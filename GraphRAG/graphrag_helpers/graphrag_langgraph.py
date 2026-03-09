from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, ServiceUnavailable, SessionExpired
from pydantic import BaseModel, Field


READ_ONLY_BLOCKLIST = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|DROP|REMOVE|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE
)


def load_neo4j_config(env_file: str | Path) -> dict[str, str]:
    config: dict[str, str] = {}
    with open(env_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    required = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing Neo4j config keys: {missing}")
    return config


class Neo4jReadClient:
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str,
        max_attempts: int = 3,
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.max_attempts = max(1, int(max_attempts))
        self.driver = self._new_driver()

    def _new_driver(self):
        return GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            connection_timeout=30,
            max_connection_lifetime=300,
        )

    def reconnect(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass
        self.driver = self._new_driver()
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def run_read(self, query: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        if READ_ONLY_BLOCKLIST.search(query):
            raise ValueError("Blocked non-read Cypher statement.")
        payload = params or {}
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with self.driver.session(database=self.database) as session:
                    records = session.run(query, payload)
                    return [dict(rec.data()) for rec in records]
            except (SessionExpired, ServiceUnavailable, DriverError, OSError) as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                try:
                    self.reconnect()
                except Exception:
                    # If reconnect fails, continue to retry loop and surface final error.
                    pass
                time.sleep(min(0.4 * (2 ** (attempt - 1)), 2.0))
        assert last_error is not None
        raise last_error


class QuestionPlan(BaseModel):
    intent: Literal["hub_partners", "shadow_explanation", "top_shadow_hubs", "general"] = "general"
    country_name: Optional[str] = None
    country_iso3: Optional[str] = None
    year: Optional[int] = None
    top_n: int = 5
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    retrieval_focus: str = Field(default="Find direct graph evidence for the user question.")


class QualityReport(BaseModel):
    is_sufficient: bool
    issues: list[str] = Field(default_factory=list)
    failure_reason: str = ""
    needs_user_clarification: bool = False
    clarification_question: Optional[str] = None
    retrieval_gap: bool = False


class RewritePlan(BaseModel):
    refined_question: str
    retrieval_hint: str


class RagState(TypedDict, total=False):
    question: str
    refined_question: str
    attempt: int
    max_retries: int
    top_n_default: int
    k_vector: int
    plan: dict[str, Any]
    cypher_evidence: list[dict[str, Any]]
    vector_evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    context_text: str
    draft_answer: str
    quality_report: dict[str, Any]
    final_answer: str


class HybridVectorIndex:
    def __init__(
        self,
        neo4j_client: Neo4jReadClient,
        embeddings: OpenAIEmbeddings,
        cache_dir: str | Path = ".cache/graphrag",
    ):
        self.neo4j = neo4j_client
        self.embeddings = embeddings
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.docs_path = self.cache_dir / "docs.json"
        self.vec_path = self.cache_dir / "vectors.npy"
        self.fp_path = self.cache_dir / "fingerprint.json"
        self.docs: list[dict[str, Any]] = []
        self.matrix: Optional[np.ndarray] = None

    def _fingerprint(self) -> dict[str, Any]:
        row = self.neo4j.run_read(
            """
            MATCH (n) WITH count(n) AS node_count
            MATCH ()-[t:TRADE]->() WITH node_count, count(t) AS trade_count
            MATCH ()-[s:SHADOW_HUB]->() WITH node_count, trade_count, count(s) AS shadow_count
            MATCH (y:Year) RETURN node_count, trade_count, shadow_count, max(y.year) AS max_year
            """
        )[0]
        return row

    def _fetch_docs(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []

        countries = self.neo4j.run_read(
            """
            MATCH (c:Country)
            RETURN c.iso3 AS iso3, c.name AS name, c.iso2 AS iso2,
                   coalesce(c.ofac_entities, 0) AS ofac_entities,
                   coalesce(c.ofac_links, 0) AS ofac_links
            ORDER BY c.iso3
            """
        )
        name_by_iso3 = {row["iso3"]: row["name"] for row in countries}
        for row in countries:
            text = (
                f"Country profile: {row['name']} ({row['iso3']}). "
                f"ISO2: {row.get('iso2')}. "
                f"OFAC-linked entities: {row.get('ofac_entities', 0)}. "
                f"OFAC link score/count: {row.get('ofac_links', 0)}."
            )
            docs.append(
                {
                    "doc_id": f"country::{row['iso3']}",
                    "source_type": "country_profile",
                    "text": text,
                    "metadata": {"iso3": row["iso3"], "name": row["name"]},
                }
            )

        shadow_rows = self.neo4j.run_read(
            """
            MATCH (c:Country)-[s:SHADOW_HUB]->(y:Year)
            RETURN c.iso3 AS iso3, c.name AS name, y.year AS year,
                   s.shadow_resid AS shadow_resid, s.shadow_rank AS shadow_rank,
                   s.betweenness AS betweenness, s.trade_total_usd AS trade_total_usd,
                   s.in_deg AS in_deg, s.out_deg AS out_deg
            ORDER BY y.year, s.shadow_rank
            """
        )
        for row in shadow_rows:
            text = (
                f"Shadow hub metrics for {row['name']} ({row['iso3']}) in {row['year']}: "
                f"shadow_resid={row.get('shadow_resid')}, shadow_rank={row.get('shadow_rank')}, "
                f"betweenness={row.get('betweenness')}, trade_total_usd={row.get('trade_total_usd')}, "
                f"in_degree={row.get('in_deg')}, out_degree={row.get('out_deg')}. "
                "A high shadow residual means the country is unusually central relative to trade volume."
            )
            docs.append(
                {
                    "doc_id": f"shadow::{row['iso3']}::{row['year']}",
                    "source_type": "shadow_metric",
                    "text": text,
                    "metadata": {
                        "iso3": row["iso3"],
                        "name": row["name"],
                        "year": row["year"],
                    },
                }
            )

        trades = self.neo4j.run_read(
            """
            MATCH (a:Country)-[t:TRADE]->(b:Country)
            RETURN a.iso3 AS exporter_iso3, b.iso3 AS importer_iso3, t.year AS year, t.trade_value_usd AS usd
            """
        )

        exports: dict[tuple[str, int], list[tuple[float, str]]] = {}
        imports: dict[tuple[str, int], list[tuple[float, str]]] = {}
        for row in trades:
            key_exp = (row["exporter_iso3"], int(row["year"]))
            key_imp = (row["importer_iso3"], int(row["year"]))
            exports.setdefault(key_exp, []).append((float(row["usd"]), row["importer_iso3"]))
            imports.setdefault(key_imp, []).append((float(row["usd"]), row["exporter_iso3"]))

        all_keys = sorted(set(exports.keys()) | set(imports.keys()), key=lambda x: (x[1], x[0]))
        for iso3, year in all_keys:
            exp_rows = sorted(exports.get((iso3, year), []), key=lambda x: x[0], reverse=True)
            imp_rows = sorted(imports.get((iso3, year), []), key=lambda x: x[0], reverse=True)
            total_exp = sum(v for v, _ in exp_rows)
            total_imp = sum(v for v, _ in imp_rows)
            top_exp = ", ".join(
                f"{name_by_iso3.get(p, p)} ({p}) ${v:,.0f}" for v, p in exp_rows[:3]
            ) or "none"
            top_imp = ", ".join(
                f"{name_by_iso3.get(p, p)} ({p}) ${v:,.0f}" for v, p in imp_rows[:3]
            ) or "none"
            name = name_by_iso3.get(iso3, iso3)
            text = (
                f"Trade summary for {name} ({iso3}) in {year}. "
                f"Total exports: ${total_exp:,.0f}. Total imports: ${total_imp:,.0f}. "
                f"Top export partners: {top_exp}. Top import partners: {top_imp}."
            )
            docs.append(
                {
                    "doc_id": f"trade_summary::{iso3}::{year}",
                    "source_type": "trade_summary",
                    "text": text,
                    "metadata": {"iso3": iso3, "name": name, "year": year},
                }
            )
        return docs

    def build(self, force_rebuild: bool = False) -> None:
        fingerprint = self._fingerprint()
        if not force_rebuild and self.docs_path.exists() and self.vec_path.exists() and self.fp_path.exists():
            cached_fp = json.loads(self.fp_path.read_text(encoding="utf-8"))
            if cached_fp == fingerprint:
                self.docs = json.loads(self.docs_path.read_text(encoding="utf-8"))
                self.matrix = np.load(self.vec_path)
                return

        self.docs = self._fetch_docs()
        texts = [doc["text"] for doc in self.docs]
        vectors = self.embeddings.embed_documents(texts)
        self.matrix = np.asarray(vectors, dtype=np.float32)

        self.docs_path.write_text(json.dumps(self.docs, ensure_ascii=True, indent=2), encoding="utf-8")
        np.save(self.vec_path, self.matrix)
        self.fp_path.write_text(json.dumps(fingerprint, ensure_ascii=True, indent=2), encoding="utf-8")

    def search(self, query: str, k: int = 6) -> list[dict[str, Any]]:
        if self.matrix is None or not self.docs:
            raise RuntimeError("Vector index is not built. Run build() first.")
        q_vec = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        row_norms = np.linalg.norm(self.matrix, axis=1)
        q_norm = np.linalg.norm(q_vec)
        denom = np.maximum(row_norms * max(q_norm, 1e-8), 1e-8)
        sims = (self.matrix @ q_vec) / denom
        k = max(1, min(k, len(self.docs)))
        top_idx = np.argsort(-sims)[:k]
        hits: list[dict[str, Any]] = []
        for idx in top_idx:
            doc = self.docs[int(idx)]
            hits.append(
                {
                    "doc_id": doc["doc_id"],
                    "score": float(sims[int(idx)]),
                    "source_type": doc["source_type"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                }
            )
        return hits


class GraphRAGAssistant:
    def __init__(
        self,
        neo4j_client: Neo4jReadClient,
        model: str = "gpt-4o-mini",
        embeddings_model: str = "text-embedding-3-small",
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set. Set it before constructing GraphRAGAssistant.")
        self.neo4j = neo4j_client
        self.planner_llm = ChatOpenAI(model=model, temperature=0)
        self.answer_llm = ChatOpenAI(model=model, temperature=0.1)
        self.eval_llm = ChatOpenAI(model=model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.vector_index = HybridVectorIndex(self.neo4j, self.embeddings)
        self.graph = self._build_graph()

    @classmethod
    def from_neo4j_env_file(
        cls,
        env_file: str | Path,
        model: str = "gpt-4o-mini",
        embeddings_model: str = "text-embedding-3-small",
    ) -> "GraphRAGAssistant":
        config = load_neo4j_config(env_file)
        client = Neo4jReadClient(
            uri=config["NEO4J_URI"],
            username=config["NEO4J_USERNAME"],
            password=config["NEO4J_PASSWORD"],
            database=config["NEO4J_DATABASE"],
        )
        return cls(client, model=model, embeddings_model=embeddings_model)

    def close(self) -> None:
        self.neo4j.close()

    def build_vector_index(self, force_rebuild: bool = False) -> None:
        self.vector_index.build(force_rebuild=force_rebuild)

    def _latest_year(self) -> int:
        row = self.neo4j.run_read("MATCH (y:Year) RETURN max(y.year) AS year")[0]
        return int(row["year"])

    def _resolve_country(self, country_name: Optional[str], country_iso3: Optional[str]) -> dict[str, Any]:
        if not country_name and not country_iso3:
            return {}
        rows = self.neo4j.run_read(
            """
            MATCH (c:Country)
            WHERE ($iso3 IS NOT NULL AND c.iso3 = toUpper($iso3))
               OR ($name IS NOT NULL AND toLower(c.name) = toLower($name))
               OR ($name IS NOT NULL AND c.iso3 = toUpper($name))
            RETURN c.iso3 AS iso3, c.name AS name
            LIMIT 3
            """,
            {"name": country_name, "iso3": country_iso3},
        )
        if rows:
            return rows[0]
        if country_name:
            fuzzy = self.neo4j.run_read(
                """
                MATCH (c:Country)
                WHERE toLower(c.name) CONTAINS toLower($name)
                RETURN c.iso3 AS iso3, c.name AS name
                ORDER BY c.name
                LIMIT 5
                """,
                {"name": country_name},
            )
            if len(fuzzy) == 1:
                return fuzzy[0]
            if len(fuzzy) > 1:
                return {"ambiguous_matches": fuzzy}
        return {}

    def _build_graph(self):
        graph_builder = StateGraph(RagState)

        def plan_question(state: RagState) -> RagState:
            question = state.get("refined_question") or state["question"]
            latest_year = self._latest_year()
            prompt = f"""
You are planning a graph retrieval strategy for an oil-trade graph with this schema:
- (:Country)-[:TRADE {{year, trade_value_usd}}]->(:Country), where direction means exporter -> importer.
- (:Country)-[:SHADOW_HUB {{shadow_resid, shadow_rank, betweenness, trade_total_usd, in_deg, out_deg}}]->(:Year {{year}})

Classify the question and extract parameters. Keep output strict.
Rules:
- intent='hub_partners' for questions asking import/export partners of a given shadow hub.
- intent='shadow_explanation' for questions asking what makes a country a shadow hub.
- intent='top_shadow_hubs' for questions asking rankings/lists of shadow hubs.
- intent='general' otherwise.
- If the question needs hub_partners but does not specify both country (name or iso3) and year, set needs_clarification=true.
- If year is missing for top_shadow_hubs or shadow_explanation, default to {latest_year}.
- top_n defaults to {state.get('top_n_default', 5)}.

Question: {question}
"""
            plan = self.planner_llm.with_structured_output(QuestionPlan).invoke(prompt)
            plan_data = plan.model_dump()
            resolved = self._resolve_country(plan.country_name, plan.country_iso3)
            if "ambiguous_matches" in resolved:
                options = ", ".join(
                    f"{m['name']} ({m['iso3']})" for m in resolved["ambiguous_matches"][:4]
                )
                plan_data["needs_clarification"] = True
                plan_data["clarification_question"] = (
                    "I found multiple country matches. Which country do you mean: "
                    f"{options}?"
                )
            elif resolved:
                plan_data["country_iso3"] = resolved["iso3"]
                plan_data["country_name"] = resolved["name"]
            if plan_data["intent"] == "hub_partners":
                if not plan_data.get("country_iso3") and not plan_data.get("country_name"):
                    plan_data["needs_clarification"] = True
                    plan_data["clarification_question"] = (
                        "Please provide the shadow hub country (name or ISO3 code)."
                    )
                if not plan_data.get("year"):
                    plan_data["needs_clarification"] = True
                    if plan_data.get("clarification_question"):
                        plan_data["clarification_question"] += " Also provide the year."
                    else:
                        plan_data["clarification_question"] = "Please provide the year for partner analysis."
            return {"plan": plan_data}

        def retrieve_hybrid(state: RagState) -> RagState:
            plan = state["plan"]
            top_n = int(plan.get("top_n") or state.get("top_n_default", 5))
            cypher_evidence: list[dict[str, Any]] = []
            citations: list[dict[str, Any]] = []
            cypher_index = 1
            vector_index = 1

            def add_citation(source_type: str, title: str, locator: str, details: str) -> str:
                nonlocal cypher_index, vector_index
                if source_type == "cypher":
                    cid = f"C{cypher_index}"
                    cypher_index += 1
                else:
                    cid = f"V{vector_index}"
                    vector_index += 1
                citations.append(
                    {
                        "id": cid,
                        "source_type": source_type,
                        "title": title,
                        "locator": locator,
                        "details": details,
                    }
                )
                return cid

            if not plan.get("needs_clarification"):
                intent = plan.get("intent")
                year = plan.get("year")
                if year is None and intent in {"top_shadow_hubs", "shadow_explanation"}:
                    year = self._latest_year()
                    plan["year"] = year

                if intent == "hub_partners":
                    iso3 = plan.get("country_iso3")
                    params = {"iso3": iso3, "year": int(year), "top_n": top_n}
                    exports = self.neo4j.run_read(
                        """
                        MATCH (c:Country {iso3: $iso3})-[e:TRADE {year: $year}]->(p:Country)
                        RETURN p.iso3 AS partner_iso3, p.name AS partner_name, e.trade_value_usd AS trade_value_usd
                        ORDER BY e.trade_value_usd DESC
                        LIMIT $top_n
                        """,
                        params,
                    )
                    imports = self.neo4j.run_read(
                        """
                        MATCH (p:Country)-[i:TRADE {year: $year}]->(c:Country {iso3: $iso3})
                        RETURN p.iso3 AS partner_iso3, p.name AS partner_name, i.trade_value_usd AS trade_value_usd
                        ORDER BY i.trade_value_usd DESC
                        LIMIT $top_n
                        """,
                        params,
                    )
                    shadow_metrics = self.neo4j.run_read(
                        """
                        MATCH (c:Country {iso3: $iso3})-[s:SHADOW_HUB]->(y:Year {year: $year})
                        RETURN c.iso3 AS iso3, c.name AS name, y.year AS year, s.shadow_resid AS shadow_resid,
                               s.shadow_rank AS shadow_rank, s.betweenness AS betweenness, s.trade_total_usd AS trade_total_usd
                        LIMIT 1
                        """,
                        params,
                    )
                    cid_exp = add_citation(
                        "cypher",
                        "Top export partners",
                        "Neo4j TRADE relationships",
                        f"country={iso3}, year={year}, rows={len(exports)}",
                    )
                    cid_imp = add_citation(
                        "cypher",
                        "Top import partners",
                        "Neo4j TRADE relationships",
                        f"country={iso3}, year={year}, rows={len(imports)}",
                    )
                    cid_shadow = add_citation(
                        "cypher",
                        "Shadow metrics",
                        "Neo4j SHADOW_HUB relationship",
                        f"country={iso3}, year={year}, rows={len(shadow_metrics)}",
                    )
                    cypher_evidence.extend(
                        [
                            {"citation_id": cid_exp, "type": "exports", "rows": exports},
                            {"citation_id": cid_imp, "type": "imports", "rows": imports},
                            {"citation_id": cid_shadow, "type": "shadow_metrics", "rows": shadow_metrics},
                        ]
                    )

                elif intent == "top_shadow_hubs":
                    params = {"year": int(year), "top_n": top_n}
                    rows = self.neo4j.run_read(
                        """
                        MATCH (c:Country)-[s:SHADOW_HUB]->(y:Year {year: $year})
                        RETURN c.iso3 AS iso3, c.name AS name, y.year AS year, s.shadow_resid AS shadow_resid,
                               s.shadow_rank AS shadow_rank, s.betweenness AS betweenness, s.trade_total_usd AS trade_total_usd
                        ORDER BY s.shadow_resid DESC
                        LIMIT $top_n
                        """,
                        params,
                    )
                    cid = add_citation(
                        "cypher",
                        "Top shadow hubs by residual",
                        "Neo4j SHADOW_HUB relationship",
                        f"year={year}, rows={len(rows)}",
                    )
                    cypher_evidence.append({"citation_id": cid, "type": "top_shadow_hubs", "rows": rows})

                else:
                    if plan.get("country_iso3"):
                        iso3 = plan["country_iso3"]
                        rows = self.neo4j.run_read(
                            """
                            MATCH (c:Country {iso3: $iso3})-[s:SHADOW_HUB]->(y:Year)
                            RETURN c.iso3 AS iso3, c.name AS name, y.year AS year, s.shadow_resid AS shadow_resid,
                                   s.shadow_rank AS shadow_rank, s.betweenness AS betweenness, s.trade_total_usd AS trade_total_usd
                            ORDER BY y.year DESC
                            LIMIT 5
                            """,
                            {"iso3": iso3},
                        )
                        cid = add_citation(
                            "cypher",
                            "Country shadow trend",
                            "Neo4j SHADOW_HUB relationship",
                            f"country={iso3}, rows={len(rows)}",
                        )
                        cypher_evidence.append({"citation_id": cid, "type": "country_shadow_trend", "rows": rows})
                    else:
                        latest_year = self._latest_year()
                        rows = self.neo4j.run_read(
                            """
                            MATCH (c:Country)-[s:SHADOW_HUB]->(y:Year {year: $year})
                            RETURN c.iso3 AS iso3, c.name AS name, y.year AS year, s.shadow_resid AS shadow_resid,
                                   s.shadow_rank AS shadow_rank
                            ORDER BY s.shadow_resid DESC
                            LIMIT 10
                            """,
                            {"year": latest_year},
                        )
                        cid = add_citation(
                            "cypher",
                            "Top shadow hubs baseline",
                            "Neo4j SHADOW_HUB relationship",
                            f"year={latest_year}, rows={len(rows)}",
                        )
                        cypher_evidence.append({"citation_id": cid, "type": "baseline_shadow_hubs", "rows": rows})

            vector_query = state.get("refined_question") or state["question"]
            if plan.get("retrieval_focus"):
                vector_query = f"{vector_query}\nFocus: {plan['retrieval_focus']}"
            if plan.get("country_iso3"):
                vector_query += f"\nCountry ISO3: {plan['country_iso3']}"
            if plan.get("year"):
                vector_query += f"\nYear: {plan['year']}"
            vector_hits = self.vector_index.search(vector_query, k=int(state.get("k_vector", 6)))
            vector_evidence: list[dict[str, Any]] = []
            for hit in vector_hits:
                cid = add_citation(
                    "vector",
                    f"Vector hit {hit['doc_id']}",
                    hit["source_type"],
                    f"score={hit['score']:.3f}",
                )
                vector_evidence.append(
                    {
                        "citation_id": cid,
                        "doc_id": hit["doc_id"],
                        "score": hit["score"],
                        "source_type": hit["source_type"],
                        "metadata": hit["metadata"],
                        "text": hit["text"],
                    }
                )

            context_lines: list[str] = []
            for ev in cypher_evidence:
                context_lines.append(
                    f"{ev['citation_id']} | {ev['type']} | rows={len(ev['rows'])} | data={json.dumps(ev['rows'], ensure_ascii=True)}"
                )
            for hit in vector_evidence:
                context_lines.append(
                    f"{hit['citation_id']} | vector | {hit['doc_id']} | score={hit['score']:.3f} | text={hit['text']}"
                )
            return {
                "plan": plan,
                "cypher_evidence": cypher_evidence,
                "vector_evidence": vector_evidence,
                "citations": citations,
                "context_text": "\n".join(context_lines),
            }

        def draft_answer(state: RagState) -> RagState:
            plan = state["plan"]
            if plan.get("needs_clarification"):
                return {"draft_answer": ""}
            prompt = f"""
You are an AI assistant for the Shadow Hubs in Global Oil Trade visualization. You answer questions
about the data, the methodology, and the tool itself. Use the PROJECT REFERENCE below for questions
about how the tool works, the math, the tech stack, etc. Use the GRAPH EVIDENCE for questions about
specific countries, trade flows, rankings, and sanctions data. Combine both when helpful.

=== PROJECT REFERENCE ===

WHAT THIS TOOL IS:
An interactive 3D globe (Globe.gl) that maps bilateral crude and refined petroleum trade flows
(UN Comtrade HS 2709 crude, HS 2710 refined) from 2019 to 2024, overlaid with OFAC sanctions
exposure and network-derived "shadow hub" scores. Users can click countries to see trade arcs,
toggle between OFAC and clustering color modes, and adjust the percentage of edges shown.

WHAT IS A SHADOW HUB:
A shadow hub is a country that is unusually central in the oil trade network relative to its actual
trade volume. We measure this using betweenness centrality residuals. Betweenness centrality counts
how many shortest paths between all pairs of countries pass through a given country. We regress
betweenness centrality against log trade volume using OLS, and the residual is the "shadow hub score."
A positive residual means the country sits on more shortest paths than its trade volume would predict —
it is structurally anomalous as an intermediary. A high shadow residual does NOT prove illicit activity.
Legitimate factors (geography, refining capacity, free trade zones) can produce elevated centrality.

ISOLATING TRUE SHADOW HUBS:
To move beyond pure mathematical anomaly, we cross-reference shadow hub scores with OFAC sanctions
exposure (count of sanctioned entities per country). The intersection of high betweenness residual
AND high sanctions exposure pinpoints countries that are both structurally anomalous and connected
to sanctioned activity — the most likely candidates for sanctions evasion intermediation.

CLUSTERING ANALYSIS — HOW SHADOW HUBS BROKER:
Shadow hub detection tells us WHICH countries are anomalous intermediaries but not HOW they operate.
The weighted clustering coefficient answers this. For each country i, it measures the fraction of
triangles among its trade partners, weighted by trade volume. Formally: if country i trades with
both j and k, a weighted triangle exists when j also trades with k. We compute this via
nx.clustering(G, weight='log_weight') on the directed trade graph for each year.

Low clustering = the country's partners do NOT trade with each other. The country is a sole bridge
between disconnected groups (exclusive broker pattern).
High clustering = partners form a dense clique. Oil circulates within a tight network rather than
passing through a single chokepoint (laundering syndicate pattern).

THREE BEHAVIORAL PATTERNS:
- Exclusive Broker (clustering coefficient < 0.20): Sole bridge between disconnected trading blocs.
  Partners rarely trade directly. All flows route through this hub. Examples: Singapore (0.18),
  Turkey (0.19). Consistent with transshipment hubs or sanctions evasion chokepoints.
- Mixed/Moderate (CC 0.20-0.40): Some partner interconnection but still structurally important.
  Examples: Spain (0.24), Kenya (0.31). Regional brokerage roles with partial direct partner trade.
- Laundering Syndicate (CC > 0.40): Partners trade heavily with each other forming a dense sub-network.
  Oil circulates within a tight cluster. Examples: Georgia (0.45), Uzbekistan (0.42). Consistent
  with coordinated trade rings or regional blocs.

KEY INSIGHT: High clustering + high shadow residual + high OFAC exposure = strongest signal for
organized sanctions evasion. These are Exclusive Brokers and Laundering Syndicates actively bypassing
global sanctions.

WHAT IS GraphRAG:
GraphRAG stands for Graph-based Retrieval-Augmented Generation. Instead of answering purely from
pre-trained knowledge, the AI first queries a Neo4j graph database containing the actual trade network
(countries, trade flows, shadow hub metrics, OFAC data), retrieves relevant evidence, then uses that
evidence to ground its response. This means answers about specific countries or years are backed by
real data, not hallucinated.

HOW CITATIONS WORK:
[C1], [C2], etc. = Cypher citations — data retrieved by direct graph database queries (e.g.
"top 5 shadow hubs in 2024"). These are exact results from Neo4j.
[V1], [V2], etc. = Vector citations — data retrieved by semantic similarity search over
pre-embedded trade summaries. The AI finds relevant text snippets matching the question.

GRAPH DATABASE SCHEMA (Neo4j):
- (:Country) nodes with properties: iso3, name, iso2, ofac_entities, ofac_links
- (:Year) nodes with property: year (2019-2024)
- (:Country)-[:TRADE {{year, trade_value_usd}}]->(:Country) — direction = exporter -> importer
- (:Country)-[:SHADOW_HUB {{shadow_resid, shadow_rank, betweenness, trade_total_usd, in_deg, out_deg}}]->(:Year)

TECH STACK:
- Frontend: Globe.gl (3D WebGL globe), Chart.js (clustering bubble chart), vanilla HTML/CSS/JS
- Backend: FastAPI (Python), serves static files and the /ask chat endpoint
- AI/ML: LangGraph (agent orchestration), GPT-4o-mini (LLM), OpenAI text-embedding-3-small (embeddings)
- Database: Neo4j AuraDB free tier (graph database), with hybrid Cypher + vector retrieval
- Hosting: Render (web service), UptimeRobot (health check pings)
- Network analysis: NetworkX (Python) for betweenness centrality, clustering coefficients, OLS regression

DATA SOURCES:
- UN Comtrade: bilateral trade flows via comtradeapicall Python package (HS 2709/2710), 2019-2024
- OFAC SDN List: U.S. Treasury Specially Designated Nationals (sdn.csv, add.csv, sdn_comments.csv)
- UN/LOCODE: port and location geocoding
- World Port Index (WPI): NGA Pub 150, supplementary port coordinates

GLOBE DISPLAY MODES:
- Clustered mode (default): dots colored by brokerage pattern — blue=Broker, orange=Mixed, red=Syndicate,
  gray=non-hub. Leaderboard shows top shadow hubs filtered by OFAC > 5 with cluster labels.
- Raw mode: dots colored by OFAC exposure — red=high (>500), orange=medium (100-500), blue=low/zero.
  Leaderboard shows raw shadow hub rankings with OFAC count.
- Dot size reflects shadow hub rank (larger = higher residual).
- Arcs appear when clicking a country, colored by trade value tier: gold=top 10%, light blue=top 25%,
  dark blue=remaining. Edge slider controls what percentage of flows are shown (default 25%).

=== END PROJECT REFERENCE ===

Question:
{state.get('refined_question') or state['question']}

Extracted plan:
{json.dumps(plan, ensure_ascii=True)}

Graph evidence (from Neo4j database):
{state['context_text']}

Rules:
- For data questions (countries, rankings, trade flows), use graph evidence and cite inline [C1]/[V2].
- For questions about the tool, methodology, math, or tech stack, use the PROJECT REFERENCE above.
- You can combine both — e.g. explain what a shadow hub is (reference) then show the top ones (evidence).
- Keep answers concise and helpful. Do not say you lack evidence for conceptual questions.
- A high shadow residual suggests anomalous intermediary structure, not proof of illicit activity.
"""
            answer = self.answer_llm.invoke(prompt).content
            return {"draft_answer": answer}

        def evaluate_answer(state: RagState) -> RagState:
            plan = state["plan"]
            issues: list[str] = []
            draft = state.get("draft_answer", "")
            if plan.get("needs_clarification"):
                q = plan.get("clarification_question") or "Please provide the missing query parameters."
                report = QualityReport(
                    is_sufficient=False,
                    issues=["missing_user_inputs"],
                    failure_reason="Required inputs are missing.",
                    needs_user_clarification=True,
                    clarification_question=q,
                    retrieval_gap=False,
                )
                return {"quality_report": report.model_dump()}

            # General/meta questions don't need graph evidence or citations
            is_general = plan.get("intent") == "general" and not plan.get("country_iso3")
            if not is_general:
                if not state.get("cypher_evidence") and not state.get("vector_evidence"):
                    issues.append("empty_evidence")
                if not re.search(r"\[[CV]\d+\]", draft):
                    issues.append("missing_citations_in_answer")
                if plan.get("year") and str(plan["year"]) not in draft:
                    issues.append("year_not_mentioned")
            country_tokens = [plan.get("country_iso3"), plan.get("country_name")]
            country_tokens = [t for t in country_tokens if t]
            if country_tokens and not any(t.lower() in draft.lower() for t in country_tokens):
                issues.append("country_not_mentioned")

            eval_prompt = f"""
Evaluate whether this answer sufficiently addresses the question using the provided evidence.

Question:
{state.get('refined_question') or state['question']}

Answer:
{draft}

Evidence summary:
{state['context_text']}

Return strict JSON using this schema:
{{
  "is_sufficient": bool,
  "issues": [string],
  "failure_reason": string,
  "needs_user_clarification": bool,
  "clarification_question": string | null,
  "retrieval_gap": bool
}}
"""
            llm_report = self.eval_llm.with_structured_output(QualityReport).invoke(eval_prompt)
            merged_issues = list(dict.fromkeys((llm_report.issues or []) + issues))
            is_sufficient = bool(llm_report.is_sufficient and not merged_issues)
            report = llm_report.model_copy(
                update={
                    "issues": merged_issues,
                    "is_sufficient": is_sufficient,
                    "failure_reason": llm_report.failure_reason
                    if llm_report.failure_reason
                    else ("; ".join(merged_issues) if merged_issues else ""),
                    "retrieval_gap": bool(llm_report.retrieval_gap or "empty_evidence" in merged_issues),
                }
            )
            return {"quality_report": report.model_dump()}

        def rewrite_query(state: RagState) -> RagState:
            quality = state["quality_report"]
            rewrite_prompt = f"""
Rewrite the question for better retrieval over Neo4j graph + vector summaries.
Keep the original intent but make entity/year constraints explicit.

Original question:
{state['question']}

Current refined question:
{state.get('refined_question') or state['question']}

Failure reason:
{quality.get('failure_reason')}

Issues:
{quality.get('issues')}
"""
            rewrite = self.planner_llm.with_structured_output(RewritePlan).invoke(rewrite_prompt)
            return {
                "refined_question": rewrite.refined_question,
                "attempt": int(state.get("attempt", 0)) + 1,
            }

        def finalize(state: RagState) -> RagState:
            draft = state.get("draft_answer", "").strip()
            citation_lines = []
            for citation in state.get("citations", []):
                citation_lines.append(
                    f"- [{citation['id']}] {citation['title']} | {citation['locator']} | {citation['details']}"
                )
            appendix = "Citations:\n" + ("\n".join(citation_lines) if citation_lines else "- None")
            final_text = f"{draft}\n\n{appendix}".strip()
            return {"final_answer": final_text}

        def ask_for_clarification(state: RagState) -> RagState:
            quality = state.get("quality_report", {})
            reason = quality.get("failure_reason") or "Insufficient evidence after retries."
            clarification = quality.get("clarification_question") or (
                "Please provide the specific country (name or ISO3) and year."
            )
            attempts = state.get("attempt", 0)
            final_text = (
                f"I could not produce a reliable answer after {attempts + 1} attempt(s).\n"
                f"Failure reason: {reason}\n"
                f"Needed clarification: {clarification}"
            )
            quality_out = {
                **quality,
                "is_sufficient": False,
                "needs_user_clarification": True,
                "clarification_question": clarification,
                "failure_reason": reason,
            }
            return {"final_answer": final_text, "quality_report": quality_out}

        def route_after_eval(state: RagState) -> str:
            quality = state["quality_report"]
            if quality.get("is_sufficient"):
                return "finalize"
            if quality.get("needs_user_clarification"):
                return "clarify"
            if int(state.get("attempt", 0)) < int(state.get("max_retries", 2)):
                return "rewrite"
            return "clarify"

        graph_builder.add_node("plan", plan_question)
        graph_builder.add_node("retrieve", retrieve_hybrid)
        graph_builder.add_node("draft", draft_answer)
        graph_builder.add_node("evaluate", evaluate_answer)
        graph_builder.add_node("rewrite", rewrite_query)
        graph_builder.add_node("finalize", finalize)
        graph_builder.add_node("clarify", ask_for_clarification)

        graph_builder.add_edge(START, "plan")
        graph_builder.add_edge("plan", "retrieve")
        graph_builder.add_edge("retrieve", "draft")
        graph_builder.add_edge("draft", "evaluate")
        graph_builder.add_conditional_edges(
            "evaluate",
            route_after_eval,
            {"finalize": "finalize", "rewrite": "rewrite", "clarify": "clarify"},
        )
        graph_builder.add_edge("rewrite", "plan")
        graph_builder.add_edge("finalize", END)
        graph_builder.add_edge("clarify", END)
        return graph_builder.compile()

    def ask(self, question: str, max_retries: int = 2, top_n_default: int = 5, k_vector: int = 6) -> dict[str, Any]:
        state: RagState = {
            "question": question,
            "refined_question": question,
            "attempt": 0,
            "max_retries": max_retries,
            "top_n_default": top_n_default,
            "k_vector": k_vector,
        }
        try:
            out = self.graph.invoke(state)
        except (SessionExpired, ServiceUnavailable, DriverError, OSError):
            # Recover once from transient Aura connection drops.
            self.neo4j.reconnect()
            out = self.graph.invoke(state)
        return {
            "answer": out.get("final_answer", ""),
            "plan": out.get("plan", {}),
            "quality_report": out.get("quality_report", {}),
            "attempt": out.get("attempt", 0),
        }

    def _default_clarification_callback(
        self, clarification_question: str, prior_result: dict[str, Any]
    ) -> Optional[str]:
        print(f"Clarification needed: {clarification_question}")
        try:
            choice = input("Would you like to add clarification and retry? [y/N]: ").strip().lower()
        except EOFError:
            print("Clarification input is not available in this environment. Skipping interactive retry.")
            return None
        if choice not in {"y", "yes"}:
            return None
        try:
            extra = input("Enter clarification: ").strip()
        except EOFError:
            print("Clarification input is not available in this environment. Skipping interactive retry.")
            return None
        return extra or None

    def _extract_needed_clarification(
        self, result: dict[str, Any], fallback: str = "Please provide the missing details."
    ) -> str:
        quality = result.get("quality_report", {}) or {}
        if quality.get("clarification_question"):
            return str(quality["clarification_question"])
        answer = result.get("answer", "") or ""
        match = re.search(r"Needed clarification:\s*(.+)", answer)
        if match:
            return match.group(1).strip()
        return fallback

    def ask_with_user_clarification(
        self,
        question: str,
        max_retries: int = 2,
        top_n_default: int = 5,
        k_vector: int = 6,
        max_clarification_rounds: int = 2,
        clarification_callback: Optional[Callable[[str, dict[str, Any]], Optional[str]]] = None,
    ) -> dict[str, Any]:
        callback = clarification_callback or self._default_clarification_callback
        clarifications: list[str] = []
        last_result: Optional[dict[str, Any]] = None

        for _ in range(max_clarification_rounds + 1):
            merged_question = question
            if clarifications:
                merged_question = (
                    f"{question}\n\nAdditional user clarifications:\n"
                    + "\n".join(f"- {item}" for item in clarifications)
                )

            result = self.ask(
                merged_question,
                max_retries=max_retries,
                top_n_default=top_n_default,
                k_vector=k_vector,
            )
            last_result = result

            quality = result.get("quality_report", {}) or {}
            needs_clarification = bool(quality.get("needs_user_clarification"))
            if not needs_clarification:
                result["clarifications_used"] = clarifications
                return result

            needed = self._extract_needed_clarification(result)
            try:
                extra = callback(needed, result)
            except EOFError:
                result["clarifications_used"] = clarifications
                quality = result.get("quality_report", {}) or {}
                if not quality.get("failure_reason"):
                    quality["failure_reason"] = "Clarification input is unavailable in this environment."
                result["quality_report"] = quality
                return result
            if not extra:
                result["clarifications_used"] = clarifications
                return result
            clarifications.append(extra)

        if last_result is None:
            last_result = self.ask(
                question,
                max_retries=max_retries,
                top_n_default=top_n_default,
                k_vector=k_vector,
            )
        last_result["clarifications_used"] = clarifications
        return last_result
