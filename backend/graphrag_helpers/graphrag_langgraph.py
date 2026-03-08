from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, ServiceUnavailable, SessionExpired
from pydantic import BaseModel, Field


READ_ONLY_BLOCKLIST = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|DROP|REMOVE|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE
)


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

    @classmethod
    def from_env(cls) -> Neo4jReadClient:
        """Create a Neo4jReadClient from environment variables."""
        uri = os.environ.get("NEO4J_URI")
        username = os.environ.get("NEO4J_USERNAME")
        password = os.environ.get("NEO4J_PASSWORD")
        database = os.environ.get("NEO4J_DATABASE")

        if not all([uri, username, password, database]):
            raise ValueError(
                "Missing Neo4j environment variables. "
                "Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE"
            )

        return cls(uri=uri, username=username, password=password, database=database)

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
                    pass
                time.sleep(min(0.4 * (2 ** (attempt - 1)), 2.0))
        if last_error:
            raise last_error
        raise RuntimeError("Unable to execute query after all retries")


class QuestionPlan(BaseModel):
    intent: Literal[
        "hub_partners",
        "shadow_explanation",
        "top_shadow_hubs",
        "emerging_hubs",
        "clustering_analysis",
        "sanctions_hubs",
        "temporal_trend",
        "comparative",
        "general",
    ] = "general"
    country_name: Optional[str] = None
    country_iso3: Optional[str] = None
    country_iso3_b: Optional[str] = None
    year: Optional[int] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
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
        cache_dir: str | Path = "/tmp/graphrag_cache",
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
                   s.in_deg AS in_deg, s.out_deg AS out_deg,
                   s.clustering_weighted AS clustering_weighted
            ORDER BY y.year, s.shadow_rank
            """
        )
        for row in shadow_rows:
            text = (
                f"Shadow hub metrics for {row['name']} ({row['iso3']}) in {row['year']}: "
                f"shadow_resid={row.get('shadow_resid')}, shadow_rank={row.get('shadow_rank')}, "
                f"betweenness={row.get('betweenness')}, trade_total_usd={row.get('trade_total_usd')}, "
                f"in_degree={row.get('in_deg')}, out_degree={row.get('out_deg')}, "
                f"clustering_weighted={row.get('clustering_weighted')}. "
                "A high shadow residual means the country is unusually central relative to trade volume. "
                "Clustering below 0.20 suggests an exclusive broker pattern; above 0.40 suggests a laundering syndicate."
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

        conceptual_docs = [
            {
                "doc_id": "concept::shadow_hub_definition",
                "source_type": "conceptual",
                "text": "A shadow hub is a country whose betweenness centrality in the global oil trade network is anomalously high relative to its trade volume. These countries are structurally positioned as intermediaries — they sit on many shortest paths between trading pairs but handle relatively little oil themselves. This asymmetry, measured by the shadow residual (the gap between actual and predicted betweenness), suggests the country may be facilitating indirect trade routes, potentially related to sanctions circumvention or illicit trade routing.",
                "metadata": {"concept": "shadow_hub_definition"},
            },
            {
                "doc_id": "concept::betweenness_centrality",
                "source_type": "conceptual",
                "text": "Betweenness centrality measures how often a country lies on the shortest path between other trading pairs in the oil network. High betweenness means the country is a critical bridge or chokepoint. When combined with low trade volume, it indicates the country is facilitating others' trade without being a major trader itself — the hallmark of an intermediary or transit hub.",
                "metadata": {"concept": "betweenness_centrality"},
            },
            {
                "doc_id": "concept::regression_ceiling",
                "source_type": "conceptual",
                "text": "The USA appears as the top shadow hub due to a statistical artifact called the regression ceiling. The shadow hub model uses regression of log(betweenness) on log(volume) trained on normal-sized countries. Because the USA has astronomically high trade volume, the model predicts its betweenness should be far higher than is mathematically possible — betweenness has an upper bound once a country is connected to nearly everyone. This creates an enormous negative residual that gets inverted in the ranking. The USA's high shadow rank reflects its extreme outlier status, not evidence of illicit intermediary behavior.",
                "metadata": {"concept": "regression_ceiling", "country": "USA"},
            },
            {
                "doc_id": "concept::ofac_enforcer_bias",
                "source_type": "conceptual",
                "text": "The USA has a high OFAC entity count not because it is sanctioned, but because it is the enforcer. The OFAC SDN list tracks addresses and bank accounts of sanctioned entities. Because the US dollar is the global reserve currency, many targeted shell companies, drug cartels, and illicit traders attempt to open US bank accounts or register fake addresses in states like Delaware. The data aggregator counts these US addresses as US entities even though they are the targets being hunted, not American perpetrators.",
                "metadata": {"concept": "ofac_enforcer_bias", "country": "USA"},
            },
            {
                "doc_id": "concept::sanctions_shockwave",
                "source_type": "conceptual",
                "text": "The sanctions shockwave describes the dramatic restructuring of global oil trade routes after 2022 when direct Russian trade was banned. Global oil routes shattered into complex indirect pathways. Spain topped the emerging hubs list because its coastal waters near Ceuta became the global epicenter for unregulated Ship-to-Ship transfers of Russian crude. Italy and Belgium acted as emergency intake valves for laundered oil returning from third-party refineries in Asia. These countries may not be intentional shadow hubs operating illicit shell companies, but they became the primary geographic bottlenecks bearing the brunt of the newly restructured shadow economy.",
                "metadata": {"concept": "sanctions_shockwave"},
            },
            {
                "doc_id": "concept::exclusive_broker_low_clustering",
                "source_type": "conceptual",
                "text": "An exclusive broker is a shadow hub with a low weighted clustering coefficient (typically below 0.20). In network terms, this means the hub's trading partners do not trade with each other — the hub bridges two completely disconnected geopolitical blocs. Singapore (clustering ~0.18) is the prime example. Western buyers refuse to trade directly with sanctioned entities, creating a permanent structural hole in the network. The broker steps into this void as the sole conduit. The low clustering coefficient mathematically proves the hub is bridging disconnected groups, and the broken triangle means no alternative route exists without the broker.",
                "metadata": {"concept": "exclusive_broker", "pattern": "low_clustering"},
            },
            {
                "doc_id": "concept::laundering_syndicate_high_clustering",
                "source_type": "conceptual",
                "text": "A laundering syndicate is a shadow hub with a high weighted clustering coefficient (typically above 0.40). Rather than acting as an isolated middleman, this hub sits inside a tightly knit group of regional actors who all trade heavily with one another. Georgia (clustering ~0.45) is the prime example. When sanctioned Russian oil enters Georgian ports, it does not pass straight through to Europe. It bounces between a closed loop of Black Sea and Caucasus neighbors — effectively washing its paper trail and disguising it as local Caspian blend before reaching the open market. The high clustering score proves the oil is traded multiple times within a friendly syndicate to mask its origin.",
                "metadata": {"concept": "laundering_syndicate", "pattern": "high_clustering"},
            },
            {
                "doc_id": "concept::clustering_interpretation",
                "source_type": "conceptual",
                "text": "The weighted clustering coefficient measures the proportion of closed triangles in a country's trade neighborhood. If Country A trades with B and C, high clustering means B and C also trade with each other. For shadow hubs, clustering reveals HOW the hub operates: Low clustering (below 0.20) indicates an exclusive broker pattern — the hub bridges disconnected groups. Medium clustering (0.20-0.35) suggests a mixed role. High clustering (above 0.40) indicates a laundering syndicate pattern — the hub is embedded in a tight clique that trades among themselves to wash the oil's paper trail.",
                "metadata": {"concept": "clustering_interpretation"},
            },
        ]
        docs.extend(conceptual_docs)

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
    def from_env(cls, model: str = "gpt-4o-mini", embeddings_model: str = "text-embedding-3-small") -> GraphRAGAssistant:
        """Create a GraphRAGAssistant from environment variables."""
        client = Neo4jReadClient.from_env()
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
- (:Country)-[:SHADOW_HUB {{shadow_resid, shadow_rank, betweenness, trade_total_usd, in_deg, out_deg, clustering_weighted}}]->(:Year {{year}})

Classify the question and extract parameters. Keep output strict.

Rules:
- intent='hub_partners' for questions asking import/export partners of a given shadow hub.
- intent='shadow_explanation' for questions asking what makes a country a shadow hub.
- intent='top_shadow_hubs' for questions asking rankings/lists of shadow hubs.
- intent='emerging_hubs' for questions about growing/emerging/rising shadow hubs or changes over time across countries. Default year_start=2021, year_end=2024.
- intent='clustering_analysis' for questions about HOW a country moves oil, broker vs syndicate, clustering coefficient. Requires a country.
- intent='sanctions_hubs' for questions about sanctioned countries that are shadow hubs, OFAC overlap.
- intent='temporal_trend' for questions about a single country's changes over time.
- intent='comparative' for questions comparing two specific countries. Extract both country_iso3 and country_iso3_b.
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

            if plan.country_iso3_b:
                resolved_b = self._resolve_country(None, plan.country_iso3_b)
                if resolved_b and "iso3" in resolved_b:
                    plan_data["country_iso3_b"] = resolved_b["iso3"]

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

            if plan_data["intent"] == "clustering_analysis":
                if not plan_data.get("country_iso3"):
                    plan_data["needs_clarification"] = True
                    plan_data["clarification_question"] = "Please provide the country for clustering analysis."
                if not plan_data.get("year"):
                    plan_data["year"] = latest_year

            if plan_data["intent"] == "comparative":
                if not plan_data.get("country_iso3") or not plan_data.get("country_iso3_b"):
                    plan_data["needs_clarification"] = True
                    plan_data["clarification_question"] = "Please provide both countries to compare (as ISO3 codes or names)."
                if not plan_data.get("year"):
                    plan_data["year"] = latest_year

            if plan_data["intent"] == "emerging_hubs":
                if not plan_data.get("year_start"):
                    plan_data["year_start"] = 2021
                if not plan_data.get("year_end"):
                    plan_data["year_end"] = 2024

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

                elif intent == "emerging_hubs":
                    year_start = plan.get("year_start", 2021)
                    year_end = plan.get("year_end", 2024)
                    params = {"year_start": year_start, "year_end": year_end, "top_n": top_n}
                    rows = self.neo4j.run_read(
                        """
                        MATCH (c:Country)-[s1:SHADOW_HUB]->(y1:Year {year: $year_start})
                        MATCH (c)-[s2:SHADOW_HUB]->(y2:Year {year: $year_end})
                        RETURN c.iso3 AS iso3, c.name AS name,
                               s1.shadow_resid AS resid_start, s2.shadow_resid AS resid_end,
                               (s2.shadow_resid - s1.shadow_resid) AS resid_change,
                               s2.clustering_weighted AS clustering
                        ORDER BY resid_change DESC
                        LIMIT $top_n
                        """,
                        params,
                    )
                    cid = add_citation(
                        "cypher",
                        "Emerging shadow hubs",
                        "Neo4j SHADOW_HUB comparison",
                        f"years={year_start}-{year_end}, rows={len(rows)}",
                    )
                    cypher_evidence.append({"citation_id": cid, "type": "emerging_hubs", "rows": rows})

                elif intent == "clustering_analysis":
                    iso3 = plan.get("country_iso3")
                    year = plan.get("year", self._latest_year())
                    params = {"iso3": iso3, "year": int(year)}
                    rows = self.neo4j.run_read(
                        """
                        MATCH (c:Country {iso3: $iso3})-[s:SHADOW_HUB]->(y:Year {year: $year})
                        RETURN c.iso3 AS iso3, c.name AS name, y.year AS year,
                               s.shadow_resid AS shadow_resid, s.shadow_rank AS shadow_rank,
                               s.betweenness AS betweenness, s.trade_total_usd AS trade_total_usd,
                               s.clustering_weighted AS clustering_weighted,
                               s.in_deg AS in_deg, s.out_deg AS out_deg
                        """,
                        params,
                    )
                    cid = add_citation(
                        "cypher",
                        "Clustering analysis",
                        "Neo4j SHADOW_HUB clustering",
                        f"country={iso3}, year={year}, rows={len(rows)}",
                    )
                    cypher_evidence.append({"citation_id": cid, "type": "clustering_analysis", "rows": rows})

                elif intent == "sanctions_hubs":
                    year = plan.get("year", self._latest_year())
                    params = {"year": int(year), "top_n": top_n}
                    rows = self.neo4j.run_read(
                        """
                        MATCH (c:Country)-[s:SHADOW_HUB]->(y:Year {year: $year})
                        WHERE c.ofac_entities > 0
                        RETURN c.iso3 AS iso3, c.name AS name, c.ofac_entities AS ofac_entities,
                               s.shadow_resid AS shadow_resid, s.shadow_rank AS shadow_rank,
                               s.clustering_weighted AS clustering_weighted
                        ORDER BY s.shadow_resid DESC
                        LIMIT $top_n
                        """,
                        params,
                    )
                    cid = add_citation(
                        "cypher",
                        "Sanctions-linked shadow hubs",
                        "Neo4j SHADOW_HUB with OFAC",
                        f"year={year}, rows={len(rows)}",
                    )
                    cypher_evidence.append({"citation_id": cid, "type": "sanctions_hubs", "rows": rows})

                elif intent == "temporal_trend":
                    iso3 = plan.get("country_iso3")
                    params = {"iso3": iso3}
                    rows = self.neo4j.run_read(
                        """
                        MATCH (c:Country {iso3: $iso3})-[s:SHADOW_HUB]->(y:Year)
                        RETURN y.year AS year, s.shadow_resid AS shadow_resid, s.shadow_rank AS shadow_rank,
                               s.betweenness AS betweenness, s.trade_total_usd AS trade_total_usd,
                               s.clustering_weighted AS clustering_weighted
                        ORDER BY y.year
                        """,
                        params,
                    )
                    cid = add_citation(
                        "cypher",
                        "Temporal trend",
                        "Neo4j SHADOW_HUB time series",
                        f"country={iso3}, rows={len(rows)}",
                    )
                    cypher_evidence.append({"citation_id": cid, "type": "temporal_trend", "rows": rows})

                elif intent == "comparative":
                    iso3_a = plan.get("country_iso3")
                    iso3_b = plan.get("country_iso3_b")
                    year = plan.get("year", self._latest_year())
                    params = {"iso3_a": iso3_a, "iso3_b": iso3_b, "year": int(year)}
                    rows = self.neo4j.run_read(
                        """
                        MATCH (c:Country)-[s:SHADOW_HUB]->(y:Year {year: $year})
                        WHERE c.iso3 IN [$iso3_a, $iso3_b]
                        RETURN c.iso3 AS iso3, c.name AS name, s.shadow_resid AS shadow_resid,
                               s.shadow_rank AS shadow_rank, s.betweenness AS betweenness,
                               s.trade_total_usd AS trade_total_usd, s.clustering_weighted AS clustering_weighted,
                               s.in_deg AS in_deg, s.out_deg AS out_deg
                        """,
                        params,
                    )
                    cid = add_citation(
                        "cypher",
                        "Comparative analysis",
                        "Neo4j SHADOW_HUB comparison",
                        f"countries={iso3_a},{iso3_b}, year={year}, rows={len(rows)}",
                    )
                    cypher_evidence.append({"citation_id": cid, "type": "comparative", "rows": rows})

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
You are answering a risk-screening question about oil-trade network structure.

Question:
{state.get('refined_question') or state['question']}

Extracted plan:
{json.dumps(plan, ensure_ascii=True)}

Evidence context:
{state['context_text']}

Rules:
- Use only evidence in context.
- Include inline citations like [C1] or [V2] next to claims.
- Keep the answer concise and analytical.
- State uncertainty if evidence is missing.
- Reminder: high shadow residual suggests anomalous intermediary structure, not proof of illicit activity.
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
            self.neo4j.reconnect()
            out = self.graph.invoke(state)
        return {
            "answer": out.get("final_answer", ""),
            "plan": out.get("plan", {}),
            "quality_report": out.get("quality_report", {}),
            "attempt": out.get("attempt", 0),
        }
