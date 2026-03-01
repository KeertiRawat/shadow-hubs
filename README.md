# 🛢️ Shadow Hubs in Global Oil Trade

**Network Analytics & Sanctions Overlay**

A multi-layer analysis of illicit oil trade networks combining OFAC sanctions data, UN Comtrade bilateral flows (2019–2024), and graph-theoretic "shadow hub" detection. Built as a final project for *Social Media & Network Analytics — Spring 2026*.

> **Shadow hubs** are countries whose betweenness centrality in the oil trade network is anomalously high relative to their trade volume — structurally positioned as intermediaries in ways that merit further investigation for potential sanctions circumvention.

---

## 🌐 Live Demo

**[Launch the Interactive Globe →](https://ahmerrill.github.io/shadow-hubs/notebooks/globe_viz.html)**

Click any country dot to reveal its trade flows. Switch years (2019–2024) to see how the network shifts post-sanctions.

---

## 📂 Repository Structure

```
shadow-hubs/
├── README.md
├── .env.example              ← credential template (copy to .env)
├── .gitignore
├── data/
│   ├── edges_country_oil_2019plus.csv    (45K directed trade flows)
│   ├── nodes_country_oil_2019plus.csv    (232 countries with OFAC counts)
│   ├── ofac_country_agg.csv              (OFAC entity aggregation)
│   └── shadow_hubs_residual_2019plus.csv (413 shadow hub scores)
├── notebooks/
│   ├── SMA-Final Project.ipynb           (main analysis notebook)
│   ├── auradb_load.cypher                (Neo4j load scripts + demo queries)
│   ├── globe_viz.html                    (3D interactive visualization)
│   └── VISUALIZATION_README.md           (globe technical docs)
└── docs/
    ├── Data dictionary - SMA Final Project.docx
    └── HANDOFF README- Final Project.docx
```

---

## 🚀 Quick Links

| Resource | Link |
|----------|------|
| **Analysis Notebook** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AHMerrill/shadow-hubs/blob/main/notebooks/SMA-Final%20Project.ipynb) |
| **Presentation** | [View in Canva](https://www.canva.com/design/DAHCviKzIKc/qz2ByeEa35NoiJhALUG57A/view) |
| **Interactive Globe** | [Launch in Browser](https://ahmerrill.github.io/shadow-hubs/notebooks/globe_viz.html) |
| **Neo4j Load Scripts** | [`notebooks/auradb_load.cypher`](notebooks/auradb_load.cypher) |
| **Data Dictionary** | [`docs/Data dictionary - SMA Final Project.docx`](docs/Data%20dictionary%20-%20SMA%20Final%20Project.docx) |
| **Datasets** | [`data/`](data/) |

---

## 🔬 Methodology

### Data Pipeline

1. **OFAC SDN List** — Parsed the U.S. Treasury's Specially Designated Nationals list, aggregating sanctioned entities by country jurisdiction. Note: OFAC targets *entities* (individuals, companies, vessels), not countries per se.

2. **UN Comtrade** — Pulled bilateral oil trade flows (HS 2709–2710) for 230+ countries across 2019–2024 via the Comtrade API.

3. **Network Construction** — Built directed weighted graphs per year, pruned noise edges, computed betweenness centrality, degree, and total trade volume per node.

4. **Shadow Hub Detection** — Regressed log-betweenness on log-volume; countries with high positive residuals are structurally positioned as intermediaries beyond what their trade volume would predict. These are "shadow hubs" — not necessarily illicit, but warranting further investigation.

### Graph Database

All data loaded into **Neo4j AuraDB** (free tier) for Cypher-based querying:

- **232** Country nodes with OFAC metadata
- **45,513** directed TRADE edges (year-partitioned)
- **6** Year nodes with **413** SHADOW_HUB relationships

See [`auradb_load.cypher`](notebooks/auradb_load.cypher) for the complete load script and 8 demo queries (top shadow hubs, OFAC-exposed hubs, emerging hubs post-2022, country trajectory over time, etc.).

### Interactive Visualization

A self-contained [3D globe](https://ahmerrill.github.io/shadow-hubs/notebooks/globe_viz.html) built with [Globe.gl](https://globe.gl/) showing:

- **Country dots** sized by shadow hub rank, colored by OFAC exposure
- **Click-to-reveal** trade flow arcs for any country
- **Year selector** (2019–2024) to watch network evolution
- **Country borders** via TopoJSON overlay

---

## 🛠️ Setup

### Prerequisites

- Python 3.9+
- [UN Comtrade API key](https://comtradeplus.un.org/) (free)
- [Neo4j AuraDB](https://console.neo4j.io/) instance (free tier)

### Credentials

```bash
cp .env.example .env
# Edit .env with your API keys
```

The notebook reads credentials from environment variables via `python-dotenv` — no hardcoded secrets.

### Running the Notebook

The fastest way is via Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AHMerrill/shadow-hubs/blob/main/notebooks/SMA-Final%20Project.ipynb)

Or locally:

```bash
pip install jupyter pandas neo4j python-dotenv comtradeapicall
jupyter notebook notebooks/SMA-Final\ Project.ipynb
```

### Loading Neo4j

1. Create a free AuraDB instance at [console.neo4j.io](https://console.neo4j.io)
2. Add your credentials to `.env`
3. Run the Cypher scripts in [`auradb_load.cypher`](notebooks/auradb_load.cypher) in order (Steps 1–4)
4. Verify with Step 5 queries

---

## 📊 Key Datasets

| File | Rows | Description |
|------|------|-------------|
| `edges_country_oil_2019plus.csv` | 45,513 | Directed oil trade flows (exporter → importer, USD value, by year) |
| `nodes_country_oil_2019plus.csv` | 232 | Countries with ISO codes, OFAC entity counts |
| `ofac_country_agg.csv` | 177 | OFAC sanctions entity aggregation by country |
| `shadow_hubs_residual_2019plus.csv` | 413 | Shadow hub scores: betweenness, residual, rank, trade volume |

---

## 👥 Team

Keerti Rawat · Muskan Khepar · Nikhil Kumar · Stiles Clements · Zan Merrill

**Course:** Social Media & Network Analytics — Spring 2026

---

## 📝 License

Academic project — created for educational analysis of global oil trade networks. Uses [Globe.gl](https://github.com/vasturiano/globe.gl) (MIT License) and [three-globe](https://github.com/vasturiano/three-globe) (MIT License).
