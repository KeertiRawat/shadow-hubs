// =============================================================
// AuraDB Load Scripts — Shadow Hubs in Global Oil Trade
// =============================================================
//
// Google Drive direct download URLs (already populated).
// Run each section in order in the AuraDB Query tab.
// =============================================================
//
// FILE ID REFERENCE:
//   edges_country_oil_2019plus.csv   → 1z9NYXXoq1kgi94HNLqgZXVRxatf_IWIp
//   nodes_country_oil_2019plus.csv   → 1RIT8qmmB9kZ0Cl5u-1_ntKqzUY9qJJNU
//   ofac_country_agg.csv             → 1tQq8kOaxhoETxfMTJfnSpQIcy8gWn6iD
//   shadow_hubs_residual_2019plus.csv→ 1S-HzP1tkNdrzV_YcxoHgzb3ZmrDifMtt


// =============================================================
// STEP 1: CONSTRAINTS & INDEXES
// Run these one at a time.
// =============================================================

CREATE CONSTRAINT country_iso3_unique IF NOT EXISTS
FOR (c:Country) REQUIRE c.iso3 IS UNIQUE;

CREATE CONSTRAINT year_unique IF NOT EXISTS
FOR (y:Year) REQUIRE y.year IS UNIQUE;

CREATE INDEX country_name_idx IF NOT EXISTS
FOR (c:Country) ON (c.name);

CREATE INDEX trade_year_idx IF NOT EXISTS
FOR ()-[t:TRADE]-() ON (t.year);


// =============================================================
// STEP 2: LOAD COUNTRY NODES
// ~232 nodes from nodes_country_oil_2019plus.csv
// =============================================================

LOAD CSV WITH HEADERS FROM 'https://drive.google.com/uc?export=download&id=1RIT8qmmB9kZ0Cl5u-1_ntKqzUY9qJJNU' AS row
WITH row WHERE row.is_group = 'False'
MERGE (c:Country {iso3: row.country_iso3})
SET c.name           = row.country_name,
    c.iso2           = row.iso2,
    c.comtrade_code  = toInteger(row.code),
    c.ofac_entities  = toInteger(row.ofac_entities),
    c.ofac_links     = toInteger(row.ofac_entity_country_links);

// Also load groups/regions as a separate label if you want them:
LOAD CSV WITH HEADERS FROM 'https://drive.google.com/uc?export=download&id=1RIT8qmmB9kZ0Cl5u-1_ntKqzUY9qJJNU' AS row
WITH row WHERE row.is_group = 'True'
MERGE (c:Country:RegionGroup {iso3: row.country_iso3})
SET c.name           = row.country_name,
    c.iso2           = row.iso2,
    c.comtrade_code  = toInteger(row.code),
    c.ofac_entities  = toInteger(row.ofac_entities),
    c.ofac_links     = toInteger(row.ofac_entity_country_links);


// =============================================================
// STEP 3: LOAD TRADE EDGES
// ~45K directed edges from edges_country_oil_2019plus.csv
//
// NOTE: This is the biggest load. AuraDB handles it fine but it
// may take 30-60 seconds. Use periodic commit for safety.
// =============================================================

:auto
LOAD CSV WITH HEADERS FROM 'https://drive.google.com/uc?export=download&id=1z9NYXXoq1kgi94HNLqgZXVRxatf_IWIp' AS row
CALL {
  WITH row
  MATCH (exp:Country {iso3: row.exporter_iso3})
  MATCH (imp:Country {iso3: row.importer_iso3})
  CREATE (exp)-[:TRADE {
    year:            toInteger(row.year),
    trade_value_usd: toFloat(row.trade_value_usd)
  }]->(imp)
} IN TRANSACTIONS OF 5000 ROWS;


// =============================================================
// STEP 4: CREATE YEAR NODES + SHADOW_HUB RELATIONSHIPS
// ~413 rows from shadow_hubs_residual_2019plus.csv
//
// Data model (per README):
//   (:Country)-[:SHADOW_HUB {metrics...}]->(:Year)
// =============================================================

// 4a. Create Year nodes
LOAD CSV WITH HEADERS FROM 'https://drive.google.com/uc?export=download&id=1S-HzP1tkNdrzV_YcxoHgzb3ZmrDifMtt' AS row
WITH DISTINCT toInteger(row.year) AS yr
MERGE (:Year {year: yr});

// 4b. Create SHADOW_HUB relationships
LOAD CSV WITH HEADERS FROM 'https://drive.google.com/uc?export=download&id=1S-HzP1tkNdrzV_YcxoHgzb3ZmrDifMtt' AS row
MATCH (c:Country {iso3: row.iso3})
MATCH (y:Year {year: toInteger(row.year)})
CREATE (c)-[:SHADOW_HUB {
  betweenness:     toFloat(row.betweenness),
  trade_total_usd: toFloat(row.trade_total_usd),
  in_deg:          toInteger(row.in_deg),
  out_deg:         toInteger(row.out_deg),
  log_btw:         toFloat(row.log_btw),
  log_vol:         toFloat(row.log_vol),
  btw_expected:    toFloat(row.btw_expected),
  shadow_resid:    toFloat(row.shadow_resid),
  shadow_rank:     toFloat(row.shadow_rank)
}]->(y);


// =============================================================
// STEP 5: VERIFICATION QUERIES
// Run these to confirm everything loaded correctly.
// =============================================================

// Count nodes and relationships
MATCH (c:Country) RETURN 'Countries' AS label, count(c) AS count
UNION ALL
MATCH (y:Year) RETURN 'Years' AS label, count(y) AS count;

MATCH ()-[t:TRADE]->() RETURN 'TRADE edges' AS label, count(t) AS count
UNION ALL
MATCH ()-[s:SHADOW_HUB]->() RETURN 'SHADOW_HUB edges' AS label, count(s) AS count;

// Check year coverage
MATCH ()-[t:TRADE]->()
RETURN DISTINCT t.year AS year, count(t) AS edges
ORDER BY year;


// =============================================================
// STEP 6: DEMO QUERIES
// From the README — these are the kinds of questions your
// GraphRAG pipeline will eventually answer.
// =============================================================

// Q1: Top 10 shadow hubs in a given year by residual
MATCH (c:Country)-[s:SHADOW_HUB]->(y:Year {year: 2024})
RETURN c.iso3 AS country, c.name AS name,
       s.shadow_resid AS residual, s.shadow_rank AS rank,
       s.betweenness AS betweenness,
       s.trade_total_usd AS trade_volume,
       c.ofac_entities AS ofac_entities
ORDER BY s.shadow_rank ASC
LIMIT 10;

// Q2: Top exporters by total trade value in a year
MATCH (exp:Country)-[t:TRADE {year: 2024}]->(imp:Country)
WITH exp, sum(t.trade_value_usd) AS total_exports, count(imp) AS partners
RETURN exp.iso3 AS country, exp.name AS name,
       total_exports, partners
ORDER BY total_exports DESC
LIMIT 10;

// Q3: Top importers by total trade value in a year
MATCH (exp:Country)-[t:TRADE {year: 2024}]->(imp:Country)
WITH imp, sum(t.trade_value_usd) AS total_imports, count(exp) AS suppliers
RETURN imp.iso3 AS country, imp.name AS name,
       total_imports, suppliers
ORDER BY total_imports DESC
LIMIT 10;

// Q4: Subgraph around a specific country (e.g., USA)
MATCH (c:Country {iso3: 'USA'})-[t:TRADE {year: 2024}]-(partner:Country)
RETURN c.name AS country, type(t) AS rel,
       CASE WHEN startNode(t) = c THEN 'exports_to' ELSE 'imports_from' END AS direction,
       partner.iso3 AS partner_iso3, partner.name AS partner_name,
       t.trade_value_usd AS value_usd
ORDER BY t.trade_value_usd DESC
LIMIT 20;

// Q5: Shadow hubs with highest OFAC exposure
MATCH (c:Country)-[s:SHADOW_HUB]->(y:Year {year: 2024})
WHERE c.ofac_entities > 0
RETURN c.iso3 AS country, c.name AS name,
       c.ofac_entities AS ofac_entities,
       s.shadow_resid AS residual, s.shadow_rank AS rank
ORDER BY s.shadow_rank ASC
LIMIT 10;

// Q6: Countries that became shadow hubs after 2022
//     (not in top 20 in 2019-2022, but top 20 in 2023 or 2024)
MATCH (c:Country)-[s:SHADOW_HUB]->(y:Year)
WITH c, y.year AS yr, s.shadow_rank AS rank
WITH c,
     collect(CASE WHEN yr <= 2022 AND rank <= 20 THEN yr END) AS early_top,
     collect(CASE WHEN yr >= 2023 AND rank <= 20 THEN yr END) AS late_top
WHERE size(early_top) = 0 AND size(late_top) > 0
RETURN c.iso3 AS country, c.name AS name, late_top AS top_years
ORDER BY c.iso3;

// Q7: How did a country's hub rank change over time?
MATCH (c:Country {iso3: 'USA'})-[s:SHADOW_HUB]->(y:Year)
RETURN y.year AS year, s.shadow_rank AS rank,
       s.shadow_resid AS residual, s.betweenness AS betweenness,
       s.trade_total_usd AS trade_volume
ORDER BY y.year;

// Q8: High residual but moderate volume (the "true" shadow hubs)
MATCH (c:Country)-[s:SHADOW_HUB]->(y:Year {year: 2024})
WHERE s.shadow_resid > 0
WITH c, s,
     percentileCont(s.trade_total_usd, 0.5) AS median_vol
RETURN c.iso3 AS country, c.name AS name,
       s.shadow_resid AS residual,
       s.trade_total_usd AS trade_volume,
       c.ofac_entities AS ofac_entities
ORDER BY s.shadow_resid DESC
LIMIT 15;
