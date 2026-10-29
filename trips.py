"""DuckDB views over the cleaned monthly trip files (used by 07_censoring.py).

- `station_alias(id, canon)`: merges the paired stations that operate as one site (271 and 272 -> 271-272).
- `trips_all`: all trips with canonical station ids and system ('previous', 'new' or 'ambiguous').
- `trips_new`: trips of the new system, the ones analysed.

System rule: before NEW_SYSTEM_START every trip belongs to the previous system and after
OLD_SYSTEM_END to the new one; in between, a trip belongs to the new system if its bicycle has trips
after the closure and none before the start, and to the previous system in the opposite case.
"""
from __future__ import annotations

import re

import config


def lit(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def create_views(con, extra_pairs: list[str] | None = None) -> None:
    pattern = lit(str(config.TRIPS / "*.parquet"))
    con.execute(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM read_parquet({pattern})")

    pairs = {r[0] for r in con.execute("""
        SELECT DISTINCT est_retiro FROM raw WHERE regexp_full_match(est_retiro, '[0-9]+-[0-9]+')
        UNION SELECT DISTINCT est_arribo FROM raw WHERE regexp_full_match(est_arribo, '[0-9]+-[0-9]+')
    """).fetchall() if r[0]}
    pairs |= {p for p in (extra_pairs or []) if isinstance(p, str) and re.fullmatch(r"\d+-\d+", p)}
    rows = [(part, pair) for pair in sorted(pairs) for part in pair.split("-") if part]
    con.execute("CREATE OR REPLACE TEMP TABLE station_alias (id VARCHAR, canon VARCHAR)")
    if rows:
        con.executemany("INSERT INTO station_alias VALUES (?, ?)", rows)

    start, end = lit(config.NEW_SYSTEM_START), lit(config.OLD_SYSTEM_END)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE bikes AS
        SELECT bici,
               bool_or(ts_retiro < TIMESTAMP {start}) AS before_start,
               bool_or(ts_retiro > TIMESTAMP {end} + INTERVAL 1 DAY) AS after_end
        FROM raw WHERE bici IS NOT NULL GROUP BY bici
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW trips_all AS
        SELECT v.ts_retiro, v.ts_arribo, v.bici,
               coalesce(a1.canon, v.est_retiro) AS est_retiro,
               coalesce(a2.canon, v.est_arribo) AS est_arribo,
               CASE
                 WHEN v.ts_retiro < TIMESTAMP {start} THEN 'previous'
                 WHEN v.ts_retiro > TIMESTAMP {end} + INTERVAL 1 DAY THEN 'new'
                 WHEN b.after_end AND NOT b.before_start THEN 'new'
                 WHEN b.before_start AND NOT b.after_end THEN 'previous'
                 ELSE 'ambiguous'
               END AS system
        FROM raw v
        LEFT JOIN station_alias a1 ON a1.id = v.est_retiro
        LEFT JOIN station_alias a2 ON a2.id = v.est_arribo
        LEFT JOIN bikes b USING (bici)
    """)
    con.execute("CREATE OR REPLACE VIEW trips_new AS SELECT * FROM trips_all WHERE system = 'new'")
