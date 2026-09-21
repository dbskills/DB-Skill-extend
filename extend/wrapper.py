#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extend index-selection wrapper (long-lived HTTP server).

Implements the Extend heuristic (Schlosser, Kossmann, and Boissier, ICDE 2019)
behind the index-selection HTTP protocol: GET /health, POST /recommend,
GET /state, POST /shutdown.

Extend is a deterministic, cost-based heuristic. It builds an index
configuration greedily in rounds: each round it evaluates every single-column
candidate and every one-column extension of an already-chosen index (multi-
column, up to max_index_width), picks the candidate with the best
benefit-to-size ratio that yields at least min_cost_improvement over the
current cost, adds it, and repeats until the storage/count budget is exceeded
or no further improvement is possible.

Candidate evaluation uses HypoPG hypothetical indexes (cost-only EXPLAIN) on
--eval-dsn (defaults to --dsn); no real indexes are created on the main DSN
unless --apply is set. There is no online training or model state; state is
limited to request counters and the last recommendation, persisted on graceful
shutdown and rehydrated on restart.
"""
import sys
import os
import json
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
import threading
import traceback
import types
import importlib
import importlib.abc
import importlib.util
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# ---------------------------------------------------------------------------
# Stub unneeded modules before importing the algorithm.
#
# benefit_estimation and dqn_selection are replaced by empty packages with
# no-op loader functions, and psqlparse by an empty module, so the image does
# not need torch, sklearn, or a psqlparse build.
# ---------------------------------------------------------------------------
_BLOCKED_PREFIXES = (
    "index_advisor_selector.index_benefit_estimation",
    "index_advisor_selector.index_selection.dqn_selection",
)
_STUB_NAMES = (
    "load_model_tree", "get_tree_est_res",
    "load_model_lib", "get_lib_est_res",
    "load_model_former", "get_former_est_res",
)


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        for prefix in _BLOCKED_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []  # mark as a package so deeper imports also resolve to stubs
        m.__package__ = spec.name
        return m

    def exec_module(self, module):
        for name in _STUB_NAMES:
            setattr(module, name, lambda *a, **kw: None)


def _install_stubs():
    sys.meta_path.insert(0, _StubFinder())
    if "psqlparse" not in sys.modules:
        _ps = types.ModuleType("psqlparse")
        _ps.parse_dict = lambda q: []
        sys.modules["psqlparse"] = _ps


_install_stubs()

REPO = os.environ.get("EXTEND_REPO", "/app/Index_EAB")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import psycopg2  # noqa: E402
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import (  # noqa: E402
    PostgresDatabaseConnector,
)
from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com  # noqa: E402
from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import (  # noqa: E402
    Workload, Query,
)
from index_advisor_selector.index_selection.heu_selection.heu_utils.cost_evaluation import (  # noqa: E402
    CostEvaluation,
)
from index_advisor_selector.index_selection.heu_selection.heu_algos.extend_algorithm import (  # noqa: E402
    ExtendAlgorithm,
)

# SelectionAlgorithm.__init__ calls drop_indexes(); make it a no-op so the
# target DSN is never mutated (runtime patch only).
PostgresDatabaseConnector.drop_indexes = lambda self: None

DEFAULT_STATE_DIR = os.path.join(REPO, "skill_state")
DEFAULT_MAX_INDEX_WIDTH = 2
DEFAULT_BUDGET_MB = 1000.0
DEFAULT_MAX_INDEXES = 10
DEFAULT_MIN_COST_IMPROVEMENT = 1.003
DEFAULT_TIME_BUDGET_SECONDS = 60  # 0 = unlimited
DEFAULT_EVAL_SAMPLE_WORKLOADS = 3
DEFAULT_EVAL_SAMPLE_SIZE = 200  # per-sample cap
DEFAULT_EVAL_SAMPLE_FRACTION = 0.2  # of the workload, per sample
DEFAULT_EVAL_SAMPLE_MIN = 30
DEFAULT_EVAL_SAMPLE_SEED = 60
DEFAULT_CANDIDATE_LIMIT = 8  # 0 = evaluate every candidate
DEFAULT_EVAL_PARALLELISM = 8  # eval-DB sessions; 0/1 = sequential


def _index_key(ind):
    return (ind.table().name, tuple(c.name for c in ind.columns))


def _prefixes_from_existing(existing_cols):
    """All leftmost prefixes of existing real indexes. A candidate matching one
    is redundant (a btree also serves prefix predicates), so it is masked out
    of the algorithm's candidate set instead of being emitted and skipped."""
    prefixes = set()
    for table, cols in existing_cols:
        for k in range(1, len(cols) + 1):
            prefixes.add((table, tuple(cols[:k])))
    return prefixes

_LITERAL_RE = re.compile(r"'[^']*'")
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


_QUALIFIED_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*)?$")
_QUOTED_IDENT_RE = re.compile(r'^"((?:[^"]|"")*)"(.*)$', re.DOTALL)


def _parse_index_key_part(part):
    """One key-column entry of a pg_indexes.indexdef -> lower-case column name,
    or None for an expression. Strips [COLLATE <name>] [opclass]
    [ASC|DESC] [NULLS FIRST|LAST]."""
    s = part.strip()
    if not s:
        return None
    m = _QUOTED_IDENT_RE.match(s)
    if m:
        name = m.group(1).replace('""', '"')
        rest = m.group(2).strip()
        tokens = rest.split() if rest else []
        quoted = True
    else:
        tokens = s.split()
        name = tokens[0]
        tokens = tokens[1:]
        quoted = False
    opclass_seen = False
    i = 0
    while i < len(tokens):
        tok = tokens[i].upper()
        if tok == "COLLATE":
            i += 2            # COLLATE <collation name>
            continue
        if tok == "NULLS":
            i += 2            # NULLS FIRST|LAST
            continue
        if tok in ("ASC", "DESC"):
            i += 1
            continue
        if not opclass_seen and _QUALIFIED_IDENT_RE.match(tokens[i].strip('"')):
            opclass_seen = True   # e.g. text_pattern_ops / pg_catalog.int4_ops
            i += 1
            continue
        return None
    if not name or (not quoted and not _IDENT_RE.match(name)):
        return None
    return name.lower()


def _parse_indexdef_key_columns(indexdef):
    """Key (btree) columns of a pg_indexes.indexdef; [] when the key cannot be
    read with certainty (expression or non-btree index). INCLUDE is not part
    of the key."""
    if not indexdef:
        return []
    m = re.search(r"\bON\b\s+\S+\s+USING\s+btree\s*\(([^)]*)\)", indexdef, re.I)
    if not m:
        m = re.search(r"\bON\b\s+\S+\s*\(([^)]*)\)", indexdef, re.I)
    if not m:
        return []
    cols = []
    for part in m.group(1).split(","):
        if not part.strip():
            continue
        name = _parse_index_key_part(part)
        if name is None:
            return []  # expression key — do not guess
        cols.append(name)
    return cols


def _word_re(name):
    return re.compile(r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])")


def _strip_sql(text):
    text = _BLOCK_COMMENT_RE.sub(" ", text)
    text = _LINE_COMMENT_RE.sub(" ", text)
    text = _LITERAL_RE.sub(" ", text)
    return text.lower()


def _split_statements(content):
    content = content.strip()
    if not content:
        return []
    out = []
    for part in content.split(";"):
        p = part.strip()
        if p:
            out.append(p)
    return out


_DIRECTIVE_RE = re.compile(r"^:[A-Za-z]$")


def _clean_stmt(text):
    """Strip SQL line comments and dbgen-style directives (:x/:o) so the
    remaining text begins with the actual SQL verb. Returns the cleaned
    statement (comments/directives removed)."""
    kept = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("--") or _DIRECTIVE_RE.match(s):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _is_read_query(text):
    t = _clean_stmt(text).lstrip().lower()
    return t.startswith("select") or t.startswith("with")


def _parse_dsn(dsn):
    return psycopg2.extensions.parse_dsn(dsn)


def _b_to_mb(b):
    # 1 MB = 1e6 bytes.
    return (b or 0) / 1_000_000.0


class _TimeBudgetExceeded(Exception):
    """Raised by the patched _evaluate_combination once the request's
    wall-clock budget is exhausted, so the wrapper can return the
    best-so-far configuration instead of grinding on."""


class ExtendServer:
    def __init__(self, state_dir=DEFAULT_STATE_DIR):
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._connectors = {}        # dsn -> PostgresDatabaseConnector
        self._schema_cache = {}      # dsn -> (tables, columns, existing)
        self._table_size_cache = {}  # dsn -> {table_name: bytes}
        self._col_types_cache = {}    # dsn -> {(table,col): pg_type}
        self._max_bytes_cache = {}    # (dsn,table,col) -> max octet_length
        self.state_dir = state_dir
        self._counters = {
            "total_requests": 0,
            "total_queries_analyzed": 0,
            "last_workload_size": 0,
            "last_num_indexes": 0,
            "last_optimization_time": 0.0,
            "last_estimated_impact": 0.0,
        }
        self._last_recommendation = []
        self._last_config = {}
        self._shutdown = False
        self._load_state()

    # -- persistence --------------------------------------------------------
    def _state_files(self):
        return (
            os.path.join(self.state_dir, "counters.json"),
            os.path.join(self.state_dir, "last_recommendation.json"),
        )

    def _load_state(self):
        try:
            cf, rf = self._state_files()
            if os.path.isfile(cf):
                with open(cf) as fh:
                    saved = json.load(fh)
                for k, v in saved.items():
                    if k in self._counters:
                        self._counters[k] = v
            if os.path.isfile(rf):
                with open(rf) as fh:
                    self._last_recommendation = json.load(fh)
        except Exception:
            pass  # fresh start on any corruption

    def _persist_state(self):
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            cf, rf = self._state_files()
            tmp_cf = cf + ".tmp"
            with open(tmp_cf, "w") as fh:
                json.dump(self._counters, fh, indent=2)
            os.replace(tmp_cf, cf)
            tmp_rf = rf + ".tmp"
            with open(tmp_rf, "w") as fh:
                json.dump(self._last_recommendation, fh, indent=2)
            os.replace(tmp_rf, rf)
        except Exception:
            pass

    # -- connections & schema ---------------------------------------------
    def _get_connector(self, dsn):
        conn = self._connectors.get(dsn)
        if conn is not None:
            try:
                cur = conn._cursor
                cur.execute("select 1")
                cur.fetchone()
                return conn
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                self._connectors.pop(dsn, None)
                self._schema_cache.pop(dsn, None)
                self._table_size_cache.pop(dsn, None)
        d = _parse_dsn(dsn)
        cfg = {"postgresql": d}
        conn = PostgresDatabaseConnector(
            cfg, autocommit=True,
            host=d.get("host"), port=str(d.get("port")),
            db_name=d.get("dbname"), user=d.get("user"),
            password=d.get("password"),
        )
        try:
            conn._cursor.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
        except Exception:
            pass  # surface later if a what-if call actually fails
        conn.drop_hypo_indexes()
        self._connectors[dsn] = conn
        return conn

    def _make_fresh_connector(self, dsn):
        """Create a brand-new connector (own PG session); the caller must close it.

        A fresh session per /recommend bounds server-side HypoPG state.
        """
        d = _parse_dsn(dsn)
        cfg = {"postgresql": d}
        conn = PostgresDatabaseConnector(
            cfg, autocommit=True,
            host=d.get("host"), port=str(d.get("port")),
            db_name=d.get("dbname"), user=d.get("user"),
            password=d.get("password"),
        )
        try:
            conn._cursor.execute("CREATE EXTENSION IF NOT EXISTS hypopg")
        except Exception:
            pass  # surface later if a what-if call actually fails
        conn.drop_hypo_indexes()
        return conn

    def _hypo_count(self, conn):
        try:
            conn._cursor.execute("select count(*) from hypopg_list_indexes()")
            return conn._cursor.fetchone()[0]
        except Exception:
            try:
                conn._cursor.execute("select count(*) from hypopg_list_indexes")
                return conn._cursor.fetchone()[0]
            except Exception:
                return None

    def _introspect(self, dsn):
        cached = self._schema_cache.get(dsn)
        if cached is not None:
            return cached
        conn = self._get_connector(dsn)
        tables, columns = heu_com.get_columns_from_db(conn)
        existing = self._existing_indexes(conn)
        sizes = self._table_sizes(conn, tables)
        col_types = self._col_types(conn)
        self._schema_cache[dsn] = (tables, columns, existing)
        self._table_size_cache[dsn] = sizes
        self._col_types_cache[dsn] = col_types
        return tables, columns, existing

    def _existing_indexes(self, conn):
        """Return (set of (table, tuple(cols)), set of index_names)."""
        cols_set, names = set(), set()
        try:
            conn._cursor.execute(
                "select schemaname, tablename, indexname, indexdef "
                "from pg_indexes where schemaname = 'public'"
            )
            rows = conn._cursor.fetchall()
        except Exception:
            return cols_set, names
        for _schema, table, name, indexdef in rows:
            names.add(name.lower())
            colparts = _parse_indexdef_key_columns(indexdef)
            if colparts:
                cols_set.add((table.lower(), tuple(colparts)))
        return cols_set, names

    def _table_sizes(self, conn, tables):
        sizes = {}
        for t in tables:
            try:
                conn._cursor.execute(
                    "select pg_total_relation_size(%s)", (t.name,)
                )
                sizes[t.name] = conn._cursor.fetchone()[0]
            except Exception:
                sizes[t.name] = 0
        return sizes

    def _col_types(self, conn):
        types = {}
        try:
            conn._cursor.execute(
                "select table_name, column_name, data_type, character_maximum_length "
                "from information_schema.columns where table_schema = 'public'"
            )
            for tbl, col, dt, cml in conn._cursor.fetchall():
                types[(tbl.lower(), col.lower())] = (dt, cml)
        except Exception:
            pass
        return types

    _BTREE_KEY_LIMIT = 2700  # PG btree index-row safety threshold (~8191 B block)

    def _max_bytes(self, dsn, table, col):
        """Cached max octet_length of a column, for buildability probing."""
        key = (dsn, table, col)
        if key in self._max_bytes_cache:
            return self._max_bytes_cache[key]
        conn = self._get_connector(dsn)
        val = 0
        try:
            conn._cursor.execute("set statement_timeout = 60000")
            conn._cursor.execute(
                'select max(octet_length("%s")) from "%s"' % (col, table)
            )
            row = conn._cursor.fetchone()
            val = (row[0] or 0) if row else 0
        except Exception:
            # Timeout / error -> assume large (conservative skip) so we never
            # emit a DDL that will fail at CREATE INDEX time.
            val = self._BTREE_KEY_LIMIT + 1
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            try:
                conn._cursor.execute("set statement_timeout = 0")
            except Exception:
                pass
        self._max_bytes_cache[key] = val
        return val

    def _index_buildable(self, dsn, table, cols, col_types):
        """Return (buildable, reason). Skip indexes whose key columns include an
        unbounded text/bytea/json column with a value exceeding the btree
        index-row limit (CREATE INDEX would otherwise fail at apply time)."""
        for c in cols:
            dtype, cml = col_types.get((table, c), (None, None))
            if dtype in ("text", "bytea", "json", "xml") or (
                dtype == "character varying" and cml is None
            ):
                if self._max_bytes(dsn, table, c) > self._BTREE_KEY_LIMIT:
                    return False, (
                        f"index on {table}({','.join(cols)}) has unbounded "
                        f"{dtype} column '{c}' exceeding btree key-size limit; skipped"
                    )
        return True, None

    # -- workload construction --------------------------------------------
    def _populate_columns(self, text, columns):
        stripped = _strip_sql(text)
        matched = []
        for col in columns:
            tn = col.table.name
            if _word_re(tn).search(stripped) and _word_re(col.name).search(stripped):
                matched.append(col)
        return matched

    def _build_workload(self, workload_field, columns):
        """Return (Workload, num_analyzed, notes)."""
        entries = self._normalize_workload(workload_field)
        queries = []
        notes = []
        analyzed = 0
        for idx, (sql, weight) in enumerate(entries, start=1):
            for stmt in _split_statements(sql):
                stmt = _clean_stmt(stmt)
                if not stmt:
                    continue
                if not _is_read_query(stmt):
                    notes.append(f"Q{idx}: non-SELECT excluded from read-cost")
                    continue
                cols = self._populate_columns(stmt, columns)
                if not cols:
                    notes.append(f"Q{idx}: no schema columns matched (absent table or unparseable)")
                    continue
                q = Query(idx, stmt, columns=cols, frequency=max(1, int(weight or 1)))
                queries.append(q)
                analyzed += 1
        if not queries:
            return None, 0, notes
        return Workload(queries), analyzed, notes

    def _query_shape(self, q):
        return tuple(sorted((c.table.name.lower(), c.name.lower())
                            for c in q.columns))

    def _limit_candidates(self, candidates, workload, limit):
        """Keep the most-referenced candidates, with one per table first."""
        if limit <= 0 or len(candidates) <= limit:
            return candidates
        scores = {}
        for q in workload.queries:
            try:
                freq = max(1, int(getattr(q, "frequency", 1) or 1))
            except (TypeError, ValueError):
                freq = 1
            for col in q.columns:
                key = (col.table.name.lower(), col.name.lower())
                scores[key] = scores.get(key, 0) + freq

        def key(ind):
            col = ind.columns[0]
            return (scores.get((col.table.name.lower(), col.name.lower()), 0),
                    col.table.name.lower(), col.name.lower())

        ranked = sorted(candidates, key=key, reverse=True)
        selected, tables = [], set()
        for ind in ranked:
            table = ind.columns[0].table.name.lower()
            if table in tables:
                continue
            tables.add(table)
            selected.append(ind)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            chosen = {id(ind) for ind in selected}
            for ind in ranked:
                if id(ind) in chosen:
                    continue
                selected.append(ind)
                chosen.add(id(ind))
                if len(selected) >= limit:
                    break
        return sorted(selected, key=key, reverse=True)

    def _sample_eval_workloads(self, workload, cfg):
        """Independent stratified samples used for the search's cost decisions."""
        try:
            n_workloads = int(cfg.get("eval_sample_workloads",
                                      DEFAULT_EVAL_SAMPLE_WORKLOADS))
        except (TypeError, ValueError):
            n_workloads = DEFAULT_EVAL_SAMPLE_WORKLOADS
        try:
            size_cap = int(cfg.get("eval_sample_size", DEFAULT_EVAL_SAMPLE_SIZE))
        except (TypeError, ValueError):
            size_cap = DEFAULT_EVAL_SAMPLE_SIZE
        try:
            fraction = float(cfg.get("eval_sample_fraction",
                                     DEFAULT_EVAL_SAMPLE_FRACTION))
        except (TypeError, ValueError):
            fraction = DEFAULT_EVAL_SAMPLE_FRACTION
        try:
            size_min = int(cfg.get("eval_sample_min", DEFAULT_EVAL_SAMPLE_MIN))
        except (TypeError, ValueError):
            size_min = DEFAULT_EVAL_SAMPLE_MIN
        try:
            seed = int(cfg.get("eval_sample_seed", DEFAULT_EVAL_SAMPLE_SEED))
        except (TypeError, ValueError):
            seed = DEFAULT_EVAL_SAMPLE_SEED
        qs = list(workload.queries)
        size = size_cap if fraction <= 0 else min(
            size_cap, max(size_min, int(len(qs) * fraction)))
        meta = {"enabled": False, "workloads": 0, "sample_size": 0,
                "seed": seed, "total_queries": len(qs)}
        if n_workloads <= 0 or size <= 0 or len(qs) <= n_workloads * size:
            return None, meta
        rng = random.Random(seed)
        groups = {}
        for q in qs:
            groups.setdefault(self._query_shape(q), []).append(q)
        shapes = sorted(groups)
        weights = {id(q): max(1, int(getattr(q, "frequency", 1) or 1))
                   for q in qs}
        samples = []
        for _ in range(n_workloads):
            chosen = []
            used = set()
            shape_order = list(shapes)
            rng.shuffle(shape_order)
            for shape in shape_order:
                if len(chosen) >= size:
                    break
                candidates = groups[shape]
                pick = candidates[rng.randrange(len(candidates))]
                if id(pick) not in used:
                    used.add(id(pick))
                    chosen.append(pick)
            remaining = [q for q in qs if id(q) not in used]
            need = size - len(chosen)
            if need > 0:
                if need >= len(remaining):
                    chosen.extend(remaining)
                else:
                    keyed = [(rng.random() ** (1.0 / weights[id(q)]), q)
                             for q in remaining]
                    keyed.sort(key=lambda kv: kv[0], reverse=True)
                    chosen.extend(q for _, q in keyed[:need])
            samples.append(Workload(chosen))
        meta.update({"enabled": True, "workloads": len(samples),
                     "sample_size": size})
        return samples, meta

    def _normalize_workload(self, workload_field):
        """Return list of (sql_text, weight)."""
        if workload_field is None:
            return []
        if isinstance(workload_field, list):
            out = []
            for item in workload_field:
                if isinstance(item, dict):
                    out.append((item.get("sql", ""), item.get("weight", item.get("frequency", 1))))
                elif isinstance(item, str):
                    out.append((item, 1))
            return out
        if isinstance(workload_field, str):
            s = workload_field.strip()
            if not s:
                return []
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    return self._normalize_workload(parsed)
                except Exception:
                    pass
            if os.path.isfile(s):
                try:
                    with open(s) as fh:
                        return [(fh.read(), 1)]
                except Exception:
                    pass
            return [(s, 1)]
        return []

    # -- recommendation ----------------------------------------------------
    def recommend(self, req):
        t_start = time.time()
        dsn = req.get("dsn")
        if not dsn:
            return self._degrade("missing --dsn", [], {}, 0)
        workload_field = req.get("workload")
        eval_dsn = req.get("eval_dsn") or dsn
        budget = req.get("budget") or {}
        cfg = self._parse_config(req.get("config"))
        apply_flag = bool(req.get("apply"))

        with self._lock:
            self._last_config = cfg
            try:
                eval_conn = self._make_fresh_connector(eval_dsn)
            except Exception as e:
                return self._degrade(f"cannot connect to eval-dsn: {e}", [], cfg, 0,
                                     dsn=dsn, eval_dsn=eval_dsn, budget=budget)
            try:
                return self._recommend_locked(dsn, eval_dsn, eval_conn,
                                               workload_field, budget, cfg,
                                               apply_flag, t_start)
            finally:
                # Close the per-request session (releases server-side HypoPG
                # state); probes/--apply use the long-lived DSN connector.
                try:
                    leaked = self._hypo_count(eval_conn)
                except Exception:
                    leaked = None
                if leaked:
                    try:
                        eval_conn.drop_hypo_indexes()
                    except Exception:
                        pass
                try:
                    eval_conn.close()
                except Exception:
                    pass

    def _recommend_locked(self, dsn, eval_dsn, eval_conn, workload_field,
                          budget, cfg, apply_flag, t_start):
        degrade_notes = []
        try:
            _, columns, _ = self._introspect(dsn)
            # Re-read the existing-index set on every request (cheap catalog
            # SELECT; indexes may change between requests).
            existing = self._existing_indexes(self._get_connector(dsn))
        except Exception as e:
            return self._degrade(f"schema introspection failed: {e}", [], cfg, 0,
                                 dsn=dsn, eval_dsn=eval_dsn, budget=budget)

        workload, num_analyzed, wnotes = self._build_workload(workload_field, columns)
        degrade_notes.extend(wnotes)
        if workload is None:
            return self._degrade("no parseable SELECT query with indexable columns",
                                  [], cfg, 0, notes=degrade_notes, dsn=dsn,
                                  eval_dsn=eval_dsn, budget=budget)

        # Mask candidates that are a prefix of an existing real index (the
        # search seed is workload.potential_indexes()).
        _blocked = _prefixes_from_existing(existing[0])
        _orig_potential = workload.potential_indexes
        try:
            candidate_limit = int(cfg.get("candidate_limit",
                                          DEFAULT_CANDIDATE_LIMIT))
        except (TypeError, ValueError):
            candidate_limit = DEFAULT_CANDIDATE_LIMIT

        def _filtered_potential(_orig=_orig_potential, _blocked=_blocked):
            kept = []
            for ind in _orig():
                try:
                    if _index_key(ind) in _blocked:
                        continue
                except Exception:
                    pass
                kept.append(ind)
            if candidate_limit > 0:
                kept = self._limit_candidates(kept, workload, candidate_limit)
            return kept

        workload.potential_indexes = _filtered_potential

        params, cap_count, write_pct = self._build_params(cfg, budget)
        try:
            time_budget = float(cfg.get("time_budget_seconds", DEFAULT_TIME_BUDGET_SECONDS))
        except (TypeError, ValueError):
            time_budget = float(DEFAULT_TIME_BUDGET_SECONDS)
        _t_algo = time.time()
        _stop_watch = threading.Event()
        _watch_state = {"ces": []}

        def _watch():
            while not _stop_watch.wait(15):
                try:
                    ces = _watch_state["ces"]
                    estimates = sum(getattr(ce, "cost_requests", 0) for ce in ces)
                    simulated = sum(
                        len(getattr(getattr(ce, "what_if", None),
                                    "simulated_indexes", {}) or {})
                        for ce in ces)
                    print(f"[watch] {time.time() - _t_algo:.0f}s "
                          f"cost_estimations={estimates} "
                          f"simulated={simulated}", flush=True)
                except Exception:
                    pass

        _watchdog = threading.Thread(target=_watch, daemon=True)
        _watchdog.start()
        # Wall-clock budget: patch _evaluate_combination to raise once the
        # budget is spent; the best-so-far combination is returned (runtime
        # patch only).
        _deadline = None if time_budget <= 0 else _t_algo + time_budget
        _partial = {"best": []}
        _orig_eval = ExtendAlgorithm._evaluate_combination
        sample_workloads, sample_meta = self._sample_eval_workloads(workload, cfg)
        try:
            parallelism = int(cfg.get("eval_parallelism",
                                      DEFAULT_EVAL_PARALLELISM))
        except (TypeError, ValueError):
            parallelism = DEFAULT_EVAL_PARALLELISM
        parallelism = max(1, parallelism)
        _orig_calc = None
        _parallel = None
        _worker_ces = []
        _worker_conns = []
        algo = None

        def _budgeted_eval(algo_self, index_combination, best, current_cost,
                           old_index_size=0):
            if _deadline is not None and time.time() > _deadline:
                raise _TimeBudgetExceeded()
            ratio = _orig_eval(algo_self, index_combination, best, current_cost,
                               old_index_size)
            if best.get("combination"):
                _partial["best"] = best["combination"]
            return ratio

        try:
            eval_conn.drop_hypo_indexes()
            ExtendAlgorithm._evaluate_combination = _budgeted_eval
            algo = ExtendAlgorithm(
                eval_conn, params, process=False,
                cand_gen=None, is_utilized=None, sel_oracle=cfg.get("sel_oracle"),
            )
            _watch_state["ces"] = [algo.cost_evaluation]
            _orig_calc = algo.cost_evaluation.calculate_cost
            cost_workloads = sample_workloads or [workload]
            if parallelism > 1:
                chunk_sets = []
                for cost_workload in cost_workloads:
                    qs = list(cost_workload.queries)
                    k = min(parallelism, max(1, len(qs)))
                    chunk_sets.append([Workload(qs[i::k]) for i in range(k)])
                n_workers = max(len(parts) for parts in chunk_sets)
                for _ in range(n_workers):
                    conn = self._make_fresh_connector(eval_dsn or dsn)
                    _worker_conns.append(conn)
                    _worker_ces.append(CostEvaluation(conn))
                _parallel = ThreadPoolExecutor(max_workers=n_workers)
                _watch_state["ces"] = _worker_ces

                def _worker_eval(idx, indexes, store_size):
                    total = 0.0
                    first_size = store_size
                    for parts in chunk_sets:
                        if idx >= len(parts):
                            continue
                        total += _worker_ces[idx].calculate_cost(
                            parts[idx], indexes, store_size=first_size)
                        first_size = False
                    return total

                def _parallel_cost(_workload, indexes, store_size=False):
                    try:
                        futures = [
                            _parallel.submit(_worker_eval, i, indexes, store_size)
                            for i in range(n_workers)
                        ]
                        return sum(f.result() for f in futures)
                    except Exception as e:
                        print("parallel cost evaluation failed ({}); falling back"
                              .format(e), flush=True)
                        total = 0.0
                        for j, cost_workload in enumerate(cost_workloads):
                            total += _orig_calc(
                                cost_workload, indexes,
                                store_size=(store_size and j == 0))
                        return total

                algo.cost_evaluation.calculate_cost = _parallel_cost
            elif sample_workloads:
                def _sampled_cost(_workload, indexes, store_size=False,
                                  _orig=_orig_calc, _samples=sample_workloads):
                    total = 0.0
                    for i, sample in enumerate(_samples):
                        total += _orig(sample, indexes,
                                       store_size=(store_size and i == 0))
                    return total

                algo.cost_evaluation.calculate_cost = _sampled_cost
            indexes = algo.calculate_best_indexes(workload)
            eval_conn.drop_hypo_indexes()
        except _TimeBudgetExceeded:
            eval_conn.drop_hypo_indexes()
            indexes = list(_partial["best"])
            degrade_notes.append(
                f"time budget of {time_budget:g}s exceeded; returning the "
                f"best-so-far configuration ({len(indexes)} indexes) after "
                f"{time.time() - _t_algo:.1f}s -- raise time_budget_seconds "
                f"or submit a smaller workload for the complete result"
            )
        except Exception as e:
            traceback.print_exc()
            eval_conn.drop_hypo_indexes()
            return self._degrade(f"algorithm failure: {e}", [], cfg, num_analyzed,
                                 notes=degrade_notes, dsn=dsn, eval_dsn=eval_dsn,
                                 budget=budget)
        finally:
            ExtendAlgorithm._evaluate_combination = _orig_eval
            if algo is not None and _orig_calc is not None:
                try:
                    algo.cost_evaluation.calculate_cost = _orig_calc
                except Exception:
                    pass
            if _parallel is not None:
                try:
                    _parallel.shutdown(wait=True)
                except Exception:
                    pass
            for ce in _worker_ces:
                try:
                    ce.complete_cost_estimation()
                except Exception:
                    pass
            for conn in _worker_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            _stop_watch.set()

        table_sizes = self._table_size_cache.get(dsn, {})
        col_types = self._col_types_cache.get(dsn, {})
        rec_list = self._to_ddl(indexes, existing, table_sizes, col_types, dsn,
                                cap_count, write_pct, degrade_notes)

        impact = self._estimate_impact(eval_conn, workload, indexes)
        dt = time.time() - t_start
        budget_used = self._budget_used(budget, rec_list)
        if not rec_list and not any("storage" in n or "budget" in n for n in degrade_notes):
            degrade_notes.append("no beneficial index within budget")

        response = {
            "recommended_indexes": rec_list,
            "metadata": {
                "strategy_type": "cost-based",
                "optimization_time": round(dt, 4),
                "estimated_impact": round(impact, 4),
                "num_queries_analyzed": num_analyzed,
                "budget_used": budget_used,
                "eval_mode": "standby" if eval_dsn != dsn else "hypothetical",
                "mode": "inference-only",
                "eval_sampling": sample_meta,
                "eval_parallelism": parallelism,
                "candidate_limit": candidate_limit,
                "config": {k: cfg[k] for k in sorted(cfg)},
                "degrade_note": "; ".join(degrade_notes) if degrade_notes else None,
            },
        }
        if apply_flag:
            response["apply_results"] = self._apply_indexes(rec_list, dsn)

        with self._state_lock:
            self._counters["total_requests"] += 1
            self._counters["total_queries_analyzed"] += num_analyzed
            self._counters["last_workload_size"] = num_analyzed
            self._counters["last_num_indexes"] = len(rec_list)
            self._counters["last_optimization_time"] = round(dt, 4)
            self._counters["last_estimated_impact"] = round(impact, 4)
            self._last_recommendation = rec_list
        return response

    # -- config & params ---------------------------------------------------
    def _parse_config(self, config_field):
        if not config_field:
            return {}
        if isinstance(config_field, dict):
            return dict(config_field)
        if isinstance(config_field, str):
            s = config_field.strip()
            if not s:
                return {}
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {}
        return {}

    def _build_params(self, cfg, budget):
        try:
            max_index_width = int(cfg.get("max_index_width", DEFAULT_MAX_INDEX_WIDTH))
        except (TypeError, ValueError):
            max_index_width = DEFAULT_MAX_INDEX_WIDTH
        budget_mb = DEFAULT_BUDGET_MB
        max_indexes = DEFAULT_MAX_INDEXES
        try:
            min_cost_improvement = float(
                cfg.get("min_cost_improvement", DEFAULT_MIN_COST_IMPROVEMENT))
        except (TypeError, ValueError):
            min_cost_improvement = DEFAULT_MIN_COST_IMPROVEMENT

        try:
            storage_mb = float(budget["storage_mb"])
        except (KeyError, TypeError, ValueError):
            storage_mb = None
        try:
            max_indexes_b = int(budget["max_indexes"])
        except (KeyError, TypeError, ValueError):
            max_indexes_b = None
        try:
            write_pct = float(budget["max_write_overhead_pct"])
        except (KeyError, TypeError, ValueError):
            write_pct = None

        cap_count = None
        if storage_mb is not None:
            constraint = "storage"
            disk_mb = storage_mb
            cap_count = max_indexes_b if max_indexes_b is not None else max_indexes
        elif max_indexes_b is not None:
            constraint = "number"
            disk_mb = budget_mb
            max_indexes = max_indexes_b
            cap_count = None
        else:
            constraint = "storage"
            disk_mb = budget_mb  # uniform default (1000 MB) when no budget is given
            cap_count = max_indexes

        params = {
            "max_index_width": max_index_width,
            "budget_MB": disk_mb,
            "max_indexes": max_indexes,
            "constraint": constraint,
            "min_cost_improvement": min_cost_improvement,
        }
        return params, cap_count, write_pct

    # -- index -> DDL ------------------------------------------------------
    def _to_ddl(self, indexes, existing, table_sizes, col_types, dsn,
                cap_count, write_pct, notes):
        existing_cols, existing_names = existing
        used_names = set()
        rec = []
        for ind in indexes:
            table = ind.table().name
            cols = [c.name for c in ind.columns]
            key = (table, tuple(cols))
            if key in existing_cols:
                notes.append(f"index on {table}({','.join(cols)}) already exists; skipped")
                continue
            buildable, breason = self._index_buildable(dsn, table, cols, col_types)
            if not buildable:
                notes.append(breason)
                continue
            base = ind.index_idx()
            name = base
            n = 2
            while name.lower() in existing_names or name.lower() in used_names:
                name = f"{base}_{n}"
                n += 1
            used_names.add(name.lower())
            storage_mb = _b_to_mb(ind.estimated_size or 0)
            tsize = table_sizes.get(table, 0)
            ratio = (storage_mb / (_b_to_mb(tsize) or 1.0)) if tsize else 0.0
            write_overhead = min(100.0, 1.0 + ratio * 5.0)
            rec.append({
                "ddl": f"CREATE INDEX {name} ON {table} ({','.join(cols)})",
                "table": table,
                "columns": cols,
                "index_name": name,
                "storage_mb": round(storage_mb, 4),
                "write_overhead_estimate": round(write_overhead, 4),
            })
            if cap_count is not None and len(rec) >= cap_count:
                break
        if write_pct is not None:
            kept = []
            for r in rec:
                if (r["write_overhead_estimate"] or 0) <= float(write_pct):
                    kept.append(r)
                else:
                    notes.append(
                        f"{r['index_name']} write_overhead {r['write_overhead_estimate']}% "
                        f"exceeds budget {write_pct}%; dropped"
                    )
            rec = kept
        return rec

    def _estimate_impact(self, eval_conn, workload, indexes):
        try:
            ce = CostEvaluation(eval_conn)
            initial = ce.calculate_cost(workload, [])
            final = ce.calculate_cost(workload, list(indexes))
            ce.complete_cost_estimation()
            eval_conn.drop_hypo_indexes()
            if initial and initial > 0:
                return (initial - final) / initial * 100.0
            return 0.0
        except Exception:
            try:
                eval_conn.drop_hypo_indexes()
            except Exception:
                pass
            return 0.0

    def _apply_indexes(self, rec_list, dsn):
        results = []
        try:
            conn = self._get_connector(dsn)
        except Exception as e:
            return [{"index_name": r["index_name"], "status": "error",
                     "error": f"connect failed: {e}"} for r in rec_list]
        for r in rec_list:
            try:
                conn._cursor.execute("DROP INDEX IF EXISTS %s" % r["index_name"])
                conn._cursor.execute(r["ddl"])
                results.append({"index_name": r["index_name"], "status": "ok", "error": None})
            except Exception as e:
                conn.rollback()
                results.append({"index_name": r["index_name"], "status": "error",
                                 "error": str(e)})
        return results

    # -- degrade & state ---------------------------------------------------
    def _budget_used(self, budget, rec_list):
        consumed_storage = sum(r.get("storage_mb") or 0 for r in rec_list)
        consumed_write = max([r.get("write_overhead_estimate") or 0 for r in rec_list] + [0.0])
        try:
            storage_mb = float(budget["storage_mb"])
        except (KeyError, TypeError, ValueError):
            storage_mb = DEFAULT_BUDGET_MB
        try:
            max_indexes = int(budget["max_indexes"])
        except (KeyError, TypeError, ValueError):
            max_indexes = DEFAULT_MAX_INDEXES
        try:
            write_pct = float(budget["max_write_overhead_pct"])
        except (KeyError, TypeError, ValueError):
            write_pct = None
        return {
            "storage_mb": storage_mb,
            "max_indexes": max_indexes,
            "max_write_overhead_pct": write_pct,
            "consumed_storage_mb": round(consumed_storage, 4),
            "consumed_indexes": len(rec_list),
            "consumed_write_overhead_pct": round(consumed_write, 4),
        }

    def _degrade(self, reason, rec_list, cfg, num_analyzed, notes=None,
                 dsn=None, eval_dsn=None, budget=None):
        notes = notes or []
        notes.append(reason)
        return {
            "recommended_indexes": rec_list,
            "metadata": {
                "strategy_type": "cost-based",
                "optimization_time": 0.0,
                "estimated_impact": 0.0,
                "num_queries_analyzed": num_analyzed,
                "budget_used": self._budget_used(budget or {}, []),
                "eval_mode": "standby" if (eval_dsn and dsn and eval_dsn != dsn) else "hypothetical",
                "mode": "inference-only",
                "config": {k: cfg[k] for k in sorted(cfg)} if cfg else {},
                "degrade_note": "; ".join(notes),
            },
        }

    def state(self):
        with self._state_lock:
            counters = dict(self._counters)
            last_rec = list(self._last_recommendation)
            cfg = dict(self._last_config)
        return {
            "idle": True,  # no background work for a deterministic heuristic
            "model_loaded": True,
            "training_in_progress": False,
            "counters": counters,
            "last_recommendation": last_rec,
            "active_config": cfg,
            "state_dir": self.state_dir,
            "strategy_type": "cost-based",
        }

    def shutdown(self):
        self._shutdown = True
        with self._state_lock:
            self._persist_state()
        for dsn, conn in list(self._connectors.items()):
            try:
                conn.drop_hypo_indexes()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        self._connectors.clear()


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
_SERVER = {}


def _make_handler(server_obj):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"status": "ok", "strategy": "extend"})
                return
            if self.path.startswith("/state"):
                self._send(200, server_obj.state())
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.startswith("/recommend"):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    req = json.loads(raw.decode("utf-8") or "{}")
                except Exception as e:
                    self._send(200, server_obj._degrade(f"invalid JSON body: {e}", [], {}, 0))
                    return
                _t0 = time.time()
                try:
                    resp = server_obj.recommend(req)
                except Exception as e:
                    traceback.print_exc()
                    resp = server_obj._degrade(f"internal error: {e}", [], {}, 0)
                print(f"[req] {time.time() - _t0:.2f}s "
                      f"recs={len(resp.get('recommended_indexes') or [])} "
                      f"analyzed={(resp.get('metadata') or {}).get('num_queries_analyzed')}",
                      flush=True)
                self._send(200, resp)
                return
            if self.path.startswith("/shutdown"):
                self._send(200, {"status": "shutting down"})
                threading.Thread(target=server_obj.shutdown, daemon=True).start()
                threading.Thread(
                    target=lambda: (time.sleep(0.5), _SERVER["httpd"].shutdown()),
                    daemon=True).start()
                return
            self._send(404, {"error": "not found"})

    return Handler


def main():
    port = int(os.environ.get("PORT", "8765"))
    state_dir = os.environ.get("EXTEND_STATE_DIR", DEFAULT_STATE_DIR)
    server_obj = ExtendServer(state_dir=state_dir)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(server_obj))
    httpd.daemon_threads = True
    _SERVER["httpd"] = httpd
    print(f"extend wrapper listening on {port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        server_obj.shutdown()


if __name__ == "__main__":
    main()
