#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Agentic DAG + Graphy Repo-RAG + Workspace ACI for Code Editing (Ollama)
Reality-first, self-correcting, self-updating via ToolFoundry + Route Context.

NEW IN THIS UPGRADE (drop-in full replacement):
  • Repo Indexer (nested folders): builds a fast, symbolic/semantic index:
      - file tree (relative paths, size, sha1, mtime, language)
      - per-file outlines (py AST classes/functions/imports; js/ts regex; md headings; json keys)
      - cross-file refs (basic Python import map)
  • Dynamic Route Context: goal → ranked file candidates + actionable tools catalog
      - Packs informed filenames, relpaths, outlines, and tool instructions for manipulation
  • WorkspaceTools (ACI): safe span-bounded editing, regex replace, unified-diff apply,
      search, read spans, verify changes, optional git commit/branch/diff, pytest run
  • QA Pipeline: captures BEFORE/AFTER, requested edit, diffs; LLM QA rating → pass/fail
      - On PASS: optional commit/integration; On FAIL: auto reflect/replan retry once
  • Plan wiring: LLM planner is conditioned on Route Context (files + tools) to operate
      on the right spans with the right tools (no raw dumping entire files).
  • Full traceability: data/trace.jsonl stores route context, applied edits, QA ratings.

Run:
  ollama serve
  python3 app.py

Optional env:
  AGENT_DEBUG=1         # print compact prompt transcripts
  AGENT_INTERLEAVE=1    # add a short assistant-preface before STRICT-JSON asks
  ACI_MAX_FILES=8000    # cap files scanned during indexing (default 8000)
  ACI_EXCLUDES=...      # comma-separated globs to ignore (default below)
  ACI_GIT_AUTOCOMMIT=1  # if QA passes, auto-commit with a generated message
  ACI_RUN_PYTEST=1      # run pytest -q if found, as a verifier

  # NEW:
  AGENT_STREAM=0        # disable live token streaming (enabled by default)
"""

# ──────────────────────────────────────────────────────────────────────────────
# I. venv bootstrap (stdlib only)
# ──────────────────────────────────────────────────────────────────────────────
import os, sys, subprocess, json, time, textwrap, inspect, importlib.util, types, traceback, re, ast, hashlib, fnmatch, difflib, io
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
PY_EXE = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
PIP_EXE = [str(PY_EXE), "-m", "pip"]
DEPS = ["ollama>=0.5.0", "pydantic>=2.7", "httpx>=0.27"]

def _in_venv() -> bool:
    return str(PY_EXE) == sys.executable or bool(os.environ.get("VIRTUAL_ENV"))

def _bootstrap_and_reexec():
    if not VENV_DIR.exists():
        print("[setup] Creating venv ...")
        import venv
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    if Path(sys.executable) != PY_EXE:
        print(f"[setup] Upgrading pip & installing deps into {VENV_DIR} ...")
        subprocess.check_call([str(PY_EXE), "-m", "pip", "install", "--upgrade", "pip", "wheel"])
        subprocess.check_call(PIP_EXE + ["install"] + DEPS)
        env = os.environ.copy()
        env["VENV_ACTIVE"] = "1"
        os.execve(str(PY_EXE), [str(PY_EXE), __file__] + sys.argv[1:], env)

if not _in_venv():
    _bootstrap_and_reexec()

# ──────────────────────────────────────────────────────────────────────────────
# II. imports (post-venv)
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_type_hints, Iterable
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime, UTC

try:
    import httpx
    import ollama
    from ollama import Client
except Exception as e:
    print("[fatal] Failed to import ollama/httpx after venv activation.")
    print(e)
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# III. paths, IO helpers, Context Bus, Systems, Telemetry, Debug
# ──────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = BASE_DIR / "config.json"
DOCS_PATH = BASE_DIR / "docs.json"
TOOLS_PATH = BASE_DIR / "tools.py"
DATA_DIR = BASE_DIR / "data"
BUS_PATH = DATA_DIR / "context_bus.jsonl"
FOUNDRY_LOG_DIR = DATA_DIR / "foundry_runs"
SYSTEMS_PATH = BASE_DIR / "systems.json"
TESTS_INDEX_PATH = DATA_DIR / "tests_index.json"
TRACE_PATH = DATA_DIR / "trace.jsonl"
REPO_INDEX_PATH = DATA_DIR / "repo_index.json"

for d in (DATA_DIR, FOUNDRY_LOG_DIR):
    d.mkdir(exist_ok=True)

DEBUG = os.environ.get("AGENT_DEBUG","0") == "1"
STRICT_SYSTEM = os.environ.get("AGENT_STRICT_SYSTEM", "1") != "0"
INTERLEAVE = os.environ.get("AGENT_INTERLEAVE","0") == "1"
# Stream live tokens by default (set AGENT_STREAM=0 to disable)
STREAM = os.environ.get("AGENT_STREAM","1") != "0"

def jdump(p: Path, obj: Any):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def jload(p: Path, default: Any = None):
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def _now_iso():
    return datetime.now(UTC).replace(microsecond=0).isoformat()

def ctx_bus_push(kind: str, payload: Dict[str, Any]):
    rec = {"ts": _now_iso(), "kind": kind, "payload": payload}
    with BUS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def ctx_bus_tail(n: int = 20) -> List[Dict[str, Any]]:
    if not BUS_PATH.exists():
        return []
    try:
        lines = BUS_PATH.read_text().splitlines()
        return [json.loads(x) for x in lines[-n:]]
    except Exception:
        return []

def trace(kind: str, obj: Dict[str, Any]):
    rec = {"ts": _now_iso(), "kind": kind, "payload": obj}
    with TRACE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _dbg(label: str, messages: List[Dict[str,str]]):
    if not DEBUG: return
    print(f"\n--- DEBUG:{label} ---")
    for i,m in enumerate(messages,1):
        head = (m.get("content","")[:400] + "…") if len(m.get("content",""))>400 else m.get("content","")
        print(f"{i:02d} {m.get('role')} :: {head}\n")
        
global TOOLS_MOD, TOOLS, DOCS, REGISTRY, TOOL_ALIASES

# ──────────────────────────────────────────────────────────────────────────────
# IV. Unified System Messages (auto-seeded systems.json)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SYSTEMS = {
  "reality_policy": (
    "Reality Policy:\n"
    "- Do NOT simulate or return placeholder values.\n"
    "- Always prefer plans that acquire or compute real results.\n"
    "- If a capability is missing, NAME a new tool with a descriptive, generic name; another agent will implement it.\n"
    "- Plans that return null/None or dummy JSON without obtaining real data will be rejected."
  ),

  "clarify": (
    "You are a precise task clarifier.\n"
    "Goal: produce the smallest useful reformulation and concrete constraints.\n"
    "Rules (highest priority):\n"
    "- Consider ONLY the provided 'goal' string. Ignore conversation history and prior turns.\n"
    "- Output 1–2 short lines total. No markdown sections, no commentary.\n"
    "- NEVER invent or propose new tools. Use only existing workspace ACI (WorkspaceTools.*).\n"
    "- If the goal requests a repo file read or edit, say that directly (e.g., 'Read config.json' or 'Edit config.json → set model=qwen3:4b').\n"
    "- Keep constraints concrete and at most 3 bullets if needed.\n\n"
    "{{REALITY_POLICY}}"
  ),

  "plan": (
    "You are a planning agent...\n"
    "OUTPUT STRICT JSON (double quotes only; no prose):\n"
    "{ \"nodes\": [ { \"id\": \"str\", \"tool\": \"str\", \"args\": object, \"after\": [\"str\"], \"retries\": int, \"timeout_s\": int, \"alias\": \"str?\" } ],\n"
    "  \"meta\": { \"parallelism\": int, \"timeout_s\": int, \"notes\": \"str?\" } }\n"
    "Hard rules:\n"
    "- Language: English only.\n"
    "- Tools: choose ONLY from this list: {{AVAILABLE_TOOLS}}\n"
    "- Use these signatures (arg names matter):\n"
    "{{TOOL_SIGNATURES}}\n"
    "- For file ops use 'rel' (path), never file contents.\n"
    "- Prefer read_span/write_span over whole-file writes.\n"
    "- If args reference previous outputs, wire dependencies with [nodeId.output.key] and ensure 'after' includes nodeId.\n"
    "- For regexes: avoid character classes like [0-9]; prefer \\d etc. If you must use brackets, escape them as \\[ and \\].\n"
    "- Include post-check nodes (verify_file_contains, optional TestTools.run_pytest when available).\n"
    "{{REALITY_POLICY_SECTION}}\n"
    ),



  "red_team": (
    "You are an adversarial planner ('red team').\n"
    "Given a goal, route_context, and an initial plan, identify missing prerequisites, failure modes, validation steps, safety checks,\n"
    "and any observability needed. Propose a PATCH as STRICT JSON only:\n"
    "{ 'add': [ nodes... ], 'modify': [ {'id': str, 'tool'?: str, 'args'?: object, 'after'?: [str], 'retries'?: int, 'timeout_s'?: int} ],\n"
    "  'remove': [str], 'notes': [str] }\n"
    "- Keep it minimal but sufficient; no prose outside JSON."
  ),

  "plan_merge": (
    "You merge an original plan with a patch. Output ONLY the merged plan JSON with the same schema as 'plan'."
  ),

  "reflect": (
    "You repair tool plans. Given failures and the original plan, output ONLY corrected JSON plan. "
    "Prefer swapping tools/args; keep minimal."
  ),

  "assemble": (
    "You are a careful assembler. Use tool results to answer succinctly and call out assumptions."
  ),

  "foundry_spec": (
    "Define a minimal, generic tool spec for a single Python function.\n"
    "Output STRICT JSON: {doc, args:[{name,type,desc}], returns}\n"
    "Constraints:\n"
    "- Design for real operation (no simulations). If network access is implied, it must be performable in production.\n"
    "- Keep names generic; the function can accept an optional 'fetcher' for DI if it performs HTTP.\n"
  ),

  "foundry_impl": (
    "Implement ONE Python function with type hints and a docstring.\n"
    "- No side effects. For NET_TOOL: include optional parameter `fetcher=None`. If None, set it to a helper using httpx.get.\n"
    "- Allowed imports by kind: PURE_FUNC={math,re,json,statistics}; NET_TOOL={httpx,json,re}; IO_TOOL={json,re}.\n"
    "- Do not print, read env, or touch filesystem. Return Python objects (dict/list), not JSON strings.\n"
    "- Name MUST equal 'name'. Return ONLY the function code (no fences, no extra text)."
  ),

  "foundry_test": (
    "Write deterministic unit tests for the function.\n"
    "Output STRICT JSON array, 3-5 cases:\n"
    "[{\"inputs\": {\"arg\": value,...}, \"expect\": any, \"rtol\": optional_float, \"explain\": \"\"}, ...]\n"
    "For NET_TOOL: pass a stub `fetcher(url)` returning fixed `.json()` payloads; do NOT perform network calls."
  ),

  "foundry_revise": (
    "Revise the implementation based on test failures. Return ONLY the updated function code.\n"
    "If failures relate to signature or edge cases, correct them. Keep within the allowed imports by kind."
  ),

  "qa_reviewer": (
    "You are a repository QA reviewer. Given a requested edit, BEFORE/AFTER snippets, a diff, and verification results,\n"
    "output STRICT JSON with keys: { 'pass': bool, 'score': int (0-100), 'reasons': [str], 'risk': 'low'|'medium'|'high', 'followups': [str] }.\n"
    "Guidelines: prefer PASS if change matches requested edit, compiles/tests (if run) pass, and verifications succeeded."
  )
}

def _seed_or_load_systems() -> Dict[str, str]:
    existing = jload(SYSTEMS_PATH, default=None)
    if existing is None:
        jdump(SYSTEMS_PATH, DEFAULT_SYSTEMS)
        return DEFAULT_SYSTEMS
    merged = dict(DEFAULT_SYSTEMS)
    merged.update(existing)
    if existing != merged:
        jdump(SYSTEMS_PATH, merged)
    return merged

SYSTEMS = _seed_or_load_systems()

def sysmsg(key: str, replacements: Optional[Dict[str, str]] = None) -> str:
    s = SYSTEMS.get(key, "")
    if replacements:
        for k, v in replacements.items():
            s = s.replace("{{"+k+"}}", v)
    return s
# ──────────────────────────────────────────────────────────────────────────────
# V. default tools.py (created/extended if missing — stdlib; ACI & Git helpers)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_TOOLS = '''\
"""
tools.py — Built-in tools for the agentic DAG (ACI: Agent-Computer Interface)
Deterministic, safe helpers. ToolFoundry appends new ones below.

New classes:
- WorkspaceTools: repo indexing, file listing, search, outlines, span edits, regex replace, unified-diff apply, verify
- GitTools: optional git integration (status/diff/commit/branch)
- TestTools: run pytest -q if present

Docstring conventions (for LLM/agent consumption)
- Purpose: One-line summary of what the tool does in production.
- When to use: Simple decision rule so the planner can select correctly.
- Args: Name (type) — meaning, units/range, constraints.
- Returns: JSON-compatible schema and key meanings.
- Failure modes: What can go wrong and how it is surfaced (return vs raise).
- Example: Minimal, copy-pastable call.
"""

from typing import List, Dict, Any, Optional, Iterable, Tuple
from pathlib import Path
import json, re, statistics, hashlib, fnmatch, difflib, io, subprocess, os, time, ast

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
REPO_INDEX_PATH = DATA_DIR / "repo_index.json"

# ---------- small utilities ----------
def _sha1_bytes(b: bytes) -> str:
    """Internal: SHA-1 of bytes, hex string."""
    import hashlib
    h = hashlib.sha1(); h.update(b); return h.hexdigest()

def _sha1_file(p: Path) -> str:
    """Internal: SHA-1 of file contents, hex string ('' on failure)."""
    try:
        return _sha1_bytes(p.read_bytes())
    except Exception:
        return ""

def _detect_lang(p: Path) -> str:
    """Internal: light language detection from file extension."""
    ext = p.suffix.lower()
    return {
        ".py":"python", ".js":"javascript", ".ts":"typescript", ".tsx":"tsx", ".jsx":"jsx",
        ".md":"markdown", ".json":"json", ".yml":"yaml", ".yaml":"yaml", ".toml":"toml",
        ".html":"html", ".css":"css", ".cpp":"cpp", ".cc":"cpp", ".cxx":"cpp", ".c":"c",
        ".java":"java", ".rs":"rust"
    }.get(ext, ext.lstrip("."))

def _line_bounds(text: str, start: int, end: int) -> Tuple[int, int]:
    """Internal: clamp 1-based [start,end] to text line count and return (s,e)."""
    lines = text.splitlines(keepends=True)
    n = len(lines)
    s = max(1, min(start, n))
    e = max(s, min(end, n))
    return s, e

def _python_outline(src: str) -> Dict[str, Any]:
    """Internal: outline Python source → {'imports': [str], 'symbols':[{'kind','name','lineno'}]}."""
    out = {"imports": [], "symbols": []}
    try:
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    out["imports"].append(a.name.split(".",1)[0])
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    out["imports"].append(n.module.split(".",1)[0])
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out["symbols"].append({"kind":"def","name":n.name,"lineno":n.lineno})
            elif isinstance(n, ast.ClassDef):
                out["symbols"].append({"kind":"class","name":n.name,"lineno":n.lineno})
    except Exception:
        pass
    out["imports"] = sorted(list(dict.fromkeys(out["imports"])))
    return out

def _js_outline(src: str) -> Dict[str, Any]:
    """Internal: outline JS/TS source → {'imports': [str], 'symbols':[{'kind','name','lineno'}]}."""
    out = {"imports": [], "symbols": []}
    try:
        for line_no, line in enumerate(src.splitlines(), 1):
            if line.strip().startswith("import "):
                m = re.search(r"from\\s+['\\\"]([^'\\\"]+)['\\\"]", line)
                if m:
                    base = m.group(1).split("/",1)[0].strip("./")
                    out["imports"].append(base)
            m = re.search(r"\\b(class|function)\\s+([A-Za-z_][\\w]*)", line)
            if m:
                out["symbols"].append({"kind":m.group(1),"name":m.group(2),"lineno":line_no})
            m2 = re.search(r"export\\s+(?:const|function|class)\\s+([A-Za-z_][\\w]*)", line)
            if m2:
                out["symbols"].append({"kind":"export","name":m2.group(1),"lineno":line_no})
    except Exception:
        pass
    out["imports"] = sorted(list(dict.fromkeys(out["imports"])))
    return out

def _md_outline(src: str) -> Dict[str, Any]:
    """Internal: outline Markdown headings → symbols [{kind:'hN', name, lineno}]."""
    syms = []
    for i, line in enumerate(src.splitlines(),1):
        if line.startswith("#"):
            h = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            syms.append({"kind":f"h{h}","name":title,"lineno":i})
    return {"imports": [], "symbols": syms}

def _json_outline(src: str) -> Dict[str, Any]:
    """Internal: outline top-level JSON keys (first 50)."""
    try:
        obj = json.loads(src)
        if isinstance(obj, dict):
            keys = list(obj.keys())[:50]
            return {"imports": [], "symbols": [{"kind":"key","name":k,"lineno":1} for k in keys]}
    except Exception:
        pass
    return {"imports": [], "symbols": []}

def _outline_for(lang: str, src: str) -> Dict[str, Any]:
    """Internal: route to language-specific outline helper."""
    if lang == "python": return _python_outline(src)
    if lang in ("javascript","typescript","tsx","jsx"): return _js_outline(src)
    if lang == "markdown": return _md_outline(src)
    if lang == "json": return _json_outline(src)
    return {"imports": [], "symbols": []}

def _default_excludes() -> List[str]:
    """Internal: default path globs to exclude during indexing/search."""
    env = os.environ.get("ACI_EXCLUDES")
    if env:
        return [x.strip() for x in env.split(",") if x.strip()]
    return [
        ".git/*", "venv/*", "node_modules/*", "dist/*", "build/*",
        "__pycache__/*", "*.min.*", "*.lock", "*.log", "*.bin", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.ico"
    ]

def _should_skip(rel: str, excludes: List[str]) -> bool:
    """Internal: fnmatch-based include/exclude check for a relative path."""
    rel = rel.replace("\\\\","/")
    for pat in excludes:
        if fnmatch.fnmatch(rel, pat):
            return True
    return False

def _unix_rel(p: Path, base: Path) -> str:
    """Internal: normalize relpath to forward slashes."""
    return str(p.relative_to(base)).replace("\\\\","/")

# ---------- basic utilities ----------
def add(a: float, b: float) -> float:
    """
    Purpose: Add two numbers.
    When to use: Basic arithmetic in plans or tests.
    Args:
      - a (float): left operand.
      - b (float): right operand.
    Returns: float — a + b
    Failure modes: None (casts to float).
    Example: add(2, 3.5) -> 5.5
    """
    return float(a) + float(b)

def mean(values: List[float]) -> float:
    """
    Purpose: Compute arithmetic mean of a list of numbers.
    When to use: Quick statistics over numeric arrays.
    Args:
      - values (List[float]): non-empty list of finite numbers.
    Returns: float — arithmetic mean.
    Failure modes: ValueError if list is empty.
    Example: mean([1,2,3]) -> 2.0
    """
    if not values:
        raise ValueError("values must be non-empty")
    return float(statistics.fmean(values))

def slugify(text: str) -> str:
    """
    Purpose: Make a filesystem-safe slug from arbitrary text.
    When to use: Generate filenames/IDs from titles.
    Args:
      - text (str): any text.
    Returns: str — lowercase, dash-separated, alnum-only; 'n-a' if empty.
    Example: slugify("Hello World!") -> "hello-world"
    """
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return s or "n-a"

def word_count(text: str) -> Dict[str, int]:
    """
    Purpose: Rough token counts for text.
    When to use: Telemetry, quick summaries.
    Args:
      - text (str): input text.
    Returns: dict — {"words": int, "lines": int, "chars": int}
    Example: word_count("a b\\nc") -> {"words":3,"lines":2,"chars":4}
    """
    lines = text.splitlines()
    words = [w for w in re.split(r"\\s+", text.strip()) if w]
    return {"words": len(words), "lines": len(lines), "chars": len(text)}

def save_json(name: str, obj: dict) -> str:
    """
    Purpose: Persist a JSON-serializable object into ./data/<name>.json.
    When to use: Cache intermediate tool outputs for later inspection.
    Args:
      - name (str): base filename without extension (use slugify).
      - obj (dict): JSON-serializable object.
    Returns: str — absolute path written.
    Failure modes: Raises on IO errors.
    Example: save_json("run1", {"ok":true})
    """
    p = (DATA_DIR / f"{name}.json")
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return str(p)

def wrap_timestamp(ts: str) -> dict:
    """
    Purpose: Normalize timestamp into a standard envelope.
    When to use: For consistent timestamp structures across tools.
    Args:
      - ts (str): ISO-8601 timestamp.
    Returns: dict — {"timestamp": "<ISO-8601>"}
    """
    return {"timestamp": ts}

class UtilityTools:
    """Miscellaneous helpers."""

    @staticmethod
    def now(tz: str = "UTC") -> dict:
        """
        Purpose: Return the current timestamp (UTC) wrapped in an object.
        When to use: Stamp events, logs, or results.
        Args:
          - tz (str, optional): kept for API parity; only "UTC" supported.
        Returns: dict — {"timestamp": "<ISO-8601-UTC>"}
        Example: UtilityTools.now() -> {"timestamp": "2025-01-01T12:00:00+00:00"}
        """
        from datetime import datetime, UTC
        iso = datetime.now(UTC).replace(microsecond=0).isoformat()
        return {"timestamp": iso}

    @staticmethod
    def now_iso(tz: str = "UTC") -> str:
        """
        Purpose: Return the current timestamp (UTC) as ISO-8601 string.
        When to use: You need just the string, not an object.
        Args:
          - tz (str, optional): kept for API parity; only "UTC" supported.
        Returns: str — "<ISO-8601-UTC>"
        """
        from datetime import datetime, UTC
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    @staticmethod
    def load_json(name: str) -> dict:
        """
        Purpose: Load a JSON file from ./data/<name>.json.
        When to use: Retrieve cached tool outputs.
        Args:
          - name (str): base filename without extension.
        Returns: dict — parsed JSON or {} if missing/invalid.
        """
        p = (DATA_DIR / f"{name}.json")
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

class WorkspaceTools:
    """
    Purpose: Agent-Computer Interface (ACI) for working inside a project workspace.
    When to use: Determining repo contents, finding candidate files, and making safe,
                 span-bounded edits with post-change verification.

    Provided capabilities:
      - Repo indexing (symbolic outlines) → data/repo_index.json
      - File listing and regex search over the indexed set
      - Span read/write with optional hash guard
      - Regex-based replace
      - Apply unified diffs via `patch` if present
      - Lightweight verification helpers
      - Ranking candidate files from keywords (Route Context)
    """

    @staticmethod
    def _base(root: Optional[str] = None) -> Path:
        """Internal: workspace base path (resolved)."""
        return Path(root or ".").resolve()

    @staticmethod
    def _excludes() -> List[str]:
        """Internal: compute effective exclude globs (env ACI_EXCLUDES overrides)."""
        env = os.environ.get("ACI_EXCLUDES")
        if env:
            return [x.strip() for x in env.split(",") if x.strip()]
        return [
            ".git/*", "venv/*", "env/*", ".venv/*", "node_modules/*", "dist/*", "build/*",
            "__pycache__/*", "*.min.*", "*.lock", "*.log", "*.bin",
            "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.ico",
            "*.pdf", "*.zip", "*.tar", "*.gz", "*.7z", "*.mp4", "*.mov", "*.mp3", "*.wav",
        ]

    @staticmethod
    def list_tools(kind: str = "all") -> Dict[str, Any]:
        """
        Purpose: Enumerate available tools in tools.py as a compact catalog for planners.
        When to use: Provide the LLM a concise, accurate list of tools + signatures.
        Args:
          - kind (str): 'all' (default) or one of {'workspace','git','test','functions'}.
        Returns: dict — {"tools":[{"name":str,"sig":str,"doc":str}], "count": int}
        """
        import inspect as _inspect
        mod = globals()
        out: List[Dict[str, Any]] = []

        def _sig(f) -> str:
            try:
                return str(_inspect.signature(f))
            except Exception:
                return "(...)"

        def _doc1(f) -> str:
            d = (_inspect.getdoc(f) or "").strip()
            return d.split("\n", 1)[0] if d else ""

        # top-level functions in tools.py
        if kind in ("all", "functions"):
            for name, obj in list(mod.items()):
                if name.startswith("_"):
                    continue
                if callable(obj) and getattr(obj, "__module__", None) == __name__ and not isinstance(obj, type):
                    out.append({"name": name, "sig": f"{name}{_sig(obj)}", "doc": _doc1(obj)})

        def _collect_class(cls, clsname: str):
            for mname, m in _inspect.getmembers(cls, predicate=_inspect.isfunction):
                if mname.startswith("_"):
                    continue
                out.append({"name": f"{clsname}.{mname}", "sig": f"{mname}{_sig(m)}", "doc": _doc1(m)})

        if kind in ("all", "workspace"):
            _collect_class(WorkspaceTools, "WorkspaceTools")
        if "GitTools" in mod and kind in ("all", "git"):
            _collect_class(mod["GitTools"], "GitTools")
        if "TestTools" in mod and kind in ("all", "test"):
            _collect_class(mod["TestTools"], "TestTools")

        out.sort(key=lambda d: d["name"])
        return {"tools": out, "count": len(out)}

    @staticmethod
    def _skip(rel: str, excludes: List[str]) -> bool:
        """Internal: fnmatch-based exclusion check."""
        rel = rel.replace("\\\\", "/")
        for pat in excludes:
            if fnmatch.fnmatch(rel, pat):
                return True
        return False

    @staticmethod
    def _detect_lang(path: Path) -> str:
        """Internal: delegate to extension-based language detector."""
        return _detect_lang(path)

    @staticmethod
    def _outline(lang: str, text: str) -> Dict[str, Any]:
        """Internal: delegate to per-language outline builder."""
        return _outline_for(lang, text)

    @staticmethod
    def index_repo(root: str = ".", max_files: Optional[int] = None, excludes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Purpose: Walk the repository and build a fast, lightweight index.
        When to use: Before planning workspace edits or searches; keeps context grounded.
        Args:
          - root (str): workspace root (default ".").
          - max_files (int|None): cap files scanned; default ACI_MAX_FILES env or 8000.
          - excludes (List[str]|None): override exclude globs; otherwise internal defaults.
        Returns: dict — {
          "root": str, "scanned": int, "excluded_globs": [str], "generated_at": int,
          "files": [
            {"rel": str, "size": int, "mtime": int, "sha1": str,
             "lang": str, "imports": [str], "symbols":[{"kind","name","lineno"}]}
          ],
          "py_import_map": {import_base: [relpath, ...]}
        }
        Failure modes: Skips unreadable files; index still returns with fewer items.
        Example: WorkspaceTools.index_repo(".", max_files=2000)
        """
        base = WorkspaceTools._base(root)
        cap = max_files or int(os.environ.get("ACI_MAX_FILES", "8000"))
        ex = excludes or WorkspaceTools._excludes()

        files: List[Dict[str, Any]] = []
        scanned = 0

        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(base)).replace("\\\\", "/")
            if WorkspaceTools._skip(rel, ex):
                continue
            try:
                b = p.read_bytes()
                sha1 = _sha1_bytes(b)
                lang = WorkspaceTools._detect_lang(p)
                text = ""
                outline = {"imports": [], "symbols": []}
                # Read small/known-text files for outlines (size guard ~1.5MB)
                if p.suffix.lower() in (".py",".js",".ts",".tsx",".jsx",".md",".json",".yml",".yaml",".toml",".html",".css") and p.stat().st_size < 1_500_000:
                    try:
                        text = b.decode("utf-8", "replace")
                        outline = WorkspaceTools._outline(lang, text)
                    except Exception:
                        text = ""
                files.append({
                    "rel": rel,
                    "size": p.stat().st_size,
                    "mtime": int(p.stat().st_mtime),
                    "sha1": sha1,
                    "lang": lang,
                    "imports": outline.get("imports", []),
                    "symbols": outline.get("symbols", []),
                })
                scanned += 1
                if scanned >= cap:
                    break
            except Exception:
                # Skip unreadable files
                continue

        # Build light-weight cross-file refs (python imports → files)
        py_map: Dict[str, List[str]] = {}
        for f in files:
            if f["lang"] == "python":
                for imp in f.get("imports", []):
                    py_map.setdefault(imp, []).append(f["rel"])

        index = {
            "root": str(base),
            "scanned": scanned,
            "excluded_globs": ex,
            "generated_at": int(time.time()),
            "files": files,
            "py_import_map": {k: sorted(set(v)) for k, v in py_map.items()},
        }
        REPO_INDEX_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        return index

    @staticmethod
    def get_index() -> Dict[str, Any]:
        """
        Purpose: Retrieve the current repo index from disk.
        When to use: Before routing or file operations; avoids re-scanning if not needed.
        Args: None
        Returns: dict — index as produced by index_repo(); {} skeleton if absent.
        Failure modes: Returns a safe empty skeleton on parse errors.
        """
        if REPO_INDEX_PATH.exists():
            try:
                return json.loads(REPO_INDEX_PATH.read_text())
            except Exception:
                pass
        return {"root": str(Path(".").resolve()), "scanned": 0, "files": []}

    @staticmethod
    def list_files(pattern: str = "*", exts: Optional[List[str]] = None, limit: int = 500) -> List[str]:
        """
        Purpose: List files from the index by glob and extension filters.
        When to use: Narrow candidate files prior to editing/searching.
        Args:
          - pattern (str): fnmatch-style path pattern, e.g. "src/*.py".
          - exts (List[str]|None): filter by suffix (e.g., [".py",".md"]).
          - limit (int): max paths to return (default 500).
        Returns: List[str] — relative paths.
        Example: WorkspaceTools.list_files("**/*.py", [".py"], 200)
        """
        idx = WorkspaceTools.get_index()
        out: List[str] = []
        for f in idx.get("files", []):
            rel = f.get("rel", "")
            if not fnmatch.fnmatch(rel, pattern):
                continue
            if exts:
                if Path(rel).suffix.lower() not in [e.lower() for e in exts]:
                    continue
            out.append(rel)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def search_regex(pattern: str, flags: str = "", max_matches: int = 200) -> List[Dict[str, Any]]:
        """
        Purpose: Search the repository (text files) using a Python regex.
        When to use: Locate occurrences before editing; build review context.
        Args:
          - pattern (str): regex (Python re syntax).
          - flags (str): combination of 'i' (IGNORECASE), 'm' (MULTILINE), 's' (DOTALL).
          - max_matches (int): cap on results for performance.
        Returns: List[{"path": str, "line_no": int, "line": str}]
        Failure modes: Skips unreadable/large files (>1.5MB).
        Example: WorkspaceTools.search_regex(r"TODO\\b", "m", 100)
        """
        idx = WorkspaceTools.get_index()
        fl = 0
        if "i" in flags: fl |= re.IGNORECASE
        if "m" in flags: fl |= re.MULTILINE
        if "s" in flags: fl |= re.DOTALL
        rx = re.compile(pattern, fl)

        matches: List[Dict[str, Any]] = []
        base = Path(idx.get("root", "."))
        for f in idx.get("files", []):
            rel = f.get("rel", "")
            p = base / rel
            try:
                if p.stat().st_size > 1_500_000:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        matches.append({"path": rel, "line_no": i, "line": line})
                        if len(matches) >= max_matches:
                            return matches
            except Exception:
                continue
        return matches

    @staticmethod
    def read_file(rel: str) -> Dict[str, Any]:
        """
        Purpose: Read an entire text file from the workspace.
        When to use: You need the whole content and a hash for guards.
        Args:
          - rel (str): relative path within repo root.
        Returns: dict — {"path": str, "text": str, "sha1": str, "lines": int}
        Failure modes: Raises on IO errors (file missing/permission).
        Example: WorkspaceTools.read_file("app.py")
        """
        base = WorkspaceTools._base(WorkspaceTools.get_index().get("root", "."))
        p = base / rel
        text = p.read_text(encoding="utf-8", errors="replace")
        sha1 = _sha1_bytes(text.encode("utf-8"))
        return {"path": rel, "text": text, "sha1": sha1, "lines": len(text.splitlines())}

    @staticmethod
    def read_span(rel: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """
        Purpose: Read a bounded line range (1-based, inclusive) from a file.
        When to use: Prepare precise edits or reviews without loading the whole file.
        Args:
          - rel (str): relative path.
          - start_line (int): 1-based inclusive start.
          - end_line (int): 1-based inclusive end (clamped to file length).
        Returns: dict — {"path": str, "start": int, "end": int, "text": str, "sha1_file": str}
        Failure modes: Raises on IO errors; clamps out-of-range to valid bounds.
        Example: WorkspaceTools.read_span("src/x.py", 10, 40)
        """
        info = WorkspaceTools.read_file(rel)
        text = info["text"]
        lines = text.splitlines(keepends=True)
        n = len(lines)
        s = max(1, min(start_line, n))
        e = max(s, min(end_line, n))
        span = "".join(lines[s-1:e])
        return {"path": rel, "start": s, "end": e, "text": span, "sha1_file": info["sha1"]}

    @staticmethod
    def write_span(rel: str, start_line: int, end_line: int, new_text: str, ensure_trailing_nl: bool = True, guard_sha1: Optional[str] = None) -> Dict[str, Any]:
        """
        Purpose: Safely replace a bounded line range with new text.
        When to use: Precise, minimal edits with optional whole-file hash guard.
        Args:
          - rel (str): relative path.
          - start_line (int): 1-based inclusive start.
          - end_line (int): 1-based inclusive end.
          - new_text (str): replacement content for that span.
          - ensure_trailing_nl (bool): if True, append '\\n' if new_text lacks newline.
          - guard_sha1 (str|None): if provided, must match current file SHA-1 or no write occurs.
        Returns: dict — {
          "path": str, "start": int, "end": int,
          "before_sha1": str, "after_sha1": str, "changed": bool,
          "reason"?: "guard_sha1_mismatch", "current_sha1"?: str
        }
        Failure modes: Guard mismatch prevents write (returns changed=False and reason).
        Example: WorkspaceTools.write_span("x.py", 12, 20, "print('hi')\\n", True, guard_sha1="...sha1...")
        """
        base = WorkspaceTools._base(WorkspaceTools.get_index().get("root", "."))
        p = base / rel
        orig = p.read_text(encoding="utf-8", errors="replace")
        if guard_sha1:
            cur = _sha1_bytes(orig.encode("utf-8"))
            if cur != guard_sha1:
                return {"path": rel, "changed": False, "reason": "guard_sha1_mismatch", "current_sha1": cur}

        lines = orig.splitlines(keepends=True)
        n = len(lines)
        s = max(1, min(start_line, n if n else 1))
        e = max(s, min(end_line, n))
        # Ensure trailing newline robustly (avoid quote/escape pitfalls)
        if ensure_trailing_nl and new_text and not new_text.endswith((chr(10), chr(13), chr(13)+chr(10))):
            new_text = new_text + chr(10)
        before_sha1 = _sha1_bytes(orig.encode("utf-8"))
        new_lines = lines[:s-1] + [new_text] + lines[e:]
        new_all = "".join(new_lines)
        changed = (new_all != orig)
        if changed:
            p.write_text(new_all, encoding="utf-8")
        after_sha1 = _sha1_bytes(new_all.encode("utf-8"))
        return {"path": rel, "start": s, "end": e, "before_sha1": before_sha1, "after_sha1": after_sha1, "changed": changed}

    @staticmethod
    def regex_replace(rel: str, pattern: str, repl: str, flags: str = "", count: int = 0) -> Dict[str, Any]:
        """
        Purpose: Perform regex substitution in a file.
        When to use: Pattern-based edits across a file (e.g., renames, comment fixes).
        Args:
          - rel (str): relative path.
          - pattern (str): Python regex pattern.
          - repl (str): replacement string.
          - flags (str): 'i' ignorecase, 'm' multiline, 's' dotall.
          - count (int): max replacements (0 = replace all).
        Returns: dict — {"path": str, "matches": int, "changed": bool, "before_sha1": str, "after_sha1": str}
        Failure modes: Raises on IO errors; unchanged if no matches found.
        Example: WorkspaceTools.regex_replace("a.py", r"\\bTODO\\b", "NOTE", "m")
        """
        base = WorkspaceTools._base(WorkspaceTools.get_index().get("root", "."))
        p = base / rel
        text = p.read_text(encoding="utf-8", errors="replace")
        fl = 0
        if "i" in flags: fl |= re.IGNORECASE
        if "m" in flags: fl |= re.MULTILINE
        if "s" in flags: fl |= re.DOTALL
        rx = re.compile(pattern, fl)
        new_text, n = rx.subn(repl, text, count=count or 0)
        changed = n > 0 and new_text != text
        if changed:
            p.write_text(new_text, encoding="utf-8")
        return {
            "path": rel,
            "matches": n,
            "changed": changed,
            "before_sha1": _sha1_bytes(text.encode("utf-8")),
            "after_sha1": _sha1_bytes(new_text.encode("utf-8")),
        }

    @staticmethod
    def apply_unified_diff(diff_text: str, strip: int = 0, dry_run: bool = False, root: str = ".") -> Dict[str, Any]:
        """
        Purpose: Apply a unified diff using the system `patch` tool if available.
        When to use: Multi-file edits generated as a patch (git diff -U3).
        Args:
          - diff_text (str): unified diff content.
          - strip (int): -pN path strip level (patch's -p).
          - dry_run (bool): if True, do not write; just validate.
          - root (str): workspace root to apply within.
        Returns: dict — {"ok": bool, "stdout": str, "stderr": str, "used": "patch"|"fallback"}
        Failure modes: On missing `patch` or errors, returns ok=False and stderr with reason.
        Example: WorkspaceTools.apply_unified_diff(diff, strip=1, dry_run=True)
        """
        try:
            base = WorkspaceTools._base(root)
            proc = subprocess.run(
                ["patch", f"-p{int(strip)}", "--batch"] + (["--dry-run"] if dry_run else []),
                input=diff_text.encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(base)
            )
            ok = (proc.returncode == 0)
            return {"ok": ok, "stdout": proc.stdout.decode("utf-8","ignore"), "stderr": proc.stderr.decode("utf-8","ignore"), "used": "patch"}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": f"patch-unavailable: {e}", "used": "fallback"}

    @staticmethod
    def verify_file_contains(rel: str, must_include: List[str]) -> Dict[str, Any]:
        """
        Purpose: Verify that a file contains all required substrings.
        When to use: Post-edit validation to ensure intended insertions exist.
        Args:
          - rel (str): relative path.
          - must_include (List[str]): substrings that must appear.
        Returns: dict — {"path": str, "ok": bool, "missing": [str]}
        Failure modes: Raises on IO errors; ok=False with missing details if not found.
        Example: WorkspaceTools.verify_file_contains("x.py", ["def foo", "return 1"])
        """
        info = WorkspaceTools.read_file(rel)
        missing = [s for s in must_include if s not in info["text"]]
        return {"path": rel, "ok": not missing, "missing": missing}

    @staticmethod
    def route_candidates(keywords: List[str], k: int = 20) -> Dict[str, Any]:
        """
        Purpose: Rank likely-relevant files for a goal using simple heuristics.
        When to use: Build a Route Context: which files to read/edit first.
        Args:
          - keywords (List[str]): keywords from a clarified goal (lowercased ok).
          - k (int): number of candidates to return.
        Returns: dict — {
          "keywords": [str],
          "candidates": [
             {"rel": str, "score": float, "lang": str, "size": int,
              "symbols": [{"kind","name","lineno"}], "imports": [str]}
          ]
        }
        Heuristics: filename hits > path hits > symbol/import hits (descending score).
        Example: WorkspaceTools.route_candidates(["config","app"], 20)
        """
        idx = WorkspaceTools.get_index()
        kw = [x.lower() for x in (keywords or []) if isinstance(x, str) and x.strip()]
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for f in idx.get("files", []):
            rel = f.get("rel","")
            base = Path(rel).name.lower()
            score = 0.0
            for w in kw:
                if w in base:
                    score += 3.0
                if w in rel.lower():
                    score += 1.0
            # outline bonus
            for sym in f.get("symbols", []):
                nm = str(sym.get("name","")).lower()
                if any(w in nm for w in kw):
                    score += 1.5
            for imp in f.get("imports", []):
                il = str(imp).lower()
                if any(w in il for w in kw):
                    score += 0.8
            if score > 0:
                scored.append((score, f))
        scored.sort(key=lambda x: (-x[0], x[1].get("rel","")))
        top = []
        for s, f in scored[:k]:
            top.append({
                "rel": f["rel"],
                "score": round(s, 3),
                "lang": f.get("lang",""),
                "size": f.get("size", 0),
                "symbols": f.get("symbols", [])[:10],
                "imports": f.get("imports", [])[:10],
            })
        return {"keywords": kw, "candidates": top}

class GitTools:
    """
    Purpose: Minimal Git helpers for review and integration.
    When to use: Only when working in a Git repo; otherwise skip gracefully.
    """

    @staticmethod
    def _run(args: List[str], root: str = ".") -> Tuple[int, str, str]:
        """Internal: run a git subcommand; returns (returncode, stdout, stderr)."""
        proc = subprocess.run(["git"] + args, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode, proc.stdout, proc.stderr

    @staticmethod
    def is_repo(root: str = ".") -> bool:
        """
        Purpose: Detect if the current workspace is a Git repository.
        When to use: Guard Git operations to avoid noisy failures.
        Args:
          - root (str): repository root.
        Returns: bool — True if inside a work tree.
        """
        code, out, _ = GitTools._run(["rev-parse", "--is-inside-work-tree"], root)
        return code == 0 and out.strip() == "true"

    @staticmethod
    def status(root: str = ".", porcelain: bool = True) -> str:
        """
        Purpose: Get Git status (optionally porcelain).
        When to use: Show dirty files before deciding what to commit.
        Args:
          - root (str): repo root.
          - porcelain (bool): True → '--porcelain=v1'.
        Returns: str — git status output.
        """
        args = ["status", "--porcelain=v1"] if porcelain else ["status"]
        _, out, _ = GitTools._run(args, root)
        return out

    @staticmethod
    def current_branch(root: str = ".") -> str:
        """
        Purpose: Get the current branch name.
        Returns: str — branch name or "" on failure.
        """
        code, out, _ = GitTools._run(["rev-parse", "--abbrev-ref", "HEAD"], root)
        return out.strip() if code == 0 else ""

    @staticmethod
    def diff(root: str = ".", staged: bool = False) -> str:
        """
        Purpose: Obtain a unified diff of unstaged or staged changes.
        When to use: Feed into QA or apply as review artifact.
        Args:
          - root (str): repo root.
          - staged (bool): True → include staged changes ('--staged').
        Returns: str — unified diff (-U3).
        """
        args = ["diff", "--unified=3"]
        if staged:
            args.append("--staged")
        _, out, _ = GitTools._run(args, root)
        return out

    @staticmethod
    def ensure_branch(root: str = ".", name: str = "", create: bool = False, checkout: bool = False) -> Dict[str, Any]:
        """
        Purpose: Ensure a branch exists and optionally checkout.
        Args:
          - root (str): repo root.
          - name (str): branch name (required).
          - create (bool): create if missing.
          - checkout (bool): checkout the branch after ensure.
        Returns: dict — {"ok": bool, "branch"?: str, "error"?: str}
        """
        if not name:
            return {"ok": False, "error": "no branch name"}
        if create:
            code, out, err = GitTools._run(["branch", name], root)
            if code != 0 and "already exists" not in err.lower():
                return {"ok": False, "error": err.strip()}
        if checkout:
            code, out, err = GitTools._run(["checkout", name], root)
            if code != 0:
                return {"ok": False, "error": err.strip()}
        return {"ok": True, "branch": name}

    @staticmethod
    def commit(root: str = ".", message: str = "", add_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Purpose: Add and commit changes.
        When to use: After QA passes to persist changes in history.
        Args:
          - root (str): repo root.
          - message (str): commit message (default generic).
          - add_paths (List[str]|None): specific paths to stage; None → stage all -A.
        Returns: dict — {"ok": bool, "stdout": str, "stderr": str} (ok=False on failure).
        """
        if add_paths:
            code, out, err = GitTools._run(["add"] + add_paths, root)
            if code != 0:
                return {"ok": False, "error": err.strip()}
        else:
            code, out, err = GitTools._run(["add", "-A"], root)
            if code != 0:
                return {"ok": False, "error": err.strip()}

        code, out, err = GitTools._run(["commit", "-m", message or "chore: agentic edit"], root)
        ok = (code == 0)
        return {"ok": ok, "stdout": out, "stderr": err}

class TestTools:
    """
    Purpose: Run repository test suites for verification.
    When to use: As a validation step after edits; enables automatic gating.
    """

    @staticmethod
    def has_pytest(root: str = ".") -> bool:
        """
        Purpose: Check if pytest is available in PATH (inside this repo environment).
        Returns: bool — True if 'pytest --version' exits with 0.
        """
        try:
            proc = subprocess.run(["pytest", "--version"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return proc.returncode == 0
        except Exception:
            return False

    @staticmethod
    def run_pytest(root: str = ".", args: Optional[List[str]] = None, timeout_s: int = 900) -> Dict[str, Any]:
        """
        Purpose: Execute pytest within the given repo.
        When to use: Post-change verification; can be wired into the plan DAG.
        Args:
          - root (str): repo root.
          - args (List[str]|None): extra pytest args; default ['-q'].
          - timeout_s (int): hard timeout for the test run (seconds).
        Returns: dict — {"ok": bool, "returncode": int, "stdout": str, "stderr": str} or {"ok": False, "error": str}
        Failure modes: Returns ok=False on absence, timeout, or non-zero exit.
        Example: TestTools.run_pytest(".", ["-q"])
        """
        if not TestTools.has_pytest(root):
            return {"ok": False, "error": "pytest not available"}
        cmd = ["pytest"] + (args or ["-q"])
        try:
            proc = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, text=True)
            ok = (proc.returncode == 0)
            return {"ok": ok, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "pytest timeout"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
'''

# If tools.py is missing, create it with our DEFAULT_TOOLS content
if not TOOLS_PATH.exists():
    print("[setup] Creating default tools.py ...")
    TOOLS_PATH.write_text(DEFAULT_TOOLS)

# ──────────────────────────────────────────────────────────────────────────────
# VI. Ollama config + model discovery (python + REST fallback)
# ──────────────────────────────────────────────────────────────────────────────
class AppConfig(BaseModel):
    host: str = Field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    model: str
    options: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.2})

def _client_for(host: str) -> Client:
    return Client(host=host)

def _http_list_models(host: str) -> List[Dict[str, Any]]:
    url = host.rstrip("/") + "/api/tags"
    r = httpx.get(url, timeout=5.0)
    r.raise_for_status()
    data = r.json()
    models = data.get("models", [])
    return models if isinstance(models, list) else []

def _py_list_models(cli: Client) -> List[Dict[str, Any]]:
    try:
        resp = ollama.list()
    except Exception:
        resp = cli.list()
    if isinstance(resp, dict) and "models" in resp:
        models = resp["models"]
    elif isinstance(resp, list):
        models = resp
    else:
        models = []
    return models if isinstance(models, list) else []

def _list_models(cli: Client, host: str) -> List[Dict[str, Any]]:
    models = _py_list_models(cli)
    if not models:
        try:
            models = _http_list_models(host)
            if models:
                print("[setup] (fallback) Listed models via REST /api/tags.")
        except Exception as e:
            print("[warn] REST /api/tags fallback failed:", e)
    return models

def initial_setup() -> AppConfig:
    host_env = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    cli = _client_for(host_env)
    print(f"[setup] Querying Ollama at {host_env} for local models ...")
    try:
        models = _list_models(cli, host_env)
    except Exception as e:
        print("[fatal] Could not contact Ollama. Is `ollama serve` running?")
        print("Error:", e); sys.exit(2)
    if not models:
        print("[setup] No local models found via client or REST. Example: `ollama pull qwen3:4b` then rerun.")
        sys.exit(3)
    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        name = m.get("name") or m.get("model") or "unknown"
        size = m.get("size")
        size_mb = f"{(size/1024/1024):.1f} MB" if isinstance(size, (int, float)) else "?"
        family = (m.get("details") or {}).get("family", "")
        print(f"  [{i}] {name:<28} ({family})  {size_mb}")
    sel = None
    while sel is None:
        raw = input("\nSelect model by number: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(models):
                sel = models[idx-1]
        except Exception:
            pass
    model_name = sel.get("name") or sel.get("model")
    cfg = AppConfig(host=host_env, model=model_name)
    jdump(CONFIG_PATH, cfg.model_dump())
    print(f"[setup] Saved config.json with model='{cfg.model}' host='{cfg.host}'")
    return cfg

def load_config() -> AppConfig:
    cfg = jload(CONFIG_PATH)
    if not cfg: return initial_setup()
    try:
        return AppConfig(**cfg)
    except ValidationError:
        return initial_setup()

CONFIG = load_config()
CLIENT = _client_for(CONFIG.host)

# ──────────────────────────────────────────────────────────────────────────────
# VII. tool discovery + per-tool docs generation (incremental) + aliases
# ──────────────────────────────────────────────────────────────────────────────
def import_tools_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("tools", str(TOOLS_PATH))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod  # type: ignore

class ArgSpec(BaseModel):
    name: str
    type: str
    required: bool = True
    default: Optional[Any] = None
    doc: Optional[str] = None

class ToolSpec(BaseModel):
    name: str           # display / registry key ("foo" or "Class.method")
    qualname: str       # same as name (kept for clarity)
    origin: str         # "function" or "class:ClassName"
    doc: str = ""       # docstring pulled from object
    args: List[ArgSpec] = Field(default_factory=list)
    returns: Optional[str] = None

def _typename(t: Any) -> str:
    if t is None:
        return "None"
    if getattr(t, "__origin__", None) is Union:
        args = [a.__name__ if hasattr(a, "__name__") else str(a) for a in t.__args__]
        return "Union[" + ", ".join(args) + "]"
    if hasattr(t, "__name__"):
        return t.__name__
    return str(t)

def discover_tools(mod: types.ModuleType) -> List[ToolSpec]:
    tools: List[ToolSpec] = []
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
            ann = get_type_hints(obj); sig = inspect.signature(obj)
            args: List[ArgSpec] = []
            for p in sig.parameters.values():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                arg_t = _typename(ann.get(p.name, Any))
                required = p.default is inspect._empty
                default = None if required else p.default
                args.append(ArgSpec(name=p.name, type=arg_t, required=required, default=default))
            ret = _typename(ann.get("return", Any))
            tools.append(ToolSpec(
                name=name, qualname=name, origin="function",
                doc=(inspect.getdoc(obj) or ""), args=args, returns=ret
            ))
        if inspect.isclass(obj) and obj.__module__ == mod.__name__ and name.endswith("Tools"):
            for meth_name, meth in inspect.getmembers(obj, predicate=inspect.isfunction):
                if meth_name.startswith("_"):
                    continue
                ann = get_type_hints(meth); sig = inspect.signature(meth)
                args: List[ArgSpec] = []
                for p in sig.parameters.values():
                    if p.name in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                        continue
                    arg_t = _typename(ann.get(p.name, Any))
                    required = p.default is inspect._empty
                    default = None if required else p.default
                    args.append(ArgSpec(name=p.name, type=arg_t, required=required, default=default))
                ret = _typename(ann.get("return", Any))
                tools.append(ToolSpec(
                    name=f"{name}.{meth_name}", qualname=f"{name}.{meth_name}",
                    origin=f"class:{name}", doc=(inspect.getdoc(meth) or ""),
                    args=args, returns=ret
                ))
    return tools

# -------- incremental per-tool docs inference --------
def _safe_default(v: Any) -> Any:
    try:
        json.dumps(v)
        return v
    except Exception:
        return repr(v)

def _tool_fingerprint(t: ToolSpec) -> str:
    """Stable signature of a tool's public surface + docstring."""
    blob = {
        "name": t.name,
        "origin": t.origin,
        "doc": (t.doc or "").strip(),
        "returns": t.returns,
        "args": [
            {"name": a.name, "type": a.type, "required": a.required, "default": _safe_default(a.default)}
            for a in t.args
        ],
    }
    s = json.dumps(blob, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h

def _generate_single_tool_doc(t: ToolSpec) -> Dict[str, Any]:
    """
    Ask the model to infer one tool's doc entry.
    Returns a dict with keys: purpose, when_to_use, args{...}, returns, example.
    """
    SYSTEM = (
        "You are ToolDocBot.\n"
        "Given ONE Python tool (its docstring, args with types/defaults, and return type), "
        "output STRICT JSON with keys: "
        "{purpose, when_to_use, args:{<arg>:desc}, returns, example}. "
        "Keep it concise and actionable. No prose outside JSON."
    )
    payload = {
        "name": t.name,
        "origin": t.origin,
        "doc": t.doc or "",
        "returns": t.returns,
        "args": [
            {"name": a.name, "type": a.type, "required": a.required, "default": _safe_default(a.default)}
            for a in t.args
        ],
    }
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        resp = CLIENT.chat(model=CONFIG.model, messages=msgs, options=CONFIG.options | {"temperature": 0.1})
        content = (resp.get("message", {}) or {}).get("content", "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            content = content.split("\n", 1)[1] if "\n" in content else content
        out = json.loads(content)
        if not isinstance(out, dict):
            raise ValueError("model returned non-object")
        return out
    except Exception:
        # Fallback: introspection-only skeleton
        return {
            "purpose": (t.doc or f"{t.name} tool").split("\n", 1)[0].strip(),
            "when_to_use": "When its purpose matches your need.",
            "args": {a.name: f"{a.type}" for a in t.args},
            "returns": t.returns,
            "example": f"{t.name}({', '.join(a.name for a in t.args)})",
        }

def ensure_docs(tools: List[ToolSpec], force: bool=False) -> Dict[str, Any]:
    if DOCS_PATH.exists() and not force:
        try:
            return jload(DOCS_PATH, {})
        except Exception:
            pass

    payload = [
        {
            "name": t.name,
            "origin": t.origin,
            "doc": t.doc,
            "args": [a.model_dump() for a in t.args],
            "returns": t.returns
        }
        for t in tools
    ]

    SYSTEM = (
        "You are ToolDocBot.\n"
        "Given Python tools (docstrings, arg types/defaults, returns), output STRICT JSON mapping tool name → doc "
        "with keys {purpose, when_to_use, args:{<arg>:desc}, returns, example}. No prose outside JSON."
    )
    USER = {"instruction": "Write JSON docs for these tools.", "tools_introspection": payload}

    print(f"[docs] Streaming docs.json generation for {len(payload)} tools ...")
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(USER, ensure_ascii=False)},
    ]
    if DEBUG:
        _dbg("TOOLDOCS", msgs)

    def _extract_json_obj(text: str) -> Dict[str, Any]:
        s = (text or "").strip()
        # strip code fences if present
        if s.startswith("```"):
            s = s.strip("`")
            s = s.split("\n", 1)[1] if "\n" in s else s
        # try strict
        try:
            return json.loads(s)
        except Exception:
            pass
        # try to salvage a top-level object from partial output
        a = s.find("{")
        b = s.rfind("}")
        if a != -1 and b != -1 and b > a:
            cand = s[a:b+1]
            try:
                return json.loads(cand)
            except Exception:
                pass
        raise ValueError("Could not parse JSON from streamed content")

    try:
        content = _chat_stream_text(msgs, temp=0.1, label="TOOLDOCS", timeout_s=None, max_chars=None)
        docs = _extract_json_obj(content)
    except KeyboardInterrupt:
        print("[docs] ^C — using partial content if possible.", flush=True)
        try:
            docs = _extract_json_obj(content)
        except Exception:
            docs = {}
    except Exception as e:
        print("[warn] Tool docs generation failed, falling back to introspection-only.", e)
        docs = {
            t.name: {
                "purpose": (t.doc or f"{t.name} tool"),
                "when_to_use": "When its purpose matches your need.",
                "args": {a.name: f"{a.type}" for a in t.args},
                "returns": t.returns,
                "example": f"{t.name}({', '.join(a.name for a in t.args)})",
            }
            for t in tools
        }

    jdump(DOCS_PATH, docs)
    print(f"[docs] Wrote {DOCS_PATH.name}: {len(docs)} entries.")
    return docs



def build_tool_registry(mod: types.ModuleType) -> Dict[str, Callable[..., Any]]:
    reg: Dict[str, Callable[..., Any]] = {}
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
            reg[name] = obj
        if inspect.isclass(obj) and obj.__module__ == mod.__name__ and name.endswith("Tools"):
            inst = obj()
            for meth_name, meth in inspect.getmembers(inst, predicate=callable):
                if meth_name.startswith("_"):
                    continue
                if inspect.ismethod(meth) or inspect.isfunction(meth):
                    reg[f"{obj.__name__}.{meth_name}"] = meth
    return reg

TOOLS_MOD = import_tools_module()
TOOLS = discover_tools(TOOLS_MOD)
DOCS = ensure_docs(TOOLS, force=False)
REGISTRY = build_tool_registry(TOOLS_MOD)

# Aliases: map simple method names to "Class.method" if unique; functions map to themselves.
TOOL_ALIASES: Dict[str, str] = {}
for full in list(REGISTRY.keys()):
    if "." in full:
        simple = full.split(".")[-1]
        # avoid overwriting if multiple classes share same method name
        TOOL_ALIASES.setdefault(simple, full)
    else:
        TOOL_ALIASES.setdefault(full, full)

def resolve_tool_name(name: str) -> str:
    if name in REGISTRY:
        return name
    return TOOL_ALIASES.get(name, name)


# ──────────────────────────────────────────────────────────────────────────────
# VIII. Repo Route Context (dynamic prompt routing substrate)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_keywords(goal: str, max_k: int = 12) -> List[str]:
    # quick heuristic: words > 2 chars, strip punctuation, dedupe
    toks = re.findall(r"[A-Za-z0-9_/.\-]{3,}", goal.lower())
    seen, out = set(), []
    for t in toks:
        if t in seen: continue
        seen.add(t); out.append(t)
        if len(out) >= max_k: break
    return out

def ensure_repo_index() -> Dict[str, Any]:
    idx = jload(REPO_INDEX_PATH)
    if idx and idx.get("files"):
        return idx
    # index using WorkspaceTools from tools.py
    try:
        index_repo = REGISTRY.get("WorkspaceTools.index_repo")
        assert callable(index_repo)
        idx = index_repo(".")
        trace("repo_indexed", {"scanned": idx.get("scanned", 0)})
        return idx
    except Exception as e:
        trace("repo_index_error", {"error": str(e)})
        return {"root": str(BASE_DIR), "files": [], "scanned": 0}

def build_route_context(goal: str) -> Dict[str, Any]:
    # Ensure index exists, then produce ranked file candidates
    idx = ensure_repo_index()
    keywords = _extract_keywords(goal)
    candidates = []
    try:
        route = REGISTRY["WorkspaceTools.route_candidates"](keywords, 30)
        candidates = route.get("candidates", [])
    except Exception as e:
        trace("route_candidates_error", {"error": str(e)})
    # Minimal tools catalog with actionable notes
    tools_catalog = [
        {"name":"WorkspaceTools.read_span", "when":"Inspect a small section before editing.", "args":"{rel,start_line,end_line}"},
        {"name":"WorkspaceTools.write_span","when":"Edit bounded lines safely.","args":"{rel,start_line,end_line,new_text,guard_sha1?}"},
        {"name":"WorkspaceTools.regex_replace","when":"Targeted pattern updates.","args":"{rel,pattern,repl,flags?,count?}"},
        {"name":"WorkspaceTools.apply_unified_diff","when":"Apply multi-file patch.","args":"{diff_text,strip?,dry_run?,root?}"},
        {"name":"WorkspaceTools.verify_file_contains","when":"Post-check expected strings.","args":"{rel,must_include[]}"},
        {"name":"GitTools.diff","when":"Review changes pre-commit.","args":"{root?,staged?}"},
        {"name":"GitTools.commit","when":"Integrate changes into repo.","args":"{root?,message?,add_paths?}"},
        {"name":"TestTools.run_pytest","when":"Run repo tests as verifier.","args":"{root?,args?,timeout_s?}"}
    ]
    return {
        "keywords": keywords,
        "index_stats": {"scanned": idx.get("scanned", 0)},
        "candidates": candidates[:20],
        "tools": tools_catalog
    }

# ──────────────────────────────────────────────────────────────────────────────
# IX. DAG models + strict plan parsing + placeholder/dep repair
# ──────────────────────────────────────────────────────────────────────────────
class Node(BaseModel):
    id: str; tool: str; args: Any=Field(default_factory=dict)
    after: List[str]=Field(default_factory=list)
    retries: int=1; timeout_s: int=30; alias: Optional[str]=None

class Plan(BaseModel):
    nodes: List[Node]
    meta: Dict[str, Any]=Field(default_factory=lambda: {"parallelism":4, "retries":1, "timeout_s":180})

class ToolRecord(BaseModel):
    node_id: str; tool: str; args_resolved: Any
    output: Any=None; stdout_short: Optional[str]=None
    error: Optional[Dict[str, Any]]=None
    started_at: str=Field(default_factory=_now_iso)
    ended_at: Optional[str]=None; elapsed_ms: Optional[int]=None

class TurnState(BaseModel):
    turn_id: str; plan_id: str; graph: Plan
    pending: List[str]; completed: List[str]=Field(default_factory=list)
    tool_outputs: Dict[str, ToolRecord]=Field(default_factory=dict)

def parse_plan_strict(s: str) -> Dict[str, Any]:
    s = s.strip()
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        raise ValueError("no JSON object found")
    obj = json.loads(s[a:b+1])
    if "nodes" not in obj or not isinstance(obj["nodes"], list):
        raise ValueError("missing or invalid 'nodes' list")
    return obj

PLACEHOLDER_RE = re.compile(
    r"\[([A-Za-z_]\w*(?:\.(?:output|args))(?:\.[A-Za-z0-9_\-]+)+)\]"
)
UNBRACKETED_RE = re.compile(r"\b([A-Za-z_]\w*)\.output(?:\.[A-Za-z0-9_\-\[\]]+)+")
def validate_and_fix_placeholders(plan: Plan) -> Plan:
    """
    - Normalizes placeholders in node.args:
        * Wraps unbracketed refs like read.output.text -> [read.output.text] (and .args.* too)
        * Only rewrites *valid* refs (nodeId.(output|args).path); leaves others intact
        * Avoids touching regex/diff/text payload fields that commonly contain brackets
    - Auto-adds 'after' dependencies for any nodeIds referenced via placeholders.
    - Coerces 'after' entries that are tool names (full or simple) to the first node using that tool.
    - Dedupes and cleans 'after' while excluding self-dependencies.
    """
    node_ids = {n.id for n in plan.nodes}

    # Map tool -> node ids; include both full and simple tool names for robustness.
    tool_to_nodes: Dict[str, List[str]] = {}
    for n in plan.nodes:
        tool_to_nodes.setdefault(n.tool, []).append(n.id)
        if "." in n.tool:  # also index simple suffix e.g. 'WorkspaceTools.regex_replace' → 'regex_replace'
            simple = n.tool.split(".")[-1]
            tool_to_nodes.setdefault(simple, []).append(n.id)

    # Keys whose string values should NEVER be scanned/re-written for placeholders
    # because they frequently contain [] (regex char classes, patch hunks, raw text, etc.).
    SKIP_KEYS = {"pattern", "repl", "replacement", "diff_text", "new_text", "content"}

    # Accept tokens like "node.output.foo.bar" or "node.args.k"
    VALID_TOKEN_RE = re.compile(r"^[A-Za-z_]\w*\.(?:output|args)\.[A-Za-z0-9_\-\.]+$")

    # Wrap both .output.* and .args.* when unbracketed
    UNBRACKETED_BOTH_RE = re.compile(
        r"\b([A-Za-z_]\w*)\.(?:output|args)(?:\.[A-Za-z0-9_\-\[\]]+)+"
    )

    def wrap_if_unbracketed(s: str) -> str:
        # Turn unbracketed node.output/args references into bracketed form
        def repl(m): return "[" + m.group(0) + "]"
        return UNBRACKETED_BOTH_RE.sub(repl, s)

    def _fix_token(token: str) -> str:
        # If it isn't a valid placeholder token, keep as-is (still bracketed).
        if not VALID_TOKEN_RE.match(token):
            return "[" + token + "]"

        parts = token.split(".")
        head = parts[0] if parts else ""
        if head not in node_ids:
            # Allow using a tool name (full or simple) and map it to the first node using that tool.
            cands = tool_to_nodes.get(head, [])
            if cands:
                parts[0] = cands[0]
        return "[" + ".".join(parts) + "]"

    def fix_in_string(s: str) -> str:
        # 1) Wrap unbracketed refs
        s = wrap_if_unbracketed(s)
        # 2) Normalize only valid placeholder tokens
        return PLACEHOLDER_RE.sub(lambda m: _fix_token(m.group(1)), s)

    def walk_args(x: Any, key_ctx: Optional[str] = None) -> Any:
        if isinstance(x, str):
            # never touch regex/diff/text payloads
            if key_ctx in SKIP_KEYS:
                return x
            return fix_in_string(x)
        if isinstance(x, list):
            return [walk_args(v, None) for v in x]
        if isinstance(x, dict):
            return {k: walk_args(v, k) for k, v in x.items()}
        return x

    # Apply normalization over args
    for n in plan.nodes:
        n.args = walk_args(n.args, None)

    # Collect deps from args placeholders to auto-wire 'after'
    def deps_from_args(args: Any) -> List[str]:
        deps: List[str] = []

        def visit(x: Any, key_ctx: Optional[str]):
            if isinstance(x, str):
                if key_ctx in SKIP_KEYS:
                    return
                # Ensure unbracketed refs are detectable
                s = wrap_if_unbracketed(x)
                for tok in PLACEHOLDER_RE.findall(s):
                    # Record heads of valid tokens only
                    if VALID_TOKEN_RE.match(tok):
                        head = tok.split(".", 1)[0]
                        deps.append(head)
            elif isinstance(x, list):
                for v in x:
                    visit(v, None)
            elif isinstance(x, dict):
                for k, v in x.items():
                    visit(v, k)

        visit(args, None)
        return deps

    # Finalize 'after' with derived deps; map tool names to node ids; prune + dedupe
    for n in plan.nodes:
        wanted: List[str] = []

        # existing after (coerce tool names → first node id)
        for dep in (n.after or []):
            dep_id = dep.split(".")[0] if isinstance(dep, str) else dep
            if dep_id in node_ids:
                wanted.append(dep_id)
            else:
                cands = tool_to_nodes.get(dep_id, [])
                if cands:
                    wanted.append(cands[0])

        # derived deps from args placeholders
        for dep_head in deps_from_args(n.args):
            dep_id = dep_head
            if dep_id not in node_ids:
                cands = tool_to_nodes.get(dep_id, [])
                if cands:
                    dep_id = cands[0]
            if dep_id in node_ids:
                wanted.append(dep_id)

        # remove self and dedupe preserving order
        seen = set()
        clean_after: List[str] = []
        for d in wanted:
            if d == n.id:
                continue
            if d not in seen:
                seen.add(d)
                clean_after.append(d)

        n.after = clean_after

    return plan


def run_node(node: Node, registry: Dict[str, Callable[..., Any]], outputs: Dict[str, ToolRecord]) -> ToolRecord:
    start = time.time(); rec = ToolRecord(node_id=node.id, tool=node.tool, args_resolved=None)
    try:
        tool_name = resolve_tool_name(node.tool)
        fn = registry.get(tool_name)
        if not fn:
            raise RuntimeError(f"Unknown tool: {node.tool}")
        args_resolved = _resolve_placeholders(node.args, outputs); rec.args_resolved = args_resolved
        if isinstance(args_resolved, dict): out = fn(**args_resolved)
        elif isinstance(args_resolved, list): out = fn(*args_resolved)
        else: out = fn(args_resolved)
        rec.output = out; rec.stdout_short = str(out)[:200]; rec.error = None
    except Exception as e:
        rec.error = {"type": e.__class__.__name__, "message": str(e), "retryable": False}
    finally:
        rec.ended_at = _now_iso(); rec.elapsed_ms = int((time.time()-start)*1000)
    return rec

def _topo_ready(state: TurnState) -> List[Node]:
    id_to_node = {n.id: n for n in state.graph.nodes}
    ready = []
    for nid in list(state.pending):
        node = id_to_node[nid]
        if all(dep in state.completed for dep in node.after): ready.append(node)
    return ready

# ──────────────────────────────────────────────────────────────────────────────
# X. Concurrency executor with retries
# ──────────────────────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed

def execute_plan(plan: Plan, registry: Dict[str, Callable[..., Any]]) -> Tuple[List[ToolRecord], Dict[str, ToolRecord]]:
    parallelism = int(plan.meta.get("parallelism", 4))
    max_retries_default = int(plan.meta.get("retries", 1))

    state = TurnState(
        turn_id=f"turn-{int(time.time()*1000)}",
        plan_id=f"plan-{hash(json.dumps(plan.model_dump(), sort_keys=True)) & 0xfffffff}",
        graph=plan,
        pending=[n.id for n in plan.nodes],
    )
    retry_counts = {n.id: int(getattr(n, "retries", max_retries_default)) for n in plan.nodes}
    records: Dict[str, ToolRecord] = {}

    while state.pending:
        ready = _topo_ready(state)
        if not ready:
            print("[executor] Deadlock or unmet deps. Breaking."); break

        batch = ready[:max(1, parallelism)]
        future_map = {}
        with ThreadPoolExecutor(max_workers=parallelism) as tp:
            for node in batch:
                future = tp.submit(run_node, node, registry, records)
                future_map[future] = node

            for fut in as_completed(future_map):
                node = future_map[fut]
                rec = fut.result()
                records[node.id] = rec
                if rec.error and retry_counts.get(node.id, 0) > 0:
                    retry_counts[node.id] -= 1
                    print(f"[node] {node.id} {node.tool} → RETRY ({rec.elapsed_ms} ms) remaining={retry_counts[node.id]}")
                else:
                    if rec.error:
                        print(f"[node] {node.id} {node.tool} → ERR ({rec.elapsed_ms} ms): {rec.error}")
                    else:
                        print(f"[node] {node.id} {node.tool} → OK ({rec.elapsed_ms} ms)")
                    if node.id in state.pending:
                        state.pending.remove(node.id)
                    state.completed.append(node.id)

    ordered = [records[nid] for nid in [n.id for n in plan.nodes] if nid in records]
    return ordered, records

# ──────────────────────────────────────────────────────────────────────────────
# XI. LLM helpers (clarify, plan, red-team, reflect, assemble, QA review)
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# XI. LLM helpers (clarify, plan, red-team, reflect, assemble, QA review)
# ──────────────────────────────────────────────────────────────────────────────
def _chat_stream_text(
    messages: List[Dict[str, str]],
    temp: float = 0.1,
    label: str = "LLM",
    timeout_s: Optional[float] = None,
    max_chars: Optional[int] = None,
) -> str:
    """
    Stream tokens from Ollama and print them live to the terminal.
    - Gracefully handles KeyboardInterrupt (returns partial content).
    - Optional timeout and max_chars to prevent indefinite streams.
    - For planning/repair/QA phases, enforces JSON format, low temperature, and ASCII-only output.
    - Falls back to a single non-streaming call on transport errors.

    Returns the full (or partial) assistant content as a string.
    """
    # default timeout from env if not provided
    if timeout_s is None:
        try:
            timeout_s = float(os.environ.get("AGENT_STREAM_TIMEOUT", "180"))
        except Exception:
            timeout_s = 180.0

    # Per-label stricter options (reduce hallucinations and multilingual drift)
    label_upper = str(label or "").upper()
    jsonish_prefixes = ("PLAN", "REFLECT", "PLAN-MERGE", "RED-TEAM", "QA")
    force_json = any(label_upper.startswith(p) for p in jsonish_prefixes)

    opts = CONFIG.options | {
        "temperature": 0.0 if force_json else temp,
        "top_p": 0.9 if force_json else CONFIG.options.get("top_p", 0.9),
        "repeat_penalty": 1.1 if force_json else CONFIG.options.get("repeat_penalty", 1.1),
    }
    if force_json:
        # Many Ollama models honor 'format': 'json' to bias toward valid JSON output
        opts["format"] = "json"

    out_parts: List[str] = []
    start_ts = time.time()

    # honor global streaming toggle
    if not STREAM:
        try:
            r = CLIENT.chat(model=CONFIG.model, messages=messages, options=opts)
            content = (r.get("message", {}) or {}).get("content", "") or ""
            content = content.strip()
            if force_json:
                # Keep output ASCII-only for planning/repair/QA phases
                try:
                    content = content.encode("ascii", "ignore").decode("ascii")
                except Exception:
                    pass
            return content
        except Exception as e:
            print(f"[stream:{label}] ERROR (non-stream fallback): {e}", flush=True)
            return ""

    try:
        print(f"\n[stream:{label}] ", end="", flush=True)
        # streaming path
        for part in CLIENT.chat(model=CONFIG.model, messages=messages, options=opts, stream=True):
            msg = part.get("message") or {}
            chunk = msg.get("content") or ""
            if chunk:
                # Print raw; collect for return
                print(chunk, end="", flush=True)
                out_parts.append(chunk)

            # timeout guard
            if timeout_s and (time.time() - start_ts) > timeout_s:
                print(f"\n[stream:{label}] timeout after {timeout_s:.0f}s — returning partial output.", flush=True)
                break

            # size guard
            if max_chars and sum(len(x) for x in out_parts) >= max_chars:
                print(f"\n[stream:{label}] reached {max_chars} chars — returning partial output.", flush=True)
                break

        print("", flush=True)
        text = "".join(out_parts).strip()
        if force_json:
            # Keep output ASCII-only for planning/repair/QA phases
            try:
                text = text.encode("ascii", "ignore").decode("ascii")
            except Exception:
                pass
        return text

    except KeyboardInterrupt:
        print(f"\n[stream:{label}] ^C — returning partial output.", flush=True)
        text = "".join(out_parts).strip()
        if force_json:
            try:
                text = text.encode("ascii", "ignore").decode("ascii")
            except Exception:
                pass
        return text

    except Exception as e:
        # If we have partial content, return it; else try non-stream fallback
        partial = "".join(out_parts).strip()
        print(f"\n[stream:{label}] ERROR: {e}", flush=True)
        if partial:
            print(f"[stream:{label}] Using partial output due to error.", flush=True)
            if force_json:
                try:
                    partial = partial.encode("ascii", "ignore").decode("ascii")
                except Exception:
                    pass
            return partial
        try:
            r = CLIENT.chat(model=CONFIG.model, messages=messages, options=opts)
            content = (r.get("message", {}) or {}).get("content", "") or ""
            content = content.strip()
            if force_json:
                try:
                    content = content.encode("ascii", "ignore").decode("ascii")
                except Exception:
                    pass
            return content
        except Exception as e2:
            print(f"[stream:{label}] Fallback also failed: {e2}", flush=True)
            return ""


def _llm(system: str, user_obj: Any, temp=0.1, preface: Optional[str]=None, label: str="LLM") -> str:
    # Ensure STRICT_SYSTEM exists (prevents NameError if not defined elsewhere)
    if "STRICT_SYSTEM" not in globals():
        # When True, we DO NOT interleave assistant prefaces; system message is highest priority.
        globals()["STRICT_SYSTEM"] = True

    msgs = [{"role":"system","content":system}]
    # Only add an assistant preface if STRICT_SYSTEM is off and a preface was explicitly provided
    if (not STRICT_SYSTEM) and INTERLEAVE and preface:
        msgs.append({"role":"assistant","content":preface})
    msgs.append({"role":"user","content":json.dumps(user_obj, ensure_ascii=False)})
    _dbg(label, msgs)

    label_upper = str(label or "").upper()
    jsonish_prefixes = ("PLAN", "REFLECT", "PLAN-MERGE", "RED-TEAM", "QA")
    force_json = any(label_upper.startswith(p) for p in jsonish_prefixes)

    opts = CONFIG.options | {
        "temperature": 0.0 if force_json else temp,
        "top_p": 0.9 if force_json else CONFIG.options.get("top_p", 0.9),
        "repeat_penalty": 1.1 if force_json else CONFIG.options.get("repeat_penalty", 1.1),
    }
    if force_json:
        opts["format"] = "json"

    r = CLIENT.chat(model=CONFIG.model, messages=msgs, options=opts)
    content = r.get("message", {}).get("content","") or ""
    content = content.strip()

    # Strip code fences universally; some models wrap JSON
    if content.startswith("```"):
        content = content.strip("`")
        content = content.split("\n", 1)[1] if "\n" in content else ""

    # ASCII-only for planning/repair/QA phases to avoid multilingual drift
    if force_json:
        try:
            content = content.encode("ascii", "ignore").decode("ascii")
        except Exception:
            pass
    return content


def clarify_goal(goal: str) -> str:
    SYSTEM = sysmsg("clarify", {"REALITY_POLICY": sysmsg("reality_policy")})
    # Strictly pass only the goal; no bus tail, no extra assistant preface
    user = {"goal": goal}
    attempt = 0
    last = ""
    while attempt < 2:
        attempt += 1
        try:
            r = _llm(SYSTEM, user, temp=0.1, preface=None, label=f"CLARIFY#{attempt}")
        except Exception:
            r = goal
        # post-filter any verbosity the model might add
        r = _strip_meta(r)
        r = _first_n_lines(r, 2)
        last = r or goal
        if last:
            break
    return last or goal


def plan_with_llm(goal: str, tools: List[ToolSpec], docs: Dict[str, Any], route_context: Dict[str, Any]) -> Plan:
    available = sorted(list(REGISTRY.keys()) + list(TOOL_ALIASES.keys()))
    signatures = _tool_signatures_for_prompt(tools)

    SYSTEM = sysmsg("plan", {
        "AVAILABLE_TOOLS": json.dumps(available),
        "REALITY_POLICY_SECTION": sysmsg("reality_policy"),
        "TOOL_SIGNATURES": signatures,  # <-- new
    })

    # Build a compact tool catalog for the planner (via WorkspaceTools.list_tools)
    try:
        tools_brief = REGISTRY["WorkspaceTools.list_tools"]("all")
    except Exception:
        # Fallback: derive a brief list from discovered ToolSpec objects
        def _sigline(t: ToolSpec) -> str:
            parts = []
            for a in t.args:
                opt = "" if a.required else "?"
                parts.append(f"{a.name}{opt}:{a.type}")
            return f"({', '.join(parts)}) -> {t.returns}"
        tools_brief = {
            "tools": [
                {
                    "name": t.name,
                    "sig": f"{t.name}{_sigline(t)}",
                    "doc": (t.doc or "").split("\n", 1)[0],
                }
                for t in tools
            ],
            "count": len(tools),
        }
    # Only pass what’s necessary for planning; no conversation tail; no preface
    base_user = {
        "goal": goal,
        "tools": tools_brief,
        "docs": docs,
        "route_context": route_context
    }

    msgs = [{"role":"system","content":SYSTEM}, {"role":"user","content":json.dumps(base_user, ensure_ascii=False)}]
    _dbg("PLAN", msgs)

    for attempt in range(1, 4):
        content = _chat_stream_text(msgs, temp=0.05, label=f"PLAN#{attempt}")
        try:
            plan_obj = parse_plan_strict(content)
            plan = Plan(**plan_obj)
            plan = validate_and_fix_placeholders(plan)
            plan = _sanitize_and_coerce_plan(plan)
            if plan.nodes:
                return plan
        except Exception as e:
            msgs.append({"role":"assistant","content":content})
            msgs.append({"role":"user","content":f"ERROR: {e}. Re-output ONLY valid JSON matching the schema with nodes/meta (no wrapper)."})
            _dbg("PLAN-RETRY", msgs[-2:])

    # Deterministic fallbacks for common config.json edits
    gl = (goal or "").lower()
    target_file = _guess_target_file(goal) or (route_context.get("candidates") or [{}])[0].get("rel") or "config.json"

    # 1) Temperature change fallback
    if re.search(r"\btemp(?:erature)?\b", gl):
        m = re.search(r"(?<![\w.])(?:0?\.\d+|[01](?:\.\d+)?)(?![\w.])", gl)
        target_temp = m.group(0) if m else "0.8"
        nodes = [
            Node(id="read", tool="WorkspaceTools.read_file", args={"rel": target_file}, after=[]),
            Node(
                id="set_temp",
                tool="WorkspaceTools.regex_replace",
                args={
                    "rel": target_file,
                    "pattern": r'"temperature"\s*:\s*[0-9.]+',
                    "repl": f'"temperature": {target_temp}',
                    "flags": "",
                    "count": 1
                },
                after=["read"]
            ),
            Node(
                id="verify_temp",
                tool="WorkspaceTools.verify_file_contains",
                args={"rel": target_file, "must_include": [f'"temperature": {target_temp}']},
                after=["set_temp"]
            ),
        ]
        seeded = Plan(nodes=nodes, meta={"parallelism":1,"timeout_s":60})

    # 2) Model change to qwen3:4b fallback
    elif ("qwen3:4b" in gl) and ("config" in target_file.lower() or "model" in gl):
        nodes = [
            Node(id="read", tool="WorkspaceTools.read_file", args={"rel": target_file}, after=[]),
            Node(
                id="set_model",
                tool="WorkspaceTools.regex_replace",
                args={
                    "rel": target_file,
                    "pattern": r'("model"\s*:\s*")[^"]+(")',
                    "repl": r'\1qwen3:4b\2',
                    "flags": "",
                    "count": 1
                },
                after=["read"]
            ),
            Node(
                id="verify_model",
                tool="WorkspaceTools.verify_file_contains",
                args={"rel": target_file, "must_include": ['"model": "qwen3:4b"']},
                after=["set_model"]
            ),
        ]
        seeded = Plan(nodes=nodes, meta={"parallelism":1,"timeout_s":60})

    # 3) Safe minimal fallback (read-only)
    else:
        seeded = Plan(
            nodes=[Node(id="read", tool="WorkspaceTools.read_file", args={"rel": target_file}, after=[])],
            meta={"parallelism":1,"timeout_s":60}
        )

    return validate_and_fix_placeholders(_sanitize_and_coerce_plan(seeded))


def merge_plan(original: Plan, patch: Dict[str, Any]) -> Plan:
    id_to_node = {n.id: n for n in original.nodes}
    for nid in patch.get("remove", []):
        id_to_node.pop(nid, None)
    for m in patch.get("modify", []):
        nid = m.get("id")
        if nid and nid in id_to_node:
            node = id_to_node[nid]
            if "tool" in m: node.tool = m["tool"]
            if "args" in m: node.args = m["args"]
            if "after" in m: node.after = m["after"]
            if "retries" in m: node.retries = int(m["retries"])
            if "timeout_s" in m: node.timeout_s = int(m["timeout_s"])
    for a in patch.get("add", []):
        try:
            node = Node(**a)
            id_to_node[node.id] = node
        except Exception:
            continue
    merged = Plan(nodes=list(id_to_node.values()), meta=original.meta)
    return validate_and_fix_placeholders(merged)


def red_team_refine(goal: str, initial: Plan, tools: List[ToolSpec], route_context: Dict[str, Any]) -> Plan:
    SYSTEM = sysmsg("red_team")
    USER = {
        "goal": goal,
        "initial_plan": initial.model_dump(),
        "available_tools": sorted(list(REGISTRY.keys()) + list(TOOL_ALIASES.keys())),
        "docs": DOCS,
        "route_context": route_context
    }
    patch_raw = _llm(SYSTEM, USER, temp=0.2, preface=None, label="RED-TEAM")
    try:
        patch_obj = parse_plan_strict(patch_raw)
    except Exception:
        try:
            patch_obj = json.loads(patch_raw)
        except Exception:
            return initial
    merged = merge_plan(initial, patch_obj)
    merged = _sanitize_and_coerce_plan(merged)
    try:
        SYSTEM_MERGE = sysmsg("plan_merge")
        merged2 = _llm(SYSTEM_MERGE, {
            "goal": goal,
            "original": initial.model_dump(),
            "patch": patch_obj
        }, temp=0.0, preface=None, label="PLAN-MERGE")
        obj = parse_plan_strict(merged2)
        merged = validate_and_fix_placeholders(Plan(**obj))
    except Exception:
        pass
    return merged


def reflect_repair(goal: str, tools: List[ToolSpec], docs: Dict[str, Any], records: List[ToolRecord], original_plan: Plan, route_context: Dict[str, Any]) -> Optional[Plan]:
    errors = [r for r in records if r.error]
    if not errors:
        return None
    SYSTEM = sysmsg("reflect")
    payload = {
        "goal": goal,
        "original_plan": original_plan.model_dump(),
        "failed": [{"node_id": r.node_id, "tool": r.tool, "error": r.error} for r in errors],
        "tools": [{"name": t.name, "args":[a.name for a in t.args]} for t in tools],
        "docs": docs,
        "route_context": route_context
    }
    msgs = [{"role":"system","content":SYSTEM}, {"role":"user","content":json.dumps(payload, ensure_ascii=False)}]
    _dbg("REFLECT", msgs)
    for attempt in range(1, 4):
        content = _chat_stream_text(msgs, temp=0.05, label=f"REFLECT#{attempt}")
        try:
            plan_obj = parse_plan_strict(content)
            plan = Plan(**plan_obj)
            return validate_and_fix_placeholders(plan)
        except Exception as e:
            msgs.append({"role":"assistant","content":content})
            msgs.append({"role":"user","content":f"ERROR: {e}. Re-output ONLY valid JSON matching the schema."})
            _dbg("REFLECT-RETRY", msgs[-2:])
    return None


def assemble_final(goal: str, clarified: str, records: Dict[str, ToolRecord]) -> str:
    # If we read a file, return its contents directly (no LLM needed for simple read).
    for nid, rec in records.items():
        if rec.tool.endswith("WorkspaceTools.read_file") or rec.tool == "WorkspaceTools.read_file":
            out = rec.output or {}
            if isinstance(out, dict) and "text" in out and "path" in out:
                path = out.get("path", "")
                text = out.get("text", "")
                return f"### {path}\n\n```\n{text}\n```"

    # Otherwise, fall back to LLM assembly
    blocks = []
    for nid, rec in records.items():
        hdr = f"[tool:{rec.tool} → node={nid}] ({rec.elapsed_ms} ms)"
        short = rec.stdout_short or ""
        blocks.append(f"{hdr}\nsummary: {short}\njson: {json.dumps(rec.output)[:400]}")
    SYSTEM = sysmsg("assemble")
    user = {"goal": goal, "clarified": clarified, "tool_blocks": blocks}
    try:
        return _llm(SYSTEM, user, temp=0.2, label="ASSEMBLE", preface="Assemble concise final answer.")
    except Exception as e:
        return f"(finalization failed) {e}\n\nTool outputs:\n" + "\n\n".join(blocks)


def qa_review(requested_edit: str, diff_text: str, verifications: Dict[str, Any]) -> Dict[str, Any]:
    SYSTEM = sysmsg("qa_reviewer")
    BEFORE = verifications.get("before_snap", "")
    AFTER = verifications.get("after_snap", "")
    USER = {
        "requested_edit": requested_edit,
        "before_excerpt": BEFORE[:2000],
        "after_excerpt": AFTER[:2000],
        "diff": diff_text[:16000],  # avoid ballooning
        "verifications": verifications
    }
    raw = _llm(SYSTEM, USER, temp=0.0, label="QA")
    try:
        obj = json.loads(raw.strip().strip("`").split("\n",1)[-1] if raw.strip().startswith("```") else raw)
        if not isinstance(obj, dict): raise ValueError("not an object")
        obj.setdefault("pass", False)
        obj.setdefault("score", 0)
        obj.setdefault("reasons", [])
        obj.setdefault("risk", "medium")
        obj.setdefault("followups", [])
        return obj
    except Exception:
        return {"pass": False, "score": 0, "reasons": ["QA parsing failure"], "risk":"high", "followups":["Re-run QA"]}

# ──────────────────────────────────────────────────────────────────────────────
# XI.a Plan sanitizers and goal heuristics
# ──────────────────────────────────────────────────────────────────────────────

ALIAS_ARG_MAP = {
    "WorkspaceTools.read_file": {"file": "rel", "path": "rel"},
    "WorkspaceTools.read_span": {"file": "rel", "path": "rel"},
    "WorkspaceTools.regex_replace": {"file": "rel", "path": "rel"},
    "WorkspaceTools.verify_file_contains": {"file": "rel", "path": "rel"},
}
def _sanitize_and_coerce_plan(plan: Plan) -> Plan:
    """Drop unknown tools; coerce common arg alias mistakes; ensure minimal schema sanity."""
    valid = []
    for n in plan.nodes:
        # normalize tool name
        resolved = resolve_tool_name(n.tool)
        if resolved not in REGISTRY:
            continue

        # coerce args
        args = n.args if isinstance(n.args, dict) else ({} if n.args is None else n.args)
        if isinstance(args, dict):
            # generic aliases for workspace tools
            mapping = ALIAS_ARG_MAP.get(resolved)
            if mapping:
                for old, new in mapping.items():
                    if old in args and new not in args:
                        args[new] = args.pop(old)

            # special-case: verify_file_contains wants List[str] in 'must_include'
            if resolved == "WorkspaceTools.verify_file_contains":
                if "must_include" not in args:
                    for k in ("content", "text", "needle", "contains"):
                        if k in args:
                            v = args.pop(k)
                            if isinstance(v, list):
                                args["must_include"] = v
                            elif v is None:
                                args["must_include"] = []
                            else:
                                args["must_include"] = [str(v)]
                            break
                # last resort default to empty list (tool will return ok=False if empty)
                args.setdefault("must_include", [])

        # write back
        n.tool = resolved
        n.args = args

        # normalize retries/timeout types
        if getattr(n, "retries", None) is None:
            n.retries = 1
        if getattr(n, "timeout_s", None) is None:
            n.timeout_s = 30

        # ensure after-list shape
        if not isinstance(n.after, list):
            n.after = []

        valid.append(n)

    plan.nodes = valid
    return plan


_GOAL_FILE_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\-]+\\.(json|ya?ml|toml|ini|txt|md|py|js|ts|tsx|jsx|html|css|rs|go|java|c|cpp))",
    re.I
)

def _guess_target_file(goal: str) -> Optional[str]:
    m = _GOAL_FILE_RE.search(goal or "")
    if m:
        return m.group("path")
    return None

# ──────────────────────────────────────────────────────────────────────────────
# XII. Cellular-Automata ToolFoundry (optional synthesis of missing tools)
# ──────────────────────────────────────────────────────────────────────────────
CAPS = {
    "PURE_FUNC": {"allowed_imports": {"math","re","json","statistics"}, "io": False, "net": False},
    "NET_TOOL":  {"allowed_imports": {"httpx","json","re"},            "io": False, "net": True},
    "IO_TOOL":   {"allowed_imports": {"json","re"},                    "io": True,  "net": False},
}

PIP_STATE_PATH = DATA_DIR / "pip_state.json"
PIP_LOCK_PATH = DATA_DIR / "pip.lock"

def ensure_pip_installed(requirements: List[str], timeout_s: int = 300) -> Dict[str, Any]:
    import importlib
    def _top_module(req: str) -> str:
        base = re.split(r"[<>=!~\[]", req, 1)[0].strip()
        if not base:
            return ""
        mapping = {"beautifulsoup4":"bs4","opencv-python":"cv2","pillow":"PIL","pyyaml":"yaml","scikit-learn":"sklearn",
                   "tensorflow-cpu":"tensorflow","tensorflow":"tensorflow","torchvision":"torchvision","pydantic-core":"pydantic_core"}
        return mapping.get(base.lower(), base.replace("-", "_"))

    def _read_state() -> Dict[str, Any]:
        return jload(PIP_STATE_PATH, default={"installed": {}, "history": []}) or {"installed": {}, "history": []}

    def _write_state(state: Dict[str, Any]):
        try: jdump(PIP_STATE_PATH, state)
        except Exception: pass

    def _lock_acquire(max_wait_s: int = 180) -> bool:
        t0 = time.time()
        while True:
            try:
                fd = os.open(str(PIP_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd); return True
            except FileExistsError:
                if time.time() - t0 > max_wait_s: return False
                time.sleep(0.2)

    def _lock_release():
        try:
            if PIP_LOCK_PATH.exists(): PIP_LOCK_PATH.unlink()
        except Exception: pass

    reqs = [r for r in (requirements or []) if isinstance(r, str) and r.strip()]
    if not reqs:
        return {"ok": True, "installed": [], "skipped": [], "errors": []}

    state = _read_state()
    installed_ok, skipped, errors = [], [], []

    pre_need = []
    for r in reqs:
        mod = _top_module(r)
        try:
            if mod:
                importlib.import_module(mod)
                skipped.append(r)
                state["installed"].setdefault(r, {"ts": _now_iso(), "via": "pre-import"})
            else:
                pre_need.append(r)
        except Exception:
            pre_need.append(r)

    if not pre_need:
        _write_state(state)
        return {"ok": True, "installed": [], "skipped": skipped, "errors": []}

    if not _lock_acquire():
        return {"ok": False, "installed": installed_ok, "skipped": skipped,
                "errors": [{"req": "<lock>", "err": "pip install lock timeout"}]}

    try:
        for r in pre_need:
            cmd = [str(PY_EXE), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", r]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, check=False, text=True)
                if proc.returncode != 0:
                    errors.append({"req": r, "err": f"pip exit {proc.returncode}: {proc.stderr.strip()[:2000]}"})
                    continue
                installed_ok.append(r)
                state["installed"][r] = {"ts": _now_iso(), "via": "pip"}
                state["history"].append({"req": r, "ts": _now_iso(), "status": "ok"})
            except subprocess.TimeoutExpired:
                errors.append({"req": r, "err": "pip install timeout"})
            except Exception as e:
                errors.append({"req": r, "err": str(e)})
    finally:
        _lock_release(); _write_state(state)

    return {"ok": len(errors) == 0, "installed": installed_ok, "skipped": skipped, "errors": errors}

def _extract_top_imports(src: str) -> List[str]:
    tops: set = set()
    try:
        tree = ast.parse(src, mode="exec")
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    tops.add(a.name.split(".",1)[0])
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    tops.add(n.module.split(".",1)[0])
    except Exception:
        pass
    return sorted(tops)

def safe_ast_check(code: str, kind: str = "PURE_FUNC", extra_allow: Optional[Union[set, List[str]]] = None) -> Optional[str]:
    allowed = set(CAPS.get(kind, CAPS["PURE_FUNC"])["allowed_imports"])
    if extra_allow: allowed |= set(extra_allow)
    banned_names = {"os","sys","subprocess","socket","shutil","requests","pathlib","open","eval","exec","compile","__import__","importlib"}
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    class V(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                base = alias.name.split(".",1)[0]
                if base not in allowed:
                    raise ValueError(f"Disallowed import: {alias.name}")
        def visit_ImportFrom(self, node: ast.ImportFrom):
            base = (node.module or "").split(".",1)[0]
            if base not in allowed:
                raise ValueError(f"Disallowed import-from: {node.module}")
        def visit_Name(self, node: ast.Name):
            if node.id in banned_names:
                raise ValueError(f"Use of banned name: {node.id}")
        def visit_Attribute(self, node: ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith("__"):
                raise ValueError(f"Banned dunder attribute: {node.attr}")
            self.generic_visit(node)
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                raise ValueError("Use of __import__ is banned")
            self.generic_visit(node)
        def generic_visit(self, node):
            if isinstance(node, (ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Lambda,
                                 ast.Global, ast.Nonlocal, ast.Await, ast.Yield, ast.YieldFrom)):
                raise ValueError(f"Banned node type: {type(node).__name__}")
            return super().generic_visit(node)
    try:
        V().visit(tree)
    except Exception as e:
        return str(e)
    return None

class FoundryResult(BaseModel):
    ok: bool; name: str; code: Optional[str]=None
    tests: List[Dict[str, Any]]=Field(default_factory=list); reason: Optional[str]=None

def tool_foundry(goal: str, desired_name: str, hint_context: Dict[str, Any]) -> FoundryResult:
    # Minimal spec/impl/test loop (reuses earlier LLM system keys)
    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if "\n" in s:
                s = s.split("\n",1)[1]
        return s.strip()

    spec = {"name": desired_name, "doc": "", "sig": {"args": [], "returns": "Any"}, "kind": "PURE_FUNC"}
    code = None; tests: List[Dict[str, Any]] = []; last_err: Any = None
    for _ in range(4):
        if not spec["doc"]:
            SYSTEM = sysmsg("foundry_spec")
            USER = {"goal":goal, "desired_name":desired_name, "known_tools": list(REGISTRY.keys()),
                    "context_hint": hint_context, "recent_context": ctx_bus_tail(10)}
            try:
                content = _strip_code_fences(_llm(SYSTEM, USER, temp=0.0, label="FOUNDRY-SPEC"))
                s = json.loads(content); spec["doc"] = s.get("doc","").strip()
                spec["sig"]["args"] = s.get("args",[]); spec["sig"]["returns"] = s.get("returns","Any")
            except Exception as e:
                last_err = f"spec-fail {e}"

        IMPL_SYS = sysmsg("foundry_impl")
        IMPL_USER = {"name": spec["name"], "spec": spec, "critic_feedback": last_err, "context_hint": hint_context}
        try:
            code_block = _strip_code_fences(_llm(IMPL_SYS, IMPL_USER, temp=0.1, label="FOUNDRY-IMPL"))
            always_ok = {"json","re","math","statistics","httpx"}
            imports = [m for m in _extract_top_imports(code_block) if m not in always_ok]
            extra_allow = set()
            if imports:
                pip_res = ensure_pip_installed(imports)
                ok_names = []
                for r in pip_res.get("installed", []) + pip_res.get("skipped", []):
                    name = re.split(r"[<>=!~\[]", r, 1)[0].strip().lower()
                    ok_names.append(name.replace("-", "_"))
                extra_allow = set(ok_names)
            err = safe_ast_check(code_block, extra_allow=extra_allow)
            if err: last_err = f"ast-unsafe: {err}"
            else: code = textwrap.dedent(code_block).strip()
        except Exception as e:
            last_err = f"impl-fail {e}"

        TEST_SYS = sysmsg("foundry_test")
        TEST_USER = {"name": spec["name"], "spec": spec, "recent_error": last_err}
        try:
            content = _strip_code_fences(_llm(TEST_SYS, TEST_USER, temp=0.15, label="FOUNDRY-TEST"))
            cand_tests = json.loads(content)
            if isinstance(cand_tests, list) and cand_tests:
                tests = cand_tests
        except Exception as e:
            last_err = f"test-fail {e}"

        if code and tests:
            try:
                SAFE_BUILTINS = {"abs":abs,"min":min,"max":max,"sum":sum,"len":len,"range":range,"enumerate":enumerate,"sorted":sorted,
                                 "map":map,"filter":filter,"all":all,"any":any,"round":round,"int":int,"float":float,"bool":bool,"str":str,
                                 "list":list,"dict":dict,"set":set,"tuple":tuple}
                g = {"__builtins__": SAFE_BUILTINS, "re": re, "json": json}
                l: Dict[str, Any] = {}
                exec(code, g, l); fn = l.get(spec["name"])
                if not callable(fn): raise RuntimeError("function not defined correctly")
                failures = []
                for i, case in enumerate(tests):
                    try:
                        got = fn(**case.get("inputs",{}))
                        exp = case.get("expect")
                        if got != exp:
                            failures.append({"i":i,"got":got,"exp":exp})
                    except Exception as e:
                        failures.append({"i":i,"error":str(e)})
                if not failures: break
                last_err = {"failures": failures}
            except Exception as e:
                last_err = {"error": str(e)}
    if not code:
        return FoundryResult(ok=False, name=desired_name, reason=str(last_err), tests=tests or [])

    # Append to tools.py
    try:
        current = TOOLS_PATH.read_text()
        if not re.search(rf"def\s+{re.escape(spec['name'])}\s*\(", current):
            with TOOLS_PATH.open("a", encoding="utf-8") as f:
                f.write("\n\n# ---- Auto-added by ToolFoundry ----\n")
                f.write(code.rstrip() + "\n")
    except Exception as e:
        return FoundryResult(ok=False, name=desired_name, reason=f"write-failed: {e}", tests=tests or [], code=code)

    # Reload registry/docs
    TOOLS_MOD = import_tools_module()
    TOOLS = discover_tools(TOOLS_MOD)
    DOCS = ensure_docs(TOOLS, force=False)
    REGISTRY = build_tool_registry(TOOLS_MOD)
    TOOL_ALIASES.clear()
    for full in list(REGISTRY.keys()):
        if "." in full:
            simple = full.split(".")[-1]
            TOOL_ALIASES.setdefault(simple, full)
        else:
            TOOL_ALIASES.setdefault(full, full)
    return FoundryResult(ok=True, name=desired_name, code=code, tests=tests)

def ensure_tools_available(plan: Plan, registry: Dict[str, Callable[..., Any]], goal: str) -> None:
    missing = [n.tool for n in plan.nodes if resolve_tool_name(n.tool) not in registry]
    if not missing: return
    print(f"[foundry] Missing tools detected: {missing}")
    for tname in missing:
        hint = {"desired_name": tname, "nodes": [n.model_dump() for n in plan.nodes]}
        res = tool_foundry(goal, tname, hint_context=hint)
        if res.ok:
            print(f"[foundry] Created tool '{tname}' with {len(res.tests)} tests; appended to tools.py")
            ctx_bus_push("foundry_success", {"tool": tname})
        else:
            print(f"[foundry] Failed to create '{tname}': {res.reason}")
            ctx_bus_push("foundry_failure", {"tool": tname, "reason": res.reason})

# ──────────────────────────────────────────────────────────────────────────────
# XIII. Post-exec verification helpers (diffs, tests, auto-commit)
# ──────────────────────────────────────────────────────────────────────────────
def collect_git_diff() -> str:
    try:
        if "GitTools.diff" in REGISTRY:
            return REGISTRY["GitTools.diff"](root=".", staged=False) or ""
    except Exception:
        pass
    return ""

def maybe_run_pytests() -> Dict[str, Any]:
    if os.environ.get("ACI_RUN_PYTEST","0") != "1":
        return {"enabled": False}
    try:
        if REGISTRY["TestTools.has_pytest"]("."):
            res = REGISTRY["TestTools.run_pytest"](root=".", args=["-q"])
            return {"enabled": True, "result": res}
        return {"enabled": True, "result": {"ok": False, "error":"pytest not available"}}
    except Exception as e:
        return {"enabled": True, "result": {"ok": False, "error": str(e)}}

def maybe_autocommit(message: str, changed_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    if os.environ.get("ACI_GIT_AUTOCOMMIT","0") != "1":
        return {"enabled": False}
    try:
        if REGISTRY["GitTools.is_repo"]("."):
            return REGISTRY["GitTools.commit"](root=".", message=message, add_paths=changed_paths or None)
        return {"enabled": True, "ok": False, "error":"not a git repo"}
    except Exception as e:
        return {"enabled": True, "ok": False, "error": str(e)}

# Helpers (place with the other small utilities)
def _first_n_lines(s: str, n: int = 2) -> str:
    lines = [ln.strip() for ln in (s or "").splitlines() if ln.strip()]
    return "\n".join(lines[:n])

def _strip_meta(s: str) -> str:
    # Remove chatter about history/tools; keep the gist
    bad = re.compile(r"(conversation history|snippets|Code Analyzer|external dependencies|to be implemented)", re.I)
    out = []
    for ln in (s or "").splitlines():
        if not bad.search(ln):
            out.append(ln)
    return "\n".join(out)

def _tool_signatures_for_prompt(tools: List[ToolSpec]) -> str:
    lines = []
    for t in tools:
        params = []
        for a in t.args:
            mark = "" if a.required else "?"
            params.append(f"{a.name}{mark}:{a.type}")
        lines.append(f"- {t.name}({', '.join(params)}) -> {t.returns}")
    return "\n".join(lines)

# Replace clarify_goal with this tighter version
def clarify_goal(goal: str) -> str:
    SYSTEM = sysmsg("clarify", {"REALITY_POLICY": sysmsg("reality_policy")})
    # Strictly pass only the goal; no bus tail, no extra assistant preface
    user = {"goal": goal}
    attempt = 0
    last = ""
    while attempt < 2:
        attempt += 1
        try:
            r = _llm(SYSTEM, user, temp=0.1, preface=None, label=f"CLARIFY#{attempt}")
        except Exception:
            r = goal
        # post-filter any verbosity the model might add
        r = _strip_meta(r)
        r = _first_n_lines(r, 2)
        last = r or goal
        if last:
            break
    return last or goal

# ──────────────────────────────────────────────────────────────────────────────
# XIV. CLI ingress
# ──────────────────────────────────────────────────────────────────────────────
def _parse_explicit_tool_request(goal: str) -> Optional[str]:
    m = re.search(r"\bcreate\s+a\s+tool\s+(?:named\s+)?([A-Za-z_]\w*)\b", goal, re.I)
    if m: return m.group(1)
    m2 = re.search(r"\bcreate\s+a\s+tool\s+to\s+([A-Za-z0-9_\-\s]+)$", goal, re.I)
    if m2:
        base = re.sub(r"[^A-Za-z0-9]+", "_", m2.group(1).strip())
        if base and base[0].isdigit():
            base = "t_" + base
        return base.lower()
    return None

def interactive_loop():
    print("\nAgentic Ollama Runner — Graphy Repo-RAG + Workspace ACI")
    print("--------------------------------------------------------")
    print(f"Model: {CONFIG.model}  Host: {CONFIG.host}")
    print("Type a goal (e.g., 'rename Config class to AppConfig in config.py and update imports').\n")

    while True:
        try:
            goal = input("Goal> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye."); break
        if not goal:
            goal = "Search for 'TODO' in *.py, replace with 'NOTE', update one file span, and show diff."
        ctx_bus_push("goal", {"text": goal})
        trace("goal", {"text": goal})

        # Explicit tool creation (lets the LLM synthesize a new tool)
        explicit = _parse_explicit_tool_request(goal)
        if explicit and resolve_tool_name(explicit) not in REGISTRY:
            print(f"[foundry] Explicit tool request detected: {explicit}")
            res = tool_foundry(goal, explicit, {"requested_by":"explicit"})
            if res.ok:
                print(f"[foundry] Created '{explicit}'.")
            else:
                print(f"[foundry] Could not create '{explicit}': {res.reason}")

        clarified = clarify_goal(goal); print("\n[clarify]\n" + clarified + "\n")
        trace("clarify", {"text": clarified})

        # Build Dynamic Route Context
        route_ctx = build_route_context(clarified)
        trace("route_context", route_ctx)

        # Initial plan
        plan = plan_with_llm(clarified, TOOLS, DOCS, route_ctx)
        plan = validate_and_fix_placeholders(plan)
        trace("plan_initial", plan.model_dump())

        # Adversarial refinement
        plan = red_team_refine(clarified, plan, TOOLS, route_ctx)
        plan = validate_and_fix_placeholders(plan)
        trace("plan_refined", plan.model_dump())

        ensure_tools_available(plan, REGISTRY, clarified)
        print("[plan]\n" + json.dumps(plan.model_dump(), indent=2))

        # Execute
        recs, recmap = execute_plan(plan, REGISTRY)
        trace("execute_done", {"ok": all(r.error is None for r in recs)})

        # reflect/repair if needed
        if any(r.error for r in recs):
            repaired = reflect_repair(clarified, TOOLS, DOCS, recs, plan, route_ctx)
            if repaired:
                repaired = validate_and_fix_placeholders(repaired)
                ensure_tools_available(repaired, REGISTRY, clarified)
                print("[reflect] Executing repaired plan ...")
                trace("plan_repaired", repaired.model_dump())
                recs2, recmap2 = execute_plan(repaired, REGISTRY)
                recmap.update(recmap2)

        # Gather diff and verifications
        diff_text = collect_git_diff()
        pytest_result = maybe_run_pytests()
        verifs = {
            "diff_present": bool(diff_text.strip()),
            "pytest": pytest_result,
            "before_snap": "",  # kept for future capture (can be filled by read_span nodes if added)
            "after_snap": "",
            "records": {k: v.model_dump() for k,v in recmap.items()}
        }
        trace("verification", {"has_diff": verifs["diff_present"], "pytest_ok": pytest_result.get("result",{}).get("ok") if pytest_result.get("enabled") else None})

        # QA review + optional auto-commit
        qa = qa_review(clarified, diff_text, verifs)
        trace("qa", qa)
        print(f"\n[qa] pass={qa.get('pass')} score={qa.get('score')} risk={qa.get('risk')}")
        if qa.get("reasons"):
            print("     reasons:", "; ".join(qa["reasons"])[:200])

        if qa.get("pass"):
            cm = f"agent: {clarified[:72]}… | score={qa.get('score')}"
            commit_res = maybe_autocommit(cm)
            if commit_res.get("enabled", True):
                print("[git]", "autocommit:", commit_res)

        final = assemble_final(clarified, clarified, recmap)
        print("\n[final]\n" + final + "\n")
        ctx_bus_push("final", {"goal": clarified, "text": final})
        trace("final", {"text": final})

# ──────────────────────────────────────────────────────────────────────────────
# XV. entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        interactive_loop()
    except Exception as e:
        print("[fatal] Unhandled exception:", e)
        traceback.print_exc()

