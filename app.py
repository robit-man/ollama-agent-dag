#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Agentic DAG + Cellular-Automata ToolFoundry on Ollama
Reality-first, streaming, self-correcting, self-updating via ToolFoundry.
System prompts are unified in systems.json (auto-seeded; editable between runs).

Run:
  ollama serve
  python3 app.py

Optional env:
  AGENT_DEBUG=1         # print compact prompt transcripts
  AGENT_INTERLEAVE=1    # add a short assistant-preface before STRICT-JSON asks
  AGENT_STREAM=1        # stream LLM tokens live (default 1)
  AGENT_COLORS=1        # enable ANSI color output (default 1)
"""

# ──────────────────────────────────────────────────────────────────────────────
# I. venv bootstrap (stdlib only)
# ──────────────────────────────────────────────────────────────────────────────
import os, sys, subprocess, json, time, textwrap, inspect, importlib.util, types, traceback, re, ast
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / "venv"
PY_EXE = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
PIP_EXE = [str(PY_EXE), "-m", "pip"]
DEPS = [
    "ollama>=0.5.0",
    "pydantic>=2.7",
    "httpx>=0.27",
    # Added deps for web + scraping + browser
    "beautifulsoup4>=4.12",
    "html5lib>=1.1",
    "selenium>=4.21",
    "webdriver-manager>=4.0.2",
    "requests>=2.31",
]

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
# II. imports (post-venv) + color helpers
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_type_hints
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

AGENT_STREAM = os.environ.get("AGENT_STREAM", "1") == "1"
AGENT_COLORS = os.environ.get("AGENT_COLORS", "1") == "1"

# Minimal ANSI coloring
class C:
    R = "\033[31m" if AGENT_COLORS else ""
    G = "\033[32m" if AGENT_COLORS else ""
    Y = "\033[33m" if AGENT_COLORS else ""
    B = "\033[34m" if AGENT_COLORS else ""
    M = "\033[35m" if AGENT_COLORS else ""
    C = "\033[36m" if AGENT_COLORS else ""
    W = "\033[37m" if AGENT_COLORS else ""
    DIM = "\033[2m" if AGENT_COLORS else ""
    UND = "\033[4m" if AGENT_COLORS else ""
    RST = "\033[0m" if AGENT_COLORS else ""

def _c(tag: str, text: str) -> str:
    m = {
        "SYS": C.M + "SYS" + C.RST,
        "USR": C.C + "USR" + C.RST,
        "AST": C.G + "AST" + C.RST,
        "DBG": C.Y + "DBG" + C.RST,
        "ERR": C.R + "ERR" + C.RST,
        "OK":  C.G + "OK"  + C.RST,
        "RUN": C.B + "RUN" + C.RST,
        "TOOL": C.Y + "TOOL" + C.RST,
        "PLAN": C.B + "PLAN" + C.RST,
        "NODE": C.M + "NODE" + C.RST,
    }
    if tag in m:
        return f"[{m[tag]}] {text}"
    return text

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

for d in (DATA_DIR, FOUNDRY_LOG_DIR):
    d.mkdir(exist_ok=True)

DEBUG = os.environ.get("AGENT_DEBUG","0") == "1"
INTERLEAVE = os.environ.get("AGENT_INTERLEAVE","0") == "1"

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
    print(_c("DBG", f"--- DEBUG:{label} ---"))
    for i,m in enumerate(messages,1):
        head = (m.get("content","")[:400] + "…") if len(m.get("content",""))>400 else m.get("content","")
        print(f"{C.DIM}{i:02d} {m.get('role')} :: {head}{C.RST}")

def _emit_chat_preview(system: str, user_obj: Any, preface: Optional[str], label: str):
    sys_head = system.strip().splitlines()[0][:80]
    print(_c("SYS", f"{label} system: {sys_head!r}"))
    if preface:
        pre = preface.strip().replace("\n", " ")[:120]
        print(_c("AST", f"{label} interleave: {pre}"))
    js = json.dumps(user_obj, ensure_ascii=False)[:160]
    print(_c("USR", f"{label} user: {js} ..."))

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
    "Reformulate the user's goal into 1–2 sentences and list constraints as bullets.\n"
    "Call out any external data, credentials, or hardware dependencies.\n\n"
    "{{REALITY_POLICY}}"
  ),

  "plan": (
    "You are a planning agent.\n"
    "Output ONLY JSON:\n"
    "{ 'nodes': [ { 'id': str, 'tool': str, 'args': object, 'after': [str], 'retries': int?, 'timeout_s': int? } ],\n"
    "  'meta': { 'parallelism': int?, 'timeout_s': int?, 'notes': str? } }\n"
    "Rules:\n"
    "- Choose tool from THIS exact list (use exact string): {{AVAILABLE_TOOLS}}\n"
    "- If no tool fits, NAME a new tool; a separate agent will implement it.\n"
    "- Wire dependencies using bracket placeholders like [nodeId.output.key] (NO angle brackets).\n"
    "- Break the problem into as many nodes as necessary (up to 50) to ensure real, verifiable results.\n"
    "- Include validation or post-check nodes when appropriate.\n"
    "- No prose outside the JSON.\n"
    "{{REALITY_POLICY_SECTION}}"
  ),

  "red_team": (
    "You are an adversarial planner ('red team').\n"
    "Given a goal and an initial plan, identify missing prerequisites, failure modes, validation steps, safety checks,\n"
    "and any observability needed. Propose a PATCH as STRICT JSON only:\n"
    "{ 'add': [ nodes... ], 'modify': [ {'id': str, 'tool'?: str, 'args'?: object, 'after'?: [str], 'retries'?: int, 'timeout_s'?: int} ],\n"
    "  'remove': [str], 'notes': [str] }\n"
    "- Use tools from the available list; if missing, NAME new tools generically.\n"
    "- Keep it minimal but sufficient; no prose outside JSON.\n"
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

  # NEW: directional auxiliary summary directive
  "aux_directive": (
    "You are a focused, evidence-grounded summarizer.\n"
    "Rules:\n"
    "- Use only the supplied page text; do not invent facts.\n"
    "- Prioritize what directly answers the user's topic.\n"
    "- 2–3 crisp sentences, plain language, no marketing fluff.\n"
    "- If the content is thin, say so briefly.\n"
  ),
}

def _seed_or_load_systems() -> Dict[str, str]:
    existing = jload(SYSTEMS_PATH, default=None)
    if existing is None:
        jdump(SYSTEMS_PATH, DEFAULT_SYSTEMS)
        return DEFAULT_SYSTEMS
    merged = dict(DEFAULT_SYSTEMS); merged.update(existing)
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
# V. default tools.py (created if missing — stdlib + safe libs; Foundry appends)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_TOOLS = '''\
"""
tools.py — Built-in tools for the agentic DAG.
Deterministic helpers + web/search/selenium. ToolFoundry appends new ones below.

Notes:
- Indentation is spaces-only to avoid TabError.
- Browser helpers are designed for agentic N-step loops:
    1) act (navigate/click/input/press/scroll/eval_js)
    2) observe (query/exists/count/observe/dom/text/attrs/screenshot/url)
    3) decide next action based on the observation
- Tools.input accepts **kwargs and aliases "prompt"→text for backward compatibility.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json, re, statistics, os, sys, csv, time, platform, shutil, subprocess, tempfile, uuid, html, io, glob, traceback
from datetime import datetime, UTC
from urllib import request, error as urlerror
import httpx

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
WORKSPACE_DIR = DATA_DIR / "workspace"
RUNS_DIR = DATA_DIR / "runs"
EVENTS_PATH = DATA_DIR / "events.jsonl"
MEMORY_PATH = DATA_DIR / "memory_kv.json"
DOCS_PATH = ROOT_DIR / "docs.json"
BUS_PATH = DATA_DIR / "context_bus.jsonl"
CONFIG_PATH = ROOT_DIR / "config.json"
SYSTEMS_PATH = ROOT_DIR / "systems.json"

for d in (DATA_DIR, WORKSPACE_DIR, RUNS_DIR):
    d.mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()

def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+","-", s.strip().lower()).strip("-")
    return s or "n-a"

def _safe_join(base: Path, p: Union[str, Path]) -> Path:
    pth = (base / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
    base_res = base.resolve()
    if str(pth).startswith(str(base_res)):
        return pth
    return base / _slug(str(p))

def _is_hidden(path: Path) -> bool:
    try:
        return any(part.startswith(".") for part in Path(path).parts)
    except Exception:
        return Path(path).name.startswith(".")

def _read_jsonl_tail(path: Path, n: int) -> List[Dict[str, Any]]:
    if not path.exists(): return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return [json.loads(x) for x in lines[-n:]]
    except Exception:
        return []

def log_event(level: str, msg: str, data: Optional[Dict[str, Any]]=None) -> str:
    rec = {"ts": _now_iso(), "level": level, "msg": msg, "data": data or {}}
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
    return str(EVENTS_PATH)

def log_message(message: str, level: str = "INFO"):
    try:
        print(f"[{level}] {message}")
    finally:
        try:
            log_event(level, message, None)
        except Exception:
            pass

# ── Basic math/string utilities ──────────────────────────────────────────────
def add(a: float, b: float) -> float:
    "Add two numbers and return the sum."
    return float(a) + float(b)

def mean(values: List[float]) -> float:
    "Compute the arithmetic mean of a list of numbers."
    if not values:
        raise ValueError("values must be non-empty")
    return float(statistics.fmean(values))

def slugify(text: str) -> str:
    "Make a filesystem-safe slug from text (lowercase, dashes, alnum-only)."
    return _slug(text)

def word_count(text: str) -> Dict[str, int]:
    "Return token counts: words, lines, characters."
    lines = text.splitlines()
    words = [w for w in re.split(r"\\s+", text.strip()) if w]
    return {"words": len(words), "lines": len(lines), "chars": len(text)}

# ── Filesystem context & IO ───────────────────────────────────────────────────
def dir_list(path: str = ".", recursive: bool=False, pattern: str="*", files_only: bool=False, include_stats: bool=True) -> List[Dict[str, Any]]:
    """
    List files/dirs under 'path'. Writes nothing. Returns list of dicts:
    {name, path, is_dir, size?, mtime?}
    """
    root = Path(path).resolve()
    if not root.exists(): return []
    items = []
    it = root.rglob(pattern) if recursive else root.glob(pattern)
    for p in it:
        if files_only and p.is_dir():
            continue
        try:
            d = {"name": p.name, "path": str(p), "is_dir": p.is_dir()}
            if include_stats:
                st = p.stat()
                d["size"] = int(st.st_size)
                d["mtime"] = int(st.st_mtime)
            items.append(d)
        except Exception:
            pass
    return items

def file_read(path: str, max_bytes: int = 131072, encoding: str="utf-8") -> str:
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        return ""
    data = p.read_bytes()[:max_bytes]
    try:
        return data.decode(encoding, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")

def file_write(path: str, text: str, encoding: str="utf-8", make_dirs: bool=True) -> str:
    p = _safe_join(WORKSPACE_DIR, path)
    if make_dirs: p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding=encoding)
    return str(p)

def file_append(path: str, text: str, encoding: str="utf-8") -> str:
    p = _safe_join(WORKSPACE_DIR, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding=encoding) as f:
        f.write(text)
    return str(p)

def json_read(path: str) -> Any:
    p = Path(path).resolve()
    if not p.exists(): return None
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return None

def json_write(path: str, obj: Any) -> str:
    p = _safe_join(WORKSPACE_DIR, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return str(p)

def csv_read(path: str, max_rows: int = 5000) -> List[List[str]]:
    p = Path(path).resolve()
    if not p.exists(): return []
    out: List[List[str]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            out.append(row)
            if i+1 >= max_rows: break
    return out

def csv_write(path: str, rows: List[List[str]]) -> str:
    p = _safe_join(WORKSPACE_DIR, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)
    return str(p)

def grep(regex: str, root: str=".", file_glob: str="**/*", max_matches: int = 500, ignore_hidden: bool=True) -> List[Dict[str, Any]]:
    """
    Grep-like search. Returns [{path, lineno, line}] up to max_matches.
    """
    pat = re.compile(regex)
    base = Path(root).resolve()
    results = []
    for path in base.glob(file_glob):
        if path.is_dir(): continue
        if ignore_hidden and _is_hidden(path): continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if pat.search(line):
                        results.append({"path": str(path), "lineno": i, "line": line.rstrip("\\n")})
                        if len(results) >= max_matches: return results
        except Exception:
            pass
    return results

def extract_frontmatter_md(path: str) -> Dict[str, Any]:
    """
    Extract simple YAML-like frontmatter from Markdown file.
    """
    text = file_read(path, max_bytes=2_000_000) or ""
    if not text.startswith("---"): return {"frontmatter": {}, "body": text}
    parts = text.split("---", 2)
    if len(parts) < 3: return {"frontmatter": {}, "body": text}
    fm_raw, body = parts[1], parts[2]
    fm = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return {"frontmatter": fm, "body": body.lstrip()}

# ── Planning artifacts & memory ───────────────────────────────────────────────
def plan_doc_create(name: str, bullets: List[str]) -> str:
    fname = _slug(name) + ".md"
    p = WORKSPACE_DIR / "plans" / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    content = "# " + name + "\\n\\n" + "\\n".join([f"- {b}" for b in bullets]) + "\\n"
    p.write_text(content, encoding="utf-8")
    return str(p)

def memory_kv_get(key: str, default: Any=None) -> Any:
    if not MEMORY_PATH.exists(): return default
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        return data.get(key, default)
    except Exception:
        return default

def memory_kv_set(key: str, value: Any) -> bool:
    data = {}
    if MEMORY_PATH.exists():
        try: data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except Exception: data = {}
    data[key] = value
    MEMORY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return True

def ctx_bus_tail_tool(n: int = 100) -> List[Dict[str, Any]]:
    return _read_jsonl_tail(BUS_PATH, n)

# ── Tooling introspection (via docs.json generated by app) ────────────────────
def tool_list() -> List[str]:
    if not DOCS_PATH.exists(): return []
    try:
        d = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
        return sorted(list(d.keys()))
    except Exception:
        return []

def tool_docs(name: str) -> Dict[str, Any]:
    if not DOCS_PATH.exists(): return {}
    try:
        d = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
        return d.get(name, {})
    except Exception:
        return {}

# ── Project discovery & Git awareness ────────────────────────────────────────
def project_discover(root: str = ".") -> Dict[str, Any]:
    r = Path(root).resolve()
    files = {p.name: str(p) for p in r.glob("*") if p.is_file()}
    hints = {
        "has_pyproject": "pyproject.toml" in files,
        "has_requirements": "requirements.txt" in files,
        "has_package_json": "package.json" in files,
        "has_dockerfile": "Dockerfile" in files,
        "has_readme": any(n.lower().startswith("readme") for n in files),
    }
    langs = []
    for ext, lang in {".py":"python",".js":"js",".ts":"ts",".md":"md",".json":"json",".sh":"shell",".ipynb":"ipynb"}.items():
        if any(str(p).endswith(ext) for p in r.rglob(f"*{ext}")):
            langs.append(lang)
    gi = {}
    try:
        proc = subprocess.run(["git","rev-parse","--abbrev-ref","HEAD"], cwd=str(r), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if proc.returncode == 0:
            gi["branch"] = proc.stdout.strip()
        proc = subprocess.run(["git","rev-parse","HEAD"], cwd=str(r), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if proc.returncode == 0:
            gi["commit"] = proc.stdout.strip()
        proc = subprocess.run(["git","status","--porcelain"], cwd=str(r), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if proc.returncode == 0:
            gi["is_dirty"] = bool(proc.stdout.strip())
    except Exception:
        pass
    return {"root": str(r), "hints": hints, "languages": sorted(set(langs)), "git": gi}

def git_info() -> Dict[str, Any]:
    return project_discover(".").get("git", {})

def git_changed_files(root: str = ".") -> List[str]:
    try:
        proc = subprocess.run(["git","status","--porcelain"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
        if proc.returncode == 0:
            files = []
            for line in proc.stdout.splitlines():
                part = line[3:].strip()
                if part: files.append(part)
            return files
    except Exception:
        pass
    return []

# ── Execution: run Python with logs ───────────────────────────────────────────
def python_run_file(path: str, args: List[str]=None, timeout_s: int = 120) -> Dict[str, Any]:
    """
    Run a Python file using current interpreter; capture stdout/stderr to RUNS_DIR log.
    """
    args = args or []
    ts = int(time.time()*1000)
    slug = _slug(Path(path).name)
    log_path = RUNS_DIR / f"{ts}_{slug}.log"
    cmd = [sys.executable, "-u", path] + list(args)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_s)
        log_path.write_text(f"$ {' '.join(cmd)}\\n\\n[stdout]\\n{proc.stdout}\\n\\n[stderr]\\n{proc.stderr}", encoding="utf-8")
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "elapsed_ms": int((time.time()-t0)*1000), "log_path": str(log_path)}
    except subprocess.TimeoutExpired as e:
        log_path.write_text(f"$ {' '.join(cmd)}\\n\\n[TIMEOUT after {timeout_s}s]\\n", encoding="utf-8")
        return {"returncode": -1, "stdout": "", "stderr": f"timeout after {timeout_s}s", "elapsed_ms": int((time.time()-t0)*1000), "log_path": str(log_path)}

def python_run(code: str, timeout_s: int = 60) -> Dict[str, Any]:
    """
    Write ephemeral code to WORKSPACE and execute via python_run_file.
    """
    scratch = WORKSPACE_DIR / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    ts = int(time.time()*1000)
    path = scratch / f"snippet_{ts}.py"
    path.write_text(code, encoding="utf-8")
    return python_run_file(str(path), [], timeout_s)

# ── Environment & system context ──────────────────────────────────────────────
def env_list() -> Dict[str, str]:
    keys = ["PATH","PYTHONPATH","VIRTUAL_ENV","HOME","USER","SHELL","COMSPEC","TEMP","TMP","NUMBER_OF_PROCESSORS"]
    out = {}
    for k in keys:
        v = os.environ.get(k)
        if v:
            if k in {"PATH","PYTHONPATH"} and len(v) > 300:
                out[k] = v[:300] + "..."
            else:
                out[k] = v
    return out

def workspace_info() -> Dict[str, Any]:
    total, used, free = shutil.disk_usage(str(ROOT_DIR))
    return {
        "root": str(ROOT_DIR),
        "workspace": str(WORKSPACE_DIR),
        "runs_dir": str(RUNS_DIR),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disk_gb": {"total": round(total/1e9,2), "used": round(used/1e9,2), "free": round(free/1e9,2)},
    }

def system_specs() -> Dict[str, Any]:
    info = workspace_info()
    mem_total = None
    try:
        if hasattr(os, "sysconf"):
            if "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
                mem_total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        pass
    info["memory_total_bytes"] = mem_total
    info["machine"] = platform.machine()
    info["processor"] = platform.processor()
    info["node"] = platform.node()
    return info

# ── Networking (urllib stdlib) ────────────────────────────────────────────────
def http_get(url: str, headers: Optional[Dict[str,str]] = None, timeout_s: int = 20, parse_json: bool=True) -> Dict[str, Any]:
    "Simple HTTP GET using stdlib. Optionally parse JSON by content-type or body shape."
    req = request.Request(url, headers=headers or {})
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            ct = resp.headers.get("Content-Type","")
            out = {"status": resp.status, "headers": dict(resp.headers)}
            if parse_json and ("json" in ct or body.strip().startswith(b"{") or body.strip().startsWith(b"[")):
                try:
                    out["json"] = json.loads(body.decode("utf-8", errors="replace"))
                except Exception:
                    out["text"] = body.decode("utf-8", errors="replace")
            else:
                out["text"] = body.decode("utf-8", errors="replace")
            return out
    except urlerror.HTTPError as e:
        return {"status": e.code, "headers": dict(e.headers or {}), "text": (e.read() or b"").decode("utf-8", errors="replace")}
    except Exception as e:
        return {"status": 0, "error": str(e)}

def http_post_json(url: str, data: Any, headers: Optional[Dict[str,str]] = None, timeout_s: int = 20) -> Dict[str, Any]:
    "Simple HTTP POST with JSON body using stdlib."
    payload = json.dumps(data).encode("utf-8")
    base_headers = {"Content-Type":"application/json"}
    if headers: base_headers.update(headers)
    req = request.Request(url, data=payload, headers=base_headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
            out = {"status": resp.status, "headers": dict(resp.headers)}
            try:
                out["json"] = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                out["text"] = body.decode("utf-8", errors="replace")
            return out
    except urlerror.HTTPError as e:
        return {"status": e.code, "headers": dict(e.headers or {}), "text": (e.read() or b"").decode("utf-8", errors="replace")}
    except Exception as e:
        return {"status": 0, "error": str(e)}

def web_read_text(url: str, timeout_s: int = 20) -> Dict[str, Any]:
    "Fetch URL and return {'status','headers','title','text'} with HTML stripped to plain text (best-effort)."
    out = http_get(url, timeout_s=timeout_s, parse_json=False)
    txt = out.get("text","")
    title = None
    if txt:
        m = re.search(r"<title[^>]*>(.*?)</title>", txt, re.I|re.S)
        if m:
            title = html.unescape(m.group(1)).strip()
        body = re.sub(r"(?s)<script.*?>.*?</script>", " ", txt)
        body = re.sub(r"(?s)<style.*?>.*?</style>", " ", body)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\\s+", " ", body).strip()
        out["text"] = body[:200000]
    if title: out["title"] = title
    return out

# ── Minimal BeautifulSoup scrape helper (used by search) ─────────────────────
def bs4_scrape(url: str, timeout_s: int = 10, full_html: bool = False) -> str:
    """
    Quick scrape via 'requests' and BeautifulSoup.
    Returns plain text by default; set full_html=True to return raw HTML.
    """
    import requests
    from bs4 import BeautifulSoup
    headers = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout_s)
        r.raise_for_status()
    except Exception as e:
        log_message(f"[bs4_scrape] error: {e}", "ERROR")
        return ""
    if full_html:
        return r.text
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script","style","noscript","form","header","footer","nav","aside"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

# ── Browser, search & scraping tools (agentic-ready) ─────────────────────────
class Tools:
    """
    High-level browser + scraping helpers designed for agentic, iterative control.
    Each method is idempotent where possible and returns concise, JSON-serializable data.
    """
    _driver = None  # Selenium WebDriver singleton

    # ── Driver lifecycle ──────────────────────────────────────────────────────
    @staticmethod
    def _find_system_chromedriver() -> Optional[str]:
        "Return a usable chromedriver path if one is available on the system."
        candidates: List[Optional[str]] = [
            shutil.which("chromedriver"),
            "/usr/bin/chromedriver",
            "/usr/local/bin/chromedriver",
            "/snap/bin/chromium.chromedriver",
            "/usr/lib/chromium-browser/chromedriver",
            "/opt/homebrew/bin/chromedriver",
        ]
        for path in filter(None, candidates):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                try:
                    subprocess.run([path, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return path
                except Exception:
                    continue
        return None

    @staticmethod
    def open_browser(headless: bool = False, force_new: bool = False, window_size: str = "1920,1080") -> str:
        """
        Launch a Chrome/Chromium browser with resilient fallbacks.
        - headless: run headless mode if True (default False for better compatibility)
        - force_new: close any existing singleton before launching
        - window_size: 'WIDTH,HEIGHT' string
        Returns a human-readable status string.
        """
        from selenium import webdriver
        from selenium.common.exceptions import WebDriverException
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        try:
            from webdriver_manager.chrome import ChromeDriverManager
        except Exception:
            ChromeDriverManager = None  # optional

        if force_new and Tools._driver:
            try:
                Tools._driver.quit()
            except Exception:
                pass
            Tools._driver = None
        if Tools._driver:
            return "Browser already open"

        chrome_bin = (
            os.getenv("CHROME_BIN")
            or shutil.which("google-chrome")
            or shutil.which("chromium-browser")
            or shutil.which("chromium")
            or "/snap/bin/chromium"
            or "/usr/bin/chromium-browser"
            or "/usr/bin/chromium"
        )

        opts = Options()
        if chrome_bin:
            opts.binary_location = chrome_bin
        if headless:
            # modern headless for Chrome 109+
            opts.add_argument("--headless=new")
        opts.add_argument(f"--window-size={window_size}")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--remote-allow-origins=*")
        opts.add_argument("--disable-infobars")
        opts.add_argument("--disable-blink-features=AutomationControlled")

        # Try Selenium-Manager first (bundled with Selenium 4.6+)
        try:
            log_message("[open_browser] Trying Selenium-Manager…", "DEBUG")
            Tools._driver = webdriver.Chrome(options=opts)
            Tools._driver.set_page_load_timeout(20)
            Tools._driver.implicitly_wait(0)  # agentic flows rely on explicit waits
            log_message("[open_browser] Launched via Selenium-Manager.", "SUCCESS")
            return "Browser launched (selenium-manager)"
        except WebDriverException as e:
            log_message(f"[open_browser] Selenium-Manager failed: {e}", "WARNING")

        # Try system chromedriver
        try:
            sys_drv = Tools._find_system_chromedriver()
            if sys_drv:
                Tools._driver = webdriver.Chrome(service=Service(sys_drv), options=opts)
                Tools._driver.set_page_load_timeout(20)
                Tools._driver.implicitly_wait(0)
                log_message("[open_browser] Launched via system chromedriver.", "SUCCESS")
                return "Browser launched (system chromedriver)"
        except WebDriverException as e:
            log_message(f"[open_browser] System chromedriver failed: {e}", "WARNING")

        # Try webdriver-manager if available and arch supports it
        if ChromeDriverManager is not None and platform.machine().lower() in ("x86_64","amd64","arm64","aarch64"):
            try:
                drv_path = ChromeDriverManager().install()
                Tools._driver = webdriver.Chrome(service=Service(drv_path), options=opts)
                Tools._driver.set_page_load_timeout(20)
                Tools._driver.implicitly_wait(0)
                log_message("[open_browser] Launched via webdriver-manager.", "SUCCESS")
                return "Browser launched (webdriver-manager)"
            except Exception as e:
                log_message(f"[open_browser] webdriver-manager failed: {e}", "ERROR")

        raise RuntimeError("No usable chromedriver. Install Chrome/Chromium or set CHROME_BIN, or place chromedriver on PATH.")

    @staticmethod
    def close_browser() -> str:
        "Close and cleanup the singleton driver."
        if Tools._driver:
            try:
                Tools._driver.quit()
                log_message("[close_browser] Browser closed.", "DEBUG")
            except Exception:
                pass
            Tools._driver = None
            return "Browser closed"
        return "No browser to close"

    # ── Navigation + waits ────────────────────────────────────────────────────
    @staticmethod
    def navigate(url: str, wait_complete: bool = True, timeout: int = 15) -> str:
        """
        Navigate current tab to URL.
        - wait_complete: wait for document.readyState==='complete'
        - timeout: seconds to wait for readiness
        """
        if not Tools._driver:
            return "Error: browser not open"
        drv = Tools._driver
        log_message(f"[navigate] → {url}", "DEBUG")
        drv.get(url)
        if wait_complete:
            end = time.time() + timeout
            while time.time() < end:
                try:
                    state = drv.execute_script("return document.readyState")
                    if state == "complete":
                        break
                except Exception:
                    pass
                time.sleep(0.1)
        return f"Navigated to {url}"

    @staticmethod
    def wait_for(selector: str, timeout: int = 8, state: str = "visible") -> str:
        """
        Wait for an element to reach a desired state.
        state ∈ {'visible','clickable','present','hidden','gone'}
        """
        if not Tools._driver:
            return "Error: browser not open"
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        drv = Tools._driver
        wait = WebDriverWait(drv, timeout, poll_frequency=0.1)

        try:
            if state == "clickable":
                wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            elif state == "visible":
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
            elif state == "present":
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            elif state == "hidden":
                wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, selector)))
            elif state == "gone":
                wait.until_not(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            else:
                return f"Error: unknown state {state!r}"
            return f"wait_for: {selector} is {state}"
        except Exception as e:
            return f"Error waiting for {selector} to be {state}: {e}"

    @staticmethod
    def current_url() -> str:
        "Return the current page URL."
        if not Tools._driver:
            return "Error: browser not open"
        return Tools._driver.current_url

    @staticmethod
    def go_back(wait_complete: bool = True, timeout: int = 10) -> str:
        "Navigate back in history."
        if not Tools._driver:
            return "Error: browser not open"
        drv = Tools._driver
        drv.back()
        if wait_complete:
            end = time.time() + timeout
            while time.time() < end:
                try:
                    if drv.execute_script("return document.readyState") == "complete":
                        break
                except Exception:
                    pass
                time.sleep(0.1)
        return "Went back"

    # ── Element interactions ──────────────────────────────────────────────────
    @staticmethod
    def click(selector: str, timeout: int = 8, focus: bool = True, scroll: bool = True) -> str:
        """
        Click the first element that matches selector.
        - focus: move focus before clicking
        - scroll: scroll element into view before clicking
        """
        if not Tools._driver:
            return "Error: browser not open"
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        try:
            drv = Tools._driver
            el = WebDriverWait(drv, timeout, poll_frequency=0.1).until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            if scroll:
                drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            if focus:
                drv.execute_script("arguments[0].focus({preventScroll:true});", el)
            el.click()
            return f"Clicked {selector}"
        except Exception as e:
            log_message(f"[click] Error clicking {selector}: {e}", "ERROR")
            return f"Error clicking {selector}: {e}"

    @staticmethod
    def input(selector: str = "", text: Optional[str] = None, timeout: int = 8, submit: bool = False, clear: bool = True, **kwargs) -> str:
        """
        Type into an input/textarea/contenteditable.
        - selector: CSS selector; if blank, use active element or first text-like input
        - text: text to send (alias: pass 'prompt' in kwargs for backwards compatibility)
        - submit: send Enter/Return at end if True
        - clear: attempt to clear existing value first
        """
        if not Tools._driver:
            return "Error: browser not open"
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.keys import Keys

        drv = Tools._driver
        if text is None:
            text = kwargs.get("prompt")
        if text is None:
            return "Error: no text provided"

        try:
            # choose element: explicit selector → active element → first typical input
            if selector:
                el = WebDriverWait(drv, timeout, poll_frequency=0.1).until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            else:
                try:
                    el = drv.switch_to.active_element
                except Exception:
                    el = None
                if not el:
                    cands = drv.find_elements(By.CSS_SELECTOR, "input[type='text'], input:not([type]), textarea, [contenteditable=''], [contenteditable='true']")
                    if not cands:
                        return "Error: no input element found"
                    el = cands[0]

            drv.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            drv.execute_script("arguments[0].focus({preventScroll:true});", el)
            if clear:
                try:
                    el.clear()
                except Exception:
                    # fallback: select-all + delete
                    el.send_keys(Keys.CONTROL, "a"); el.send_keys(Keys.DELETE)
            el.send_keys(str(text))
            if submit:
                el.send_keys(Keys.RETURN)
            return f"Sent {text!r} to {selector or '[active/input]'}"
        except Exception as e:
            log_message(f"[input] Error typing into {selector or '[auto]'}: {e}", "ERROR")
            return f"Error typing into {selector or '[auto]'}: {e}"

    @staticmethod
    def press_keys(keys: str) -> str:
        """
        Send key chords to the active element or <body>.
        Example keys: 'CTRL+A', 'ESC', 'ENTER', 'TAB', 'SHIFT+TAB'
        """
        if not Tools._driver:
            return "Error: browser not open"
        from selenium.webdriver.common.keys import Keys
        drv = Tools._driver
        mapping = {
            "CTRL":"CONTROL", "CONTROL":"CONTROL",
            "ALT":"ALT", "SHIFT":"SHIFT",
            "ENTER":"ENTER", "RETURN":"RETURN", "ESC":"ESCAPE",
            "ESCAPE":"ESCAPE", "TAB":"TAB", "BACKSPACE":"BACKSPACE",
        }
        seq = []
        for token in keys.replace("+"," + ").split():
            name = token.strip().upper()
            if name == "+": continue
            key = getattr(Keys, mapping.get(name, name), None)
            if key is None:
                seq.append(token)  # literal text
            else:
                seq.append(key)
        try:
            target = None
            try:
                target = drv.switch_to.active_element
            except Exception:
                pass
            if not target:
                target = drv.find_element("css selector","body")
            for part in seq:
                target.send_keys(part)
            return f"Pressed keys: {keys}"
        except Exception as e:
            return f"Error pressing keys {keys!r}: {e}"

    @staticmethod
    def select_option(selector: str, by: str = "value", value: str = "") -> str:
        """
        Select an option in a <select> element.
        - by: 'value' | 'text' | 'index'
        - value: value/text/index string
        """
        if not Tools._driver:
            return "Error: browser not open"
        from selenium.webdriver.support.ui import Select
        try:
            el = Tools._driver.find_element("css selector", selector)
            sel = Select(el)
            if by == "value":
                sel.select_by_value(value)
            elif by == "text":
                sel.select_by_visible_text(value)
            elif by == "index":
                sel.select_by_index(int(value))
            else:
                return f"Error: unknown 'by' {by!r}"
            return f"Selected {value!r} by {by} in {selector}"
        except Exception as e:
            return f"Error selecting option in {selector}: {e}"

    @staticmethod
    def scroll(to: str = "bottom", amount: int = 600) -> str:
        """
        Scroll the page.
        - to: 'top' | 'bottom' | 'amount'
        - amount: pixels if to=='amount' (positive: down, negative: up)
        """
        if not Tools._driver:
            return "Error: browser not open"
        drv = Tools._driver
        try:
            if to == "top":
                drv.execute_script("window.scrollTo({top:0, behavior:'instant'});")
            elif to == "bottom":
                drv.execute_script("window.scrollTo({top:document.body.scrollHeight, behavior:'instant'});")
            elif to == "amount":
                drv.execute_script("window.scrollBy(0, arguments[0]);", int(amount))
            else:
                return f"Error: unknown target {to!r}"
            return f"Scrolled {to}"
        except Exception as e:
            return f"Error scrolling: {e}"

    # ── Observation & queries (for agentic decisions) ─────────────────────────
    @staticmethod
    def exists(selector: str, timeout: int = 0) -> bool:
        "Return True if at least one element exists; optionally wait up to timeout seconds."
        if not Tools._driver:
            return False
        from selenium.webdriver.common.by import By
        if timeout > 0:
            # simple poll
            end = time.time() + timeout
            while time.time() < end:
                try:
                    if Tools._driver.find_elements(By.CSS_SELECTOR, selector):
                        return True
                except Exception:
                    pass
                time.sleep(0.1)
            return False
        try:
            return bool(Tools._driver.find_elements(By.CSS_SELECTOR, selector))
        except Exception:
            return False

    @staticmethod
    def count(selector: str) -> int:
        "Return number of elements matching selector."
        if not Tools._driver:
            return 0
        from selenium.webdriver.common.by import By
        try:
            return len(Tools._driver.find_elements(By.CSS_SELECTOR, selector))
        except Exception:
            return 0

    @staticmethod
    def get_text(selector: str, all: bool = False, sep: str = "\\n") -> Union[str, List[str]]:
        """
        Extract visible text from elements.
        - all=False: text from first match
        - all=True: list of texts for all matches
        """
        if not Tools._driver:
            return "Error: browser not open"
        from selenium.webdriver.common.by import By
        drv = Tools._driver
        try:
            els = drv.find_elements(By.CSS_SELECTOR, selector)
            if not els:
                return "" if not all else []
            if not all:
                return els[0].text.strip()
            return [e.text.strip() for e in els]
        except Exception as e:
            return f"Error get_text({selector}): {e}"

    @staticmethod
    def get_attr(selector: str, attr: str, all: bool = False) -> Union[str, List[str]]:
        "Get attribute from first/all elements matching selector."
        if not Tools._driver:
            return "Error: browser not open"
        from selenium.webdriver.common.by import By
        drv = Tools._driver
        try:
            els = drv.find_elements(By.CSS_SELECTOR, selector)
            if not els:
                return "" if not all else []
            if not all:
                return (els[0].get_attribute(attr) or "").strip()
            return [(e.get_attribute(attr) or "").strip() for e in els]
        except Exception as e:
            return f"Error get_attr({selector}, {attr}): {e}"

    @staticmethod
    def observe(max_nodes: int = 50) -> Dict[str, Any]:
        """
        Return a compact snapshot of the current page for agentic evaluation:
        {url, title, readyState, viewport:{w,h,scrollY}, dom:[{tag,id,cls,text,href,aria,label}...]}
        """
        if not Tools._driver:
            return {"error": "browser not open"}
        drv = Tools._driver
        try:
            data = drv.execute_script("""
                const lim = arguments[0] || 50;
                const nodes = [];
                const q = document.querySelectorAll("h1,h2,h3,h4,button,a,input,textarea,select,[role],[aria-label]");
                const take = Array.from(q).slice(0, lim);
                for (const el of take) {
                  const tag = el.tagName.toLowerCase();
                  const id = el.id || "";
                  const cls = (el.className || "").toString();
                  const text = (el.innerText || el.value || "").trim().slice(0, 160);
                  const href = (el.getAttribute && el.getAttribute("href")) || "";
                  const role = el.getAttribute && el.getAttribute("role") || "";
                  const aria = el.getAttribute && el.getAttribute("aria-label") || "";
                  const name = el.getAttribute && (el.getAttribute("name") || el.getAttribute("data-testid") || "");
                  nodes.push({tag, id, cls, text, href, role, aria, name});
                }
                return {
                  url: location.href,
                  title: document.title,
                  readyState: document.readyState,
                  viewport: { w: innerWidth, h: innerHeight, scrollY: scrollY },
                  dom: nodes
                };
            """, int(max_nodes))
            return data
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def eval_js(script: str) -> Any:
        "Evaluate a short JavaScript snippet in the page context and return its JSON-serializable result."
        if not Tools._driver:
            return {"error": "browser not open"}
        try:
            return Tools._driver.execute_script("return (function(){try{return (" + script + ");}catch(e){return String(e);}})();")
        except Exception as e:
            return {"error": str(e)}

    # ── Page source & screenshots ─────────────────────────────────────────────
    @staticmethod
    def get_html() -> str:
        "Return full page HTML."
        if not Tools._driver:
            return "Error: browser not open"
        return Tools._driver.page_source

    @staticmethod
    def screenshot(filename: str = "screenshot.png") -> str:
        "Save a screenshot to the given filename (relative to working dir). Returns the filename or error."
        if not Tools._driver:
            return "Error: browser not open"
        try:
            Tools._driver.save_screenshot(filename)
            return filename
        except Exception as e:
            return f"Error taking screenshot: {e}"

    # ── Simple DuckDuckGo web search (agentic-friendly) ───────────────────────
    @staticmethod
    def search_internet(topic: str, num_results: int = 5, wait_sec: int = 2, deep_scrape: bool = True, summarize: bool = False) -> List[Dict[str, Any]]:
        """
        Headless DuckDuckGo search with optional deep scrape.
        Returns list of {title, url, snippet, content?}.
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException

        results: List[Dict[str, Any]] = []
        try:
            Tools.close_browser()
            Tools.open_browser(headless=True, force_new=True)
            drv = Tools._driver
            wait = WebDriverWait(drv, wait_sec, poll_frequency=0.1)
            drv.get("https://duckduckgo.com/")
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            box = drv.find_element(By.CSS_SELECTOR, "input[name='q']")
            box.send_keys(topic)
            box.submit()
            try:
                wait.until(lambda d: "?q=" in d.current_url)
            except TimeoutException:
                pass
            anchors = drv.find_elements(By.CSS_SELECTOR, "a.result__a, a[data-testid='result-title-a']")[:num_results]
            for a in anchors:
                href = a.get_attribute("href")
                title = a.text.strip() or drv.execute_script("return arguments[0].innerText;", a)
                snippet = ""
                try:
                    parent = a.find_element(By.XPATH, "./ancestor::*[contains(@class,'result')][1]")
                    sn = parent.find_element(By.CSS_SELECTOR, ".result__snippet, span[data-testid='result-snippet']")
                    snippet = sn.text.strip()
                except Exception:
                    pass
                content = ""
                if deep_scrape and href:
                    try:
                        content = bs4_scrape(href, timeout_s=8, full_html=False)[:10000]
                    except Exception:
                        content = ""
                entry = {"title": title, "url": href, "snippet": snippet}
                if content:
                    entry["content"] = content
                results.append(entry)
        except Exception as e:
            log_message(f"[search_internet] error: {e}", "ERROR")
        finally:
            Tools.close_browser()
        return results
'''


if not TOOLS_PATH.exists():
    print(_c("RUN", "Creating default tools.py ..."))
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
                print(_c("RUN", "(fallback) Listed models via REST /api/tags."))
        except Exception as e:
            print(_c("ERR", f"REST /api/tags fallback failed: {e}"))
    return models

def initial_setup() -> AppConfig:
    host_env = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    cli = _client_for(host_env)
    print(_c("RUN", f"Querying Ollama at {host_env} for local models ..."))
    try:
        models = _list_models(cli, host_env)
    except Exception as e:
        print(_c("ERR", "Could not contact Ollama. Is `ollama serve` running?")); print("Error:", e); sys.exit(2)
    if not models:
        print(_c("ERR", "No local models found. Example: `ollama pull qwen3:4b` then rerun."))
        sys.exit(3)
    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        name = m.get("name") or m.get("model") or "unknown"
        size = m.get("size")
        size_mb = f"{(size/1024/1024):.1f} MB" if isinstance(size, (int, float)) else "?"
        family = (m.get("details") or {}).get("family", "")
        print(f"  [{i}] {name:<24} ({family})  {size_mb}")
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
    print(_c("OK", f"Saved config.json with model='{cfg.model}' host='{cfg.host}'"))
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
# VII. Streaming LLM wrapper (moved BEFORE docs gen to avoid NameError)
# ──────────────────────────────────────────────────────────────────────────────
def _chat_stream_and_collect(messages: List[Dict[str, str]], label: str = "LLM", options: Optional[Dict[str, Any]] = None) -> str:
    opts = (CONFIG.options or {}).copy()
    if options:
        opts.update(options)
    if AGENT_STREAM:
        try:
            print(_c("AST", f"{label} streaming ..."))
            content, start = [], time.time()
            for chunk in CLIENT.chat(model=CONFIG.model, messages=messages, options=opts, stream=True):
                part = (chunk.get("message") or {}).get("content","")
                if part:
                    content.append(part)
                    sys.stdout.write(C.G + part + C.RST)
                    sys.stdout.flush()
            sys.stdout.write("\n"); sys.stdout.flush()
            txt = "".join(content).strip()
            print(_c("OK", f"{label} done in {int((time.time()-start)*1000)} ms"))
            return txt
        except Exception as e:
            print(_c("ERR", f"{label} stream failed, falling back to non-stream: {e}"))
    resp = CLIENT.chat(model=CONFIG.model, messages=messages, options=opts)
    return resp.get("message", {}).get("content","").strip()

def _llm(system: str, user_obj: Any, temp=0.1, preface: Optional[str]=None, label: str="LLM") -> str:
    msgs = [{"role":"system","content":system}]
    if INTERLEAVE and preface:
        msgs.append({"role":"assistant","content":preface})
    msgs.append({"role":"user","content":json.dumps(user_obj, ensure_ascii=False)})
    _emit_chat_preview(system, user_obj, preface, label)
    if DEBUG: _dbg(label, msgs)
    return _chat_stream_and_collect(msgs, label=label, options={"temperature":temp})

# ──────────────────────────────────────────────────────────────────────────────
# VIII. tool discovery + docs generation + aliases
# ──────────────────────────────────────────────────────────────────────────────
def import_tools_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("tools", str(TOOLS_PATH))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod  # type: ignore

class ArgSpec(BaseModel):
    name: str; type: str; required: bool=True; default: Optional[Any]=None; doc: Optional[str]=None

class ToolSpec(BaseModel):
    name: str; qualname: str; origin: str; doc: str=""; args: List[ArgSpec]=Field(default_factory=list); returns: Optional[str]=None

def _typename(t: Any) -> str:
    if t is None: return "None"
    if getattr(t, "__origin__", None) is Union:
        args = [a.__name__ if hasattr(a, "__name__") else str(a) for a in t.__args__]
        return "Union[" + ", ".join(args) + "]"
    if hasattr(t, "__name__"): return t.__name__
    return str(t)

def discover_tools(mod: types.ModuleType) -> List[ToolSpec]:
    tools: List[ToolSpec] = []
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"): continue
        if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
            ann = get_type_hints(obj); sig = inspect.signature(obj)
            args: List[ArgSpec] = []
            for p in sig.parameters.values():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD): continue
                arg_t = _typename(ann.get(p.name, Any))
                required = p.default is inspect._empty; default = None if required else p.default
                args.append(ArgSpec(name=p.name, type=arg_t, required=required, default=default))
            ret = _typename(ann.get("return", Any))
            tools.append(ToolSpec(name=name, qualname=name, origin="function", doc=(inspect.getdoc(obj) or ""), args=args, returns=ret))
        if inspect.isclass(obj) and obj.__module__ == mod.__name__ and name.endswith("Tools"):
            for meth_name, meth in inspect.getmembers(obj, predicate=inspect.isfunction):
                if meth_name.startswith("_"): continue
                ann = get_type_hints(meth); sig = inspect.signature(meth)
                args: List[ArgSpec] = []
                for p in sig.parameters.values():
                    if p.name in ("self", "cls"): continue
                    if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD): continue
                    arg_t = _typename(ann.get(p.name, Any))
                    required = p.default is inspect._empty; default = None if required else p.default
                    args.append(ArgSpec(name=p.name, type=arg_t, required=required, default=default))
                ret = _typename(ann.get("return", Any))
                tools.append(ToolSpec(name=f"{name}.{meth_name}", qualname=f"{name}.{meth_name}", origin=f"class:{name}", doc=(inspect.getdoc(meth) or ""), args=args, returns=ret))
    return tools

# Robust JSON extractor to avoid "Expecting value" when the model adds prose/fences/empty
def _extract_json_map(content: str) -> Dict[str, Any]:
    s = (content or "").strip()
    if not s:
        raise ValueError("empty content")
    # fenced ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.S)
    if m:
        return json.loads(m.group(1))
    # generic fenced ```
    if s.startswith("```"):
        s2 = s.strip("`")
        if "\n" in s2:
            s2 = s2.split("\n", 1)[1]
        s2 = s2.strip()
        return json.loads(s2)
    # try to carve the first balanced JSON object
    start = s.find("{")
    if start != -1:
        depth = 0
        end_idx = None
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        if end_idx:
            return json.loads(s[start:end_idx])
    # final attempt
    return json.loads(s)

def ensure_docs(tools: List["ToolSpec"], force: bool=False) -> Dict[str, Any]:
    """
    Generate and persist tool docs one function at a time to avoid overwhelming the doc writer.

    Behavior:
      - Loads existing docs from DOCS_PATH (mapping: tool_name -> doc).
      - For each tool, computes a stable fingerprint of its signature+docstring.
      - If not force and fingerprint matches what's stored, skips regeneration.
      - Otherwise, calls the doc writer for that SINGLE tool only.
      - Saves/updates docs.json incrementally after each tool.
      - Also writes a per-tool JSON under data/tool_docs/<tool>.json for clarity.

    Return:
      - The full docs mapping loaded+updated from disk.
    """
    existing = jload(DOCS_PATH, default={}) or {}
    docs: Dict[str, Any] = dict(existing)

    # Local helper: stable fingerprint so we only re-generate when needed
    def _fingerprint(t: "ToolSpec") -> str:
        import hashlib
        base = {
            "name": t.name,
            "origin": t.origin,
            "doc": t.doc or "",
            "args": [(a.name, a.type, bool(a.required), (None if a.default is None else str(a.default))) for a in t.args],
            "returns": t.returns or "",
        }
        raw = json.dumps(base, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # One-tool-at-a-time prompt
    SYSTEM = (
        "You are ToolDocBot v2.\n"
        "Goal: Given ONE Python tool (its name, origin, docstring, arg names/types/defaults, and return type), "
        "produce a STRICT, single JSON object that fully captures how an autonomous agent should use it in complex, "
        "real-world runs (including browser/network/stateful contexts). Do not return any prose outside JSON. "
        "Do not include code fences.\n"
        "\n"
        "JSON schema to output (keys exactly as below):\n"
        "{\n"
        '  "purpose": str,                               # one sentence, crisp\n'
        '  "when_to_use": [str, ...],                    # bullet-y triggers; be specific\n'
        '  "preconditions": [str, ...],                  # env/state that MUST be true (e.g., open browser, network, cwd)\n'
        '  "inputs": {                                   # one entry per arg; reflect real constraints\n'
        '    "<arg>": {\n'
        '      "desc": str,\n'
        '      "type": str,                              # echo type info (Union, List[T], etc.)\n'
        '      "required": bool,\n'
        '      "default": any,                           # null if none\n'
        '      "constraints": {                          # only include fields that apply\n'
        '        "format": "url|path|css-selector|regex|json|number|text|enum|timeout-s|headers|payload",\n'
        '        "regex": str,\n'
        '        "enum": [any, ...],\n'
        '        "min": number,\n'
        '        "max": number,\n'
        '        "min_len": int,\n'
        '        "max_len": int,\n'
        '        "units": "px|ms|s|bytes|…" ,\n'
        '        "case_sensitive": bool,\n'
        '        "example": any,\n'
        '        "coupled_with": ["other_arg", ...]      # if values must be provided together\n'
        '      }\n'
        '    }, ...\n'
        '  },\n'
        '  "behavior": {\n'
        '    "idempotent": bool,                         # true if repeated calls don’t alter state\n'
        '    "deterministic": bool,                      # true if same inputs ⇒ same outputs\n'
        '    "side_effects": [str, ...],                 # files written, navigation, clicks, network calls, memory writes\n'
        '    "state_dependencies": [str, ...],           # e.g., Tools._driver is open; document.readyState==complete\n'
        '    "timeouts": { "default_s": int, "notes": str },\n'
        '    "retries": { "recommended": int, "strategy": "none|fixed|exponential", "backoff_s": number },\n'
        '    "performance": { "expected_latency_ms": [low, high], "scaling_note": str }\n'
        '  },\n'
        '  "returns": {\n'
        '    "type": str,                                # e.g., Dict[str, Any]\n'
        '    "schema": any,                              # example-shaped object/array with key fields populated\n'
        '    "notes": str,                               # nulls/sentinels, empty-shape behavior\n'
        '    "success_criteria": [str, ...]              # how an agent can assert success from the output\n'
        '  },\n'
        '  "failure_modes": [                            # concrete, inspectable failure taxonomy\n'
        '    { "case": str, "detect_by": str, "message": str, "recovery": str },\n'
        '    ...\n'
        '  ],\n'
        '  "agent_playbook": {\n'
        '    "happy_path": [str, ...],                   # step list the agent should follow with this tool\n'
        '    "next_best_actions_on_failure": [str, ...], # fallbacks (e.g., wait_for → observe → retry with backoff)\n'
        '    "pair_with_tools": [ { "tool": str, "why": str }, ... ]\n'
        '  },\n'
        '  "examples": [                                 # 1–3 concrete calls\n'
        '    { "call": str, "explain": str },\n'
        '    ...\n'
        '  ]\n'
        "}\n"
        "\n"
        "Authoring rules:\n"
        "- Derive constraints from arg types, names, defaults, and docstring cues. If not knowable, omit the field; do NOT invent.\n"
        "- If origin is 'class:Tools' or tool name starts with 'Tools.', assume a Selenium/DOM context and include browser preconditions "
        "  (e.g., driver open via open_browser, page load readiness, selector existence) and side effects (click, input, scroll).\n"
        "- If the tool name/doc references URL/HTTP/fetch, add network assumptions: DNS, TLS, headers, auth, rate limits, status handling.\n"
        "- If the tool writes files (json_write, csv_write, screenshot), list paths/dirs touched in side_effects and success_criteria.\n"
        "- Keep outputs compact but concrete. Prefer lists of short bullets over long prose.\n"
        "- Output ONLY the JSON object. No markdown, no commentary, no code fences."
    )


    # Create a tidy per-tool folder (no global constant needed)
    tool_docs_dir = DATA_DIR / "tool_docs"
    tool_docs_dir.mkdir(exist_ok=True)

    updated = 0
    for t in tools:
        fp = _fingerprint(t)
        prev = docs.get(t.name)
        if not force and isinstance(prev, dict) and prev.get("__fingerprint__") == fp:
            # No change → keep existing
            continue

        payload = {
            "name": t.name,
            "origin": t.origin,
            "doc": t.doc,
            "args": [
                {
                    "name": a.name,
                    "type": a.type,
                    "required": bool(a.required),
                    "default": None if a.required else a.default,
                }
                for a in t.args
            ],
            "returns": t.returns,
        }

        USER = {
            "instruction": "Write JSON docs for THIS SINGLE TOOL only.",
            "tool": payload,
        }

        print(_c("PLAN", f"Docgen → {t.name}"))
        try:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(USER, ensure_ascii=False)},
            ]
            if DEBUG:
                _dbg(f"TOOLDOCS:{t.name}", msgs)

            content = _chat_stream_and_collect(msgs, label=f"TOOLDOCS:{t.name}", options={"temperature": 0.1})
            try:
                doc_obj = _extract_json_map(content)
                if not isinstance(doc_obj, dict) or not doc_obj:
                    raise ValueError("empty or non-dict JSON")
            except Exception:
                # Fallback: introspection-only doc
                doc_obj = {
                    "purpose": (t.doc or f"{t.name} tool").strip(),
                    "when_to_use": "Use when this tool's purpose matches your need.",
                    "args": {a.name: f"{a.type} (required)" if a.required else f"{a.type} (optional, default={a.default})" for a in t.args},
                    "returns": t.returns or "Any",
                    "example": f"{t.name}({', '.join(a.name for a in t.args)})",
                }
        except Exception as e:
            # Hard fallback if LLM call itself failed
            doc_obj = {
                "purpose": (t.doc or f"{t.name} tool").strip(),
                "when_to_use": "Use when this tool's purpose matches your need.",
                "args": {a.name: f"{a.type} (required)" if a.required else f"{a.type} (optional, default={a.default})" for a in t.args},
                "returns": t.returns or "Any",
                "example": f"{t.name}({', '.join(a.name for a in t.args)})",
                "__error__": f"docgen_failed: {e}",
            }

        # Stamp meta + fingerprint
        doc_obj["__fingerprint__"] = fp
        doc_obj["__meta__"] = {
            "updated_at": _now_iso(),
            "origin": t.origin,
        }

        # Save to in-memory map and to disk immediately
        docs[t.name] = doc_obj
        try:
            jdump(DOCS_PATH, docs)
        except Exception as e:
            print(_c("ERR", f"Failed writing {DOCS_PATH}: {e}"))

        # Also save a per-tool JSON for clarity
        try:
            fname = t.name.replace(".", "_")
            (tool_docs_dir / f"{fname}.json").write_text(json.dumps({t.name: doc_obj}, indent=2, ensure_ascii=False))
        except Exception as e:
            print(_c("ERR", f"Failed writing per-tool doc for {t.name}: {e}"))

        updated += 1

    print(_c("OK", f"Docs ready: {len(docs)} total, {updated} updated."))
    return docs



def build_tool_registry(mod: types.ModuleType) -> Dict[str, Callable[..., Any]]:
    reg: Dict[str, Callable[..., Any]] = {}
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"): continue
        if inspect.isfunction(obj) and obj.__module__ == mod.__name__:
            reg[name] = obj
        if inspect.isclass(obj) and obj.__module__ == mod.__name__ and name.endswith("Tools"):
            inst = obj()
            for meth_name, meth in inspect.getmembers(inst, predicate=callable):
                if meth_name.startswith("_"): continue
                if inspect.ismethod(meth) or inspect.isfunction(meth):
                    reg[f"{obj.__name__}.{meth_name}"] = meth
    return reg

TOOLS_MOD = import_tools_module()
TOOLS = discover_tools(TOOLS_MOD)
DOCS = ensure_docs(TOOLS, force=False)
REGISTRY = build_tool_registry(TOOLS_MOD)

# Aliases: "now" => "UtilityTools.now", etc.
TOOL_ALIASES: Dict[str, str] = {}
for full in list(REGISTRY.keys()):
    if "." in full:
        simple = full.split(".")[-1]
        TOOL_ALIASES.setdefault(simple, full)
    else:
        TOOL_ALIASES.setdefault(full, full)

def resolve_tool_name(name: str) -> str:
    if name in REGISTRY:
        return name
    return TOOL_ALIASES.get(name, name)

# ──────────────────────────────────────────────────────────────────────────────
# IX. DAG models + strict plan parsing + placeholder/dep repair
# ──────────────────────────────────────────────────────────────────────────────
class Node(BaseModel):
    id: str; tool: str; args: Any=Field(default_factory=dict)
    after: List[str]=Field(default_factory=list)
    retries: int=1; timeout_s: int=30; alias: Optional[str]=None

class Plan(BaseModel):
    nodes: List[Node]
    meta: Dict[str, Any]=Field(default_factory=lambda: {"parallelism":4, "retries":1, "timeout_s":120})

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

PLACEHOLDER_RE = re.compile(r"\[([^\]]+)\]")
UNBRACKETED_RE = re.compile(r"\b([A-Za-z_]\w*)\.output(?:\.[A-Za-z0-9_\-\[\]]+)+")

def validate_and_fix_placeholders(plan: Plan) -> Plan:
    node_ids = {n.id for n in plan.nodes}
    tool_to_nodes: Dict[str, List[str]] = {}
    for n in plan.nodes:
        tool_to_nodes.setdefault(n.tool, []).append(n.id)

    def wrap_if_unbracketed(s: str) -> str:
        def repl(m): return "[" + m.group(0) + "]"
        return UNBRACKETED_RE.sub(repl, s)

    def fix_in_string(s: str) -> str:
        s = wrap_if_unbracketed(s)
        def _fix_token(token: str) -> str:
            parts = token.split(".")
            if parts and parts[0] not in node_ids:
                cands = tool_to_nodes.get(parts[0], [])
                if cands:
                    parts[0] = cands[0]
            return "[" + ".".join(parts) + "]"
        return PLACEHOLDER_RE.sub(lambda m: _fix_token(m.group(1)), s)

    def walk_args(x: Any) -> Any:
        if isinstance(x, str): return fix_in_string(x)
        if isinstance(x, list): return [walk_args(v) for v in x]
        if isinstance(x, dict): return {k: walk_args(v) for k, v in x.items()}
        return x

    for n in plan.nodes:
        n.args = walk_args(n.args)

    for n in plan.nodes:
        clean_after = []
        for dep in n.after:
            dep_id = dep.split(".")[0] if isinstance(dep, str) else dep
            if dep_id in node_ids:
                clean_after.append(dep_id)
            else:
                cands = tool_to_nodes.get(dep_id, [])
                if cands:
                    clean_after.append(cands[0])
        n.after = list(dict.fromkeys(clean_after))
    return plan

def _resolve_placeholders(args: Any, outputs: Dict[str, ToolRecord]) -> Any:
    def lookup(path: str):
        parts = path.split(".")
        if len(parts) >= 2 and parts[1].startswith("output"):
            nid = parts[0]
            rec = outputs.get(nid)
            if not rec or rec.output is None:
                raise KeyError(f"Missing output for {nid}")
            cur = rec.output
            for p in parts[2:]:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    raise KeyError(f"Missing path: {path}")
            return cur
        raise KeyError(f"Unsupported placeholder: {path}")

    def replace(x: Any) -> Any:
        if isinstance(x, str):
            out = x
            for token in PLACEHOLDER_RE.findall(x):
                if "?" in token or "|" in token:
                    left, *rest = token.split("|", 1)
                    default = rest[0] if rest else None
                    left = left.replace("?.",".")
                    try:
                        val = lookup(left)
                    except Exception:
                        if default is None: raise
                        val = default
                else:
                    val = lookup(token)
                out = out.replace(f"[{token}]", json.dumps(val) if not isinstance(val, str) else val)
            return out
        if isinstance(x, list): return [replace(v) for v in x]
        if isinstance(x, dict): return {k: replace(v) for k, v in x.items()}
        return x
    return replace(args)

# ──────────────────────────────────────────────────────────────────────────────
# X. Concurrency executor with live color logs
# ──────────────────────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_node(node: "Node", registry: Dict[str, Callable[..., Any]], outputs: Dict[str, "ToolRecord"]) -> "ToolRecord":
    start = time.time(); rec = ToolRecord(node_id=node.id, tool=node.tool, args_resolved=None)
    print(_c("NODE", f"start {node.id} → {node.tool}"))
    try:
        tool_name = resolve_tool_name(node.tool)
        fn = registry.get(tool_name)
        if not fn:
            raise RuntimeError(f"Unknown tool: {node.tool}")
        args_resolved = _resolve_placeholders(node.args, outputs); rec.args_resolved = args_resolved
        print(_c("TOOL", f"{node.id} args: {json.dumps(args_resolved)[:200]}"))
        if isinstance(args_resolved, dict): out = fn(**args_resolved)
        elif isinstance(args_resolved, list): out = fn(*args_resolved)
        else: out = fn(args_resolved)
        rec.output = out; rec.stdout_short = str(out)[:200]; rec.error = None
    except Exception as e:
        rec.error = {"type": e.__class__.__name__, "message": str(e), "retryable": False}
        print(_c("ERR", f"{node.id} {node.tool} failed: {rec.error}"))
    finally:
        rec.ended_at = _now_iso(); rec.elapsed_ms = int((time.time()-start)*1000)
        tag = "OK" if not rec.error else "ERR"
        print(_c(tag, f"done {node.id} {node.tool} ({rec.elapsed_ms} ms)"))
    return rec

def _topo_ready(state: "TurnState") -> List["Node"]:
    id_to_node = {n.id: n for n in state.graph.nodes}
    ready = []
    for nid in list(state.pending):
        node = id_to_node[nid]
        if all(dep in state.completed for dep in node.after): ready.append(node)
    return ready

def execute_plan(plan: "Plan", registry: Dict[str, Callable[..., Any]]) -> Tuple[List["ToolRecord"], Dict[str, "ToolRecord"]]:
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
            print(_c("ERR", "Deadlock or unmet deps. Breaking.")); break

        batch = ready[:max(1, parallelism)]
        print(_c("PLAN", f"dispatch {len(batch)} node(s): {', '.join(n.id for n in batch)}"))
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
                    print(_c("Y", f"RETRY {node.id} remaining={retry_counts[node.id]}"))
                else:
                    if node.id in state.pending:
                        state.pending.remove(node.id)
                    state.completed.append(node.id)

    ordered = [records[nid] for nid in [n.id for n in plan.nodes] if nid in records]
    return ordered, records

# ──────────────────────────────────────────────────────────────────────────────
# XI. LLM stages (clarify, plan, red-team, merge, reflect, assemble)
# ──────────────────────────────────────────────────────────────────────────────
ANTI_SIM_PHRASES = re.compile(r"\b(simulat(?:e|ion)|placeholder|null(?:\s+values?)?|doesn'?t actually|mock(?:ed|ing)?|fake|dummy)\b", re.IGNORECASE)
def response_smells_simulated(text: str) -> bool: return bool(ANTI_SIM_PHRASES.search(text))

def _llm_stage(system_key: str, user: Dict[str, Any], temp: float, label: str, preface: Optional[str] = None) -> str:
    sys = sysmsg(system_key, {"REALITY_POLICY": sysmsg("reality_policy"), "AVAILABLE_TOOLS": "", "REALITY_POLICY_SECTION": sysmsg("reality_policy")})
    return _llm(sys, user, temp=temp, preface=preface, label=label)

def clarify_goal(goal: str) -> str:
    SYSTEM = sysmsg("clarify", {"REALITY_POLICY": sysmsg("reality_policy")})
    tail = ctx_bus_tail(10)
    user = {"goal": goal, "recent_context": tail}
    for attempt in range(1, 4):
        pre = "Do NOT suggest simulation or placeholders. List concrete constraints and real-data needs."
        r = _llm(SYSTEM, user, temp=0.1, preface=pre, label=f"CLARIFY#{attempt}") or goal
        if not response_smells_simulated(r):
            return r
        user["violation"] = "Detected simulation/placeholder language; restate with real-data intent only."
    return r

def plan_with_llm(goal: str, tools: List["ToolSpec"], docs: Dict[str, Any]) -> "Plan":
    available = sorted(list(REGISTRY.keys()) + list(TOOL_ALIASES.keys()))
    tools_brief = [{"name": t.name, "args":[a.name for a in t.args], "returns": t.returns} for t in tools]
    SYSTEM = sysmsg("plan", {"AVAILABLE_TOOLS": json.dumps(available), "REALITY_POLICY_SECTION": sysmsg("reality_policy")})
    base_user = {"goal": goal, "tools": tools_brief, "docs": docs, "recent_context": ctx_bus_tail(10)}
    msgs = [{"role":"system","content":SYSTEM}]
    if INTERLEAVE: msgs.append({"role":"assistant","content":"Avoid placeholders; break into verifiable steps; include validations."})
    msgs.append({"role":"user","content":json.dumps(base_user, ensure_ascii=False)})
    _emit_chat_preview(SYSTEM, base_user, msgs[1]["content"] if len(msgs) > 1 and msgs[1]["role"]=="assistant" else None, "PLAN")
    if DEBUG: _dbg("PLAN", msgs)
    for attempt in range(1, 4):
        content = _chat_stream_and_collect(msgs, label=f"PLAN#{attempt}", options={"temperature":0.05})
        try:
            plan_obj = parse_plan_strict(content)
            plan = Plan(**plan_obj)
            return validate_and_fix_placeholders(plan)
        except Exception as e:
            msgs.append({"role":"assistant","content":content})
            msgs.append({"role":"user","content":f"ERROR: {e}. Re-output ONLY valid JSON with nodes/meta."})
    seeded = Plan(nodes=[
        Node(id="n1", tool="word_count", args={"text": goal}),
        Node(id="n2", tool="json_write", args={"path":"last_word_count.json","obj":"[n1.output]"}, after=["n1"]),
    ])
    return validate_and_fix_placeholders(seeded)

def merge_plan(original: "Plan", patch: Dict[str, Any]) -> "Plan":
    id_to_node = {n.id: n for n in original.nodes}
    for nid in patch.get("remove", []): id_to_node.pop(nid, None)
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
        try: id_to_node[a["id"]] = Node(**a)
        except Exception: pass
    merged = Plan(nodes=list(id_to_node.values()), meta=original.meta)
    return validate_and_fix_placeholders(merged)

def red_team_refine(goal: str, initial: "Plan", tools: List["ToolSpec"]) -> "Plan":
    available = sorted(list(REGISTRY.keys()) + list(TOOL_ALIASES.keys()))
    SYSTEM = sysmsg("red_team")
    USER = {"goal": goal, "initial_plan": initial.model_dump(), "available_tools": available, "docs": DOCS, "recent_context": ctx_bus_tail(10)}
    patch_raw = _llm(SYSTEM, USER, temp=0.2, label="RED-TEAM")
    try:
        patch_obj = json.loads(patch_raw)
    except Exception:
        return initial
    merged = merge_plan(initial, patch_obj)
    try:
        SYSTEM_MERGE = sysmsg("plan_merge")
        merged2 = _llm(SYSTEM_MERGE, {"goal": goal, "original": initial.model_dump(), "patch": patch_obj}, temp=0.0, label="PLAN-MERGE")
        obj = parse_plan_strict(merged2); merged = validate_and_fix_placeholders(Plan(**obj))
    except Exception:
        pass
    return merged

def reflect_repair(goal: str, tools: List["ToolSpec"], docs: Dict[str, Any], records: List["ToolRecord"], original_plan: "Plan") -> Optional["Plan"]:
    errors = [r for r in records if r.error]
    if not errors: return None
    SYSTEM = sysmsg("reflect")
    payload = {
        "goal": goal, "original_plan": original_plan.model_dump(),
        "failed": [{"node_id": r.node_id, "tool": r.tool, "error": r.error} for r in errors],
        "tools": [{"name": t.name, "args":[a.name for a in t.args]} for t in tools], "docs": DOCS,
        "recent_context": ctx_bus_tail(10),
    }
    msgs = [{"role":"system","content":SYSTEM}]
    if INTERLEAVE: msgs.append({"role":"assistant","content":"Repair minimally. Keep IDs stable where possible."})
    msgs.append({"role":"user","content":json.dumps(payload, ensure_ascii=False)})
    _emit_chat_preview(SYSTEM, payload, msgs[1]["content"] if len(msgs)>1 and msgs[1]["role"]=="assistant" else None, "REFLECT")
    if DEBUG: _dbg("REFLECT", msgs)
    for _ in range(3):
        content = _chat_stream_and_collect(msgs, label="REFLECT", options={"temperature":0.05})
        try:
            plan_obj = parse_plan_strict(content)
            return validate_and_fix_placeholders(Plan(**plan_obj))
        except Exception as e:
            msgs.append({"role":"assistant","content":content})
            msgs.append({"role":"user","content":f"ERROR: {e}. Re-output ONLY valid JSON."})
    return None

def assemble_final(goal: str, clarified: str, records: Dict[str, "ToolRecord"]) -> str:
    blocks = []
    for nid, rec in records.items():
        hdr = f"[tool:{rec.tool} → node={nid}] ({rec.elapsed_ms} ms)"
        short = rec.stdout_short or ""
        blocks.append(f"{hdr}\nsummary: {short}\njson: {json.dumps(rec.output)[:300]}")
    SYSTEM = sysmsg("assemble")
    user = {"goal": goal, "clarified": clarified, "tool_blocks": blocks}
    return _llm(SYSTEM, user, temp=0.2, label="ASSEMBLE")

# ──────────────────────────────────────────────────────────────────────────────
# XII. ToolFoundry (iterative code synth & tests)
# ──────────────────────────────────────────────────────────────────────────────
CAPS = {
    "PURE_FUNC": {"allowed_imports": {"math","re","json","statistics"}, "io": False, "net": False},
    "NET_TOOL":  {"allowed_imports": {"httpx","json","re"},            "io": False, "net": True},
    "IO_TOOL":   {"allowed_imports": {"json","re"},                    "io": True,  "net": False},
}

PIP_STATE_PATH = DATA_DIR / "pip_state.json"
PIP_LOCK_PATH  = DATA_DIR / "pip.lock"

def ensure_pip_installed(requirements: List[str], timeout_s: int = 300) -> Dict[str, Any]:
    import importlib
    def _top_module(req: str) -> str:
        base = re.split(r"[<>=!~\[]", req, 1)[0].strip()
        if not base: return ""
        mapping = {"beautifulsoup4":"bs4","opencv-python":"cv2","pillow":"PIL","pyyaml":"yaml","scikit-learn":"sklearn","pydantic-core":"pydantic_core"}
        return mapping.get(base.lower(), base.replace("-", "_"))

    def _read_state(): return jload(PIP_STATE_PATH, default={"installed": {}, "history": []}) or {"installed": {}, "history": []}
    def _write_state(state):
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
    if not reqs: return {"ok": True, "installed": [], "skipped": [], "errors": []}

    state = _read_state()
    installed_ok, skipped, errors = [], [], []
    pre_need = []
    for r in reqs:
        mod = _top_module(r)
        try:
            if mod:
                importlib.import_module(mod); skipped.append(r)
                state["installed"].setdefault(r, {"ts": _now_iso(), "via": "pre-import"})
            else:
                pre_need.append(r)
        except Exception:
            pre_need.append(r)

    if not pre_need:
        _write_state(state); return {"ok": True, "installed": [], "skipped": skipped, "errors": []}

    print(_c("RUN", f"pip install: {', '.join(pre_need)}"))
    if not _lock_acquire():
        return {"ok": False, "installed": installed_ok, "skipped": skipped, "errors": [{"req": "<lock>", "err": "pip lock timeout"}]}

    try:
        for r in pre_need:
            cmd = [str(PY_EXE), "-m", "pip", "install", "--disable-pip-version-check", "--no-input", r]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, check=False, text=True)
                if proc.returncode != 0:
                    errors.append({"req": r, "err": f"pip exit {proc.returncode}: {proc.stderr.strip()[:4000]}"})
                    continue
                mod = _top_module(r)
                try:
                    if mod:
                        importlib.invalidate_caches(); importlib.import_module(mod)
                    installed_ok.append(r); state["installed"][r] = {"ts": _now_iso(), "via": "pip"}
                    state["history"].append({"req": r, "ts": _now_iso(), "status": "ok"})
                except Exception as ie:
                    errors.append({"req": r, "err": f"installed but import failed: {ie}"})
                    state["history"].append({"req": r, "ts": _now_iso(), "status": "import-failed", "detail": str(ie)})
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
                for a in n.names: tops.add(a.name.split(".",1)[0])
            elif isinstance(n, ast.ImportFrom):
                if n.module: tops.add(n.module.split(".",1)[0])
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
        def visit_Import(self, node):
            for alias in node.names:
                base = alias.name.split(".",1)[0]
                if base not in allowed: raise ValueError(f"Disallowed import: {alias.name}")
        def visit_ImportFrom(self, node):
            base = (node.module or "").split(".",1)[0]
            if base not in allowed: raise ValueError(f"Disallowed import-from: {node.module}")
        def visit_Name(self, node):
            if node.id in banned_names: raise ValueError(f"Use of banned name: {node.id}")
        def visit_Attribute(self, node):
            if isinstance(node.attr, str) and node.attr.startswith("__"): raise ValueError(f"Banned dunder attribute: {node.attr}")
            self.generic_visit(node)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__": raise ValueError("Use of __import__ is banned")
            self.generic_visit(node)
        def generic_visit(self, node):
            if isinstance(node, (ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Lambda, ast.Global, ast.Nonlocal, ast.Await, ast.Yield, ast.YieldFrom)):
                raise ValueError(f"Banned node type: {type(node).__name__}")
            return super().generic_visit(node)
    try:
        V().visit(tree)
    except Exception as e:
        return str(e)
    return None

def _deep_equal(a: Any, b: Any, rtol: float = 1e-9) -> bool:
    try:
        if isinstance(a, (int,float)) or isinstance(b, (int,float)):
            fa, fb = float(a), float(b)
            return abs(fa - fb) <= rtol * (1.0 + abs(fb))
    except Exception:
        pass
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()): return False
        return all(_deep_equal(a[k], b[k], rtol) for a in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_equal(x, y, rtol) for x, y in zip(a, b))
    return a == b

class FoundryResult(BaseModel):
    ok: bool; name: str; code: Optional[str]=None
    tests: List[Dict[str, Any]]=Field(default_factory=list); reason: Optional[str]=None

def _kind_from_spec(spec_doc: str) -> str:
    s = (spec_doc or "").lower()
    if any(k in s for k in ["http","api","network","fetch","request","url"]): return "NET_TOOL"
    if any(k in s for k in ["file","load","save","read","write","disk","json file"]): return "IO_TOOL"
    return "PURE_FUNC"

def _tests_index_update(tool: str, tests: List[Dict[str, Any]], status: str, last_err: Any = None):
    idx = jload(TESTS_INDEX_PATH, default={"tools":{}})
    t = idx["tools"].setdefault(tool, {"runs": []})
    t["runs"].append({"ts": _now_iso(), "count": len(tests or []), "status": status, "error": last_err if isinstance(last_err, str) else (json.dumps(last_err) if last_err else None)})
    jdump(TESTS_INDEX_PATH, idx)

def run_tests_and_collect_failures(code: str, tests: List[Dict[str, Any]], fname: str, kind: str) -> Optional[Dict[str, Any]]:
    try:
        import math, re as _re, json as _json, statistics as _statistics
        SAFE_BUILTINS = {"abs":abs,"min":min,"max":max,"sum":sum,"len":len,"range":range,"enumerate":enumerate,"sorted":sorted,"map":map,"filter":filter,"all":all,"any":any,"round":round,"int":int,"float":float,"bool":bool,"str":str,"list":list,"dict":dict,"set":set,"tuple":tuple}
        safe_globals = {"__builtins__": SAFE_BUILTINS, "re": _re, "json": _json, "statistics": _statistics}
        if kind == "PURE_FUNC": safe_globals["math"] = math
        if kind == "NET_TOOL":  safe_globals["httpx"] = __import__("httpx")
        safe_locals: Dict[str, Any] = {}
        exec(code, safe_globals, safe_locals)
        fn = safe_locals.get(fname)
        if not callable(fn): raise RuntimeError("function not defined correctly")
        failures = []
        for i, case in enumerate(tests):
            try:
                got = fn(**case.get("inputs",{}))
                exp = case.get("expect"); tol = float(case.get("rtol", 1e-9)) if isinstance(case, dict) else 1e-9
                if not _deep_equal(got, exp, rtol=tol):
                    failures.append({"i":i,"got":got,"exp":exp,"rtol":tol,"explain":case.get("explain","")})
            except Exception as e:
                failures.append({"i":i,"error": str(e), "explain":case.get("explain","")})
        return {"failures": failures} if failures else None
    except Exception as e:
        return {"error": f"runtime {e}"}

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if "\n" in s: s = s.split("\n",1)[1]
    return s.strip()

def tool_foundry(goal: str, desired_name: str, hint_context: Dict[str, Any]) -> FoundryResult:
    global TOOLS_MOD, TOOLS, DOCS, REGISTRY, TOOL_ALIASES
    spec = {"name": desired_name, "doc": "", "sig": {"args": [], "returns": "Any"}, "kind": "PURE_FUNC"}
    code = None; tests: List[Dict[str, Any]] = []; last_err: Any = None; accepted = False

    MAX_ITERS = 5
    for it in range(MAX_ITERS):
        if not spec["doc"]:
            SYSTEM = sysmsg("foundry_spec")
            USER = {"goal":goal, "desired_name":desired_name, "known_tools": list(REGISTRY.keys()), "context_hint": hint_context, "recent_context": ctx_bus_tail(10)}
            try:
                content = _strip_code_fences(_llm(SYSTEM, USER, temp=0.0, label="FOUNDRY-SPEC"))
                s = json.loads(content)
                spec["doc"] = s.get("doc","").strip(); spec["sig"]["args"] = s.get("args",[]); spec["sig"]["returns"] = s.get("returns","Any")
                spec["kind"] = _kind_from_spec(spec["doc"]) or spec["kind"]
            except Exception as e:
                last_err = f"spec-fail {e}"

        IMPL_SYS = sysmsg("foundry_impl")
        IMPL_USER = {"name": spec["name"], "spec": spec, "critic_feedback": last_err, "context_hint": hint_context}
        try:
            code_block = _strip_code_fences(_llm(IMPL_SYS, IMPL_USER, temp=0.1, label="FOUNDRY-IMPL"))
            always_ok = {"json","re","math","statistics","httpx"}
            imports = [m for m in _extract_top_imports(code_block) if m not in always_ok]
            explicit_reqs = []
            for k in ("pip", "requires", "dependencies"):
                v = (spec.get(k) or [])
                if isinstance(v, list): explicit_reqs.extend([str(x) for x in v if isinstance(x, str)])
            reqs = sorted(set(explicit_reqs + imports))
            extra_allow = set()
            if reqs:
                pip_res = ensure_pip_installed(reqs)
                ok_names = []
                for r in pip_res.get("installed", []) + pip_res.get("skipped", []):
                    name = re.split(r"[<>=!~\[]", r, 1)[0].strip().lower()
                    ok_names.append(name.replace("-", "_"))
                extra_allow = set(ok_names)

            err = safe_ast_check(code_block, kind=spec["kind"], extra_allow=extra_allow)
            if err: last_err = f"ast-unsafe: {err}"
            else:   code = textwrap.dedent(code_block).strip()
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
            last_err = run_tests_and_collect_failures(code, tests, spec["name"], spec["kind"])
        else:
            last_err = {"error":"missing code or tests"}

        if not last_err:
            accepted = True; _tests_index_update(spec["name"], tests, "ok", None); break
        _tests_index_update(spec["name"], tests, "fail", last_err)

        REV_SYS = sysmsg("foundry_revise")
        REV_USER = {"name": spec["name"], "spec": spec, "failures": last_err, "previous_code": code}
        try:
            revised = _strip_code_fences(_llm(REV_SYS, REV_USER, temp=0.1, label="FOUNDRY-REVISE"))
            err = safe_ast_check(revised, kind=spec["kind"])
            if not err: code = textwrap.dedent(revised).strip()
        except Exception:
            pass

        (FOUNDRY_LOG_DIR / f"{spec['name']}_{it}.json").write_text(json.dumps({
            "it": it, "spec": spec, "code": code, "tests": tests, "last_err": last_err
        }, indent=2))

    if not accepted:
        reason = ("no code synthesized" if code is None else "no tests produced" if not tests else (last_err if isinstance(last_err,str) else json.dumps(last_err)))
        print(_c("ERR", f"Foundry failed for {desired_name}: {reason}"))
        return FoundryResult(ok=False, name=desired_name, reason=reason, tests=tests or [], code=code)

    try:
        current = TOOLS_PATH.read_text()
        if not re.search(rf"def\\s+{re.escape(spec['name'])}\\s*\\(", current):
            with TOOLS_PATH.open("a", encoding="utf-8") as f:
                f.write("\\n\\n# ---- Auto-added by ToolFoundry ----\\n")
                f.write(code.rstrip() + "\\n")
    except Exception as e:
        return FoundryResult(ok=False, name=desired_name, reason=f"write-failed: {e}", tests=tests or [], code=code)

    print(_c("OK", f"Tool '{desired_name}' created; reloading registry ..."))
    reload_registry_and_docs()
    return FoundryResult(ok=True, name=desired_name, code=code, tests=tests)

def reload_registry_and_docs():
    global TOOLS_MOD, TOOLS, DOCS, REGISTRY, TOOL_ALIASES
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

def ensure_tools_available(plan: "Plan", registry: Dict[str, Callable[..., Any]], goal: str) -> None:
    missing = [n.tool for n in plan.nodes if resolve_tool_name(n.tool) not in registry]
    if not missing: return
    print(_c("PLAN", f"Missing tools: {', '.join(missing)}"))
    for tname in missing:
        hint = {"desired_name": tname, "nodes": [n.model_dump() for n in plan.nodes]}
        res = tool_foundry(goal, tname, hint_context=hint)
        if res.ok:
            print(_c("OK", f"Created tool '{tname}' with {len(res.tests)} tests"))
            ctx_bus_push("foundry_success", {"tool": tname})
        else:
            print(_c("ERR", f"Failed to create '{tname}': {res.reason}"))
            ctx_bus_push("foundry_failure", {"tool": tname, "reason": res.reason})

# ──────────────────────────────────────────────────────────────────────────────
# XIII. CLI ingress
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
    print("\n" + _c("RUN", "Agentic Ollama Runner — streaming + color (Reality-first, Generic)"))
    print("----------------------------------------------------------------------------")
    print(_c("RUN", f"Model: {CONFIG.model}  Host: {CONFIG.host}"))
    print("Type a goal, or press Enter for a demo. Ask for missing capabilities to see ToolFoundry spawn.\n")
    while True:
        try:
            goal = input(C.UND + "Goal> " + C.RST).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye."); break
        if not goal:
            goal = "Compute the median of [3,9,1,4,7], then slugify 'Hello World!', and save both results."
        ctx_bus_push("goal", {"text": goal}); trace("goal", {"text": goal})

        explicit = _parse_explicit_tool_request(goal)
        if explicit and resolve_tool_name(explicit) not in REGISTRY:
            print(_c("PLAN", f"Explicit tool request detected: {explicit}"))
            res = tool_foundry(goal, explicit, {"requested_by":"explicit"})
            if res.ok: print(_c("OK", f"Created '{explicit}'"))
            else: print(_c("ERR", f"Could not create '{explicit}': {res.reason}"))

        clarified = clarify_goal(goal); print("\n" + _c("AST", "[clarify]") + "\n" + clarified + "\n")
        trace("clarify", {"text": clarified})

        plan = plan_with_llm(goal, TOOLS, DOCS)
        plan = validate_and_fix_placeholders(plan)
        trace("plan_initial", plan.model_dump())

        plan = red_team_refine(goal, plan, TOOLS)
        plan = validate_and_fix_placeholders(plan)
        trace("plan_refined", plan.model_dump())

        ensure_tools_available(plan, REGISTRY, goal)
        print(_c("PLAN", "Final plan:"))
        print(json.dumps(plan.model_dump(), indent=2))

        recs, recmap = execute_plan(plan, REGISTRY)
        trace("execute_done", {"ok": all(r.error is None for r in recs)})

        if any(r.error for r in recs):
            repaired = reflect_repair(goal, TOOLS, DOCS, recs, plan)
            if repaired:
                repaired = validate_and_fix_placeholders(repaired)
                ensure_tools_available(repaired, REGISTRY, goal)
                print(_c("PLAN", "Executing repaired plan ..."))
                trace("plan_repaired", repaired.model_dump())
                recs2, recmap2 = execute_plan(repaired, REGISTRY)
                recmap.update(recmap2)

        final = assemble_final(goal, clarified, recmap)
        print("\n" + _c("OK", "[final]") + "\n" + final + "\n")
        ctx_bus_push("final", {"goal": goal, "text": final}); trace("final", {"text": final})

# ──────────────────────────────────────────────────────────────────────────────
# XIII. entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        interactive_loop()
    except Exception as e:
        print(_c("ERR", f"Unhandled exception: {e}"))
        traceback.print_exc()
