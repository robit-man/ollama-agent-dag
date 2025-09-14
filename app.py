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
HINTS_PATH = DATA_DIR / "foundry_hints.json"

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
    "- Break the problem into as many nodes as necessary (up to 100) to ensure real, verifiable results.\n"
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
  )
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
# V. default tools.py (created if missing — stdlib only; Foundry appends)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_TOOLS = '''\
"""
tools.py — Built-in tools for the agentic DAG.
Deterministic, safe, stdlib-only helpers. ToolFoundry appends new ones below.

Workspace:
- All file operations are scoped to ./data/workspace to prevent path escapes.
- Plans are saved under ./data/plans, runs under ./data/runs.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import json, re, statistics, os, sys, subprocess, time
from datetime import datetime, UTC

DATA_DIR   = Path(__file__).resolve().parent / "data"
WORK_DIR   = DATA_DIR / "workspace"
PLANS_DIR  = DATA_DIR / "plans"
RUNS_DIR   = DATA_DIR / "runs"
for d in (DATA_DIR, WORK_DIR, PLANS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)

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
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return s or "n-a"

def word_count(text: str) -> Dict[str, int]:
    "Return token counts: words, lines, characters."
    lines = text.splitlines()
    words = [w for w in re.split(r"\\s+", text.strip()) if w]
    return {"words": len(words), "lines": len(lines), "chars": len(text)}

def save_json(name: str, obj: dict) -> str:
    "Save a JSON object into ./data/<name>.json and return the written path."
    p = (DATA_DIR / f"{name}.json")
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return str(p)

class UtilityTools:
    "Miscellaneous helpers."
    @staticmethod
    def now(tz: str = "UTC") -> dict:
        iso = datetime.now(UTC).replace(microsecond=0).isoformat()
        return {"timestamp": iso}

    @staticmethod
    def now_iso(tz: str = "UTC") -> str:
        return datetime.now(UTC).replace(microsecond=0).isoformat()

    @staticmethod
    def load_json(name: str) -> dict:
        p = (DATA_DIR / f"{name}.json")
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

# ── Filesystem utilities (scoped to WORK_DIR) ────────────────────────────────
def _safe_path(rel: str) -> Path:
    """
    Resolve a relative path against WORK_DIR and prevent escaping the workspace.
    """
    base = WORK_DIR.resolve()
    p = (base / rel).resolve()
    if not str(p).startswith(str(base)):
        raise ValueError("path escapes workspace")
    return p

def ensure_dir(rel: str) -> str:
    "Create a directory (and parents) under workspace; returns absolute path."
    p = _safe_path(rel)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def file_write(path: str, text: str, mode: str = "w", mkdirs: bool = True) -> str:
    """
    Write text to a file in workspace. 'mode' is 'w' or 'a'.
    Returns absolute path.
    """
    if mode not in ("w", "a"):
        raise ValueError("mode must be 'w' or 'a'")
    p = _safe_path(path)
    if mkdirs:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open(mode, encoding="utf-8") as f:
        f.write(text)
    return str(p)

def file_append(path: str, text: str) -> str:
    "Append text to a file in workspace; returns absolute path."
    return file_write(path, text, mode="a", mkdirs=True)

def file_read(path: str, max_bytes: Optional[int] = None) -> Dict[str, Any]:
    """
    Read text from a file in workspace. Optionally cap bytes read.
    Returns {'text', 'truncated', 'size'}.
    """
    p = _safe_path(path)
    data = p.read_bytes()
    size = len(data)
    truncated = False
    if max_bytes is not None and size > max_bytes:
        data = data[:max_bytes]
        truncated = True
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        text = ""
        truncated = True
    return {"text": text, "truncated": truncated, "size": size}

def file_delete(path: str) -> bool:
    "Delete a file under workspace. Returns True if removed, False if missing."
    p = _safe_path(path)
    if p.exists() and p.is_file():
        p.unlink()
        return True
    return False

def dir_list(path: str = ".", recursive: bool = False, pattern: Optional[str] = None, files_only: bool = False) -> List[Dict[str, Any]]:
    """
    List directory contents under workspace.
    Returns list of {path, is_dir, size, mtime}.
    'pattern' uses glob semantics (e.g., '*.md')
    """
    base = _safe_path(path)
    if not base.exists():
        return []
    if recursive:
        it = base.rglob(pattern or "*")
    else:
        it = base.glob(pattern or "*")
    out: List[Dict[str, Any]] = []
    for p in it:
        if files_only and not p.is_file():
            continue
        try:
            stat = p.stat()
            out.append({
                "path": str(p.relative_to(WORK_DIR)),
                "is_dir": p.is_dir(),
                "size": 0 if p.is_dir() else int(stat.st_size),
                "mtime": datetime.fromtimestamp(stat.st_mtime, UTC).replace(microsecond=0).isoformat()
            })
        except Exception:
            continue
    return out

# ── Planning document helpers (Markdown) ─────────────────────────────────────
def plan_markdown(name: str, goal: str, steps: List[str], notes: str = "", tags: Optional[List[str]] = None) -> str:
    """
    Create a structured Markdown planning doc in ./data/plans.
    Returns absolute path to the file.
    """
    slug = slugify(name)
    ts = datetime.now(UTC).replace(microsecond=0).isoformat()
    p = (PLANS_DIR / f"{slug}.md")
    steps_md = "\\n".join(f"- {s}" for s in steps)
    tags = tags or []
    body = f"""---
title: {name}
created: {ts}
tags: {tags}
---

# Goal
{goal}

# Plan
{steps_md}

# Notes
{notes}
"""
    p.write_text(body, encoding="utf-8")
    return str(p)

# ── Run Python files with full logging ───────────────────────────────────────
def run_python_file(path: str, args: Optional[List[str]] = None, log_prefix: Optional[str] = None, timeout_s: int = 300) -> Dict[str, Any]:
    """
    Execute a Python file within current interpreter, capture stdout/stderr, and log to ./data/runs/<stamp>/.
    Returns dict with exit_code, duration_ms, stdout_tail, stderr_tail, stdout_path, stderr_path, meta_path, log_dir.
    """
    target = _safe_path(path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError("target python file not found in workspace")
    args = args or []
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = slugify(log_prefix or target.stem)
    run_dir = RUNS_DIR / f"{base}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(target), *[str(a) for a in args]]
    start = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, text=True)
        rc = int(proc.returncode)
        out = proc.stdout or ""
        err = proc.stderr or ""
    except subprocess.TimeoutExpired as te:
        rc = -1
        out, err = "", f"TimeoutExpired after {timeout_s}s: {te}"
    dur_ms = int((time.time() - start) * 1000)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    meta_path = run_dir / "meta.json"
    stdout_path.write_text(out, encoding="utf-8")
    stderr_path.write_text(err, encoding="utf-8")
    meta = {
        "cmd": cmd,
        "started": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "duration_ms": dur_ms,
        "exit_code": rc,
        "target": str(target),
        "args": args
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "exit_code": rc,
        "duration_ms": dur_ms,
        "stdout_tail": out[-2000:],
        "stderr_tail": err[-2000:],
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "meta_path": str(meta_path),
        "log_dir": str(run_dir)
    }
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
# VII. tool discovery + docs generation + aliases
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

def _chat_stream_and_collect(messages: List[Dict[str, str]], label: str = "LLM", options: Optional[Dict[str, Any]] = None) -> str:
    opts = (CONFIG.options or {}).copy()
    if options: opts.update(options)
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

def ensure_docs(tools: List[ToolSpec], force: bool=False) -> Dict[str, Any]:
    if DOCS_PATH.exists() and not force:
        try: return jload(DOCS_PATH, {})
        except Exception: pass
    payload = [{"name": t.name, "origin": t.origin, "doc": t.doc, "args": [a.model_dump() for a in t.args], "returns": t.returns} for t in tools]
    SYSTEM = (
        "You are ToolDocBot.\n"
        "Given Python tools (docstrings, arg types/defaults, returns), output STRICT JSON mapping tool name → doc "
        "with keys {purpose, when_to_use, args:{<arg>:desc}, returns, example}. No prose outside JSON."
    )
    USER = {"instruction":"Write JSON docs for these tools.","tools_introspection":payload}
    print(_c("PLAN", "Generating tool docs (streaming) ..."))
    try:
        msg = [{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps(USER, ensure_ascii=False)}]
        if DEBUG: _dbg("TOOLDOCS", msg)
        content = _chat_stream_and_collect(msg, label="TOOLDOCS")
        docs = json.loads(content)
    except Exception as e:
        print(_c("ERR", f"Tool docs generation failed, falling back to introspection-only. {e}"))
        docs = { t.name: {"purpose":(t.doc or f"{t.name} tool"),"when_to_use":"When its purpose matches your need.",
                           "args":{a.name:f"{a.type}" for a in t.args},"returns":t.returns,
                           "example":f"{t.name}({', '.join(a.name for a in t.args)})"} for t in tools }
    jdump(DOCS_PATH, docs); print(_c("OK", f"Wrote {DOCS_PATH.name} with {len(docs)} entries."))
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
# VIII. DAG models + strict plan parsing + placeholder/dep repair
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
        if len(parts) >= 2 and parts[1].startsWith("output"):  # fix later if needed
            pass
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
# VIII.a Streaming LLM wrapper
# ──────────────────────────────────────────────────────────────────────────────
def _llm(system: str, user_obj: Any, temp=0.1, preface: Optional[str]=None, label: str="LLM") -> str:
    msgs = [{"role":"system","content":system}]
    if INTERLEAVE and preface:
        msgs.append({"role":"assistant","content":preface})
    msgs.append({"role":"user","content":json.dumps(user_obj, ensure_ascii=False)})
    _emit_chat_preview(system, user_obj, preface, label)
    if DEBUG: _dbg(label, msgs)
    return _chat_stream_and_collect(msgs, label=label, options={"temperature":temp})

# ──────────────────────────────────────────────────────────────────────────────
# IX. Concurrency executor with live color logs
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
                    print(_c("PLAN", f"RETRY {node.id} remaining={retry_counts[node.id]}"))
                else:
                    if node.id in state.pending:
                        state.pending.remove(node.id)
                    state.completed.append(node.id)

    ordered = [records[nid] for nid in [n.id for n in plan.nodes] if nid in records]
    return ordered, records

# ──────────────────────────────────────────────────────────────────────────────
# X. Reality policy + LLM stages (clarify, plan, red-team, merge, reflect, assemble)
# ──────────────────────────────────────────────────────────────────────────────
ANTI_SIM_PHRASES = re.compile(r"\b(simulat(?:e|ion)|placeholder|null(?:\s+values?)?|doesn'?t actually|mock(?:ed|ing)?|fake|dummy)\b", re.IGNORECASE)
def response_smells_simulated(text: str) -> bool: return bool(ANTI_SIM_PHRASES.search(text))

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
    # fallback seed
    seeded = Plan(nodes=[
        Node(id="n1", tool="word_count", args={"text": goal}),
        Node(id="n2", tool="save_json", args={"name":"last_word_count","obj":"[n1.output]"}, after=["n1"]),
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
        "tools": [{"name": t.name, "args":[a.name for a in t.args]} for t in tools], "docs": docs,
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
# XI. ToolFoundry (iterative + pip auto-install + AST safety + tests index)
#      + Adaptive testing prompts + offline NET tests + hint memory
# ──────────────────────────────────────────────────────────────────────────────
CAPS = {
    "PURE_FUNC": {"allowed_imports": {"math","re","json","statistics"}, "io": False, "net": False},
    "NET_TOOL":  {"allowed_imports": {"httpx","json","re"},            "io": False, "net": True},
    "IO_TOOL":   {"allowed_imports": {"json","re"},                    "io": True,  "net": False},
}

PIP_STATE_PATH = DATA_DIR / "pip_state.json"
PIP_LOCK_PATH  = DATA_DIR / "pip.lock"

def ensure_pip_installed(requirements: List[str], timeout_s: int = 300) -> Dict[str, Any]:
    """
    Ensure the given pip requirement strings are installed into THIS venv.
    Returns: {"ok": bool, "installed": [str], "skipped": [str], "errors": [{"req": str, "err": str}]}
    """
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
            # Relaxed: allow Raise/Try/Lambda; ban async/with/global/nonlocal/await/yield.
            if isinstance(node, (ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal, ast.Await, ast.Yield, ast.YieldFrom)):
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
        return all(_deep_equal(a[k], b[k], rtol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_equal(x, y, rtol) for x, y in zip(a, b))
    return a == b

# Adaptive testing hints
def _load_hints() -> Dict[str, Any]:
    try:
        return json.loads(HINTS_PATH.read_text())
    except Exception:
        return {}

def _save_hints(obj: Dict[str, Any]) -> None:
    HINTS_PATH.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def _categorize_failures(fail: Any) -> List[str]:
    tags = set()
    if not fail:
        return []
    if isinstance(fail, dict) and "error" in fail and not fail.get("failures"):
        msg = str(fail.get("error","")).lower()
        if "ast-unsafe" in msg: tags.add("ast_unsafe")
        if "network" in msg or "httpx" in msg: tags.add("net_in_tests")
        if "syntaxerror" in msg: tags.add("syntax")
        if "timeout" in msg: tags.add("timeout")
        return sorted(tags)
    for f in fail.get("failures", []):
        em = str(f.get("error") or f.get("explain") or "").lower()
        if "typeerror" in em: tags.add("type_mismatch")
        if "keyerror" in em or "missing path" in em: tags.add("key_path")
        if "valueerror" in em or "domain" in em: tags.add("domain")
        if "network" in em or "httpx" in em: tags.add("net_in_tests")
        if "rtol" in em or "precision" in em: tags.add("precision")
        if "json" in em and "decode" in em: tags.add("json_decode")
        if "division by zero" in em: tags.add("zero_div")
        if "indexerror" in em: tags.add("index_bounds")
        if "assert" in em or "expected" in em: tags.add("assertions")
    return sorted(tags)

def _summarize_code_ast(code: Optional[str]) -> Dict[str, Any]:
    if not code:
        return {"has_doc": False, "num_args": 0, "raises": [], "has_fetcher": False, "branches": 0}
    try:
        tree = ast.parse(code)
        raises, args_cnt, fetcher, branches = set(), 0, False, 0
        class V(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef):
                nonlocal args_cnt, fetcher
                args_cnt = max(args_cnt, len([a for a in node.args.args]))
                for a in node.args.args:
                    if a.arg == "fetcher": fetcher = True
                self.generic_visit(node)
            def visit_Raise(self, node: ast.Raise):
                if node.exc and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    raises.add(node.exc.func.id)
                self.generic_visit(node)
            def visit_If(self, node: ast.If):
                nonlocal branches
                branches += 1
                self.generic_visit(node)
        V().visit(tree)
        has_doc = isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str)
        return {"has_doc": bool(has_doc), "num_args": int(args_cnt), "raises": sorted(raises), "has_fetcher": bool(fetcher), "branches": int(branches)}
    except Exception:
        return {"has_doc": False, "num_args": 0, "raises": [], "has_fetcher": False, "branches": 0}

_ADAPT_HINTS = {
    "assertions": "Add explicit expected outputs and invariants; include at least one negative test.",
    "precision": "Include a floating-point comparison case with strict rtol=1e-9 and another relaxed rtol (e.g., 1e-6).",
    "type_mismatch": "Add tests with wrong types (e.g., str for numeric) expecting a clear ValueError or TypeError.",
    "key_path": "Add tests for missing keys / paths and assert a helpful error or fallback behavior.",
    "json_decode": "Include a malformed JSON payload test and assert a clean error path.",
    "index_bounds": "Test empty and single-element sequences plus out-of-range indices.",
    "zero_div": "Include a zero-div case and assert the chosen behavior (raise ValueError vs. return None).",
    "domain": "Add domain boundary tests (min/max, empty input, None).",
    "ast_unsafe": "Reduce features: avoid disallowed imports or dunder usage.",
    "syntax": "Ensure valid Python syntax; import only allowed modules.",
    "timeout": "Avoid slow operations; tests must terminate quickly.",
    "net_in_tests": "For NET_TOOL, ALWAYS pass a stub fetcher(url) that returns deterministic .json(). Do not do real HTTP in tests."
}

def _compose_adaptive_test_system(base_system: str, tags: List[str], code_summary: Dict[str, Any], tool_kind: str) -> str:
    hints = []
    for t in tags:
        h = _ADAPT_HINTS.get(t)
        if h: hints.append(f"- {h}")
    if tool_kind == "NET_TOOL" and not code_summary.get("has_fetcher", False):
        hints.append("- Ensure the function accepts an optional `fetcher(url)` for dependency injection in tests.")
    if code_summary.get("branches", 0) >= 2:
        hints.append("- Cover both the 'happy path' and each conditional branch with separate test cases.")
    if not code_summary.get("has_doc", False):
        hints.append("- Write tests that verify documented behavior; if missing docstring, infer and validate key behaviors.")
    if not hints:
        return base_system
    adaptive_section = (
        "ADAPTIVE TESTING GUIDANCE:\n"
        "Incorporate the following requirements when generating tests for this tool:\n" +
        "\n".join(hints) +
        "\nRespond with STRICT JSON array of test cases only."
    )
    return base_system + "\n\n" + adaptive_section

def _bump_hint_counters(tool_name: str, tags: List[str]) -> None:
    db = _load_hints()
    ent = db.setdefault(tool_name, {"counts": {}, "last_tags": []})
    for t in tags:
        ent["counts"][t] = int(ent["counts"].get(t, 0)) + 1
        ent["last_tags"] = tags
    _save_hints(db)

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
        if kind == "NET_TOOL":
            class _HttpxNoNet:
                @staticmethod
                def get(*args, **kwargs):
                    raise RuntimeError("Network disabled during tests; provide a stub `fetcher(url)`.")
            safe_globals["httpx"] = _HttpxNoNet
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

def tool_foundry(goal: str, desired_name: str, hint_context: Dict[str, Any]) -> "FoundryResult":
    global TOOLS_MOD, TOOLS, DOCS, REGISTRY, TOOL_ALIASES
    class FoundryResult(BaseModel):
        ok: bool; name: str; code: Optional[str]=None
        tests: List[Dict[str, Any]]=Field(default_factory=list); reason: Optional[str]=None

    spec = {"name": desired_name, "doc": "", "sig": {"args": [], "returns": "Any"}, "kind": "PURE_FUNC"}
    code = None; tests: List[Dict[str, Any]] = []; last_err: Any = None; accepted = False

    MAX_ITERS = 5
    for it in range(MAX_ITERS):
        # SPEC
        if not spec["doc"]:
            SYSTEM = sysmsg("foundry_spec")
            USER = {
                "goal":goal,
                "desired_name":desired_name,
                "known_tools": list(REGISTRY.keys()),
                "context_hint": hint_context,
                "recent_context": ctx_bus_tail(10),
                "observed_failures": _categorize_failures(last_err),
            }
            try:
                content = _strip_code_fences(_llm(SYSTEM, USER, temp=0.0, label="FOUNDRY-SPEC"))
                s = json.loads(content)
                spec["doc"] = s.get("doc","").strip(); spec["sig"]["args"] = s.get("args",[]); spec["sig"]["returns"] = s.get("returns","Any")
                spec["kind"] = _kind_from_spec(spec["doc"]) or spec["kind"]
            except Exception as e:
                last_err = f"spec-fail {e}"

        # IMPL
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
            if err: last_err = {"error": f"ast-unsafe: {err}"}
            else:   code = textwrap.dedent(code_block).strip()
        except Exception as e:
            last_err = {"error": f"impl-fail {e}"}

        # TEST (Adaptive)
        base_test_sys = sysmsg("foundry_test")
        last_tags = _categorize_failures(last_err)
        _bump_hint_counters(spec["name"], last_tags)
        code_summary = _summarize_code_ast(code)
        TEST_SYS = _compose_adaptive_test_system(base_test_sys, last_tags, code_summary, spec.get("kind","PURE_FUNC"))

        TEST_USER = {
            "name": spec["name"],
            "spec": spec,
            "recent_error": last_err,
            "code_summary": code_summary,
            "policy": {"offline_tests_for_net_tools": True}
        }
        try:
            content = _strip_code_fences(_llm(TEST_SYS, TEST_USER, temp=0.10, label="FOUNDRY-TEST"))
            cand_tests = json.loads(content)
            if isinstance(cand_tests, list) and cand_tests:
                tests = cand_tests
        except Exception as e:
            last_err = {"error": f"test-fail {e}"}

        if DEBUG:
            print(_c("DBG", f"[adapt] {spec['name']} failure tags: {last_tags}  summary: {code_summary}"))

        # CRITIC
        if code and tests:
            last_err = run_tests_and_collect_failures(code, tests, spec["name"], spec["kind"])
        else:
            last_err = {"error":"missing code or tests"}

        if not last_err:
            accepted = True; _tests_index_update(spec["name"], tests, "ok", None); break
        _tests_index_update(spec["name"], tests, "fail", last_err)

        # REVISION
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

    # Append to tools.py (idempotent)
    try:
        current = TOOLS_PATH.read_text()
        if not re.search(rf"def\s+{re.escape(spec['name'])}\s*\(", current):
            with TOOLS_PATH.open("a", encoding="utf-8") as f:
                f.write("\n\n# ---- Auto-added by ToolFoundry ----\n")
                f.write(code.rstrip() + "\n")
    except Exception as e:
        return FoundryResult(ok=False, name=desired_name, reason=f"write-failed: {e}", tests=tests or [], code=code)

    # Reload registry & docs & aliases
    print(_c("OK", f"Tool '{desired_name}' created; reloading registry ..."))
    reload_registry_and_docs()
    return FoundryResult(ok=True, name=desired_name, code=code, tests=tests)

class FoundryResult(BaseModel):
    ok: bool; name: str; code: Optional[str]=None
    tests: List[Dict[str, Any]]=Field(default_factory=list); reason: Optional[str]=None

def _kind_from_spec(spec_doc: str) -> str:
    s = (spec_doc or "").lower()
    if any(k in s for k in ["http","api","network","fetch","request","url"]): return "NET_TOOL"
    if any(k in s for k in ["file","load","save","read","write","disk","json file"]): return "IO_TOOL"
    return "PURE_FUNC"

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
# XII. CLI ingress
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
