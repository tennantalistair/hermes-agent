"""
Hermes Worker — Paperclip Unified Multi-Agent Integration

One worker handles ALL Hermes-type agents across ALL companies.
Polls Paperclip's heartbeat_runs table, resolves per-agent config + API keys,
executes the issue (clone repo, run agent, create PR), writes results back.

Entry point (Render):
    python supabase_paperclip_worker.py

Required env vars:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    GITHUB_TOKEN                 # GitHub PAT for cloning/pushing/PRs

Optional:
    PAPERCLIP_POLL_INTERVAL_SECONDS   (default: 3)
    PAPERCLIP_MAX_RUNS_PER_PROCESS    (default: unlimited)
    LOG_LEVEL                          (default: INFO)
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================

@dataclass
class WorkerConfig:
    supabase_url: str
    supabase_service_role_key: str
    github_token: str
    poll_interval_seconds: float = 3.0
    max_runs_per_process: Optional[int] = None
    # Reusable clone cache: avoid re-cloning the same repo within one container
    _clone_cache: Dict[str, Path] = field(default_factory=dict)


def _load_config_from_env() -> WorkerConfig:
    def req(name: str) -> str:
        v = os.getenv(name, "").strip()
        if not v:
            raise RuntimeError(f"Missing required env var: {name}")
        return v

    return WorkerConfig(
        supabase_url=req("SUPABASE_URL"),
        supabase_service_role_key=req("SUPABASE_SERVICE_ROLE_KEY"),
        github_token=req("GITHUB_TOKEN"),
        poll_interval_seconds=float(os.getenv("PAPERCLIP_POLL_INTERVAL_SECONDS", "3")),
        max_runs_per_process=(
            int(os.getenv("PAPERCLIP_MAX_RUNS_PER_PROCESS"))
            if os.getenv("PAPERCLIP_MAX_RUNS_PER_PROCESS")
            else None
        ),
    )


# =============================================================================
# Supabase client
# =============================================================================

def _create_supabase_client(cfg: WorkerConfig):
    from supabase import create_client
    return create_client(cfg.supabase_url, cfg.supabase_service_role_key)


def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return {"_unserializable": str(type(obj))}


def _get_data(res) -> List[Dict[str, Any]]:
    """Extract rows from supabase-py response (handles both object and dict forms)."""
    if hasattr(res, "data") and res.data is not None:
        return res.data
    if isinstance(res, dict):
        return res.get("data") or []
    return []


# =============================================================================
# Paperclip data access — unified multi-agent
# =============================================================================

def _select_next_runs(supabase) -> List[Dict[str, Any]]:
    """
    Fetch ALL queued runs for Hermes-type agents.
    Filters to adapter_type IN ('hermes', 'hermes_local').
    Returns runs across all companies and all agent roles.
    """
    # First, get all Hermes agent IDs
    agent_res = (
        supabase.table("agents")
        .select("id, adapter_type, company_id, role, name, runtime_config, capabilities")
        .in_("adapter_type", ["hermes", "hermes_local"])
        .execute()
    )
    agents_rows = _get_data(agent_res)
    if not agents_rows:
        return []

    hermes_agent_ids = [a["id"] for a in agents_rows]
    agent_map = {a["id"]: a for a in agents_rows}

    # Fetch queued runs for these agents
    res = (
        supabase.table("heartbeat_runs")
        .select("*")
        .in_("agent_id", hermes_agent_ids)
        .eq("status", "queued")
        .order("created_at", desc=False)
        .limit(10)  # batch up to 10 at a time
        .execute()
    )
    runs = _get_data(res)

    # Attach agent info to each run
    for run in runs:
        agent = agent_map.get(run.get("agent_id"), {})
        run["_agent"] = agent

    return runs


def _claim_run(supabase, run_id: str) -> bool:
    """Atomic claim: update queued → running."""
    res = (
        supabase.table("heartbeat_runs")
        .update({"status": "running", "started_at": "now()"})
        .eq("id", run_id)
        .eq("status", "queued")
        .execute()
    )
    rows = _get_data(res)
    return bool(rows)


def _fetch_issue_context(supabase, run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Walk the Paperclip join chain for this specific run.
    Returns flat context dict for execution.
    """
    run_id = run["id"]
    agent = run.get("_agent", {})

    # Find the issue via execution_run_id
    res = (
        supabase.table("issues")
        .select(
            "id, title, description, project_workspace_id, "
            "company_id, project_id, assignee_agent_id"
        )
        .eq("execution_run_id", run_id)
        .limit(1)
        .execute()
    )
    issue_rows = _get_data(res)
    if not issue_rows:
        return {"_error": "No issue found for this run"}
    issue = issue_rows[0]

    workspace_id = issue.get("project_workspace_id")
    repo_url = None
    cwd = None
    repo_ref = "main"

    if workspace_id:
        ws_res = (
            supabase.table("project_workspaces")
            .select("repo_url, cwd, repo_ref, default_ref")
            .eq("id", workspace_id)
            .limit(1)
            .execute()
        )
        ws_rows = _get_data(ws_res)
        if ws_rows:
            ws = ws_rows[0]
            repo_url = ws.get("repo_url")
            cwd = ws.get("cwd") or ""
            repo_ref = ws.get("repo_ref") or ws.get("default_ref") or "main"

    # Fetch company info
    company_id = issue.get("company_id")
    company_name = None
    if company_id:
        comp_res = (
            supabase.table("companies")
            .select("name")
            .eq("id", company_id)
            .limit(1)
            .execute()
        )
        comp_rows = _get_data(comp_res)
        if comp_rows:
            company_name = comp_rows[0].get("name")

    return {
        "issue_id": issue["id"],
        "issue_title": issue.get("title", ""),
        "issue_description": issue.get("description", ""),
        "company_id": company_id,
        "company_name": company_name,
        "project_id": issue.get("project_id"),
        "repo_url": repo_url,
        "repo_ref": repo_ref,
        "cwd": cwd,
        "agent_id": run.get("agent_id"),
        "agent_role": agent.get("role"),
        "agent_name": agent.get("name"),
        "agent_runtime_config": agent.get("runtime_config") or {},
        "agent_capabilities": agent.get("capabilities"),
        "heartbeat_run_id": run_id,
        "run_context": run.get("context_snapshot") or {},
    }


def _fetch_agent_api_key(supabase, agent_id: str, company_id: str) -> Optional[str]:
    """Look up per-agent per-company API key for LLM billing."""
    res = (
        supabase.table("agent_api_keys")
        .select("token")
        .eq("agent_id", agent_id)
        .eq("company_id", company_id)
        .limit(1)
        .execute()
    )
    rows = _get_data(res)
    if rows:
        return rows[0].get("token")
    # Fallback: any key for this agent
    res = (
        supabase.table("agent_api_keys")
        .select("token")
        .eq("agent_id", agent_id)
        .limit(1)
        .execute()
    )
    rows = _get_data(res)
    return rows[0].get("token") if rows else None


# =============================================================================
# Git helpers
# =============================================================================

def _auth_git_url(repo_url: str, token: str) -> str:
    if not token:
        return repo_url
    if repo_url.startswith("https://github.com/"):
        return repo_url.replace("https://", f"https://{token}@")
    return repo_url


def _clone_repo(repo_url: str, ref: str, github_token: str, cache: Dict[str, Path]) -> Path:
    """Clone or reuse from cache."""
    cache_key = f"{repo_url}@{ref}"
    if cache_key in cache:
        logger.info("Reusing cached clone of %s @ %s", repo_url, ref)
        return cache[cache_key]

    authed_url = _auth_git_url(repo_url, github_token)
    tmp = Path(tempfile.mkdtemp(prefix="hermes_"))
    clone_dest = tmp / "repo"

    logger.info("Cloning %s (ref=%s) → %s", repo_url, ref, clone_dest)
    _run(["git", "clone", "--quiet", "--depth=1", authed_url, str(clone_dest)])
    _run(["git", "-C", str(clone_dest), "fetch", "--quiet", "origin", ref])
    _run(["git", "-C", str(clone_dest), "checkout", "--quiet", ref])

    cache[cache_key] = clone_dest
    return clone_dest


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        if result.returncode != 0:
            logger.warning("Command %s exited %d: %s", cmd, result.returncode, result.stderr.strip())
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.error("Command timed out after %ds: %s", timeout, cmd)
        raise


def _create_pr(repo_url: str, branch: str, title: str, body: str, github_token: str) -> str:
    """Create a GitHub PR via API. Returns PR URL."""
    parts = repo_url.rstrip("/").split("/")
    owner_repo = "/".join(parts[-2:]).replace(".git", "")

    import urllib.request

    url = f"https://api.github.com/repos/{owner_repo}/pulls"
    payload = json.dumps({
        "title": title,
        "head": branch,
        "base": "main",
        "body": body,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("html_url", "")


# =============================================================================
# Hermes execution — per-agent config
# =============================================================================

def _run_hermes_agent(
    repo_path: Path,
    cwd: str,
    prompt: str,
    runtime_config: Dict[str, Any],
    github_token: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Hermes with per-agent config from runtime_config.

    runtime_config fields supported:
      - model: str (e.g. "anthropic/claude-opus-4.6")
      - max_turns: int (default 30)
      - skills: List[str] (optional skill names)
      - provider: str (optional override)
    """
    work_dir = (repo_path / cwd) if cwd else repo_path
    if not work_dir.exists():
        work_dir = repo_path

    model = runtime_config.get("model", "")
    max_turns = int(runtime_config.get("max_turns", 30))
    provider = runtime_config.get("provider", "")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "hermes"),
        "chat",
        "--max-turns", str(max_turns),
        "-q",
        "--",
        prompt,
    ]
    if model:
        cmd.extend(["-m", model])
    if provider:
        cmd.extend(["--provider", provider])

    env = {**os.environ}
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key

    logger.info(
        "Running Hermes (model=%s, turns=%d, agent_role=%s) in %s",
        model or "default", max_turns,
        runtime_config.get("role", "unknown"),
        work_dir,
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(work_dir),
            timeout=900,  # 15 min max per run
            env=env,
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout or "",
            "stderr": (result.stderr or "")[-500:],
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "Hermes timed out after 900 seconds",
        }


# =============================================================================
# Execute a single run
# =============================================================================

def _execute_run(
    ctx: Dict[str, Any], cfg: WorkerConfig, supabase
) -> Dict[str, Any]:
    """Execute one Paperclip issue with the resolved agent + company config."""
    repo_url = ctx.get("repo_url")
    if not repo_url:
        return {"error": "No repo_url found in issue context"}

    repo_ref = ctx.get("repo_ref", "main")
    cwd = ctx.get("cwd", "")
    issue_title = ctx.get("issue_title", "Hermes task")
    issue_description = ctx.get("issue_description", "")
    company_id = ctx.get("company_id")
    agent_id = ctx.get("agent_id")
    runtime_config = ctx.get("agent_runtime_config", {})
    company_name = ctx.get("company_name") or "unknown"

    prompt = issue_description.strip() if issue_description.strip() else issue_title

    # Clone repo
    try:
        repo_path = _clone_repo(repo_url, repo_ref, cfg.github_token, cfg._clone_cache)
    except Exception as exc:
        return {"error": f"Clone failed: {exc}"}

    # Per-company, per-agent API key
    api_key = None
    if agent_id and company_id:
        try:
            api_key = _fetch_agent_api_key(supabase, agent_id, company_id)
        except Exception as exc:
            logger.warning("Could not fetch agent API key: %s", exc)

    # Run Hermes with agent's runtime config
    result = _run_hermes_agent(
        repo_path, cwd, prompt, runtime_config, cfg.github_token, api_key
    )

    pr_url = None
    branch_name = f"hermes-{uuid.uuid4().hex[:8]}"

    if result.get("exit_code") == 0:
        try:
            body = (
                f"**Hermes Task Result**\n"
                f"Company: {company_name}\n"
                f"Agent Role: {ctx.get('agent_role', 'unknown')}\n\n"
                f"```\n{result.get('stdout', '')}\n```"
            )
            pr_url = _create_pr(
                repo_url, branch_name,
                f"Hermes [{company_name}]: {issue_title[:50]}",
                body[:65536],
                cfg.github_token,
            )
        except Exception as exc:
            logger.warning("PR creation failed: %s", exc)

    return {
        "repo_url": repo_url,
        "repo_ref": repo_ref,
        "pr_url": pr_url,
        "branch": branch_name,
        "company": company_name,
        "agent_role": ctx.get("agent_role"),
        "hermes_result": result,
    }


def _update_run(
    supabase, run_id: str, status: str, result: Dict[str, Any], *, error: str = ""
):
    patch: Dict[str, Any] = {
        "status": status,
        "finished_at": "now()",
    }
    if status == "completed":
        patch["result_json"] = _safe_json(result)
        patch["exit_code"] = 0
    else:
        patch["result_json"] = _safe_json({"error": error})
        patch["exit_code"] = 1
        patch["error"] = str(error)

    supabase.table("heartbeat_runs").update(patch).eq("id", run_id).execute()


def _post_issue_comment(supabase, issue_id: str, body: str):
    if not issue_id:
        return
    try:
        supabase.table("issue_comments").insert({
            "issue_id": issue_id,
            "body": body,
        }).execute()
    except Exception as exc:
        logger.debug("Could not post issue comment: %s", exc)


# =============================================================================
# Main worker loop
# =============================================================================

def run_forever():
    logging.basicConfig(
        level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_config_from_env()
    supabase = _create_supabase_client(cfg)

    processed = 0
    logger.info("Paperclip unified worker starting — polling every %.1fs", cfg.poll_interval_seconds)

    while True:
        if cfg.max_runs_per_process and processed >= cfg.max_runs_per_process:
            logger.info("Reached PAPERCLIP_MAX_RUNS_PER_PROCESS=%s, exiting", cfg.max_runs_per_process)
            return

        try:
            runs = _select_next_runs(supabase)
            if not runs:
                time.sleep(cfg.poll_interval_seconds)
                continue

            for run in runs:
                run_id = run["id"]
                agent_name = run.get("_agent", {}).get("name", "?")
                company = run.get("_agent", {}).get("company_id", "")[:8]

                claimed = _claim_run(supabase, run_id)
                if not claimed:
                    logger.debug("Run %s already claimed by another worker", run_id[:8])
                    continue

                logger.info(
                    "Claimed run %s — agent=%s company=%s",
                    run_id[:8], agent_name, company,
                )

                ctx = _fetch_issue_context(supabase, run)
                error = ctx.get("_error")

                if error:
                    logger.warning("Skipping run %s: %s", run_id[:8], error)
                    _update_run(supabase, run_id, "failed", {}, error=error)
                    processed += 1
                    continue

                try:
                    result = _execute_run(ctx, cfg, supabase)
                except Exception as exc:
                    logger.exception("Execution failed for run %s", run_id[:8])
                    result = {"error": str(exc)}

                status = "completed" if "error" not in result else "failed"
                _update_run(supabase, run_id, status, result, error=result.get("error", ""))

                # Post result comment on issue
                issue_id = ctx.get("issue_id")
                if issue_id:
                    pr = result.get("pr_url")
                    if status == "completed" and pr:
                        msg = f"✅ Hermes completed task — [View PR]({pr})"
                    elif status == "completed":
                        msg = "✅ Hermes completed task"
                    else:
                        msg = f"⚠️ Hermes task failed: {result.get('error', 'unknown error')}"
                    _post_issue_comment(supabase, issue_id, msg)

                processed += 1

        except Exception:
            logger.exception("Worker loop error — retrying")
            time.sleep(max(1.0, cfg.poll_interval_seconds))


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path, override=True)
        except ImportError:
            pass

    run_forever()
