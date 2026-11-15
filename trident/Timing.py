"""Optional bridge from TRIDENT runs to the shared embedding timing report."""
from __future__ import annotations

import os
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Any, Optional

_reports = {}
_warned_unavailable = False


def _report_class():
    """Load the workspace report module when available; timing must stay optional."""
    try:
        from mimyc.embedding_reports import RunReport
        return RunReport
    except Exception:
        pass
    # Source checkouts of TRIDENT can be run without installing the workspace package.
    for candidate in (os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")), "/opt/project/src"):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            from mimyc.embedding_reports import RunReport
            return RunReport
        except Exception:
            continue
    return None


def create_report(job_dir: str, run_id: str, *, backend: str = "trident", settings: Optional[dict] = None):
    global _warned_unavailable
    cls = _report_class()
    if cls is None:
        if not _warned_unavailable:
            warnings.warn("Shared embedding timing report is unavailable; TRIDENT processing will continue without timing output.", RuntimeWarning)
            _warned_unavailable = True
        return None
    try:
        task = (settings or {}).get("task")
        required = {"seg": ["segmentation"], "coords": ["coords"], "feat": ["patch_features"], "all": ["segmentation", "coords", "patch_features"]}.get(task)
        if task == "feat" and (settings or {}).get("slide_encoder"):
            required = ["patch_features", "slide_features"]
        if task == "all" and (settings or {}).get("slide_encoder"):
            required = ["segmentation", "coords", "patch_features", "slide_features"]
        if task is None:
            required = ["segmentation", "coords", "patch_features"]
        if run_id in _reports:
            return _reports[run_id]
        report = cls(output_dir=job_dir, backend=backend, settings=settings or {}, required_stages=required, run_id=run_id)
        _reports[run_id] = report
        return report
    except Exception as exc:
        warnings.warn(f"Could not initialize timing report: {exc}", RuntimeWarning)
        return None


def get_report(run_id: Optional[str]):
    if not run_id:
        return None
    return _reports.get(str(run_id))


def register_device(run_id: Optional[str], device: Any) -> None:
    report = get_report(run_id)
    if report is None:
        return
    try:
        report.register_worker_device(os.getpid(), device)
    except Exception:
        pass
    try:
        _load_sync_cuda()(device)
    except Exception:
        pass


def _load_sync_cuda():
    try:
        from mimyc.embedding_reports import sync_cuda
        return sync_cuda
    except Exception:
        _report_class()
        from mimyc.embedding_reports import sync_cuda
        return sync_cuda


def record_stage(run_id: Optional[str], slide: str, stage: str, status: str,
                 started: Optional[float] = None, *, scope: str = "", patch_count=None,
                 extra=None, device=None) -> None:
    report = get_report(run_id)
    if report is None:
        return
    try:
        register_device(run_id, device) if device is not None else None
        elapsed = max(0.0, time.monotonic() - started) if started is not None else None
        report_status = "cached" if status == "skipped" and extra and extra.get("reason") in {"already_segmented", "already_generated", "already_extracted"} else status
        report.record(slide, stage, report_status, duration_s=elapsed, scope=scope,
                      patch_count=patch_count, extra=extra)
    except Exception:
        # Reports are auxiliary and cannot change TRIDENT's extraction behavior.
        pass


@contextmanager
def timed_stage(run_id: Optional[str], slide: str, stage: str, *, scope: str = "", device=None, patch_count=None):
    register_device(run_id, device)
    started = time.monotonic()
    status = "completed"
    try:
        yield
    except BaseException as exc:
        status = "error"
        raise
    finally:
        record_stage(run_id, slide, stage, status, started, scope=scope,
                     patch_count=patch_count, device=device)


def export_state_report(job_dir: str, run_id: Optional[str] = None) -> Optional[str]:
    """Export timing rows from existing TRIDENT state attempts, marking provenance unknown."""
    try:
        # Read state without importing TRIDENT's heavy model/imaging package.
        import importlib.util
        state_module = sys.modules.get("trident.State")
        if state_module is None:
            spec = importlib.util.spec_from_file_location("_trident_timing_state", os.path.join(os.path.dirname(__file__), "State.py"))
            state_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(state_module)
        load_all_states = state_module.load_all_states
        from datetime import datetime, timezone
        cls = _report_class()
        if cls is None:
            return None
        rid = run_id or ("historical-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        report = cls(output_dir=job_dir, backend="trident", settings={"historical_export": True, "device": "unknown"}, run_id=rid)
        for state in load_all_states(job_dir).values():
            slide_meta = state.get("slide") or {}
            slide = slide_meta.get("path") or slide_meta.get("slide_path") or slide_meta.get("name") or slide_meta.get("id", "unknown")
            for task, data in (state.get("tasks") or {}).items():
                family, _, scope = task.partition(":")
                stage = {"segmentation": "segmentation", "coords": "coords", "patch_features": "patch_features", "slide_features": "slide_features"}.get(family, family)
                attempts = data.get("attempts") or [{}]
                for attempt in attempts:
                    outcome = attempt.get("result") or data.get("status", "unknown")
                    report.record(slide, stage, outcome,
                                  duration_s=attempt.get("duration_s"), scope=scope,
                                  extra={"provenance": "historical_state", "device": "unknown",
                                         "wall_time": "unknown", "reason": data.get("reason"),
                                         "attempt_id": attempt.get("attempt_id"),
                                         "started_at": attempt.get("started_at"),
                                         "finished_at": attempt.get("finished_at"),
                                         "original_run_id": attempt.get("run_id"),
                                         "state_file_slide_id": slide_meta.get("id")})
        report.finalize(status="historical_export")
        return rid
    except Exception:
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export TRIDENT state attempts to the shared timing report.")
    parser.add_argument("job_dir")
    parser.add_argument("--run-id")
    options = parser.parse_args()
    result = export_state_report(options.job_dir, options.run_id)
    if result is None:
        raise SystemExit("Could not export timing report; shared embedding report module is unavailable.")
    print(result)
