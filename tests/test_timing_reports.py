import importlib.util
import os
import sys
import tempfile
from pathlib import Path
import types
import unittest
import json
import subprocess
from unittest.mock import patch


TIMING_PATH = os.path.join(os.path.dirname(__file__), "..", "trident", "Timing.py")


class FakeReport:
    instances = []
    events = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.rows = []
        self.devices = []
        self.run_dir = Path(kwargs["output_dir"]) / "runs" / kwargs["run_id"]
        self.__class__.instances.append(self)

    def record(self, *args, **kwargs):
        self.rows.append((args, kwargs))
        self.events.append("record")

    def register_worker_device(self, worker, device):
        self.devices.append((worker, device))

    def finalize(self, status):
        self.status = status


def load_timing():
    reports = types.ModuleType("mimyc.embedding_reports")
    reports.RunReport = FakeReport
    reports.sync_cuda = lambda device: FakeReport.events.append(("sync", device))
    mimyc = types.ModuleType("mimyc")
    mimyc.__path__ = []
    sys.modules["mimyc"] = mimyc
    sys.modules["mimyc.embedding_reports"] = reports
    spec = importlib.util.spec_from_file_location("trident_timing_under_test", TIMING_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimingReportTests(unittest.TestCase):
    def setUp(self):
        self.module_patch = patch.dict(sys.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        FakeReport.instances.clear()
        FakeReport.events.clear()
        self.timing = load_timing()

    def test_required_stages_and_cached_outcome(self):
        report = self.timing.create_report("out", "r1", settings={"task": "all"})
        self.assertEqual(report.kwargs["required_stages"], ["segmentation", "coords", "patch_features"])
        self.timing.record_stage("r1", "slide.svs", "patch_features", "skipped", scope="uni:20x", extra={"reason": "already_extracted"})
        self.assertEqual(report.rows[-1][0][2], "cached")

    def test_same_run_id_reopens_report_and_cuda_is_synchronized_at_boundary(self):
        self.timing.create_report("out", "parent", settings={"task": "feat"})
        child_report = self.timing.create_report("out", "parent", settings={"task": "feat"})
        self.timing.record_stage("parent", "slide.svs", "patch_features", "completed", started=0.0, device="cuda:2")
        self.assertIs(child_report, FakeReport.instances[-1])
        self.assertIn(("sync", "cuda:2"), FakeReport.events)
        self.assertLess(FakeReport.events.index(("sync", "cuda:2")), FakeReport.events.index("record"))

    def test_historical_export_marks_unknown_provenance(self):
        state_module = types.ModuleType("trident.State")
        state_module.load_all_states = lambda _job: {"s": {"slide": {"name": "slide.svs"}, "tasks": {
            "coords:20x": {"status": "completed", "attempts": [{"duration_s": None}]}
        }}}
        trident = types.ModuleType("trident")
        trident.__path__ = []
        sys.modules["trident"] = trident
        sys.modules["trident.State"] = state_module
        with tempfile.TemporaryDirectory() as temp:
            run_id = self.timing.export_state_report(temp, "old-run")
        self.assertEqual(run_id, "old-run")
        row = FakeReport.instances[-1].rows[0]
        self.assertIsNone(row[1]["duration_s"])
        self.assertEqual(row[1]["extra"]["device"], "unknown")
        self.assertEqual(row[1]["extra"]["wall_time"], "unknown")
        self.assertTrue(FakeReport.instances[-1].kwargs["settings"]["historical_export"])

    def test_historical_cli_works_without_imaging_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            states = Path(temp) / "wsi_states"
            states.mkdir()
            (states / "slide.json").write_text(json.dumps({"slide": {"name": "a"}, "tasks": {
                "patch_features:encoder": {"status": "skipped", "attempts": [{"result": "completed", "duration_s": 2, "attempt_id": "original"}]}
            }}))
            subprocess.run([sys.executable, TIMING_PATH, temp, "--run-id", "history"], check=True, capture_output=True, text=True, timeout=15)
            run_dir = Path(temp) / "runs" / "history"
            summary = json.loads((run_dir / "timing_summary.json").read_text())
            self.assertIsNone(summary["total_invocation_wall_s"])
            self.assertEqual(summary["stage_scope"]["patch_features:encoder"]["mean"], 2)
            hardware = json.loads((run_dir / "hardware.json").read_text())
            self.assertEqual(hardware["status"], "unknown")
            self.assertNotIn("host", hardware)


if __name__ == "__main__":
    unittest.main()
