"""CPU-only checks for the experiment interface and result interpretation."""
import contextlib
from datetime import timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import experiment


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def config(self, value):
        path = self.root / "input.json"
        path.write_text(json.dumps(value))
        return experiment.configuration(path)

    def test_defaults_and_explicit_values(self):
        cfg = self.config({"benchmark": {"iterations": 7}})
        self.assertEqual(cfg["name"], "exp")
        self.assertEqual(cfg["benchmark"], {"warmup": 5, "iterations": 7})
        self.assertEqual(cfg["profiling"]["backends"], [])

    def test_missing_timezone_database_does_not_fail_configuration(self):
        with patch.object(experiment, "ZoneInfo", side_effect=experiment.ZoneInfoNotFoundError("unavailable")), \
             patch.object(experiment, "timezone_database_available", return_value=False):
            cfg = self.config({})
        self.assertEqual(cfg["timezone"], "Europe/Warsaw")

    def test_missing_timezone_database_resolves_to_utc(self):
        with patch.object(experiment, "ZoneInfo", side_effect=experiment.ZoneInfoNotFoundError("unavailable")), \
             patch.object(experiment, "timezone_database_available", return_value=False):
            resolved = experiment.resolve_timezone("Europe/Warsaw")
        self.assertEqual(resolved, timezone.utc)

    def test_invalid_timezone_name_is_rejected_when_database_is_available(self):
        with patch.object(experiment, "ZoneInfo", side_effect=experiment.ZoneInfoNotFoundError("unavailable")), \
             patch.object(experiment, "timezone_database_available", return_value=True):
            with self.assertRaises(experiment.ZoneInfoNotFoundError):
                experiment.resolve_timezone("Bad/Zone")

    def test_invalid_configs_are_rejected(self):
        for value in ({"name": "../outside"}, {"device": True}, {"benchmark": {"iterations": 0}},
                      {"data": {"shape": [0]}}, {"profiling": {"backends": ["unknown"]}},
                      {"profiling": {"backends": ["msprof", "msprof"]}}, {"typo": 1},
                      {"operator": {"ref": "--help"}}, {"allow_dirty": "false"}):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.config(value)

    def test_failure_preserves_status_and_log(self):
        cfg = self.config({})
        run = self.root / "run"
        (run / "logs").mkdir(parents=True)
        def fail_idle(device, log):
            log.write_text("device busy")
            raise RuntimeError("NPU 6 has a running process")
        with patch.object(experiment, "ROOT", self.root), patch.object(experiment, "device_lock", lambda d: contextlib.nullcontext()), \
             patch.object(experiment, "ensure_idle", fail_idle), contextlib.redirect_stdout(io.StringIO()):
            code = experiment.execute(cfg, run)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "failed")
        self.assertEqual((run / "logs/npu-smi-before.log").read_text(), "device busy")

    def test_profiler_success_needs_the_custom_kernel_and_numeric_data(self):
        report = self.root / "op_summary.csv"
        report.write_text("Op Name,Duration(us)\nOtherOperator,12.3\n")
        self.assertFalse(experiment.profile_reports(self.root, "add_custom")["matched_reports"])
        report.write_text("KernelName,TaskId\nadd_custom_1,1\n")
        self.assertFalse(experiment.profile_reports(self.root, "add_custom")["matched_reports"])
        report.write_text("Op Name,Duration(us)\nAddCustom,12.3\n")
        parsed = experiment.profile_reports(self.root, "add_custom")
        self.assertEqual(parsed["matched_reports"], ["op_summary.csv"])
        self.assertEqual(parsed["sample_rows"][0]["values"]["Duration(us)"], "12.3")

    def test_interrupted_run_retains_final_status(self):
        cfg = self.config({})
        run = self.root / "interrupted"
        (run / "logs").mkdir(parents=True)
        with patch.object(experiment, "ROOT", self.root), \
             patch.object(experiment, "device_lock", lambda d: contextlib.nullcontext()), \
             patch.object(experiment, "ensure_idle", side_effect=KeyboardInterrupt("cancelled")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.execute(cfg, run), 1)
        status = json.loads((run / "status.json").read_text())
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["error"], "cancelled")

    @unittest.skipIf(os.name == 'nt', 'Process group cancellation is a Linux runner contract')
    def test_timeout_terminates_the_workload_and_keeps_its_log(self):
        log = self.root / 'timeout.log'
        program = 'import os,time; print(os.getpid(), flush=True); time.sleep(60)'
        with patch.object(experiment, "ROOT", self.root), self.assertRaises(subprocess.TimeoutExpired):
            experiment.command([sys.executable, '-c', program], log, timeout=0.5)
        pid = int(log.read_text().splitlines()[1])
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_run_directory_collision_preserves_previous_results(self):
        config = self.root / "input.json"
        config.write_text('{}')
        with patch.object(experiment, "ROOT", self.root), patch.object(experiment, "execute", return_value=0), \
             patch('sys.argv', ['experiment.py', str(config)]):
            self.assertEqual(experiment.main(), 0)
            run = next((self.root / 'runs').iterdir())
            sentinel = run / 'existing-result.txt'
            sentinel.write_text('preserve me')
            with self.assertRaises(FileExistsError):
                experiment.main()
            self.assertEqual(sentinel.read_text(), 'preserve me')

    def test_invalid_configuration_has_diagnostic_artifact(self):
        config = self.root / 'invalid.json'
        config.write_text('{"name": "../outside"}')
        with patch.object(experiment, 'ROOT', self.root), patch('sys.argv', ['experiment.py', str(config)]), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(experiment.main(), 2)
        run = next((self.root / 'runs').iterdir())
        self.assertEqual(json.loads((run / 'status.json').read_text())['stage'], 'configuration')
        self.assertEqual((run / 'config.original.json').read_text(), config.read_text())

    @unittest.skipIf(os.name == 'nt', 'File locking is exercised on the Linux server')
    def test_second_experiment_cannot_reserve_the_same_device(self):
        with patch.object(experiment, 'ROOT', self.root), experiment.device_lock(6):
            with self.assertRaises(RuntimeError):
                with experiment.device_lock(6):
                    self.fail('Second lock should not be acquired')


if __name__ == "__main__":
    unittest.main()
