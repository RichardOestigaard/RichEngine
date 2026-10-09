"""The autotune knob tables and the tuning.json persistence path.

Every served family names its TunableKnobs table; a Knob's gates keep paths
that cannot run — or that a weaker bandwidth class never wins — out of the
sweep, and the chip-keyed record merges onto a serve's environment only
where the user has not set the variable.
"""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from install import autotune, layout, models
from install.tuning import TUNING_TABLES, TuneContext, eligible, knobs_for

TUNING_HPP = Path(__file__).resolve().parents[2] / "runtime/Tuning.hpp"


def _context(**overrides):
    fields = {
        "family": "Qwen3.8-27B",
        "chip": "M4 Max",
        "gpu_family": 10,
        "bandwidth_gbps": 546,
        "model_type": "qwen3_5_text",
        "target_format": "mlx",
        "draft_kind": "dflash2",
        "diffusion": False,
    }
    return TuneContext(**fields | overrides)


class TestKnobTables(unittest.TestCase):
    def test_every_served_family_has_a_table(self):
        from install import families

        missing = [f.name for f in families.FAMILIES if knobs_for(f.name) is None]
        self.assertEqual(missing, [])

    def test_unknown_family_has_no_table(self):
        self.assertIsNone(knobs_for("No-Such-Family"))

    def test_tables_name_the_family_they_serve(self):
        for name, table in TUNING_TABLES.items():
            with self.subTest(name):
                self.assertEqual(table.family, name)
                for knob in table.knobs:
                    self.assertTrue(knob.env.startswith("RICHENGINE_"))
                    self.assertTrue(knob.values)

    def test_every_knob_names_a_variable_the_engine_reads(self):
        # A knob sweeping an env name the runtime does not read measures the
        # baseline again and again: the sweep cannot tell. Text parity with
        # Tuning.hpp catches the drift.
        engine_vars = set(re.findall(r"RICHENGINE_[A-Z_0-9]+", TUNING_HPP.read_text()))
        for name, table in TUNING_TABLES.items():
            for knob in table.knobs:
                with self.subTest(knob.env):
                    self.assertIn(knob.env, engine_vars)

    def test_no_knob_names_a_compile_time_geometry_constant(self):
        # The kernel splits, tile rows and thread/shard counts live in
        # metal/abi/ExecutionGeometry.h as #defines baked into the metallib:
        # the engine never reads them from the environment, so a knob naming
        # one would only re-measure the baseline — the parity test above
        # would miss it if the name also appeared in Tuning.hpp.
        geometry = (
            TUNING_HPP.parent / "metal/abi/ExecutionGeometry.h"
        ).read_text()
        compile_time = set(re.findall(r"RICHENGINE_[A-Z_0-9]+", geometry))
        for name, table in TUNING_TABLES.items():
            for knob in table.knobs:
                with self.subTest(knob.env):
                    self.assertNotIn(knob.env, compile_time)

    def test_draft_geometry_knobs_present_and_perf_only(self):
        # Qwen3.8-27B's table carries the env-readable performance knobs;
        # none trades output quality, so the Quick sweep keeps them.
        table = TUNING_TABLES["Qwen3.8-27B"]
        by_env = {knob.env: knob for knob in table.knobs}
        for env in (
            "RICHENGINE_PROPOSAL_CAP",
            "RICHENGINE_DRAFT_BYPASS_EXPECT",
            "RICHENGINE_SUBMIT_AHEAD",
            "RICHENGINE_ICB_OFF",
            "RICHENGINE_PREPARED_CACHE_OFF",
            "RICHENGINE_PATCHABLE_OFF",
            "RICHENGINE_NO_FUSED_GATE",
        ):
            with self.subTest(env):
                self.assertIn(env, by_env)
                self.assertFalse(by_env[env].quality_sensitive)

    def test_batch_width_matches_the_engine(self):
        # measure() submits autotune.BATCH_WIDTH concurrent lanes for the
        # batched number; it must equal ExecutionLimits::maximumBatchWidth.
        model_hpp = (TUNING_HPP.parent / "model/Model.hpp").read_text()
        width = int(re.search(r"maximumBatchWidth = (\d+)", model_hpp).group(1))
        self.assertEqual(autotune.BATCH_WIDTH, width)


class TestEligibility(unittest.TestCase):
    def test_gpu_family_gate(self):
        table = TUNING_TABLES["Qwen3.8-27B"]
        fast = {k.env for k in eligible(table, _context(gpu_family=10))}
        slow = {k.env for k in eligible(table, _context(gpu_family=8))}
        self.assertIn("RICHENGINE_MTL4", fast)
        self.assertNotIn("RICHENGINE_MTL4", slow)

    def test_bandwidth_gate(self):
        table = TUNING_TABLES["Qwen3.8-27B"]
        wide = {k.env for k in eligible(table, _context(bandwidth_gbps=546))}
        narrow = {k.env for k in eligible(table, _context(bandwidth_gbps=68))}
        self.assertIn("RICHENGINE_DFLASH_POOL", wide)
        self.assertNotIn("RICHENGINE_DFLASH_POOL", narrow)

    def test_unknown_bandwidth_is_unproven_not_excluded(self):
        # An unrecognized chip (0 GB/s) still runs the bandwidth-gated
        # knobs; measurement settles what the table could not.
        table = TUNING_TABLES["Qwen3.8-27B"]
        unknown = {k.env for k in eligible(table, _context(bandwidth_gbps=0))}
        self.assertIn("RICHENGINE_DFLASH_POOL", unknown)

    def test_format_gate(self):
        table = TUNING_TABLES["Qwen3.8-27B"]
        gguf = {k.env for k in eligible(table, _context(target_format="gguf"))}
        mlx = {k.env for k in eligible(table, _context(target_format="mlx"))}
        self.assertIn("RICHENGINE_GGUF_PACKED_ON", gguf)
        self.assertNotIn("RICHENGINE_GGUF_PACKED_ON", mlx)

    def test_draft_gate(self):
        table = TUNING_TABLES["Granite-4.2-3B"]
        drafted = {k.env for k in eligible(table, _context(draft_kind="dflash"))}
        undrafted = {k.env for k in eligible(table, _context(draft_kind="none"))}
        self.assertNotIn("RICHENGINE_DRAFT_BYPASS", undrafted | drafted)
        self.assertIn("RICHENGINE_NGRAM_PREDRAFT", undrafted)

    def test_draft_geometry_gate(self):
        # The proposal cap and bypass EWMA need a real draft; the n-gram
        # predraft's lanes leave them out even where the table lists them.
        table = TUNING_TABLES["Qwen3.8-27B"]
        drafted = {k.env for k in eligible(table, _context(draft_kind="dflash2"))}
        undrafted = {
            k.env for k in eligible(table, _context(draft_kind="none"))
        }
        for env in ("RICHENGINE_PROPOSAL_CAP", "RICHENGINE_DRAFT_BYPASS_EXPECT"):
            self.assertIn(env, drafted)
            self.assertNotIn(env, undrafted)


class TestContextFor(unittest.TestCase):
    def _assembly(self, directory, record):
        link = Path(directory) / ".resolved" / "abc123"
        link.mkdir(parents=True)
        (link / layout.ASSEMBLY_RECORD).write_text(json.dumps(record))
        return link

    def test_assembly_record(self):
        with tempfile.TemporaryDirectory() as directory:
            link = self._assembly(
                directory,
                {"family": "Qwen3.8-27B", "target_format": "mlx", "files": []},
            )
            context = autotune.context_for(link)
            self.assertEqual(context.family, "Qwen3.8-27B")
            self.assertEqual(context.target_format, "mlx")
            self.assertEqual(context.model_type, "qwen3_5_text")
            self.assertFalse(context.diffusion)

    def test_uninstalled_directory_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(models.ModelError):
                autotune.context_for(Path(directory))


class TestKeepRule(unittest.TestCase):
    """The sweep keeps a knob only when a metric wins somewhere and none
    loses anywhere."""

    def _metrics(self, single=100.0, batched=300.0, prefill=1000.0):
        return {
            "1": {
                "tokens_per_second": single,
                "prefill_tokens_per_second": prefill,
            },
            str(autotune.BATCH_WIDTH): {
                "tokens_per_second": batched,
                "prefill_tokens_per_second": prefill,
            },
        }

    def test_improvement_on_one_metric_keeps(self):
        candidate = self._metrics(single=110.0)
        self.assertTrue(autotune._keeps(candidate, self._metrics()))

    def test_batch_win_cannot_hide_a_single_lane_loss(self):
        candidate = self._metrics(single=90.0, batched=320.0)
        self.assertFalse(autotune._keeps(candidate, self._metrics()))

    def test_prefill_improvement_counts(self):
        candidate = self._metrics(prefill=1200.0)
        self.assertTrue(autotune._keeps(candidate, self._metrics()))

    def test_prefill_regression_rejects(self):
        candidate = self._metrics(single=105.0, prefill=900.0)
        self.assertFalse(autotune._keeps(candidate, self._metrics()))

    def test_noise_is_neither(self):
        candidate = self._metrics(single=100.5, batched=299.0)
        self.assertFalse(autotune._keeps(candidate, self._metrics()))

    def test_sigma_raises_the_bar(self):
        # A 5% rep spread makes the 1.5% floor meaningless: a +3% 'win'
        # inside it is not kept, and a -3% 'loss' inside it is tolerated.
        sigma = {
            ("1", "tokens_per_second"): 0.05,
            ("1", "prefill_tokens_per_second"): 0.0,
            (str(autotune.BATCH_WIDTH), "tokens_per_second"): 0.0,
            (str(autotune.BATCH_WIDTH), "prefill_tokens_per_second"): 0.0,
        }
        marginal = self._metrics(single=103.0)
        self.assertFalse(autotune._keeps(marginal, self._metrics(), sigma))
        self.assertTrue(autotune._keeps(marginal, self._metrics()))
        losing = self._metrics(single=97.0, batched=320.0)
        self.assertTrue(autotune._keeps(losing, self._metrics(), sigma))

    def test_baseline_sigma_reads_per_rep_spread(self):
        samples = {
            "1": {
                "tokens_per_second": [100.0, 102.0, 98.0],
                "prefill_tokens_per_second": [1000.0, 1000.0, 1000.0],
            },
            "4": {"tokens_per_second": [300.0, 300.0, 300.0]},
        }
        sigma = autotune._baseline_sigma(samples)
        self.assertAlmostEqual(sigma[("1", "tokens_per_second")], 2.0 / 100.0)
        self.assertEqual(sigma[("1", "prefill_tokens_per_second")], 0.0)
        self.assertNotIn(("4", "prefill_tokens_per_second"), sigma)


class TestBadPathPruning(unittest.TestCase):
    """The formulas that skip already-proven-bad paths: the deep-loss
    check, the ordered-knob prune, the priors bound."""

    def _metrics(self, tps_1=100.0, tps_4=300.0):
        return {
            "1": {"tokens_per_second": tps_1},
            "4": {"tokens_per_second": tps_4},
        }

    def test_deep_loss_needs_a_loss_at_every_width(self):
        base = self._metrics()
        deep = self._metrics(tps_1=80.0, tps_4=270.0)
        self.assertTrue(autotune._deep_loss(deep, base, autotune.PRUNE_DEPTH))
        mixed = self._metrics(tps_1=80.0, tps_4=299.0)
        self.assertFalse(autotune._deep_loss(mixed, base, autotune.PRUNE_DEPTH))
        shallow = self._metrics(tps_1=95.0, tps_4=290.0)
        self.assertFalse(autotune._deep_loss(shallow, base, autotune.PRUNE_DEPTH))
        self.assertFalse(autotune._deep_loss({}, base, autotune.PRUNE_DEPTH))

    def test_ordered_knobs_run_closest_to_default_first(self):
        by_env = {
            knob.env: knob
            for table in TUNING_TABLES.values()
            for knob in table.knobs
        }
        union = by_env["RICHENGINE_MOE_UNION"]
        self.assertTrue(union.ordered)
        self.assertEqual(union.values[0], "16")
        cap = by_env["RICHENGINE_PROPOSAL_CAP"]
        self.assertTrue(cap.ordered)
        self.assertEqual(cap.values[0], "6")

    def _knob(self, env="RICHENGINE_X"):
        from install.tuning import Knob

        return Knob(env, ("1",), "test")

    def test_priors_need_enough_observations(self):
        knob = self._knob()
        stats = {"RICHENGINE_X=1": {"wins": 0, "losses": 4}}
        self.assertFalse(autotune._prior_skips(stats, knob, "1"))

    def test_priors_skip_a_persistent_loser(self):
        knob = self._knob()
        stats = {"RICHENGINE_X=1": {"wins": 0, "losses": 20}}
        self.assertTrue(autotune._prior_skips(stats, knob, "1"))

    def test_priors_keep_a_sometimes_winner(self):
        knob = self._knob()
        stats = {"RICHENGINE_X=1": {"wins": 3, "losses": 17}}
        self.assertFalse(autotune._prior_skips(stats, knob, "1"))

    def test_priors_skip_only_the_proven_path(self):
        knob = self._knob()
        stats = {"RICHENGINE_X=1": {"wins": 0, "losses": 20}}
        self.assertFalse(autotune._prior_skips(stats, knob, "2"))

    def test_update_priors_tallies_wins_and_losses(self):
        results = {
            "baseline": {"env": {}, "metrics": {}},
            "RICHENGINE_A=1": {"env": {"RICHENGINE_A": "1"}},
            "RICHENGINE_B=1": {"env": {"RICHENGINE_B": "1"}},
            "RICHENGINE_C=1": {
                "env": {"RICHENGINE_C": "1"},
                "pressured": "thermal",
            },
            "recheck:RICHENGINE_B=1": {"env": {"RICHENGINE_B": "1"}},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "priors.json"
            autotune._update_priors(
                path, "M4 Max", results, {"RICHENGINE_A": "1"}
            )
            stats = autotune._read_priors(path)["M4 Max"]
        self.assertEqual(stats["RICHENGINE_A=1"], {"wins": 1, "losses": 0})
        # First-pass loss + recheck loss = two losses.
        self.assertEqual(stats["RICHENGINE_B=1"], {"wins": 0, "losses": 2})
        self.assertNotIn("RICHENGINE_C=1", stats)
        self.assertNotIn("baseline", stats)


class TestPersistence(unittest.TestCase):
    def _record(self, **entry):
        base = {"schema": autotune.TUNING_SCHEMA, "env": {}}
        base.update(entry)
        return base

    def test_round_trip_per_chip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = autotune.tuning_path(Path(directory))
            path.write_text(
                json.dumps(
                    {
                        "M4 Max": self._record(env={"RICHENGINE_MTL4": "1"}),
                        "M1": self._record(env={"RICHENGINE_MTL4": "0"}),
                    }
                )
            )
            tuned = autotune.load_tuning(Path(directory), chip="M4 Max")
            self.assertEqual(tuned, {"RICHENGINE_MTL4": "1"})
            self.assertEqual(autotune.load_tuning(Path(directory), chip="M2"), {})

    def test_older_schema_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            autotune.tuning_path(Path(directory)).write_text(
                json.dumps(
                    {"M4": {"schema": 1, "env": {"RICHENGINE_MTL4": "1"}}}
                )
            )
            self.assertEqual(autotune.load_tuning(Path(directory), chip="M4"), {})
            self.assertEqual(
                autotune.tuning_note(Path(directory), chip="M4"),
                "recorded by an older sweep; re-run 'richengine tune'",
            )

    def test_other_engine_build_is_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "richengine"
            binary.write_bytes(b"engine-v1")
            autotune.tuning_path(Path(directory)).write_text(
                json.dumps(
                    {
                        "M4": self._record(
                            env={"RICHENGINE_MTL4": "1"},
                            engine=autotune.engine_fingerprint(binary),
                        )
                    }
                )
            )
            self.assertEqual(
                autotune.load_tuning(
                    Path(directory), chip="M4", binary=binary
                ),
                {"RICHENGINE_MTL4": "1"},
            )
            binary.write_bytes(b"engine-v2-rebuilt")
            self.assertEqual(
                autotune.load_tuning(
                    Path(directory), chip="M4", binary=binary
                ),
                {},
            )
            self.assertIsNotNone(
                autotune.tuning_note(Path(directory), chip="M4", binary=binary)
            )

    def test_malformed_record_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            autotune.tuning_path(Path(directory)).write_text("not json")
            self.assertEqual(autotune.load_tuning(Path(directory), chip="M4"), {})

    def test_untuned_model_inherits(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(autotune.tuned_environment(Path(directory), chip="X"))

    def test_user_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            autotune.tuning_path(Path(directory)).write_text(
                json.dumps(
                    {"X": self._record(env={"RICHENGINE_MTL4": "0"})}
                )
            )
            os.environ["RICHENGINE_MTL4"] = "1"
            try:
                env = autotune.tuned_environment(Path(directory), chip="X")
            finally:
                del os.environ["RICHENGINE_MTL4"]
            self.assertEqual(env["RICHENGINE_MTL4"], "1")


class _FakeProcess:
    """A finished or blocked sweep child for Tuner tests: iterable stdout,
    a wait() that returns the code or parks on the event."""

    def __init__(self, lines=(), returncode=0, block=None):
        self.stdout = iter(lines)
        self.pid = 424242
        self._returncode = returncode
        self._block = block
        self.terminated = False

    def wait(self):
        if self._block is not None:
            self._block.wait(10)
        return self._returncode

    def terminate(self):
        self.terminated = True
        if self._block is not None:
            self._block.set()


class TestTunerJob(unittest.TestCase):
    """server/tuner.py: request validation happens before any subprocess
    spawns, then the job lifecycle mirrors Installer's."""

    def _install(self, root: Path, model="owner/repo"):
        """A minimal assembly: the link dir holding a model.json record."""
        link = root / model
        link.mkdir(parents=True)
        (link / layout.ASSEMBLY_RECORD).write_text("{}")
        return link

    def _tuner(self):
        from server import tuner

        return tuner.Tuner()

    def test_bad_model_is_400(self):
        from server import errors

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(errors.APIError) as caught:
                self._tuner().start(Path(root), "bad id with spaces")
            self.assertEqual(caught.exception.status, 400)

    def test_uninstalled_model_is_404(self):
        from server import errors

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(errors.APIError) as caught:
                self._tuner().start(Path(root), "owner/repo")
            self.assertEqual(caught.exception.status, 404)

    def test_loaded_model_needs_allow_loaded(self):
        from unittest import mock

        from server import errors, tuner

        with tempfile.TemporaryDirectory() as root:
            self._install(Path(root))
            with self.assertRaises(errors.APIError) as caught:
                self._tuner().start(Path(root), "owner/repo", serving=True)
            self.assertEqual(caught.exception.status, 409)
            with mock.patch.object(
                tuner.subprocess, "Popen", return_value=_FakeProcess()
            ):
                status = self._tuner().start(
                    Path(root), "owner/repo", serving=True, allow_loaded=True
                )
            self.assertTrue(status["running"])
            self.assertEqual(status["model"], "owner/repo")

    def test_second_tune_while_running_is_409(self):
        import threading
        import time
        from unittest import mock

        from install import paths as install_paths
        from server import errors, tuner

        block = threading.Event()
        with tempfile.TemporaryDirectory() as root:
            self._install(Path(root))
            instance = self._tuner()
            with (
                # An empty runtime dir: no live serve holds a serve-*.lock.
                mock.patch.object(install_paths, "RUNTIME", Path(root) / "rt"),
                mock.patch.object(
                    tuner.subprocess,
                    "Popen",
                    return_value=_FakeProcess(block=block),
                ),
            ):
                instance.start(Path(root), "owner/repo")
                with self.assertRaises(errors.APIError) as caught:
                    instance.start(Path(root), "owner/repo")
                self.assertEqual(caught.exception.status, 409)
            block.set()
        for _ in range(100):
            if instance.status()["done"]:
                break
            time.sleep(0.02)
        self.assertTrue(instance.status()["done"])

    def test_finished_job_reports_kept_knobs(self):
        import threading
        import time
        from unittest import mock

        from server import tuner

        chip, _, _ = autotune.detect_chip()
        with tempfile.TemporaryDirectory() as root:
            link = self._install(Path(root))
            autotune.tuning_path(link).write_text(
                json.dumps(
                    {
                        chip: {
                            "schema": autotune.TUNING_SCHEMA,
                            "env": {"RICHENGINE_MTL4": "1"},
                        }
                    }
                )
            )
            done = threading.Event()
            process = _FakeProcess(lines=("kept MTL4=1\n",), block=done)
            instance = self._tuner()
            from install import paths as install_paths

            with (
                mock.patch.object(install_paths, "RUNTIME", Path(root) / "rt"),
                mock.patch.object(tuner.subprocess, "Popen", return_value=process),
            ):
                instance.start(Path(root), "owner/repo")
                done.set()  # stdout is drained; only wait() parks
            for _ in range(100):
                if instance.status()["done"]:
                    break
                time.sleep(0.02)
            status = instance.status()
            self.assertTrue(status["ok"])
            self.assertEqual(status["kept"], ["MTL4=1"])
            self.assertEqual(status["tail"], ["kept MTL4=1"])

    def test_cancel_when_idle_is_false(self):
        self.assertFalse(self._tuner().cancel())


if __name__ == "__main__":
    unittest.main()
