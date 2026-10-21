"""install/local.py: local model directories, as LM Studio and oMLX keep
them — scanning, probing, name resolution and the selections they serve."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests import fixture_files
from dev.tests.installer_fixtures import (
    MOE,
    fake_hub,
    selection,
    text_config,
)
from install import local, models, upstream


def moe_config(quantization):
    return {
        "text_config": text_config(MOE),
        "quantization": quantization,
    }


def mlx_dir(path, config, *, weights=b"weights"):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    (path / "model.safetensors").write_bytes(weights)
    return path


MOE_GGUF = {
    "general.architecture": "qwen35moe",
    "qwen35moe.embedding_length": 2048,
    "qwen35moe.block_count": 40,
    "qwen35moe.context_length": 262144,
    "qwen35moe.attention.head_count": 16,
    "qwen35moe.attention.head_count_kv": 2,
    "qwen35moe.attention.key_length": 256,
    "qwen35moe.expert_count": 256,
    "qwen35moe.expert_used_count": 8,
    "tokenizer.ggml.tokens": ["token"] * 248320,
}

OQ = {
    "bits": 4,
    "group_size": 64,
    "mode": "affine",
    "model.embed_tokens": {"bits": 8, "group_size": 64},
    "model.layers.0.mlp.experts.0.down_proj": {"bits": 5, "group_size": 128},
}
AFFINE = {"bits": 4, "group_size": 64, "mode": "affine"}


class LocalTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # The scan roots are fixture directories, never the machine's.
        self.lmstudio = self.root / ".lmstudio" / "models"
        self.omlx = self.root / ".omlx"
        for name, value in (
            ("install.local.LMSTUDIO_MODELS", self.lmstudio),
            ("install.local.OMLX_DIR", self.omlx),
        ):
            patch = mock.patch(name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def candidates(self):
        return {candidate.name: candidate for candidate in local.scan()}

    def test_scan_finds_models_in_the_cache_layouts(self):
        mlx_dir(
            self.lmstudio / "author" / "moe-oq4e",
            moe_config(OQ),
        )
        mlx_dir(
            self.lmstudio / "author" / "moe-4bit",
            moe_config(AFFINE),
        )
        # A flat directory the root itself holds.
        mlx_dir(self.lmstudio / "flat-oQ8", moe_config({"bits": 8, "group_size": 64}))
        # Hidden folders (imatrix stores, partial downloads) never list.
        (self.lmstudio / ".oqe_imatrix").mkdir(parents=True)
        (self.lmstudio / "author" / ".cache").mkdir()
        # oMLX's settings.json names another root.
        self.omlx.mkdir()
        (self.omlx / "settings.json").write_text(
            json.dumps({"model": {"model_dirs": [str(self.root / "extra")]}})
        )
        mlx_dir(self.root / "extra" / "omlx-model", moe_config(AFFINE))

        found = self.candidates()
        self.assertEqual(
            sorted(found),
            [
                "author/moe-4bit",
                "author/moe-oq4e",
                "flat-oQ8",
                "omlx-model",
            ],
        )
        oq = found["author/moe-oq4e"]
        self.assertEqual(
            (oq.format, oq.family, oq.reason), ("mlx", "Qwen3.6-35B-A3B", None)
        )
        self.assertIn("mixed", oq.quantization)
        self.assertIn("group-64/128", oq.quantization)
        self.assertEqual(found["author/moe-4bit"].format, "mlx")
        self.assertIn("4-bit", found["author/moe-4bit"].quantization)
        self.assertEqual(found["omlx-model"].family, "Qwen3.6-35B-A3B")

    def test_probe_reports_what_cannot_serve(self):
        # Outside the oQ matrix, no tokenizer, an unknown architecture: each
        # lists with its reason instead of being skipped silently.
        mlx_dir(
            self.lmstudio / "a" / "too-low",
            moe_config({"bits": 3, "group_size": 64}),
        )
        mlx_dir(
            self.lmstudio / "a" / "bad-override",
            moe_config(AFFINE | {"model.x": {"bits": 6, "group_size": 32}}),
        )
        unknown = text_config(MOE, model_type="something_else")
        mlx_dir(
            self.lmstudio / "a" / "alien",
            {"text_config": unknown, "quantization": AFFINE},
        )
        found = self.candidates()
        self.assertIn("3 bits/group-64", found["a/too-low"].reason)
        self.assertIn("model.x", found["a/bad-override"].reason)
        self.assertIn("architecture", found["a/alien"].reason)

    def test_probe_reads_a_gguf_header(self):
        path = self.lmstudio / "unsloth" / "moe-gguf"
        path.mkdir(parents=True)
        fixture_files.write_gguf(path / "model-Q4_K_M.gguf", MOE_GGUF, [])
        candidate = self.candidates()["unsloth/moe-gguf"]
        self.assertEqual(candidate.format, "gguf")
        self.assertEqual(candidate.family, "Qwen3.6-35B-A3B")
        self.assertIsNone(candidate.reason)

    def test_find_resolves_names(self):
        mlx_dir(self.lmstudio / "a" / "shared", moe_config(AFFINE))
        mlx_dir(self.lmstudio / "b" / "shared", moe_config(AFFINE))
        mlx_dir(self.lmstudio / "a" / "unique", moe_config(AFFINE))
        self.assertEqual(
            local.find("unique"), (self.lmstudio / "a" / "unique").resolve()
        )
        self.assertEqual(
            local.find("a/shared"), (self.lmstudio / "a" / "shared").resolve()
        )
        with self.assertRaisesRegex(models.ModelError, "several local models"):
            local.find("shared")
        with self.assertRaisesRegex(models.ModelError, "no local model"):
            local.find("missing")

    def test_a_local_directory_makes_a_selection(self):
        directory = mlx_dir(self.lmstudio / "a" / "moe-oq4e", moe_config(OQ))
        chosen = models.Selection.of(self.root / "models", str(directory))
        self.assertEqual(chosen.repo_id, str(directory.resolve()))
        self.assertEqual(chosen.model, "moe-oq4e")
        # No repository ID: the selection link is the hashed kind.
        self.assertEqual(chosen.link.parent.name, ".selections")
        # The same directory by its scanned name selects it too, and by
        # the local: prefix.
        self.assertEqual(
            models.Selection.of(self.root / "models", "moe-oq4e").link, chosen.link
        )
        self.assertEqual(
            models.Selection.of(self.root / "models", "local:a/moe-oq4e").link,
            chosen.link,
        )

    def test_repository_ids_stay_repository_ids(self):
        chosen = models.Selection.of(self.root / "models", "owner/repo:VAR")
        self.assertEqual(chosen.repo_id, "owner/repo")
        self.assertEqual(chosen.variant, "VAR")
        with self.assertRaisesRegex(models.ModelError, "owner/repo"):
            models.Selection.of(self.root / "models", "missing-name")
        with self.assertRaisesRegex(models.ModelError, "model directory not found"):
            models.Selection.of(self.root / "models", "/no/such/directory")

    def test_a_local_oq_checkpoint_installs(self):
        fake_hub(self, self.root / "hub", target=MOE)
        directory = mlx_dir(self.lmstudio / "author" / "moe-oq4e", moe_config(OQ))
        chosen = selection(self.root, str(directory))
        from install import assembly

        upstream.prepare(chosen)
        record = assembly.verify(chosen.link)
        self.assertEqual(record["target_format"], "mlx-affine")
        self.assertEqual(record["family"], "Qwen3.6-35B-A3B")
        # The source is the local directory, recorded without a revision.
        self.assertEqual(
            record["sources"]["target"],
            {"repo": str(directory.resolve()), "revision": None},
        )
        # Its weights link into the LM Studio directory, not the Hub cache.
        linked = (chosen.link / "target" / "model.safetensors").resolve()
        self.assertEqual(linked, directory.resolve() / "model.safetensors")
        # Starting it again keeps the installed assembly.
        upstream.prepare(chosen)


if __name__ == "__main__":
    unittest.main()
