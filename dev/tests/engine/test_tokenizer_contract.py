import io
import unittest
from unittest import mock

from server import constraints as generation_constraints
from server import server

CONFIG = {
    "eos_token_id": [248046, 248044],
    "text_config": {"vocab_size": 248320, "eos_token_id": 248044},
}


class TokenizerContractTests(unittest.TestCase):
    def tokenizer(self):
        vocabulary = {
            "text": 0,
            '<|endoftext|>': 248044,
            '<|im_end|>': 248046,
            "</think>": 248069,
        }
        return mock.Mock(
            get_vocab=mock.Mock(return_value=vocabulary),
            encode=mock.Mock(side_effect=lambda text, **_: [vocabulary[text]]),
            eos_token_id=248046,
        )

    def test_padded_model_vocabulary_need_not_have_a_token_at_every_index(self):
        contract = generation_constraints.validate_tokenizer(
            self.tokenizer(), CONFIG
        )
        self.assertEqual(contract.vocabulary, 248320)
        self.assertEqual(contract.eos_tokens, (248044, 248046))
        self.assertEqual(contract.marker, '<|im_end|>')
        self.assertEqual(contract.think_end_id, 248069)

    def test_mismatched_stop_or_thinking_tokens_are_rejected(self):
        for token in ('<|endoftext|>', '<|im_end|>', "</think>"):
            with self.subTest(token=token):
                tokenizer = self.tokenizer()
                tokenizer.get_vocab()[token] = 0
                with self.assertRaisesRegex(
                    server.engine_runtime.EngineUnhealthy, "native token"
                ):
                    generation_constraints.validate_tokenizer(tokenizer, CONFIG)

    def test_normalizer_cannot_split_a_native_special_token(self):
        tokenizer = self.tokenizer()
        tokenizer.encode.return_value = [1, 2]
        tokenizer.encode.side_effect = None
        with self.assertRaisesRegex(
            server.engine_runtime.EngineUnhealthy, "native token"
        ):
            generation_constraints.validate_tokenizer(tokenizer, CONFIG)

    def test_out_of_range_token_is_rejected(self):
        for token_id in (-1, 248320, True):
            with self.subTest(token_id=token_id):
                tokenizer = self.tokenizer()
                tokenizer.get_vocab()["community-token"] = token_id
                with self.assertRaisesRegex(
                    server.engine_runtime.EngineUnhealthy, "vocabulary"
                ):
                    generation_constraints.validate_tokenizer(tokenizer, CONFIG)

    def test_eos_configuration_must_match_a_native_stop_token(self):
        tokenizer = self.tokenizer()
        tokenizer.eos_token_id = 0
        with self.assertRaisesRegex(server.engine_runtime.EngineUnhealthy, "EOS token"):
            generation_constraints.validate_tokenizer(tokenizer, CONFIG)

    def test_config_without_vocabulary_or_eos_is_rejected(self):
        for config in (
            {},
            {"text_config": {"vocab_size": "248320", "eos_token_id": 248044}},
            {"text_config": {"vocab_size": 248320}},
            {"text_config": {"vocab_size": 248320, "eos_token_id": -1}},
        ):
            with self.subTest(config=config):
                with self.assertRaises(server.engine_runtime.EngineUnhealthy):
                    generation_constraints.validate_tokenizer(self.tokenizer(), config)

    def test_startup_rejects_tokenizer_before_starting_the_native_worker(self):
        import json as json_module
        import tempfile
        from pathlib import Path

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        Path(directory.name, "config.json").write_text(json_module.dumps(CONFIG))
        args = server.parse_args(
            ["model", "--tokenizer", directory.name, "--model", "owner/model"]
        )
        tokenizer = self.tokenizer()
        tokenizer.get_vocab()["</think>"] = 0
        http = mock.Mock()
        with (
            mock.patch.object(server, "parse_args", return_value=args),
            mock.patch.object(server.signal, "signal"),
            mock.patch.object(server, "load_thinking_key", return_value=None),
            mock.patch.object(
                server.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            mock.patch.object(server, "FrontendServer", return_value=http),
            mock.patch.object(server.engine_runtime, "MultiplexedRuntime") as native,
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
            mock.patch("sys.stdout", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            server.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("native token", stderr.getvalue())
        native.assert_not_called()
        http.server_activate.assert_not_called()
        http.server_close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
