"""Network-free adapter tests; fake clients never transmit data."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gemini_text as g


def approved_synthetic_request():
    return dict(provider="developer-api", model="gemini-synthetic-test", task="caption",
                source_text="合成测试：n=20，选择率50.0%，P=0.040。",
                instructions="写一条简短图注，不改变数据。", consent_confirmed=True,
                data_minimized_confirmed=True, max_output_tokens=512)


class GeminiSafetyTests(unittest.TestCase):
    def test_unapproved_public_template_is_blocked(self):
        path = Path(__file__).resolve().parents[1] / "assets" / "gemini_request.example.json"
        with self.assertRaises(g.RequestError):
            g.validate_request(json.loads(path.read_text(encoding="utf-8")))

    def test_no_consent_or_unreviewed_data_cannot_proceed(self):
        for key in ("consent_confirmed", "data_minimized_confirmed"):
            request = approved_synthetic_request()
            request[key] = False
            with self.subTest(key=key), self.assertRaises(g.RequestError):
                g.validate_request(request)

    def test_credentials_are_not_accepted_in_request_json(self):
        request = approved_synthetic_request()
        request["api_key"] = "synthetic-not-a-key"
        with self.assertRaises(g.RequestError):
            g.validate_request(request)

    def test_sensitive_content_stops_without_echoing_it(self):
        for content in ["someone@example.invalid", "192.0.2.1", "13800138000",
                        "-----BEGIN PRIVATE KEY-----"]:
            request = approved_synthetic_request()
            request["source_text"] = content
            with self.subTest(content_type=content[:4]):
                with self.assertRaises(g.RequestError) as caught:
                    g.validate_request(request)
                self.assertNotIn(content, str(caught.exception))

    def test_long_manuscript_task_or_private_model_alias_is_rejected(self):
        for key, value in [("task", "manuscript"), ("model", "quality"),
                           ("max_output_tokens", 100000), ("provider", "unknown-endpoint")]:
            request = approved_synthetic_request()
            request[key] = value
            with self.subTest(key=key), self.assertRaises(g.RequestError):
                g.validate_request(request)

    def test_preview_does_not_read_credentials_or_import_sdk(self):
        forbidden_env = mock.MagicMock()
        forbidden_env.get.side_effect = AssertionError("Do not read credentials during dry-run")
        with mock.patch.object(g.os, "environ", forbidden_env), mock.patch.object(g.importlib, "import_module") as loader:
            preview = g.request_preview(approved_synthetic_request())
        loader.assert_not_called()
        forbidden_env.get.assert_not_called()
        self.assertEqual(preview["status"], "validated_locally_not_sent")
        self.assertNotIn("source_text", preview)

    def test_cli_without_send_cannot_call_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "synthetic.json"
            request_path.write_text(json.dumps(approved_synthetic_request()), encoding="utf-8")
            with mock.patch.object(g, "_send") as sender, contextlib.redirect_stdout(io.StringIO()):
                code = g.main(["--request", str(request_path)])
            sender.assert_not_called()
            self.assertEqual(code, 0)

    def test_missing_output_or_existing_output_blocks_before_send(self):
        with tempfile.TemporaryDirectory() as directory:
            request_path, output_path = Path(directory) / "request.json", Path(directory) / "result.json"
            request_path.write_text(json.dumps(approved_synthetic_request()), encoding="utf-8")
            output_path.write_text("user-file", encoding="utf-8")
            for extra in [[], ["--output", str(output_path)]]:
                with self.subTest(extra=extra), mock.patch.object(g, "_send") as sender, contextlib.redirect_stderr(io.StringIO()):
                    code = g.main(["--request", str(request_path), "--send", *extra])
                sender.assert_not_called()
                self.assertEqual(code, 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "user-file")

    def test_sdk_failure_message_and_payload_are_not_logged(self):
        secret_marker = "SYNTHETIC_PRIVATE_FAILURE_DETAIL"
        with tempfile.TemporaryDirectory() as directory:
            request_path, output_path = Path(directory) / "request.json", Path(directory) / "result.json"
            request_path.write_text(json.dumps(approved_synthetic_request()), encoding="utf-8")
            stderr = io.StringIO()
            with mock.patch.object(g, "_send", side_effect=RuntimeError(secret_marker)), contextlib.redirect_stderr(stderr):
                code = g.main(["--request", str(request_path), "--send", "--output", str(output_path)])
            self.assertEqual(code, 2)
            self.assertNotIn(secret_marker, stderr.getvalue())
            self.assertNotIn(secret_marker, output_path.read_text(encoding="utf-8"))

    def test_fake_sdk_complete_and_truncated_responses(self):
        for finish, expected in [("STOP", True), ("MAX_TOKENS", False)]:
            response = SimpleNamespace(text="合成短说明。", model_version="gemini-synthetic-test",
                candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
                usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=3, total_token_count=8))
            client = mock.MagicMock()
            client.models.generate_content.return_value = response
            sdk = SimpleNamespace(Client=mock.MagicMock(return_value=client))
            with self.subTest(finish=finish), mock.patch.object(g.importlib, "import_module", return_value=sdk), mock.patch.object(g.os, "environ", {"GEMINI_API_KEY": "synthetic-not-a-key"}):
                result = g._send(approved_synthetic_request())
            self.assertEqual(result["complete"], expected)
            self.assertEqual(result["usage"]["total_token_count"], 8)
            self.assertEqual(sdk.Client.call_args.kwargs["http_options"]["retry_options"]["attempts"], 1)
            client.close.assert_called_once()
            self.assertNotIn("api_key", result)

    def test_vertex_needs_own_config_and_does_not_inherit_developer_key(self):
        request = approved_synthetic_request()
        request["provider"] = "vertex-ai"
        for environment in [{}, {"GOOGLE_CLOUD_PROJECT": "synthetic-project", "GOOGLE_CLOUD_LOCATION": "global",
                                 "GEMINI_API_KEY": "synthetic-not-a-key"}]:
            sdk = SimpleNamespace(Client=mock.MagicMock())
            with self.subTest(environment_fields=list(environment)), mock.patch.object(g.os, "environ", environment), mock.patch.object(g.importlib, "import_module", return_value=sdk):
                with self.assertRaises(g.RequestError):
                    g._send(request)
            sdk.Client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
