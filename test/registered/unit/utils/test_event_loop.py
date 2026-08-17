import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.utils.event_loop import (
    EVENT_LOOP_CONFIG,
    install_event_loop,
    resolve_event_loop_config,
    run_event_loop,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEventLoop(CustomTestCase):
    def test_windows_uses_winloop(self):
        config = resolve_event_loop_config("win32")

        self.assertEqual(config.backend, "winloop")
        self.assertEqual(config.uvicorn_loop, "none")
        self.assertEqual(config.granian_loop, "winloop")

    def test_non_windows_uses_uvloop(self):
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                config = resolve_event_loop_config(platform)
                self.assertEqual(config.backend, "uvloop")
                self.assertEqual(config.uvicorn_loop, "uvloop")
                self.assertEqual(config.granian_loop, "uvloop")

    def test_install_uses_selected_backend_policy(self):
        backend = MagicMock()
        policy = backend.EventLoopPolicy.return_value

        with (
            patch(
                "sglang.srt.utils.event_loop.importlib.import_module",
                return_value=backend,
            ) as import_module,
            patch(
                "sglang.srt.utils.event_loop.asyncio.set_event_loop_policy"
            ) as set_policy,
        ):
            install_event_loop()

        import_module.assert_called_once_with(EVENT_LOOP_CONFIG.backend)
        set_policy.assert_called_once_with(policy)

    def test_run_uses_selected_backend(self):
        backend = MagicMock()
        main = object()
        result = object()
        backend.run.return_value = result

        with patch(
            "sglang.srt.utils.event_loop.importlib.import_module",
            return_value=backend,
        ) as import_module:
            self.assertIs(run_event_loop(main), result)

        import_module.assert_called_once_with(EVENT_LOOP_CONFIG.backend)
        backend.run.assert_called_once_with(main)


if __name__ == "__main__":
    unittest.main()
