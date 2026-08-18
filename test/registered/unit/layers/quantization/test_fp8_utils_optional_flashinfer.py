from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-small")

import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.test_utils import CustomTestCase


class TestFp8UtilsOptionalFlashinfer(CustomTestCase):
    def test_import_succeeds_without_flashinfer(self):
        script = textwrap.dedent(
            """
            import builtins

            real_import = builtins.__import__

            def import_without_flashinfer(name, *args, **kwargs):
                if name == "flashinfer" or name.startswith("flashinfer."):
                    raise ModuleNotFoundError("No module named 'flashinfer'")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = import_without_flashinfer

            import sglang.srt.layers.quantization.fp8_utils as fp8_utils

            assert fp8_utils._is_cuda, "This regression test requires CUDA"
            """
        )
        env = os.environ.copy()
        env["SGLANG_IS_FLASHINFER_AVAILABLE"] = "0"
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env=env,
            text=True,
            timeout=60,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
