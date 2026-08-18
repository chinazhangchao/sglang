"""Unit tests for JIT Ninja executable discovery."""

import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from sglang.kernels.jit.utils.compile import ninja
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNinjaExecutable(CustomTestCase):
    def test_uses_executable_from_python_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executable = pathlib.Path(temp_dir) / (
                "ninja.exe" if os.name == "nt" else "ninja"
            )
            executable.touch()

            with patch.object(ninja.ninja_package, "BIN_DIR", temp_dir):
                self.assertEqual(ninja._ninja_executable(), str(executable))


if __name__ == "__main__":
    unittest.main()
