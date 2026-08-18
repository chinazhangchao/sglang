"""Unit tests for Windows fallbacks that avoid the POSIX-only JIT toolchain."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.clamp_position import clamp_position_cuda
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestWindowsJitFallbacks(CustomTestCase):
    @patch("sglang.kernels.ops.attention.clamp_position.os.name", "nt")
    def test_clamp_position_uses_torch(self):
        seq_lens = torch.tensor([0, 1, 5], dtype=torch.int64)

        torch.testing.assert_close(
            clamp_position_cuda(seq_lens), torch.tensor([0, 0, 4])
        )


if __name__ == "__main__":
    unittest.main()
