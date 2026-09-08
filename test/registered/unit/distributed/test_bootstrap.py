"""Unit tests for distributed bootstrap backend selection."""

import unittest
from unittest.mock import patch

from sglang.srt.distributed.bootstrap import _resolve_backend, _resolve_dist_init_method
from sglang.srt.runtime_context import get_context, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestResolveBackend(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    @patch("sglang.srt.distributed.bootstrap.dist.is_nccl_available", return_value=False)
    @patch(
        "sglang.srt.distributed.bootstrap.get_default_distributed_backend",
        return_value="nccl",
    )
    @patch("sglang.srt.distributed.bootstrap.os.name", "nt")
    def test_windows_without_nccl_uses_gloo(self, _default_backend, _nccl_available):
        with get_context().override_server_args():
            self.assertEqual(_resolve_backend(device="cuda"), "gloo")

    def test_wildcard_ipv4_host_uses_loopback_for_rendezvous(self):
        with get_context().override_server_args(host="0.0.0.0"):
            self.assertEqual(
                _resolve_dist_init_method(dist_port=12345),
                "tcp://127.0.0.1:12345",
            )

    def test_wildcard_ipv6_host_uses_loopback_for_rendezvous(self):
        with get_context().override_server_args(host="::"):
            self.assertEqual(
                _resolve_dist_init_method(dist_port=12345),
                "tcp://[::1]:12345",
            )


if __name__ == "__main__":
    unittest.main()
