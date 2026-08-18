"""Unit tests for distributed bootstrap backend selection."""

import unittest
from unittest.mock import patch

from sglang.srt.distributed.bootstrap import _resolve_backend, _resolve_dist_init_method
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestResolveBackend(CustomTestCase):
    @patch("sglang.srt.distributed.bootstrap.dist.is_nccl_available", return_value=False)
    @patch(
        "sglang.srt.distributed.bootstrap.get_default_distributed_backend",
        return_value="nccl",
    )
    @patch("sglang.srt.distributed.bootstrap.os.name", "nt")
    def test_windows_without_nccl_uses_gloo(self, _default_backend, _nccl_available):
        server_args = ServerArgs(model_path="dummy")

        self.assertEqual(
            _resolve_backend(device="cuda", server_args=server_args), "gloo"
        )

    def test_wildcard_ipv4_host_uses_loopback_for_rendezvous(self):
        server_args = ServerArgs(model_path="dummy", host="0.0.0.0")

        self.assertEqual(
            _resolve_dist_init_method(server_args=server_args, dist_port=12345),
            "tcp://127.0.0.1:12345",
        )

    def test_wildcard_ipv6_host_uses_loopback_for_rendezvous(self):
        server_args = ServerArgs(model_path="dummy", host="::")

        self.assertEqual(
            _resolve_dist_init_method(server_args=server_args, dist_port=12345),
            "tcp://[::1]:12345",
        )


if __name__ == "__main__":
    unittest.main()
