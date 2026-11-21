"""A second decode receiver on a live bootstrap_room fails alone and leaves the first one's state alone."""

import threading
import unittest
from collections import defaultdict

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVReceiver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOM = 1000

# prefill_complete sets defer_decode_allocation and runs only with Mooncake.
POLICIES = (
    ("early", NixlKVManager, NixlKVReceiver, False),
    ("prefill_complete", MooncakeKVManager, MooncakeKVReceiver, True),
)


def _decode_manager(manager_type, defer_decode_allocation):
    mgr = manager_type.__new__(manager_type)
    mgr.defer_decode_allocation = defer_decode_allocation
    mgr.deferred_bootstrap = None
    mgr.request_status = {}
    mgr.failure_lock = threading.Lock()
    mgr.failure_records = {}
    mgr.required_prefill_response_num_table = {}
    mgr.prefill_response_tracker = defaultdict(set)
    mgr.addr_to_rooms_tracker = defaultdict(set)
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    mgr._staging_outstanding = {}
    mgr._send_records = None
    return mgr


class TestDecodeDuplicateRoom(CustomTestCase):
    def test_duplicate_receiver_fails_without_touching_owner(self):
        for policy, manager_type, receiver_type, defer in POLICIES:
            with self.subTest(policy=policy):
                mgr = _decode_manager(manager_type, defer)
                owner = receiver_type(mgr, "prefill:8998", ROOM)
                self.assertEqual(owner.poll(), KVPoll.Bootstrapping)
                duplicate = receiver_type(mgr, "prefill:8998", ROOM)
                self.assertEqual(duplicate.poll(), KVPoll.Failed)
                self.assertEqual(owner.poll(), KVPoll.Bootstrapping)
                with self.assertRaisesRegex(KVTransferError, "already in use"):
                    duplicate.failure_exception()
                self.assertNotIn(ROOM, mgr.failure_records)
                # Retiring the duplicate must not drop the owner's status entry.
                self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)
                self.assertEqual(owner.poll(), KVPoll.Bootstrapping)

    def test_room_is_reusable_after_owner_clears(self):
        for policy, manager_type, receiver_type, defer in POLICIES:
            with self.subTest(policy=policy):
                mgr = _decode_manager(manager_type, defer)
                receiver_type(mgr, "prefill:8998", ROOM).clear()
                again = receiver_type(mgr, "prefill:8998", ROOM)
                self.assertEqual(again.poll(), KVPoll.Bootstrapping)


if __name__ == "__main__":
    unittest.main()
