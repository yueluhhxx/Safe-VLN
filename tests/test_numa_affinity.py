import tempfile
import unittest
from pathlib import Path

from vlnce_baselines.common.numa_affinity import (
    GpuInventoryEntry,
    format_cpu_list,
    match_inventory_device,
    maybe_bind_current_process_to_gpu_numa,
    normalize_pci_bus_id,
    parse_cpu_list,
    parse_nvidia_smi_inventory,
    visible_device_for_rank,
)


INVENTORY_TEXT = """\
0, GPU-aaaaaaaa-0000, 00000000:07:00.0
3, GPU-bbbbbbbb-1111, 00000000:87:00.0
7, GPU-bbbbaaaa-2222, 00000000:BD:00.0
"""


class NumaAffinityParsingTests(unittest.TestCase):
    def test_cpu_list_parse_and_format(self):
        cpus = parse_cpu_list("0-3, 8,10-11")
        self.assertEqual(cpus, {0, 1, 2, 3, 8, 10, 11})
        self.assertEqual(format_cpu_list(cpus), "0-3,8,10-11")

    def test_cpu_list_rejects_malformed_ranges(self):
        for value in ("", "3-1", "1,,2", "a", "1-2-3"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cpu_list(value)

    def test_pci_bus_normalization_matches_sysfs(self):
        self.assertEqual(normalize_pci_bus_id("00000000:AF:00.0"), "0000:af:00.0")
        self.assertEqual(normalize_pci_bus_id("af:00.0"), "0000:af:00.0")
        self.assertEqual(normalize_pci_bus_id("0000:af:00.0"), "0000:af:00.0")

    def test_visible_device_mapping_respects_rank_and_order(self):
        self.assertEqual(visible_device_for_rank(0, "3,7"), "3")
        self.assertEqual(visible_device_for_rank(1, "3,7"), "7")
        self.assertEqual(visible_device_for_rank(2, None), "2")
        with self.assertRaises(ValueError):
            visible_device_for_rank(2, "3,7")
        with self.assertRaises(ValueError):
            visible_device_for_rank(0, "")

    def test_inventory_matches_index_uuid_prefix_and_bus(self):
        inventory = parse_nvidia_smi_inventory(INVENTORY_TEXT)
        self.assertEqual(match_inventory_device("3", inventory).index, "3")
        self.assertEqual(
            match_inventory_device("GPU-bbbbb", inventory).pci_bus_id,
            "0000:87:00.0",
        )
        self.assertEqual(match_inventory_device("bd:00.0", inventory).index, "7")
        with self.assertRaises(ValueError):
            match_inventory_device("GPU-bbbb", inventory)


class NumaAffinityApplicationTests(unittest.TestCase):
    def test_disabled_unless_value_is_exactly_one(self):
        for value in (None, "0", "true"):
            environment = {} if value is None else {"SAFE_VLN_BIND_NUMA": value}
            result = maybe_bind_current_process_to_gpu_numa(environ=environment)
            self.assertFalse(result.enabled)
            self.assertFalse(result.bound)

    def test_maps_rank_to_sysfs_and_intersects_allowed_cpus(self):
        inventory = parse_nvidia_smi_inventory(INVENTORY_TEXT)
        set_calls = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pci_root = root / "pci"
            node_root = root / "node"
            gpu_path = pci_root / "0000:bd:00.0"
            gpu_path.mkdir(parents=True)
            (gpu_path / "numa_node").write_text("2\n", encoding="utf-8")
            cpu_path = node_root / "node2"
            cpu_path.mkdir(parents=True)
            (cpu_path / "cpulist").write_text("0-3,8\n", encoding="utf-8")

            result = maybe_bind_current_process_to_gpu_numa(
                environ={
                    "SAFE_VLN_BIND_NUMA": "1",
                    "LOCAL_RANK": "1",
                    "CUDA_VISIBLE_DEVICES": "3,7",
                },
                pci_sysfs_root=pci_root,
                node_sysfs_root=node_root,
                inventory_provider=lambda: inventory,
                affinity_getter=lambda _pid: {2, 3, 4, 8},
                affinity_setter=lambda pid, cpus: set_calls.append(
                    (pid, set(cpus))
                ),
            )

        self.assertTrue(result.enabled)
        self.assertTrue(result.bound)
        self.assertEqual(result.local_rank, 1)
        self.assertEqual(result.visible_device, "7")
        self.assertEqual(result.pci_bus_id, "0000:bd:00.0")
        self.assertEqual(result.numa_node, 2)
        self.assertEqual(result.cpus, (2, 3, 8))
        self.assertEqual(set_calls, [(0, {2, 3, 8})])

    def test_topology_failure_is_returned_not_raised(self):
        result = maybe_bind_current_process_to_gpu_numa(
            environ={
                "SAFE_VLN_BIND_NUMA": "1",
                "LOCAL_RANK": "bad",
            }
        )
        self.assertTrue(result.enabled)
        self.assertFalse(result.bound)
        self.assertIn("ValueError", result.reason)

    def test_negative_numa_node_fails_open(self):
        gpu = GpuInventoryEntry("0", "GPU-zero", "0000:07:00.0")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gpu_path = root / "pci" / gpu.pci_bus_id
            gpu_path.mkdir(parents=True)
            (gpu_path / "numa_node").write_text("-1\n", encoding="utf-8")
            result = maybe_bind_current_process_to_gpu_numa(
                environ={
                    "SAFE_VLN_BIND_NUMA": "1",
                    "LOCAL_RANK": "0",
                    "CUDA_VISIBLE_DEVICES": "0",
                },
                pci_sysfs_root=root / "pci",
                node_sysfs_root=root / "node",
                inventory_provider=lambda: (gpu,),
            )
        self.assertFalse(result.bound)
        self.assertIn("no NUMA node", result.reason)


if __name__ == "__main__":
    unittest.main()
