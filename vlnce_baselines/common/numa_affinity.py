"""Best-effort CPU affinity for GPU-local Habitat workers.

Habitat's subprocesses inherit the affinity of the rank process which creates
them.  Binding that parent process before ``VectorEnv`` is constructed keeps
simulation work on the NUMA node local to the rank's physical GPU.  Everything
in this module is deliberately fail-open: topology discovery or affinity
errors are reported to the caller and never prevent training from starting.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Set, Tuple


NUMA_BIND_ENV = "SAFE_VLN_BIND_NUMA"
_PCI_BUS_RE = re.compile(
    r"^(?:(?P<domain>[0-9a-fA-F]{4,8}):)?"
    r"(?P<bus>[0-9a-fA-F]{2}):(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)


@dataclass(frozen=True)
class GpuInventoryEntry:
    """One physical GPU reported by ``nvidia-smi``."""

    index: str
    uuid: str
    pci_bus_id: str


@dataclass(frozen=True)
class NumaBindingResult:
    """Outcome returned to the launch path for logging and diagnostics."""

    enabled: bool
    bound: bool
    reason: str
    local_rank: Optional[int] = None
    visible_device: Optional[str] = None
    pci_bus_id: Optional[str] = None
    numa_node: Optional[int] = None
    cpus: Tuple[int, ...] = ()


def parse_cpu_list(value: str) -> Set[int]:
    """Parse Linux ``cpulist`` syntax such as ``0-3,8,10-11``."""

    value = value.strip()
    if not value:
        raise ValueError("CPU list is empty")

    cpus: Set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("CPU list contains an empty component")
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2 or not all(bound.isdigit() for bound in bounds):
                raise ValueError("invalid CPU range: {!r}".format(part))
            first, last = (int(bound) for bound in bounds)
            if first > last:
                raise ValueError("descending CPU range: {!r}".format(part))
            cpus.update(range(first, last + 1))
        else:
            if not part.isdigit():
                raise ValueError("invalid CPU index: {!r}".format(part))
            cpus.add(int(part))
    return cpus


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Return a compact Linux-style representation of a CPU collection."""

    ordered = sorted(set(cpus))
    if not ordered:
        return ""
    ranges = []
    first = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(first) if first == previous else "{}-{}".format(first, previous))
        first = previous = cpu
    ranges.append(str(first) if first == previous else "{}-{}".format(first, previous))
    return ",".join(ranges)


def normalize_pci_bus_id(value: str) -> str:
    """Normalize NVIDIA's 8-digit PCI domain to Linux sysfs notation."""

    match = _PCI_BUS_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError("invalid PCI bus id: {!r}".format(value))
    domain = (match.group("domain") or "0000")[-4:]
    return "{}:{}:{}.{}".format(
        domain,
        match.group("bus"),
        match.group("device"),
        match.group("function"),
    ).lower()


def parse_nvidia_smi_inventory(output: str) -> Tuple[GpuInventoryEntry, ...]:
    """Parse ``index,uuid,pci.bus_id`` CSV output without requiring a GPU."""

    entries = []
    for line_number, raw_line in enumerate(output.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1] or not fields[2]:
            raise ValueError(
                "invalid nvidia-smi inventory line {}: {!r}".format(
                    line_number, raw_line
                )
            )
        entries.append(
            GpuInventoryEntry(
                index=fields[0],
                uuid=fields[1],
                pci_bus_id=normalize_pci_bus_id(fields[2]),
            )
        )
    if not entries:
        raise ValueError("nvidia-smi returned an empty GPU inventory")
    return tuple(entries)


def visible_device_for_rank(
    local_rank: int, cuda_visible_devices: Optional[str]
) -> str:
    """Map a torchrun local rank to its physical visible-device token."""

    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")
    if cuda_visible_devices is None:
        return str(local_rank)

    raw_tokens = cuda_visible_devices.split(",")
    tokens = [token.strip() for token in raw_tokens]
    if not tokens or any(not token for token in tokens):
        raise ValueError("CUDA_VISIBLE_DEVICES is empty or malformed")
    if local_rank >= len(tokens):
        raise ValueError(
            "LOCAL_RANK {} is outside {} visible CUDA devices".format(
                local_rank, len(tokens)
            )
        )
    token = tokens[local_rank]
    if token.lower() in {"-1", "none", "void", "nodevfiles"}:
        raise ValueError("CUDA_VISIBLE_DEVICES disables the selected GPU")
    return token


def match_inventory_device(
    visible_device: str, inventory: Sequence[GpuInventoryEntry]
) -> GpuInventoryEntry:
    """Resolve an NVIDIA index, UUID/prefix, or PCI bus token uniquely."""

    token = visible_device.strip()
    token_lower = token.lower()
    if token.isdigit():
        matches = [entry for entry in inventory if entry.index == token]
    else:
        try:
            normalized_bus = normalize_pci_bus_id(token)
        except ValueError:
            normalized_bus = None
        if normalized_bus is not None:
            matches = [
                entry for entry in inventory if entry.pci_bus_id == normalized_bus
            ]
        else:
            exact = [entry for entry in inventory if entry.uuid.lower() == token_lower]
            matches = exact or [
                entry
                for entry in inventory
                if entry.uuid.lower().startswith(token_lower)
            ]

    if len(matches) != 1:
        raise ValueError(
            "visible GPU token {!r} matched {} physical GPUs".format(
                visible_device, len(matches)
            )
        )
    return matches[0]


def query_nvidia_smi_inventory() -> Tuple[GpuInventoryEntry, ...]:
    """Query the physical GPU inventory with a short, bounded subprocess."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "exit status {}".format(
            completed.returncode
        )
        raise RuntimeError("nvidia-smi inventory query failed: {}".format(message))
    return parse_nvidia_smi_inventory(completed.stdout)


def _failure(
    reason: str,
    local_rank: Optional[int] = None,
    visible_device: Optional[str] = None,
    pci_bus_id: Optional[str] = None,
    numa_node: Optional[int] = None,
) -> NumaBindingResult:
    return NumaBindingResult(
        enabled=True,
        bound=False,
        reason=reason,
        local_rank=local_rank,
        visible_device=visible_device,
        pci_bus_id=pci_bus_id,
        numa_node=numa_node,
    )


def maybe_bind_current_process_to_gpu_numa(
    environ: Optional[Mapping[str, str]] = None,
    pci_sysfs_root: Path = Path("/sys/bus/pci/devices"),
    node_sysfs_root: Path = Path("/sys/devices/system/node"),
    inventory_provider: Optional[
        Callable[[], Sequence[GpuInventoryEntry]]
    ] = None,
    affinity_getter: Optional[Callable[[int], Iterable[int]]] = None,
    affinity_setter: Optional[Callable[[int, Iterable[int]], None]] = None,
) -> NumaBindingResult:
    """Bind this rank to its GPU-local CPUs when explicitly enabled.

    The optional providers make topology and mapping behavior testable without
    a GPU.  No exception escapes this function, including missing sysfs,
    missing ``nvidia-smi``, restricted cpusets, and unsupported affinity APIs.
    """

    env = os.environ if environ is None else environ
    if env.get(NUMA_BIND_ENV) != "1":
        return NumaBindingResult(
            enabled=False,
            bound=False,
            reason="{} is not 1".format(NUMA_BIND_ENV),
        )

    local_rank = None
    visible_device = None
    pci_bus_id = None
    numa_node = None
    try:
        local_rank = int(env.get("LOCAL_RANK", "0"))
        visible_device = visible_device_for_rank(
            local_rank, env.get("CUDA_VISIBLE_DEVICES")
        )
        provider = inventory_provider or query_nvidia_smi_inventory
        gpu = match_inventory_device(visible_device, provider())
        pci_bus_id = gpu.pci_bus_id

        numa_path = Path(pci_sysfs_root) / pci_bus_id / "numa_node"
        numa_node = int(numa_path.read_text(encoding="utf-8").strip())
        if numa_node < 0:
            return _failure(
                "GPU PCI device has no NUMA node",
                local_rank,
                visible_device,
                pci_bus_id,
                numa_node,
            )

        cpu_path = Path(node_sysfs_root) / "node{}".format(numa_node) / "cpulist"
        node_cpus = parse_cpu_list(cpu_path.read_text(encoding="utf-8"))

        get_affinity = affinity_getter or getattr(os, "sched_getaffinity", None)
        set_affinity = affinity_setter or getattr(os, "sched_setaffinity", None)
        if get_affinity is None or set_affinity is None:
            return _failure(
                "OS CPU-affinity APIs are unavailable",
                local_rank,
                visible_device,
                pci_bus_id,
                numa_node,
            )

        allowed_cpus = set(get_affinity(0))
        target_cpus = node_cpus.intersection(allowed_cpus)
        if not target_cpus:
            return _failure(
                "GPU-local CPUs do not intersect the process cpuset",
                local_rank,
                visible_device,
                pci_bus_id,
                numa_node,
            )
        set_affinity(0, target_cpus)
        ordered_cpus = tuple(sorted(target_cpus))
        return NumaBindingResult(
            enabled=True,
            bound=True,
            reason=(
                "bound rank {} GPU {} ({}) to NUMA node {} CPUs {}"
            ).format(
                local_rank,
                visible_device,
                pci_bus_id,
                numa_node,
                format_cpu_list(ordered_cpus),
            ),
            local_rank=local_rank,
            visible_device=visible_device,
            pci_bus_id=pci_bus_id,
            numa_node=numa_node,
            cpus=ordered_cpus,
        )
    except Exception as error:  # fail-open is part of the launch contract
        return _failure(
            "{}: {}".format(type(error).__name__, error),
            local_rank,
            visible_device,
            pci_bus_id,
            numa_node,
        )
