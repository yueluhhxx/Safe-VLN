"""Safe-VLN occupancy-mask utilities.

The scan and radial-map conventions intentionally match Sim2Sim-VLNCE:

* four 90 degree depth patches are concatenated in sensor-id order;
* raw scan indices increase clockwise, with forward at one eighth of a turn
  because each camera's optical axis is at the center of its 90 degree patch;
* radial occupancy uses ``1`` for a surface, ``-1`` for free space, and
  ``0`` for unknown space.

The waypoint predictor uses the same forward/clockwise heading convention,
but stores maps as ``[heading, range]``.  Keeping the conversion here as pure
torch functions makes the geometry testable without starting Habitat.
"""

from typing import Sequence

import torch


SIM2SIM_FREE = -1.0
SIM2SIM_UNKNOWN = 0.0
SIM2SIM_SURFACE = 1.0
SAFE_FREE = 0.0
SAFE_BLOCKED = -1.0
DEFAULT_OCCUPANCY_DELTA = 1.0e-5


def validate_occupancy_geometry(
    num_patches,
    patch_width,
    heading_bins,
    range_bins,
    max_range,
    heading_offset_bins=-15,
):
    """Validate the sensor-to-waypoint discretization.

    ``heading_offset_bins`` is the signed roll from laser headings to
    waypoint headings.  Sim2Sim concatenation starts at the left edge of the
    forward camera: the forward optical axis is therefore at ``H / 8``.  For
    120 headings the production correction is ``-15`` so ETPNav bin zero is
    forward.  It remains explicit to prevent a silent image-edge offset.
    """

    integer_values = {
        "num_patches": num_patches,
        "patch_width": patch_width,
        "heading_bins": heading_bins,
        "range_bins": range_bins,
        "heading_offset_bins": heading_offset_bins,
    }
    for name, value in integer_values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("{} must be an integer, got {!r}".format(name, value))

    if num_patches != 4:
        raise ValueError("a 360 degree scan requires exactly four 90 degree patches")
    if patch_width <= 0 or heading_bins <= 0 or range_bins <= 0:
        raise ValueError("patch_width, heading_bins, and range_bins must be positive")
    if num_patches * patch_width % heading_bins != 0:
        raise ValueError("raw laser samples must divide evenly into heading bins")
    if heading_bins % 8 != 0:
        raise ValueError("heading_bins must represent the 45 degree patch-edge offset")
    if max_range <= 0:
        raise ValueError("max_range must be positive")
    if not -heading_bins < heading_offset_bins < heading_bins:
        raise ValueError("heading_offset_bins must have magnitude below heading_bins")
    expected_offset = -(heading_bins // 8)
    if heading_offset_bins != expected_offset:
        raise ValueError(
            "heading_offset_bins must be {} (roll raw forward bin H/8 to zero)".format(
                expected_offset
            )
        )


def depth_patches_to_laser_scan(
    patches: Sequence[torch.Tensor],
    patch_width=360,
):
    """Join four raw depth patches into Sim2Sim's clockwise raw scan.

    Habitat 0.1.7 depth observations are ``[B, H, W, 1]``.  A ``[B, H, W]``
    tensor is also accepted to keep the function convenient for unit tests.
    Sim2Sim's ``exact`` height pooling is reproduced by selecting the middle
    row; the configured Safe-VLN sensor has height one.
    """

    if len(patches) != 4:
        raise ValueError("expected four depth patches, got {}".format(len(patches)))

    rows = []
    reference = None
    for patch_index, patch in enumerate(patches):
        if not torch.is_tensor(patch):
            raise TypeError("depth patch {} is not a tensor".format(patch_index))
        if patch.dim() == 4:
            if patch.shape[-1] != 1:
                raise ValueError("depth patches must have one channel")
            patch = patch[..., 0]
        elif patch.dim() != 3:
            raise ValueError("depth patches must be [B,H,W,1] or [B,H,W]")

        if patch.shape[1] <= 0 or patch.shape[2] != patch_width:
            raise ValueError(
                "depth patch {} has shape {}; expected width {}".format(
                    patch_index, tuple(patch.shape), patch_width
                )
            )
        identity = (patch.shape[0], patch.device, patch.dtype)
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("all depth patches must share batch, device, and dtype")

        rows.append(patch[:, patch.shape[1] // 2, :])

    return torch.cat(rows, dim=1)


def laser_scan_to_radial_occupancy(
    scan,
    heading_bins=120,
    range_bins=12,
    max_range=3.0,
    is_normalized=True,
    agg_percentile=50.0,
):
    """Convert a batched range scan to Sim2Sim ternary radial occupancy.

    The returned layout is ``[B, range, heading]``.  Raw rays within each
    heading bin are aggregated with the requested percentile; Safe-VLN uses
    the median (50th percentile), i.e. twelve 0.25 degree rays per 3 degree
    heading bin with the default 1440-ray scan.
    """

    if not torch.is_tensor(scan) or scan.dim() != 2:
        raise ValueError("scan must be a [batch, raw_heading_samples] tensor")
    if heading_bins <= 0 or range_bins <= 0 or max_range <= 0:
        raise ValueError("heading_bins, range_bins, and max_range must be positive")
    if scan.shape[1] % heading_bins != 0:
        raise ValueError("scan samples must divide evenly into heading bins")
    if not 0.0 <= float(agg_percentile) <= 100.0:
        raise ValueError("agg_percentile must be in [0, 100]")
    if scan.shape[1] == 0:
        raise ValueError("scan must contain at least one ray")

    # torch.quantile on CUDA did not support fp16 in the PyTorch version used
    # by ETPNav.  Occupancy has no gradient path, so aggregate in float32.
    samples = scan.detach().to(dtype=torch.float32)
    if is_normalized:
        samples = samples * float(max_range)
    samples = torch.where(
        torch.isnan(samples), torch.full_like(samples, -1.0), samples
    )
    samples = samples.reshape(scan.shape[0], heading_bins, -1)
    ranges = torch.quantile(
        samples, float(agg_percentile) / 100.0, dim=-1
    )

    finite = torch.isfinite(ranges)
    measurable = (finite & (ranges >= 0.0)) | torch.isposinf(ranges)
    beyond_max = measurable & (ranges >= float(max_range))

    # Avoid converting +/-inf to integer.  Values at max_range belong to the
    # extra [max_range, inf] histogram bin in Sim2Sim and therefore mark every
    # represented range bin as free.
    finite_ranges = torch.where(finite, ranges, torch.zeros_like(ranges))
    bin_width = float(max_range) / float(range_bins)
    surface_bins = torch.floor(finite_ranges / bin_width).to(torch.long)

    radial_indices = torch.arange(
        range_bins, device=scan.device, dtype=torch.long
    ).view(1, range_bins, 1)
    measured_bins = surface_bins.unsqueeze(1)
    valid_surface = measurable & ~beyond_max

    occupancy = torch.full(
        (scan.shape[0], range_bins, heading_bins),
        SIM2SIM_UNKNOWN,
        device=scan.device,
        dtype=torch.float32,
    )
    free = measurable.unsqueeze(1) & (
        beyond_max.unsqueeze(1) | (radial_indices < measured_bins)
    )
    surface = valid_surface.unsqueeze(1) & (radial_indices == measured_bins)
    occupancy[free] = SIM2SIM_FREE
    occupancy[surface] = SIM2SIM_SURFACE
    return occupancy


def radial_occupancy_to_safe_mask(
    occupancy,
    heading_bins=120,
    range_bins=12,
    heading_offset_bins=-15,
):
    """Map Sim2Sim ternary occupancy to a Safe-VLN waypoint mask.

    Free cells become zero.  Both obstacle surfaces and unobserved space
    become ``-1`` as required by Safe-VLN.  Output layout is
    ``[B, heading, range]`` to match ETPNav's waypoint heatmap.
    """

    if not torch.is_tensor(occupancy) or occupancy.dim() != 3:
        raise ValueError("occupancy must be a [batch, range, heading] tensor")
    if tuple(occupancy.shape[1:]) != (range_bins, heading_bins):
        raise ValueError(
            "occupancy shape {} does not match [B, {}, {}]".format(
                tuple(occupancy.shape), range_bins, heading_bins
            )
        )
    if not isinstance(heading_offset_bins, int) or isinstance(
        heading_offset_bins, bool
    ):
        raise ValueError("heading_offset_bins must be an integer")
    if not -heading_bins < heading_offset_bins < heading_bins:
        raise ValueError("heading_offset_bins must have magnitude below heading_bins")

    valid_values = (
        (occupancy == SIM2SIM_FREE)
        | (occupancy == SIM2SIM_UNKNOWN)
        | (occupancy == SIM2SIM_SURFACE)
    )
    if not bool(torch.all(valid_values).item()):
        raise ValueError("occupancy must contain only Sim2Sim values -1, 0, or 1")

    safe_mask = torch.where(
        occupancy == SIM2SIM_FREE,
        torch.full_like(occupancy, SAFE_FREE),
        torch.full_like(occupancy, SAFE_BLOCKED),
    ).transpose(1, 2)
    if heading_offset_bins:
        safe_mask = torch.roll(safe_mask, shifts=heading_offset_bins, dims=1)
    return safe_mask.contiguous()


def normalize_probability_rows(probabilities, eps=1.0e-12):
    """Clamp and normalize ``[N, K]`` rows with a uniform fallback."""

    if not torch.is_tensor(probabilities) or probabilities.dim() != 2:
        raise ValueError("probabilities must be a [rows, classes] tensor")
    if probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one class")
    clean = torch.where(
        torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities)
    ).clamp_min(0.0)
    totals = clean.sum(dim=1, keepdim=True)
    uniform = torch.full_like(clean, 1.0 / float(clean.shape[1]))
    return torch.where(totals > eps, clean / totals.clamp_min(eps), uniform)


def regional_waypoint_to_global_indices(
    region_index,
    sampled_index,
    heading_bins=120,
    range_bins=12,
    region_width=10,
    heading_offset=5,
):
    """Invert a sample from a rolled regional waypoint probability map.

    A positive ``heading_offset`` roll moves global heading ``h`` to
    ``(h + heading_offset) % heading_bins`` before the map is split into
    panorama regions.  In particular, the forward region contains headings
    ``[115, ..., 119, 0, ..., 4]``, not ``[0, ..., 9]``.  This inverse also
    preserves the sampled range bin.  Integer scalars and broadcastable
    integer tensors are both supported without moving tensors between
    devices or changing the probability map's dtype/gradient path.
    """

    if heading_bins <= 0 or range_bins <= 0 or region_width <= 0:
        raise ValueError("heading_bins, range_bins, and region_width must be positive")
    if heading_bins % region_width != 0:
        raise ValueError("heading_bins must divide evenly into panorama regions")
    local_heading = sampled_index // range_bins
    heading = (
        region_index * region_width + local_heading - heading_offset
    ) % heading_bins
    return heading, sampled_index % range_bins


def fuse_occupancy_mask(
    probabilities,
    safe_mask,
    delta=DEFAULT_OCCUPANCY_DELTA,
    eps=1.0e-12,
):
    """Apply Safe-VLN's ``norm(H + delta * M)`` robustly.

    Inputs and output use ETPNav layout ``[B, heading, range]``.  Negative or
    non-finite probability values are removed before normalization.  The
    masked map is clamped to zero, normalized per batch item, and falls back
    to the unmasked normalized map if masking removes all probability mass.
    """

    if not torch.is_tensor(probabilities) or probabilities.dim() != 3:
        raise ValueError("probabilities must be a [batch, heading, range] tensor")
    if not torch.is_tensor(safe_mask) or safe_mask.shape != probabilities.shape:
        raise ValueError("safe_mask must have exactly the probability-map shape")
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    valid_mask = (safe_mask == SAFE_FREE) | (safe_mask == SAFE_BLOCKED)
    if not bool(torch.all(valid_mask).item()):
        raise ValueError("safe_mask must contain only 0 (free) or -1 (blocked)")

    batch_size = probabilities.shape[0]
    flat = probabilities.reshape(batch_size, -1)
    base = normalize_probability_rows(flat, eps=eps).reshape_as(probabilities)
    updated = (base + float(delta) * safe_mask.to(base.dtype)).clamp_min(0.0)
    totals = updated.reshape(batch_size, -1).sum(dim=1, keepdim=True)
    normalized = updated / totals.clamp_min(eps).view(batch_size, 1, 1)
    fallback = totals <= eps
    return torch.where(fallback.view(batch_size, 1, 1), base, normalized)


def build_safe_occupancy_mask(
    patches,
    patch_width=360,
    heading_bins=120,
    range_bins=12,
    max_range=3.0,
    is_normalized=True,
    agg_percentile=50.0,
    heading_offset_bins=-15,
):
    """End-to-end pure conversion from four depth patches to a Safe mask."""

    validate_occupancy_geometry(
        len(patches),
        patch_width,
        heading_bins,
        range_bins,
        max_range,
        heading_offset_bins,
    )
    scan = depth_patches_to_laser_scan(patches, patch_width=patch_width)
    occupancy = laser_scan_to_radial_occupancy(
        scan,
        heading_bins=heading_bins,
        range_bins=range_bins,
        max_range=max_range,
        is_normalized=is_normalized,
        agg_percentile=agg_percentile,
    )
    return radial_occupancy_to_safe_mask(
        occupancy,
        heading_bins=heading_bins,
        range_bins=range_bins,
        heading_offset_bins=heading_offset_bins,
    )
