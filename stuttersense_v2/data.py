"""Sampling, identity auditing, duplicate grouping, and leakage-safe splits."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = {
    "speaker": ("speaker_id", "speaker"),
    "episode": ("episode_id", "episode"),
    "recording": ("recording_id", "recording"),
}
MANIFEST_REQUIRED = {
    "sample_id",
    "group_id",
    "group_kind",
    "identity_source",
    "identity_revision",
    "identity_verified",
}


@dataclass(frozen=True)
class IdentityAudit:
    status: str
    group_kind: str | None
    group_column: str | None
    source: str
    available_columns: tuple[str, ...]
    unresolved_risk: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitAudit:
    identity_status: str
    grouping_basis: str
    group_kind: str | None
    group_column: str | None
    partition_sizes: dict[str, int]
    duplicate_partition_overlap: int
    group_partition_overlap: int
    unresolved_leakage_risk: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reservoir_sample(
    iterable: Iterable[Any],
    sample_size: int | None,
    seed: int,
    scan_limit: int | None = None,
) -> tuple[list[tuple[int, Any]], int]:
    """Algorithm R sampling without replacement over the complete scanned stream."""
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be positive or null")
    if scan_limit is not None and scan_limit < 1:
        raise ValueError("scan_limit must be positive or null")
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[int, Any]] = []
    scanned = 0
    for position, item in enumerate(iterable):
        if scan_limit is not None and position >= scan_limit:
            break
        if sample_size is None or len(reservoir) < sample_size:
            reservoir.append((position, item))
        else:
            replacement = int(rng.integers(0, position + 1))
            if replacement < sample_size:
                reservoir[replacement] = (position, item)
        scanned += 1
    reservoir.sort(key=lambda pair: pair[0])
    return reservoir, scanned


def waveform_sha256(waveform: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(waveform, dtype="<f4"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def audit_dataset_identity(
    columns: Sequence[str],
    requested_kind: str | None = None,
    requested_column: str | None = None,
    identity_source: str | None = None,
    identity_revision: str | None = None,
) -> IdentityAudit:
    """Recognize only explicit identity fields; never infer identity from order or names."""
    available = tuple(sorted(set(columns)))
    if requested_kind is None and requested_column is None:
        discovered = [
            (kind, column)
            for kind, candidates in IDENTITY_COLUMNS.items()
            for column in candidates
            if column in available
        ]
        if len(discovered) == 1:
            kind, column = discovered[0]
            return IdentityAudit(
                "AVAILABLE_NOT_ATTESTED", kind, column, "dataset schema", available,
                "The field exists but its semantics must be verified before claiming identity-disjoint evaluation.",
            )
        return IdentityAudit(
            "NOT_VERIFIED", None, None, "dataset schema", available,
            "No uniquely verified speaker, episode, or recording identity is available.",
        )
    if requested_kind not in IDENTITY_COLUMNS:
        raise ValueError("requested_kind must be speaker, episode, recording, or null")
    if requested_column not in IDENTITY_COLUMNS[requested_kind]:
        raise ValueError("Requested column is not an explicit identifier for the requested group kind")
    if requested_column not in available:
        raise ValueError(f"Requested identity column is absent: {requested_column}")
    if not identity_source or not identity_revision:
        raise ValueError("Verified dataset identity requires a source and pinned revision")
    return IdentityAudit(
        "VERIFIED_BY_CONFIGURATION", requested_kind, requested_column,
        f"{identity_source} @ {identity_revision}", available, None,
    )


def load_identity_manifest(path: str | Path, sample_ids: Sequence[str]) -> tuple[np.ndarray, IdentityAudit]:
    """Load an explicitly attested sample-to-group mapping with one row per sample."""
    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    missing = MANIFEST_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Identity manifest is missing columns: {sorted(missing)}")
    if frame.empty or frame["sample_id"].duplicated().any():
        raise ValueError("Identity manifest must contain one unique row per sample_id")
    if set(frame["identity_verified"].str.lower()) != {"true"}:
        raise ValueError("Every manifest row must explicitly set identity_verified=true")
    if frame["group_id"].eq("").any() or frame["identity_source"].eq("").any():
        raise ValueError("group_id and identity_source must be non-empty")
    if frame["identity_revision"].eq("").any():
        raise ValueError("identity_revision must pin the mapping version")
    sources = set(frame["identity_source"])
    revisions = set(frame["identity_revision"])
    if len(sources) != 1 or len(revisions) != 1:
        raise ValueError("Identity manifest must use one consistent source and revision")
    kinds = set(frame["group_kind"])
    if len(kinds) != 1 or next(iter(kinds)) not in IDENTITY_COLUMNS:
        raise ValueError("Manifest group_kind must be one consistent speaker, episode, or recording value")
    requested = pd.DataFrame({"sample_id": np.asarray(sample_ids, dtype=str)})
    merged = requested.merge(frame, on="sample_id", how="left", validate="many_to_one")
    if merged["group_id"].isna().any():
        missing_ids = merged.loc[merged["group_id"].isna(), "sample_id"].head(5).tolist()
        raise ValueError(f"Identity manifest does not cover all samples; examples: {missing_ids}")
    unused = set(frame["sample_id"]) - set(requested["sample_id"])
    if unused:
        raise ValueError("Identity manifest contains samples outside the selected dataset")
    kind = next(iter(kinds))
    source = next(iter(sources))
    revision = next(iter(revisions))
    audit = IdentityAudit(
        "VERIFIED_BY_MANIFEST", kind, "group_id",
        f"{source} @ {revision}", tuple(frame.columns), None,
    )
    return merged["group_id"].to_numpy(dtype=str), audit


def build_dependence_groups(
    audio_hashes: Sequence[str], identity_values: Sequence[str] | None = None
) -> np.ndarray:
    """Build connected components induced by duplicates and verified identity groups."""
    hashes = np.asarray(audio_hashes, dtype=object)
    if hashes.ndim != 1 or len(hashes) == 0 or any(not str(value) for value in hashes):
        raise ValueError("audio_hashes must be a non-empty one-dimensional identifier sequence")
    n = len(hashes)
    parent = np.arange(n)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    sequences = [hashes]
    if identity_values is not None:
        identities = np.asarray(identity_values, dtype=object)
        if identities.shape != hashes.shape or any(not str(value) for value in identities):
            raise ValueError("identity_values must align with hashes and contain no empty values")
        sequences.append(identities)
    for values in sequences:
        first: dict[str, int] = {}
        for index, value in enumerate(values):
            key = str(value)
            if key in first:
                union(index, first[key])
            else:
                first[key] = index
    roots = [find(index) for index in range(n)]
    labels = {root: label for label, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([labels[root] for root in roots], dtype=int)


def grouped_partition(
    groups: Sequence[int | str],
    fractions: Sequence[float],
    seed: int,
    names: Sequence[str] = ("train", "validation", "test"),
) -> dict[str, np.ndarray]:
    """Assign complete groups to partitions with deterministic size balancing."""
    group_array = np.asarray(groups)
    fraction_array = np.asarray(fractions, dtype=float)
    if group_array.ndim != 1 or len(group_array) < len(names):
        raise ValueError("groups must contain enough samples for every partition")
    if len(fraction_array) != len(names) or np.any(fraction_array <= 0):
        raise ValueError("fractions and names must align and be positive")
    if not np.isclose(fraction_array.sum(), 1.0):
        raise ValueError("fractions must sum to one")
    unique, counts = np.unique(group_array, return_counts=True)
    if len(unique) < len(names):
        raise ValueError("At least one dependence group is required per partition")
    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(unique))
    order = np.lexsort((tie_break, -counts))
    targets = fraction_array * len(group_array)
    assigned: list[list[Any]] = [[] for _ in names]
    sizes = np.zeros(len(names), dtype=float)
    for position in order:
        group, size = unique[position], counts[position]
        deficits = (targets - sizes) / np.maximum(targets, 1.0)
        candidates = np.flatnonzero(np.isclose(deficits, deficits.max()))
        destination = int(rng.choice(candidates))
        assigned[destination].append(group)
        sizes[destination] += size
    partitions = {
        name: np.flatnonzero(np.isin(group_array, assigned[index]))
        for index, name in enumerate(names)
    }
    if any(len(values) == 0 for values in partitions.values()):
        raise RuntimeError("Group assignment produced an empty partition")
    return partitions


def audit_partitions(
    partitions: dict[str, Sequence[int]],
    audio_hashes: Sequence[str],
    dependence_groups: Sequence[int | str],
    identity_audit: IdentityAudit,
) -> SplitAudit:
    hashes = np.asarray(audio_hashes, dtype=object)
    groups = np.asarray(dependence_groups, dtype=object)
    names = tuple(partitions)
    duplicate_overlap = 0
    group_overlap = 0
    seen_indices: set[int] = set()
    for i, left_name in enumerate(names):
        left = np.asarray(partitions[left_name], dtype=int)
        if seen_indices.intersection(left.tolist()):
            raise ValueError("Sample indices overlap across partitions")
        seen_indices.update(left.tolist())
        for right_name in names[i + 1 :]:
            right = np.asarray(partitions[right_name], dtype=int)
            duplicate_overlap += len(set(hashes[left]).intersection(hashes[right]))
            group_overlap += len(set(groups[left]).intersection(groups[right]))
    if seen_indices != set(range(len(hashes))):
        raise ValueError("Partitions do not cover every sample exactly once")
    if duplicate_overlap or group_overlap:
        raise ValueError("Leakage audit failed: duplicates or dependence groups cross partitions")
    verified = identity_audit.status.startswith("VERIFIED")
    basis = "exact duplicates plus verified identity" if verified else "exact duplicates only"
    unresolved = None if verified else (
        "Speaker/episode independence is NOT VERIFIED; duplicate protection does not prevent identity leakage."
    )
    return SplitAudit(
        identity_audit.status,
        basis,
        identity_audit.group_kind,
        identity_audit.group_column,
        {name: len(values) for name, values in partitions.items()},
        duplicate_overlap,
        group_overlap,
        unresolved,
    )


def iter_partition_pairs(partitions: dict[str, Sequence[int]]) -> Iterator[tuple[str, str]]:
    names = tuple(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            yield left, right
