"""
Extract structural descriptor time series from mdCATH HDF5 files.

Converts raw MD trajectory coordinates into 8 "sensor" channels per frame:
    0. RMSD          — backbone RMSD vs initial structure (nm) [pre-computed]
    1. Rg            — radius of gyration (nm) [pre-computed as gyrationRadius]
    2. SASA          — solvent-accessible surface area (nm^2) [computed via mdtraj]
    3. Q             — native contact fraction [computed from Ca coords]
    4. helix_frac    — fraction of residues in alpha-helix [from DSSP]
    5. sheet_frac    — fraction of residues in beta-sheet [from DSSP]
    6. coil_frac     — fraction of residues in coil [from DSSP]
    7. end_to_end    — N-to-C terminal Ca distance (nm) [from Ca coords]

Output: (n_frames, 8) array per replicate, saved as .npy files in descriptors dir.

Usage:
    python protein/descriptors.py              # process TARGET_DOMAINS
    python protein/descriptors.py 5sicI00      # process specific domain
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein.config import (
    CONTACT_CUTOFF_NM,
    CONTACT_SEQ_SEPARATION,
    DATA_DIR,
    DESCRIPTOR_CHANNELS,
    DESCRIPTORS_DIR,
    N_CHANNELS,
    TARGET_DOMAINS,
    TEMPERATURES,
    N_REPLICATES,
)


def _extract_ca_indices(pdb_text: str) -> np.ndarray:
    """Extract Ca atom indices from PDB text.

    Parses ATOM records and finds lines where atom name is 'CA'.
    Returns 0-based indices into the full atom array.
    """
    lines = pdb_text.split("\n")
    ca_idx = []
    atom_idx = 0
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            if atom_name == "CA":
                ca_idx.append(atom_idx)
            atom_idx += 1
    return np.array(ca_idx, dtype=int)


def _compute_native_contacts(
    ca_coords: np.ndarray,
    ref_ca_coords: np.ndarray,
    cutoff: float = CONTACT_CUTOFF_NM,
    seq_sep: int = CONTACT_SEQ_SEPARATION,
) -> float:
    """Compute native contact fraction Q for a single frame.

    Args:
        ca_coords: (n_residues, 3) Ca coordinates for current frame (nm)
        ref_ca_coords: (n_residues, 3) Ca coordinates for reference (nm)
        cutoff: distance cutoff for defining a contact (nm)
        seq_sep: minimum sequence separation |i-j| for a contact

    Returns:
        Q: fraction of native contacts preserved (0 to 1)
    """
    n = len(ref_ca_coords)
    if n < seq_sep + 1:
        return 1.0

    # Identify native contacts from reference structure
    ref_dists = np.linalg.norm(
        ref_ca_coords[:, None, :] - ref_ca_coords[None, :, :], axis=2
    )
    # Mask: upper triangle with sequence separation > seq_sep
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + seq_sep + 1, n):
            mask[i, j] = True

    native_mask = mask & (ref_dists < cutoff)
    n_native = native_mask.sum()

    if n_native == 0:
        return 1.0

    # Count how many native contacts are preserved in current frame
    curr_dists = np.linalg.norm(
        ca_coords[:, None, :] - ca_coords[None, :, :], axis=2
    )
    preserved = (curr_dists[native_mask] < cutoff).sum()

    return float(preserved / n_native)


def _compute_dssp_fractions(dssp_frame: np.ndarray) -> tuple[float, float, float]:
    """Compute helix, sheet, coil fractions from a single DSSP frame.

    DSSP encoding in mdCATH:
        H = alpha-helix, G = 3-10 helix, I = pi-helix  -> helix
        E = extended strand, B = beta-bridge            -> sheet
        T = turn, S = bend, ' ' = coil/loop             -> coil

    Args:
        dssp_frame: (n_residues,) array of encoded DSSP bytes

    Returns:
        (helix_frac, sheet_frac, coil_frac)
    """
    n = len(dssp_frame)
    if n == 0:
        return 0.0, 0.0, 0.0

    helix = 0
    sheet = 0
    coil = 0

    for code in dssp_frame:
        if isinstance(code, bytes):
            code = code.decode()
        if code in ("H", "G", "I"):
            helix += 1
        elif code in ("E", "B"):
            sheet += 1
        else:
            coil += 1

    return float(helix / n), float(sheet / n), float(coil / n)


def _compute_sasa_trajectory(
    coords: np.ndarray,
    pdb_text: str,
    stride: int = 1,
) -> np.ndarray:
    """Compute SASA per frame using mdtraj.

    Args:
        coords: (n_frames, n_atoms, 3) in nm
        pdb_text: PDB string for topology
        stride: compute every N-th frame (1 = all frames)

    Returns:
        (n_frames,) SASA in nm^2 (interpolated if stride > 1)
    """
    import tempfile
    import mdtraj as md

    n_frames = coords.shape[0]

    # Write PDB to temp file for topology
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb_text)
        pdb_path = f.name

    try:
        topology = md.load(pdb_path).topology

        # Compute SASA in batches to manage memory
        frame_indices = list(range(0, n_frames, stride))
        sasa_sparse = np.zeros(len(frame_indices))

        batch_size = 50
        for batch_start in range(0, len(frame_indices), batch_size):
            batch_idx = frame_indices[batch_start:batch_start + batch_size]
            batch_coords = coords[batch_idx]
            traj = md.Trajectory(batch_coords, topology)
            sasa_batch = md.shrake_rupley(traj, mode="residue")
            # Total SASA per frame = sum over residues
            sasa_sparse[batch_start:batch_start + len(batch_idx)] = sasa_batch.sum(axis=1)

        # Interpolate back to full frame count if stride > 1
        if stride > 1:
            sasa_full = np.interp(
                np.arange(n_frames),
                frame_indices,
                sasa_sparse,
            )
            return sasa_full
        return sasa_sparse

    finally:
        Path(pdb_path).unlink(missing_ok=True)


def extract_descriptors(
    h5_path: Path,
    domain_id: str,
    temperature: int,
    replicate: int,
    ref_ca_coords: np.ndarray | None = None,
    ca_indices: np.ndarray | None = None,
    pdb_text: str | None = None,
    sasa_stride: int = 5,
) -> np.ndarray:
    """Extract 8-channel descriptor time series for one replicate.

    Args:
        h5_path: path to mdCATH HDF5 file
        domain_id: e.g. '5sicI00'
        temperature: e.g. 320, 450
        replicate: 0-indexed replicate number
        ref_ca_coords: (n_ca, 3) reference Ca coords for native contacts.
            If None, uses frame 0 of this trajectory.
        ca_indices: indices of Ca atoms in the full atom array.
            If None, extracted from PDB in HDF5.
        pdb_text: PDB string for SASA topology. If None, read from HDF5.
        sasa_stride: compute SASA every N frames (interpolate between)

    Returns:
        (n_frames, 8) array of descriptor channels
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[domain_id]
        rep_grp = grp[str(temperature)][str(replicate)]

        n_frames = int(rep_grp.attrs["numFrames"])

        # Pre-computed channels
        rmsd = rep_grp["rmsd"][:]           # (n_frames,) nm
        rg = rep_grp["gyrationRadius"][:]   # (n_frames,) nm
        dssp = rep_grp["dssp"][:]           # (n_frames, n_residues)
        coords_raw = rep_grp["coords"][:]    # (n_frames, n_atoms, 3) Angstroms
        coords = coords_raw / 10.0            # convert to nm (mdtraj, contacts use nm)

        # Extract Ca indices if not provided
        if ca_indices is None:
            pdb_protein = grp["pdbProteinAtoms"][()].decode("utf-8")
            ca_indices = _extract_ca_indices(pdb_protein)

        # Get PDB text for SASA if not provided
        if pdb_text is None:
            pdb_text = grp["pdb"][()].decode("utf-8")

    # Ca coordinates for all frames
    ca_coords = coords[:, ca_indices, :]  # (n_frames, n_ca, 3)

    # Reference Ca coords for native contacts
    if ref_ca_coords is None:
        ref_ca_coords = ca_coords[0]

    # Compute native contact fraction Q per frame
    q_values = np.zeros(n_frames)
    for i in range(n_frames):
        q_values[i] = _compute_native_contacts(ca_coords[i], ref_ca_coords)

    # Compute DSSP fractions per frame
    helix_frac = np.zeros(n_frames)
    sheet_frac = np.zeros(n_frames)
    coil_frac = np.zeros(n_frames)
    for i in range(n_frames):
        helix_frac[i], sheet_frac[i], coil_frac[i] = _compute_dssp_fractions(dssp[i])

    # End-to-end distance: first Ca to last Ca
    end_to_end = np.linalg.norm(ca_coords[:, -1, :] - ca_coords[:, 0, :], axis=1)

    # SASA via mdtraj (expensive — use stride + interpolation)
    print(f"    Computing SASA (stride={sasa_stride})...")
    sasa = _compute_sasa_trajectory(coords, pdb_text, stride=sasa_stride)

    # Assemble descriptor array
    descriptors = np.column_stack([
        rmsd,         # 0: RMSD (nm)
        rg,           # 1: Rg (nm)
        sasa,         # 2: SASA (nm^2)
        q_values,     # 3: native contact fraction Q
        helix_frac,   # 4: helix fraction
        sheet_frac,   # 5: sheet fraction
        coil_frac,    # 6: coil fraction
        end_to_end,   # 7: end-to-end distance (nm)
    ])

    assert descriptors.shape == (n_frames, N_CHANNELS), \
        f"Descriptor shape {descriptors.shape} != expected ({n_frames}, {N_CHANNELS})"

    return descriptors


def extract_domain_descriptors(
    domain_id: str,
    sasa_stride: int = 5,
) -> dict:
    """Extract descriptors for all temperatures and replicates of a domain.

    Saves per-replicate .npy files to DESCRIPTORS_DIR/{domain_id}/.

    Returns:
        dict with metadata and per-replicate descriptor arrays
    """
    h5_path = DATA_DIR / f"mdcath_dataset_{domain_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"{h5_path} not found. Run: python protein/download.py {domain_id}"
        )

    out_dir = DESCRIPTORS_DIR / domain_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract Ca indices and PDB text once (shared across all replicates)
    with h5py.File(h5_path, "r") as f:
        grp = f[domain_id]
        pdb_protein = grp["pdbProteinAtoms"][()].decode("utf-8")
        ca_indices = _extract_ca_indices(pdb_protein)
        n_residues = int(grp.attrs["numResidues"])

        # Reference Ca coords: 320K replicate 0 frame 0
        ref_coords = grp["320"]["0"]["coords"][0] / 10.0  # Angstroms -> nm
        ref_ca_coords = ref_coords[ca_indices]             # (n_ca, 3)

    print(f"\nDomain {domain_id}: {n_residues} residues, {len(ca_indices)} CA atoms")

    results = {
        "domain_id": domain_id,
        "n_residues": n_residues,
        "n_ca_atoms": len(ca_indices),
        "descriptors": {},
    }

    for temp in TEMPERATURES:
        for rep in range(N_REPLICATES):
            key = f"{temp}K_rep{rep}"
            npy_path = out_dir / f"{key}.npy"

            if npy_path.exists():
                print(f"  {key}: loading cached descriptors")
                desc = np.load(npy_path)
            else:
                print(f"  {key}: extracting descriptors...")
                desc = extract_descriptors(
                    h5_path, domain_id, temp, rep,
                    ref_ca_coords=ref_ca_coords,
                    ca_indices=ca_indices,
                    pdb_text=pdb_protein,
                    sasa_stride=sasa_stride,
                )
                np.save(npy_path, desc)
                print(f"    Saved {npy_path} — shape {desc.shape}")

            results["descriptors"][key] = desc

    # Save reference Ca coords for reproducibility
    np.save(out_dir / "ref_ca_coords.npy", ref_ca_coords)

    return results


def compute_rul(
    q_trajectory: np.ndarray,
    q_threshold: float,
) -> tuple[np.ndarray, int | None]:
    """Compute RUL (remaining useful life) from native contact trajectory.

    RUL = frames remaining until Q first drops below threshold and stays below.
    "Stays below" = Q never returns above threshold after the crossing point.

    Args:
        q_trajectory: (n_frames,) native contact fraction Q
        q_threshold: failure threshold (e.g. 0.3)

    Returns:
        rul: (n_frames,) RUL values. np.inf if trajectory never unfolded.
        failure_frame: frame index of permanent crossing, or None if never unfolded.
    """
    n = len(q_trajectory)

    # Find first frame where Q drops below threshold and stays below.
    # "Stays below" means Q never returns above threshold for the remainder.
    failure_frame = None
    for i in range(n):
        if q_trajectory[i] < q_threshold:
            # Check if it stays below for the rest
            if np.all(q_trajectory[i:] < q_threshold):
                failure_frame = i
                break

    if failure_frame is None:
        # Trajectory never unfolded permanently
        return np.full(n, np.inf), None

    rul = np.zeros(n)
    for i in range(n):
        if i >= failure_frame:
            rul[i] = 0
        else:
            rul[i] = failure_frame - i

    return rul, failure_frame


def main():
    domains = sys.argv[1:] if len(sys.argv) > 1 else TARGET_DOMAINS

    for domain_id in domains:
        results = extract_domain_descriptors(domain_id)

        # Print summary
        from protein.config import Q_FAILURE_THRESHOLD, UNFOLDING_TEMP, HEALTHY_TEMP
        print(f"\n  Summary for {domain_id}:")

        for temp in TEMPERATURES:
            for rep in range(N_REPLICATES):
                key = f"{temp}K_rep{rep}"
                desc = results["descriptors"][key]
                q = desc[:, 3]  # native contacts
                rul, ff = compute_rul(q, Q_FAILURE_THRESHOLD)

                status = f"unfolded at frame {ff}" if ff is not None else "stable"
                print(f"    {key}: Q_mean={q.mean():.3f}, Q_min={q.min():.3f}, "
                      f"RMSD_max={desc[:, 0].max():.3f} nm — {status}")

    print("\nDone.")


if __name__ == "__main__":
    main()
