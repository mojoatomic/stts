"""
Download individual mdCATH domain HDF5 files from HuggingFace.

Each domain is a separate HDF5 file (~50-200 MB depending on domain size).
Do NOT download the full 3TB dataset — only the selected domains.

Usage:
    python protein/download.py                    # download TARGET_DOMAINS from config
    python protein/download.py 5sicI00 2wjwA00   # download specific domains

HDF5 file structure (per domain):
    /{domain_id}/z                          — (n_atoms,) atomic numbers
    /{domain_id}/pdb                        — PDB string (full system)
    /{domain_id}/pdbProteinAtoms            — PDB string (protein only)
    /{domain_id}/psf                        — PSF topology string
    /{domain_id}/attrs: numChains, numProteinAtoms, numResidues
    /{domain_id}/{temp}/{replica}/coords    — (n_frames, n_atoms, 3) in nm
    /{domain_id}/{temp}/{replica}/forces    — (n_frames, n_atoms, 3) in kcal/mol/A
    /{domain_id}/{temp}/{replica}/rmsd      — (n_frames,) in nm
    /{domain_id}/{temp}/{replica}/rmsf      — (n_residues,) in nm
    /{domain_id}/{temp}/{replica}/gyrationRadius — (n_frames,) in nm
    /{domain_id}/{temp}/{replica}/dssp      — (n_frames, n_residues) encoded strings
    /{domain_id}/{temp}/{replica}/box       — simulation box
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from protein.config import DATA_DIR, HUGGINGFACE_REPO, TARGET_DOMAINS


def download_domain(domain_id: str) -> Path:
    """Download a single mdCATH domain HDF5 file from HuggingFace.

    Uses huggingface_hub if available, falls back to direct URL download.

    Returns:
        Path to downloaded HDF5 file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"mdcath_dataset_{domain_id}.h5"
    local_path = DATA_DIR / filename

    if local_path.exists():
        print(f"  Already downloaded: {local_path}")
        return local_path

    try:
        from huggingface_hub import hf_hub_download
        print(f"  Downloading {domain_id} via huggingface_hub...")
        downloaded = hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            filename=f"data/{filename}",
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        # huggingface_hub downloads into local_dir/data/filename
        # Move to local_dir/filename for our expected layout
        dl_path = Path(downloaded)
        if dl_path != local_path and dl_path.exists():
            dl_path.rename(local_path)
        print(f"  Saved to {local_path}")
        return local_path
    except ImportError:
        pass

    # Fallback: direct HTTPS download
    import urllib.request
    url = (
        f"https://huggingface.co/datasets/{HUGGINGFACE_REPO}"
        f"/resolve/main/data/{filename}"
    )
    print(f"  Downloading {domain_id} from {url}...")
    urllib.request.urlretrieve(url, str(local_path))
    print(f"  Saved to {local_path}")
    return local_path


def inspect_domain(h5_path: Path, domain_id: str) -> dict:
    """Print summary of a downloaded domain HDF5 file.

    Returns dict with domain metadata for selection screening.
    """
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        grp = f[domain_id]
        n_residues = int(grp.attrs["numResidues"])
        n_protein_atoms = int(grp.attrs["numProteinAtoms"])

        # Extract CA indices from PDB
        pdb_text = grp["pdbProteinAtoms"][()].decode("utf-8")
        pdb_lines = pdb_text.split("\n")
        ca_indices = [
            i for i, line in enumerate(pdb_lines)
            if len(line.split()) > 2 and line.split()[2] == "CA"
        ]

        print(f"\n  Domain: {domain_id}")
        print(f"  Residues: {n_residues}, Protein atoms: {n_protein_atoms}, CA atoms: {len(ca_indices)}")

        info = {
            "domain_id": domain_id,
            "n_residues": n_residues,
            "n_protein_atoms": n_protein_atoms,
            "n_ca_atoms": len(ca_indices),
            "temperatures": {},
        }

        for temp in ["320", "348", "379", "413", "450"]:
            if temp not in grp:
                continue
            temp_info = {}
            for rep in sorted(grp[temp].keys()):
                rep_grp = grp[temp][rep]
                n_frames = int(rep_grp.attrs["numFrames"])
                rmsd = rep_grp["rmsd"][:]
                rg = rep_grp["gyrationRadius"][:]
                temp_info[rep] = {
                    "n_frames": n_frames,
                    "rmsd_mean": float(np.mean(rmsd)),
                    "rmsd_max": float(np.max(rmsd)),
                    "rg_mean": float(np.mean(rg)),
                }
            info["temperatures"][temp] = temp_info

            # Print summary for this temperature
            rmsds = [v["rmsd_mean"] for v in temp_info.values()]
            print(f"  {temp}K: {len(temp_info)} reps, "
                  f"mean RMSD={np.mean(rmsds):.3f} nm, "
                  f"frames={list(temp_info.values())[0]['n_frames']}")

    return info


def main():
    domains = sys.argv[1:] if len(sys.argv) > 1 else TARGET_DOMAINS

    print(f"Downloading {len(domains)} mdCATH domain(s)...")
    for domain_id in domains:
        h5_path = download_domain(domain_id)
        inspect_domain(h5_path, domain_id)

    print("\nDone.")


if __name__ == "__main__":
    main()
