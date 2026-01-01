#!/usr/bin/env python
import os

# Avoid OpenMP crash on macOS with multiple libomp copies
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Don't go online to check newer versions
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import sys
import time
import tarfile
from pathlib import Path

import torch
from transformers import EsmForProteinFolding

# enable dropout to make folding non-deterministic
# this helps to estimate model certainty over multiple runs
# and enables comparison across two structures without
# folding stochasticity determining the outcome
def enable_dropout(model: torch.nn.Module) -> None:
    """
    Enable dropout layers during inference.
    This forces all nn.Dropout modules into train() mode,
    while keeping the rest of the model in eval() mode.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def pick_device() -> torch.device:
    """
    Choose the best available device: CUDA -> MPS (Apple) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_first_fasta_sequence(path: Path) -> str:
    """
    Load the first sequence from a FASTA file.
    """
    text = path.read_text().strip().splitlines()
    if not text:
        raise SystemExit(f"FASTA file {path} is empty.")

    # concatenate all non-header lines
    lines = [l.strip() for l in text if l and not l.startswith(">")]
    if not lines:
        raise SystemExit(f"No sequence lines found in FASTA file {path}.")
    return "".join(lines)


def validate_sequence(sequence: str) -> str:
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("Sequence is empty.")

    # Basic validation: standard 20 amino acids
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    bad = {aa for aa in sequence if aa not in valid}
    if bad:
        raise ValueError(
            f"Invalid amino acid characters in sequence: {''.join(sorted(bad))}"
        )
    return sequence


def load_model(
    device: torch.device,
    chunk_size: int | None = None,
    use_fp16: bool = False,
) -> EsmForProteinFolding:
    """
    Load ESMFold once, move to device, configure chunking + fp16, and enable dropout.
    """
    t0 = time.time()
    from_kwargs = {"low_cpu_mem_usage": True}

    # Load ESMFold from Hugging Face
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        **from_kwargs,
    )

    # Optional chunking: trades speed for lower peak memory
    if chunk_size is not None:
        try:
            model.trunk.set_chunk_size(chunk_size)
            print(f"Using trunk chunk_size={chunk_size}", file=sys.stderr)
        except AttributeError:
            print(
                "Warning: model.trunk.set_chunk_size not available on this model",
                file=sys.stderr,
            )

    # Move to device
    model = model.to(device)

    # Put the ESM transformer trunk in fp16 on CUDA only
    # (structure module + heads stay in fp32 to avoid numerical issues)
    if device.type == "cuda" and use_fp16:
        model.esm = model.esm.half()
        torch.backends.cuda.matmul.allow_tf32 = True  # nice perf boost on 40xx

    model.eval()
    enable_dropout(model)  # keep dropout active if you want stochasticity

    t1 = time.time()
    print(f"Model load time: {t1 - t0:.2f} sec", file=sys.stderr)
    print(f"Using device: {device}", file=sys.stderr)

    return model


def infer_pdb(model: EsmForProteinFolding, sequence: str) -> str:
    """Run inference and return a PDB string."""
    t0 = time.time()
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)
    t1 = time.time()
    print(f"Inference time: {t1 - t0:.2f} sec", file=sys.stderr)
    return pdb_str


def output_path_for_run(faa_path: Path, run_idx: int) -> Path:
    """
    Turn:
      /x/LMNA-...-wt.faa
    into:
      /x/LMNA-...-wt-<run_idx>.pdb
    """
    stem = faa_path.stem  # filename without suffix
    return faa_path.with_name(f"{stem}-{run_idx}.pdb")


def iter_faa_files(root: Path) -> list[Path]:
    """Return all .faa files under root (recursive), sorted for stable order."""
    return sorted(p for p in root.rglob("*.faa") if p.is_file())


def run_single(args: argparse.Namespace) -> None:
    device = pick_device()
    model = load_model(device=device, chunk_size=args.chunk_size, use_fp16=args.fp16)

    if args.sequence:
        seq = validate_sequence(args.sequence)
    else:
        seq = validate_sequence(load_first_fasta_sequence(args.fasta))

    pdb_str = infer_pdb(model, seq)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(pdb_str)
    print(f"Wrote PDB to {args.output}", file=sys.stderr)

def archive_path_for_faa(faa_path: Path) -> Path:
    # e.g. LMNA-...-es1.faa -> LMNA-...-es1.pdbs.tar.gz
    return faa_path.with_suffix(".pdbs.tar.gz")


def pdb_paths_for_faa(faa_path: Path, runs: int) -> list[Path]:
    return [output_path_for_run(faa_path, r) for r in range(runs)]


def all_pdbs_exist(faa_path: Path, runs: int) -> bool:
    return all(p.exists() for p in pdb_paths_for_faa(faa_path, runs))


def create_archive(faa_path: Path, runs: int, overwrite: bool) -> Path | None:
    """
    Create .tar.gz containing all run PDBs for the given .faa.
    Returns the archive path if created/exists, otherwise None.
    """
    pdbs = pdb_paths_for_faa(faa_path, runs)
    if not all(p.exists() for p in pdbs):
        return None  # not ready

    archive_path = archive_path_for_faa(faa_path)

    # If archive already exists and we're not overwriting, treat as "done"
    if archive_path.exists() and not overwrite:
        return archive_path

    # Create/overwrite archive
    with tarfile.open(archive_path, "w:gz") as tar:
        for p in pdbs:
            tar.add(p, arcname=p.name)

    # Verify it exists and is non-empty-ish
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        raise RuntimeError(f"Archive creation failed: {archive_path}")

    print(f"  ✓ archived: {archive_path}", file=sys.stderr)
    return archive_path


def delete_archived_pdbs(faa_path: Path, runs: int) -> int:
    """
    Delete PDBs for this .faa (only if they exist).
    Returns number deleted.
    """
    deleted = 0
    for p in pdb_paths_for_faa(faa_path, runs):
        if p.exists():
            p.unlink()
            deleted += 1
    if deleted:
        print(f"  ✓ deleted {deleted} PDBs for {faa_path.name}", file=sys.stderr)
    return deleted


def maybe_archive_and_cleanup(faa_path: Path, runs: int, overwrite: bool) -> bool:
    """
    If all PDBs exist, ensure archive exists (or overwrite), then delete PDBs.
    Returns True if archive+cleanup completed, else False.
    """
    if not all_pdbs_exist(faa_path, runs):
        return False

    archive_path = create_archive(faa_path, runs, overwrite=overwrite)
    if archive_path is None:
        return False

    # Only delete if archive definitely exists
    if not archive_path.exists():
        raise RuntimeError(f"Expected archive missing after creation: {archive_path}")

    delete_archived_pdbs(faa_path, runs)
    return True


def run_esp_esp(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    bin_root = (data_root / args.bin).resolve()

    if not bin_root.exists() or not bin_root.is_dir():
        raise SystemExit(f"Bin directory not found: {bin_root}")

    faa_files = iter_faa_files(bin_root)
    if not faa_files:
        raise SystemExit(f"No .faa files found under: {bin_root}")

    # --- First: sweep/repair pass (archive+cleanup anything already complete) ---
    swept = 0
    for faa_path in faa_files:
        try:
            if maybe_archive_and_cleanup(
                faa_path=faa_path,
                runs=args.runs,
                overwrite=args.overwrite,
            ):
                swept += 1
        except Exception as e:
            print(f"[sweep] Warning: {faa_path} (error: {e})", file=sys.stderr)

    if swept:
        print(f"[sweep] Archived+cleaned {swept} completed .faa entries.", file=sys.stderr)

    device = pick_device()
    model = load_model(device=device, chunk_size=args.chunk_size, use_fp16=args.fp16)

    print(f"Found {len(faa_files)} .faa files under {bin_root}", file=sys.stderr)
    print(f"Runs per file: {args.runs}", file=sys.stderr)
    print(f"Default behavior: skip existing PDBs; use --overwrite to recompute", file=sys.stderr)

    total_written = 0
    for i, faa_path in enumerate(faa_files, start=1):
        # If already archived and we're not overwriting, we can skip folding entirely
        archive_path = archive_path_for_faa(faa_path)
        if archive_path.exists() and not args.overwrite:
            # still: if there are stray PDBs around, clean them up safely
            try:
                delete_archived_pdbs(faa_path, args.runs)
            except Exception as e:
                print(f"[{i}/{len(faa_files)}] Warning: cleanup failed for {faa_path} ({e})", file=sys.stderr)
            print(f"[{i}/{len(faa_files)}] Skipping (archive exists): {archive_path.name}", file=sys.stderr)
            continue

        # Load sequence
        try:
            seq = validate_sequence(load_first_fasta_sequence(faa_path))
        except Exception as e:
            print(f"[{i}/{len(faa_files)}] Skipping {faa_path} (error: {e})", file=sys.stderr)
            continue

        print(f"[{i}/{len(faa_files)}] Folding {faa_path} (len={len(seq)})", file=sys.stderr)

        # Generate missing PDBs (or overwrite all if requested)
        for r in range(args.runs):
            out_path = output_path_for_run(faa_path, r)

            if out_path.exists() and not args.overwrite:
                print(f"  - exists, skipping: {out_path.name}", file=sys.stderr)
                continue

            pdb_str = infer_pdb(model, seq)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(pdb_str)
            total_written += 1
            print(f"  - wrote: {out_path.name}", file=sys.stderr)

        # After processing this .faa: archive if complete, then delete PDBs
        try:
            maybe_archive_and_cleanup(
                faa_path=faa_path,
                runs=args.runs,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"[{i}/{len(faa_files)}] Warning: archive/cleanup failed for {faa_path} ({e})", file=sys.stderr)

    print(f"Done. Wrote {total_written} PDB files.", file=sys.stderr)



def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Fold amino acid sequences with ESMFold (single or batch esp-esp mode)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- single mode (your original behavior) ----
    p_single = sub.add_parser("single", help="Fold a single sequence (sequence or fasta) to one PDB.")
    group = p_single.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", type=str, help="Amino acid sequence as a single string.")
    group.add_argument("--fasta", type=Path, help="Path to a FASTA file (first sequence used).")
    p_single.add_argument("--output", type=Path, required=True, help="Output PDB file path.")
    p_single.add_argument("--chunk-size", type=int, default=128, help="ESMFold trunk chunk size.")
    p_single.add_argument("--fp16", action="store_true", help="Use fp16 for ESM trunk on CUDA.")
    p_single.set_defaults(func=run_single)

    # ---- esp-esp batch mode ----
    p_batch = sub.add_parser(
        "esp-esp",
        help="Batch mode: find all .faa under <data-root>/<bin>/ and fold each N times to PDBs beside the .faa."
    )
    p_batch.add_argument(
        "--bin",
        required=True,
        type=str,
        help="Bin name such as 501-600, 101-200, 201-300, etc.",
    )
    p_batch.add_argument(
        "--data-root",
        required=True,
        type=str,
        help="Directory that contains the bin folders (e.g. esp-esp/data/protein-folding/mane-plus-clinical).",
    )
    p_batch.add_argument(
        "--runs",
        type=int,
        default=10,
        help="How many stochastic folds to run per .faa (default: 10).",
    )
    p_batch.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDB files (default: skip existing).",
    )
    p_batch.add_argument("--chunk-size", type=int, default=128, help="ESMFold trunk chunk size.")
    p_batch.add_argument("--fp16", action="store_true", help="Use fp16 for ESM trunk on CUDA.")
    p_batch.set_defaults(func=run_esp_esp)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
