#!/usr/bin/env python
import os

# Avoid OpenMP crash on macOS with multiple libomp copies
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Don't go online to check newer versions
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import EsmForProteinFolding

# enable dropout to make folding non-deterministic
# this helps to estimate model certainty over multiple runs
# and enables comparison across two structures without
# folding stochasticity determining the outcome
def enable_dropout(model):
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
    #return torch.device("cpu")   # hard-force CPU to circumvent OOM issues on MPS
    
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

    lines = [l.strip() for l in text if l and not l.startswith(">")]
    if not lines:
        raise SystemExit(f"No sequence lines found in FASTA file {path}.")
    return "".join(lines)


def fold_sequence_to_pdb(sequence: str, device: torch.device) -> str:
    """
    Run ESMFold (Hugging Face port) on a single amino acid sequence
    and return a PDB file as a string.
    """
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

    print(f"Sequence length: {len(sequence)}", file=sys.stderr)
    print(f"Using device: {device}", file=sys.stderr)

    # Load ESMFold from Hugging Face
    # TIMING START
    t_load0 = time.time()
    
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    #model.trunk.set_chunk_size(32) # trades lower peak memory for slower inference, use 64, 32, 16, 8 for more savings
    model = model.to(device)
    model.eval()
    enable_dropout(model)   # <-- enables dropout

    t_load1 = time.time()
    print(f"Model load time: {t_load1 - t_load0:.2f} sec", file=sys.stderr)

    # INFERENCE TIMER
    t_inf0 = time.time()
    
    # High-level helper: returns a PDB string
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)

    t_inf1 = time.time()
    print(f"Inference time: {t_inf1 - t_inf0:.2f} sec", file=sys.stderr)

    # Optional total time
    print(f"Total time: {t_inf1 - t_load0:.2f} sec", file=sys.stderr)

    return pdb_str


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fold an amino acid sequence with ESMFold and write a PDB file."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sequence",
        type=str,
        help="Amino acid sequence as a single string (e.g. 'ACDEFGHIKLMNPQRSTVWY').",
    )
    group.add_argument(
        "--fasta",
        type=Path,
        help="Path to a FASTA file. Only the first sequence will be used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PDB file path.",
    )

    args = parser.parse_args(argv)

    # Get sequence either from CLI or from FASTA
    if args.sequence:
        seq = args.sequence
    else:
        seq = load_first_fasta_sequence(args.fasta)

    device = pick_device()
    pdb_str = fold_sequence_to_pdb(seq, device=device)

    # Write PDB
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(pdb_str)
    print(f"Wrote PDB to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
