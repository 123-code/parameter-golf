"""
Colab/T4 runner for Parameter Golf architecture comparison.
Run this in a Google Colab notebook with a T4 GPU runtime.

Usage in Colab:
    !git clone https://github.com/<your-fork>/parameter-golf.git
    %cd parameter-golf
    !pip install numpy tqdm torch sentencepiece huggingface-hub datasets
    !python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

    # Run baseline (9 unique layers, dim=512):
    !python colab_train.py --arch baseline --iterations 2000

    # Run looped (3 blocks x 4 loops = 12 effective layers, dim=640):
    !python colab_train.py --arch looped --iterations 2000

    # Compare the val_bpb numbers at the same step count to decide which scales better.
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Colab-friendly Parameter Golf trainer")
    parser.add_argument("--arch", choices=["baseline", "looped"], default="looped",
                        help="Architecture to train")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Number of training iterations")
    parser.add_argument("--batch-tokens", type=int, default=131072,
                        help="Batch tokens (reduced for T4 16GB VRAM)")
    parser.add_argument("--val-every", type=int, default=200,
                        help="Validate every N steps")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length")
    parser.add_argument("--dim", type=int, default=None,
                        help="Model dimension (default: 512 for baseline, 640 for looped)")
    args = parser.parse_args()

    # T4-friendly settings: smaller batch, no wallclock cap
    os.environ["TRAIN_BATCH_TOKENS"] = str(args.batch_tokens)
    os.environ["MAX_WALLCLOCK_SECONDS"] = "0"
    os.environ["ITERATIONS"] = str(args.iterations)
    os.environ["VAL_LOSS_EVERY"] = str(args.val_every)
    os.environ["TRAIN_LOG_EVERY"] = "50"
    os.environ["TRAIN_SEQ_LEN"] = str(args.seq_len)

    if args.arch == "baseline":
        # Baseline: 9 layers, dim 512
        dim = args.dim or 512
        os.environ["MODEL_DIM"] = str(dim)
        os.environ["NUM_LAYERS"] = "9"
        os.environ["RUN_ID"] = f"colab_baseline_d{dim}_i{args.iterations}"
        print(f"\n{'='*60}")
        print(f"  BASELINE: 9 layers, dim={dim}")
        print(f"  Iterations: {args.iterations}, Batch tokens: {args.batch_tokens}")
        print(f"{'='*60}\n")
        import runpy
        runpy.run_path("train_gpt.py", run_name="__main__")

    elif args.arch == "looped":
        # Looped: 3 unique blocks x 4 loops = 12 effective layers
        dim = args.dim or 640
        os.environ["MODEL_DIM"] = str(dim)
        os.environ["MATRIX_LR"] = "0.02"
        os.environ["SCALAR_LR"] = "0.02"
        os.environ["GRAD_CLIP_NORM"] = "1.0"
        os.environ["RUN_ID"] = f"colab_looped_d{dim}_i{args.iterations}"
        print(f"\n{'='*60}")
        print(f"  LOOPED: 3 blocks x 4 loops = 12 effective layers, dim={dim}")
        print(f"  Iterations: {args.iterations}, Batch tokens: {args.batch_tokens}")
        print(f"{'='*60}\n")
        import runpy
        runpy.run_path("train_gpt_looped.py", run_name="__main__")


if __name__ == "__main__":
    main()
