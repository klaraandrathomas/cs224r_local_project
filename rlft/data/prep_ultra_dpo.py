# ========== FILE 1: prep_ultra_dpo.py ==========
"""
Process UltraFeedback preference dataset for DPO.
Output: (prompt, chosen, rejected) triplets for training DPO.
Run:
    python -m rlft.data.prep_ultra_dpo \
        --dataset HuggingFaceH4/ultrafeedback_binarized \
        --model Qwen/Qwen2.5-0.5B \
        --outdir data/processed/ultra_dpo
"""
from datasets import load_dataset, Dataset
from pathlib import Path
import argparse, json


def preprocess_ultra_dpo(dataset: str, outdir: str):
    ds = load_dataset(dataset, split="train_dpo")
    triplets = ds.map(lambda ex: {
        "prompt": ex["prompt"],
        "chosen": ex["chosen"],
        "rejected": ex["rejected"]
    })
    Path(outdir).mkdir(parents=True, exist_ok=True)
    triplets.save_to_disk(outdir)
    print(f"✅ Saved Ultra DPO triplets to {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    p.add_argument("--outdir", default="data/processed/ultra_dpo")
    args = p.parse_args()
    preprocess_ultra_dpo(args.dataset, args.outdir)
