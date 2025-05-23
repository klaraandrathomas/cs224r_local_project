# ========== FILE 3: prep_countdown_prompts.py ==========
"""
Extract prompts for Countdown RLOO sampling.
Run:
    python -m rlft.data.prep_countdown_prompts \
        --dataset Jiayi-Pan/Countdown-Tasks-3to4 \
        --outdir data/processed/countdown_prompts
"""
from datasets import load_dataset
from pathlib import Path
import argparse

def preprocess(dataset: str, outdir: str):
    ds = load_dataset(dataset, split="train")
    ds_prompt = ds.map(lambda r: {"prompt": r["input"]})
    Path(outdir).mkdir(parents=True, exist_ok=True)
    ds_prompt.save_to_disk(outdir)
    print(f"✅ Saved countdown prompts to {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Jiayi-Pan/Countdown-Tasks-3to4")
    p.add_argument("--outdir", default="data/processed/countdown_prompts")
    args = p.parse_args()
    preprocess(args.dataset, args.outdir)
