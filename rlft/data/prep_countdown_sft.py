# ========== FILE 2: prep_countdown_sft.py ==========
"""
Prepare Countdown SFT dataset for supervised fine-tuning.
Run:
    python -m rlft.data.prep_countdown_sft \
        --dataset Asap7772/cog_behav_all_strategies \
        --model Qwen/Qwen2.5-0.5B \
        --outdir data/processed/countdown_sft
"""
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from pathlib import Path
import argparse, json

PAD_IGNORE = -100
PROMPT_MAX, RESP_MAX = 256, 1024
SEQ_LEN = PROMPT_MAX + RESP_MAX

def build_token_fn(tokenizer):
    def fn(batch):
        full = [f"{p}\n{r}" for p, r in zip(batch["prompt"], batch["response"])]
        enc = tokenizer(full, max_length=SEQ_LEN, truncation=True, padding="max_length", return_tensors="pt")
        labels = enc.input_ids.clone()
        for i, p in enumerate(batch["prompt"]):
            p_len = len(tokenizer(p, max_length=PROMPT_MAX, truncation=True, add_special_tokens=False).input_ids)
            labels[i, :p_len] = PAD_IGNORE
        enc["labels"] = labels
        return enc
    return fn


def preprocess(dataset: str, model: str, outdir: str):
    ds = load_dataset(dataset, split="train")
    ds = ds.map(lambda r: {"prompt": r["input"], "response": r["target"]})
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds_tok = ds.map(build_token_fn(tok), batched=True, remove_columns=["prompt", "response"])
    Path(outdir).mkdir(parents=True, exist_ok=True)
    ds_tok.train_test_split(test_size=0.1).save_to_disk(outdir)
    print(f"✅ Processed Countdown SFT to {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Asap7772/cog_behav_all_strategies")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--outdir", default="data/processed/countdown_sft")
    args = p.parse_args()
    preprocess(args.dataset, args.model, args.outdir)