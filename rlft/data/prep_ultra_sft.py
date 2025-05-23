# rlft/data/prep_ultra_sft.py
"""
Pre-tokenise UltraFeedback (preference) dataset for SFT.

• Use only (prompt, chosen) pairs (like filtered behavior cloning)
• Truncate to 256-token prompt + 1024-token response
• Mask prompt tokens with PAD_IGNORE (=-100) so loss only applies to response
• Save as HF-arrow (train/val splits)

Run:
    python -m rlft.data.prep_ultra_sft \
        --dataset HuggingFaceH4/ultrafeedback_binarized \
        --model Qwen/Qwen2.5-0.5B \
        --outdir data/processed/ultra_sft \
        --val 0.1
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict

import argparse, json, logging, gc
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer

# ─────────────────────── logging ───────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─────────────────────── constants ───────────────────────
PROMPT_MAX = 256
RESP_MAX = 1024
SEQ_LEN = PROMPT_MAX + RESP_MAX
PAD_IGNORE = -100


# ─────────── extract (prompt, response) for SFT ───────────
def extract_ultra_sft_pair(row: Dict) -> Dict[str, str]:
    return {"prompt": row["prompt"], "response": row["chosen"]}


# ─────────────── build tokenizer and tokenisation fn ───────────────
def build_tokeniser(model_name: str):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenise_factory(tok):
    def _fn(batch):
        merged = [f"{p}\n{r}" for p, r in zip(batch["prompt"], batch["response"])]
        enc = tok(
            merged,
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        labels = enc.input_ids.clone()
        for i, p in enumerate(batch["prompt"]):
            p_len = len(tok(p, truncation=True, max_length=PROMPT_MAX, add_special_tokens=False).input_ids)
            p_len = min(p_len, SEQ_LEN)
            labels[i, :p_len] = PAD_IGNORE
        enc["labels"] = labels
        return enc
    return _fn


# ─────────────── main pipeline ───────────────
def preprocess(
    *,
    dataset: str = "HuggingFaceH4/ultrafeedback_binarized",
    model: str = "Qwen/Qwen2.5-0.5B",
    outdir: Path | str = "data/processed/ultra_sft",
    batch: int = 64,
    chunk: int = 2000,
    val: float = 0.1,
    peek: int | None = None,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(dataset, split="train")
    if peek:
        raw = raw.select(range(peek))
        log.warning("peek mode: using first %d examples", len(raw))

    tok = build_tokeniser(model)
    tok_fn = tokenise_factory(tok)

    tokenised_parts = []
    for start in range(0, len(raw), chunk):
        sub = raw.select(range(start, min(start + chunk, len(raw))))
        pairs = [extract_ultra_sft_pair(r) for r in sub]
        ds_pairs = Dataset.from_list(pairs)

        ds_tok = ds_pairs.map(
            tok_fn,
            batched=True,
            batch_size=batch,
            remove_columns=["prompt", "response"],
            desc=f"tokenising {start}->{start+chunk}",
        )
        tokenised_parts.append(ds_tok)

        del sub, pairs, ds_pairs
        gc.collect()
        log.info("processed %d / %d raw rows", start + chunk, len(raw))

    full = concatenate_datasets(tokenised_parts)

    # Split + save
    split = full.train_test_split(test_size=val, seed=42)
    (outdir / "train").mkdir(exist_ok=True, parents=True)
    (outdir / "val").mkdir(exist_ok=True, parents=True)
    split["train"].save_to_disk(outdir / "train")
    split["test"].save_to_disk(outdir / "val")

    with open(outdir / "config.json", "w") as f:
        json.dump(
            {
                "model": model,
                "seq_len": SEQ_LEN,
                "prompt_max": PROMPT_MAX,
                "resp_max": RESP_MAX,
            },
            f,
            indent=2,
        )

    log.info("✅ wrote %d train / %d val samples to %s",
             len(split["train"]), len(split["test"]), outdir)


# ─────────────── CLI Entrypoint ───────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--outdir", default="data/processed/ultra_sft")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--peek", type=int, default=None, help="use first N rows only")
    args = p.parse_args()
    preprocess(**vars(args))


if __name__ == "__main__":
    main()
