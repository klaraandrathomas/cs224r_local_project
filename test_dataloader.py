#!/usr/bin/env python3
"""
test_dataloader.py – sanity-check the SFT dataloader

Usage
-----
python test_dataloader.py --model_name Qwen/Qwen2.5-0.5B --batches 2

What it does
------------
* loads the first N batches from the Warm-Start dataset
* reconstructs the decoded text for each sample
* prints:
    - sample index
    - prompt (user side)
    - response (assistant side)
    - input_ids length
    - first 20 token IDs
"""

import argparse, textwrap, torch
from countdown_dataloader import get_sft_dataloader  # must include the tok_sft fix
from transformers import AutoTokenizer

# -------------------------------------------------------------------------
def split_prompt_response(decoded: str):
    """Return (prompt, response) split on '<|assistant|>' marker."""
    tag = "<|assistant|>"
    if tag in decoded:
        prompt, resp = decoded.split(tag, 1)
        # remove system header from prompt for readability
        if prompt.startswith("<|system|>"):
            prompt = prompt.split("\n\n", 1)[-1]
        return prompt.strip(), resp.strip()
    return "", decoded.strip()

# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--batches", type=int, default=1,
                    help="How many batches to inspect")
    ap.add_argument("--bsz", type=int, default=2)
    args = ap.parse_args()

    # make dataloader & tokenizer
    tok, loader = get_sft_dataloader(args.model_name, batch_size=args.bsz)
    tokenizer = tok  # same object

    print(f"\n🔍 Inspecting {args.batches} batch(es) "
          f"with batch-size {args.bsz}\n")

    for b_idx, batch in enumerate(loader, 1):
        if b_idx > args.batches:
            break

        print("="*80)
        print(f"BATCH {b_idx}")
        print("="*80)

        # Work on CPU for printing
        for s_idx in range(len(batch["input_ids"])):
            ids = batch["input_ids"][s_idx].tolist()
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            prompt_txt, resp_txt = split_prompt_response(decoded)

            print(f"\n--- sample {s_idx} ---")
            print("Prompt :")
            print(textwrap.indent(prompt_txt, "  "))
            print("Response :")
            print(textwrap.indent(resp_txt, "  "))
            print(f"input_ids len: {len(ids)}")
            print(f"first 20 ids : {ids[:20]}\n")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
