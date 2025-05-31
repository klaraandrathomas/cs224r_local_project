#!/usr/bin/env python3
"""
eval_countdown.py
─────────────────
• Loads a fine-tuned checkpoint
• Generates answers for Countdown 3-to-4-number prompts
• Scores each answer with compute_score from countdown.py
• Supports partial evaluation via --batches or --samples
"""

import argparse, textwrap, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown_dataloader import get_eval_dataloader   # uses SmolTalk prompt builder
from countdown import compute_score                    # official rule-based scorer


# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, tok, loader, device, max_batches=None, max_samples=None):
    model.eval()

    total, hits, reward_sum = 0, 0, 0.0
    sample_cap = max_samples or float("inf")

    for b_idx, batch in enumerate(loader, 1):
        if max_batches and b_idx > max_batches:
            break
        if total >= sample_cap:
            break

        # move tensors to GPU if available
        cuda_inp = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                    if k in ("input_ids", "attention_mask")}

        out = model.generate(**cuda_inp,
                             max_new_tokens=64,
                             temperature=0.0)


        for i in range(len(batch["input_ids"])):
            if total >= sample_cap:
                break

            prompt_ids  = batch["input_ids"][i]
            answer_ids  = out[i][len(prompt_ids):]

            prompt_txt  = tok.decode(prompt_ids, skip_special_tokens=True)
            answer_txt  = tok.decode(answer_ids, skip_special_tokens=True)
            print(answer_txt)

            gt = {"numbers": batch["nums"][i],   # supplied by EvalCollator
                  "target":  batch["target"][i]}
            s  = compute_score(answer_txt, gt)

            reward_sum += s
            hits       += int(s == 1.0)
            total      += 1

            # ───── pretty print ─────
            print(f"\n── Batch {b_idx} • Sample {i+1} ──")
            print("Prompt:")
            print(textwrap.indent(prompt_txt, "  "))
            print("Answer:")
            print(textwrap.indent(answer_txt, "  "))
            print(f"Score : {s:.3f}")

    avg_reward = reward_sum / total
    exact_acc  = hits / total
    return avg_reward, exact_acc, total


# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Path to fine-tuned HF checkpoint")
    ap.add_argument("--batches", type=int, default=None,
                    help="Limit evaluation to N dataloader batches")
    ap.add_argument("--samples", type=int, default=None,
                    help="Limit evaluation to N total prompts")
    args = ap.parse_args()

    if args.batches is None and args.samples is None:
        args.batches = 1   # default: just 1 batch for a quick look

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n▶ loading model from {args.ckpt_dir} …")
    tok   = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_dir,
                                                 trust_remote_code=True).to(device)

    loader = get_eval_dataloader(tok, batch_size=8, max_len=1024)

    print(f"\n▶ evaluating {args.batches or '∞'} batches / {args.samples or '∞'} samples …\n")
    avg, exact, seen = evaluate(model, tok, loader, device,
                                max_batches=args.batches,
                                max_samples=args.samples)

    print("\n" + "="*60)
    print(f"Evaluated {seen} samples")
    print(f"Average reward : {avg:.4f}")
    print(f"Exact accuracy : {exact:.2%}")
    print("="*60)


if __name__ == "__main__":
    main()
