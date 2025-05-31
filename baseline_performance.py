"""
baseline_eval_cog.py
────────────────────
Evaluate the *base* model (no fine-tuning) on the Countdown Warm-Start test
split and report BLEU-4, ROUGE-L, and BERTScore.

python baseline_eval_cog.py --model_name Qwen/Qwen2.5-0.5B --samples 200
"""

import argparse, csv, math, torch, textwrap
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

SYSTEM_MSG = ("You are a helpful assistant skilled at solving "
              "Countdown maths puzzles.")


def build_chat_prompt(raw_q: str) -> str:
    """Reproduce the training chat string for a single example."""
    return (
        f"<|system|>\n{SYSTEM_MSG}\n\n"
        f"<|user|>\n{raw_q.strip()}\n\n"
        f"<|assistant|>\n"
    )

@torch.no_grad()
def generate(model, tok, prompts, device, max_new=256, bsz=4):
    preds = []
    for i in range(0, len(prompts), bsz):
        enc = tok(prompts[i:i+bsz], return_tensors="pt",
                  padding=True).to(device)
        gen = model.generate(**enc,
                             max_new_tokens=max_new,
                             temperature=0.0)
        for j in range(len(enc.input_ids)):
            preds.append(tok.decode(
                gen[j][enc.input_ids.shape[1]:],
                skip_special_tokens=False))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True,
                    help="base checkpoint to evaluate")
    ap.add_argument("--samples", type=int, default=None,
                    help="limit to first N test examples")
    ap.add_argument("--max_new", type=int, default=256,
                    help="max_new_tokens for generation")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n▶ loading base model {args.model_name}")
    tok   = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = (AutoModelForCausalLM
             .from_pretrained(args.model_name, trust_remote_code=True)
             .to(device).eval())

    test_ds = load_dataset("Asap7772/cog_behav_all_strategies",
                           split="test")
    if args.samples:
        test_ds = test_ds.select(range(args.samples))

    prompts = [build_chat_prompt(ex["query"])   for ex in test_ds]
    refs    = [ex["completion"].strip()         for ex in test_ds]

    print(f"▶ generating {len(prompts)} samples …")
    preds = generate(model, tok, prompts, device,
                     max_new=args.max_new, bsz=4)


    bleu = corpus_bleu(preds, [refs]).score

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds))
    rouge_l /= len(refs)

    P, R, F = bert_score(preds, refs, lang="en", verbose=False)
    bert_f1 = F.mean().item() * 100

    print("BASELINE METRICS")
    print(f"BLEU-4   : {bleu:6.2f}")
    print(f"ROUGE-L  : {rouge_l*100:6.2f}")
    print(f"BERTScore: {bert_f1:6.2f}")

    # save detailed CSV
    out_csv = Path("results") / f"baseline_{args.model_name.replace('/','_')}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "reference", "prediction"])
        w.writerows(zip(prompts, refs, preds))
    print(f"baseline predictions saved to {out_csv}")

if __name__ == "__main__":
    main()
