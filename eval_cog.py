"""
eval_cog.py
───────────
Loads a fine-tuned checkpoint, generates answers for the Countdown *test* split,
computes BLEU-4, ROUGE-L, BERTScore (F1), writes a CSV with prompt / reference / prediction

$ python eval_cog.py --ckpt_dir ./sft_ckpt --samples 200
"""

import argparse, csv, math, torch, textwrap
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from countdown import extract_solution
import json

SYSTEM_MSG = "You are a helpful assistant skilled at solving Countdown maths puzzles."

def build_chat_prompt(raw_q: str) -> str:
    """
    Build the chat string seen at train-time:
        <|system|> … <|user|> … <|assistant|>
    `raw_q` is dataset["query"] which already contains the narrative
    and 'Assistant: Let me solve this step by step.'
    """
    return (
        f"<|system|>\n{SYSTEM_MSG}\n\n"   # system
        f"<|user|>\n{raw_q.strip()}\n\n"
        f"<|assistant|>\n"
    )


@torch.no_grad()
def generate(model, tok, prompts, device, max_new=256, bsz=4):
    outs = []
    for i in range(0, len(prompts), bsz):
        batch_text = prompts[i:i+bsz]
        enc = tok(batch_text, return_tensors="pt",
                  padding=True).to(device)
        gen = model.generate(**enc,
                             max_new_tokens=max_new,
                             temperature=0.0)
        for j in range(len(batch_text)):
            ans = tok.decode(gen[j][enc.input_ids.shape[1]:],
                             skip_special_tokens=False)
            outs.append(ans)
    return outs

def extract_nums_target(prompt_text):
    """Extract the [numbers] and the target number from prompt text."""
    # Extract numbers inside square brackets
    num_match = re.search(r'Using the numbers \[(.*?)\]', prompt_text)
    nums = [int(n) for n in num_match.group(1).strip().split()] if num_match else []

    # Extract target number
    target_match = re.search(r'equals (\d+)', prompt_text)
    target = int(target_match.group(1)) if target_match else None

    return nums, target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--samples", type=int, default=None,
                    help="limit to first N test examples")
    ap.add_argument("--max_new", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok   = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    model = (AutoModelForCausalLM
             .from_pretrained(args.ckpt_dir, trust_remote_code=True)
             .to(device).eval())

    test_ds = load_dataset("Asap7772/cog_behav_all_strategies",
                           split="test")
    if args.samples:
        test_ds = test_ds.select(range(args.samples))

    prompts = [build_chat_prompt(ex["query"]) for ex in test_ds]
    refs    = [ex["completion"].strip() for ex in test_ds]

    print(f"▶ generating {len(prompts)} samples …")
    preds = generate(model, tok, prompts, device,
                     max_new=args.max_new, bsz=4)


    bleu = corpus_bleu(preds, [refs]).score

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds))
    rouge_l /= len(refs)

    P, R, F = bert_score(preds, refs, lang="en", verbose=False)
    bert_f1 = F.mean().item() * 100

    print("Metrics")
    print(f"BLEU-4   : {bleu:6.2f}")
    print(f"ROUGE-L  : {rouge_l*100:6.2f}")
    print(f"BERTScore: {bert_f1:6.2f}")


    out_csv = Path("results") / f"eval_{Path(args.ckpt_dir).name}.csv"
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "reference", "prediction"])
        w.writerows(zip(prompts, refs, preds))
    print(f"✔ results written to {out_csv}")

    #output JSON
    out_json = Path("results") / f"eval_{Path(args.ckpt_dir).name}.jsonl"
    records = []
    for p, pred in zip(prompts, preds):
        nums, target = extract_nums_target(p)
        solution = extract_solution(pred)
        if nums and target and solution:  # only if all pieces are there
            records.append({
                "num": nums,
                "response": solution,
                "target": target
            })
    with open(out_json, "w") as f:
        for r in records:
            json.dump(r, f)
            f.write("\n")
    print(f"✔ JSONL results written to {out_json}")

if __name__ == "__main__":
    main()
