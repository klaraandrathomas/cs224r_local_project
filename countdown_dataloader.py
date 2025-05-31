"""
countdown_dataloader.py

Functions to tokenize and buildm dataloaders
"""
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

_pad = DataCollatorWithPadding

CHAT_TEMPLATE = """<|system|>
You are a helpful assistant skilled at solving Countdown maths puzzles.

<|user|>
{prompt}

<|assistant|>
{response}"""


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = CHAT_TEMPLATE
    return tok


def _mask_labels(enc: Dict[str, Any], tok):
    asst = tok.convert_tokens_to_ids("<|assistant|>")
    labels = []
    for ids in enc["input_ids"]:
        lab = ids.copy()
        try:
            cut = ids.index(asst)
        except ValueError:
            cut = 0
        lab[: cut + 1] = [-100] * (cut + 1)
        labels.append(lab)
    enc["labels"] = labels
    return enc


def _extract_prompt_and_resp(example):
    q = example["query"]
    if "Assistant:" in q:
        prompt = q.split("Assistant:", 1)[0].strip()
    else:
        prompt = q.strip()
    response = example["completion"].strip()
    return prompt, response


#  Warm-Start SFT tokeniser
def tok_sft(batch, tok, max_len: int):
    system_msg = (
        "You are a helpful assistant skilled at solving Countdown maths puzzles."
    )

    all_input_ids, all_attention, all_labels = [], [], []

    for raw_q, comp in zip(batch["query"], batch["completion"]):
        # split query -> prompt / response
        prompt_txt = raw_q.split("Assistant:", 1)[0].strip() if "Assistant:" in raw_q else raw_q.strip()
        resp_txt   = comp.strip()
        # build chat sections
        prompt_part = (
            f"<|system|>\n{system_msg}\n\n"
            f"<|user|>\n{prompt_txt}\n\n"
            f"<|assistant|>\n"
        )

        # tokenise separately
        prompt_ids  = tok(prompt_part,  add_special_tokens=False).input_ids
        resp_ids    = tok(resp_txt,    add_special_tokens=False).input_ids

        if len(prompt_ids) + len(resp_ids) > max_len:
            resp_ids = resp_ids[: max_len - len(prompt_ids)]

        ids   = prompt_ids + resp_ids
        mask  = [1] * len(ids)  
        lbls  = [-100] * len(prompt_ids) + resp_ids 

        all_input_ids.append(ids)
        all_attention.append(mask)
        all_labels.append(lbls)

    return {
        "input_ids":      all_input_ids,
        "attention_mask": all_attention,
        "labels":         all_labels,
    }


def _build_eval_prompt(nums, tgt):
    nums_str = " ".join(map(str, nums))
    return (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the "
        "reasoning process in the mind and then provides the user with the answer.\n"
        f"User: Using the numbers [{nums_str}], create an equation that equals {tgt}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only "
        "be used once. Show your work in <think> </think> tags. And return the final "
        "answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\n"
        "Assistant: Let me solve this step by step."
    )


def tok_prompt_eval(batch, tok, max_len: int):
    system_msg = "You are a helpful assistant skilled at solving Countdown maths puzzles."

    chat_texts = []
    for nums, tgt in zip(batch["nums"], batch["target"]):
        user_prompt = _build_eval_prompt(nums, tgt)
        chat_texts.append(
            f"<|system|>\n{system_msg}\n\n"
            f"<|user|>\n{user_prompt}\n\n"
            f"<|assistant|>\n"
        )

    return tok(
        chat_texts,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )


class EvalCollator:
    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, batch):
        padded = self.pad([{k: ex[k] for k in ("input_ids", "attention_mask")} for ex in batch])
        padded["nums"]   = [ex["nums"]   for ex in batch]
        padded["target"] = [ex["target"] for ex in batch]
        return padded



def collate_sft(batch, pad_id: int):
    """
    batch : list of dicts returned by tok_sft
    returns: dict of torch.LongTensor, shape (B, L_max)
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)

    def pad(seq, value):
        return seq + [value] * (max_len - len(seq))

    out = {
        "input_ids":      [],
        "attention_mask": [],
        "labels":         [],
    }
    for ex in batch:
        out["input_ids"].append(pad(ex["input_ids"],      pad_id))
        out["attention_mask"].append(pad(ex["attention_mask"], 0))
        out["labels"].append(pad(ex["labels"],        -100))

    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


# Public dataloader init function
def get_sft_dataloader(model_name: str,
                       batch_size: int = 4,
                       max_len: int = 1024):
    """
    Returns
    -------
    tokenizer : transformers.AutoTokenizer
    loader    : torch.utils.data.DataLoader  (shuffled, ready for training)
    """
    tok = build_tokenizer(model_name)
    warm = load_dataset("Asap7772/cog_behav_all_strategies", split="train")

    warm_tok = warm.map(
        tok_sft,
        fn_kwargs={"tok": tok, "max_len": max_len},
        batched=True,
        remove_columns=warm.column_names,
        num_proc=8,
        load_from_cache_file=False,   # ← force remap
    )



    collate = DataCollatorWithPadding(tok)
    loader = DataLoader(warm_tok, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True,
                        collate_fn=lambda b, pad_id=tok.pad_token_id: collate_sft(b, pad_id), 
                        persistent_workers=True)
    return tok, loader



def get_eval_dataloader(tokenizer,
                        batch_size: int = 8,
                        max_len: int = 1024):
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    ds_tok = ds.map(
        tok_prompt_eval,
        fn_kwargs={"tok": tokenizer, "max_len": max_len},
        batched=True,
        num_proc=4,
        load_from_cache_file=False,   # 🔄 force re-tokenisation
    )

    return DataLoader(
        ds_tok,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=EvalCollator(tokenizer),
        num_workers=2,
        pin_memory=True,
    )
