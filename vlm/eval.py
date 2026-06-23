"""Evaluation harness for a trained VLM: VQA accuracy + caption metrics.

    python -m vlm.eval --ckpt checkpoints/stage2 --task vqa \
        --source '{"type":"hf","id":"lmms-lab/VQAv2","split":"validation"}' --limit 5000

    python -m vlm.eval --ckpt checkpoints/stage2 --task caption \
        --source '{"type":"hf","id":"nlphuji/flickr30k","split":"test",
                   "image_col":"image","caption_col":"caption"}'

VQA uses the official accuracy: acc(pred) = min(#matching annotator answers / 3, 1),
averaged over questions, after canonical answer normalization. Caption uses corpus
BLEU-4 (sacrebleu) and CIDEr when pycocoevalcap is installed.
"""

from __future__ import annotations

import argparse
import json
import re

import torch
from tqdm import tqdm

from .data import num_image_tokens_for
from .infer import load_model

# --- VQA answer normalization (port of the official VQAv2 eval) ----------------------------
_CONTRACTIONS = {"dont": "don't", "isnt": "isn't", "arent": "aren't", "cant": "can't",
                 "wont": "won't", "wasnt": "wasn't", "doesnt": "doesn't"}
_ARTICLES = {"a", "an", "the"}
_NUMBER = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
           "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!.]")


def normalize_answer(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    ans = _PUNCT.sub("", ans)
    toks = [_NUMBER.get(t, _CONTRACTIONS.get(t, t)) for t in ans.split() if t not in _ARTICLES]
    return " ".join(toks).strip()


def vqa_accuracy(pred: str, gt_answers: list[str]) -> float:
    pred = normalize_answer(pred)
    gts = [normalize_answer(a) for a in gt_answers]
    matches = sum(pred == g for g in gts)
    return min(matches / 3.0, 1.0)


# --- data loading -------------------------------------------------------------------------
def load_eval_records(source: dict, limit: int | None):
    if source["type"] == "jsonl":
        with open(source["path"]) as f:
            recs = [json.loads(line) for line in f]
    elif source["type"] == "hf":
        from datasets import load_dataset
        recs = list(load_dataset(source["id"], split=source.get("split", "validation")))
    else:
        raise ValueError(source["type"])
    return recs[:limit] if limit else recs


# --- generation ---------------------------------------------------------------------------
def _build_input(tokenizer, n_img, prompt):
    messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    img_id = tokenizer.convert_tokens_to_ids("<image>")
    pos = ids.index(img_id)
    return ids[:pos] + [img_id] * n_img + ids[pos + 1:]


@torch.no_grad()
def generate_answer(model, tokenizer, processor, n_img, image, prompt, device, max_new_tokens):
    from PIL import Image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device, model.projector.fc1.weight.dtype)
    ids = torch.tensor([_build_input(tokenizer, n_img, prompt)], device=device)
    out = model.generate(input_ids=ids, pixel_values=pixel_values,
                         max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# --- metrics ------------------------------------------------------------------------------
def caption_metrics(predictions, references):
    """predictions: list[str]; references: list[list[str]]."""
    out = {}
    try:
        import sacrebleu
        # sacrebleu wants references transposed: refs[i] is the i-th ref for every example.
        max_refs = max(len(r) for r in references)
        refs_t = [[(r[i] if i < len(r) else r[0]) for r in references] for i in range(max_refs)]
        out["BLEU-4"] = round(sacrebleu.corpus_bleu(predictions, refs_t).score, 2)
    except ImportError:
        out["BLEU-4"] = "install sacrebleu"
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [p] for i, p in enumerate(predictions)}
        out["CIDEr"] = round(Cider().compute_score(gts, res)[0] * 100, 2)
    except ImportError:
        out["CIDEr"] = "install pycocoevalcap"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", choices=["vqa", "caption"], required=True)
    ap.add_argument("--source", required=True, help="JSON dataset spec")
    ap.add_argument("--base-lm", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--out", default=None, help="optional path to dump per-example predictions")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(args.ckpt, base_lm=args.base_lm)
    model.to(device).eval()
    n_img = num_image_tokens_for(model.vision_tower.config)
    records = load_eval_records(json.loads(args.source), args.limit)

    preds, refs, scores, dump = [], [], [], []
    max_new = args.max_new_tokens or (16 if args.task == "vqa" else 64)

    for rec in tqdm(records, desc=f"eval/{args.task}"):
        if args.task == "vqa":
            q = rec.get("question") or rec["text"]
            pred = generate_answer(model, tokenizer, processor, n_img, rec["image"],
                                   f"{q}\nAnswer in one word or phrase.", device, max_new)
            gts = rec.get("answers") or rec.get("answer")
            gts = [a["answer"] if isinstance(a, dict) else a for a in (gts if isinstance(gts, list) else [gts])]
            acc = vqa_accuracy(pred, gts)
            scores.append(acc)
            dump.append({"q": q, "pred": pred, "gt": gts, "acc": acc})
        else:
            cap = rec.get("caption") or rec.get("captions")
            cap = cap if isinstance(cap, list) else [cap]
            pred = generate_answer(model, tokenizer, processor, n_img, rec["image"],
                                   "Describe this image in one sentence.", device, max_new)
            preds.append(pred)
            refs.append(cap)
            dump.append({"pred": pred, "refs": cap})

    print("\n=== Results ===")
    if args.task == "vqa":
        print(f"VQA accuracy: {100 * sum(scores) / len(scores):.2f}  (n={len(scores)})")
    else:
        for k, v in caption_metrics(preds, refs).items():
            print(f"{k}: {v}")
        print(f"(n={len(preds)})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"Wrote per-example predictions to {args.out}")


if __name__ == "__main__":
    main()
