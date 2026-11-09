
import argparse
import gc
import glob
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels" / "build"))
sys.path.insert(0, str(ROOT / "model"))

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from model_utils import reorder_model_llama, reorder_model_qwen


def compute_perplexity(model, data, batch_size=1):
    """Exact MR-GPTQ compute_perplexity implementation."""
    device = next(model.parameters()).device
    nll_running, tokens_processed = 0.0, 0
    for i in trange(0, len(data), batch_size, desc="Computing perplexity", leave=False):
        inputs = torch.cat(data[i:min(i + batch_size, len(data))]).to(device)
        logits = model(inputs).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).float()
        a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
        b = tokens_processed / (tokens_processed + shift_labels.numel())
        nll_running = a * loss.item() + b * nll_running
        tokens_processed += shift_labels.numel()
    return torch.tensor(nll_running).exp().item()


def get_wikitext2(tokenizer, sequence_length):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokens = tokenizer("\n\n".join(raw["text"]), return_tensors="pt").input_ids
    return [
        tokens[:, i * sequence_length:(i + 1) * sequence_length]
        for i in range(tokens.numel() // sequence_length)
    ]


def get_c4_eval(tokenizer, sequence_length):
    # Exact MR-GPTQ source shard, reused from the local Hugging Face Hub cache.
    # Read the first 1100 JSONL records directly because this host has the raw
    # cached .json.gz file but not its datasets Arrow conversion.
    import gzip

    c4_validation = Path(
        "/home/pengliang/.cache/huggingface/hub/datasets--allenai--c4/"
        "snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/"
        "en/c4-validation.00000-of-00008.json.gz"
    )
    if not c4_validation.is_file():
        raise FileNotFoundError(f"Cached C4 validation shard not found: {c4_validation}")

    texts = []
    with gzip.open(c4_validation, "rt", encoding="utf-8") as handle:
        for _ in range(1100):
            line = handle.readline()
            if not line:
                break
            texts.append(json.loads(line)["text"])

    if len(texts) != 1100:
        raise RuntimeError(f"Expected 1100 C4 validation records, got {len(texts)}")

    tokens = tokenizer(" ".join(texts), return_tensors="pt").input_ids
    n = min(tokens.numel() // sequence_length, 256)
    return [tokens[:, i * sequence_length:(i + 1) * sequence_length] for i in range(n)]

def build_model(args):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False

    if args.mode == "arcquant":
        model_name = Path(args.model.rstrip("/")).name.lower()
        prefix = args.cache_dir / f"{model_name}_"
        reorder_index = torch.load(
            str(prefix) + f"reorder_index_{args.calib_dataset}_{args.act_sort_metric}.pt",
            weights_only=False,
        )
        select_nums = torch.load(
            str(prefix) + f"select_num_{args.calib_dataset}_{args.act_sort_metric}.pt",
            weights_only=False,
        )
        reorder_model = (
            reorder_model_qwen
            if model.config.model_type in {"qwen2", "qwen3"}
            else reorder_model_llama
        )
        model = reorder_model(
            model, device=torch.device("cuda:0"), kv_cache=False,
            reorder_index=reorder_index, select_nums=select_nums,
            quant_type="NVFP4",
        )

    model = model.to("cuda:0").eval()
    model.requires_grad_(False)
    return model


TASKS = {
    "piqa":             ("piqa", 0, 64, None, {}),
    "winogrande":       ("winogrande", 5, 64, None, {}),
    "boolq":            ("boolq", 0, 8, None, {}),
    "arc_challenge":    ("arc_challenge", 25, 8, None, {}),
    "hellaswag":        ("hellaswag", 10, 8, 4000, {}),
    "mmlu":             ("mmlu", 5, 4, 200, {}),
    "arc_easy":         ("arc_easy", 25, 8, 4000, {}),
    "lambada_openai":   ("lambada_openai", 0, 16, None, {}),
    "gsm8k":            ("gsm8k", 5, 16, None, {}),
    "mbpp":             ("mbpp", 3, 16, None, {}),
}


def evaluate_tasks(model, tokenizer, requested, smoke_limit):
    results = {}
    original_max_pos = model.config.max_position_embeddings
    model.config.max_position_embeddings = min(original_max_pos, 2048)

    for key in requested:
        task, shots, batch_size, default_limit, kwargs = TASKS[key]
        limit = smoke_limit if smoke_limit is not None else default_limit
        use_cache = task in {"gsm8k", "mbpp"}
        model.config.use_cache = use_cache
        if model.generation_config is not None:
            model.generation_config.use_cache = use_cache

        gc.collect()
        torch.cuda.empty_cache()
        lm = HFLM(pretrained=model, tokenizer=tokenizer,
                  batch_size=batch_size, max_length=2048)
        output = simple_evaluate(
            model=lm, tasks=[task], num_fewshot=shots, limit=limit,
            confirm_run_unsafe_code=True, **kwargs,
        )
        results.update(output["results"])
        print(make_table({"results": output["results"], "versions": {},
                          "n-shot": {}, "higher_is_better": {}}))
        del lm

    model.config.max_position_embeddings = original_max_pos
    model.config.use_cache = False
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("fp16", "arcquant"), default="fp16")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--calib-dataset", default="c4")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "saved")
    parser.add_argument("--act-sort-metric", default="max")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--tasks", nargs="*", choices=sorted(TASKS))
    parser.add_argument("--smoke-limit", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = build_model(args)

    wiki = compute_perplexity(model, get_wikitext2(tokenizer, args.sequence_length), 1)
    print(f"Wikitext-2 perplexity: {wiki:.4f}")

    c4 = compute_perplexity(model, get_c4_eval(tokenizer, args.sequence_length), 1)
    print(f"C4 perplexity: {c4:.4f}")

    results = evaluate_tasks(model, tokenizer, args.tasks or [], args.smoke_limit)

    def metric(task_result):
        if not task_result:
            return None
        for marker in ("pass_at_1", "pass@1", "exact_match", "acc_norm", "acc", "f1"):
            for key, value in task_result.items():
                if marker in key and "stderr" not in key:
                    return value
        return None

    # Keep a compact, MR-GPTQ-style summary for parsing and tabular logs.
    compact_results = {
        "piqa": metric(results.get("piqa")),
        "winogrande": metric(results.get("winogrande")),
        "boolq": metric(results.get("boolq")),
        "arc_challenge": metric(results.get("arc_challenge")),
        "hellaswag": metric(results.get("hellaswag")),
        "mmlu": metric(results.get("mmlu")),
        "arc_easy": metric(results.get("arc_easy")),
        "lambada_openai": metric(results.get("lambada_openai")),
        "gsm8k": metric(results.get("gsm8k")),
        "mbpp": metric(results.get("mbpp")),
    }
    print("[RESULT_SUMMARY] " + json.dumps({
        "mode": args.mode, "dtype": args.dtype, "seed": args.seed,
        "wikitext2_ppl": wiki, "c4_ppl": c4, "results": compact_results,
    }))

if __name__ == "__main__":
    main()
