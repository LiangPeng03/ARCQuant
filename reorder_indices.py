from datasets import load_dataset
import torch.nn as nn
import gc
from utilize import * 
import torch
from collections import defaultdict
import functools
from typing import List
import time
import pandas as pd
import numpy as np
import tqdm
import argparse
import math
import os
import time
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="path of the hf model")
parser.add_argument(
    "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "humaneval", "pile"], 
    help="The calibration dataset to use."
)
parser.add_argument("--act_sort_metric", type=str, help="the metric used to sort the activations.")
parser.add_argument("--samples", type=int, default=128)
parser.add_argument("--seqlen", type=int, default=2048)
parser.add_argument("--seed", type=int, default=0,
                    help="Seed used to sample the calibration data.")
parser.add_argument("--calibration-tokens", type=Path,
                    help="Read the exact calibration token tensor from MR-GPTQ.")
parser.add_argument("--cache-dir", type=Path, default=Path("saved"))


args = parser.parse_args()


DATASET_LOADERS = {
    "wikitext2": get_wikitext2,
    "c4": get_c4,
    "pile": get_pile,
    "humaneval": get_humaneval
}
        
def main():
    model, enc = load_model(args.model)
    folder_path = args.cache_dir
    path = args.model.rstrip('/')
    model_name = path.split('/')[-1]
    folder_path.mkdir(parents=True, exist_ok=True)

    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    start_time = time.time()
    
    print(f"Using {args.dataset} dataset for calibration.")
    get_dataset = DATASET_LOADERS[args.dataset]

    dataset_name = args.dataset.lower()
    prefix = folder_path / model_name.lower()
    act_scales_filename = f'{prefix}_act_scales_{dataset_name}_{args.act_sort_metric}.pt'
    tokens = None
    if args.calibration_tokens is not None:
        if args.dataset != "c4" or (args.samples, args.seqlen) != (128, 2048) or args.cache_dir == Path("saved"):
            parser.error("Shared C4 tokens require --dataset c4, 128 x 2048, and a separate --cache-dir.")
        tokens = torch.load(args.calibration_tokens, map_location="cpu", weights_only=True)
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long or tuple(tokens.shape) != (128, 2048):
            raise ValueError("Expected a torch.long calibration tensor shaped (128, 2048)")
        print(f"Shared C4 calibration tokens: {args.calibration_tokens}")

    print("Getting activation stats...")
    if tokens is not None or not os.path.exists(act_scales_filename):
        print("Generating activation stats...")
        if tokens is None:
            dataloader, _ = get_dataset(
                nsamples=args.samples, seed=args.seed, seqlen=args.seqlen, tokenizer=enc
            )
        else:
            dataloader = [(sample,) for sample in tokens.split(1)]

        act_scales = get_act_stats(
            model, dataloader, "cuda:0", metric=args.act_sort_metric, seqlen=args.seqlen
        )
        torch.save(act_scales, act_scales_filename)
        del dataloader
    else:
        print("Loading pre-saved activation stats...")
        act_scales = torch.load(act_scales_filename)
        

    print("Getting reording index...")
    reorder_index = get_reorder_index(model, act_scales, metric=args.act_sort_metric)
    
    print("Getting proportions...")

    if tokens is None:
        _, inps = get_dataset(
            nsamples=32, seed=args.seed, tokenizer=enc, seqlen=args.seqlen
        )
    else:
        inps = list(tokens[:32].split(1))
    select_num, average_bits = search_select_proportions(model, inps, "cuda", args.seqlen, reorder_index)
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    reorder_filename = f'{prefix}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt'
    select_num_filename = f'{prefix}_select_num_{dataset_name}_{args.act_sort_metric}.pt'
    avg_bits_filename = f'{prefix}_average_bits_{dataset_name}_{args.act_sort_metric}.pt'

    print(f"Saving reorder index to {reorder_filename}")
    torch.save(reorder_index, reorder_filename)
    print(f"Saving select num to {select_num_filename}")
    torch.save(select_num, select_num_filename)
    print(f"Saving average bits to {avg_bits_filename}")
    torch.save(average_bits, avg_bits_filename)
    
if __name__ == "__main__":
    main()
