import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import math_reward


def normalize_messages(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return [dict(m) for m in x]


def get_ground_truth(x):
    if hasattr(x, "as_py"):
        x = x.as_py()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return x
    if isinstance(x, dict):
        return x["ground_truth"]
    raise TypeError(f"Unsupported reward_model type: {type(x)}")


def compute_reward(response, ground_truth):
    try:
        s = math_reward.compute_score(response, ground_truth)
    except TypeError:
        s = math_reward.compute_score(
            solution_str=response,
            ground_truth=ground_truth,
        )

    if isinstance(s, dict):
        for key in ("score", "reward", "acc"):
            if key in s:
                s = s[key]
                break

    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=False,
    )

    df = pd.read_parquet(args.data)

    examples = []

    for idx, row in df.iterrows():
        messages = normalize_messages(row["prompt"])

        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        # Match training/validation filter:
        # data.max_prompt_length = 1024
        if len(prompt_ids) > 1024:
            continue

        gt = get_ground_truth(row["reward_model"])

        examples.append({
            "index": int(idx),
            "prompt_token_ids": prompt_ids,
            "ground_truth": gt,
        })

    print(f"Original test size: {len(df)}")
    print(f"Filtered eval size: {len(examples)}")
    print("Expected filtered size: 4998")

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=5120,
        enforce_eager=True,
        enable_prefix_caching=True,
        seed=args.seed,
    )

    # do_sample=True is implicit because temperature > 0.
    sampling = SamplingParams(
        n=1,
        temperature=0.6,
        top_p=0.95,
        top_k=-1,
        max_tokens=4096,
    )

    result_file = outdir / f"step_{args.step}_sampled_results.jsonl"

    total = 0
    reward_sum = 0.0
    correct = 0

    with result_file.open("w", encoding="utf-8") as fout:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]

            prompts = [
                {"prompt_token_ids": x["prompt_token_ids"]}
                for x in batch
            ]

            outputs = llm.generate(
                prompts,
                sampling_params=sampling,
                use_tqdm=False,
            )

            for ex, output in zip(batch, outputs):
                response = output.outputs[0].text
                token_ids = output.outputs[0].token_ids

                reward = compute_reward(
                    response,
                    ex["ground_truth"],
                )

                total += 1
                reward_sum += reward
                correct += int(reward >= 0.999999)

                fout.write(json.dumps({
                    "index": ex["index"],
                    "step": args.step,
                    "reward": reward,
                    "response_length": len(token_ids),
                    "response": response,
                }, ensure_ascii=False) + "\n")

            if total % 256 < args.batch_size:
                print(
                    f"step={args.step} "
                    f"{total}/{len(examples)} "
                    f"acc={correct/total:.6f} "
                    f"mean_reward={reward_sum/total:.6f}"
                )

    summary = {
        "step": args.step,
        "num_examples": total,
        "correct": correct,
        "accuracy": correct / total,
        "mean_reward": reward_sum / total,
        "seed": args.seed,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": -1,
        "n": 1,
        "max_new_tokens": 4096,
    }

    with (outdir / f"step_{args.step}_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\nFINAL")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
