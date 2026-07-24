import argparse
import asyncio
import json
import logging
import time
from types import SimpleNamespace

import aiohttp
from tqdm import tqdm

from slime.rollout.rm_hub import async_rm
from slime.utils.data import Dataset
from slime.utils.metric_utils import compute_pass_rate, compute_statistics, dict_add_prefix
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample


LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AIME with SGLang using Slime reward/metric code.")
    parser.add_argument(
        "--server-url", required=True, help="SGLang router/server URL, for example http://127.0.0.1:30000"
    )
    parser.add_argument("--model-path", required=True, help="HF model/tokenizer path")
    parser.add_argument("--dataset", required=True, help="AIME jsonl path")
    parser.add_argument("--output", required=True, help="Output metrics json path")
    parser.add_argument("--samples-output", required=True, help="Output samples jsonl path")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--dataset-name", default="aime")
    parser.add_argument("--n", type=int, default=32, help="Samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--request-timeout", type=int, default=14400)
    parser.add_argument("--disable-thinking", action="store_true", default=False)
    parser.add_argument("--seed-base", type=int, default=None, help="Optional deterministic sampling seed base")
    parser.add_argument("--skip-special-tokens", action="store_true", default=False)
    return parser.parse_args()


def make_metric_dict(dataset_name: str, rewards: list[float], samples: list[Sample], group_size: int):
    response_lengths = [sample.effective_response_length for sample in samples]
    truncated = [sample.status == Sample.Status.TRUNCATED for sample in samples]

    metrics = {
        f"eval/{dataset_name}": float(sum(rewards) / len(rewards)),
        f"eval/{dataset_name}-truncated_ratio": float(sum(truncated) / len(truncated)),
    }
    metrics |= dict_add_prefix(compute_statistics(response_lengths), f"eval/{dataset_name}/response_len/")
    metrics[f"eval/{dataset_name}/truncated_ratio"] = metrics[f"eval/{dataset_name}-truncated_ratio"]
    metrics |= dict_add_prefix(
        compute_pass_rate(flat_rewards=rewards, group_size=group_size),
        f"eval/{dataset_name}-",
    )
    return metrics


def update_sample_from_meta(sample: Sample, meta_info: dict):
    token_logprobs = meta_info.get("output_token_logprobs") or []
    if token_logprobs:
        sample.response_length = len(token_logprobs)
        sample.rollout_log_probs = [item[0] for item in token_logprobs]
    else:
        sample.response_length = int(meta_info.get("completion_tokens") or meta_info.get("completion_token_num") or 0)

    finish_reason = meta_info.get("finish_reason") or {}
    finish_type = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
    if finish_type == "length":
        sample.status = Sample.Status.TRUNCATED
    elif finish_type == "abort":
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED


async def post_generate(session, url, payload, max_retries=8):
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except Exception:
            if attempt + 1 == max_retries:
                raise
            await asyncio.sleep(min(2**attempt, 30))


async def evaluate(args):
    tokenizer = load_tokenizer(args.model_path, trust_remote_code=True)
    processor = load_processor(args.model_path, trust_remote_code=True)
    dataset = Dataset(
        path=args.dataset,
        tokenizer=tokenizer,
        processor=processor,
        max_length=None,
        prompt_key=args.prompt_key,
        label_key=args.label_key,
        apply_chat_template=True,
        disable_thinking=args.disable_thinking,
    )

    rm_args = SimpleNamespace(eval_rm_type="math", rm_type=None, custom_rm_path=None)
    sampling_params_base = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "stop": None,
        "stop_token_ids": None,
        "skip_special_tokens": args.skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    semaphore = asyncio.Semaphore(args.concurrency)
    samples: list[Sample] = []
    start_time = time.time()
    generate_url = args.server_url.rstrip("/") + "/generate"

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def run_one(prompt_sample: Sample, prompt_index: int, sample_in_group: int, flat_index: int):
            sample = Sample(
                group_index=prompt_index,
                index=flat_index,
                prompt=prompt_sample.prompt,
                label=prompt_sample.label,
                metadata=dict(prompt_sample.metadata or {}),
            )
            prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
            sample.tokens = prompt_ids

            sampling_params = dict(sampling_params_base)
            if args.seed_base is not None:
                sampling_params["sampling_seed"] = args.seed_base + sample_in_group

            payload = {
                "input_ids": prompt_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            async with semaphore:
                output = await post_generate(session, generate_url, payload)

            sample.response = output["text"]
            update_sample_from_meta(sample, output.get("meta_info", {}))
            sample.reward = await async_rm(rm_args, sample, evaluation=True)
            return sample

        tasks = []
        flat_index = 0
        for prompt_index, prompt_sample in enumerate(dataset.samples):
            for sample_in_group in range(args.n):
                tasks.append(asyncio.create_task(run_one(prompt_sample, prompt_index, sample_in_group, flat_index)))
                flat_index += 1

        with tqdm(total=len(tasks), desc=f"Eval {args.dataset_name}") as pbar:
            for task in asyncio.as_completed(tasks):
                samples.append(await task)
                pbar.update(1)

    samples.sort(key=lambda item: item.index)
    rewards = [float(sample.reward) for sample in samples]
    metrics = make_metric_dict(args.dataset_name, rewards, samples, args.n)
    metrics["num_prompts"] = len(dataset.samples)
    metrics["num_samples"] = len(samples)
    metrics["elapsed_sec"] = time.time() - start_time
    metrics["samples_per_sec"] = len(samples) / metrics["elapsed_sec"]

    with open(args.samples_output, "w", encoding="utf-8") as f:
        for sample in samples:
            row = sample.to_dict()
            row["status"] = sample.status.value
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    LOGGER.info("metrics: %s", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
