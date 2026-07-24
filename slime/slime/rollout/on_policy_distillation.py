import math

from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    """Score a completed student sequence with an SGLang teacher."""
    import aiohttp

    from slime.utils.processing_utils import encode_image_for_rollout_engine

    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def zero_reward_func(args, sample, **kwargs):
    """Return no task reward, for pure OPD with a Megatron teacher."""
    return 0.0


def _extract_teacher_response_log_probs(sample: Sample, reward: dict, sample_index: int) -> list[float]:
    """Validate one SGLang response and return log-probs for response tokens only."""
    context = f"teacher response for sample {sample_index}"

    if not isinstance(reward, dict):
        raise ValueError(f"{context} must be a JSON object, got {type(reward).__name__}.")
    meta_info = reward.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError(f"{context} is missing object field 'meta_info'.")
    token_log_probs = meta_info.get("input_token_logprobs")
    if not isinstance(token_log_probs, (list, tuple)):
        raise ValueError(f"{context} is missing list field 'meta_info.input_token_logprobs'.")

    response_length = sample.response_length
    if not isinstance(response_length, int) or isinstance(response_length, bool) or response_length < 0:
        raise ValueError(f"sample {sample_index} has invalid response_length={response_length!r}.")
    if response_length > len(sample.tokens):
        raise ValueError(
            f"sample {sample_index} has response_length={response_length}, "
            f"but only {len(sample.tokens)} total tokens."
        )

    # SGLang returns one entry per input token. The first entry has no
    # conditional log-probability; response tokens are a suffix of the input.
    if len(token_log_probs) != len(sample.tokens):
        raise ValueError(
            f"{context} contains {len(token_log_probs)} token entries, "
            f"but the submitted sequence contains {len(sample.tokens)} tokens."
        )
    if response_length == 0:
        return []

    response_entries = token_log_probs[-response_length:]
    response_tokens = sample.tokens[-response_length:]
    values = []
    for position, (entry, expected_token) in enumerate(zip(response_entries, response_tokens, strict=True)):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(
                f"{context} has malformed input_token_logprobs entry at response position {position}; "
                "expected [log_prob, token_id, ...]."
            )
        log_prob, returned_token = entry[:2]
        if isinstance(log_prob, bool) or not isinstance(log_prob, (int, float)):
            raise ValueError(
                f"{context} has a non-numeric log-probability at response position {position}: {log_prob!r}."
            )
        if not math.isfinite(log_prob):
            raise ValueError(
                f"{context} has a non-finite log-probability at response position {position}: {log_prob!r}."
            )
        if isinstance(returned_token, bool) or not isinstance(returned_token, int):
            raise ValueError(f"{context} has an invalid token id at response position {position}: {returned_token!r}.")
        if returned_token != expected_token:
            raise ValueError(
                f"{context} token mismatch at response position {position}: "
                f"expected {expected_token}, got {returned_token}."
            )
        values.append(float(log_prob))

    if len(values) != response_length:
        raise ValueError(f"{context} produced {len(values)} log-probabilities, expected {response_length}.")
    return values


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Attach validated SGLang teacher log-probs and return zero task rewards."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    teacher_log_probs = [
        _extract_teacher_response_log_probs(sample, reward, sample_index)
        for sample_index, (sample, reward) in enumerate(zip(samples, raw_rewards, strict=True))
    ]
    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = t_log_probs

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
