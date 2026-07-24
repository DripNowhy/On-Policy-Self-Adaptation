import asyncio
from types import SimpleNamespace

import pytest

from slime.rollout.on_policy_distillation import post_process_rewards, reward_func, zero_reward_func
from slime.utils.types import Sample


def teacher_reward(tokens: list[int], log_probs: list[float | None]):
    assert len(tokens) == len(log_probs)
    return {
        "meta_info": {
            "input_token_logprobs": [
                [log_prob, token_id, f"token-{token_id}"] for token_id, log_prob in zip(tokens, log_probs, strict=True)
            ]
        }
    }


def test_post_process_extracts_response_suffix_and_returns_zero_task_reward():
    sample = Sample(
        tokens=[10, 11, 12, 13],
        response_length=2,
        reward=teacher_reward([10, 11, 12, 13], [None, -0.1, -0.4, -0.7]),
    )

    rewards, raw_rewards = post_process_rewards(SimpleNamespace(reward_key=None), [sample])

    assert rewards == [0.0]
    assert raw_rewards == [0.0]
    assert sample.teacher_log_probs == pytest.approx([-0.4, -0.7])
    assert isinstance(sample.teacher_log_probs, list)


def test_zero_length_response_stays_empty_instead_of_selecting_full_sequence():
    sample = Sample(
        tokens=[10, 11, 12],
        response_length=0,
        reward=teacher_reward([10, 11, 12], [None, -0.1, -0.2]),
    )

    post_process_rewards(SimpleNamespace(reward_key=None), [sample])

    assert sample.teacher_log_probs == []


@pytest.mark.parametrize(
    ("reward", "message"),
    [
        (None, "must be a JSON object"),
        ({}, "missing object field 'meta_info'"),
        ({"meta_info": {}}, "missing list field 'meta_info.input_token_logprobs'"),
        ({"meta_info": {"input_token_logprobs": [[None, 10]]}}, "contains 1 token entries"),
        (
            {"meta_info": {"input_token_logprobs": [[None, 10], [-0.1]]}},
            "malformed input_token_logprobs entry",
        ),
        (
            {"meta_info": {"input_token_logprobs": [[None, 10], ["bad", 11]]}},
            "non-numeric log-probability",
        ),
        (
            {"meta_info": {"input_token_logprobs": [[None, 10], [float("nan"), 11]]}},
            "non-finite log-probability",
        ),
        (
            {"meta_info": {"input_token_logprobs": [[None, 10], [-0.1, 99]]}},
            "token mismatch",
        ),
    ],
)
def test_malformed_teacher_response_is_rejected(reward, message):
    sample = Sample(tokens=[10, 11], response_length=1, reward=reward)

    with pytest.raises(ValueError, match=message):
        post_process_rewards(SimpleNamespace(reward_key=None), [sample])

    assert sample.teacher_log_probs is None


def test_invalid_response_length_is_rejected():
    sample = Sample(
        tokens=[10],
        response_length=2,
        reward=teacher_reward([10], [None]),
    )

    with pytest.raises(ValueError, match="only 1 total tokens"):
        post_process_rewards(SimpleNamespace(reward_key=None), [sample])


def test_batch_is_not_partially_mutated_when_a_later_teacher_response_is_invalid():
    valid = Sample(
        tokens=[10, 11],
        response_length=1,
        reward=teacher_reward([10, 11], [None, -0.1]),
    )
    invalid = Sample(
        tokens=[20, 21],
        response_length=1,
        reward=teacher_reward([20, 99], [None, -0.2]),
    )

    with pytest.raises(ValueError, match="sample 1 token mismatch"):
        post_process_rewards(SimpleNamespace(reward_key=None), [valid, invalid])

    assert valid.teacher_log_probs is None
    assert invalid.teacher_log_probs is None


def test_megatron_pure_opd_reward_is_zero():
    reward = asyncio.run(zero_reward_func(SimpleNamespace(), Sample()))
    assert reward == 0.0


def test_sglang_request_uses_only_sampled_token_log_probs(monkeypatch):
    posted = {}

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return {"meta_info": {"input_token_logprobs": []}}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, json):
            posted["url"] = url
            posted["payload"] = json
            return FakeResponse()

    monkeypatch.setattr("aiohttp.ClientSession", FakeSession)
    args = SimpleNamespace(rm_url="http://teacher.example/generate")
    sample = Sample(tokens=[10, 11])

    result = asyncio.run(reward_func(args, sample))

    assert result == {"meta_info": {"input_token_logprobs": []}}
    assert posted["url"] == args.rm_url
    assert posted["payload"]["input_ids"] == [10, 11]
    assert set(posted["payload"]) == {"input_ids", "sampling_params", "return_logprob", "logprob_start_len"}
