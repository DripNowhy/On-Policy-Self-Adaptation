"""Reward hooks for teacher-free OPSA training."""


async def reward_func(args, sample, **kwargs):
    """Return zero task reward; OPSA supplies the token-level learning signal."""

    if isinstance(sample, list):
        return [0.0] * len(sample)
    return 0.0


def post_process_rewards(args, samples, **kwargs):
    """Return zero scalar rewards for the standard rollout post-processing API."""

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
