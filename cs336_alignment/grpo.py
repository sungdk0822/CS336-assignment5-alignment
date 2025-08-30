import torch


# uv run pytest -k test_compute_group_normalized_rewards
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    '''
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] 
            Scores the rollout responses against
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".
        rollout_responses: list[str] 
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] 
            The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
        group_size: int 
            Number of responses per question (group).
        advantage_eps: float 
            Small constant to avoid division by zero in normalization.
        normalize_by_std: bool 
            If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]
            advantages 
                shape (rollout_batch_size,). Group-normalized rewards for each rollout
                response.
            raw_rewards 
                shape (rollout_batch_size,). Unnormalized rewards for each rollout
                response.
            metadata 
                your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    '''
    rewards = []
    format_rewards = []
    answer_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        reward, format_reward, answer_reward = reward_dict['reward'], reward_dict['format_reward'], reward_dict['answer_reward']
        rewards.append(reward)
        format_rewards.append(format_reward)
        answer_rewards.append(answer_reward)
    
    assert len(rewards) % group_size == 0
    advantages = torch.tensor(rewards)
    rewards = torch.tensor(rewards)
    format_rewards = torch.tensor(format_rewards)
    answer_rewards = torch.tensor(answer_rewards)
    
    for i in range(advantages.size(0) // group_size):
        advantages[i * group_size : (i + 1) * group_size] -= rewards[i * group_size : (i + 1) * group_size].mean()
        if normalize_by_std:
            advantages[i * group_size : (i + 1) * group_size] /= rewards[i * group_size : (i + 1) * group_size].std() + advantage_eps
    
    metadata = {
        'reward_max': rewards.max(),
        'reward_min': rewards.min(),
        'reward_mean': rewards.mean(),
        'reward_std': rewards.std(),
        'format_reward_max': format_rewards.max(),
        'format_reward_min': format_rewards.min(),
        'format_reward_mean': format_rewards.mean(),
        'format_reward_std': format_rewards.std(),
        'answer_reward_max': answer_rewards.max(),
        'answer_reward_min': answer_rewards.min(),
        'answer_reward_mean': answer_rewards.mean(),
        'answer_reward_std': answer_rewards.std()
    }

    return (advantages, rewards, metadata)


# uv run pytest -k test_compute_naive_policy_gradient_loss
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    '''
    Args:
        raw_rewards_or_advantages: torch.Tensor 
            Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor 
            Shape (batch_size, sequence_length), logprobs for each token.
    Returns:
        torch.Tensor 
            Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
            be aggregated across the batch and sequence dimensions in the training loop).
    '''
    return -raw_rewards_or_advantages * policy_log_probs