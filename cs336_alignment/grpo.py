import torch
from typing import Literal


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


# uv run pytest -k test_compute_grpo_clip_loss
def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Args:
        advantages: torch.Tensor 
            Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor 
            Shape (batch_size, sequence_length), per-token log
            probs from the policy being trained.
        old_log_probs: torch.Tensor 
            Shape (batch_size, sequence_length), per-token log probs
            from the old policy.
        cliprange: float 
            Clip parameter ε (e.g. 0.2).
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            loss 
                torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
            metadata 
                dict containing whatever you want to log. We suggest logging whether each
                token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
                the min was lower than the LHS.
    '''
    importance_ratio = (policy_log_probs - old_log_probs).exp()
    clipped_importance_ratio = torch.clamp(importance_ratio, 1 - cliprange, 1 + cliprange)

    is_clipped = importance_ratio * advantages > clipped_importance_ratio * advantages
    metadata = {'is_clipped': is_clipped}

    return (-torch.minimum(importance_ratio * advantages, clipped_importance_ratio * advantages), metadata)


# uv run pytest -k test_compute_policy_gradient_loss
def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal['no_baseline', 'reinforce_with_baseline', 'grpo_clip'],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Args:
        policy_log_probs 
            (batch_size, sequence_length), per-token log-probabilities from the
            policy being trained.
        loss_type 
            One of 'no_baseline', 'reinforce_with_baseline', or 'grpo_clip'.
        raw_rewards 
            Required if loss_type == 'no_baseline'; shape (batch_size, 1).
        advantages 
            Required for 'reinforce_with_baseline' and 'grpo_clip'; shape (batch_size, 1).
        old_log_probs 
            Required for 'grpo_clip'; shape (batch_size, sequence_length).
        cliprange 
            Required for 'grpo_clip'; scalar ε used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            loss 
                (batch_size, sequence_length), per-token loss.
            metadata 
                dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    '''
    if loss_type == 'no_baseline':
        assert raw_rewards is not None
        metadata = {}
        return (compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), metadata)
    if loss_type == 'reinforce_with_baseline':
        assert advantages is not None
        metadata = {}
        return (compute_naive_policy_gradient_loss(advantages, policy_log_probs), metadata)
    if loss_type == 'grpo_clip':
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)


# uv run pytest -k test_masked_mean
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    '''
    Args:
        tensor: torch.Tensor 
            The data to be averaged.
        mask: torch.Tensor 
            Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None 
            Dimension over which to average. If None, compute the mean over all
            masked elements.
    Returns:
        torch.Tensor 
            The masked mean; shape matches tensor.mean(dim) semantics.
    '''
    num_included_elements_along_dim = mask.sum(dim)
    num_all_elements_along_dim = mask.size(dim) if dim is not None else mask.numel()
    mean_including_zeros = (tensor * mask).mean(dim)
    mean_excluding_zeros = mean_including_zeros * num_all_elements_along_dim / num_included_elements_along_dim

    return mean_excluding_zeros