import os
import random
import torch
import wandb
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.util import init_vllm, load_policy_into_vllm_instance, get_gsm8k
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal
from vllm import SamplingParams


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
    metadata = {'clip_ratio': (is_clipped.sum() / is_clipped.numel()).item()}

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


# uv run pytest -k test_grpo_microbatch_train_step
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal['no_baseline', 'reinforce_with_baseline', 'grpo_clip'],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Args:
        policy_log_probs 
            (batch_size, sequence_length), per-token log-probabilities from the policy being trained.
        response_mask 
            (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps 
            Number of microbatches per optimizer step.
        loss_type 
            One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards 
            Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages 
            Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs 
            Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange 
            Clip parameter ε for GRPO-Clip.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            loss 
                scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
                this so we can log it.
            metadata 
                Dict with metadata from the underlying loss call, and any other statistics you
                might want to log.
    '''
    '''
    from assignment5 pdf p.26:
        given the raw rewards or advantages and log probs, 
        we will compute the per-token loss, 
        use masked_mean to aggregate to a scalar loss per example, 
        average over the batch dimension, 
        adjust for gradient accumulation, 
        and backpropagate.
    '''
    # compute the per-token loss
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )

    # use masked_mean to aggregate to a scalar loss per example
    loss = masked_mean(per_token_loss, response_mask, dim=-1)

    # average over the batch dimension
    loss = loss.mean(dim=0)

    # adjust for gradient accumulation
    loss /= gradient_accumulation_steps

    # backpropagate
    loss.backward()

    return (loss, metadata)


if __name__ == '__main__':
    model_id = 'Qwen/Qwen2.5-Math-1.5B'
    # model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = f'outputs/{model_id.split('/')[-1]}/{time}'
    os.makedirs(output_dir, exist_ok=True)

    n_grpo_steps: int = 200
    eval_steps: int = 10
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 512
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 256 # On-policy
    micro_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.1
    loss_type: Literal[
        'no_baseline',
        'reinforce_with_baseline',
        'grpo_clip',
    ] = 'reinforce_with_baseline'
    use_std_normalization: bool = True
    cliprange: float = 0.1
    use_wandb: bool = True
    reward_fn = r1_zero_reward_fn

    assert train_batch_size % gradient_accumulation_steps == 0, (
        'train_batch_size must be divisible by gradient_accumulation_steps'
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        'rollout_batch_size must be divisible by group_size'
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        'train_batch_size must be greater than or equal to group_size'
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    policy_model_device = 'cuda:0'
    policy.to(policy_model_device)
    print(f'policy_model_device = {policy.device}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    vllm = init_vllm(model_id, 'cuda', 2025, gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        logprobs=0,
        stop=['</answer>'],
        include_stop_str_in_output = True
    )

    train_questions, train_answers = get_gsm8k('r1_zero', 'train')
    validation_questions, validation_answers = get_gsm8k('r1_zero', 'test')
    assert len(train_questions) == len(train_answers)
    assert len(validation_questions) == len(validation_answers)
    num_train_samples = len(train_questions)
    num_validation_samples = len(validation_questions)
    
    print(num_train_samples)
    print(num_validation_samples)

    if use_wandb:
        wandb.login()
        run = wandb.init(
            project='cs336-assignment5',
            name=f'{time}',
            config={
                'model_id': model_id,
                'n_grpo_steps': n_grpo_steps,
                'learning_rate': learning_rate,
                'advantage_eps': advantage_eps,
                'rollout_batch_size': rollout_batch_size,
                'group_size': group_size,
                'sampling_temperature': sampling_temperature,
                'sampling_min_tokens': sampling_min_tokens,
                'sampling_max_tokens': sampling_max_tokens,
                'epochs_per_rollout_batch': epochs_per_rollout_batch,
                'train_batch_size': train_batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'gpu_memory_utilization': gpu_memory_utilization,
                'loss_type': loss_type,
                'use_std_normalization': use_std_normalization,
                'cliprange': cliprange
            },
        )
        # setup wandb metrics
        wandb.define_metric("train_step") # the x‑axis for training
        wandb.define_metric("eval_step") # the x‑axis for evaluation
        # everything that starts with train/ is tied to train_step
        wandb.define_metric("train/*", step_metric="train_step")
        # everything that starts with eval/ is tied to eval_step
        wandb.define_metric("eval/*", step_metric="eval_step")

    for step in tqdm(range(n_grpo_steps)):
        accumulated_loss = 0.0
        metadata = {}

        def merge_metadata(metadata: dict, temp_metadata: dict, step: int):
            for key, value in temp_metadata.items():
                if key not in metadata.keys():
                    metadata[key] = value
                else:
                    if 'max' in key:
                        metadata[key] = max(metadata[key], value)
                    elif 'min' in key:
                        metadata[key] = min(metadata[key], value)
                    elif 'mean' in key:
                        metadata[key] = (metadata[key] * step + value) / (step + 1)
                    elif 'std' in key:
                        metadata[key] = 0.0 # todo: compute std
                    elif 'ratio' in key:
                        metadata[key] = (metadata[key] * step + value) / (step + 1)

            return metadata

        for gradient_accumulation_step in range(gradient_accumulation_steps):
            # sample a batch of questions D_b from D (microbatch version)
            indices = random.sample(range(num_train_samples), micro_train_batch_size)
            repeated_indices = [i for i in indices for _ in range(group_size)]
            repeated_micro_batch_questions = [train_questions[i] for i in repeated_indices]
            repeated_micro_batch_answers = [train_answers[i] for i in repeated_indices]

            # set the old policy model π_θ_old ← π_θ
            load_policy_into_vllm_instance(policy, vllm)

            # sample G outputs for each question q ∈ D_b
            outputs = vllm.generate(repeated_micro_batch_questions, sampling_params, use_tqdm=False)
            prompts = [output.prompt for output in outputs]
            responses = [output.outputs[0].text for output in outputs]

            # compute rewards for each sampled output by running reward function
            # compute advantages with(loss_type == 'grpo_clip') or without(loss_type == 'reinforce_with_baseline') group normalization
            normalize_by_std = True if loss_type == 'grpo_clip' else False
            advantages, rewards, temp_metadata = compute_group_normalized_rewards(
                reward_fn,
                responses,
                repeated_micro_batch_answers,
                group_size,
                advantage_eps,
                normalize_by_std
            )
            metadata = merge_metadata(metadata, temp_metadata, gradient_accumulation_step)

            if loss_type == 'no_baseline':
                advantages = None
            else:
                rewards = None
            
            # compute policy_log_probs
            result = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids, labels, response_mask = result['input_ids'], result['labels'], result['response_mask']

            input_ids = input_ids.to(policy_model_device)
            labels = labels.to(policy_model_device)
            response_mask = response_mask.to(policy_model_device)

            policy_log_probs = get_response_log_probs(policy, input_ids, labels)['log_probs']
            advantages = advantages.to(policy_model_device)

            # compute old_log_probs if loss_type == 'grpo_clip'
            old_log_probs = None
            if loss_type == 'grpo_clip':
                with torch.inference_mode():
                    old_log_probs = get_response_log_probs(policy, input_ids, labels)['log_probs']

            advantages.unsqueeze_(-1)
            loss, temp_metadata = grpo_microbatch_train_step(
                policy_log_probs,
                response_mask,
                gradient_accumulation_steps,
                loss_type,
                rewards,
                advantages,
                old_log_probs,
                cliprange
            )

            accumulated_loss += loss
            metadata = merge_metadata(metadata, temp_metadata, gradient_accumulation_step)

        train_log = {}
        for key, value in metadata.items():
            train_log['train/' + key] = value

        if use_wandb: # todo: add gradient norm, token entropy
            wandb.log(
                train_log, 
                step=step
            )
        else:
            print(f'step {step}, train_log = {train_log}')

        # todo: use gradient clipping with clip value 1.0

        # epochs_per_rollout_batch is n_train_steps_per_rollout_batch in p.21 Algorithm 3
        for train_step in range(epochs_per_rollout_batch):
            # update the policy model π_θ by maximizing the GRPO-Clip objective

            if train_step != 0:
                optimizer.zero_grad()
                # todo: modify to compute over all input_ids and labels, not just those from the last accumulation step
                policy_log_probs = get_response_log_probs(policy, input_ids, labels)['log_probs']
                loss, train_step_metadata = compute_policy_gradient_loss(
                    policy_log_probs,
                    loss_type,
                    rewards,
                    advantages,
                    old_log_probs,
                    cliprange
                )
                loss.backward()

            optimizer.step()

        if step % eval_steps == 0:
            eval_metadata = {}
            load_policy_into_vllm_instance(policy, vllm)
            for eval_accumulation_step in tqdm(range(num_validation_samples // micro_eval_batch_size), desc='eval accumulation'):
                
                # todo: modify to perform evaluation on the entire validation set instead of random sampling
                
                # sample a batch of questions D_b from D (microbatch version)
                indices = random.sample(range(num_validation_samples), micro_eval_batch_size)
                repeated_indices = [i for i in indices for _ in range(group_size)]
                repeated_micro_batch_questions = [validation_questions[i] for i in repeated_indices]
                repeated_micro_batch_answers = [validation_answers[i] for i in repeated_indices]

                # sample G outputs for each question q ∈ D_b
                outputs = vllm.generate(repeated_micro_batch_questions, sampling_params, use_tqdm=False)
                prompts = [output.prompt for output in outputs]
                responses = [output.outputs[0].text for output in outputs]

                # compute rewards for each sampled output by running reward function
                # compute advantages with(loss_type == 'grpo_clip') or without(loss_type == 'reinforce_with_baseline') group normalization
                normalize_by_std = True if loss_type == 'grpo_clip' else False
                advantages, rewards, temp_metadata = compute_group_normalized_rewards(
                    reward_fn,
                    responses,
                    repeated_micro_batch_answers,
                    group_size,
                    advantage_eps,
                    normalize_by_std
                )

                eval_metadata = merge_metadata(eval_metadata, temp_metadata, eval_accumulation_step)
            
            eval_log = {}
            for key, value in eval_metadata.items():
                eval_log['eval/' + key] = value

            if use_wandb: # todo: add gradient norm, token entropy
                wandb.log(
                    eval_log, 
                    step=step
                )
            else:
                print(f'step {step}, eval_log = {eval_log}')

        # todo: periodically save model

    # save the model weights
    policy.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)