import pandas as pd
import torch
from cs336_alignment.config import r1_zero_prompt
from typing import Literal
from transformers import PreTrainedModel
from unittest.mock import patch
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed


def get_gsm8k(
    prompt_template: Literal['r1_zero', 'question_only'],
    split: Literal['train', 'test']
) -> tuple[list[str], list[str]]:
    df = pd.read_json(f'data/gsm8k/{split}.jsonl', lines=True)
    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    if prompt_template == 'r1_zero':
        questions = [r1_zero_prompt.format(question=q) for q in questions]

    answers = [a.split('####')[1].lstrip() for a in answers]

    return (questions, answers)


# from assignment5 pdf p.13
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


# from assignment5 pdf p.14
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())