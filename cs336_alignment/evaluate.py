import pandas as pd
from cs336_alignment.config import r1_zero_prompt
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from typing import Callable, List, Literal
from vllm import LLM, SamplingParams


def evaluate_vllm(
    vllm_model: LLM,
    eval_sampling_params: SamplingParams,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    assert len(prompts) == len(ground_truths)

    outputs = vllm_model.generate(prompts, eval_sampling_params)

    serialized_results = []
    reward_dict_sum = None
    
    for output, ground_truth in zip(outputs, ground_truths):
        prompt = output.prompt
        response = output.outputs[0].text
        reward_dict = reward_fn(response, ground_truth)

        if reward_dict_sum is None:
            reward_dict_sum = reward_dict
        else:
            for key in reward_dict_sum:
                reward_dict_sum[key] += reward_dict[key]

        serialized_results.append({'prompt': prompt, 'response': response, 'ground_ground_truth':ground_truth, 'reward_dict': reward_dict})

    print('eval results')
    for key in reward_dict_sum:
        print(f'{key}: {(reward_dict_sum[key] / len(prompts)):3f}')

    # todo: save serialized_results to disk


def gsm8k_baseline(
    prompt_template: Literal['r1_zero', 'question_only']
) -> None:
    llm = LLM(model='Qwen/Qwen2.5-Math-1.5B')
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=['</answer>'], include_stop_str_in_output = True
    )

    df = pd.read_json('data/gsm8k/test.jsonl', lines=True)
    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    if prompt_template == 'r1_zero':
        questions = [r1_zero_prompt.format(question=q) for q in questions]

    answers = [a.split('####')[1].lstrip() for a in answers]

    evaluate_vllm(llm, sampling_params, r1_zero_reward_fn, questions, answers)


if __name__ == '__main__':
    gsm8k_baseline('r1_zero')
    # gsm8k_baseline('question_only')