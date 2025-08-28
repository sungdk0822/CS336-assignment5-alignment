import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-Math-1.5B',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')


# uv run pytest -k test_tokenize_prompt_and_output
def tokenize_prompt_and_output(
    prompt_strs: list[str], 
    output_strs: list[str], 
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    '''
    Tokenize the prompt and output strings, 
    and construct a mask that is 1 for the response tokens 
    and 0 for other tokens (prompt or padding).

    Let prompt_and_output_lens be a list containing the lengths of
    the tokenized prompt and output strings. Then the returned dictionary should have the
    following keys:
        input_ids 
            torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
            the tokenized prompt and output strings, with the final token sliced off.
        labels 
            torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
            shifted input ids, i.e., the input ids without the first token.
        response_mask 
            torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1): 
            a mask on the response tokens in the labels
    '''
    encoded_prompts = tokenizer(prompt_strs)['input_ids']
    encoded_outputs = tokenizer(output_strs)['input_ids']
    assert len(encoded_prompts) == len(encoded_outputs)

    prompt_and_output_lens = []
    for prompt, output in zip(encoded_prompts, encoded_outputs):
        prompt_and_output_lens.append(len(prompt) + len(output))
    max_prompt_and_output_lens = max(prompt_and_output_lens)
    batch_size = len(encoded_prompts)

    pad_token_id = tokenizer.pad_token_id
    input_ids = torch.full((batch_size, max_prompt_and_output_lens - 1), pad_token_id, dtype=torch.int)
    labels = torch.full((batch_size, max_prompt_and_output_lens - 1), pad_token_id, dtype=torch.int)
    response_mask = torch.zeros(batch_size, max_prompt_and_output_lens - 1, dtype=torch.int)

    for i in range(batch_size):
        prompt, output = encoded_prompts[i], encoded_outputs[i]
        len_prompt = len(prompt)
        len_output = len(output)
        if len_prompt + len_output < max_prompt_and_output_lens - 1:
            input_ids[i, : len_prompt + len_output] = torch.tensor((prompt + output), dtype=torch.int)
        else:
            input_ids[i, : len_prompt + len_output - 1] = torch.tensor((prompt + output)[:-1], dtype=torch.int)
            
        labels[i, : len_prompt + len_output - 1] = torch.tensor((prompt + output)[1:], dtype=torch.int)
        response_mask[i, len_prompt - 1 : len_prompt + len_output - 1] = 1

    return {
        'input_ids': input_ids,
        'labels': labels,
        'response_mask': response_mask
    }


if __name__ == '__main__':
    prompt_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    output_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')

    tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)