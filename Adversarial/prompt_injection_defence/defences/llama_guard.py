from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "/home/ubuntu/LLM/Llama-Guard-2-8B"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    user_num_tokens = len(tokenizer.tokenize(chat[0]['content']))
    assistant_num_tokens = len(tokenizer.tokenize(chat[1]['content']))

    output = model.generate(
        input_ids=input_ids, 
        max_new_tokens=100, 
        pad_token_id=tokenizer.eos_token_id, 
        return_dict_in_generate=True, 
        output_scores=True
    )

    prompt_len = input_ids.shape[-1]
    generated_token_ids = output.sequences[0][prompt_len:]
    # print(generated_token_ids)
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids, skip_special_tokens=True)

    probs = torch.cat(output.scores).softmax(dim=-1)
    generated_token_probs, _ = torch.max(probs, dim=-1)

    # for token_id, token, prob in zip(generated_token_ids, generated_tokens, generated_token_probs):
    #     print(f'{token_id.item():<7} {token:<7} {prob.item():.4f}')

    response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    # print(f"Generated response: {response!r}")
    return response, user_num_tokens, assistant_num_tokens

# moderate([
#     {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
#     {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
# ])
# `safe`

def llama_guard_detect(query):
    input = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": ""}
    ]

    output, _, _ = moderate(input)
    
    if 'unsafe' in output:
        return 1
    return 0