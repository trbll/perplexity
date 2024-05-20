from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import json
from random import randint

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, pad_token_id=tokenizer.eos_token_id).to(torch_device)

def get_top_k_tokens(prompt, k, temperature=1.0):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(torch_device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(torch_device)
    outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
    
    logits = outputs.logits
    last_logits = logits[0, -1, :]

    # Apply temperature scaling
    scaled_logits = last_logits / temperature

    probs = torch.softmax(scaled_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k)
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
    return top_k_tokens, top_k_probs, top_k_indices

for run in range(1): # Change range if you want to batch run generations

    # CONFIGS
    original_prompt = "In this article, we"
    prompt = original_prompt
    k = 100
    N = 512
    seed = 105 #randint(0, 1000) # Change seed; e.g., by run id, randomly, etc.
    temp = 1.0

    set_seed(seed)

    print(f'Run: {run}, Seed: {seed}')

    steps_info = []

    for step_num in range(N):
        top_k_tokens, top_k_probs, top_k_indices = get_top_k_tokens(prompt, k, temp)
        top_k_probs_normalized = top_k_probs / top_k_probs.sum()
        
        selected_index = torch.multinomial(top_k_probs, 1).item()
        selected_token = tokenizer.decode([top_k_indices[selected_index]])

        # print( 'Prompt: ', prompt )
        prompt += selected_token

        # Record step information
        step_info = {
            'step_num': step_num + 1,
            'selected_id': top_k_indices[selected_index].item(),
            'top_ks': []
        }

        for idx, (token, prob_raw, prob_norm) in enumerate(zip(top_k_indices, top_k_probs, top_k_probs_normalized)):
            step_info['top_ks'].append({
                'token_id': token.item(),
                'token_text': tokenizer.decode([token]),
                'prob_raw': prob_raw.item(),
                'prob_norm': prob_norm.item(),
                'selected': idx == selected_index
            })
        steps_info.append(step_info)

    # Compile the final JSON structure
    output_json = {
        'model': checkpoint,
        'seed': seed,
        'initial_prompt': original_prompt,
        'final_text': prompt,
        'k': k,
        'nSteps': N,
        'steps': steps_info
    }

    # Creating a unique filename
    json_filename = f'json/output_statistics_k{k}_N{N}_seed{seed}.json'

    # Save the JSON to a file
    with open(json_filename, 'w') as f:
        json.dump(output_json, f, indent=4)

    # Double save most recent run with hardcoded filename
    # Makes quick evaluation easier (don't have to change filename in command line parameter)
    dummy_json_filename = f'json/most_recent_output_statistics.json'

    # Save the JSON to a file
    with open(dummy_json_filename, 'w') as f:
        json.dump(output_json, f, indent=4)

    # Print final prompt and JSON file info
    # print(f"Final generated text: {prompt}")
    # print(f"Statistics saved to: {json_filename}")