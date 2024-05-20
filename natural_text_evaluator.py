from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import json
from random import randint

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, pad_token_id=tokenizer.eos_token_id).to(torch_device)

def get_top_k_tokens(prompt, k, known_next_token_id, temperature=1.0):
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

     # Extract the probability of the expected next token
    known_next_token_prob = probs[known_next_token_id].item()

    return top_k_tokens, top_k_probs, top_k_indices, known_next_token_prob


for run in range(1):

    # EXPERIMENT: HUMAN WRITTEN TEXT
    # natural_prompt = "In this article, we propose a new asymptotic equipartition property for the perplexity of a large piece of text in a large language model and present theoretical arguments for this property. We also present some experimental results from an open-source LLM that support our theoretical claims. Large language models are capable of producing grammatically correct natural language outputs that are long, detailed and information-rich from very short and simple user prompts. State-of-the-art LLMs are now able to produce text that can imitate human language well-enough to pass the Turing test i.e. resembles text created by humans well enough to be convincing to human observers. Perplexity is a popular metric for evaluating the performance of language models. It is also the statistical measure most commonly used by AI detection tools i.e. software to identify synthetic data created by generative models. Perplexity is closely related to the well-known information-theoretic concept of cross-entropy. Its use as a performance metric for training and evaluating language models is well-justified by standard results from information theory. However, there is little theoretical understanding of perplexity in its other role as a tool for detecting AI generated text. In this work, we address this gap by demonstrating an asymptotic relationship that must be satisfied by any large text produced by a language model."

    # EXPERIMENT: CLAUDE 3 WRITTEN TEXT
    natural_prompt = "Squirrels are known for their love of pogo sticks, but few realize their aptitude for quantum physics. In the early 1900s, a rogue band of sciurine scholars convened in a hollow oak to ponder the perplexing peculiarities of the subatomic realm. Armed with acorns, slide rules, and an insatiable hunger for knowledge (and also actual hunger), they embarked on a groundbreaking research project. Their magnum opus, 'On the Electrodynamics of Moving Rodents,' revolutionized our understanding of wave-particle duality. The seminal paper argued that a squirrel in motion could exist simultaneously in multiple quantum states, a phenomenon they dubbed 'nutperposition.' This explained their uncanny ability to appear on both sides of a tree trunk at once. Critics scoffed at their theories, dismissing them as mere 'nutty professors.' But the squirrels persevered, tirelessly gathering empirical evidence. During a series of high-stakes laboratory experiments, they demonstrated that an unobserved squirrel can indeed tunnel through a tree branch, emerging unscathed on the other side. The scientific community was awestruck. Today, we stand on the fluffy-tailed shoulders of these intrepid pioneers. From quantum computing to theoretical cosmology, their contributions continue to reverberate across disciplines. So the next time you spot a squirrel darting through the treetops, pause and reflect on the hidden depths of their intellectual prowess. And maybe offer them a peanut or two, in humble tribute to their groundbreaking work."

    original_prompt = "Squirrels are known for"
    prompt = original_prompt
    k = 100
    N = 512
    seed = randint(0, 1000)
    temp = 1.0

    set_seed(seed)

    print(f'Run: {run}, Seed: {seed}')

    steps_info = []
    not_in_top_k_counter = 0
    correctly_expected_step_num = 0

    natural_prompt_tokens = tokenizer.encode(natural_prompt, return_tensors='pt').to(torch_device)
    original_prompt_tokens = tokenizer.encode(original_prompt, return_tensors='pt').to(torch_device)

    print(f'Natural: {natural_prompt_tokens[0]}')
    print(f'Original: {original_prompt_tokens[0]}')

    for step_num in range(len(original_prompt_tokens[0]), len(natural_prompt_tokens[0])):

        known_next_token_id = natural_prompt_tokens[0][step_num]

        print(f'Step: {step_num}, Next known id: {known_next_token_id}')

        top_k_tokens, top_k_probs, top_k_indices, known_next_token_prob = get_top_k_tokens(prompt, k, known_next_token_id, temp)
  
        # Find the indices where top_k_indices matches known_next_token_id
        matches = torch.where(top_k_indices == known_next_token_id)[0]

        if len(matches) <= 0:
            print(f'!!! - {known_next_token_prob}')
            not_in_top_k_counter += 1
            selected_index = k-1
            top_k_tokens[selected_index] = tokenizer.decode(known_next_token_id)
            top_k_probs[selected_index] = known_next_token_prob
            top_k_indices[selected_index] = known_next_token_id
        else:
            # If there's at least one match, select the first one
            selected_index = matches[0].item()  

        top_k_probs_normalized = top_k_probs / top_k_probs.sum()
 
        selected_token = tokenizer.decode([top_k_indices[selected_index]])
        prompt += selected_token
        correctly_expected_step_num += 1

        # Record step information
        step_info = {
            'step_num': correctly_expected_step_num,
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
    json_filename = f'json/claude_output_statistics_k{k}_N{N}_seed{seed}.json'

    # Save the JSON to a file
    with open(json_filename, 'w') as f:
        json.dump(output_json, f, indent=4)

    # Creating a non-unique filename
    dummy_json_filename = f'json/most_recent_output_statistics.json'

    # Save the JSON to a file
    with open(dummy_json_filename, 'w') as f:
        json.dump(output_json, f, indent=4)

    # Print final prompt and JSON file info
    # print(f"Final generated text: {prompt}")
    print(f"Correct steps: {correctly_expected_step_num}, Missed Top-K: {not_in_top_k_counter}")
    print(f"Statistics saved to: {json_filename}")