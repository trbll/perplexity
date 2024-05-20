import os
import json
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np

def compute_log_perplexity(json_data):

    # Extract probabilities of the selected tokens at each step

    log_probs = []
    running_step_averages = []
    
    for step in json_data['steps']:
        for token in step['top_ks']:
            if token['selected']:
                selected_prob = token['prob_norm']
                log_prob = -math.log2(selected_prob)
                log_probs.append( log_prob )
                running_average = sum( log_probs ) / len( log_probs )
                running_step_averages.append( running_average )

    log_perplexity = running_step_averages[-1]

    return log_perplexity, running_step_averages

def compute_average_entropy(json_data):

    entropies = []
    variances = []
    running_step_average_entropy = []
    running_step_sigmas = []
    
    for step in json_data['steps']:
        step_entropy = 0
        s_p = 0 # second moment of the law of perplexity
        for token in step['top_ks']:
            prob = token['prob_norm']
            step_entropy -= prob * math.log2(prob) if prob > 0 else 0  # To avoid math domain error
            s_p += (prob * math.pow(math.log2(prob),2)) if prob > 0 else 0

        entropies.append(step_entropy)
        running_average = sum( entropies ) / len( entropies )
        running_step_average_entropy.append( running_average )

        step_var = s_p - math.pow(step_entropy,2)
        variances.append(step_var)
        single_sigma = math.sqrt(sum(variances)) / len(variances)
        running_step_sigmas.append( single_sigma )

    # Calculate the average of the entropies
    average_entropy = sum(entropies) / len(entropies) if entropies else 0

    sigma = math.sqrt(sum(variances)) / len(variances)

    return average_entropy, running_step_average_entropy, sigma, running_step_sigmas

def plot_perplexity_vs_entropy(stepwise_log_perplexity, stepwise_avg_entropy, stepwise_sigmas, topK, nSteps, seed):

    # Plotting both stepwise log perplexity and stepwise average entropy on the same plot
    plt.figure(figsize=(12, 7))  # Set the figure size for better visibility

    # Plot stepwise average entropy
    plt.plot(stepwise_avg_entropy, label='Average Entropy', marker='x', linestyle='--', color='red')

    # Plot stepwise log perplexity
    plt.plot(stepwise_log_perplexity, label='Log Perplexity', marker='o', linestyle='-', color='blue')

    # Calculate the upper and lower bounds for the entropy uncertainty
    upper_bound = np.array(stepwise_avg_entropy) + np.array(stepwise_sigmas)
    lower_bound = np.array(stepwise_avg_entropy) - np.array(stepwise_sigmas)

    double_upper_bound = np.array(upper_bound) + np.array(stepwise_sigmas)
    double_lower_bound = np.array(lower_bound) - np.array(stepwise_sigmas)

    # Fill the area between the upper and lower bounds to show uncertainty
    plt.fill_between(range(len(stepwise_avg_entropy)), lower_bound, upper_bound, color='red', alpha=0.2, label=r'Entropy Uncertainty, $1\sigma$')
    plt.fill_between(range(len(stepwise_avg_entropy)), double_lower_bound, double_upper_bound, color='gray', alpha=0.2, label=r'Entropy Uncertainty, $2\sigma$')

    # Adding labels and title
    plt.xlabel('Step')
    plt.ylabel('Entropy (Bits)')
    plt.suptitle('Stepwise Log Perplexity vs Average Entropy', fontsize=14)
    plt.title(f'Perplexity={stepwise_log_perplexity[-1]:.4f}, Avg. Entropy={stepwise_avg_entropy[-1]:.4f}, Sigma={stepwise_sigmas[-1]:.4f}\nK={topK}, N={nSteps}, seed={seed}', fontsize=10, color='grey')

    # Adding legend to differentiate the lines and the filled area
    plt.legend()

    # Adding grid for better readability
    plt.grid(True)

    plt.savefig(f'plots/plot_k{topK}_N{nSteps}_seed{seed}.png', format='png', bbox_inches='tight')

    # Show the plot
    plt.show()

    plt.close()

def process_json_file(file_path):

    with open(file_path, 'r') as f:
    
        # Load the JSON data
        json_data = json.load(f)

        topK = json_data['k']
        nSteps = json_data['nSteps']
        seed = json_data['seed']

        # Compute the log-perplexity
        ( log_perplexity, stepwise_log_perplexity ) = compute_log_perplexity(json_data)
        # print(f"Log-perplexity: {log_perplexity}")
        # print(f"Stepwise Log-perplexity: {stepwise_log_perplexity}")

        # Compute and print the average entropy
        ( average_entropy, stepwise_avg_entropy, sigma, stepwise_sigmas) = compute_average_entropy(json_data)
        # print(f"Average Entropy: {average_entropy}")
        # print(f"Stepwise Average Entropy: {stepwise_avg_entropy}")
        # print(f"Sigma Entropy: {sigma}")
        # print(f"Stepwise Sigmas: {stepwise_sigmas}")

        diff = abs( log_perplexity - average_entropy ) / sigma 

        # Plot the perplexity vs average entropy graph over nRuns 
        plot_perplexity_vs_entropy(stepwise_log_perplexity, stepwise_avg_entropy, stepwise_sigmas, topK, nSteps, seed)

        return log_perplexity, average_entropy, sigma

# Set up argument parser
parser = argparse.ArgumentParser(description='Process a JSON file for statistical analysis.')
parser.add_argument('path', type=str, help='Path to the JSON file or directory to be processed')

# Parse arguments
args = parser.parse_args()

# Check if the path is a directory or a file
if os.path.isdir(args.path):

    nFiles = 0
    perplexities = []
    entropies = []
    sigmas = []

    # If it's a directory, process each file in the directory
    for filename in os.listdir(args.path):

        if 'most_recent_output_statistics' in filename:
            continue

        file_path = os.path.join(args.path, filename)

        if os.path.isfile(file_path) and filename.endswith('.json'):

            print(f'File {nFiles} | Processing: {filename}')
            
            (log_perplexity, avg_entropy, sigma) = process_json_file(file_path)

            perplexities.append(log_perplexity)
            entropies.append(avg_entropy)
            sigmas.append(sigma)

            nFiles += 1

    avg_perplexities = sum(perplexities) / len(perplexities)
    avg_entropies = sum(entropies) / len(entropies)
    avg_sigmas = sum(sigmas) / len(sigmas)

    print(f'Processed files: {nFiles}')
    print(f'Avg. Perplexity: {avg_perplexities}')
    print(f'Avg. Entropies: {avg_entropies}')
    print(f'Avg. Sigmas: {avg_sigmas}')

else:
    # If it's a file, process the file
    (log_perplexity, avg_entropy, sigma) = process_json_file(args.path)
    print(f'Log Perplexity: {log_perplexity}')
    print(f'Entropy: {avg_entropy}')
    print(f'Sigma: {sigma}')