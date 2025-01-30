import argparse
import json
import os
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from copy import deepcopy
from UnSec.tools.llm_controllers import LLMBot
from UnSec.tools.fileios import *
from UnSec.tools.themer import Themer
import networkx as nx


def generate_graph_candidate_sentences(gpt_results):
    """
    Generate a new dictionary containing candidate sentences by incorporating the most similar and most dissimilar nodes.

    Args:
        gpt_results (dict): Original gpt_results containing nodes and corresponding information.

    Returns:
        dict: A new dictionary containing candidate sentences for each node with additional combinations.
    """
    updated_results = {}

    for node_id, node_info in gpt_results.items():
        base_sentences = node_info['candidate_sentences']
        hard_negative_nodes = node_info['hard_negative_nodes_with_scores']
        easy_negative_nodes = node_info['easy_negative_nodes_with_scores']

        # Create combinations for candidate sentences
        graph_candidate_sentences = []
        similarity_combinations = []

        # Select the most similar and the most dissimilar nodes
        if hard_negative_nodes:
            most_similar_node, most_similar_score = hard_negative_nodes[0]
        if easy_negative_nodes:
            most_dissimilar_node, most_dissimilar_score = easy_negative_nodes[-1]

        for base_sentence in base_sentences:
            # Add combination with the most similar and most dissimilar node information
            new_sentence = (
                f"{base_sentence}. It is similar to {most_similar_node} and dissimilar to {most_dissimilar_node}."
            )
            graph_candidate_sentences.append(new_sentence)
            similarity_combinations.append({
                'similar_node': most_similar_node,
                'similarity_score': most_similar_score,
                'dissimilar_node': most_dissimilar_node,
                'dissimilarity_score': most_dissimilar_score
            })

        # Update node_info with the newly generated sentences and similarity combinations
        node_info['graph_candidate_sentences'] = graph_candidate_sentences
        node_info['similarity_combinations'] = similarity_combinations

        # Save the updated node_info to a new dictionary
        updated_results[node_id] = node_info

    return updated_results


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='inat_llm_graph_answers')
    parser.add_argument('--out_path', default='inat_llm_graph_answers_ablation3')


    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not os.path.exists(args.out_path):
        raise FileExistsError(f"Output path {args.out_path} does not exist.")

    # Define all the dataset levels for iNat dataset
    if args.dataset_name == 'inat':
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
    else:
        level_names = ['l3', 'l2', 'l1']

    for level_name in level_names:
        # Load the GPT hierarchy results for each level
        gpt_results = load_json(
            os.path.join(args.gpt_results_root, f"cleaned_{args.dataset_name}_gpt_graph_hrchy_{level_name}.json"))

        # Generate graph candidate sentences
        updated_results = generate_graph_candidate_sentences(gpt_results)

        # Define the output path and save the results
        output_path = os.path.join(args.out_path, f"cleaned_{args.dataset_name}_gpt_graph_hrchy_{level_name}.json")
        dump_json(output_path, updated_results)
        print(f"Saved updated graph candidate sentences to {output_path}")
