import openai
import json
import argparse
import os
from tqdm import tqdm

# Set API credentials
openai.api_base = 'xx'
openai.api_key = 'xx'  # Replace with your OpenAI API key

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--hierarchy_root', type=str, default='./GraSecon_cls/hrchy_breeds')
parser.add_argument('--breed_level', type=str, default='all', choices=['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'all'], help="Level of the hierarchy to process")
args = parser.parse_args()

# Define hierarchy paths
hier_paths = {
    # 'l1': f"{args.hierarchy_root}/composed_breed_l2_num_class=10.json",
    # # 'l2': f"{args.hierarchy_root}/composed_breed_l3_num_class=29.json",
    # 'l3': f"{args.hierarchy_root}/composed_breed_l4_num_class=128.json",
    # 'l4': f"{args.hierarchy_root}/composed_breed_l5_num_class=466.json",
    'l5': f"{args.hierarchy_root}/composed_breed_l6_num_class=591.json",
    'l6': f"{args.hierarchy_root}/composed_breed_l7_num_class=98.json",
}

# Helper function to generate descriptions using OpenAI API
def generate_description(class_name, all_classnames):
    prompt = f"""
    For zero-shot learning and open-world object detection, succinctly describe '{class_name}' focusing on its distinguishing visual features compared to its all similar classes. 
    Include at least 3 distinct attributes in the description. Avoid any irrelevant, meaningless descriptions. Answer as concisely and accurately as possible. No more than 50 words.
    Please describe mainly the external visual features rather than the internal structure.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",#gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=70,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating description for {class_name}: {str(e)}")
        return None

# Load JSON file
def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Save JSON file
def dump_json(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

# Step 1: Check for missing details
def check_missing_details(levels, hierarchy_root):
    missing_details = {}

    for level in levels:
        input_file_path = hier_paths.get(level)

        # Check if input file exists
        if not os.path.exists(input_file_path):
            print(f"Input file {input_file_path} does not exist, skipping...")
            continue

        # Read input JSON file
        with open(input_file_path, 'r') as file:
            hierarchical_structure = json.load(file)

        # Check for nodes without detail_sentences
        for key, value in hierarchical_structure.items():
            if isinstance(value, dict):
                node_name = value.get("node_name")
                if node_name and "detail_sentences" not in value:
                    if level not in missing_details:
                        missing_details[level] = []
                    missing_details[level].append(node_name)

    return missing_details

# Step 2: Process and generate descriptions for missing nodes
def process_missing_details(missing_details, hierarchy_root):
    log_file_path = 'process_log.json'
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            process_log = json.load(log_file)
    else:
        process_log = {}

    unmatched_categories = []

    for level, nodes in missing_details.items():
        input_file_path = hier_paths.get(level)
        output_file_path = os.path.join(os.path.dirname(input_file_path), f"composed_detail_breed_{level}_num_class={len(nodes)}.json")

        # Check if input file exists
        if not os.path.exists(input_file_path):
            print(f"Input file {input_file_path} does not exist, skipping...")
            continue

        # Read input JSON file
        with open(input_file_path, 'r') as file:
            hierarchical_structure = json.load(file)

        # Get all node names
        all_classnames = [value.get("node_name") for key, value in hierarchical_structure.items() if isinstance(value, dict) and value.get("node_name")]

        # Process missing descriptions
        total_nodes = len(nodes)
        progress_bar = tqdm(total=total_nodes, desc="Generating missing descriptions", unit="node")

        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as file:
                hierarchical_structure_output = json.load(file)
        else:
            hierarchical_structure_output = {}

        # Generate descriptions for missing nodes
        for node_name in nodes:
            for node_id, value in hierarchical_structure.items():
                if isinstance(value, dict) and value.get("node_name") == node_name:
                    if "detail_sentences" in value:
                        continue
                    
                    description = generate_description(node_name, all_classnames)

                    if description:
                        if "detail_sentences" not in value:
                            value["detail_sentences"] = []
                        value["detail_sentences"].append(description)

                        hierarchical_structure_output[node_id] = value
                        dump_json(output_file_path, hierarchical_structure_output)
                    else:
                        unmatched_categories.append(node_name)

                    progress_bar.update(1)

        progress_bar.close()

        if unmatched_categories:
            print(f"Unmatched categories: {unmatched_categories}")

        print(f"Missing descriptions added and saved to {output_file_path}.")

# Step 3: Process the levels and save the final result
from tqdm import tqdm

def process_levels():
    levels_to_process = [args.breed_level] if args.breed_level != 'all' else hier_paths.keys()

    for level in levels_to_process:
        print(f"Processing level: {level}")
        
        hier_path = hier_paths.get(level)
        if not hier_path:
            print(f"Invalid level {level}, skipping...")
            continue
        
        class_tree = load_json(hier_path)

        # Process each node in the class tree with progress bar
        total_nodes = len(class_tree)
        progress_bar = tqdm(total=total_nodes, desc=f"Processing nodes for level {level}", unit="node")

        # Process each node in the class tree
        for node_id, node_data in class_tree.items():
            if isinstance(node_data, dict):
                node_name = node_data.get("node_name")
                
                if node_name:
                    # Generate description for the node directly
                    description = generate_description(node_name, list(class_tree.keys()))
                    
                    if description:
                        if "detail_sentences" not in node_data:
                            node_data["detail_sentences"] = []
                        node_data["detail_sentences"].append(description)

            # Update the progress bar after processing each node
            progress_bar.update(1)

        # Close the progress bar when done
        progress_bar.close()

        # Save the updated hierarchy to file
        output_path = os.path.join(os.path.dirname(hier_path), f"composed_detail_breed_{level}_num_class={len(class_tree)}.json")
        dump_json(output_path, class_tree)
        print(f"Updated hierarchy for {level} saved to {output_path}")


if __name__ == "__main__":
    process_levels()
      
    # Step 1: Check for missing details
    missing_details = check_missing_details([args.breed_level] if args.breed_level != 'all' else hier_paths.keys(), args.hierarchy_root)

    # Step 2: Process and generate descriptions for missing nodes
    process_missing_details(missing_details, args.hierarchy_root)

    # Step 3: Process the levels and save the final result
    # process_levels()
