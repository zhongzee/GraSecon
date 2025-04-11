import os
import subprocess

def activate_environment_and_move():
    # Activate the conda environment
    os.system('conda activate GraSecon')

    # Change directory
    try:
        os.chdir('GraSecon')
    except FileNotFoundError:
        print("Directory 'GraSecon' not found. Exiting.")
        exit()

def build_nexus():
    nexus_paths = {
        "ViT-B/32": ".././nexus/lvis/GraSecon_llm"
    }

    for clip_model, out_path in nexus_paths.items():
        command = [
            'python',
            '-W',
            'ignore',
            'build_miss_inat_fsod_aggr_w_llm_hrchy.py',
            '--dataset_name',
            'lvis',
            '--prompter',
            'isa',
            '--aggregator',
            'mean',
            '--clip_model',
            clip_model,
            '--out_path',
            out_path
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            exit()

# Main script execution
if __name__ == "__main__":
    activate_environment_and_move()
    build_nexus()
