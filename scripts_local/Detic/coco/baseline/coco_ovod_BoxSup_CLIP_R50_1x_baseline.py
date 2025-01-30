import subprocess


def run_command():
    # Activate the Conda environment (handled differently in Python)
    # Assuming 'UnSec' is the name of your conda environment
    conda_path = "/root/miniconda3/bin/conda"
    activate_command = f"source {conda_path} activate UnSec"

    # Define the Python command to execute the training script with parameters
    metadata_root = "./nexus/coco/baseline"
    command = [
        "python", "train_net_detic_coco.py",
        "--num-gpus", "1",
        "--config-file", "./configs_detic/BoxSup_OVCOCO_CLIP_R50_1x.yaml",
        "--eval-only",
        "MODEL.WEIGHTS", "./models/detic/coco_ovod/BoxSup_OVCOCO_CLIP_R50_1x.pth",
        "MODEL.RESET_CLS_TESTS", "True",
        "MODEL.TEST_CLASSIFIERS", f"('{metadata_root}/coco_clip_hrchy_l1.npy',)",
        "MODEL.TEST_NUM_CLASSES", "(80,)",
        "MODEL.MASK_ON", "False"
    ]

    # Combine commands to include environment activationaggr_mean/coco_clip_hrchy_l1.npy
    full_command = f"{activate_command} && {' '.join(command)}"

    # Execute the command
    process = subprocess.run(full_command, shell=True, executable='/bin/bash', text=True, capture_output=True)

    # Print the outputs
    print("STDOUT:", process.stdout)
    print("STDERR:", process.stderr)


if __name__ == "__main__":
    run_command()
