import subprocess

def run_command():
    # Environment and metadata root settings
    # metadata_root = "./nexus/lvis/baseline"
    metadata_root = "./nexus/lvis/baseline"

    # Define the Python command to execute the training script with parameters
    command = [
        "/root/miniconda3/envs/GraSecon2/bin/python3", "train_net_detic.py",
        "--num-gpus", "1",
        "--config-file", "./configs_detic/Detic_LbaseCCimg_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml",
        "--eval-only",
        "DATASETS.TEST", "('lvis_v1_val',)",
        "MODEL.WEIGHTS", "./models/detic/lvis_ovod/Detic_LbaseCCimg_CLIP_R5021k_640b64_4x_ft4x_max-size.pth",
        "MODEL.RESET_CLS_TESTS", "True",
        "MODEL.TEST_CLASSIFIERS", f"('{metadata_root}/lvis_clip_hrchy_l1.npy',)",
        "MODEL.TEST_NUM_CLASSES", "(1203,)",
        "MODEL.MASK_ON", "False"
    ]

    # Execute the command
    process = subprocess.run(command, text=True, capture_output=True)

    # Print the outputs
    print("STDOUT:", process.stdout)
    print("STDERR:", process.stderr)

if __name__ == "__main__":
    run_command()
