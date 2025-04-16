import os
import argparse
from ferminet.checkpoint import restore
import numpy as np


def load_last_n_checkpoints(folder_path, n):
    checkpoint_num = [
        f.split("_")[-1].replace(".npz", "")
        for f in os.listdir(folder_path)
        if "qmcjax_ckpt_" in f
    ]
    checkpoint_num.sort(key=lambda x: int(x))
    start_index = max(0, len(checkpoint_num) - n)
    last_n_checkpoints_nums = checkpoint_num[start_index:]

    # takes from 1 GPU only. Apparently getting data from all GPUs is hard?
    position_data = []
    spin_data = []
    for num in last_n_checkpoints_nums:
        _, data, _, _, _, _ = restore(f"{folder_path}/qmcjax_ckpt_{num}.npz")
        spin_data.append(data.spins)
        position_data.append(data.positions)


    np.save(f"{folder_path}/inference_out.npy", position_data)
    np.save(f"{folder_path}/inference_out_spins.npy", spin_data)

    return position_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load the last N checkpoints from a folder."
    )
    parser.add_argument(
        "folder_path", type=str, help="The path to the folder containing checkpoints."
    )
    parser.add_argument("n", type=int, help="The number of recent checkpoints to load.")

    args = parser.parse_args()

    print(f"Loading last {args.n} checkpoints from {args.folder_path}")
    position_data = load_last_n_checkpoints(args.folder_path, args.n)
    print(f"Position data shape: {position_data.shape}")
