import os
from tqdm import tqdm
import numpy as np

data_root = "/home/thomas/dev/isaac_projects/AMASS_Retargeted_for_G1/g1"

def check_dof_positions_in_npz(root_folder):
    """
    Recursively walk through 'root_folder', find each .npz file, and check whether
    'dof_positions' values are all within [-π, π]. Log violations.
    """
    files_to_process = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".npz"):
                files_to_process.append(os.path.join(dirpath, filename))

    violating_files = []

    for npz_path in tqdm(files_to_process, desc="Checking files for [-π, π] bounds"):
        try:
            data = np.load(npz_path)
            data_dict = dict(data)
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue

        if "dof_positions" in data_dict:
            dof_positions = data_dict["dof_positions"]
            if not np.all((dof_positions >= -np.pi) & (dof_positions <= np.pi)):
                violating_files.append(npz_path)

    if violating_files:
        print("\nFiles with 'dof_positions' outside [-π, π]:")
        for file in violating_files:
            print(file)
    else:
        print("\n✅ All files are within [-π, π] bounds.")

if __name__ == "__main__":
    check_dof_positions_in_npz(data_root)
