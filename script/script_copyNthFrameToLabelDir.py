import os
import shutil

nth_element = 30
filetype = "rgb"  # disparity

# Define the folder path
folder_path = "data/darus_data_download/data/20230807_162939_ManipulationSequences_mountedWireHarness_partial/data"  # replace with your folder path

save_dir = "labels"

save_folder_path = os.path.join(os.path.dirname(folder_path), save_dir)

# Get all the files of desired type in the directory
all_files = [
    f
    for f in os.listdir(folder_path)
    if (
        os.path.isfile(os.path.join(folder_path, f))
        and (f.split(".")[-2]).split("_")[-1] == filetype
    )
]

# Retrieve every n-th file
nth_files = all_files[0::nth_element]

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

# Print the result
for file in nth_files:
    shutil.copy(os.path.join(folder_path, file), os.path.join(save_folder_path, file))
