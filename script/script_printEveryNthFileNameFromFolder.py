import os

nth_element = 50
filetype = "rgb"  # disparity

# Define the folder path
folder_path = "data/darus_data_download/data/20230517_093927_manipulationsequence_manual_labeled_yshape/data"  # replace with your folder path

# Get all the files in the directory
all_files = [
    f
    for f in os.listdir(folder_path)
    if (
        os.path.isfile(os.path.join(folder_path, f))
        and (f.split(".")[-2]).split("_")[-1] == filetype
    )
]

# Retrieve every 2nd file
nth_files = all_files[1::nth_element]

# Print the result
for file in nth_files:
    print('"{}"'.format(file), end=" ")
