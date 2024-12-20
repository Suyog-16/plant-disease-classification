import os
def walkthrough_dir(dir_path):
    for dir_path,dir_names,filenames in os.walk(dir_path):
        print(f"There are {len(dir_names)} directories and {len(filenames)} images in {dir_path}")
    