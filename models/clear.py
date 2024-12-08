import os

# clear all checkpoint* files under the directory
def clear_checkpoint_files(directory):
    for folder in os.listdir(directory):
        if not os.path.isdir(os.path.join(directory, folder)):
            continue
        for subfolder in os.listdir(os.path.join(directory, folder)):
            if subfolder.startswith("checkpoint"):
                # remove folder
                os.system(f"rm -rf {os.path.join(directory, folder, subfolder)}")

if __name__ == "__main__":
    clear_checkpoint_files("models")
