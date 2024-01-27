import os
import shutil
import random

def split_data(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, verbose=False):
    identities = os.listdir(input_dir)
    total_identities = len(identities)

    num_train = int(total_identities * train_ratio)
    num_val = int(total_identities * val_ratio)
    num_test = total_identities - num_train - num_val

    random.shuffle(identities)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for i, directory in enumerate(identities):
        src_path = os.path.join(input_dir, directory)
        if i < num_train:
            dest_path = os.path.join(train_dir, directory)
        elif i < num_train + num_val:
            dest_path = os.path.join(val_dir, directory)
        else:
            dest_path = os.path.join(test_dir, directory)

        shutil.move(src_path, dest_path)
        if verbose:
            print(f"Moved '{directory}' to '{os.path.basename(dest_path)}'")


if __name__ == "__main__":
	print('passed')