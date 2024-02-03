import os
import shutil
import random
import time


def split_data(
    input_path,
    output_path,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    verbose=False,
    num_identities=None,
):
    if verbose:
        print("Splitting data into train, val, test")

    train_path = os.path.join(output_path, "train")
    val_path = os.path.join(output_path, "val")
    test_path = os.path.join(output_path, "test")

    if os.path.exists(train_path):
        print("Terminating. Have you already splitted the data?")
        return train_path, val_path, test_path

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    if num_identities == -1:
        num_identities = len(os.listdir(input_path))

    start_time = time.time()
    if verbose:
        print("Moving files...")

    for identity, (root, dirs, files) in enumerate(os.walk(input_path)):
        if identity == 0:
            if verbose:
                print("Skipping root directory")

            continue
        if identity > num_identities:
            break

        identity_train_path = os.path.join(train_path, str(identity - 1))
        identity_val_path = os.path.join(val_path, str(identity - 1))
        identity_test_path = os.path.join(test_path, str(identity - 1))

        os.makedirs(identity_train_path, exist_ok=True)
        os.makedirs(identity_val_path, exist_ok=True)
        os.makedirs(identity_test_path, exist_ok=True)

        random.shuffle(files)
        train_files = files[: int(len(files) * train_ratio)]
        val_files = files[
            int(len(files) * train_ratio) : int(len(files) * (train_ratio + val_ratio))
        ]
        test_files = files[int(len(files) * (train_ratio + val_ratio)) :]

        for img in train_files:
            shutil.copy(os.path.join(root, img), identity_train_path)

        for img in val_files:
            shutil.copy(os.path.join(root, img), identity_val_path)

        for img in test_files:
            shutil.copy(os.path.join(root, img), identity_test_path)

    if verbose:
        print("Done!!!")
        print(f"Time taken: {time.time() - start_time} seconds")
    return train_path, val_path, test_path


if __name__ == "__main__":
    print("passed")
