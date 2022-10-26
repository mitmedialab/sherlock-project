import os
import zipfile

import gdown


def download_data():
    """Download raw and preprocessed data files.
    The data is downloaded from Google Drive and stored in the 'data/' directory.
    """
    data_dir = "../data/data/"
    zip_filepath = "../data/data.zip"
    print(f"Downloading the raw data into {data_dir}.")

    if not os.path.exists(data_dir):
        print("Downloading data directory.")
        gdown.download(
            # url="https://drive.google.com/u/0/uc?id=1RWB7djA5cJ9Nuw41SxKtEhwDyqqfcJst&export=download",
            url = "https://drive.google.com/u/1/uc?id=1EjC6V3kx5XlwHYPVpwI8SOuVMVUEGCgm&export=download",
            output=zip_filepath,
        )

        with zipfile.ZipFile(zip_filepath, "r") as zf:
            zf.extractall("../data/")

    print("Data was downloaded.")
