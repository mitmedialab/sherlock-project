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
            url="https://drive.google.com/uc?id=1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU",
            output=zip_filepath,
        )

        with zipfile.ZipFile(zip_filepath, "r") as zf:
            zf.extractall("../data/")

    print("Data was downloaded.")
