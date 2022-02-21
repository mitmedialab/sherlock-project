import os
import zipfile

import gdown


def download_data():
    """Download raw and preprocessed data files.
    The data is downloaded from Google Drive and stored in the 'data/' directory.
    """
    data_dir = "../data/"
    zip_filepath = "../data/data.zip"
    print(f"Downloading the raw data into {data_dir}.")

    if not os.path.exists(zip_filepath):
        print("Downloading data directory.")
        gdown.download(
            url="https://drive.google.com/uc?id=1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU",
            output=data_dir,
        )

        with zipfile.ZipFile(zip_filepath, "r") as zf:
            zf.extractall(data_dir)

    print("Data was downloaded.")
