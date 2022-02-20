import os
import zipfile

import gdown


def download_data():
    """Download raw and preprocessed data files.
    The data is downloaded from Google Drive and stored in the 'data/' directory.
    """
    data_dir = "../data/data/"
    data_zip = "../data/data.zip"
    print(f"Downloading the raw data into {data_dir}.")

    if not os.path.exists(data_dir):
        print("Downloading data directory.")
        dir_name = data_zip
        gdown.download(
            url="https://drive.google.com/uc?id=1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU",
            output=dir_name,
        )

        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extractall(data_dir)

    print("Data was downloaded.")
