import os

import pkg_resources
from google_drive_downloader import GoogleDriveDownloader as gd

DATA_PATH = pkg_resources.resource_filename("sherlock", "data/")


def make_data_path(path):
    return os.path.join(DATA_PATH, path)


def download_data():
    """Download raw and preprocessed data files.
    The data is downloaded from Google Drive and stored in the 'data/' directory.
    """
    data_dir = make_data_path("data.zip")
    print(f"Downloading the raw and preprocessed data into {data_dir}.")

    if not os.path.exists(data_dir):
        print("Downloading data directory.")
        gd.download_file_from_google_drive(
            file_id="1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU",
            dest_path=data_dir,
            unzip=True,
            showsize=True,
        )

    print("Data was downloaded.")
