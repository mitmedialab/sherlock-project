import os

from google_drive_downloader import GoogleDriveDownloader as gd


def download_data():
    """Download raw and preprocessed data files.
    The data is downloaded from Google Drive and stored in the 'data/' directory.
    """
    data_dir = '../data/data.zip'
    print(f"Downloading the raw and preprocessed data into {data_dir}.")

    if not os.path.exists(data_dir):
        print('Downloading data directory.')
        dir_name = data_dir
        gd.download_file_from_google_drive(
            file_id='1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU',
            dest_path=dir_name,
            unzip=True,
            showsize=True
        )

    print('Data was downloaded.')