import os

import random

import logging
from PIL import Image
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import utils

SEED = 212

def download_dataset(data_dir='./dataset'):
    os.makedirs(data_dir, exist_ok=True)
    filename='SIGNS.zip'
    gdd.download_file_from_google_drive(file_id='1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC',
                                        dest_path=os.path.join(data_dir, filename),
                                        unzip=True)


def resize_and_save(filename, output_dir_split, size=64):
    image = Image.open(filename)
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir_split, filename.split('/')[-1]))

def preprocess_dataset(data_dir='./dataset', SIZE=64):
    assert os.path.isdir(data_dir), f"Could not find the dataset at {data_dir}"

    train_data_dir = os.path.join(data_dir, 'SIGNS dataset/train_signs')
    test_data_dir = os.path.join(data_dir, 'SIGNS dataset/test_signs')

    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    train_filenames, val_filenames = train_test_split(filenames,
                                                      test_size=0.2,
                                                      random_state=SEED)
    filenames = {
        'train': train_filenames,
        'val': val_filenames,
        'test': test_filenames
    }
    output_dir = os.path.join(data_dir, f"{SIZE}x{SIZE}_SIGNS")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        logging.info(f"Warning: output dir {output_dir} already exists")

    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(output_dir, f'{split}_signs')
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            logging.info(f'Warning: dir {output_dir_split} already exists.')

        logging.info(f'Preprocessing {split} data, saving data into {output_dir_split}')
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

if __name__ == '__main__':
    data_dir = './dataset'
    utils.set_logger(os.path.join(data_dir, 'dataset_preparation.log'))

    logging.info('Preparing dataset...')

    logging.info('Downloading dataset...')
    download_dataset(data_dir)
    logging.info(f'-done.')

    logging.info('split and resize dataset...')
    preprocess_dataset(data_dir, SIZE=64)
    logging.info('- done.')