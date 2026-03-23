"""
One-time utility for reorganizing a flat image dataset into class subfolders.
Parses the label from each filename (e.g. "image_00002_1.png" -> class "1")
and moves files into corresponding subdirectories for compatibility with
standard image loading pipelines.
"""

import argparse
import logging
import os
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def organize_images_into_subfolders(base_path):
    for split in ['train', 'test_data']:
        split_path = os.path.join(base_path, split)
        if not os.path.isdir(split_path):
            logger.warning(f"Directory '{split_path}' not found, skipping")
            continue

        files = [f for f in os.listdir(split_path) if f.endswith('.png')]
        logger.info(f"Found {len(files)} images in '{split_path}'")

        for fname in files:
            try:
                label = fname.split('_')[-1].split('.')[0]
                label_dir = os.path.join(split_path, label)
                os.makedirs(label_dir, exist_ok=True)

                src_path = os.path.join(split_path, fname)
                dst_path = os.path.join(label_dir, fname)

                shutil.move(src_path, dst_path)
            except Exception as e:
                logger.error(f"Error processing file '{fname}': {e}")

    logger.info("Done organizing files into class subfolders")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Organize flat image dataset into class subfolders")
    parser.add_argument('--dataset-dir', type=str, required=True, help="Path to dataset root (containing train/ and test_data/)")
    args = parser.parse_args()

    organize_images_into_subfolders(args.dataset_dir)
