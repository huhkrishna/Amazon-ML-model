import os
import pandas as pd
from src.utils import download_images
from src.model import EntityExtractor

DATASET_FOLDER = 'dataset/'
IMAGE_FOLDER = 'images/'
OUTPUT_FOLDER = 'output/'

if __name__ == "__main__":
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    print("Image Links:")
    print(test_df['image_link'].head())

    download_images(test_df['image_link'], IMAGE_FOLDER)

    print("Downloaded Images:")
    print(os.listdir(IMAGE_FOLDER))

    extractor = EntityExtractor()
    extractor.generate_predictions(
        os.path.join(DATASET_FOLDER, 'test.csv'), 
        os.path.join(OUTPUT_FOLDER, 'test_out.csv'),
        os.path.join(DATASET_FOLDER, 'train.csv')
    )

    output_df = pd.read_csv(os.path.join(OUTPUT_FOLDER, 'test_out.csv'))
    print("Output Predictions:")
    print(output_df.head())

    os.system(f'python src/sanity.py --test_filename {os.path.join(DATASET_FOLDER, "sample_test.csv")} --output_filename {os.path.join(OUTPUT_FOLDER, "test_out.csv")}')
