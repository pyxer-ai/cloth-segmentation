import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset


def rle_encoding(mask: np.ndarray) -> str:
    """
    Converts a mask into run-length encoding (RLE) format using vectorized operations.

    Args:
        mask (`np.ndarray`): The mask where each pixel value represents the class ID.

    Returns:
        `str`: The RLE string.
    """
    pixels = mask.T.flatten()  # Transpose to switch to column-major order
    flat_pixels = np.r_[-1, pixels, -1]  # Add sentinel values for easier run detection
    runs = np.diff(np.where(flat_pixels != np.roll(flat_pixels, 1))[0])
    start_positions = np.where(flat_pixels[1:] != flat_pixels[:-1])[0] + 1

    run_lengths = runs[1::2]
    starts = start_positions[1::2] + 1

    encoded_rle = ' '.join(f"{start} {length}" for start, length in zip(starts, run_lengths))

    return encoded_rle


def create_train_csv(dataset, output_csv: str, class_mapping: dict, ignore_classes: set):
    """
    Creates a train.csv file with the columns ImageId, EncodedPixels, Height, Width, and ClassId.

    Args:
        dataset (`DatasetDict`): The Hugging Face DatasetDict object containing the dataset.
        output_csv (`str`): The path to the output CSV file.
        class_mapping (`dict`): A dictionary mapping original class IDs to new class IDs.
        ignore_classes (`set`): A set of class IDs to be ignored.
    """
    records = []

    for index, item in tqdm(enumerate(dataset['train']), total=len(dataset['train'])):
        image = np.array(item['mask'])
        height, width = image.shape
        image_id = f"image_{index}"  # Generate a unique name for each image

        # Initialize masks for each new class ID
        masks = {new_class_id: np.zeros((height, width), dtype=np.uint8) for new_class_id in class_mapping.values()}
        
        for original_class_id in range(18):
            if original_class_id in ignore_classes:
                continue

            new_class_id = class_mapping.get(original_class_id, original_class_id)
            if new_class_id not in masks:
                continue

            class_mask = (image == original_class_id).astype(np.uint8)
            masks[new_class_id] += class_mask

        for new_class_id, combined_mask in masks.items():
            if combined_mask.sum() > 0:
                encoded_pixels = rle_encoding(combined_mask)
                records.append({
                    "ImageId": image_id,
                    "EncodedPixels": encoded_pixels,
                    "Height": height,
                    "Width": width,
                    "ClassId": new_class_id
                })
        

    df = pd.DataFrame(records, columns=["ImageId", "EncodedPixels", "Height", "Width", "ClassId"])
    df.to_csv(output_csv, index=False)
    return df

def main():
    ds = load_dataset("mattmdjaga/human_parsing_dataset")
    class_mapping = {
        0: 0,
        4: 1,
        5: 2,
        6: 2,
        7: 3,
    }
    ignore_classes = {1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    df_atr = create_train_csv(ds, "train.csv", class_mapping, ignore_classes)

if __name__ == "__main__":
    main()