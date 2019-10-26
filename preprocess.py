import sys
import os
import numpy as np
import PIL.Image

WIDTH = 1600
HEIGHT = 256

if len(sys.argv) < 2:
    print("Usage: python3 preprocess.py [PATH_TO_DATASET_ROOT]")
    exit(1)

dataset_root = sys.argv[1]

train_csv = os.path.join(dataset_root, "train.csv")
mask_dir = os.path.join(dataset_root, "train_masks")

with open(train_csv, 'r') as f:
    try:
        os.mkdir(mask_dir)
    except OSError as e:
        pass

    # The first line of csv is header
    for l in f.readlines()[1:]:
        # Get rid of '\n'
        l = l[:-1]

        # Decode the csv
        pos = l.find(',')
        sample = l[0:pos]
        encoded_pixels = l[pos+1:]
        if len(encoded_pixels) == 0:
            continue

        encoded_pixels = np.array(encoded_pixels.split(' '), dtype='int')
        encoded_pixels = encoded_pixels.reshape(-1, 2)

        # Build the class mask
        mask = np.zeros([HEIGHT, WIDTH], dtype='uint8')
        for (pixel_1d, length) in encoded_pixels:
            for pixel in range(pixel_1d, pixel_1d+length):
                pixel = pixel - 1
                x = int(pixel / HEIGHT)
                y = pixel % HEIGHT
                mask[y,x] = 255

        # Write the class mask to a png file
        mask_path = os.path.join(mask_dir, f'{sample}.png')
        PIL.Image.fromarray(mask).save(mask_path)
