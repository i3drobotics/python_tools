import os
import argparse
import glob

import numpy as np
import cv2

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))

    default_input_folder = "/home/i3dr/Desktop/"
    default_output_folder = script_dir+"/output/"
    parser = argparse.ArgumentParser(description="Convert RGB gray image to single channel gray image")
    parser.add_argument("-i","--input_folder", help="Input folder containing RGB gray images", default=default_input_folder)
    parser.add_argument("-w","--wildcard", help="Wildcard for image in folder", default="*_rect.png")
    parser.add_argument("-o","--output_folder", help="Output directory", default=default_output_folder)

    args = parser.parse_args()

    #load images
    image_fns = glob.glob(args.input_folder + '/' + args.wildcard)

    for image_fn in image_fns:
        output_fn = args.input_folder + os.path.splitext(os.path.basename(image_fn))[0] + "_gray.png"
        img = cv2.imread(image_fn,cv2.IMREAD_GRAYSCALE)

        # Convert source image to unsigned 8 bit integer Numpy array
        arr = np.uint8(img)
        cv2.imwrite(output_fn,arr)
        print(output_fn)

    return

if __name__ == '__main__':
    main()