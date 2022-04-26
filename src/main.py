import sys
import imageio
import os
import config
import cv2
import traceback
import numpy as np

from mincut import Mincut

if __name__ == '__main__':
    try:
        # create a menu to show input images
        filename = None
        if len(os.listdir(config.INPUT_DIR)) <= 1:
            raise FileNotFoundError

        print("*********** Menu ***********")
        i = 1
        switcher = {}
        for filename in os.listdir(config.INPUT_DIR)[1:]:
            if filename == '.DS_Store':
                continue
            print("{}. {}".format(i, filename))
            switcher[i] = filename
            i += 1
        choice = input("Please enter your choice for image synthesis using graph-cut:\t")
        filename = switcher.get(int(choice), 0)
        if filename == 0:
            print("Please enter valid choice")
            sys.exit(1)
        else:
            filepath = os.path.join(config.INPUT_DIR, filename)
            # read an image
            img = imageio.imread(filepath)
            scale_percent = config.RESIZE_SCALE # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            a = np.array(img,dtype=np.int32)[:,:,0:3]
            mincut = Mincut(a,config.OUT_DIM[0],config.OUT_DIM[1])
            result = mincut.patch()
            rst = result.astype('uint8')
            # write output image
            imageio.imwrite(os.path.join(config.OUTPUT_DIR,"out-{}.jpeg".format(os.path.splitext(filename)[0])),rst)
    except FileNotFoundError:
        print("Input folder empty. Please populate the folder with images")
        print(str(traceback.format_exc()))
        sys.exit(1)
