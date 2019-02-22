# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"], 0)

retval, orig_thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('orig_thresh', orig_thresh)
cv2.waitKey()
bin_thresh = (orig_thresh == 0).astype(np.uint8)


def neighbours(x, y, image):
    """Return 8-neighbours of point p1 of picture, in clockwise order"""
    i = image
    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1
    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    n = neighbours + neighbours[0:1]    # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))


def zhangSuen(image):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P4 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P6 == 0 and   # Condition 3
                    transitions(n) == 1 and  # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing1.append((x, y))
        for x, y in changing1:
            image[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P2 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P8 == 0 and   # Condition 3
                    transitions(n) == 1 and  # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing2.append((x, y))
        for x, y in changing2:
            image[y][x] = 0
    return image * 255


if __name__ == '__main__':
    after = zhangSuen(bin_thresh)
    cv2.imshow('after', after)
    cv2.waitKey()
    cv2.destroyAllWindows()

