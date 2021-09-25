import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i1", "--image1", required = True, help = "Path to the image1")
    ap.add_argument("-i2", "--image2", required = True, help = "Path to the image2")
    args = vars(ap.parse_args())

    image1 = cv2.imread(args["image1"])
    image2 = cv2.imread(args["image2"])
    i1h, i1w = image1.shape[0], image1.shape[1]
    i2h, i2w = image2.shape[0], image2.shape[1]
    image1 = cv2.resize(image1, (int(i1w * 0.1), int(i1h * 0.1)), interpolation = cv2.INTER_AREA)
    image2 = cv2.resize(image2, (int(i2w * 0.1), int(i2h * 0.1)), interpolation = cv2.INTER_AREA)
    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)
    cv2.waitKey(0)

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image 1 Gray", image1_gray)
    cv2.imshow("Image 2 Gray", image2_gray)
    cv2.waitKey(0)

    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

    plt.figure()
    plt.title("Image1 Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist1)
    plt.xlim([0, 256])

    plt.figure()
    plt.title("Image2 Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist2)
    plt.xlim([0, 256])

    plt.show()

    
