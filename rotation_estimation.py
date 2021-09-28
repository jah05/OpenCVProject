import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math

def tilt_angle_contour(b1, b2, i1, i2):
    (x1tl, y1tl, w1, h1) = cv2.boundingRect(b1)
    (x2tl, y2tl, w2, h2) = cv2.boundingRect(b2)

    x1br = x1tl + w1
    y1br = y1tl + h1

    x2br = x2tl + w2
    y2br = y2tl + h2

    m1 = float((y1tl - y1br)) / (x1tl - x1br)
    m2 = float((y2tl - y2br)) / (x2tl - x2br)

    angle1 = math.atan(m1)
    angle2 = math.atan(m2)

    angle = math.degrees(abs(angle1 - angle2))
    cv2.rectangle(i1, (x1tl, y1tl), (x1br, y1br), (0,0,255), 0)
    cv2.rectangle(i2, (x2tl, y2tl), (x2br, y2br), (0,0,255), 0)
    cv2.imshow("Image 1", i1)
    cv2.imshow("Image 2", i2)

    print(angle)
    return angles

def threshold_based(i1, i2):
    hist1 = cv2.calcHist([image1_gray], [0], None, [8], [0, 256])
    hist2 = cv2.calcHist([image2_gray], [0], None, [8], [0, 256])

    plt.figure()
    plt.title("Image1 Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist1)

    plt.figure()
    plt.title("Image2 Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist2)

    plt.show()
    


def contour_based(i1, i2):
    edges1 = cv2.Canny(i1, 30, 150)
    (contours1, _) = cv2.findContours(edges1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges2 = cv2.Canny(i2, 30, 150)
    (contours2, _) = cv2.findContours(edges2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    d1 = cv2.cvtColor(i1.copy(), cv2.COLOR_GRAY2BGR)
    d2 = cv2.cvtColor(i2.copy(), cv2.COLOR_GRAY2BGR)

    cv2.drawContours(d1, contours1[1], 1, (0, 0, 255), 2)
    cv2.imshow("Contours1", d1)

    cv2.drawContours(d2, contours2[1], 1, (0, 0, 255), 2)
    cv2.imshow("Contours2", d2)
    cv2.waitKey(0)

    tilt_angle = tilt_angle_contour(border1, border2, d1, d2)


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
    # cv2.imshow("Image 1", image1)
    # cv2.imshow("Image 2", image2)
    # cv2.waitKey(0)

    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # image1_gray = cv2.GaussianBlur(image1_gray, (5, 5), 0)
    # image2_gray = cv2.GaussianBlur(image2_gray, (5, 5), 0)
    image1_gray = cv2.medianBlur(image1_gray, 15)
    image2_gray = cv2.medianBlur(image2_gray, 15)

    cv2.imshow("Image 1 Gray", image1_gray)
    cv2.imshow("Image 2 Gray", image2_gray)
    cv2.waitKey(0)



    threshold_based(image1_gray, image2_gray)
    # contour_based(image1_gray, image2_gray)
    cv2.waitKey(0)
