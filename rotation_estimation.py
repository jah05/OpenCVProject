import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math

def tilt_angle(i1, i2):
    edges1 = cv2.Canny(i1, 30, 150)
    (contours1, _) = cv2.findContours(edges1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges2 = cv2.Canny(i2, 30, 150)
    (contours2, _) = cv2.findContours(edges2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    d1 = cv2.cvtColor(i1.copy(), cv2.COLOR_GRAY2BGR)
    d2 = cv2.cvtColor(i2.copy(), cv2.COLOR_GRAY2BGR)

    if len(contours2) != 0 and len(contours1) != 0:
        cv2.drawContours(d1, contours1[0], 1, (0, 0, 255), 2)
        cv2.imshow("Contours 1", d1)

        cv2.drawContours(d2, contours2[0], 1, (0, 0, 255), 2)
        cv2.imshow("Contours 2", d2)
        cv2.waitKey(0)

        (x1tl, y1tl, w1, h1) = cv2.boundingRect(contours1[0])
        (x2tl, y2tl, w2, h2) = cv2.boundingRect(contours2[0])

        x1br = x1tl + w1
        y1br = y1tl + h1

        x2br = x2tl + w2
        y2br = y2tl + h2

        m1 = float((y1tl - y1br)) / (x1tl - x1br)
        m2 = float((y2tl - y2br)) / (x2tl - x2br)

        angle1 = math.atan(m1)
        angle2 = math.atan(m2)

        angle = math.degrees(abs(angle1 - angle2))
        cv2.rectangle(d1, (x1tl, y1tl), (x1br, y1br), (0,0,255), 0)
        cv2.rectangle(d2, (x2tl, y2tl), (x2br, y2br), (0,0,255), 0)
        cv2.imshow("Bounding 1", d1)
        cv2.imshow("Bounding 2", d2)
        cv2.waitKey(0)

    else:
        angle = -1

    return angle

def threshold_based(i1, i2):
    hist1 = cv2.calcHist([image1_gray], [0], None, [32], [0, 256])
    hist2 = cv2.calcHist([image2_gray], [0], None, [32], [0, 256])

    peak1 = (0, hist1[0]) # index, value
    peak2 = (0, hist1[0])
    for i in range(1, hist1.shape[0]-1):
        if hist1[i-1] <= hist1[i] and hist1[i] >= hist1[i+1]:
            if hist1[i] > peak1[1]:
                peak2 = peak1
                peak1 = (i, hist1[i])
            elif hist1[i] > peak2[1]:
                peak2 = (i, hist1[i])

    plt.figure()
    plt.title("Image1 Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist1)
    plt.plot([peak1[0]], [peak1[1]], marker="o")
    plt.plot([peak2[0]], [peak2[1]], marker="o")

    plt.figure()
    plt.title("Image2 Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot([peak1[0]], [hist2[peak1[0]]], marker="o")
    plt.plot([peak2[0]], [hist2[peak2[0]]], marker="o")
    plt.plot(hist2)

    plt.show()

    (T, thresh1) = cv2.threshold(i1, int((peak1[0] + peak2[0]) * 8 // 2), 255, cv2.THRESH_BINARY_INV)
    (T, thresh2) = cv2.threshold(i2, int((peak1[0] + peak2[0]) * 8 // 2), 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Threshold 1", thresh1)
    cv2.imshow("Threshold 2", thresh2)
    cv2.waitKey(0)

    mask = np.zeros((thresh1.shape[0], thresh1.shape[1]), dtype="uint8")
    cv2.rectangle(mask, (0, 0), (thresh1.shape[1], thresh1.shape[0]//2), (255, 255, 255), -1)
    cv2.imshow("Mask", mask)

    masked1 = cv2.bitwise_and(thresh1, thresh1, mask=mask)
    masked2 = cv2.bitwise_and(thresh2, thresh2, mask=mask)
    cv2.imshow("Masked 1", masked1)
    cv2.imshow("Masked 2", masked2)
    cv2.waitKey(0)

    sum1 = 0
    for row in masked1:
        for col in row:
            sum1 += col

    sum2 = 0
    for row in masked2:
        for col in row:
            sum2 += col


    if sum2 > sum1:
        return "camera down"
    else:
        return "camera up"

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

    image1_gray = cv2.medianBlur(image1_gray, 15)
    image2_gray = cv2.medianBlur(image2_gray, 15)

    cv2.imshow("Image 1 Gray", image1_gray)
    cv2.imshow("Image 2 Gray", image2_gray)
    cv2.waitKey(0)

    angle = tilt_angle(image1_gray, image2_gray)
    if angle < 0.5:
        result = threshold_based(image1_gray, image2_gray)
        print(result)
    else:
        print(str(angle) + " degrees")
