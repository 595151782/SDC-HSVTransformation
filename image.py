# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 函数 HSV 提取：
# 用于在画面中寻找中央实线（黄色）

def extractHsvComponent(image, Max, Min):
    # 图像转为 HSV 空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 处理上限，下限
    lower, upper = np.array(Min), np.array(Max)
    # 构建 AND 矩阵
    mask = cv2.inRange(hsv, lower, upper)
    # 进行 AND 运算（符合的保留，不符合的变黑）
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


def solidLineDetection(orig, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines != None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 500 * (-b))
            y1 = int(y0 + 500 * (a))
            x2 = int(x0 - 500 * (-b))
            y2 = int(y0 - 500 * (a))
            cv2.line(orig, (x1, y1), (x2, y2), (0, 0, 255), 15)
    return orig



def markCenterLine(img):
    hsv = extractHsvComponent(img, [25, 255, 255], [20, 80, 80])
    lin = solidLineDetection(img, hsv)
    return lin


def do_nothing(image):
    pass

# 创建黑图像
if __name__ == '__main__':
    img = cv2.imread("2017-07-05 (1).png")
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 960, 540)
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image2', 960, 540)

    # 创建滑块,注册回调函数
    # cv2.createTrackbar('MaxH', 'image', 0, 180, do_nothing)
    # cv2.createTrackbar('MinH', 'image', 0, 180, do_nothing)
    # cv2.createTrackbar('MaxS', 'image', 0, 255, do_nothing)
    # cv2.createTrackbar('MinS', 'image', 0, 255, do_nothing)
    # cv2.createTrackbar('MaxV', 'image', 0, 255, do_nothing)
    # cv2.createTrackbar('MinV', 'image', 0, 255, do_nothing)
    cv2.createTrackbar('MinL', 'image', 0, 1000, do_nothing)
    cv2.createTrackbar('MaxG', 'image', 0, 1000, do_nothing)

    while(1):
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 获得滑块的位置
        # Minimum = [cv2.getTrackbarPos('MinH', 'image'),
        #            cv2.getTrackbarPos('MinS', 'image'),
        #            cv2.getTrackbarPos('MinV', 'image')]
        # Maximum = [cv2.getTrackbarPos('MaxH', 'image'),
        #            cv2.getTrackbarPos('MaxS', 'image'),
        #            cv2.getTrackbarPos('MaxV', 'image')]
        minLen, maxGap = cv2.getTrackbarPos('MinL', 'image'), cv2.getTrackbarPos('MaxG', 'image')
        # 设置图像颜色
        # res = extractHsvComponent(img, Maximum, Minimum)
        # lin = dashedLineDetection(res, img)
        # lin = dashedLineDetection(img, img, minLen, maxGap)
        lin = markCenterLine(img, minLen, maxGap)
        cv2.imshow('image2', lin)

    cv2.imwrite(
        "2017-07-05_{0}_{1}.png".format(str(Minimum), str(Maximum)), res)
    cv2.imwrite(
        "2017-07-05_{0}_{1}_lin.png".format(str(Minimum), str(Maximum)), lin)
    cv2.destroyAllWindows()
