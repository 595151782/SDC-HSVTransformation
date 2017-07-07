# -*- coding: utf-8 -*-
import numpy as np
import cv2
from image import markCenterLine

if __name__ == '__main__':
    cap = cv2.VideoCapture("E:/行车记录仪/20150106_092349.MOV") # 此处替换为视频地址

    while(True):
        # frame 为返回的图像
        ret, frame = cap.read()

        # 我们对图像进行处理，标示出中央车线
        gray = markCenterLine(frame)
        # 显示出来
        cv2.imshow('frame', gray)
        # 按下 q 键停止
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()