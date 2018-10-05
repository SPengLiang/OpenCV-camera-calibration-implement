#!usr/bin/env/ python
# _*_ coding:utf-8 _*_
import cv2 as cv
import numpy as np
import os

if __name__ == "__main__":
    file_dir = r'..\pic'
    #标定所用图像
    pic_name = os.listdir(file_dir)

    #由于棋盘为二维平面，设定世界坐标系在棋盘上，一个单位代表一个棋盘宽度，产生世界坐标系三维坐标
    real_coor = np.zeros((9*6, 3), np.float32)
    real_coor[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1, 2)

    real_points = []
    pic_points = []

    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv.imread(pic_path)

        #寻找到棋盘角点
        succ, pic_coor = cv.findChessboardCorners(pic_data, (9, 6), None)

        if succ:
            #添加每幅图的对应3D-2D坐标
            pic_cor = pic_coor.reshape(-1, 2)
            pic_points.append(pic_coor)

            real_points.append(real_coor)

    #cameraMatrix为相机内参，distCoeffs为畸变矫正参数，rvecs为旋转向量，tvecs为平移向量
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(real_points, pic_points, (480, 640), None, None)


    #下面为矫正步骤
    org = cv.imread(r'..\pic\left02.jpg')

    h, w = org.shape[:2]
    newcameramtx, roi=cv.getOptimalNewCameraMatrix(cameraMatrix,distCoeffs, (w,h), 1,(w,h))

    #显示较正前图像
    cv.imshow('org', org)
    cv.waitKey(0)

    dst = cv.undistort(org, cameraMatrix, distCoeffs, None)

    #显示较正后图像
    cv.imshow('undistort', dst)
    cv.waitKey(0)

    #矫正图像保存路径
    cv.imwrite(r'..\undistort\undistort.jpg', dst)