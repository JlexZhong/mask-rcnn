import sys
import os
import cv2
import numpy as np
import random
import time
import json
import base64
from pathlib import Path

from math import cos ,sin ,pi,fabs,radians

# https://blog.csdn.net/qq_34510308/article/details/104280171?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163136938916780265449572%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=163136938916780265449572&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-4-104280171.pc_search_result_hbase_insert&utm_term=labelme%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA&spm=1018.2226.3001.4187

#读取json
def readJson(jsonfile):
    with open(jsonfile,encoding='utf-8') as f:
        jsonData = json.load(f)
    return jsonData

def rotate_bound(image, angle):
    """
    旋转图像
    :param image: 图像
    :param angle: 角度
    :return: 旋转后的图像
    """
    h, w,_ = image.shape
    # print(image.shape)
    (cX, cY) = (w // 2, h // 2)
    # print(cX,cY)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # print(nW,nH)
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # print( M[0, 2], M[1, 2])
    image_rotate = cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))
    return image_rotate,cX,cY,angle


def dumpRotateImage(img, degree):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    # print(width // 2,height // 2)
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return imgRotation,matRotation


def rotate_xy(x, y, angle, cx, cy):
    """
    点(x,y) 绕(cx,cy)点旋转
    """
    # print(cx,cy)
    angle = angle * pi / 180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new


#转base64
def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

#坐标旋转
def rotatePoint(Srcimg_rotate,jsonTemp,M,imagePath):
    json_dict = {}
    for key, value in jsonTemp.items():
        if key=='imageHeight':
            json_dict[key]=Srcimg_rotate.shape[0]
            # print('gao',json_dict[key])
        elif key=='imageWidth':
            json_dict[key] = Srcimg_rotate.shape[1]
            # print('kuai',json_dict[key])
        elif key=='imageData':
            json_dict[key] = image_to_base64(Srcimg_rotate)
        elif key=='imagePath':
            json_dict[key] = imagePath
        else:
            json_dict[key] = value
    for item in json_dict['shapes']:
        for key, value in item.items():
            if key == 'points':
                for item2 in range(len(value)):
                    pt1=np.dot(M,np.array([[value[item2][0]],[value[item2][1]],[1]]))
                    value[item2][0], value[item2][1] = pt1[0][0], pt1[1][0]
    return json_dict

#保存json
def writeToJson(filePath,data):
    fb = open(filePath,'w')
    fb.write(json.dumps(data,indent=2)) # ,encoding='utf-8'
    fb.close()

if __name__=='__main__':
    
    # 源标注数据路径
    before_path = './before_mineral/'
    result_path = './mineral_dataset_by_datagen/'
    
    image_list = []
    json_list = []

    rotation_angle = [90,180,270,360]
    
    for file in os.listdir(before_path):
        if file.endswith('.jpg'):
            image_list.append(file)
        if file.endswith('.json'):
            json_list.append(file)
    for i in range(len(image_list)):
        
        Srcimg = cv2.imread(before_path + image_list[i])  #  gai label1,label_name
        jsonData = readJson(before_path + json_list[i])  #  读取json
        for angle in rotation_angle:
            Srcimg_rotate,M = dumpRotateImage(Srcimg, angle)  #  逆时针旋转90、180、270、360度
            jsonData2=rotatePoint(Srcimg_rotate,jsonData,M,image_list[i])  # 坐标也要旋转
            
            cv2.imwrite(result_path + image_list[i],Srcimg_rotate)
            writeToJson(result_path + json_list[i], jsonData2)
        print('completed:' + str(i))
    
    print('Successfully!')
