from cv2 import imshow, waitKey, resize
from rknnlite.api import RKNNLite
import cv2
import numpy as np
import os

def sigmoid(image):
    clipped_image = np.clip(image, -100.0, 100.0)
    result = 1.0 / (1.0 + np.exp(-clipped_image))
    return result

def filter(image , n=30):
    text_num, text_label = cv2.connectedComponents(image,connectivity=8)
    for i in range (1, text_num+1):
        pts = np.where(text_label == i)
        if len(pts[0]) < n:
            text_label[pts] = 0
    text_label = (text_label > 0).astype(np.uint8)
    return text_label

def order_points(pts):

    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    return np.array([tl, tr, br, bl])

def rgb_to_gray(image):
    return np.dot(image[...,:3],[0.2989, 0.5870, 0.0721]).astype(np.uint8)

def roi_transform(feature, box, size=(32, 180)):
    resize_h, resize_w = size
    x1, y1, x2, y2, x3, y3, x4, y4 = box[0]

    mapped_x1, mapped_y1 = (0, 0)
    mapped_x4, mapped_y4 = (0, resize_h)

    mapped_x2, mapped_y2 = (resize_w, 0)

    src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
    dst_pts = np.float32([
        (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
    ])
    affine_matrix = cv2.getAffineTransform(src_pts, dst_pts)
    feature_rotated_np = cv2.warpAffine(feature, affine_matrix,(resize_w,resize_h), flags=cv2.INTER_LINEAR)
    gray_scale_img = rgb_to_gray(feature_rotated_np)

    return gray_scale_img

def fpn_func(rknn_lite, image):
    image=cv2.resize(image,(512,512))
    IMG2 = np.expand_dims(image, 0)
    output = rknn_lite.inference(inputs=[IMG2])
    output_array = output[0]
    pointer = sigmoid(output_array[0, 0, :, :])
    dail = sigmoid(output_array[0, 1, :, :])
    text = sigmoid(output_array[0, 2, :, :])

    pointer_mask = np.where(pointer >0.5, 1, 0).astype(np.uint8)
    dail_mask = np.where(dail > 0.5, 1, 0).astype(np.uint8)
    text_mask = np.where(text > 0.7, 1, 0).astype(np.uint8)

    dail_label =filter(dail_mask,  n=30)
    text_label = filter(text_mask, n=30)

    text_edges = text_label * 255
    text_contours, hierarchy = cv2.findContours(text_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ref_point = []
    for i in range(len(text_contours)):
        rect = cv2.minAreaRect(text_contours[i])
        ref_point.append((int(rect[0][0]), int(rect[0][1])))

    if not ref_point:
        return None,None,None,None,None

    dail_edges = dail_label * 255
    dail_contours, _ = cv2.findContours(dail_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    std_point = []
    for i in range(len(dail_contours)):
        rect = cv2.minAreaRect(dail_contours[i])
        std_point.append((int(rect[0][0]), int(rect[0][1])))

    if len(std_point) == 0:
        return None,None,None,None,None

    if len(std_point) < 2:
        std_point.append(ref_point[0])
    else:
        if std_point[0][1] >= std_point[1][1]:
            pass
        else:
            std_point[0], std_point[1] = std_point[1], std_point[0]
    max_dis = 10000
    index = 0
    if len(text_contours)!=0:
        for i in range(len(text_contours)):
            min_rect = cv2.minAreaRect(text_contours[i])
            test_point = (min_rect[0][0], min_rect[0][1])
            dis = (test_point[0]-std_point[1][0])**2 + (test_point[1]-std_point[1][1])**2
            if dis<max_dis:
                max_dis=dis
                index=i
        rect_point = cv2.boxPoints(cv2.minAreaRect(text_contours[index]))
        bboxes = order_points(np.int0(rect_point)).reshape(1,8)
        rois = roi_transform(image, bboxes)
        return rois, pointer_mask, dail_label, text_label, std_point
    # 没有提取到数值区域返回空
    return None,None,None,None,None

# if __name__ == "__main__":
#
#     rknn_lite = initRKNN()
#
#     image = cv2.imread("10.jpg")
#
#     result = fpn_func(rknn_lite,image)
#
#     # imshow("1",result[0])
#     # waitKey(0)
#
#     rknn_lite.release()

