import cv2
import numpy as np
import os

OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.5, 0.25, 640

CLASSES = ("digital", "pointer")

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):

    n,c,h,w = position.shape
    p_num = 4
    mc = c//p_num
    y = position.reshape(n,p_num,mc,h,w)
    
    # Vectorized softmax
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # subtract max for numerical stability
    y = e_y / np.sum(e_y, axis=2, keepdims=True)
    
    acc_metrix = np.arange(mc).reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE//grid_h, IMG_SIZE//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def yolov8_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, yolo_info_for_draw=None, meter_value=0.0):
# 检查输入有效性
    if image is None:
       return

# 绘制YOLO检测框
    if yolo_info_for_draw is not None:
         boxes, scores, classes, ratio, padding = yolo_info_for_draw

         if boxes is not None and len(boxes) > 0:
# 原始类别定义
            CLASSES = ("digital", "pointer")

            for box, score, cl in zip(boxes, scores, classes):
               top, left, right, bottom = box

# 反算原始尺寸
               top = (top - padding[0])/ratio[0]
               left = (left - padding[1])/ratio[1]
               right = (right - padding[0])/ratio[0]
               bottom = (bottom - padding[1])/ratio[1]

               top, left, right, bottom = int(top), int(left), int(right), int(bottom)

# 绘制矩形和文字
               cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
               label_txt = f'{CLASSES[cl]} {score:.2f}'
               cv2.putText(
                    image,
                    label_txt,
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

# 绘制仪表值
    value_text = f"Meter Value: {meter_value:.2f}"
    cv2.putText(
        image,
        value_text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

def new_draw(image, yolo_info_for_draw=None, meter_value_to_display=None):
    """
    在图像上绘制 YOLO 检测框和仪表读数。

    Args:
        image (np.array): 要绘制的图像。
        yolo_info_for_draw (tuple/None): YOLO 检测结果 (boxes, scores, classes, ratio, padding)。
                                         如果为 None，则不绘制YOLO框。
        meter_value_to_display (float/None): 仪表读数值。如果为 None，则不绘制读数。
    """

    # --- 绘制 YOLO 框 ---
    # 只有当 yolo_info_for_draw 有效且包含检测框时才绘制
    if yolo_info_for_draw is not None and len(yolo_info_for_draw) > 0:
        boxes = yolo_info_for_draw[0]  # 假设 boxes 是 yolo_info_for_draw 的第一个元素

        # 检查 boxes 是否不是 None 且不为空
        if boxes is not None and len(boxes) > 0:
            # 假设 yolo_info_for_draw 可能还包含 scores, classes, ratio, padding
            # 根据你的实际 yolo_func 返回结构调整
            # 例如: boxes, scores, classes, ratio, padding = yolo_info_for_draw

            for box in boxes:
                # box 格式可能是 [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 可选：绘制类别和置信度，这需要 yolo_info_for_draw 包含这些信息
                # text = f"Class: {classes[i]}, Score: {scores[i]:.2f}"
                # cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- 绘制仪表读数 ---
    # 只有当 meter_value_to_display 不是 None 时才绘制
        if meter_value_to_display is not None:
            text = f"Value: {meter_value_to_display:.2f}"
            # 选择一个合适的位置绘制，例如图像的左上角，避免与YOLO框重叠
            cv2.putText(image, text, (50, 50),  # 调整位置以适合你的显示
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)



def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border

    return im, ratio, (left, top)

def crop_image(image, boxes):
    crop_roi = []
    for point in boxes:
        xmin,ymin,xmax,ymax = map(int, point)
        if xmax > xmin and ymax > ymin:
            object_roi = image[ymin:ymax, xmin:xmax]
            crop_roi.append(object_roi)
    return crop_roi

def yolo_func(rknn_lite, image):

    pre_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pre_image, ratio, padding = letterbox(pre_image)

    IMG2 = np.expand_dims(pre_image, 0)
    
    outputs = rknn_lite.inference(inputs=[IMG2],data_format=['nhwc'])

    boxes, classes, scores = yolov8_post_process(outputs)

    if boxes is not None:
        crop_roi = crop_image(pre_image, boxes)
        if crop_roi:
            return  crop_roi[0], (boxes, scores, classes, ratio, padding)
        else:
            return None, (boxes, scores, classes, ratio, padding)
    return None, (None, None, None, None, None)

# if __name__ == "__main__":
#     image = cv2.imread("1.jpg")
#     rknn_lite = initRKNN()
#     _,crop_roi = myFunc(rknn_lite, image)
#     crop_roi = cv2.cvtColor(crop_roi[0], cv2.COLOR_BGR2RGB)
#     imshow("1",crop_roi)
#     waitKey(0)
#     rknn_lite.release()



