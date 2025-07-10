import cv2
from skimage import morphology
import numpy as np
import collections
import os

# 全局变量，用于存储历史读数，实现平滑
# 使用 deque 实现固定长度的滑动窗口
_readings_history = collections.deque(maxlen=10)  # 存储最近 10 个读数，可以根据需要调整


def get_distance_point2line(point, line):
    line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


def judge(p1, p2, p3):
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = p2[0] * p1[1] - p1[0] * p2[1]
    value = A * p3[0] + B * p3[1] + C
    return value


def angle(v1, v2):
    lx = np.sqrt(v1.dot(v1))
    ly = np.sqrt(v2.dot(v2))
    cos_angle = v1.dot(v2) / (lx * ly)
    # 钳位 cos_angle 在 [-1, 1] 之间，防止浮点误差导致 arccos 报错
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2


def find_lines(fpn_result, value_data):
    # 如果 fpn_result 或 value_data 为 None，直接返回 None，不进行计算
    if fpn_result is None or value_data is None:
        _readings_history.clear()  # 清空历史，避免None污染后续平均
        return None

    # 解包 fpn_result。根据你的注释，fpn_result 可能是一个包含多个元素的元组
    # 假设 fpn_result 至少包含 pointer_mask 和 std_point
    try:
        _, pointer_mask, dail_label, text_label, std_point = fpn_result
    except ValueError as e:
        print(f"Error unpacking fpn_result: {e}. fpn_result was: {fpn_result}")
        _readings_history.clear()
        return None

    # 检查 pointer_mask 是否有效
    if pointer_mask is None or pointer_mask.size == 0:
        _readings_history.clear()
        return None

    pointer_edge = morphology.skeletonize(pointer_mask)
    pointer_edge = pointer_edge * 255
    pointer_edge = pointer_edge.astype(np.uint8)

    pointer_lines = cv2.HoughLinesP(pointer_edge, 1, np.pi / 180, 10,
                                    np.array([]), minLineLength=10, maxLineGap=400)
    coin1, coin2 = None, None
    try:
        # 确保 pointer_lines 不为空且有元素
        if pointer_lines is not None and len(pointer_lines) > 0:
            for x1, y1, x2, y2 in pointer_lines[0]:  # 假设只需要第一条线
                coin1 = (x1, y1)
                coin2 = (x2, y2)
        else:
            _readings_history.clear()  # 没有检测到指针线
            return None
    except TypeError:  # 如果 pointer_lines 是 None
        _readings_history.clear()
        return None

    h, w = pointer_mask.shape  # 使用 pointer_mask 的尺寸
    center = (0.5 * w, 0.5 * h)
    dis1 = (coin1[0] - center[0]) ** 2 + (coin1[1] - center[1]) ** 2
    dis2 = (coin2[0] - center[0]) ** 2 + (coin2[1] - center[1]) ** 2

    # 保证pointer_line的第一个点是指针靠近中心的那个点
    if dis1 <= dis2:
        pointer_line = (coin1, coin2)
    else:
        pointer_line = (coin2, coin1)

    # 检查 std_point 是否有效
    if std_point is None or len(std_point) < 2 or std_point[0] is None or std_point[1] is None:
        _readings_history.clear()
        return None

    a1 = std_point[0]  # 0点
    a2 = std_point[1]  # 分度值点

    one = np.array([pointer_line[0][0], pointer_line[0][1]])  # 指针根部
    vec_to_a1 = np.array(a1) - one  # 指针根部到0点向量
    vec_to_a2 = np.array(a2) - one  # 指针根部到分度值点向量
    vec_pointer = np.array(pointer_line[1]) - one  # 指针向量

    # 计算角度
    std_ang = angle(vec_to_a1, vec_to_a2)  # 0点到分度值点的角度
    now_ang = angle(vec_to_a1, vec_pointer)  # 0点到指针的角度

    flag = judge(pointer_line[0], a1, pointer_line[1])  # 判断指针是否在0点线的左侧或右侧

    if flag > 0:  # 如果指针在0点线的"左侧"（根据你的judge函数逻辑）
        now_ang = 360 - now_ang  # 修正角度，通常用于顺时针计量

    # 避免除以零
    if std_ang == 0:
        _readings_history.clear()
        return None

    # 计算原始读数
    raw_value = (value_data / std_ang) * now_ang  # value_data 应该是总刻度值

    distance = get_distance_point2line([a1[0], a1[1]],
                                       [pointer_line[0][0], pointer_line[0][1], pointer_line[1][0], pointer_line[1][1]])

    if flag > 0 and distance < 40:  # 如果指针在0点附近且判断为负向，且距离很近，认为是0
        raw_value = 0.00
    else:
        raw_value = round(raw_value, 3)

    # 将当前有效读数添加到历史记录中
    if raw_value is not None:
        _readings_history.append(raw_value)

    # 如果历史记录中有数据，返回平均值；否则返回 None
    if _readings_history:
        smoothed_value = sum(_readings_history) / len(_readings_history)
        return smoothed_value
    else:
        return None
