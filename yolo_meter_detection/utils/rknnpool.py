from queue import Queue, Full, Empty  # 导入 Full 和 Empty 异常
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed
import time  # 导入 time 模块
import os

def initRKNN(rknnModel=os.path.join(os.path.dirname(__file__), '..', '..', 'rknnModel', 'yolov8.rknn'), id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print(f"Load RKNN model {rknnModel} failed, ret={ret}")
        raise RuntimeError(f"Failed to load RKNN model: {rknnModel}")

    # 核心掩码分配逻辑，保持不变
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:  # 用于全核模式
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        print(f"警告: NPU ID {id} 未明确定义。默认分配到 NPU_CORE_0。")
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

    if ret != 0:
        print(f"Init runtime environment for {rknnModel} (core id: {id}) failed, ret={ret}")
        raise RuntimeError(f"Failed to initialize RKNN runtime for {rknnModel} on core {id}")

    print(f"RKNN runtime for {rknnModel} (core id: {id}) 初始化完成。")
    return rknn_lite


def initRKNNs(rknnModel=os.path.join(os.path.dirname(__file__), '..', '..', 'rknnModel', 'yolov5s.rknn'), TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        try:
            # YOLO 默认分配到 NPU_CORE_0 或 NPU_CORE_1
            rknn_list.append(initRKNN(rknnModel, 2))
        except RuntimeError as e:
            print(f"错误: 初始化 RKNN 实例 {i} 失败: {e}")
            for rknn_lite in rknn_list:  # 释放已初始化的资源
                rknn_lite.release()
            raise  # 重新抛出错误，让调用者处理
    return rknn_list


def initRKNN_other(rknnModel=os.path.join(os.path.dirname(__file__), '..', '..', 'rknnModel', 'yolov5s.rknn'), TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        try:
            # 明确分配给 NPU_CORE_2
            rknn_list.append(initRKNN(rknnModel, i%2))
        except RuntimeError as e:
            print(f"错误: 初始化 RKNN '其他' 实例 {i} 失败: {e}")
            for rknn_lite in rknn_list:
                rknn_lite.release()
            raise
    return rknn_list


class yolo_Executor:
    def __init__(self, rknnModel, TPEs, func, max_queue_size):
        self.TPEs = TPEs
        self.queue = Queue(maxsize=max_queue_size)  # 关键修改：有界队列
        self.rknnPool = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame, block=True, timeout=None):
        try:
            self.queue.put(self.pool.submit(
                self.func, self.rknnPool[self.num % self.TPEs], frame),
                block=block, timeout=timeout)
            self.num += 1
        except Full:
            raise  # 队列满时抛出异常

    def get(self, block=True, timeout=None):
        try:
            fut = self.queue.get(block=block, timeout=timeout)
            return fut.result(), True
        except Empty:
            return None, False
        except Exception as e:
            print(f"从 YOLO Executor 获取结果时出错: {e}")
            return None, False

    def release(self):
        self.pool.shutdown(wait=True)  # 等待所有任务完成
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
        print("YOLO Executor 资源已释放。")


class fpn_Executor:
    def __init__(self, rknnModel, TPEs, func, max_queue_size):  # 添加 max_queue_size
        self.TPEs = TPEs
        self.queue = Queue(maxsize=max_queue_size)  # 设置有界队列
        self.rknnPool = initRKNN_other(rknnModel, TPEs)  # 分配给NPU_CORE_2
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame, block=True, timeout=None):  # 添加 block 和 timeout
        try:
            self.queue.put(self.pool.submit(
                self.func, self.rknnPool[self.num % self.TPEs], frame),
                block=block, timeout=timeout)
            self.num += 1
        except Full:
            raise

    def get(self, block=True, timeout=None):  # 添加 block 和 timeout
        try:
            fut = self.queue.get(block=block, timeout=timeout)
            return fut.result(), True
        except Empty:
            return None, False
        except Exception as e:
            print(f"从 FPN Executor 获取结果时出错: {e}")
            return None, False

    def release(self):
        self.pool.shutdown(wait=True)
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
        print("FPN Executor 资源已释放。")


class crnn_Executor():
    def __init__(self, rknnModel, TPEs, func, max_queue_size):  # 添加 max_queue_size
        self.TPEs = TPEs
        self.queue = Queue(maxsize=max_queue_size)  # 设置有界队列
        self.rknnPool = initRKNN_other(rknnModel, TPEs)  # 分配给NPU_CORE_2
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame, block=True, timeout=None):  # 添加 block 和 timeout
        try:
            self.queue.put(self.pool.submit(
                self.func, self.rknnPool[self.num % self.TPEs], frame),
                block=block, timeout=timeout)
            self.num += 1
        except Full:
            raise

    def get(self, block=True, timeout=None):  # 添加 block 和 timeout
        try:
            fut = self.queue.get(block=block, timeout=timeout)
            return fut.result(), True
        except Empty:
            return None, False
        except Exception as e:
            print(f"从 CRNN Executor 获取结果时出错: {e}")
            return None, False

    def release(self):
        self.pool.shutdown(wait=True)
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
        print("CRNN Executor 资源已释放。")
