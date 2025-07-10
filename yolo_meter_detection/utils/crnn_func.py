from rknnlite.api import RKNNLite
import cv2
import numpy as np
import unicodedata as ud
import os
import sys

class CRNNProcessor:
    def __init__(self, package_share_dir):
# 延迟初始化，等到实际使用时才加载
          self.keys = None
          self.modelPath = None
          self.package_share_dir = package_share_dir
          
          self.init()

    def init(self):
# 设置路径
          self.modelPath = os.path.join(self.package_share_dir, 'rknnModel', 'CRNN.rknn')
          codec_path = os.path.join(self.package_share_dir, 'codec.txt')

# 读取codec
          if not os.path.exists(codec_path):
               raise FileNotFoundError(f"codec.txt文件未找到: {codec_path}")

          with open(codec_path, 'r') as f:
               self.keys = f.readline().strip()
               
          self.converter = CTCLabelConverter(self.keys)

# 验证模型
          if not os.path.exists(self.modelPath):
               raise FileNotFoundError(f"模型文件未找到: {self.modelPath}")
               
          self.initRKNN()

    def initRKNN(self, id=2):
          if not self.keys or not self.modelPath:
              raise RuntimeError("CRNNFunction 未初始化")

          self.rknn_lite = RKNNLite()
          ret = self.rknn_lite.load_rknn(self.modelPath)
# 其他初始化代码...
# 其他方法保持不变...
# 创建实例，延迟初始化

#def initRKNN(rknnModel=modelPath, id=2):
#    rknn_lite = RKNNLite()
#    ret = rknn_lite.load_rknn(rknnModel)
          if ret != 0:
               print("Load RKNN rknnModel failed")
               exit(ret)
          if id == 0:
               ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
          elif id == 1:
               ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
          elif id == 2:
               ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
          elif id == -1:
               ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
          else:
               ret = self.rknn_lite.init_runtime()
          if ret != 0:
               print("Init runtime environment failed")
               exit(ret)
          #print(rknnModel, "\t\tdone")
          return self.rknn_lite
    def preprocess(self, image):
         """预处理图像"""
         image = cv2.resize(image, (180, 32))
         image = np.expand_dims(image, axis=-1) # 添加通道维度
         return np.expand_dims(image, 0) # 添加批次维度

    def infer(self, image):
         """执行推理"""
         processed = self.preprocess(image)
         output = self.rknn_lite.inference(inputs=[processed])

         if output[0] is None or output[0].size == 0:
             return None

         pred = output[0].argmax(axis=2).flatten()
         return self.converter.decode(pred, pred.shape[0]) 

class CTCLabelConverter:
    def __init__(self, alphabet):
        self.alphabet =alphabet + '-'
        self.dict={}
        for i,char in enumerate(alphabet):
            self.dict[char]=i+1

    def decode(self, t_np, length_val, raw=False):
        value = None
        assert t_np.size == length_val, \
            "text with length: {} does not match declared length: {}".format(t_np.size, length_val)

        if raw:
            # 原始解码: 直接映射ID到字符
            output = ''.join([self.alphabet[i - 1] for i in t_np])
            # 处理阿拉伯语文本（如果需要）
            if len(output) > 0 and 'ARABIC' in ud.name(output[0]):
                output = output[::-1]
            return output
        else:
            char_list = []
            for i in range(length_val):  # 遍历序列
                if t_np[i] != 0 and (not (i > 0 and t_np[i - 1] == t_np[i])):
                    char_list.append(self.alphabet[t_np[i] - 1])

            output = ''.join(char_list)
            # 处理阿拉伯语文本（如果需要）
            if len(output) > 0 and 'ARABIC' in ud.name(output[0]):
                output = output[::-1]
        if output and output[0] != "":
                value = float(output)
        return value

def crnn_func(rknn_lite,image):
     raise NotImplementedError("此函数不再使用")


crnn_processor = None
# if __name__ == "__main__":
#
#     # image = cv2.imread("80.jpg",cv2.IMREAD_GRAYSCALE)
#     # cv2.imshow("test", image)
#     # cv2.waitKey(0)
#
#     rknn_lite.release()














