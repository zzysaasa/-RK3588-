import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
import threading
import queue
import os
import yaml

from std_msgs.msg import Float32
from rcl_interfaces.msg import ParameterDescriptor
from ament_index_python.packages import get_package_share_directory
from yolo_meter_detection.utils.rknnpool import yolo_Executor, fpn_Executor, crnn_Executor
from yolo_meter_detection.utils.yolo_func import yolo_func
from yolo_meter_detection.utils.fpn_func import fpn_func
from yolo_meter_detection.utils.crnn_func import CRNNProcessor, crnn_processor
from yolo_meter_detection.utils.meter import find_lines

class MeterDetectionNode(Node):
       def __init__(self):
            super().__init__('meter_detection_node')

# 声明参数
            self.declare_parameter('camera_topic', '/camera/image_raw', ParameterDescriptor(description='Input camera topic'))
            self.declare_parameter('video_device', '/dev/video21', ParameterDescriptor(description='Video device path'))
            self.declare_parameter('output_image_topic', '/meter_detection/image', ParameterDescriptor(description='Output image topic'))
            self.declare_parameter('value_topic', '/meter_detection/value', ParameterDescriptor(description='Value output topic'))
            self.declare_parameter('model_path', 'rknnModel', ParameterDescriptor(description='Path to RKNN models'))

# 获取参数值
            self.camera_topic = self.get_parameter('camera_topic').value
            self.video_device = self.get_parameter('video_device').value
            self.output_image_topic = self.get_parameter('output_image_topic').value
            self.value_topic = self.get_parameter('value_topic').value
            model_path = self.get_parameter('model_path').value

# 确保使用绝对路径
            self.base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(self.base_path, os.pardir, os.pardir, model_path)
            self.get_logger().info(f"Model path: {model_path}")

# 创建ROS2发布者和订阅者
            self.bridge = CvBridge()
            self.subscription = self.create_subscription(Image,self.camera_topic,self.image_callback,10)
            self.detection_pub = self.create_publisher(Image, self.output_image_topic, 10)
            self.value_pub = self.create_publisher(Float32, self.value_topic, 10)

# 初始化状态
            self.running = True
            self.latest_original_frame = None
            self.latest_original_frame_lock = threading.Lock()
            
            self.crnn_processor = None

# 初始化队列
            self.yolo_input_queue = queue.Queue(maxsize=5)
            self.fpn_input_queue = queue.Queue(maxsize=5)
            self.crnn_input_queue = queue.Queue(maxsize=5)
            self.display_yolo_output_queue = queue.Queue(maxsize=1)

# 初始化模型
            self.init_models(model_path)

# 启动工作线程
            self.yolo_thread = threading.Thread(target=self.yolo_worker)
            self.fpn_thread = threading.Thread(target=self.fpn_worker)
            self.crnn_thread = threading.Thread(target=self.crnn_worker)

            self.yolo_thread.start()
            self.fpn_thread.start()
            self.crnn_thread.start()

            self.get_logger().info("YOLO检测节点已启动")

       def init_models(self, model_path_param):
       
             package_share_dir = get_package_share_directory('yolo_meter_detection')
             
             try:
# 初始化功能模块
                self.crnn_processor = CRNNProcessor(package_share_dir)
             except Exception as e:
                self.get_logger().error(f"初始化CRNN模块失败: {str(e)}")
                self.running = False
  
# 设置模型路径
             model_path = os.path.join(package_share_dir, 'rknnModel')
             codec_path = os.path.join(package_share_dir, 'codec.txt')

# 记录路径
             self.get_logger().info(f"使用共享目录: {package_share_dir}")
             self.get_logger().info(f"模型路径: {model_path}")
             self.get_logger().info(f"Codec 路径: {codec_path}")

# 确保路径存在
             if not os.path.exists(model_path):
                 self.get_logger().error(f"模型路径不存在: {model_path}")
                 self.running = False
                 return

             if not os.path.exists(codec_path):
                 self.get_logger().error(f"codec.txt 文件未找到: {codec_path}")
                 self.running = False
                 return

# 读取codec文件
             try:
                with open(codec_path, 'r') as f:
                    self.keys = f.readlines()[0]
             except Exception as e:
                self.get_logger().error(f"读取codec.txt失败: {str(e)}")
                self.running = False
                return

# 构建模型路径
             yolo_model = os.path.join(model_path, "yolov8.rknn")
             fpn_model = os.path.join(model_path, "FPN.rknn")
             crnn_model = os.path.join(model_path, "CRNN.rknn")


# 初始化执行器
             self.get_logger().info("初始化模型...")

             try:
# 根据硬件能力配置线程数
                  self.yolo_TPEs = 3
                  self.fpn_TPEs = 2
                  self.crnn_TPEs = 2
                  crnn_wrapped = self.crnn_wrapper

                  self.yolo_pool = yolo_Executor(rknnModel=yolo_model, TPEs=self.yolo_TPEs, func=yolo_func,max_queue_size=self.yolo_TPEs)

                  self.fpn_pool = fpn_Executor(rknnModel=fpn_model,TPEs=self.fpn_TPEs,func=fpn_func,max_queue_size=self.fpn_TPEs)

                  self.crnn_pool = crnn_Executor(rknnModel=crnn_model,TPEs=self.crnn_TPEs,func=crnn_wrapped,max_queue_size=self.crnn_TPEs)
             except Exception as e:
                  self.get_logger().error(f"模型初始化失败: {str(e)}")
                  self.running = False
                  raise

             self.get_logger().info("模型初始化完成")
       
       def crnn_wrapper(self, rknn_lite, image):
          return self.crnn_processor.infer(image)

       def image_callback(self, msg):
         """处理传入的图像消息"""
         try:
# 转换ROS图像消息为OpenCV格式
           cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

# 更新供工作线程使用的最新帧
           with self.latest_original_frame_lock:
              self.latest_original_frame = cv_image.copy()

# 将帧推送到YOLO处理队列
           try:
               self.yolo_input_queue.put((time.time(), cv_image.copy()), block=False)
           except queue.Full:
               self.get_logger().debug("YOLO输入队列已满，丢弃帧", throttle_duration_sec=1.0)
         except Exception as e:
           self.get_logger().error(f"图像回调错误: {str(e)}")

       def yolo_worker(self):
           """YOLO处理线程"""
           self.get_logger().info("YOLO工作线程启动")
           while self.running and rclpy.ok():
               try:
# 从队列获取图像
                    frame_time, original_frame = self.yolo_input_queue.get(timeout=0.01)

# 处理图像
                    yolo_result = None
                    try:
                         self.yolo_pool.put(original_frame, block=False)
                         yolo_result, _ = self.yolo_pool.get(timeout=0.01)
                    except (queue.Full, queue.Empty):
                         yolo_result = None

# 处理检测结果
                    cropped_image, yolo_info = yolo_result if yolo_result else (None, None)
                    boxes_detected = yolo_info and len(yolo_info[0]) > 0

                    if boxes_detected and cropped_image is not None:
# 将裁剪后的图像传递给FPN
                       try:
                          self.fpn_input_queue.put((frame_time, cropped_image, yolo_info), block=False)
                       except queue.Full:
                          pass

# 更新显示队列
                    try:
                       while not self.display_yolo_output_queue.empty():
                          self.display_yolo_output_queue.get_nowait()
                       self.display_yolo_output_queue.put((frame_time, yolo_info))
                    except queue.Full:
                       pass

               except queue.Empty:
                 time.sleep(0.001)
               except Exception as e:
                 self.get_logger().error(f"YOLO处理错误: {str(e)}")
                 self.running = False

       def fpn_worker(self):
          """FPN处理线程"""
          self.get_logger().info("FPN工作线程启动")
          while self.running and rclpy.ok():
              try:
                frame_time, cropped_image, yolo_info = self.fpn_input_queue.get(timeout=0.01)

# 处理图像
                fpn_result = None
                try:
                   self.fpn_pool.put(cropped_image, block=False)
                   fpn_result, _ = self.fpn_pool.get(timeout=0.01)
                except (queue.Full, queue.Empty):
                   fpn_result = None

                if fpn_result:
                   fpn_image, *_ = fpn_result
# 传递到CRNN处理
                   try:
                       self.crnn_input_queue.put((frame_time, fpn_image, fpn_result, yolo_info), block=False)
                   except queue.Full:
                       pass
              except queue.Empty:
                   time.sleep(0.001)
              except Exception as e:
                   self.get_logger().error(f"FPN处理错误: {str(e)}")
                   self.running = False

       def crnn_worker(self):
           """CRNN处理线程"""
           self.get_logger().info("CRNN工作线程启动")
           while self.running and rclpy.ok():
                try:
                     frame_time, crnn_image, fpn_result, yolo_info = self.crnn_input_queue.get(timeout=0.01)

# 处理图像
                     value_data = None
                     try:
                        self.crnn_pool.put(crnn_image, block=False)
                        value_data, _ = self.crnn_pool.get(timeout=0.01)
                     except (queue.Full, queue.Empty):
                        value_data = None

# 识别仪表值
                     final_value = find_lines(fpn_result, value_data) if value_data else None

                     if final_value is not None:
# 发布识别值
                          value_msg = Float32()
                          value_msg.data = final_value
                          self.value_pub.publish(value_msg)

# 处理可视化
                          with self.latest_original_frame_lock:
                            original_frame = self.latest_original_frame.copy() if self.latest_original_frame else None

                          if original_frame is not None:
# 尝试获取YOLO信息
                               try:
                                  _, yolo_info_for_draw = self.display_yolo_output_queue.get_nowait()
                               except queue.Empty:
                                  yolo_info_for_draw = None

# 在图像上绘制结果
                               self.draw_and_publish(original_frame, yolo_info_for_draw, final_value, frame_time)

                except queue.Empty:
                    time.sleep(0.001)
                except Exception as e:
                    self.get_logger().error(f"CRNN处理错误: {str(e)}")
                    self.running = False

       def draw_and_publish(self, frame, yolo_info, value, frame_time):
             """在图像上绘制结果并发布"""
# 基本绘制 (需要根据你的实际需求实现)
             if yolo_info is not None:
# 这里调用你的draw函数
                 pass

# 添加文本显示值
             cv2.putText(frame, f"Value: {value:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 添加时间戳
             cv2.putText(frame, f"Latency: {time.time()-frame_time:.3f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 发布图像
             try:
                 ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                 self.detection_pub.publish(ros_image)
             except Exception as e:
                 self.get_logger().error(f"图像发布错误: {str(e)}")

       def __del__(self):
             """资源清理"""
             if hasattr(self, 'running'):
                  self.running = False

 # 检查属性是否存在再尝试清理
             if hasattr(self, 'yolo_thread') and self.yolo_thread and self.yolo_thread.is_alive():
                  self.yolo_thread.join(timeout=2.0)
             if hasattr(self, 'fpn_thread') and self.fpn_thread and self.fpn_thread.is_alive():
                  self.fpn_thread.join(timeout=2.0)
             if hasattr(self, 'crnn_thread') and self.crnn_thread and self.crnn_thread.is_alive():
                  self.crnn_thread.join(timeout=2.0)

# 检查模型池是否存在再尝试释放
             if hasattr(self, 'yolo_pool'):
                  self.yolo_pool.release()
             if hasattr(self, 'fpn_pool'):
                  self.fpn_pool.release()
             if hasattr(self, 'crnn_pool'):
                  self.crnn_pool.release()
 
             self.get_logger().info("检测节点资源已释放")

def main(args=None):
     rclpy.init(args=args)
     node = MeterDetectionNode()

     try:
       rclpy.spin(node)
     except KeyboardInterrupt:
       pass

     node.destroy_node()
     rclpy.shutdown()

if __name__ == '__main__':
     main()
