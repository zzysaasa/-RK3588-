cmake_minimum_required(VERSION 3.5)
project(yolo_meter_detection)

# 查找 ament_python（纯Python包所需）
find_package(ament_cmake REQUIRED)

# 纯Python包配置
ament_python_install_package(${PROJECT_NAME})

# 安装启动文件
install(DIRECTORY launch
DESTINATION share/${PROJECT_NAME}
)

# 安装配置文件和模型
install(DIRECTORY cfg rknnModel
DESTINATION share/${PROJECT_NAME}
)

# 安装通用文件
install(FILES package.xml codec.txt
DESTINATION share/${PROJECT_NAME}
)

# 注册包含
ament_package()
