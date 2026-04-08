cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
project(EigenProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 寻找基础库
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
# 显式指定组件，确保包含 visualization
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(Ceres REQUIRED)

add_executable(eigen_test main.cpp)

# 2. 添加 PCL 的宏定义
add_definitions(${PCL_DEFINITIONS})

# 3. 包含头文件目录 (修正了括号问题)
target_include_directories(eigen_test PUBLIC 
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

# 4. 链接库
target_link_libraries(eigen_test 
    Eigen3::Eigen 
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
)

# 将 Assets 文件夹拷贝到编译输出目录
add_custom_command(TARGET eigen_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/assets"
    "$<TARGET_FILE_DIR:eigen_test>/assets")