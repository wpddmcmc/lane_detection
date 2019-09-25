# RoadCar_Detect

---

**Author:** 陈明聪 Michael.CHEN

---

### 1.项目介绍 -Introduction

自动驾驶，曲线道路检测以及车辆检测
Auto-driving, Curl Road & Car detecting

### 2.开发环境 -Environment

**系统环境 System Dependency:** Ubuntu16.04

**软件环境 Software Dependency:** Cmake2.8 or upper, C++11, OpenCV3.4.0(only), darknet(**Please download the fix version in Inspur_RTD**)
**请下载Inspur_RTD组织内的fix版本darknet**
**硬件环境 Hardware Environment:** 

```
Computer1: CPU:i7-4720HQ GPU:Nvidia GTX960M-2GB RAM:12GB
Computer2: CPU:i7-8750H GPU:Nvidia GTX1060-6GB RAM:8GB/16GB
```

### 3.文件结构 -File Structure

**src:** 源文件 source code file
**include: **头文件 head file
**lib:**库文件 library file
**build: **编译目录 compile file
**build/data: **YOLO数据列表依赖 yolo data list file
**yolov3-tiny: **YOLO网络文件 yolo network file
**video: **测试视频 test video demo
### 4.安装教程 -Install

#### i.确保开发环境依赖安装完成 -Make sure ur Software Dependency has been installed

#### ii.配置编译 -Configure compile

配置```CMakeLists.txt ```

Configure ```CMakeLists.txt``` .

```bash
gedit CMakeLists.txt
```

如果装有多版本OpenCV,修改文件第8行，指定想要使用的OpenCV路径

if you have OpenCV version more than one, uncommit and change line8,set the build path which one you want to use.

```cmake
#if u have OpenCV version more than one, set the build path which one u want to use
set(OpenCV_DIR "YOUR_PATH")
```

举例如下:

Ex:

```cmake
#if u have OpenCV version more than one, set the build path which one u want to use
set(OpenCV_DIR "/home/test/app/opencv-3.4.0/build/")
```

如果使用Inter Realsense 相机，取消第14行注释

if use realsense,uncommit line14.

```cmake
#if use realsense,uncommit this
#set(DEPENDENCIES realsense ${OPENGL_LIBRARIES})
```

第50,51行配置include_directories中darknet路径以及53行cuda路径

In line50,51, configure the darknet path and cuda path in include_directories

```cmake
include_directories (
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
    /usr/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
#darknet path
    YOUR_PATH/include
    YOUR_PATH/src
#cuda path
    YOUR_PATH/include    
)
```

举例如下:

Ex:

```cmake
include_directories (
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
    /usr/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
#darknet path
    /home/test/app/darknet/include
    /home/test/app/darknet/src
#cuda path
    /usr/local/cuda-9.0/include    
)
```

第62行配置target_link_libraries中darknet库路径

In line62, configure the darknet library path in target_link_libraries

```cmake
target_link_libraries(autocar
    ${OpenCV_LIBS}
    /usr/lib
    /usr/local/lib
    ${DEPENDENCIES}
    #darknet lib path
    YOUR_PATH/libdarknet.so
)
```

举例如下:

Ex:

```cmake
target_link_libraries(autocar
    ${OpenCV_LIBS}
    /usr/lib
    /usr/local/lib
    ${DEPENDENCIES}
    #darknet lib path
    /home/test/app/darknet/libdarknet.so
)
```

#### iii.编译代码 -Compile

进入```build```文件夹进行编译

Entry ```build/``` and compile

```bash
cd build
cmake ..
make
```

#### iv.运行程序 -Run

```bash
./autocar
```

### 5.使用说明 -Using Note

#### i.代码修改 -Code change

修改完代码**需重新在```build```文件夹下执行```make```编译**

if you changed the code , then ***need to ```make```to compile again***.

#### ii.代码文件变更 -Code file change

如有源代码文件移动(新增,删除文件，重命名)**需在```build/```文件夹下重新执行```cmake ..```配置并```make```编译**,即4.iii

If any file changes(add, delete, rename), you**need to ```cmake ..``` and ```make```in ```build/```**

#### iii. 摄像头使用 -Camera using

```ImageConsProd.cpp``` 源文件第31行，取消注释```#define USE_CAMERA```，即可使用摄像头(默认0，于49行```VideoCapture cap(0);```更改摄像头编号), 修改完成**需要重新```make```编译**

In source file ```ImageConsProd.cpp``` line31, uncommit ```#define USE_CAMERA```. Then you can use camera (default 0, you can chage camera index at line49  ```VideoCapture cap(0);```), after that you**need to ``make`` to compile**

#### iv.视频文件使用 -Video file using

```ImageConsProd.cpp``` 源文件第31行，注释```#define USE_CAMERA```，即可使用视频, 修改完成**需要重新```make```编译**

In source file ``` ImageConsProd.cpp``` line31, commit ```#define USE_CAMERA```. Then you can use video file, after that you **need to ``make`` to compile**

选定视频文件，需要将视频放入```video/```文件夹,然后在```param/```文件夹下的```param_config.xml```内第4行修改使用的视频文件名. **注:不需要编译，只需保存修改**

The choosing of video file. Put video file into ```video/``` folder, then change the video name you want to use in line4 of ```param_config.xml``` in ```param/```. **Note: just save change, no need to re-compile**

#### v.Debug 模式 -Debug Mode

Debug模式下，读入视频或者摄像头图像，检测键盘，每敲击一下键盘读入一帧图像

In Debug Mode, detect keyboard after load video or camera image. Each frame load with each click of keyboard.

Debug 模式打开与关闭

Debug Mode ON/OFF

修改```param/```文件夹下的```param_config.xml```第3行，1代表打开，0代表关闭

Change line3 of ```param_config.xml``` in ```param/```, 1 ON, 0 OFF.

```xml
<?xml version="1.0"?>
<opencv_storage>
<debug_mode>1</debug_mode>
<video_name>vedioname.mp4</video_name>
</opencv_storage>
```

