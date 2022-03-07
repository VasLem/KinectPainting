Drawing with Kinect on any surface
==================
A new degree of freedom.
-------
This project is intended to allow to everyone with a depth sensor and a projector to create interactive drawings on any surface, 
expanding the idea of blackboard and creating new horizons for creativity. It offers an immersive experience, 
upgrading simple gestures to drawing actions that can make any crazy idea come to life.

Dependencies/Requirements
----
0. Currently, a **Kinect v2 Sensor** hardware is required, along with a **Kinect Adapter** cable. In future releases these requirements  will be made optional. If you do not own such a device, you can test it offline, either by feeding it with recorded frames from a directory or by feeding it with a rosbag file.

1. **Linux OS** , due to the fact that ROS platform is needed for all the tools to cooperate synchronously. My OS is Ubuntu 16.04, but any setup can work, as long as all the dependencies can be fullfilled. I am planning to make a preconfigured VM for Windows some day, till then anyone can try to get it working. 

0. **Various preinstallation packages**:
		
		sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev cmake-qt-gui
1. If NVIDIA GPU is available, one can install **CUDA** beforehand, after verifying that his GPU is supported in https://developer.nvidia.com/cuda-gpus , but **Caution**, the installation is hard and many problems may arise if the NVIDIA drivers are not installed correctly on the system, which are not very well documented and much experimentation migh be needed if luck is absent. 
If you are not an experienced user or lack of patience, skip this step, 
although take into consideration that this step can not be done after completion of the next steps, 
but the procedure will have to be repeated. Installing CUDA will make everything run faster.

	To install CUDA, visit https://developer.nvidia.com/cuda-downloads
2. **ROS Kinetic**

	 Installation Instructions:  http://wiki.ros.org/kinetic/Installation/Ubuntu

3. **OpenCV 3.1.0 dev**

	You have to install  from source due to a bug of contour approximation in image edges.
	Clone either commit from Nov 18, 2016 for safety: https://github.com/opencv/opencv/tree/12569dc73058686d3e0d7724aafa70cf524f8e26 , or clone current version: https://github.com/opencv/opencv :

		git clone <selected_repository>

	To compile(assume DIR is the Path where you cloned OPENCV) :
	
		cd DIR
		cd opencv
		mkdir build
		cd build
		cmake-gui ..
	
	When cmake-gui runs, if you have CUDA installed check if all CUDA flags are set correctly.
	In `CMAKE_INSTALL_PREFIX` add the directory where you want OPENCV to be installed 
	(let's name it INSTALL_DIR). Set 
	`CMAKE_BUILD_TYPE` to `RELEASE`. Search of any flags starting with "WITH_" prefix and tick 
	whatever you might need, while I urge you to tick `WITH_QT` , `WITH_TBB`, `WITH_OPENCL`, `WITH_PNG`,
	`WITH_EIGEN`, `WITH_CUFFT` and `WITH_CUBLAS`. If you have NVIDIA GPU you might want to tick 
	`WITH_NVCUVID` and if MATLAB is available, `WITH_MATLAB` should be ticked too. Check if every other 
	directory is set correctly, press Configure and after that, if no errors arise, press 
	Generate. After Generate finishes close cmake-gui window, go to terminal and:

		make -j4
		make install
	
	This will generally take 30 minutes and above. You can change the -j4 to -j5 or above, but
	the computer might hang and be almost unusable during installation, while the speed of the 
	installation might decrease due to memory and hardware bottlenecks.

	Due to the fact that ROS installation will have probably dominated your Python environment 
	after installation, the easiest way to make OPENCV library accessible from Python is to copy
	cv2.so file from INSTALL_DIR/lib/python2.7/dist-packages to 
	/opt/ros/kinetic/lib/python2.7/dist-packages.

	You can check if everything went ok by running Python console and trying:
		
		import cv2
		print cv2.__version__
	
	If the above prints ` '3.1.0-dev' `, then the installation was successful

4. **Some Python libraries** might be required. As I did not hold record of this requirement, 
some troubleshooting is needed. One can run `python init_code.py` . If any module error is raised, then he should should `pip install` the selected module and, if he wants, he can name it in the issues page of the repository, so that I can add it in this section.

	Currently I am using wxPython for the GUI part. To install wxPython the following is needed:
		
		sudo apt-get install libwebkit-dev		 
		sudo pip install --upgrade --trusted-host wxpython.org --pre -f http://wxpython.org/Phoenix/snapshot-builds/ wxPython_Phoenix 

	For the coding book stage I am using scikit-fuzzy toolbox:


		sudo pip install scikit-fuzzy
	
	It is suggested to also install progressbar2:
		
		sudo pip install progressbar2
		

5. **Kinect 2 Libraries** are required, although this will change in upcoming releases, so that to remove this requirement and make it optional. Actually, any depth device with a software that can publish to ROS can be used in this project, so there is no reason to demand those specific libraries to start with.

	To install them, visit https://github.com/code-iai/iai_kinect2 and follow the installation instructions, skipping ROS installation.
Usage
----

The basic configuration can be done through the `config.yaml` file. The primary program to run is `init_code.py` .

Report Issues/ Feedback/ Copyright
---

Any issue that anyone has can  be reported, as I am a solo team and some things might have skipped my attention, but I do not guarantee immediate response.
