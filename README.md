### Solutions for class: [Introduction to Parallel Programming](https://www.udacity.com/course/intro-to-parallel-programming--cs344)

#### Building on Windows Visual Studio
##### Prerequisites
* Install Visual Studio 2013:
	
	**Note**: `Visual Studio Express` and `Visual Studio 2015` are not supported!(I tried but not work ^_^)

	[Nvidia reference](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html#axzz44vwAc5Qx)

* Install Cuda 7.5:
	Also refer to above link. [download](https://developer.nvidia.com/cuda-downloads)

* Install CMake:
	The latest version is OK. [download](https://cmake.org/) 

* Install OpenCV:
	I installed 2.4.12, other versions should also work. [download](http://opencv.org/)
	* Run the EXE to extract the files. This EXE does not have an installer. Instead, you put your files where you want, and then add an environment variable
	* Adding the environment variable named "OpenCV_DIR" (no quotes) to the "build" subfolder in the folder where you extracted.(The exact folder you need will have one very important file in it: OpenCVConfig.cmake - this tells CMake which variables to set for you.)
	* Add a dir of "OpenCV binary DLLs" to Windows $PATH.(like f:/software/opencv/build/x86/vc12/bin)

##### Compile the solution
```
git clone https://github.com/wykvictor/cs344.git
cd cs344
mkdir build
cd build
cmake ..
```

**Done!** Just use Visual Studio to open the project-solution in dir build/ and compile everything.

=======
### Original README.md forked from [udacity/cs344](https://github.com/udacity/cs344)

##### Introduction to Parallel Programming class code

#### Building on OS X

These instructions are for OS X 10.9 "Mavericks".

* Step 1. Build and install OpenCV. The best way to do this is with
Homebrew. However, you must slightly alter the Homebrew OpenCV
installation; you must build it with libstdc++ (instead of the default
libc++) so that it will properly link against the nVidia CUDA dev kit. 
[This entry in the Udacity discussion forums](http://forums.udacity.com/questions/100132476/cuda-55-opencv-247-os-x-maverick-it-doesnt-work) describes exactly how to build a compatible OpenCV.

* Step 2. You can now create 10.9-compatible makefiles, which will allow you to
build and run your homework on your own machine:
```
mkdir build
cd build
cmake ..
make
```

