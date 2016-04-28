############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

file( GLOB  Lesson7_tiling_hdr tiling/*.hpp tiling/*.h )
SET (Lesson7_tiling_files tiling/tiling.cu)
CUDA_ADD_EXECUTABLE(Lesson7_tiling ${Lesson7_tiling_files} ${Lesson7_tiling_hdr})

file( GLOB  Lesson7_thrust_hdr thrust/*.h )
SET (Lesson7_thrust_files thrust/thrust_example.cu thrust/gettime.cc)
CUDA_ADD_EXECUTABLE(Lesson7_thrust ${Lesson7_thrust_files} ${Lesson7_thrust_hdr})

file( GLOB  Lesson7_opencv_hdr opencv/*.h )
SET (Lesson7_opencv_files opencv/opencv.cu opencv/gettime.cc)
CUDA_ADD_EXECUTABLE(Lesson7_opencv ${Lesson7_opencv_files} ${Lesson7_opencv_hdr})
