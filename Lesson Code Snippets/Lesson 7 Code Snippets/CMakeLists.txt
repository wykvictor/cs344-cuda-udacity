############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

file( GLOB  Lesson5_tiling_hdr tiling/*.hpp tiling/*.h )

SET (Lesson5_tiling_files tiling/tiling.cu)

CUDA_ADD_EXECUTABLE(Lesson5_tiling ${Lesson5_tiling_files} ${Lesson5_tiling_hdr})



