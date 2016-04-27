############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

CUDA_ADD_EXECUTABLE(Lesson2_atomics atomics.cu)

CUDA_ADD_EXECUTABLE(Lesson2_memory memory.cu)

CUDA_ADD_EXECUTABLE(Lesson2_hello_world hello.cu)
