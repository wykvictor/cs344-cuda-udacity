############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

CUDA_ADD_EXECUTABLE(Lesson3_histo histo.cu)

CUDA_ADD_EXECUTABLE(Lesson3_reduce reduce.cu)

CUDA_ADD_EXECUTABLE(Lesson3_reduce_minmax reduce_minmax.cu)

CUDA_ADD_EXECUTABLE(Lesson3_reduce_minmax_2 reduce_minmax_2.cu)
