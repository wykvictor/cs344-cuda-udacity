#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#include "gettime.h"

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}

/*double tic() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return ((double)t.tv_sec * 1000 + ((double)t.tv_usec) / 1000.);
}*/

double tic() {
  LARGE_INTEGER m_nFreq;
  LARGE_INTEGER m_Time;
  QueryPerformanceFrequency(&m_nFreq);
  QueryPerformanceCounter(&m_Time);
  return (double)m_Time.QuadPart * 1000. / m_nFreq.QuadPart;
}