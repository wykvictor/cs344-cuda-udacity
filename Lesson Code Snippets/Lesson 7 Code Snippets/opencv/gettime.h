#ifndef GETTIME_H
#define GETTIME_H

#include <winsock.h>

// MSVC defines this in winsock2.h!?
/*struct timeval {
    long tv_sec;
    long tv_usec;
};
*/
int gettimeofday(struct timeval * tp, struct timezone * tzp);

double tic();

#endif
