#include <sys/time.h>
#include <stdint.h>

uint64_t micros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * (uint64_t)1000000 + tv.tv_usec;
}
