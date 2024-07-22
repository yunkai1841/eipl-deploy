#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

#include <cstdint>

void preprocess(uint8_t *src, float *dst, int src_width, int src_height, int dst_width, int dst_height, int channel);

#endif // __PREPROCESS_H__