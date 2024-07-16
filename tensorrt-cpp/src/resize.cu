#include <cuda_runtime.h>
#include <cstdint>

#include "common/check.h"

__global__ void resize_kernel(const float* input, float* output, int input_height, int input_width, int output_height, int output_width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < output_width && y < output_height) {
    float scale_x = (float)input_width / output_width;
    float scale_y = (float)input_height / output_height;
    int x0 = (int)(x * scale_x);
    int x1 = (int)((x + 1) * scale_x);
    int y0 = (int)(y * scale_y);
    int y1 = (int)((y + 1) * scale_y);
    float sum = 0;
    for (int i = y0; i < y1; i++) {
      for (int j = x0; j < x1; j++) {
        sum += input[i * input_width + j];
      }
    }
    output[y * output_width + x] = sum / ((y1 - y0) * (x1 - x0));
  }
}
