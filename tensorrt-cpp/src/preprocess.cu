#include <cuda_runtime.h>
#include <cstdint>

/**
 * Preprocess for image data
 *
 * 1. Resize image
 * raw image data: src_width * src_height * channel
 * output image data size: channel * dst_width * dst_height
 *
 * 2. Normalize image
 * raw image data is uint8_t (0, 255), normalize to float32 (0, 1)
 */

__global__ void preprocess_kernel(uint8_t *src, float *dst, int src_width, int src_height, int dst_width, int dst_height, int channel)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < dst_width && y < dst_height && c < channel)
  {
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    int src_x = min((int)(x * scale_x), src_width - 1);
    int src_y = min((int)(y * scale_y), src_height - 1);
    dst[c * dst_width * dst_height + y * dst_width + x] = src[c * src_width * src_height + src_y * src_width + src_x] / 255.0f;
  }
}

void preprocess(uint8_t *src, float *dst, int src_width, int src_height, int dst_width, int dst_height, int channel)
{
  dim3 block(32, 32, 1);
  dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y, (channel + block.z - 1) / block.z);
  preprocess_kernel<<<grid, block>>>(src, dst, src_width, src_height, dst_width, dst_height, channel);
}
