#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "common/timer.h"
#include "common/check.h"

__global__ void resize_kernel(const float *input, float *output, int input_height, int input_width, int output_height, int output_width)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < output_width && y < output_height)
  {
    float scale_x = (float)input_width / output_width;
    float scale_y = (float)input_height / output_height;
    int x0 = (int)(x * scale_x);
    int x1 = (int)((x + 1) * scale_x);
    int y0 = (int)(y * scale_y);
    int y1 = (int)((y + 1) * scale_y);
    float sum = 0;
    for (int i = y0; i < y1; i++)
    {
      for (int j = x0; j < x1; j++)
      {
        sum += input[i * input_width + j];
      }
    }
    output[y * output_width + x] = sum / ((y1 - y0) * (x1 - x0));
  }
}

__global__ void resize_rgb_kernel(const uchar3 *input, uchar3 *output, int input_height, int input_width, int output_height, int output_width)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < output_width && y < output_height)
  {
    float scale_x = (float)input_width / output_width;
    float scale_y = (float)input_height / output_height;
    int x0 = (int)(x * scale_x);
    int x1 = (int)((x + 1) * scale_x);
    int y0 = (int)(y * scale_y);
    int y1 = (int)((y + 1) * scale_y);
    int sum_r = 0, sum_g = 0, sum_b = 0;
    for (int i = y0; i < y1; i++)
    {
      for (int j = x0; j < x1; j++)
      {
        uchar3 pixel = input[i * input_width + j];
        sum_r += pixel.x;
        sum_g += pixel.y;
        sum_b += pixel.z;
      }
    }
    uchar3 pixel;
    pixel.x = sum_r / ((y1 - y0) * (x1 - x0));
    pixel.y = sum_g / ((y1 - y0) * (x1 - x0));
    pixel.z = sum_b / ((y1 - y0) * (x1 - x0));
    output[y * output_width + x] = pixel;
  }
}

int main()
{
  // Load image
  cv::Mat image = cv::imread("data/image.png", cv::IMREAD_COLOR);
  if (image.empty())
  {
    std::cerr << "Failed to load image." << std::endl;
    return 1;
  }

  // Preprocess image
  cv::Mat resized;
  // bool useGPU = true;
  // if (useGPU) {
  //     TimerScope timer("Opencv GPU Preprocess");
  //     cv::cuda::GpuMat gpuImage;
  //     gpuImage.upload(image);
  //     cv::cuda::GpuMat gpuResized;
  //     cv::cuda::resize(gpuImage, gpuResized, cv::Size(224, 224));
  //     gpuResized.download(resized);
  // } else
  {
    TimerScope timer("Preprocess");
    cv::resize(image, resized, cv::Size(224, 224));
  }
  cv::imshow("Resized", resized);
  cv::waitKey(0);

  // Preprocess image with my kernel
  cv::Mat image2 = cv::imread("data/image.png", cv::IMREAD_COLOR);
  cv::Mat resized2(224, 224, CV_8UC3);

  uchar3 *d_input;
  uchar3 *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, image2.rows * image2.cols * sizeof(uchar3)));
  CUDA_CHECK(cudaMalloc(&d_output, resized2.rows * resized2.cols * sizeof(uchar3)));
  CUDA_CHECK(cudaMemcpy(d_input, image2.data, image2.rows * image2.cols * sizeof(uchar3), cudaMemcpyHostToDevice));

  dim3 block(32, 32);
  dim3 grid((resized2.cols + block.x - 1) / block.x, (resized2.rows + block.y - 1) / block.y);
  {
    TimerScope timer("Preprocess with my kernel");
    resize_rgb_kernel<<<grid, block>>>(d_input, d_output, image2.rows, image2.cols, resized2.rows, resized2.cols);
  }
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(resized2.data, d_output, resized2.rows * resized2.cols * sizeof(uchar3), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  cv::imshow("Resized with my kernel", resized2);
  cv::waitKey(0);
}
