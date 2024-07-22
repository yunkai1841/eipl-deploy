#ifndef __DATA_LOADER_H__
#define __DATA_LOADER_H__

#include <vector>
#include <string>
#include <fstream>
#include <random>

#include <opencv2/opencv.hpp>

#include "NvInfer.h"

#include "cuda_runtime_api.h"

#include "common/logger.h"
#include "common/check.h"

typedef std::vector<int> data_shape;

class Buffer
{
public:
    Buffer(std::size_t size) : size_(size)
    {
        host_data_ptr_.reset(new char[size]);
        CUDA_CHECK(cudaMalloc((void **)&(device_data_ptr_), size_));
        LOG_INFO("Buffer size: " << size_);
    }

    ~Buffer() {
        host_data_ptr_.reset();
        if (device_data_ptr_ != nullptr) {
            CUDA_CHECK(cudaFree(device_data_ptr_));
        }
    }

    char *host_data() { return host_data_ptr_.get(); }

    char *device_data() {
        return device_data_ptr_;
    }

    std::size_t size() { return size_; }

    void copy_to_device(cudaStream_t stream)
    {
        // CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>device_data_ptr_, size_, stream));
        CUDA_CHECK(cudaMemcpyAsync(device_data_ptr_, host_data_ptr_.get(), size_, cudaMemcpyHostToDevice, stream));
        // CUDA_CHECK(cudaMemcpy(device_data_ptr_, host_data_ptr_.get(), size_, cudaMemcpyHostToDevice));
        LOG_INFO("Copy data to device");
    }

    void copy_to_host(cudaStream_t stream)
    {
        CUDA_CHECK(cudaMemcpyAsync(host_data_ptr_.get(), device_data_ptr_, size_, cudaMemcpyDeviceToHost, stream));
        LOG_INFO("Copy data to host");
    }

    void fill_random_float()
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        float *data = reinterpret_cast<float *>(host_data_ptr_.get());
        for (std::size_t i = 0; i < size_ / sizeof(float); i++)
        {
            data[i] = distribution(generator);
        }
    }

    void fill_random_int()
    {
        std::default_random_engine generator;
        std::uniform_int_distribution<char> distribution(0, 255);
        char *data = host_data_ptr_.get();
        for (std::size_t i = 0; i < size_ / sizeof(int); i++)
        {
            data[i] = distribution(generator);
        }
    }

private:
    std::shared_ptr<char> host_data_ptr_;
    char* device_data_ptr_;
    // std::shared_ptr<char> device_data_ptr_;
    std::size_t size_;
};

class DataLoader
{
public:
    DataLoader(data_shape shape) : shape_(shape){};
    DataLoader(std::string data_path) : data_path_(data_path){};
    DataLoader(std::vector<std::string> data_paths) : data_paths_(data_paths){};
    ~DataLoader();

    Buffer load_data();

    Buffer load_data(int index);

    void save_data(Buffer &buffer, std::string file_path);


private:
    std::string data_path_;
    std::vector<std::string> data_paths_;
    std::vector<int> shape_;
};

#endif // __DATA_LOADER_H__