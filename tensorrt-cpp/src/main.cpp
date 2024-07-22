#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include <gflags/gflags.h>

#include "NvInfer.h"

#include "cuda_runtime_api.h"

#include "common/logger.h"
#include "common/timer.h"
#include "data_loader.h"
#include "preprocess.h"

using namespace std;

DEFINE_string(trt_file, "", "Path to TensorRT engine file");
DEFINE_bool(verbose, false, "Enable verbose logging");


int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Load the TensorRT engine from file
    LOG_INFO("Loading TensorRT engine from file: " << FLAGS_trt_file);
    std::ifstream trt_file(FLAGS_trt_file, std::ios::binary);
    if (!trt_file.is_open())
    {
        std::cerr << "Failed to open file: " << FLAGS_trt_file << std::endl;
        return 1;
    }

    trt_file.seekg(0, trt_file.end);
    size_t size = trt_file.tellg();
    trt_file.seekg(0, trt_file.beg);

    std::unique_ptr<char[]> trt_data(new char[size]);
    trt_file.read(trt_data.get(), size);
    trt_file.close();

    // Deserialize the engine
    Logger logger(FLAGS_verbose ? nvinfer1::ILogger::Severity::kVERBOSE : nvinfer1::ILogger::Severity::kWARNING);
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(trt_data.get(), size);
    if (!engine)
    {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return 1;
    }
    LOG_INFO("Engine deserialized successfully");

    // dummy data
    const int src_width = 640;
    const int src_height = 480;
    const int dst_width = 64;
    const int dst_height = 64;
    const int channel = 3;

    Buffer input_buffer(src_width * src_height * channel * sizeof(uint8_t));
    Buffer preprocessed_buffer(dst_width * dst_height * channel * sizeof(float));

    input_buffer.fill_random_int();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    input_buffer.copy_to_device(stream);

    cudaStreamSynchronize(stream);

    // Preprocess the input data
    {
        TimerScope timer_scope("Preprocess");
        preprocess(
            reinterpret_cast<uint8_t *>(input_buffer.device_data()),
            reinterpret_cast<float *>(preprocessed_buffer.device_data()),
            src_width, src_height, dst_width, dst_height, channel);    
    }

    // Clean up
    delete engine;
    delete runtime;

    return 0;
}