#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include <gflags/gflags.h>

#include "NvInfer.h"

#include "common/logger.h"
#include "common/timer.h"

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

    // Clean up
    delete engine;
    delete runtime;

    return 0;
}