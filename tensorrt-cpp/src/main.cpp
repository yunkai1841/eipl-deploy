#include <iostream>
#include <fstream>
#include <memory>
#include <string>

#include <gflags/gflags.h>

#include "NvInfer.h"

using namespace std;

DEFINE_string(trt_file, "", "Path to TensorRT engine file");


int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Load the TensorRT engine from file
    std::ifstream trt_file(FLAGS_trt_file, std::ios::binary);
    if (!trt_file.is_open()) {
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
    nvinfer1::ILogger& gLogger = nvinfer1::Logger(nvinfer1::Logger::Severity::kINFO);
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trt_data.get(), size);
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return 1;
    }

    // Clean up
    engine->destroy();
    runtime->destroy();

    return 0;
}