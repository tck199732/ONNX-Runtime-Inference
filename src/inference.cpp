
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <onnxruntime_cxx_api.h>

#include "cnpy.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

void unpackFeatures(cnpy::NpyArray x, cnpy::NpyArray& x_out,
                    cnpy::NpyArray& mask_out)
{
    auto shape = x.shape;
    auto ndims = x.shape.size();
    assert(static_cast<int>(ndims) == 2);

    auto numNodes = shape[0];
    auto numFeatures = shape[1];
    auto numPairs = numFeatures / 2;

    x_out = cnpy::NpyArray({numNodes, numPairs, 2}, sizeof(float), false);
    mask_out = cnpy::NpyArray({numNodes, numPairs}, sizeof(bool), false);

    for (int iNode = 0; iNode < numNodes; ++iNode)
    {
        for (int iPair = 0; iPair < numPairs; ++iPair)
        {
            float energy = x.data<float>()[iNode * numFeatures + 2 * iPair];
            float time = x.data<float>()[iNode * numFeatures + 2 * iPair + 1];
            x_out.data<float>()[iNode * numPairs * 2 + iPair * 2] = energy;
            x_out.data<float>()[iNode * numPairs * 2 + iPair * 2 + 1] = time;
            mask_out.data<bool>()[iNode * numPairs + iPair] =
                (std::abs(energy) > 0.0f) || (std::abs(time) > 0.0f);
        }
    }
    return;
}

void oc_inference(const std::vector<float>& x, const std::vector<float>& beta,
                  std::vector<int>& object_ids, double beta_thres = 0.4,
                  double dist_thres = 0.8, int bkg_idx = -1);

int main(int argc, char* argv[])
{
    int64_t batchSize = 1; // fixed batch size, 1 event per file

    bool useCUDA{true};
    const char* useCUDAFlag = "--use_cuda";
    const char* useCPUFlag = "--use_cpu";
    if (argc == 1)
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) == 0))
    {
        useCUDA = true;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCPUFlag) == 0))
    {
        useCUDA = false;
    }
    else if ((argc == 2) && (strcmp(argv[1], useCUDAFlag) != 0))
    {
        useCUDA = false;
    }
    else
    {
        throw std::runtime_error{"Too many arguments."};
    }

    if (useCUDA)
    {
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    }
    else
    {
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }

    std::string instanceName{"object-condensation-inference"};
    std::string modelFilepath{"./data/models/my_model.onnx"};
    std::string inputFilepath = "./data/test_data/oc/00000000.npz";

    /**
     *  PROCESS DATA TO DESIRED INPUT FORMAT HERE
     *  - Load npy arrays of a graph sample (1 event)
     *  - create flatten arrays for onnx input
     */

    cnpy::npz_t data = cnpy::npz_load(inputFilepath);
    const auto& node_features_array = data["node_features"];
    const auto& edge_index_array = data["edge_index"];
    const auto& edge_attr_array = data["edge_attr"];
    const auto& node_targets_array = data["node_targets"];
    const auto& node_positions_array = data["node_positions"];

    cnpy::NpyArray x_npy;
    cnpy::NpyArray fea_mask_npy;
    unpackFeatures(node_features_array, x_npy, fea_mask_npy);

    int64_t numNodes = x_npy.shape[0];

    // Input Tensor Values
    std::vector<float> x = x_npy.as_vec<float>();
    std::vector<float> pos = node_positions_array.as_vec<float>();

    auto fea_mask = std::make_unique<bool[]>(fea_mask_npy.num_vals);
    auto node_mask = std::make_unique<bool[]>(numNodes);

    for (int i = 0; i < fea_mask_npy.num_vals; ++i)
    {
        fea_mask[i] = fea_mask_npy.data<bool>()[i];
    }
    for (int i = 0; i < numNodes; ++i)
    {
        node_mask[i] = true;
    }

    // Output Tensor Values
    std::vector<float> x_c(pos.size());
    std::vector<float> beta(numNodes);

    /**
     * Setting up ONNX environment
     */

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    if (useCUDA)
    {
        // Using CUDA backend
        // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extsize_t inputTensorSize =
    // vectorProduct(inputDims);ended optimizations (Includes level 1 + more
    // complex optimizations like node fusions) ORT_ENABLE_ALL -> To Enable All
    // possible optimizations

    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::vector<std::string> inputNameStrs;
    std::vector<ONNXTensorElementDataType> inputTypes;
    std::vector<std::vector<int64_t>> inputDims;

    for (int iInput = 0; iInput < numInputNodes; iInput++)
    {
        Ort::AllocatedStringPtr inputNamePtr =
            session.GetInputNameAllocated(iInput, allocator);
        const char* inputName = inputNamePtr.get();
        inputNameStrs.push_back(std::string(inputName));

        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(iInput);

        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputType = inputTensorInfo.GetElementType();
        auto inputShape = inputTensorInfo.GetShape();

        if (inputShape.at(0) == -1)
        {
            std::cout << "Got dynamic batch size. Setting input batch size to "
                      << batchSize << "." << std::endl;
            inputShape.at(0) = batchSize;
        }
        else
        {
            assert(("Input batch size should match the batch size of the input "
                    "graph sample.",
                    inputShape.at(0) == batchSize));
        }

        if (inputShape.at(1) == -1)
        {
            std::cout << "Got dynamic node size. Setting node size to "
                      << numNodes << "." << std::endl;
            inputShape.at(1) = numNodes;
        }
        else
        {
            assert(("Input node size should match the node size of the input "
                    "graph sample.",
                    inputShape.at(1) == numNodes));
        }

        inputTypes.push_back(inputType);
        inputDims.push_back(inputShape);
    }

    std::vector<std::string> outputNameStrs;
    std::vector<ONNXTensorElementDataType> outputTypes;
    std::vector<std::vector<int64_t>> outputDims;

    for (int iOutput = 0; iOutput < numOutputNodes; iOutput++)
    {
        Ort::AllocatedStringPtr outputNamePtr =
            session.GetOutputNameAllocated(iOutput, allocator);
        const char* outputName = outputNamePtr.get();
        outputNameStrs.push_back(std::string(outputName));

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(iOutput);

        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputType = outputTensorInfo.GetElementType();
        auto outputShape = outputTensorInfo.GetShape();

        if (outputShape.at(0) == -1)
        {
            std::cout << "Got dynamic batch size. Setting output batch size to "
                      << batchSize << "." << std::endl;
            outputShape.at(0) = batchSize;
        }
        else
        {
            assert(
                ("Output batch size should match the batch size of the input "
                 "graph sample.",
                 outputShape.at(0) == batchSize));
        }

        if (outputShape.at(1) == -1)
        {
            std::cout << "Got dynamic node size. Setting node size to "
                      << numNodes << "." << std::endl;
            outputShape.at(1) = numNodes;
        }
        else
        {
            assert(("Output node size should match the node size of the input "
                    "graph sample.",
                    outputShape.at(1) == numNodes));
        }

        outputTypes.push_back(outputType);
        outputDims.push_back(outputShape);
    }

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    for (int iInput = 0; iInput < numInputNodes; iInput++)
    {
        std::cout << "Input Name: " << inputNameStrs[iInput] << std::endl;
        std::cout << "Input Type: " << inputTypes[iInput] << std::endl;
        std::cout << "Input Dimensions: " << inputDims[iInput] << std::endl;
    }

    for (int iOutput = 0; iOutput < numOutputNodes; iOutput++)
    {
        std::cout << "Output Name: " << outputNameStrs[iOutput] << std::endl;
        std::cout << "Output Type: " << outputTypes[iOutput] << std::endl;
        std::cout << "Output Dimensions: " << outputDims[iOutput] << std::endl;
    }

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;
    std::vector<const char*> inputNames{
        inputNameStrs[0].c_str(), inputNameStrs[1].c_str(),
        inputNameStrs[2].c_str(), inputNameStrs[3].c_str()};

    std::vector<const char*> outputNames{outputNameStrs[0].c_str(),
                                         outputNameStrs[1].c_str()};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, x.data(), x.size(), inputDims[0].data(),
        inputDims[0].size()));

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, pos.data(), pos.size(), inputDims[1].data(),
        inputDims[1].size()));

    inputTensors.push_back(Ort::Value::CreateTensor<bool>(
        memoryInfo, fea_mask.get(),
        numNodes * (node_features_array.shape[1] / 2), inputDims[2].data(),
        inputDims[2].size()));

    inputTensors.push_back(Ort::Value::CreateTensor<bool>(
        memoryInfo, node_mask.get(), numNodes, inputDims[3].data(),
        inputDims[3].size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, x_c.data(), x_c.size(), outputDims[0].data(),
        outputDims[0].size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, beta.data(), beta.size(), outputDims[1].data(),
        outputDims[1].size()));

    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 4, outputNames.data(),
                outputTensors.data(), 2);

    // Post-processing of output tensors to get object condensation clusters
}
