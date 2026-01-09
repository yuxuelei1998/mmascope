#include <cuda_fp8.h>
#include <cuda_fp8.hpp>
#include <cuda_runtime.h>
#include <mma.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;
namespace wmma = nvcuda::wmma;

enum Opcode { MPDPA };
enum RoundMode { RND_ZERO, RND_MINUS_INF, RND_PLUS_INF, RND_NEAREST };

struct TestCase {
    Opcode opcode;
    RoundMode roundMode;
    uint8_t vectorA[16];
    uint8_t vectorB[16];
    uint32_t scalarC;
};

struct Result { uint32_t result; };

std::map<std::string, Opcode> opcodeMap = {{"MPDPA", MPDPA}};
std::map<std::string, RoundMode> roundModeMap = {
    {"RND_ZERO", RND_ZERO}, {"RND_MINUS_INF", RND_MINUS_INF},
    {"RND_PLUS_INF", RND_PLUS_INF}, {"RND_NEAREST", RND_NEAREST}
};

inline float uint32ToFloat(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(u));
    return f;
}

inline uint32_t floatToUint32(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(f));
    return u;
}

// Fixed shape to 16x16x16 for FP8 API compatibility
// K=32 is not supported for FP8 accumulator fragment in generic WMMA API

// Kernel for E4M3
__global__ void wmmaKernelE4M3(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, const float* C, float* D, int numTests) {
    int testIdx = blockIdx.x;
    if (testIdx >= numTests) return;

    // A: 16x16 (256 elements)
    // B: 16x16 (256 elements)
    int offset = testIdx * 16 * 16;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_fp8_e4m3, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_fp8_e4m3, wmma::col_major> b_frag; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> d_frag;

    wmma::load_matrix_sync(a_frag, A + offset, 16);
    wmma::load_matrix_sync(b_frag, B + offset, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = C[testIdx];
    }
    
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(D + offset, d_frag, 16, wmma::mem_row_major);
}

// Kernel for E5M2
__global__ void wmmaKernelE5M2(const __nv_fp8_e5m2* A, const __nv_fp8_e5m2* B, const float* C, float* D, int numTests) {
    int testIdx = blockIdx.x;
    if (testIdx >= numTests) return;

    int offset = testIdx * 16 * 16;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_fp8_e5m2, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_fp8_e5m2, wmma::col_major> b_frag; 
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> d_frag;

    wmma::load_matrix_sync(a_frag, A + offset, 16);
    wmma::load_matrix_sync(b_frag, B + offset, 16);
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = C[testIdx];
    }
    
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(D + offset, d_frag, 16, wmma::mem_row_major);
}

void executeWMMA(const TestCase* testCases, Result* results, int numTests, bool isE5M2) {
    const int M = 16;
    const int N = 16;
    const int K = 16; // Fixed K=16
    const int sizeA = M * K;
    const int sizeB = K * N;
    const int sizeD = M * N;

    void *d_A, *d_B; 
    float *d_C, *d_D;

    cudaMalloc(&d_A, numTests * sizeA * sizeof(uint8_t));
    cudaMalloc(&d_B, numTests * sizeB * sizeof(uint8_t));
    cudaMalloc(&d_C, numTests * sizeof(float));
    cudaMalloc(&d_D, numTests * sizeD * sizeof(float));

    std::vector<uint8_t> h_A(numTests * sizeA);
    std::vector<uint8_t> h_B(numTests * sizeB);
    std::vector<float> h_C(numTests);
    std::vector<float> h_D(numTests * sizeD);

    memset(h_A.data(), 0, h_A.size());
    memset(h_B.data(), 0, h_B.size());

    for (int i = 0; i < numTests; i++) {
        int offsetA = i * sizeA;
        int offsetB = i * sizeB;
        
        // Fill row 0 of A (16 elements)
        for (int j = 0; j < 16; j++) {
             h_A[offsetA + j] = testCases[i].vectorA[j];
        }
        // Fill col 0 of B (16 elements)
        for (int j = 0; j < 16; j++) {
             h_B[offsetB + j] = testCases[i].vectorB[j];
        }
        h_C[i] = uint32ToFloat(testCases[i].scalarC);
    }

    cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(float), cudaMemcpyHostToDevice);

    if (isE5M2) {
        wmmaKernelE5M2<<<numTests, 32>>>((__nv_fp8_e5m2*)d_A, (__nv_fp8_e5m2*)d_B, d_C, d_D, numTests);
    } else {
        wmmaKernelE4M3<<<numTests, 32>>>((__nv_fp8_e4m3*)d_A, (__nv_fp8_e4m3*)d_B, d_C, d_D, numTests);
    }
    
    cudaDeviceSynchronize();

    cudaMemcpy(h_D.data(), d_D, numTests * sizeD * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numTests; i++) {
        results[i].result = floatToUint32(h_D[i * sizeD]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
}

uint8_t parseHex8(const std::string& hexStr) {
    return static_cast<uint8_t>(std::stoul(hexStr, nullptr, 16));
}

uint32_t parseHex32(const std::string& hexStr) {
    return static_cast<uint32_t>(std::stoul(hexStr, nullptr, 16));
}

std::vector<TestCase> readInputFile(const std::string& filename) {
    std::vector<TestCase> testCases;
    std::ifstream file(filename);
    if (!file.is_open()) return testCases;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(iss, token, ',')) {
            token.erase(0, token.find_first_not_of(' '));
            token.erase(token.find_last_not_of(' ') + 1);
            if (!token.empty()) tokens.push_back(token);
        }

        if (tokens.size() == 35) {
            TestCase tc;
            auto opcodeIter = opcodeMap.find(tokens[0]);
            auto roundIter = roundModeMap.find(tokens[1]);
            if (opcodeIter != opcodeMap.end() && roundIter != roundModeMap.end()) {
                tc.opcode = opcodeIter->second;
                tc.roundMode = roundIter->second;
                for (int i = 0; i < 16; i++) tc.vectorA[i] = parseHex8(tokens[2 + i]);
                for (int i = 0; i < 16; i++) tc.vectorB[i] = parseHex8(tokens[2 + 16 + i]);
                tc.scalarC = parseHex32(tokens[2 + 32]);
                testCases.push_back(tc);
            }
        }
    }
    return testCases;
}

void writeOutputFile(const std::string& filename, const std::vector<TestCase>& testCases, const std::vector<Result>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    for (size_t i = 0; i < testCases.size(); ++i) {
        file << "0x" << std::hex << std::setw(8) << std::setfill('0') << results[i].result << "\n";
    }
}

void processFile(const std::string& inputFilePath) {
    std::cout << "Processing: " << inputFilePath << std::endl;
    std::vector<TestCase> testCases = readInputFile(inputFilePath);
    if (testCases.empty()) return;

    bool isE5M2 = (inputFilePath.find("e5m2") != std::string::npos);
    std::cout << "Detected Type: " << (isE5M2 ? "FP8 E5M2" : "FP8 E4M3") << std::endl;

    int numTests = testCases.size();
    std::vector<Result> results(numTests);

    executeWMMA(testCases.data(), results.data(), numTests, isE5M2);

    fs::path inputPath(inputFilePath);
    std::string outputFileName = inputPath.stem().string() + "_16x16_wmma_output" + inputPath.extension().string();
    fs::path outputDir = "../numeric_fingerprints";
    if (!fs::exists(outputDir)) fs::create_directories(outputDir);
    std::string outputPath = (outputDir / outputFileName).string();
    
    writeOutputFile(outputPath, testCases, results);
    std::cout << "Output written to: " << outputPath << std::endl;
}

int main() {
    std::string folderPath = "../fp8_dp16a";
    if (!fs::exists(folderPath)) return 1;

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            processFile(entry.path().string());
        }
    }
    return 0;
}