#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
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
using namespace rocwmma;

enum Opcode { MPDPA };
enum RoundMode { RND_ZERO, RND_MINUS_INF, RND_PLUS_INF, RND_NEAREST };

struct TestCase {
    Opcode opcode;
    RoundMode roundMode;
    uint16_t vectorA[16];
    uint16_t vectorB[16];
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

// 16x16x16 MFMA Kernel using rocWMMA
__global__ void wmmaKernel(const __half* A, const __half* B, const float* C, float* D, int numTests) {
    int testIdx = blockIdx.x;
    if (testIdx >= numTests) return;

    int offset = testIdx * 16 * 16;
    
    // Define fragments
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fragment<accumulator, 16, 16, 16, float> d_frag;

    // Load matrices
    load_matrix_sync(a_frag, A + offset, 16);
    load_matrix_sync(b_frag, B + offset, 16);
    
    // Fill accumulator
    fill_fragment(c_frag, 0.0f);
    
    // Manually set C fragment values to the scalar C from input
    // Note: implementation detail dependent on fragment structure,
    // essentially we want C matrix to be all the same scalar if that's the test intent,
    // OR if the input C was meant to be a full matrix.
    // The original CUDA code set all elements of C_frag to C[testIdx].
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = C[testIdx];
    }
    
    // D = A * B + C
    mma_sync(d_frag, a_frag, b_frag, c_frag);
    
    // Store result
    store_matrix_sync(D + offset, d_frag, 16, mem_row_major);
}

void executeWMMA(const TestCase* testCases, Result* results, int numTests) {
    const int matrixSize = 16 * 16;
    __half *d_A, *d_B;
    float *d_C, *d_D;

    hipMalloc(&d_A, numTests * matrixSize * sizeof(__half));
    hipMalloc(&d_B, numTests * matrixSize * sizeof(__half));
    hipMalloc(&d_C, numTests * sizeof(float));
    hipMalloc(&d_D, numTests * matrixSize * sizeof(float));

    std::vector<__half> h_A(numTests * matrixSize, __half(0.0f));
    std::vector<__half> h_B(numTests * matrixSize, __half(0.0f));
    std::vector<float> h_C(numTests);
    std::vector<float> h_D(numTests * matrixSize);

    for (int i = 0; i < numTests; i++) {
        int offset = i * matrixSize;
        
        for (int j = 0; j < 16; j++) {
            h_A[offset + 0 * 16 + j] = __float2half_rn(uint32ToFloat(testCases[i].vectorA[j] << 16)); 
            // Input hex is likely 16-bit int representation of half? 
            // Original CUDA code: __ushort_as_half
            // In HIP we can interpret bits directly if needed, or cast.
            // Let's assume input vectorA is uint16_t bits of the half.
            unsigned short valA = testCases[i].vectorA[j];
            h_A[offset + 0 * 16 + j] = *reinterpret_cast<__half*>(&valA);
        }
        
        for (int j = 0; j < 16; j++) {
            unsigned short valB = testCases[i].vectorB[j];
            h_B[offset + j * 16 + 0] = *reinterpret_cast<__half*>(&valB);
        }
        
        h_C[i] = uint32ToFloat(testCases[i].scalarC);
    }

    hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C.data(), h_C.size() * sizeof(float), hipMemcpyHostToDevice);

    wmmaKernel<<<numTests, 64>>>(d_A, d_B, d_C, d_D, numTests); // Wave size 64 for AMD usually
    hipDeviceSynchronize();

    hipMemcpy(h_D.data(), d_D, numTests * matrixSize * sizeof(float), hipMemcpyDeviceToHost);
    
    for (int i = 0; i < numTests; i++) {
        results[i].result = floatToUint32(h_D[i * matrixSize]);
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_D);
}

uint16_t parseHex16(const std::string& hexStr) {
    return static_cast<uint16_t>(std::stoul(hexStr, nullptr, 16));
}

uint32_t parseHex32(const std::string& hexStr) {
    return static_cast<uint32_t>(std::stoul(hexStr, nullptr, 16));
}

std::vector<TestCase> readInputFile(const std::string& filename) {
    std::vector<TestCase> testCases;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << filename << std::endl;
        return testCases;
    }

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

        if (tokens.size() != 35) {
            continue;
        }

        TestCase tc;
        auto opcodeIter = opcodeMap.find(tokens[0]);
        auto roundIter = roundModeMap.find(tokens[1]);
        if (opcodeIter == opcodeMap.end() || roundIter == roundModeMap.end()) {
            continue;
        }

        tc.opcode = opcodeIter->second;
        tc.roundMode = roundIter->second;

        try {
            for (int i = 0; i < 16; i++) {
                tc.vectorA[i] = parseHex16(tokens[2 + i]);
            }
            for (int i = 0; i < 16; i++) {
                tc.vectorB[i] = parseHex16(tokens[2 + 16 + i]);
            }
            tc.scalarC = parseHex32(tokens[2 + 32]);
            testCases.push_back(tc);
        } catch (...) {
            continue;
        }
    }

    std::cout << "Read " << testCases.size() << " test cases from " << filename << std::endl;
    return testCases;
}

void writeOutputFile(const std::string& filename,
                    const std::vector<TestCase>& testCases,
                    const std::vector<Result>& results) {
    std::ofstream file(filename);
    if (!file.is_open()) return;

    for (size_t i = 0; i < testCases.size(); ++i) {
        const Result& res = results[i];
        file << "0x" << std::hex << std::setw(8) << std::setfill('0') << res.result << "\n";
    }
}

void processFile(const std::string& inputFilePath) {
    std::vector<TestCase> testCases = readInputFile(inputFilePath);
    if (testCases.empty()) return;

    int numTests = testCases.size();
    std::vector<Result> results(numTests);

    executeWMMA(testCases.data(), results.data(), numTests);

    fs::path inputPath(inputFilePath);
    std::string outputFileName = inputPath.stem().string() + "_16x16_wmma_output" + inputPath.extension().string();
    fs::path outputDir = "../numeric_fingerprints";
    
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    std::string outputPath = (outputDir / outputFileName).string();
    writeOutputFile(outputPath, testCases, results);
    std::cout << "Output written to: " << outputPath << std::endl;
}

int main() {
    // Basic AMD setup verification
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No AMD devices found." << std::endl;
        return 1;
    }
    
    std::string folderPath = "../fp16_dp16a";
    if (!fs::exists(folderPath)) {
        std::cerr << "Error: Input folder " << folderPath << " does not exist." << std::endl;
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::cout << "Processing: " << entry.path() << std::endl;
            processFile(entry.path().string());
        }
    }
    return 0;
}
