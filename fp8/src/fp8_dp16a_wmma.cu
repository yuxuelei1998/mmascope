#include <cuda_fp8.h>
#include <cuda_runtime.h>
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

// FP8 Headers might be missing in older toolkits, but we use PTX so we just need basic types.
// If __nv_fp8_e4m3 is missing, we use uint8_t and reinterpret.

namespace fs = std::filesystem;

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

// PTX Wrapper for mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
__device__ void mma_m16n8k32_e4m3(float *d, uint32_t *a, uint32_t *b, float *c) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

__device__ void mma_m16n8k32_e5m2(float *d, uint32_t *a, uint32_t *b, float *c) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

__global__ void ptxFp8Kernel(const uint8_t* A, const uint8_t* B, const float* C, float* D, int numTests, bool isE5M2) {
    int testIdx = blockIdx.x;
    if (testIdx >= numTests) return;
    
    int laneId = threadIdx.x % 32;
    // Pointers to the raw data for this test case
    const uint8_t* valA = A + testIdx * (16 * 32); 
    const uint8_t* valB = B + testIdx * (32 * 16);
    
    uint32_t regA[4];
    uint32_t regB[2]; 
    
    // Fill registers with broadcasted data (simplification for probe)
    // A: 16 bytes available. We read 16 bytes into the 4 registers.
    const uint32_t* srcA = (const uint32_t*)(valA);
    const uint32_t* srcB = (const uint32_t*)(valB);
    
    regA[0] = srcA[laneId % 4];
    regA[1] = srcA[(laneId+1) % 4];
    regA[2] = srcA[(laneId+2) % 4];
    regA[3] = srcA[(laneId+3) % 4];
    
    regB[0] = srcB[laneId % 4]; 
    regB[1] = srcB[(laneId+1) % 4];
    
    float acc[4];
    float initC = C[testIdx];
    acc[0] = initC; acc[1] = initC; acc[2] = initC; acc[3] = initC;
    
    float res[4];
    
    if (isE5M2) {
        mma_m16n8k32_e5m2(res, regA, regB, acc);
    } else {
        mma_m16n8k32_e4m3(res, regA, regB, acc);
    }
    
    // Store result D (just first element for now)
    if (laneId == 0) {
        D[testIdx * 16 * 16] = res[0]; 
    }
}

void executeWMMA(const TestCase* testCases, Result* results, int numTests, bool isE5M2) {
    // A: 16x32 = 512 bytes (logical), inputs are smaller but padded logic handles it.
    const int sizeA = 16 * 32; 
    const int sizeB = 32 * 16;
    const int sizeD = 16 * 16;
    
    uint8_t *d_A, *d_B;
    float *d_C, *d_D;
    
    cudaMalloc(&d_A, numTests * sizeA);
    cudaMalloc(&d_B, numTests * sizeB);
    cudaMalloc(&d_C, numTests * sizeof(float));
    cudaMalloc(&d_D, numTests * sizeD * sizeof(float));
    
    std::vector<uint8_t> h_A(numTests * sizeA, 0);
    std::vector<uint8_t> h_B(numTests * sizeB, 0);
    std::vector<float> h_C(numTests);
    std::vector<float> h_D(numTests * sizeD);
    
    for (int i = 0; i < numTests; i++) {
        for (int j = 0; j < 16; j++) h_A[i*sizeA + j] = testCases[i].vectorA[j];
        for (int j = 0; j < 16; j++) h_B[i*sizeB + j] = testCases[i].vectorB[j];
        h_C[i] = uint32ToFloat(testCases[i].scalarC);
    }
    
    cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch with 32 threads matching snippet logic (1 warp per test)
    ptxFp8Kernel<<<numTests, 32>>>(d_A, d_B, d_C, d_D, numTests, isE5M2);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_D.data(), d_D, numTests * sizeD * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i=0; i<numTests; i++) {
        results[i].result = floatToUint32(h_D[i * sizeD]);
    }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
}

// ... Rest of mapping/file helpers same as before ...
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