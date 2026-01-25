#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

using e4m3 = __nv_fp8_e4m3;
using e5m2 = __nv_fp8_e5m2;

__device__ __forceinline__ uint32_t get_smem_ptr(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template<typename T_A, typename T_B>
__global__ void mma_m16n8k32_kernel(const uint8_t* A, const uint8_t* B, float* D, int numTests) {
    int testIdx = blockIdx.x;
    if (testIdx >= numTests) return;

    int tid = threadIdx.x;
    
    extern __shared__ uint8_t smem[];
    uint8_t* sm_A = smem;
    uint8_t* sm_B = smem + 512;
    
    for (int i = tid; i < 512 + 256; i += 32) {
        smem[i] = 0;
    }
    __syncthreads();

    if (tid < 32) {
        sm_A[tid] = A[testIdx * 32 + tid];
    }
    
    if (tid < 32) {
        sm_B[tid] = B[testIdx * 32 + tid];
    }
    
    __syncthreads();
    
    uint32_t ra[4];
    uint32_t rb[2];
    float rd[4];
    float rc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint32_t sm_a_ptr = get_smem_ptr(sm_A);
    uint32_t sm_b_ptr = get_smem_ptr(sm_B);

    int load_stride_A = 0;
    uint32_t addr_A = sm_a_ptr + tid * 16;
    
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(ra[0]), "=r"(ra[1]), "=r"(ra[2]), "=r"(ra[3])
        : "r"(addr_A)
    );
    
    uint32_t addr_B = sm_b_ptr + tid * 8;

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(rb[0]), "=r"(rb[1])
        : "r"(addr_B)
    );
    
    if (std::is_same<T_A, e4m3>::value) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(rd[0]), "=f"(rd[1]), "=f"(rd[2]), "=f"(rd[3])
            : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
              "r"(rb[0]), "r"(rb[1]),
              "f"(rc[0]), "f"(rc[1]), "f"(rc[2]), "f"(rc[3])
        );
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(rd[0]), "=f"(rd[1]), "=f"(rd[2]), "=f"(rd[3])
            : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
              "r"(rb[0]), "r"(rb[1]),
              "f"(rc[0]), "f"(rc[1]), "f"(rc[2]), "f"(rc[3])
        );
    }

    if (tid == 0) {
        D[testIdx] = rd[0];
    }
}

struct TestCase {
    uint8_t a[32];
    uint8_t b[32];
};

std::vector<TestCase> readInputFile(const std::string& filepath) {
    std::vector<TestCase> cases;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return cases;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() < 66) continue;
        
        TestCase tc;
        try {
            for(int i=0; i<32; ++i) {
                size_t start = tokens[2+i].find("0x");
                if (start == std::string::npos) start = 0; else start += 2;
                tc.a[i] = (uint8_t)std::stoul(tokens[2+i].substr(start), nullptr, 16);
            }
            for(int i=0; i<32; ++i) {
                size_t start = tokens[34+i].find("0x");
                if (start == std::string::npos) start = 0; else start += 2;
                tc.b[i] = (uint8_t)std::stoul(tokens[34+i].substr(start), nullptr, 16);
            }
            cases.push_back(tc);
        } catch(...) {
            continue;
        }
    }
    return cases;
}

union FloatUint {
    float f;
    uint32_t u;
};

void runTests(const std::string& inputPath, const std::string& outputPath, bool isE5M2) {
    auto cases = readInputFile(inputPath);
    if (cases.empty()) return;
    
    int numTests = cases.size();
    
    uint8_t *d_A, *d_B;
    float *d_D;
    
    cudaMalloc(&d_A, numTests * 32);
    cudaMalloc(&d_B, numTests * 32);
    cudaMalloc(&d_D, numTests * sizeof(float));
    
    std::vector<uint8_t> flatA(numTests * 32);
    std::vector<uint8_t> flatB(numTests * 32);
    
    for(int i=0; i<numTests; ++i) {
        memcpy(&flatA[i*32], cases[i].a, 32);
        memcpy(&flatB[i*32], cases[i].b, 32);
    }
    
    cudaMemcpy(d_A, flatA.data(), flatA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), flatB.size(), cudaMemcpyHostToDevice);
    
    int smemSize = 4096; 
    if (isE5M2) {
        mma_m16n8k32_kernel<e5m2, e5m2><<<numTests, 32, smemSize>>>(d_A, d_B, d_D, numTests);
    } else {
        mma_m16n8k32_kernel<e4m3, e4m3><<<numTests, 32, smemSize>>>(d_A, d_B, d_D, numTests);
    }
    cudaDeviceSynchronize();
    
    std::vector<float> results(numTests);
    cudaMemcpy(results.data(), d_D, numTests * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::ofstream outFile(outputPath);
    for(float f : results) {
        FloatUint fu;
        fu.f = f;
        outFile << "0x" << std::hex << std::setw(8) << std::setfill('0') << fu.u << std::endl;
    }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);
    std::cout << "Generated " << outputPath << std::endl;
}

int main() {
    std::string baseDir = "../fp8_dp32a/";
    std::string outDir = "../numeric_fingerprints/";
    
    runTests(baseDir + "fp8_e5m2.txt", outDir + "fp8_e5m2_wmma_output.txt", true);
    
    runTests(baseDir + "fp8_e4m3.txt", outDir + "fp8_e4m3_wmma_output.txt", false);
    
    return 0;
}