#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

// FP8 Types mapping
using e4m3 = __nv_fp8_e4m3;
using e5m2 = __nv_fp8_e5m2;

// -------------------------------------------------------------------------
// WGMMA Kernel m64n8k32 (N=8)
// -------------------------------------------------------------------------
// A: M64 x K32 (in SMEM) => 64*32 = 2048 elements
// B: K32 x N8  (in SMEM) => 32*8  = 256 elements
// D: M64 x N8  (in Registers) => 512 elements
// 128 threads.
// D per thread: 512 / 128 = 4 floats.

__device__ __forceinline__ uint64_t make_smem_desc(void* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    // Simple linear descriptor: Address >> 4 in bits [0:13]
    // Swizzle: 0 (linear) for simplicity/probe safety.
    // Leading Dimension: K=32 bytes? (Assuming packed row major for A, col major for B)
    // For WGMMA, LDM encoded in bits [16:29].
    // Let's rely on basic address-only desc first as it often works for flat layouts in these probes.
    desc = (uint64_t)(addr >> 4); 
    return desc;
}

template<typename T_A, typename T_B>
__global__ void wgmma_m64n8k32_kernel(const uint8_t* A, const uint8_t* B, float* D, int numTests) {
    int testIdx = blockIdx.x;
    if (testIdx >= numTests) return;

    int tid = threadIdx.x;
    
    // Shared Memory Configuration
    extern __shared__ uint8_t smem[];
    uint8_t* sm_A = smem;                 // 2048 bytes
    uint8_t* sm_B = smem + 2048;          // 256 bytes (32*8)
    
    // Zero out Shared Memory first (to handle padding)
    for (int i = tid; i < 2048 + 256; i += 128) {
        smem[i] = 0;
    }
    __syncthreads();

    // Load Input Data (Vector to Matrix mapping)
    // Input A: 32 bytes (Row 0 of A). A is 64x32.
    // Input B: 32 bytes (Col 0 of B?). B is 32x8.
    // We fill A[0..31] with input A.
    // We fill B[0..31] (Column 0) with input B? 
    // Wait, B is 32 rows x 8 cols. 
    // If B is column major, B[0..31] is indeed the first column.
    
    // Load A (32 bytes)
    if (tid < 32) {
        sm_A[tid] = A[testIdx * 32 + tid];
    }
    
    // Load B (32 bytes) - Mapping to first column of 32x8 matrix.
    // Memory layout: 32x8.
    // If Row Major: element (r, c) is at r*8 + c.
    // Col 0 elements: 0, 8, 16, 24... 
    // If Col Major: element (r, c) is at c*32 + r.
    // Col 0 elements: 0, 1, 2... 
    // WGMMA usually expects B in Col Major (or specific block layout). 
    // Let's assume Col Major for B for simplicity of loading vector -> Col 0.
    if (tid < 32) {
        sm_B[tid] = B[testIdx * 32 + tid];
    }
    
    __syncthreads();
    
    // WGMMA Fence
    asm volatile("wgmma.fence.sync.aligned;");
    
    // Descriptors
    uint64_t descA = make_smem_desc(sm_A);
    uint64_t descB = make_smem_desc(sm_B);
    
    // Registers for Accumulator D (4 floats)
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // Initialize to 0 (C=0 effectively)

    // Execute WGMMA (m64n8k32)
    // Syntax inferred: D, A, B. D and C are same registers if accumulating. 
    // Here we init D to 0, so D = A*B + 0.
    
    int scaleD = 1;
    if (std::is_same<T_A, e4m3>::value) {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %6, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
            "{%0, %1, %2, %3}, %4, %5, p, %6, 1, 1, 0, 0;\n"
            "}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "l"(descA), "l"(descB), "r"(scaleD)
        );
    } else {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %6, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2 "
            "{%0, %1, %2, %3}, %4, %5, p, %6, 1, 1, 0, 0;\n"
            "}\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "l"(descA), "l"(descB), "r"(scaleD)
        );
    }

    // Commit and Wait
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    
    // Store Result
    // D is M64 x N8. 
    // We want D[0,0]. 
    // We need to know which thread holds fragment (0,0).
    // For m64n8k32, standard layout: fragment 0 is often in thread 0.
    // Let's assume thread 0 holds D[0,0] in d[0].
    
    if (tid == 0) {
        D[testIdx] = d[0];
    }
}

// -------------------------------------------------------------------------
// Host Code
// -------------------------------------------------------------------------

struct TestCase {
    uint8_t a[32];
    uint8_t b[32];
    // We ignore the expected result in file for now, we just compute ours.
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
        
        // Expected format: MPDPA, RND, 32 of A, 32 of B, Result
        // Total 67 tokens. Data starts at index 2.
        // A: 2..33 (32 bytes)
        // B: 34..65 (32 bytes)
        
        if (tokens.size() < 66) continue; // Basic check
        
        TestCase tc;
        try {
            for(int i=0; i<32; ++i) {
                // Parse hex string " 0x00" -> uint8_t
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
            continue; // Skip malformed lines
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
    cudaMalloc(&d_D, numTests * sizeof(float)); // Only store 1 float per test
    
    std::vector<uint8_t> flatA(numTests * 32);
    std::vector<uint8_t> flatB(numTests * 32);
    
    for(int i=0; i<numTests; ++i) {
        memcpy(&flatA[i*32], cases[i].a, 32);
        memcpy(&flatB[i*32], cases[i].b, 32);
    }
    
    cudaMemcpy(d_A, flatA.data(), flatA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), flatB.size(), cudaMemcpyHostToDevice);
    
    // Launch
    int smemSize = 4096; 
    if (isE5M2) {
        wgmma_m64n8k32_kernel<e5m2, e5m2><<<numTests, 128, smemSize>>>(d_A, d_B, d_D, numTests);
    } else {
        wgmma_m64n8k32_kernel<e4m3, e4m3><<<numTests, 128, smemSize>>>(d_A, d_B, d_D, numTests);
    }
    cudaDeviceSynchronize();
    
    std::vector<float> results(numTests);
    cudaMemcpy(results.data(), d_D, numTests * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write Output
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
    
    // E5M2
    runTests(baseDir + "fp8_e5m2.txt", outDir + "fp8_e5m2_wmma_output.txt", true);
    
    // E4M3
    runTests(baseDir + "fp8_e4m3.txt", outDir + "fp8_e4m3_wmma_output.txt", false);
    
    return 0;
}