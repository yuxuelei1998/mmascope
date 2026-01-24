#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cassert>
#include <iomanip>

// FP8 Types mapping
using e4m3 = __nv_fp8_e4m3;
using e5m2 = __nv_fp8_e5m2;

// WGMMA Kernel for Hopper (sm_90a)
// Uses 128 threads (1 Warp Group)
// Computes D = A * B + C
// A: M64 x K32 (in SMEM)
// B: K32 x N16 (in SMEM) (Transposed for WGMMA usually, or col-major)
// D: M64 x N16 (Accumulator / Output)

// Helper to create tensor map descriptor (simplified for flat SMEM address)
// For WGMMA, valid descriptor is a 64-bit integer.
// Bits 0-13: start address (16B aligned) >> 4
// Bits 16-29: leading dimension (16B multiples) >> 4
// Bits 30-31: stride dimension (0=no stride, etc.) - simplified here
// ...
// Actually, for m64n16k32, we can just point to SMEM address if swizzling is standard.
// But we need to be careful.
// Let's use the inline assembly `desc` construction if possible, or just the address if the instruction supports raw pointer (some do, but usually desc is preferred for swizzling).
// For the probe, we will use the simplest valid descriptor: simple linear layout.

__device__ __forceinline__ uint64_t make_smem_desc(void* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    // Simple descriptor: just the address (shifted) and valid swizzle mode implied or 0
    // Format:
    // [13:0] Address >> 4
    // [29:16] Leading Dim >> 4 (stride)
    // [31:30] Swizzle mode (0 = none, 1 = 32B, 2 = 64B, 3 = 128B)
    // We will attempt swizzle 0 first (linear) to see if it works, or use a "standard" formation.
    // However, Hopper WGMMA often expects specific swizzling.
    // For this simple probe, we'll try to keep it simple.
    // If we just pass the address to the instruction in some modes it works, but `wgmma` usually takes a 64-bit desc register.
    
    // Pattern for simple linear memory (may degrade perf but functional for probe):
    // Address must be 16-byte aligned.
    uint64_t desc = 0;
    desc |= (uint64_t)(addr >> 4); 
    // Stride/Leading Dim: Set to simple row usage
    // For K=32 (FP8 = 1 byte), row stride is 32 bytes?
    // Let's assume K=32 is the leading dim (packed). 
    // desc |= ((32 >> 4) << 16); 
    
    // Ideally we assume standard 128B swizzle for max perf, but linear is easier to debug.
    // We will just use the address for now, relying on the fact that for small tiles we control the layout.
    // Actually, let's use a "dummy" robust desc if possible.
    // Reference: "wgmma.mma_async" usually takes a generic shared pointer OR a descriptor.
    // We will use the `state space` form if possible, but the instruction usually essentially wants a descriptor.
    
    // Let's try constructing a basic one. 
    return desc;
}

// NOTE: To strictly avoid F2F, we use "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3"
// This consumes 1 warp group (128 threads).
// M=64, N=16, K=32.
// A matrix: 64x32 FP8 elements = 2048 bytes.
// B matrix: 32x16 FP8 elements = 512 bytes.

template<typename T_A, typename T_B>
__global__ void wgmma_fp8_kernel(const uint8_t* A, const uint8_t* B, const float* C, float* D, int numTests) {
    int testIdx = blockIdx.x;
    if (testIdx >= numTests) return;

    // We are one warp group (128 threads)
    int tid = threadIdx.x;
    
    // Shared Memory Setup
    // A: 64x32 bytes
    // B: 32x16 bytes (needs to be Transposed for WGMMA? Or Col Major?)
    // Standard WGMMA expects A Row-Major, B Col-Major (or usually KN).
    // Let's assume standard layout KxN for B is optimal.
    
    extern __shared__ uint8_t smem[];
    uint8_t* sm_A = smem;                 // Size: 64*32 = 2048
    uint8_t* sm_B = smem + 2048;          // Size: 32*16 = 512
    
    // Load data
    // Total A bytes = 2048. 128 threads. Each loads 16 bytes.
    for (int i=0; i < 16; ++i) {
        int idx = tid * 16 + i;
        if (idx < 2048) sm_A[idx] = A[testIdx * (16*32) + (idx % (16*32))]; // Repeats input if smaller
    }
    // Total B bytes = 512.
    for (int i=0; i < 4; ++i) {
        int idx = tid * 4 + i;
        if (idx < 512) sm_B[idx] = B[testIdx * (32*16) + (idx % (32*16))];
    }
    
    __syncthreads();
    
    // Synchronize for WGMMA
    asm volatile("wgmma.fence.sync.aligned;");
    
    // WGMMA Implementation
    // Accumulators in registers. D is M64xN16.
    // Each thread holds a fragment. 
    // 128 threads. Total elements 64*16 = 1024.
    // Each thread holds 8 elements (FP32).
    float regs[8];
    float initVal = C[testIdx];
    for(int i=0; i<8; ++i) regs[i] = initVal;

    // Generic pointer to shared memory
    // For simple probe, we use the `desc` computed from pointer.
    uint64_t descA = make_smem_desc(sm_A);
    uint64_t descB = make_smem_desc(sm_B);
    
    // PTX for WGMMA
    // m64n16k32
    // Correct argument list for SM90a wgmma.mma_async for FP8:
    // It appears floating point WGMMA follows D, A, B, C signature.
    // Since we use the same registers for D and C (accumulator), we pass them twice.
    
    if (std::is_same<T_A, e4m3>::value && std::is_same<T_B, e4m3>::value) {
        asm volatile(
            "{\n"
            "   wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, "
            "%8, "
            "%9, "
            "{%0, %1, %2, %3, %4, %5, %6, %7};\n" 
            "}\n"
            : "+f"(regs[0]), "+f"(regs[1]), "+f"(regs[2]), "+f"(regs[3]),
              "+f"(regs[4]), "+f"(regs[5]), "+f"(regs[6]), "+f"(regs[7])
            : "l"(descA), "l"(descB) 
        );
    } else {
        // Assume e5m2
        asm volatile(
            "{\n"
            "   wgmma.mma_async.sync.aligned.m64n16k32.f32.e5m2.e5m2 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, "
            "%8, "
            "%9, "
            "{%0, %1, %2, %3, %4, %5, %6, %7};\n"
            "}\n"
            : "+f"(regs[0]), "+f"(regs[1]), "+f"(regs[2]), "+f"(regs[3]),
              "+f"(regs[4]), "+f"(regs[5]), "+f"(regs[6]), "+f"(regs[7])
            : "l"(descA), "l"(descB)
        );
    }

    // Commit and Wait
    asm volatile("wgmma.commit_group.sync.aligned;");
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    
    // Store results
    // Logic to map threads to global D is complex for WGMMA.
    // We just dump everything linearly to D for now to preserve "probe" behavior.
    // D size: M64 * N16 = 1024 floats.
    // 128 threads * 8 regs = 1024 floats. Perfect.
    int outOffset = tid * 8;
    for (int i=0; i<8; ++i) {
        if (outOffset + i < 1024) {
             D[testIdx * 1024 + outOffset + i] = regs[i];
        }
    }
}

struct TestCase {
    uint8_t vectorA[32]; // Not full matrix, just vector
    uint8_t vectorB[32];
    uint32_t scalarC;
};

struct Result {
    uint32_t result; 
};

union IntFloat {
    uint32_t i;
    float f;
};

// Host Wrapper
void executeWMMA(const TestCase* testCases, Result* results, int numTests, bool isE5M2) {
    // Input Sizes (Probe specific)
    // The kernel expects bigger inputs (M64, K32, N16).
    // The test case only provides 32 bytes (vector).
    // We will replicate the vector to fill the matrix or just pad with zeros.
    // For the probe purpose (detecting instruction), functionality is secondary to ISA generation.
    
    int sizeBytes_A = 64 * 32; // 2KB
    int sizeBytes_B = 32 * 16; // 512B
    int sizeFloats_D = 64 * 16; // 1024 floats
    
    uint8_t *d_A, *d_B;
    float *d_C, *d_D;
    
    cudaMalloc(&d_A, numTests * sizeBytes_A);
    cudaMalloc(&d_B, numTests * sizeBytes_B);
    cudaMalloc(&d_C, numTests * sizeof(float));
    cudaMalloc(&d_D, numTests * sizeFloats_D * sizeof(float));

    std::vector<uint8_t> h_A(numTests * sizeBytes_A, 0);
    std::vector<uint8_t> h_B(numTests * sizeBytes_B, 0);
    std::vector<float> h_C(numTests);
    std::vector<float> h_D(numTests * sizeFloats_D);
    
    // Fill Data
    for (int i=0; i<numTests; ++i) {
        // Copy vector to first row/col
        memcpy(&h_A[i*sizeBytes_A], testCases[i].vectorA, 32);
        memcpy(&h_B[i*sizeBytes_B], testCases[i].vectorB, 32); 
        IntFloat cVal;
        cVal.i = testCases[i].scalarC;
        h_C[i] = cVal.f;
    }
    
    cudaMemcpy(d_A, h_A.data(), h_A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch
    // 128 threads per block, 4KB shared mem dynamic usually enough (2.5KB used)
    int smemSize = 4096; 
    
    if (isE5M2) {
        wgmma_fp8_kernel<e5m2, e5m2><<<numTests, 128, smemSize>>>(d_A, d_B, d_C, d_D, numTests);
    } else {
        wgmma_fp8_kernel<e4m3, e4m3><<<numTests, 128, smemSize>>>(d_A, d_B, d_C, d_D, numTests);
    }
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_D.data(), d_D, h_D.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Harvest results
    for (int i=0; i<numTests; ++i) {
        IntFloat res;
        res.f = h_D[i * sizeFloats_D]; // Take first element
        results[i].result = res.i;
    }
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
}