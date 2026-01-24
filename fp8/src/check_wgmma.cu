#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

// Dummy types
using e4m3 = __nv_fp8_e4m3;

__global__ void probe_wgmma() {
    // Registers for D (8 floats)
    float d[8];
    // Registers for C (8 floats)
    float c[8];
    
    // Descriptors (uint64_t)
    uint64_t descA = 0;
    uint64_t descB = 0;
    
    // Variant 1: D, A, B (3 args)
    // asm volatile(
    //     "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
    //     "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9;\n"
    //     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    //     : "l"(descA), "l"(descB)
    // );

    // Variant 2: D, A, B, C (4 args)
    // asm volatile(
    //     "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
    //     "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, {%10, %11, %12, %13, %14, %15, %16, %17};\n"
    //     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    //     : "l"(descA), "l"(descB), 
    //       "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7])
    // );

    // Variant 3: D, A, B, scale (immediate 1)
    // asm volatile(
    //     "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
    //     "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1;\n"
    //     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    //     : "l"(descA), "l"(descB)
    // );
}

// We will uncomment one by one in the run loop or just include all in different functions

// Variant 5: D, A, B, scale-D (register)
__global__ void v5() {
    float d[8]; uint64_t a=0, b=0; 
    int scaleD = 1;
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, %10;\n" 
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) 
    : "l"(a), "l"(b), "r"(scaleD));
}

// Variant 6: D, A, B, scale-D, scale-A, scale-B (registers)
__global__ void v6() {
    float d[8]; uint64_t a=0, b=0;
    int sD=1, sA=1, sB=1;
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, %10, %11, %12, 0;\n" 
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) 
    : "l"(a), "l"(b), "r"(sD), "r"(sA), "r"(sB));
}

// Variant 7: Different shape m64n32k32 (maybe n16 is not supported for e4m3?)
// N=32 -> 64x32 output. 128 threads.
// 2048 elems / 128 = 16 floats.
// This requires d[16]. 
__global__ void v7() {
    float d[16]; uint64_t a=0, b=0;
    // asm volatile to avoid lengthy string, just check conceptual syntax
    // This is getting complex to write inline.
    // Let's stick to checking if V5 or V6 works.
}
