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

// Variant 8: Shape m64n32k32 (N=32)
// For e4m3, maybe N=16 is not aligned enough?
// M=64, N=32, K=32.
// Output: 64x32 = 2048 elements.
// 128 threads. Each thread holds 16 elements (f32).
// Registers: 16 floats.
__global__ void v8() {
    float d[16]; uint64_t descA=0, descB=0;
    // Syntax: D, A, B. (Assuming 3 args).
    // Note: inline asm for 16 registers is tedious.
    // We will pack them? No, inline asm requires explicit list.
    // We'll define d0..d15.
    
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(descA), "l"(descB)
    );
}

// Variant 9: D, A, B, C for m64n32k32
__global__ void v9() {
   float d[16]; float c[16]; uint64_t A=0, B=0;
   // Just trying compilation, logic doesn't matter.
   asm volatile(
       "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
       "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
       "%16, %17, "
       "{%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33};\n"
       : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
         "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
       : "l"(A), "l"(B), 
         "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]),
         "f"(c[8]), "f"(c[9]), "f"(c[10]), "f"(c[11]), "f"(c[12]), "f"(c[13]), "f"(c[14]), "f"(c[15])
   );
}

