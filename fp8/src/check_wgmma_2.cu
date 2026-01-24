#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

using e4m3 = __nv_fp8_e4m3;

// Test larger shapes which are more likely to be valid for FP8 atoms.
// m64n64k32: 64x64 output. 128 threads.
// 4096 elements / 128 = 32 floats per thread.
// Registers: d0..d31.

__global__ void v17_m64n64() {
    float d[32]; 
    uint64_t descA=0, descB=0;
    // Standard 3-arg syntax: D, A, B.
    // e4m3 inputs, f32 accumulator.
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),  "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(descA), "l"(descB)
    );
}

// Variant 18: m64n64k32 with ScaleD (immediate 1)
__global__ void v18_m64n64_scale() {
    float d[32]; uint64_t descA=0, descB=0;
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, 1;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),  "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(descA), "l"(descB)
    );
}
