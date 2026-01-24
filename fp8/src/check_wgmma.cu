#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

using e4m3 = __nv_fp8_e4m3;

// Registers for m64n32k32:
// M=64, N=32. 64*32 = 2048 elements.
// 128 threads.
// 2048 / 128 = 16 elements per thread.
// f32 accumulator => 16 float registers.

// Variant 1: D, A, B (3 args) for m64n32k32
__global__ void v1_m64n32() {
    float d[16]; uint64_t descA=0, descB=0;
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(descA), "l"(descB)
    );
}

// Variant 2: D, A, B, C (4 args) for m64n32k32
// D and C are same regs here.
__global__ void v2_m64n32() {
    float d[16]; float c[16]; uint64_t descA=0, descB=0;
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17, "
        "{%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(descA), "l"(descB),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]),
          "f"(c[8]), "f"(c[9]), "f"(c[10]), "f"(c[11]), "f"(c[12]), "f"(c[13]), "f"(c[14]), "f"(c[15])
    );
}

// Variant 3: D, A, B, scale-D (immediate) 
__global__ void v3_m64n32() {
    float d[16]; uint64_t descA=0, descB=0;
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17, 1;\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(descA), "l"(descB)
    );
}

// Variant 4: Argument list from H100_GEMM but for m64n32k32
// D, A, B, p, 1, 1, 0, 0
__global__ void v4_m64n32() {
    float d[16]; uint64_t descA=0, descB=0;
    int scaleD = 1; 
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17, "
        "p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
        : "l"(descA), "l"(descB), "r"(scaleD)
    );
}
