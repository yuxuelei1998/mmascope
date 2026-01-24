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
__global__ void v1() {
    float d[8]; uint64_t a=0, b=0;
    // Standard 3-arg with implicit scale 1 (failed before, maybe .desc qualifier needed?)
    // Trying explicit .desc modifier on operands if supported? No, usually it's instruction level.
    // Let's try explicit scale 1 again but with specific f32 pattern.
    // Actually, documentation says scaling is for integer. FP WGMMA doesn't support generic scaling?
    // Let's try NO scale, but with proper descriptor types (already u64).
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9;\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "l"(a), "l"(b));
}

__global__ void v2() {
    float d[8]; float c[8]; uint64_t a=0, b=0;
    // 4-operand with accumulator. 
    // Maybe the 'scale' args are actually `p` (predicate) for something else?
    // Let's try 4 args again: D, A, B, C.
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, {%10, %11, %12, %13, %14, %15, %16, %17};\n" 
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) 
    : "l"(a), "l"(b), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
}

__global__ void v3() {
    float d[8]; uint64_t a=0, b=0;
    // Try with `scale-D` (0 or 1).
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1;\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "l"(a), "l"(b));
}

__global__ void v4() {
    float d[8]; uint64_t a=0, b=0;
    // Maybe `scale` is explicitly `p`?
    // Only Int8 mandates arguments for sat/scale. 
    // For FP8, maybe it is: D, A, B, scale-D? 
    // Let's try `p` again but with scale-D as valid immediate.
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0;\n" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]) : "l"(a), "l"(b));
}

