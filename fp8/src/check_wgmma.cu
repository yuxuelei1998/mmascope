#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>

using e4m3 = __nv_fp8_e4m3;

// Variant 10: Logic from H100_GEMM ptx.cuh
// Syntax: {D}, descA, descB, p, ScaleA, ScaleB, TransA, TransB
// D: 8 registers (for m64n16k32)
// p: predicate (1 for accumulate, 0 for zero)
// ScaleA/B: 1?
// TransA/B: 0?
__global__ void v10() {
    float d[8]; uint64_t descA=0, descB=0;
    int scaleD = 1; // Accumulate = 1
    
    // Note: We need to set p based on scaleD logic from reference.
    // Argument list:
    // D, descA, descB, p, ScaleA(imm), ScaleB(imm), TransA(imm), TransB(imm)
    // Total args after operands: p, 1, 1, 0, 0
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, "
        "%9, "
        "p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "l"(descA), "l"(descB), "r"(scaleD)
    );
}

// Variant 11: Accumulator C explicitly in registers?
// Reference showed D used as input/output (+=).
// If `scaleD` (via p) is 1, it accumulates. D is operand 0..7.
// So operands 0..7 are in/out ("+f").
// This matches v10.

// Variant 12: Try m64n16k32 vs m64n32k32 again if v10 fails?
// Reference struct name: SM90_64x128x16...
// This implies m64n128k16.
// Maybe e4m3 requires specific shape?
