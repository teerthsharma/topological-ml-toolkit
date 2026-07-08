/*
 * x86-64 scalar assembly reference for an L2 squared-distance hot path.
 *
 * ABI contract, System V style:
 *   float topoml_l2_sq_f32_asm(const float* left, const float* right, long dim)
 *
 * This file is intentionally small and auditable. AVX2/AVX-512 variants should
 * be added beside it only after CPUID dispatch and correctness equivalence tests
 * are in place.
 */

        .text
        .globl topoml_l2_sq_f32_asm
        .type topoml_l2_sq_f32_asm, @function

topoml_l2_sq_f32_asm:
        pxor    %xmm0, %xmm0
        test    %rdx, %rdx
        jle     .done

.loop:
        movss   (%rdi), %xmm1
        movss   (%rsi), %xmm2
        subss   %xmm2, %xmm1
        mulss   %xmm1, %xmm1
        addss   %xmm1, %xmm0
        add     $4, %rdi
        add     $4, %rsi
        dec     %rdx
        jne     .loop

.done:
        ret

        .size topoml_l2_sq_f32_asm, .-topoml_l2_sq_f32_asm
