/*
 * x86-64 scalar assembly reference for an L2 squared-distance hot path.
 *
 * ABI contract, System V style:
 *   float topoml_l2_sq_f32_asm(const float* left, const float* right, long dim)
 *   float topoml_l2_sq_f32_avx512(const float* left, const float* right, long dim)
 *
 * The AVX-512 entry point must only be called after CPUID has confirmed
 * AVX512F support. The Python wrapper enforces that gate and falls back to the
 * scalar ABI on machines without AVX-512F.
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

        .globl topoml_l2_sq_f32_avx512
        .type topoml_l2_sq_f32_avx512, @function

topoml_l2_sq_f32_avx512:
        vxorps  %zmm0, %zmm0, %zmm0
        test    %rdx, %rdx
        jle     .avx512_done

.avx512_loop:
        cmp     $16, %rdx
        jl      .avx512_reduce
        vmovups (%rdi), %zmm1
        vmovups (%rsi), %zmm2
        vsubps  %zmm2, %zmm1, %zmm1
        vmulps  %zmm1, %zmm1, %zmm1
        vaddps  %zmm1, %zmm0, %zmm0
        add     $64, %rdi
        add     $64, %rsi
        sub     $16, %rdx
        jmp     .avx512_loop

.avx512_reduce:
        sub     $64, %rsp
        vmovups %zmm0, (%rsp)
        xorps   %xmm0, %xmm0
        lea     (%rsp), %rax
        mov     $16, %rcx

.avx512_reduce_loop:
        movss   (%rax), %xmm1
        addss   %xmm1, %xmm0
        add     $4, %rax
        dec     %rcx
        jne     .avx512_reduce_loop
        add     $64, %rsp

.avx512_tail:
        test    %rdx, %rdx
        jle     .avx512_done
        movss   (%rdi), %xmm1
        movss   (%rsi), %xmm2
        subss   %xmm2, %xmm1
        mulss   %xmm1, %xmm1
        addss   %xmm1, %xmm0
        add     $4, %rdi
        add     $4, %rsi
        dec     %rdx
        jne     .avx512_tail

.avx512_done:
        vzeroupper
        ret

        .size topoml_l2_sq_f32_avx512, .-topoml_l2_sq_f32_avx512
