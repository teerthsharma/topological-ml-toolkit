/*
 * x86-64 backend dispatch probes.
 *
 * These routines are intentionally simple ABI-level building blocks. The
 * production backend must wrap them with OS-specific calling-convention glue,
 * CPUID checks, and safe fallbacks before enabling any optimized ASM path.
 */

        .text
        .globl topoml_cpuid_leaf7_ebx
        .type topoml_cpuid_leaf7_ebx, @function

/*
 * uint32_t topoml_cpuid_leaf7_ebx(void)
 *
 * Returns CPUID.(EAX=7,ECX=0):EBX feature bits.
 * AVX2: bit 5
 * AVX512F: bit 16
 * AVX512DQ: bit 17
 * AVX512CD: bit 28
 * AVX512BW: bit 30
 * AVX512VL: bit 31
 */
topoml_cpuid_leaf7_ebx:
        push    %rbx
        mov     $7, %eax
        xor     %ecx, %ecx
        cpuid
        mov     %ebx, %eax
        pop     %rbx
        ret
        .size topoml_cpuid_leaf7_ebx, .-topoml_cpuid_leaf7_ebx

        .globl topoml_has_avx2
        .type topoml_has_avx2, @function

topoml_has_avx2:
        call    topoml_cpuid_leaf7_ebx
        shr     $5, %eax
        and     $1, %eax
        ret
        .size topoml_has_avx2, .-topoml_has_avx2

        .globl topoml_has_avx512f
        .type topoml_has_avx512f, @function

topoml_has_avx512f:
        call    topoml_cpuid_leaf7_ebx
        shr     $16, %eax
        and     $1, %eax
        ret
        .size topoml_has_avx512f, .-topoml_has_avx512f
