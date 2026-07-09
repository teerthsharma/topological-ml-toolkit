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

        .globl topoml_cpuid_leaf1_ecx
        .type topoml_cpuid_leaf1_ecx, @function

/*
 * uint32_t topoml_cpuid_leaf1_ecx(void)
 *
 * Returns CPUID.(EAX=1):ECX feature bits.
 * OSXSAVE: bit 27
 * AVX: bit 28
 */
topoml_cpuid_leaf1_ecx:
        push    %rbx
        mov     $1, %eax
        xor     %ecx, %ecx
        cpuid
        mov     %ecx, %eax
        pop     %rbx
        ret
        .size topoml_cpuid_leaf1_ecx, .-topoml_cpuid_leaf1_ecx

        .globl topoml_xgetbv0
        .type topoml_xgetbv0, @function

/*
 * uint64_t topoml_xgetbv0(void)
 *
 * Returns XCR0 when OSXSAVE is available, otherwise zero.
 */
topoml_xgetbv0:
        call    topoml_cpuid_leaf1_ecx
        mov     %eax, %edx
        shr     $27, %edx
        and     $1, %edx
        jz      .xgetbv0_no_osxsave
        xor     %ecx, %ecx
        xgetbv
        shl     $32, %rdx
        or      %rdx, %rax
        ret
.xgetbv0_no_osxsave:
        xor     %eax, %eax
        xor     %edx, %edx
        ret
        .size topoml_xgetbv0, .-topoml_xgetbv0

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

        .globl topoml_has_avx512f_os
        .type topoml_has_avx512f_os, @function

topoml_has_avx512f_os:
        call    topoml_has_avx512f
        test    %eax, %eax
        jz      .avx512f_os_no
        call    topoml_xgetbv0
        and     $0xe6, %eax
        cmp     $0xe6, %eax
        sete    %al
        movzbl  %al, %eax
        ret
.avx512f_os_no:
        xor     %eax, %eax
        ret
        .size topoml_has_avx512f_os, .-topoml_has_avx512f_os
