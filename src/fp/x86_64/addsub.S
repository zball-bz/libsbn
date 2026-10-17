/* fp kernel add / sub, x86-64 (Zen 5, AVX-512). The algorithm and the region names (R1 / R2 / R3) are
   those of src/fp/addsub.cpp; the contract is kernel.hpp's.

   fxr sbn3f_add(limb *z, const fx *a, const fx *b, int64_t leps)      rdi z, rsi a, rdx b, rcx leps
   fxr sbn3f_sub(limb *z, const fx *a, const fx *b, int64_t leps)      result: rax e, rdx ns

   Dispatch: e - ns (= the grid minus the sign word) is equal for both operands iff the effective signs
   and the grids are equal: the fast add path. Equal grids with opposite signs: the fast sub path. Equal
   grids with truncation stay on the fast paths (both windows advance t limbs).
   Fast paths: d2 = min(na, nb), d3s = na - nb. The general paths (different grids) order the operands so
   that A's window starts lowest and work with lengths: d1 = hi0 - lo (R1, A only), t = nA - d1 (A's limbs
   from hi0 up; t <= 0: disjoint windows), d2 = min(t, nB), d3s = t - nB, n_top = d1 + d2 + |d3s|.
   Truncation (after ordering ga <= gb, L > ga): A's window starts at L, B's at max(gb, L); both join there.
   Cores: R2 of up to 16 limbs is one computed entry into a straight-line adc / sbb chain indexed by r11 =
   d2 (every step 15 bytes, the entry end - 15 d2; 8-byte memory operands only: n <= 8 never touches a
   vector memory form). R2 of 17 limbs and more: d2 / 8 blocks of 8 limbs from the overlap start by the
   AVX-512 carry stream (8 lanes per block, the carry chain in a GPR: T = p + 2g, t = T + c, c' = t >> 8,
   C = t ^ p), then the remaining d2 mod 8 limbs by the same chain. Every vector store is a disjoint
   64-byte block (aligned when the operands are): no overlap, no mask. R3 of up to 8 limbs: a second
   indexed chain (adc $0 / sbb $0, 14 bytes a step); a longer R3 ripples, then copies. R1 copies of up to
   16 limbs: an indexed copy chain (10 bytes a step).
   Leaf routine: caller-saved registers and the red zone (-8 .. -40 below rsp, general paths only);
   zmm0-3, k1-k2; vzeroupper on every path that used a vector register. */

#ifndef FPK_PFX
#define FPK_PFX sbn3f_
#endif
#define FPK_CAT2(a, b) a##b
#define FPK_CAT(a, b) FPK_CAT2(a, b)
#define SYM(x) FPK_CAT(FPK_PFX, x)

/* red-zone slots */
#define S_SIGN  -8(%rsp)
#define S_ETOP  -16(%rsp)
#define S_NS    -24(%rsp)
#define S_BIN   -32(%rsp)
#define S_D3    -40(%rsp)

/* ---- macros ------------------------------------------------------------------------------------ */

/* one 8-lane block: z = a op b plus the lane carries; r8 = carry in / out (0/1); r9 = C (bit j: lane j
   receives a carry, bit 8: carry out); clobbers r10, zmm0-2, k1, k2; zmm3 = all ones */
.macro BLOCK op, ma, mb, mz
	vmovdqu64 \ma, %zmm0
	vmovdqu64 \mb, %zmm2
.ifc \op,add
	vpaddq %zmm2, %zmm0, %zmm1
	vpcmpuq $1, %zmm0, %zmm1, %k1       /* generate: S < A */
	vpcmpeqq %zmm3, %zmm1, %k2          /* propagate: S == ~0 */
.else
	vpsubq %zmm2, %zmm0, %zmm1
	vpcmpuq $1, %zmm2, %zmm0, %k1       /* generate: A < B */
	vptestnmq %zmm1, %zmm1, %k2         /* propagate: S == 0 */
.endif
	kmovb %k1, %r9d
	kmovb %k2, %r10d
	lea (%r10,%r9,2), %r9
	add %r8, %r9
	mov %r9, %r8
	shr $8, %r8
	xor %r10, %r9
	kmovb %r9d, %k1
.ifc \op,add
	vpsubq %zmm3, %zmm1, %zmm1{%k1}
.else
	vpaddq %zmm3, %zmm1, %zmm1{%k1}
.endif
	vmovdqu64 %zmm1, \mz
.endm

/* q >= 1 blocks of 8 limbs: rsi A, rdx B, rdi Z at the first block, rcx = q, r8 = carry in (0/1).
   Out: r8 = carry out, rsi / rdx / rdi past the last block. Clobbers rcx, r9, r10, zmm0-3, k1, k2. */
.macro VBLOCKS op
	shl $3, %rcx
	lea (%rsi,%rcx,8), %rsi
	lea (%rdx,%rcx,8), %rdx
	lea (%rdi,%rcx,8), %rdi
	neg %rcx
	vpternlogq $0xff, %zmm3, %zmm3, %zmm3
.Lvb\@:
	BLOCK \op, "(%rsi,%rcx,8)", "(%rdx,%rcx,8)", "(%rdi,%rcx,8)"
	add $8, %rcx
	jnz .Lvb\@
	vzeroupper
.endm

/* copy len >= 0 limbs src -> dst; src and dst advance to the end; len and tmp are clobbered.
   len >= 9: a scalar head of len mod 8 limbs, then disjoint 64-byte blocks. */
.macro COPY dst, src, len, tmp
	cmp $9, \len
	jb .Lcs\@
	test $7, \len
	jz .Lcb\@
.Lch\@:
	mov (\src), \tmp
	mov \tmp, (\dst)
	add $8, \src
	add $8, \dst
	dec \len
	test $7, \len
	jnz .Lch\@
.Lcb\@:
	vmovdqu64 (\src), %zmm0
	vmovdqu64 %zmm0, (\dst)
	add $64, \src
	add $64, \dst
	sub $8, \len
	jnz .Lcb\@
	vzeroupper
	jmp .Lce\@
.Lcs\@:
	test \len, \len
	jz .Lce\@
.Lcl\@:
	mov (\src), \tmp
	mov \tmp, (\dst)
	add $8, \src
	add $8, \dst
	dec \len
	jnz .Lcl\@
.Lce\@:
.endm

/* store len >= 0 copies of val (a GPR) at dst; dst advances to the end; len clobbered */
.macro FILL dst, len, val
	cmp $9, \len
	jb .Lfs\@
	test $7, \len
	jz .Lfb\@
.Lfh\@:
	mov \val, (\dst)
	add $8, \dst
	dec \len
	test $7, \len
	jnz .Lfh\@
.Lfb\@:
	vpbroadcastq \val, %zmm0
.Lfv\@:
	vmovdqu64 %zmm0, (\dst)
	add $64, \dst
	sub $8, \len
	jnz .Lfv\@
	vzeroupper
	jmp .Lfe\@
.Lfs\@:
	test \len, \len
	jz .Lfe\@
.Lfl\@:
	mov \val, (\dst)
	add $8, \dst
	dec \len
	jnz .Lfl\@
.Lfe\@:
.endm

/* dst = -src over len >= 1 limbs; bout = the borrow out (src != 0); src, dst advance */
.macro NEG dst, src, len, tmp, bout
	clc
.Lng\@:
	mov $0, \tmp
	sbb (\src), \tmp
	mov \tmp, (\dst)
	lea 8(\src), \src
	lea 8(\dst), \dst
	dec \len
	jnz .Lng\@
	mov $0, \bout
	adc \bout, \bout
.endm

/* the computed-entry chains. Step k works on the limb 8 k bytes below base + 8 index; every step of a
   chain has the same length (disp8 down to -128), so the entry for d limbs is end - len d. */
.macro CHAIN2 insn                      /* R2: rsi, rdx, rdi; index r11; 15 bytes */
.irp k, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
	mov -8*\k(%rsi,%r11,8), %rax
	\insn -8*\k(%rdx,%r11,8), %rax
	mov %rax, -8*\k(%rdi,%r11,8)
.endr
.endm
.macro CHAIN3 insn                      /* R3: H rdx, Z rdi; index r11; 14 bytes */
.irp k, 8, 7, 6, 5, 4, 3, 2, 1
	mov -8*\k(%rdx,%r11,8), %rax
	\insn $0, %rax
	mov %rax, -8*\k(%rdi,%r11,8)
.endr
.endm
.macro CHAINCP                          /* R1 copy: rsi -> rdi; index rcx; data r11; 10 bytes */
.irp k, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
	mov -8*\k(%rsi,%rcx,8), %r11
	mov %r11, -8*\k(%rdi,%rcx,8)
.endr
.endm

/* ---- entries ------------------------------------------------------------------------------------ */

	.text
	.p2align 5
	.globl SYM(sub)
	.type SYM(sub), @function
SYM(sub):
	mov 16(%rsi), %r10
	mov 16(%rdx), %rax
	btc $63, %rax
	jmp .Lbody
	.size SYM(sub), .-SYM(sub)

	.p2align 5
	.globl SYM(add)
	.type SYM(add), @function
SYM(add):
	mov 16(%rsi), %r10                  /* ans */
	mov 16(%rdx), %rax                  /* bns (sub: the sign flipped) */
.Lbody:
	mov 8(%rsi), %r9                    /* ea */
	mov 8(%rdx), %r11                   /* eb */
	mov (%rsi), %rsi                    /* ad */
	mov (%rdx), %rdx                    /* bd */
	sub %r10, %r9                       /* ea - ans = ga - sa (sa the sign word) */
	sub %rax, %r11                      /* gb - sb */
	cmp %r11, %r9
	jne .Lnotfast                       /* equal iff the same sign and the same grid */
	mov %r10, %r8
	btr $63, %r8                        /* na */
	sub %r8, %r10                       /* the sign word */
	xor %r10, %r9                       /* ga */
	btr $63, %rax                       /* nb */
	dec %rcx
	sar $6, %rcx                        /* L1 = floor((leps - 1) / 64) */
	cmp %r9, %rcx
	jg .Lag_trunc
.Lafast_cont:
	mov %r8, %rcx
	sub %rax, %rcx                      /* d3s = na - nb; CF: na < nb */
	mov %r8, %r11
	cmovb %rax, %r11                    /* nmax */
	cmovb %r8, %rax                     /* d2 = nmin */
	add %r11, %r9                       /* e_top = ga + nmax */
	or %r11, %r10                       /* ns_top = sign | nmax */

/* add core: rsi A, rdx B, rdi Z (at the overlap start), rax d2 >= 1, rcx d3s (> 0: A continues above
   mid, < 0: B does, |d3s| limbs), r9 e_top, r10 ns_top (without a carry) */
.Lacore:
	cmp $16, %rax
	ja .Lacore_long
	mov %rax, %r11
	imul $15, %rax, %rax
	lea .Lac2_end(%rip), %r8
	sub %rax, %r8                       /* CF = 0 */
	jmp *%r8
.Lac2_beg:
	CHAIN2 adc
.Lac2_end:
	mov $0, %eax
	adc %eax, %eax                      /* the carry */
	test %rcx, %rcx
	jnz .Lacore_r3
.Lacore_fin:                            /* rax carry, rdi + 8 r11 = Z at top */
	mov %rax, (%rdi,%r11,8)
	lea (%r10,%rax), %rdx
	add %r9, %rax
	ret

.Lacore_r3:                             /* rax carry; rsi, rdx, rdi at the overlap start, r11 = d2; rcx d3s */
	lea (%rdi,%r11,8), %rdi             /* Z at mid */
	test %rcx, %rcx
	js 1f
	lea (%rsi,%r11,8), %rdx             /* H = A at mid */
	jmp 2f
1:	lea (%rdx,%r11,8), %rdx             /* H = B at mid */
	neg %rcx
2:	mov %rcx, %r11                      /* d3 */
	cmp $1, %rcx
	jne 3f
	mov (%rdx), %rsi                    /* R3 of one limb */
	add %rax, %rsi
	mov %rsi, (%rdi)
	mov $0, %eax
	adc %eax, %eax
	jmp .Lacore_fin
3:	cmp $8, %rcx
	ja .Lacore_r3long
	imul $14, %rcx, %rcx
	lea .Lac3_end(%rip), %rsi
	sub %rcx, %rsi
	bt $0, %eax
	jmp *%rsi
.Lac3_beg:
	CHAIN3 adc
.Lac3_end:
	mov $0, %eax
	adc %eax, %eax
	jmp .Lacore_fin

.Lacore_long:                           /* d2 >= 17: d2 / 8 blocks, then the tail of d2 mod 8 by the chain */
	mov %r9, S_ETOP
	mov %r10, S_NS
	mov %rcx, S_D3
	mov %rax, %rcx
	shr $3, %rcx                        /* q >= 2 */
	and $7, %eax
	mov %rax, %r11                      /* the tail index (the blocks keep r11) */
	xor %r8d, %r8d
	VBLOCKS add
	mov %r8, %rcx                       /* the carry */
	imul $15, %r11, %rax
	lea .Lac2_end(%rip), %r8
	sub %rax, %r8
	bt $0, %ecx                         /* CF = the carry out of the blocks */
	mov S_ETOP, %r9
	mov S_NS, %r10
	mov S_D3, %rcx
	jmp *%r8

.Lacore_r3long:                         /* d3 >= 9: rdx H at mid, rdi Z at mid, rcx d3, rax carry */
	test %eax, %eax
	jz 2f
1:	mov (%rdx), %rsi
	add $1, %rsi
	mov %rsi, (%rdi)
	lea 8(%rdx), %rdx
	lea 8(%rdi), %rdi
	jnc 3f
	dec %rcx
	jnz 1b
	xor %r11d, %r11d                    /* rax = 1: the carry left the top */
	jmp .Lacore_fin
3:	dec %rcx
	xor %eax, %eax
2:	COPY %rdi, %rdx, %rcx, %rsi
	xor %eax, %eax
	xor %r11d, %r11d
	jmp .Lacore_fin

/* not the fast add path: r10 ans, rax bns, r9 = ga - sa, r11 = gb - sb, rcx leps */
.Lnotfast:
	mov %r10, %r8
	xor %rax, %r8
	js .Lopp
	mov %r10, %r8                       /* same sign, different grids */
	btr $63, %r8                        /* na */
	sub %r8, %r10                       /* the sign word */
	xor %r10, %r9                       /* ga */
	xor %r10, %r11                      /* gb */
	btr $63, %rax                       /* nb */
	dec %rcx
	sar $6, %rcx                        /* L1 */

/* same sign, different grids: r10 sign word, r8 na, rax nb, r9 ga, r11 gb, rsi ad, rdx bd, rcx L1, rdi z */
.Lag:
	cmp %r11, %r9
	jle 1f
	xchg %rsi, %rdx                     /* A = the lower grid */
	xchg %r9, %r11
	xchg %r8, %rax
1:	cmp %r9, %rcx
	jg .Lag_full                        /* truncation */
/* rsi A (from lo), rdx B (from hi0 >= lo), r9 lo, r11 hi0, r8 nA, rax nB, r10 sign word, rdi z */
.Lag_w:
	mov %r11, %rcx
	sub %r9, %rcx                       /* d1 = hi0 - lo */
	sub %rcx, %r8                       /* t = nA - d1 */
	jle .Lag_disj                       /* A entirely below B */
	mov %r8, %r11
	sub %rax, %r11                      /* d3s = t - nB */
	cmovl %r8, %rax                     /* d2 = min(t, nB) */
	mov %r11, %r8
	neg %r8
	cmovs %r11, %r8                     /* |d3s| */
	add %rax, %r8
	add %rcx, %r8                       /* n_top = d1 + d2 + |d3s| */
	add %r8, %r9                        /* e_top */
	or %r8, %r10                        /* ns_top */
	test %rcx, %rcx
	jz 2f
	cmp $1, %rcx
	jne 4f
	mov (%rsi), %r8                     /* R1 of one limb */
	mov %r8, (%rdi)
	add $8, %rsi
	add $8, %rdi
	mov %r11, %rcx
	jmp .Lacore
4:	mov %r11, S_D3
	cmp $16, %rcx
	ja 3f
	imul $10, %rcx, %r8
	lea .Lacp_end(%rip), %r11
	sub %r8, %r11
	jmp *%r11
.Lacp_beg:
	CHAINCP
.Lacp_end:
	lea (%rsi,%rcx,8), %rsi
	lea (%rdi,%rcx,8), %rdi
	mov S_D3, %rcx
	jmp .Lacore
3:	COPY %rdi, %rsi, %rcx, %r11         /* R1 of more than 16 limbs */
	mov S_D3, %rcx
	jmp .Lacore
2:	mov %r11, %rcx
	jmp .Lacore

/* same grid with truncation (from the fast path): r10 sign word, r8 na, rax nb, r9 ga, rcx L1 > ga */
.Lag_trunc:
	mov %rcx, %r11
	sub %r9, %r11                       /* t = L1 - ga limbs dropped from both */
	cmp %r11, %r8
	jle 1f
	cmp %r11, %rax
	jle 1f
	sub %r11, %r8
	sub %r11, %rax
	lea (%rsi,%r11,8), %rsi
	lea (%rdx,%r11,8), %rdx
	mov %rcx, %r9                       /* the grid is now L1 */
	jmp .Lafast_cont
1:	mov %r9, %r11                       /* a window empties: the general path (gb = ga) */

/* truncation (ga <= gb, L > ga; also a same-grid window that empties): r10 sign word, r8 na, rax nb,
   r9 ga, r11 gb, rsi ad, rdx bd, rcx L1. A's window starts at L; B's at max(gb, L). */
.Lag_full:
	mov %r10, S_SIGN
	mov %rcx, %r10
	sub %r9, %r10                       /* tA = L - ga > 0 */
	sub %r10, %r8                       /* nA */
	jle .Lag_aneg                       /* A negligible */
	lea (%rsi,%r10,8), %rsi
	mov %rcx, %r10
	sub %r11, %r10                      /* L - gb */
	jle 1f
	sub %r10, %rax                      /* nB */
	jle .Lag_bneg                       /* B negligible */
	lea (%rdx,%r10,8), %rdx
	mov %rcx, %r11                      /* hi0 = L */
1:	mov %rcx, %r9                       /* lo = L */
	mov S_SIGN, %r10
	jmp .Lag_w

/* A negligible: the result is B's window from max(gb, L) with the sign in S_SIGN. r11 gb, rax nb,
   rdx bd, rcx L */
.Lag_aneg:
	mov %rcx, %r10
	sub %r11, %r10                      /* L - gb */
	jle 1f
	sub %r10, %rax                      /* nB */
	jle .Lzero
	lea (%rdx,%r10,8), %rdx
	mov %rcx, %r11
1:	lea (%r11,%rax), %r8                /* e */
	mov %rax, %r9
	or S_SIGN, %r9                      /* ns */
	COPY %rdi, %rdx, %rax, %r11
	mov %r8, %rax
	mov %r9, %rdx
	ret
/* B negligible: A's window rsi, r8 = nA limbs from L = rcx, the sign in S_SIGN */
.Lag_bneg:
	lea (%rcx,%r8), %r9                 /* e */
	mov %r8, %r10
	or S_SIGN, %r10                     /* ns */
	COPY %rdi, %rsi, %r8, %r11
	mov %r9, %rax
	mov %r10, %rdx
	ret

.Lag_disj:                              /* A entirely below B: A, zeros, B. r9 lo, rcx d1, r8 t <= 0, rax nB */
	lea (%r9,%rcx), %r11
	add %rax, %r11
	mov %r11, S_ETOP                    /* e = hi0 + nB */
	lea (%rcx,%rax), %r11
	or %r10, %r11
	mov %r11, S_NS                      /* n = d1 + nB */
	mov %r8, %r9
	neg %r9                             /* the gap */
	add %rcx, %r8                       /* nA */
	COPY %rdi, %rsi, %r8, %r11
	xor %r11d, %r11d
	FILL %rdi, %r9, %r11
	COPY %rdi, %rdx, %rax, %r11
	mov S_ETOP, %rax
	mov S_NS, %rdx
	ret

.Lzero:
	xor %eax, %eax
	xor %edx, %edx
	ret

/* ---- opposite effective signs ------------------------------------------------------------------- */
.Lopp:                                  /* r10 ans, rax bns, r9 = ga - sa, r11 = gb - sb, rcx leps */
	mov %r10, %r8
	btr $63, %r8                        /* na */
	sub %r8, %r10                       /* sign word of a */
	xor %r10, %r9                       /* ga */
	xor %r10, %r11
	btc $63, %r11                       /* gb (b's sign is the other one) */
	btr $63, %rax                       /* nb */
	sar $6, %rcx                        /* L0 = floor(leps / 64) */
	cmp %r11, %r9
	jne .Lsg                            /* different grids */
	cmp %r9, %rcx
	jg .Lsg_trunc
.Lsfast_cont:
	cmp %rax, %r8
	jne .Ls_ord
	mov -8(%rsi,%r8,8), %rcx
	cmp -8(%rdx,%r8,8), %rcx
	je .Ls_eqtop
.Ls_ord:                                /* CF: b is the bigger */
	mov %rsi, %rcx
	cmovb %rdx, %rsi                    /* big */
	cmovb %rcx, %rdx                    /* small */
	mov %r8, %rcx
	cmovb %rax, %r8                     /* nmax */
	cmovb %rcx, %rax                    /* d2 = nmin */
	sbb %rcx, %rcx
	shl $63, %rcx
	xor %rcx, %r10                      /* sign of big */
	add %r8, %r9                        /* e_top */
	or %r8, %r10                        /* ns_top */
	mov %r8, %rcx
	sub %rax, %rcx                      /* d3 */

/* sub core: rsi big, rdx small, rdi Z (at the overlap start), rax d2 >= 1, rcx d3 >= 0 (big continues
   above mid), r9 e_top, r10 ns_top; borrow in 0 (.Lscore) or S_BIN (.Lscore_bin) */
.Lscore:
	cmp $16, %rax
	ja .Lscore_long
	mov %rax, %r11
	imul $15, %rax, %rax
	lea .Lsc2_end(%rip), %r8
	sub %rax, %r8                       /* CF = 0 */
	jmp *%r8
.Lscore_bin:
	cmp $16, %rax
	ja .Lscore_long_binm
	mov %rax, %r11
	imul $15, %rax, %rax
	lea .Lsc2_end(%rip), %r8
	sub %rax, %r8
	btq $0, S_BIN
	jmp *%r8
.Lsc2_beg:
	CHAIN2 sbb
.Lsc2_end:                              /* CF borrow (0 unless d3 > 0); rax the top limb of R2 */
	sbb %r8, %r8                        /* -borrow */
	test %rcx, %rcx
	jnz .Lscore_r3
	test %rax, %rax
	jz .Lsub_strip
.Lscore_fin:
	mov %r9, %rax
	mov %r10, %rdx
	ret

.Lscore_r3:                             /* r8 -borrow; rsi big, rdi at the overlap start, r11 = d2; rcx d3 */
	lea (%rdi,%r11,8), %rdi             /* Z at mid */
	lea (%rsi,%r11,8), %rdx             /* H = big at mid */
	mov %rcx, %r11
	cmp $1, %rcx
	jne 1f
	mov (%rdx), %rax                    /* R3 of one limb: H - borrow */
	add %r8, %rax
	mov %rax, (%rdi)
	jmp .Lsc3_end
1:	cmp $8, %rcx
	ja .Lscore_r3long
	imul $14, %rcx, %rcx
	lea .Lsc3_end(%rip), %rsi
	sub %rcx, %rsi
	neg %r8                             /* CF = borrow */
	jmp *%rsi
.Lsc3_beg:
	CHAIN3 sbb
.Lsc3_end:
	test %rax, %rax
	jnz .Lscore_fin
.Lsub_strip:                            /* rdi + 8 r11: one past the top limb, which is zero */
	lea -8(%rdi,%r11,8), %rdi
1:	dec %r9
	dec %r10
	sub $8, %rdi
	cmpq $0, (%rdi)
	je 1b
	jmp .Lscore_fin

.Lscore_long_binm:
	mov S_BIN, %r8
	jmp .Lscore_long_bin
.Lscore_long:
	xor %r8d, %r8d
.Lscore_long_bin:                       /* d2 >= 17, r8 = borrow in: blocks, then the tail by the chain */
	mov %r9, S_ETOP
	mov %r10, S_NS
	mov %rcx, S_D3
	mov %rax, %rcx
	shr $3, %rcx
	and $7, %eax
	mov %rax, %r11
	VBLOCKS sub
	mov %r8, %rcx                       /* the borrow */
	imul $15, %r11, %rax
	lea .Lsc2_end(%rip), %r8
	sub %rax, %r8
	mov -8(%rdi), %rax                  /* the top limb so far (an empty tail leaves it in rax) */
	bt $0, %ecx                         /* CF = the borrow out of the blocks */
	mov S_ETOP, %r9
	mov S_NS, %r10
	mov S_D3, %rcx
	jmp *%r8

.Lscore_r3long:                         /* d3 >= 9: rdx big at mid, rdi Z at mid, rcx d3, r8 -borrow */
	test %r8, %r8
	jz 2f
1:	mov (%rdx), %rsi
	sub $1, %rsi
	mov %rsi, (%rdi)
	lea 8(%rdx), %rdx
	lea 8(%rdi), %rdi
	jnc 3f
	dec %rcx
	jnz 1b
	jmp 4f
3:	dec %rcx
2:	COPY %rdi, %rdx, %rcx, %rsi
4:	xor %r11d, %r11d
	cmpq $0, -8(%rdi)
	je .Lsub_strip
	jmp .Lscore_fin

.Ls_eqtop:                              /* equal lengths and equal tops: compare downwards */
	lea -1(%r8), %rcx
1:	sub $1, %rcx
	jb .Lzero
	mov (%rsi,%rcx,8), %r11
	cmp (%rdx,%rcx,8), %r11
	je 1b
	jmp .Ls_ord

/* opposite signs, different grids: r10 sign word of a, r8 na, rax nb, r9 ga, r11 gb, rsi ad, rdx bd,
   rcx L0, rdi z */
.Lsg:
	cmp %r11, %r9
	jle 1f
	xchg %rsi, %rdx                     /* A = the lower grid */
	xchg %r9, %r11
	xchg %r8, %rax
	btc $63, %r10                       /* the sign of the new A */
1:	cmp %r9, %rcx
	jg .Lsg_full                        /* truncation */
/* rsi A (from lo), rdx B (from hi0 >= lo), r9 lo, r11 hi0, r8 nA, rax nB, r10 sign of A, rdi z */
.Lsg_w:
	mov %r10, S_SIGN
	mov %r11, %rcx
	sub %r9, %rcx                       /* d1 */
	sub %rcx, %r8                       /* t = nA - d1 */
	jle .Lsg_disj                       /* A entirely below B: B is the bigger */
	mov %r8, %r11
	sub %rax, %r11                      /* d3s = t - nB */
	cmovl %r8, %rax                     /* d2 = min(t, nB) */
	mov %r11, %r8
	neg %r8
	cmovs %r11, %r8                     /* d3 = |d3s| */
	lea (%rcx,%rax), %r10
	add %r8, %r10                       /* n_top */
	add %r10, %r9
	mov %r9, S_ETOP                     /* e_top = lo + n_top */
	mov %r10, S_NS                      /* n_top; the sign comes with the order */
	mov %r8, %r9                        /* d3 */
	xor %r10d, %r10d                    /* r10: -1 when B is the bigger */
	test %r11, %r11
	jg 8f                               /* A's top is higher */
	jl 5f                               /* B's top is higher */
	lea (%rsi,%rcx,8), %r11             /* equal tops: A at hi0, compare the overlap downwards */
	mov %rax, %r8
4:	sub $1, %r8
	jb 7f
	mov (%r11,%r8,8), %r10
	cmp (%rdx,%r8,8), %r10
	je 4b
	sbb %r10, %r10                      /* CF: A < B */
	jmp 8f
7:	mov %rcx, %r8                       /* an equal overlap: A's R1 decides */
6:	sub $1, %r8
	jb .Lzero
	cmpq $0, (%rsi,%r8,8)
	je 6b
	xor %r10d, %r10d
	jmp 8f
5:	mov $-1, %r10
8:	xor %r8d, %r8d                      /* R1 into z; r8: the borrow into R2 */
	test %rcx, %rcx
	jz 3f
	cmp $1, %rcx
	jne 1f
	mov (%rsi), %r11                    /* R1 of one limb: A, or -A with its borrow */
	test %r10, %r10
	jz 9f
	neg %r11
	setnz %r8b
9:	mov %r11, (%rdi)
	add $8, %rsi
	add $8, %rdi
	jmp 3f
1:	test %r10, %r10
	jnz 2f
	COPY %rdi, %rsi, %rcx, %r11         /* R1, A the bigger */
	jmp 3f
2:	NEG %rdi, %rsi, %rcx, %r11, %r8     /* R1, A the smaller */
3:	mov %r8, S_BIN                      /* rsi A at hi0, rdx B at hi0, rdi z at hi0, rax d2, r9 d3 */
	mov %rsi, %r11
	test %r10, %r10
	cmovnz %rdx, %rsi                   /* big */
	cmovnz %r11, %rdx                   /* small */
	shl $63, %r10
	xor S_SIGN, %r10
	or S_NS, %r10                       /* ns_top: the sign of the bigger | n_top */
	mov %r9, %rcx                       /* d3 */
	mov S_ETOP, %r9
	jmp .Lscore_bin

/* same grid with truncation (from the fast path): r10 sign word of a, r8 na, rax nb, r9 ga, rcx L0 > ga */
.Lsg_trunc:
	mov %rcx, %r11
	sub %r9, %r11                       /* t */
	cmp %r11, %r8
	jle 1f
	cmp %r11, %rax
	jle 1f
	sub %r11, %r8
	sub %r11, %rax
	lea (%rsi,%r11,8), %rsi
	lea (%rdx,%r11,8), %rdx
	mov %rcx, %r9
	jmp .Lsfast_cont
1:	mov %r9, %r11                       /* a window empties: the general path (gb = ga) */

/* truncation (ga <= gb, L > ga; also a same-grid window that empties): r10 sign of A, r8 nA, rax nB,
   r9 ga, r11 gb, rsi ad, rdx bd, rcx L0 */
.Lsg_full:
	mov %r10, S_SIGN
	mov %rcx, %r10
	sub %r9, %r10                       /* tA */
	sub %r10, %r8                       /* nA */
	jle 2f                              /* A negligible: -B */
	lea (%rsi,%r10,8), %rsi
	mov %rcx, %r10
	sub %r11, %r10                      /* L - gb */
	jle 1f
	sub %r10, %rax                      /* nB */
	jle .Lag_bneg                       /* B negligible: A */
	lea (%rdx,%r10,8), %rdx
	mov %rcx, %r11
1:	mov %rcx, %r9
	mov S_SIGN, %r10
	jmp .Lsg_w
2:	btcq $63, S_SIGN
	jmp .Lag_aneg

.Lsg_disj:                              /* A entirely below B: -A, ones, B - 1. r9 lo, rcx d1, r8 t, rax nB */
	lea (%r9,%rcx), %r11
	add %rax, %r11
	mov %r11, S_ETOP                    /* e = hi0 + nB */
	lea (%rcx,%rax), %r11
	mov S_SIGN, %r10
	btc $63, %r10
	or %r10, %r11
	mov %r11, S_NS                      /* the sign of B | d1 + nB */
	mov %r8, %r9
	neg %r9                             /* the gap */
	add %rcx, %r8                       /* nA >= 1 */
	NEG %rdi, %rsi, %r8, %r11, %r10     /* the borrow out is 1: A's top limb is nonzero */
	mov $-1, %r11
	FILL %rdi, %r9, %r11
	mov %rax, %rcx                      /* nB >= 1 */
1:	mov (%rdx), %rax
	sub $1, %rax
	mov %rax, (%rdi)
	lea 8(%rdx), %rdx
	lea 8(%rdi), %rdi
	jnc 2f
	dec %rcx
	jnz 1b
	jmp 3f
2:	dec %rcx
	COPY %rdi, %rdx, %rcx, %r11
3:	mov S_ETOP, %r9
	mov S_NS, %r10
	xor %r11d, %r11d
	cmpq $0, -8(%rdi)
	je .Lsub_strip
	jmp .Lscore_fin
	.size SYM(add), .-SYM(add)

	.section .note.GNU-stack,"",@progbits
