#pragma once
// Generated from config/native-variants.json; no runtime environment choices.
#ifndef CR_NP
#define CR_NP 6
#endif
#ifndef SBN3_P48_LEAF
#define SBN3_P48_LEAF 8
#endif
#ifndef CR_SLOT
#define CR_SLOT 48
#endif
#ifndef CR_SLOTO
#define CR_SLOTO CR_SLOT
#endif
#ifndef CR_TB
#define CR_TB 2
#endif
#ifndef CR_BPAD
#define CR_BPAD 1
#endif
#ifndef CR_IROW_W
#define CR_IROW_W 4
#endif
#ifndef CR_ROWW
#define CR_ROWW 1
#endif
#ifndef CR_OCH
#define CR_OCH 4096
#endif
#ifndef CR_BLK_MIN
#define CR_BLK_MIN 64
#endif
#ifndef CR_FAST_INV
#define CR_FAST_INV 1
#endif
#ifndef CR_CONV8_SPLIT
#define CR_CONV8_SPLIT 1
#endif
#ifndef CR_M8U2
#define CR_M8U2 1
#endif
#ifndef CR_LEAF8P
#define CR_LEAF8P 1
#endif
#ifndef CR_FLAT_PACKA
#define CR_FLAT_PACKA 2
#endif
#ifndef CR_FLAT_TRIM_ZERO
#define CR_FLAT_TRIM_ZERO 2
#endif
#ifndef CR_FLAT_INVERSE
#define CR_FLAT_INVERSE 0
#endif
#ifndef CR_FLAT_PRIME_SPREAD
#define CR_FLAT_PRIME_SPREAD 1
#endif
#ifndef CR_FLAT_TAIL_PAIR
#define CR_FLAT_TAIL_PAIR 2
#endif
#ifndef CR_PRMF
#define CR_PRMF 1
#endif
#ifndef CR_FPACK
#define CR_FPACK 0
#endif
#ifndef CR_GARNER_DOT
#define CR_GARNER_DOT 0
#endif
#ifndef SBN3_EXPERIMENTAL_NATIVE
#if CR_NP < 4 || CR_NP > 10
#error "CR_NP override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if SBN3_P48_LEAF != 8
#error "SBN3_P48_LEAF override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_SLOT != 48
#error "CR_SLOT override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_SLOTO != 48
#error "CR_SLOTO override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_TB != 2
#error "CR_TB override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_BPAD != 1
#error "CR_BPAD override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_IROW_W != 4
#error "CR_IROW_W override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_ROWW != 1
#error "CR_ROWW override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_OCH != 4096
#error "CR_OCH override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_BLK_MIN != 64
#error "CR_BLK_MIN override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_FAST_INV != 1
#error "CR_FAST_INV override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_CONV8_SPLIT != 1
#error "CR_CONV8_SPLIT override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_M8U2 != 1
#error "CR_M8U2 override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_LEAF8P != 1
#error "CR_LEAF8P override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_FLAT_PACKA != 2
#error "CR_FLAT_PACKA override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_FLAT_TRIM_ZERO != 2
#error "CR_FLAT_TRIM_ZERO override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_FLAT_INVERSE != 0
#error "CR_FLAT_INVERSE override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_FLAT_PRIME_SPREAD != 1
#error "CR_FLAT_PRIME_SPREAD override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_FLAT_TAIL_PAIR != 2
#error "CR_FLAT_TAIL_PAIR override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_PRMF != 1
#error "CR_PRMF override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_FPACK != 0
#error "CR_FPACK override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if CR_GARNER_DOT != 0
#error "CR_GARNER_DOT override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if defined(CR_STRIDE_OVERRIDE)
#error "CR_STRIDE_OVERRIDE override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if defined(CR_FLAT_SHARE_TABLES)
#error "CR_FLAT_SHARE_TABLES override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if defined(CR_EMIT_PROF)
#error "CR_EMIT_PROF override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if defined(CR_FLAT_PROFILE)
#error "CR_FLAT_PROFILE override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if defined(CR_P48_CHECK)
#error "CR_P48_CHECK override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if defined(CR_ROWS_PROF)
#error "CR_ROWS_PROF override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#if defined(CR_TILE_PROF)
#error "CR_TILE_PROF override requires SBN3_EXPERIMENTAL_NATIVE"
#endif
#endif
