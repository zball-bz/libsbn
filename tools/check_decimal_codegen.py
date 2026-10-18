#!/usr/bin/env python3
"""Audit the optional public-header decimal inline consumer, not a kernel-only probe."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,default=ROOT/'build/native/decimal_inline_test')
    parser.add_argument('--object',type=Path,default=ROOT/'build/native/src/kernels/x86v4ifma/decimal_tiny.cpp.o')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    names=['decimal_inline_roundtrip64','decimal_inline_roundtrip128','decimal_inline_u8',
           'decimal_inline_fraction64','decimal_inline_fraction128']
    assembly=subprocess.check_output(['llvm-objdump','-d','--no-show-raw-insn',
        '--disassemble-symbols='+','.join(names),str(args.binary)],text=True)
    for name in names:
        if not re.search(r'<'+name+r'>:',assembly):raise SystemExit('missing audited symbol: '+name)
    if re.search(r'\bcall\w*\b[^\n]*sbn3_decimal_',assembly):raise SystemExit('decimal API call survived native LTO')
    if 'vpmadd52' not in assembly:raise SystemExit('expected live IFMA arithmetic absent from audit consumer')
    undefined=subprocess.check_output(['llvm-nm','-u',str(args.object)],text=True)
    if re.search(r'__udivti3|__divti3|__multi3|malloc|calloc|realloc|mmap|mlock|_Zn',undefined):
        raise SystemExit('runtime arithmetic/allocation helper in decimal object')
    plain=subprocess.check_output(['llvm-objdump','-d','--no-show-raw-insn',str(args.object)],text=True)
    if re.search(r'\b(?:idiv|div)[bwlq]?\s',plain):raise SystemExit('hardware divide in decimal object')
    result={'status':'passed','binary':str(args.binary),'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            'object':str(args.object),'object_sha256':hashlib.sha256(args.object.read_bytes()).hexdigest(),
            'symbols':names,'checks':['public-header native LTO removes decimal calls',
                'arithmetic remains live','no runtime arithmetic/allocation helper','no hardware division']}
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('decimal codegen: public-header native LTO, no decimal calls or divide/allocation helpers PASS')
if __name__=='__main__':main()
