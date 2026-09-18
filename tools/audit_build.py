#!/usr/bin/env python3
"""Audit source/ABI/target boundaries of the native artifact; no benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from reference_paths import reference_root

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(*args):return subprocess.check_output(args,text=True,cwd=ROOT)
def require(ok,message):
    if not ok:raise SystemExit(message)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify-donors",action="store_true")
    args=parser.parse_args()
    subprocess.run(['python3',str(ROOT/'tools/check_policy_integrity.py')],check=True)
    manifest=json.loads((ROOT/'config/sources.json').read_text())
    entries=manifest['entries'];registered={e['path'] for e in entries}
    actual={str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*') if p.suffix in {'.cpp','.c','.s','.S'}}
    require(actual==registered,'production TU set differs from explicit manifest')
    public=[]
    for p in (ROOT/'include').rglob('*.h'):
        s=p.read_text()
        require(not re.search(r'__m(?:128|256|512)|immintrin|x86intrin|src/backend',s),f'ISA/private header leak: {p}')
        # Audit external declarations, not names mentioned in comments,
        # callback return types such as sbn3_query_result (*bounds)(...), or
        # static inline definitions supplied by the public header itself.
        declarations=re.sub(r'/\*.*?\*/|//[^\n]*','',s,flags=re.S)
        public+=re.findall(r'\b(sbn3_[a-z0-9_]+)\s*\((?!\s*\*)[^;{}]*\)\s*;',declarations)
    archive=ROOT/'build/native/libsbn_v3.a'
    symbols=command('llvm-nm','-g','--defined-only',str(archive))
    require(not re.search(r'\b[TW] sbn_(?!3)',symbols),'legacy symbol in production archive')
    defined={line.split()[-1] for line in symbols.splitlines() if re.search(r'\b[TW]\b',line)}
    require(set(public)<=defined,f'missing public implementations: {set(public)-defined}')
    for p in (ROOT/'src').rglob('*'):
        if p.suffix not in {'.h','.hpp','.cpp','.c'}:continue
        s=re.sub(r'/\*.*?\*/|//[^\n]*','',p.read_text(),flags=re.S)
        require(not re.search(r'\b(?:malloc|calloc|realloc|aligned_alloc|posix_memalign)\s*\(',s),f'ordinary allocation: {p}')
        require(not re.search(r'\bnew\s+(?:[A-Za-z_]|\[)',s),f'ordinary new: {p}')
        if 'backend' in p.parts:require(not re.search(r'\bgetenv\s*\(',s),f'hidden environment choice: {p}')
    objects=[]
    for e in entries:
        obj=ROOT/'build/native'/(e['path']+'.o');receipt=json.loads(Path(str(obj)+'.json').read_text())
        require(receipt['object_sha256']==sha(obj),f'object checksum: {obj}')
        for name,digest in receipt['dependencies'].items():require(sha(Path(name))==digest,f'stale dependency: {name}')
        disassembly=command('llvm-objdump','-d','--no-show-raw-insn',str(obj))
        if e['group']=='common':require(not re.search(r'%zmm|%k[0-7]',disassembly),f'AVX-512 in common TU: {e["path"]}')
        if re.search(r'/np(?:[4-9]|10)\.cpp$',e['path']):
            require('vpmadd52' in disassembly,'native IFMA instance missing')
            undefined=command('llvm-nm','-u',str(obj))
            require(not re.search(r'\b(?:malloc|calloc|realloc|free|mmap|mlock|mprotect|madvise|posix_memalign|aligned_alloc|getenv)\b|\b_Zn[aw]',undefined),'allocation/page operation in native instance')
        objects.append({'path':e['path'],'group':e['group'],'sha256':sha(obj),'dependencies':len(receipt['dependencies'])})
    imports=json.loads((ROOT/'config/provenance/imports.json').read_text())['imports']
    for row in imports:
        require(re.fullmatch(r'[0-9a-f]{64}',row['sha256']), 'invalid donor identity')
        if args.verify_donors:require(sha(reference_root(True)/row['source'])==row['sha256'],f'donor changed: {row["source"]}')
    result={'schema':1,'status':'passed','archive_sha256':sha(archive),'public_symbols':sorted(set(public)),
            'objects':objects,'donor_sources_verified':args.verify_donors,'checks':['explicit TU set','public C declarations implemented','no public ISA types',
            'no legacy production symbols','no ordinary source allocation','no backend environment choices',
            'native object/source hashes','common objects without AVX-512 registers','IFMA instance present',
            'native instance without allocator/page imports','donor hashes unchanged' if args.verify_donors else 'donor identity metadata (external sources not requested)']}
    out=ROOT/'build/native/audit.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(f'OK: {len(objects)} TUs, {len(set(public))} public symbols; audit {out.relative_to(ROOT)}')
if __name__=='__main__':main()
