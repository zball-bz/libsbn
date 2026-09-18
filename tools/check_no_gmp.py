#!/usr/bin/env python3
"""Audit actual build dependencies and ELF symbols, including static linkage."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def audit(receipts):
    binaries=set();units=0
    for path in receipts:
        x=json.loads(path.read_text());binaries.add(ROOT/x['archive']['path'])
        for u in x['units']:
            units+=1
            assert not any(Path(p).name in ('gmp.h','gmpxx.h') for p in u['dependencies']),u['source']
            assert not any(a.startswith('-lgmp') or 'libgmp' in a for a in u['command']),u['source']
        for b in x.get('test_binaries',[])+x.get('benchmarks',[])+x.get('examples',[]):binaries.add(ROOT/b['path'])
    rows=[]
    for path in sorted(binaries):
        symbols=subprocess.check_output(['llvm-nm',str(path)],text=True,stderr=subprocess.DEVNULL)
        assert not re.search(r'\b_*(?:gmp|GMP)[a-zA-Z0-9_]*',symbols),path
        if path.suffix!='.a':
            needed=subprocess.check_output(['llvm-readelf','-d',str(path)],text=True)
            assert 'libgmp' not in needed,path
        rows.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'schema':1,'status':'passed','units':units,'binaries':rows,'checks':['no GMP headers in dependency closure','no GMP compiler/link flags','no static or dynamic GMP symbols','no libgmp DT_NEEDED']}
def main():
    p=argparse.ArgumentParser();p.add_argument('receipts',nargs='+',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=audit(a.receipts)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(f"no-GMP audit: {r['units']} TU records, {len(r['binaries'])} binaries PASS")
if __name__=='__main__':main()
