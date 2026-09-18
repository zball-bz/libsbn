#!/usr/bin/env python3
"""Cross-check test-only arithmetic/certified witnesses against Python integers."""
import argparse, hashlib, json, math, os, random, subprocess, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--executable',type=Path,default=ROOT/'build/native/oracle_test');ap.add_argument('--output',type=Path,default=ROOT/'build/oracle-python.json');args=ap.parse_args()
    rng=random.Random(0x51926735);cases=[]
    def add(op,a,b,result):cases.append((op,a,b,result))
    for bits in [1,2,63,64,65,127,128,255,512,1024,4096,8192,16384,65536,131072]:
        for k in range(3):
            a=rng.getrandbits(bits);b=rng.getrandbits(max(1,bits//2))|1
            add('mul',a,b,a*b);add('div',a,b,a//b)
            for sign in [1,-1]:
                x=sign*a;add('add',x,b,x+b);add('sub',x,b,x-b);add('mod',x,b,x%b)
                shift=rng.randrange(bits+65);add('shl',x,shift,x<<shift);add('shr',x,shift,sign*(a>>shift));add('floor_shr',x,shift,x>>shift);add('low',x,shift,x% (1<<shift))
    for bits in [1,2,63,128,255,512,1024,4096]:
        for _ in range(3):
            x=rng.getrandbits(bits);add('sqrt',x,0,math.isqrt(x))
    for n in [64,129,1024,4096]:
        x=1<<(128*n+1);add('sqrt',x,0,math.isqrt(x))
    for n in [1,2,16,129,2048]:
        for sign in [1,-1]:
            value=sign*rng.getrandbits(256*n)
            for delta in [-1,1]:
                modulus=(1<<(64*n))+delta;add('mod',value,modulus,value%modulus)
    def hx(x):return ('-' if x<0 else '')+format(abs(x),'x')
    payload=''.join(f'{op} {hx(a)} {hx(b)}\n' for op,a,b,_ in cases)
    args.output.parent.mkdir(parents=True,exist_ok=True);trace=args.output.with_suffix('.challenges.jsonl');trace.unlink(missing_ok=True)
    begin=time.monotonic();run=subprocess.run([str(args.executable),'--server'],input=payload,text=True,capture_output=True,env=dict(os.environ,SBN3_ORACLE_TRACE=str(trace)),timeout=120)
    assert run.returncode==0,run.stderr
    rows=run.stdout.splitlines();assert len(rows)==len(cases),(len(rows),len(cases))
    for i,(line,case) in enumerate(zip(rows,cases)):assert int(line,16)==case[3],(i,case[0],case[1].bit_length())
    report={'schema':1,'status':'passed','cases':len(cases),'elapsed_seconds':time.monotonic()-begin,'reference':'CPython built-in integers and math.isqrt; no external multiprecision module','executable_sha256':hashlib.sha256(args.executable.read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(payload.encode()).hexdigest(),'trace':str(trace),'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest() if trace.exists() else None}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(f"oracle/Python: {len(cases)} cases PASS in {report['elapsed_seconds']:.3f}s")
if __name__=='__main__':main()
