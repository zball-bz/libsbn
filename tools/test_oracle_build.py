"""One explicit reference TU for the retained comparison-tool builders.

The caller owns benchmark.lock. Its object receipt records every source input.
"""
import hashlib,json,shlex,subprocess
from pathlib import Path

def build_oracle(root):
    root=Path(root);folder=root/'build/test-oracle';folder.mkdir(parents=True,exist_ok=True)
    obj=folder/'oracle.o';dep=folder/'oracle.d';record=folder/'receipt.json'
    command=['clang++','-O2','-g','-std=c++20','-march=x86-64','-fno-vectorize','-fno-slp-vectorize','-fno-exceptions','-fno-rtti','-Iinclude','-Itests/no_gmp','-MD','-MF',str(dep),'-c','tests/oracle/oracle.cpp','-o',str(obj)]
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    compiler=subprocess.check_output(['clang++','--version'],text=True).splitlines()[0];assert '21.1.8' in compiler
    old=json.loads(record.read_text()) if record.exists() else {}
    valid=obj.exists() and old.get('command')==command and old.get('compiler')==compiler and old.get('sha256')==sha(obj)
    if valid:valid=bool(old.get('dependencies')) and all(Path(p).exists() and sha(p)==h for p,h in old['dependencies'].items())
    if not valid:
        subprocess.run(command,cwd=root,check=True)
        files={(root/p).resolve() for p in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])}
        record.write_text(json.dumps({'command':command,'compiler':compiler,'sha256':sha(obj),'dependencies':{str(p):sha(p) for p in sorted(files)}},indent=2)+'\n')
    return obj
