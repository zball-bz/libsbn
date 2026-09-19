#!/usr/bin/env python3
"""Clang-only explicit-manifest builder. Never discovers production TUs by glob."""
import fcntl
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from reference_paths import reference_command, reference_root, reference_path, experiments_root

V3 = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd):
    print(shlex.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=V3, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', choices=['lib', 'check', 'bench', 'examples'])
    ap.add_argument('--sanitize', choices=['none', 'address', 'thread'], default='none')
    ap.add_argument('--extended',action='store_true')
    ap.add_argument('--checked',action='store_true',help='Enable small-operation contract diagnostics')
    ap.add_argument('--only-test',action='append',help='Run a registered test during focused iteration (repeatable)')
    ap.add_argument('--with-reference', action='store_true', help='Include optional frozen-reference benchmarks')
    ap.add_argument('--only-benchmark', action='append', help='Build only selected registered benchmarks')
    ap.add_argument('--build-only', action='store_true', help='Build test executables without running them')
    args = ap.parse_args()
    (V3/'build').mkdir(exist_ok=True)
    build_lock=(V3/'build/benchmark.lock').open('a')
    fcntl.flock(build_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run(['python3', 'tools/generate_tuning.py', '--check'])
    run(['python3', 'tools/generate_radix_bounds.py', '--check'])
    run(['python3', 'tools/check_native_variants.py'])
    config = json.loads((V3/'config/native.json').read_text())
    sources = json.loads((V3/'config/sources.json').read_text())['entries']
    variant = 'native' if args.sanitize == 'none' else args.sanitize
    if args.checked and args.sanitize=='none':variant='checked'
    build = V3/'build'/variant
    build.mkdir(parents=True, exist_ok=True)
    compiler = {k: config['toolchain'][k] for k in ['cc','cxx','ar']}
    versions = {k: subprocess.check_output([exe,'--version'], text=True).splitlines()[0]
                for k,exe in compiler.items()}
    if 'clang' not in versions['cc'].lower() or 'clang' not in versions['cxx'].lower():
        raise SystemExit('Clang is required for every C/C++ TU')
    pin=config['toolchain'].get('version_pin')
    if pin and any(pin not in versions[k] for k in ['cc','cxx','ar']):
        raise SystemExit(f'toolchain version must match {pin}')
    sanitize = [] if args.sanitize == 'none' else [f'-fsanitize={args.sanitize}', '-fno-omit-frame-pointer']
    tests = json.loads((V3/'config/tests.json').read_text())['tests'] if args.target == 'check' else []
    tests=[t for t in tests if args.extended or t.get('tier')!='extended']
    if args.only_test:
        requested=set(args.only_test)
        missing=requested-{t['name'] for t in tests}
        if missing:raise SystemExit(f'test not registered/enabled: {sorted(missing)}')
        tests=[t for t in tests if t['name'] in requested]
    benchmarks = json.loads((experiments_root(True)/'config/benchmarks.json').read_text())['benchmarks'] if args.target == 'bench' else []
    if args.only_benchmark:
        selected=set(args.only_benchmark)
        if selected-{b['name'] for b in benchmarks}:raise SystemExit('unknown benchmark')
        benchmarks=[b for b in benchmarks if b['name'] in selected]
    elif not args.with_reference:
        benchmarks=[b for b in benchmarks if not b.get('requires_reference')]
    if any(b.get('requires_reference') for b in benchmarks):reference_root(True)
    examples = json.loads((V3/'config/examples.json').read_text())['examples'] if args.target == 'examples' else []
    receipt = {'schema':1,'variant':variant,'compiler':versions,
               'small_checks':bool(args.checked or args.sanitize!='none'),'units':[], 'tests':[]}

    def compile_one(src, language, group, prefix='', extra_flags=()):
        obj = build/(prefix+src.replace('../','external/')+'.o')
        obj.parent.mkdir(parents=True, exist_ok=True)
        dep = Path(str(obj)+'.d')
        record = Path(str(obj)+'.json')
        cxx = language == 'cxx'
        exe = compiler['cxx' if cxx else 'cc']
        flags = ['-g','-Wall','-Wextra','-Werror','-Iinclude','-Isrc']
        if language in ['c','cxx']: flags.insert(0, '-O2')
        flags += config['groups'][group]['target_flags']
        flags += list(extra_flags)
        if prefix:flags += ['-Itests/no_gmp']
        if language in ['c','cxx']:
            flags += ['-std=c++20','-fno-exceptions','-fno-rtti'] if cxx else ['-std=c11']
            flags += sanitize
            if args.checked:flags += ['-DSBN3_CHECK_SMALL=1']
        if language != 'asm': flags += ['-MD','-MF',str(dep)]
        cmd = reference_command([exe,*flags,'-c',src,'-o',str(obj)])
        previous = json.loads(record.read_text()) if record.exists() else {}
        valid = obj.exists() and previous.get('command') == cmd and previous.get('compiler') == versions['cxx' if cxx else 'cc']
        if valid:valid=previous.get('object_sha256')==digest(obj)
        dependencies = previous.get('dependencies',{})
        if valid:
            valid = bool(dependencies) and all(Path(p).is_file() and digest(Path(p))==h for p,h in dependencies.items())
        elapsed = 0
        if not valid:
            start=time.monotonic();run(cmd);elapsed=time.monotonic()-start
            files=[(V3/reference_path(src)).resolve()]
            if dep.exists():
                body=dep.read_text().replace('\\\n',' ').split(':',1)[1]
                files += [(V3/p).resolve() for p in shlex.split(body)]
            dependencies={str(p):digest(p) for p in sorted(set(files))}
            previous={'command':cmd,'compiler':versions['cxx' if cxx else 'cc'],
                      'dependencies':dependencies,'compile_seconds':elapsed,'object_sha256':digest(obj)}
            record.write_text(json.dumps(previous,indent=2)+'\n')
        receipt['units'].append(dict(source=src,object=str(obj.relative_to(V3)),
                                    reused=valid,**previous))
        return obj

    objects=[compile_one(e['path'],e['language'],e['group'],extra_flags=e.get('extra_flags',[])) for e in sources]
    archive=build/'libsbn_v3.a'
    membership={'objects':{str(p.relative_to(V3)):digest(p) for p in objects},'archiver':versions['ar']}
    membership_file=build/'archive-members.json'
    if not archive.exists() or not membership_file.exists() or json.loads(membership_file.read_text())!=membership:
        # Recreate the archive so retired manifest entries cannot survive inside it.
        temporary=build/'libsbn_v3.next.a'
        if temporary.exists(): temporary.unlink()
        run([compiler['ar'],'rcs',temporary,*objects]);os.replace(temporary,archive)
        membership_file.write_text(json.dumps(membership,indent=2)+'\n')
    receipt['archive']={'path':str(archive.relative_to(V3)),'sha256':digest(archive)}
    for example in examples:
        objs=[compile_one(e['path'],e['language'],e.get('group','reference'),'examples/',e.get('extra_flags',[])) for e in example['sources']]
        exe=build/example['name']
        run([compiler['cc'],'-fuse-ld=lld',*sanitize,*objs,archive,'-pthread','-lm','-o',exe])
        receipt.setdefault('examples',[]).append({'name':example['name'],'path':str(exe.relative_to(V3)),'sha256':digest(exe)})
    for benchmark in benchmarks:
        objs=[compile_one(e['path'],e['language'],e.get('group','common'),'bench/',e.get('extra_flags',[])) for e in benchmark['sources']]
        exe=build/benchmark['name']
        run([compiler['cxx'],'-fuse-ld=lld',*objs,archive,'-pthread','-lm','-o',exe])
        receipt.setdefault('benchmarks',[]).append({'name':benchmark['name'],'path':str(exe.relative_to(V3)),'sha256':digest(exe)})
    for test in tests:
        if any('gmp' in f.lower() for f in test.get('link_flags',[])):raise SystemExit('GMP is forbidden in maintained tests')
        objs=[compile_one(e['path'],e['language'],e.get('group','reference'),'tests/',e.get('extra_flags',[])) for e in test['sources']]
        exe=build/test['name']
        cmd=[compiler['cxx'],'-fuse-ld=lld',*sanitize,*objs,archive,'-pthread','-lm',*test.get('link_flags',[]),'-o',exe]
        run(cmd)
        receipt.setdefault('test_binaries',[]).append({'name':test['name'],'path':str(exe.relative_to(V3)),'sha256':digest(exe),'timeout_seconds':test.get('timeout_seconds',60),'environment':test.get('environment',{})})
        if args.build_only:continue
        trace=build/'oracle-traces'/(test['name']+'.jsonl');trace.parent.mkdir(exist_ok=True);trace.unlink(missing_ok=True)
        env=os.environ.copy()
        env['SBN3_ORACLE_TRACE']=str(trace)
        env.update(test.get('environment',{}))
        start=time.monotonic()
        result=subprocess.run([str(exe)],cwd=V3,env=env,timeout=test.get('timeout_seconds',60),text=True,capture_output=True)
        print(result.stdout,end='');print(result.stderr,end='')
        receipt['tests'].append({'name':test['name'],'command':[str(exe)],
                                 'elapsed_seconds':time.monotonic()-start,'returncode':result.returncode,
                                 'stdout':result.stdout,'stderr':result.stderr,
                                 'oracle_trace':str(trace.relative_to(V3)) if trace.exists() else None,
                                 'oracle_trace_sha256':digest(trace) if trace.exists() else None})
        (build/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if result.returncode: raise SystemExit(result.returncode)
    (build/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if tests:(build/'test-build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if benchmarks:(build/'benchmark-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if examples:(build/'examples-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'OK: {archive.relative_to(V3)}; receipt {variant}/receipt.json')


if __name__=='__main__':
    main()
