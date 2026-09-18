#!/usr/bin/env python3
"""Explicit native gate preparation: raise this process's MEMLOCK only."""
import os
from pathlib import Path
import resource
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
needed=1<<30
soft,hard=resource.getrlimit(resource.RLIMIT_MEMLOCK)
if soft!=resource.RLIM_INFINITY and soft<needed:
    if hard==resource.RLIM_INFINITY or hard>=needed:resource.setrlimit(resource.RLIMIT_MEMLOCK,(needed,hard))
    else:subprocess.run(['sudo','-n','prlimit','--pid',str(os.getpid()),f'--memlock={needed}:{needed}'],check=True)
sys.exit(subprocess.call([sys.executable,str(root/'tools/build.py'),'check','--extended',*sys.argv[1:]],cwd=root))
