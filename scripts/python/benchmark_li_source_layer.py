#!/usr/bin/env python3
"""Run the pinned study block benchmark with a source-built operator ELF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-optimization',type=Path,required=True)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--label',choices=('source_baseline','source_candidate'),required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    args = parser.parse_args()
    study = args.study_optimization.resolve()
    identity = args.identity.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError('Choose a fresh output root')
    original = study/'layer_bench.py'
    data = original.read_bytes()
    assert hashlib.sha256(data).hexdigest() == 'eed37257434603a570836464323a1381bf7e1a6c069b8b5a444c4396d411ed76'
    adapter = study/'build/librow_direct.so'
    assert os.environ.get('LD_PRELOAD') == str(adapter)
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == '4c6447504e1ae2f1fb69d5d93fcc33681fb3284111813d571eba524fa0541e2d'
    source = data.decode()
    old = "choices=['row_ring4', 'row_contig']"
    assert source.count(old) == 1
    source = source.replace(old,"choices=['source_baseline', 'source_candidate']")
    output.mkdir(parents=True)
    alias = output/'build'/args.label
    alias.parent.mkdir()
    alias.symlink_to(identity.parent,target_is_directory=True)
    (output/'benchmark-source.py').write_text(source)
    (output/'source-identity.json').write_text(identity.read_text())
    module = types.ModuleType('pinned_li_layer_benchmark')
    module.__file__ = str(original)
    exec(compile(source,str(original),'exec'),module.__dict__)
    module.ROOT = output
    sys.argv = [str(original),'--variant',args.label,'--device',str(args.device),
                '--warmups','12','--repeats','8','--rounds','2']
    module.main()


if __name__ == '__main__':
    main()
