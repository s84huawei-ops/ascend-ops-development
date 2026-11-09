#!/usr/bin/env python3
"""Compile op-src with one explicit installed CANN recipe for isolated A/B tests.

This produces a device ELF, not an installed custom-op package. Production
integration still uses build_operator_fixed.sh and the target server's toolchain.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.suffix in ('.h', '.cpp')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True,
                        help='Installed kernel JSON containing supportInfo')
    parser.add_argument('--soc', required=True)
    args = parser.parse_args()
    source, output, metadata = (p.resolve() for p in
                               (args.source_dir, args.output_dir, args.metadata))
    if output.exists():
        raise RuntimeError('Output already exists; choose a fresh directory')
    entry = source / 'lightning_indexer.cpp'
    if not entry.is_file():
        raise RuntimeError('source-dir must be the operator op_kernel directory')
    info = json.loads(metadata.read_text())['supportInfo']
    if info['implMode'] != 'high_performance':
        raise RuntimeError('Expected an explicit high_performance recipe')
    output.mkdir(parents=True)
    before = hashes(source)
    from asc_op_compile_base.common.platform import set_current_compile_soc_info, get_soc_spec
    from asc_op_compile_base.common.buildcfg import build_config
    from asc_op_compile_base.common.context import op_context
    set_current_compile_soc_info(args.soc)
    assert get_soc_spec('SOC_VERSION') == args.soc
    os.environ['ASCENDC_BUILD_LOG_DIR'] = str(output / 'logs')
    os.environ['TILINGKEY_PAR_COMPILE'] = '0'
    os.environ['TILINGKEY_PARALLEL_JOB'] = '1'
    cann = Path(os.environ['ASCEND_HOME_PATH'])
    recipe_path = cann / 'opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/dynamic/lightning_indexer.py'
    spec = importlib.util.spec_from_file_location('explicit_li_recipe', recipe_path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    recipe.get_kernel_source = lambda *unused: str(entry)
    original = recipe.compile_op

    def compile_explicit(src, origin, opinfo, options, channel, op_options, *extra, **kwargs):
        explicit = json.loads(op_options)
        explicit['auto_sync'] = False
        options = list(options) + ['-O3', '-g', '--cce-auto-sync=off']
        record = dict(source_dir=str(source), source_hashes=before, soc=args.soc,
                      metadata=str(metadata), metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
                      recipe=str(recipe_path), options=options, op_compile_option=explicit,
                      support_info=info, code_channel=channel,
                      compiler_version=subprocess.check_output(['bisheng', '--version'], text=True))
        (output / 'recipe.json').write_text(json.dumps(record, indent=2))
        return original(src, origin, opinfo, options, channel, json.dumps(explicit), *extra, **kwargs)

    recipe.compile_op = compile_explicit
    tensors = [dict(t, ori_shape=t['shape'], ori_format=t['format'])
               for t in info['inputs'] + info['outputs']]
    attrs = {v['name']: v['value'] for v in info['attrs']}
    os.chdir(output)
    with op_context.OpContext('dynamic'), build_config(
            kernel_meta_parent_dir=str(output), save_temp_cce_file=True, enable_deterministic_mode=0):
        recipe.lightning_indexer(*tensors, **attrs, kernel_name=metadata.stem,
                                 impl_mode='high_performance')
    assert hashes(source) == before, 'Source changed during compilation'
    elf = output / 'kernel_meta' / (metadata.stem + '.o')
    identity = dict(elf=str(elf), sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
                    source_hashes=before, soc=args.soc)
    (output / 'identity.json').write_text(json.dumps(identity, indent=2))
    print('BUILT', elf, flush=True)


if __name__ == '__main__':
    main()
