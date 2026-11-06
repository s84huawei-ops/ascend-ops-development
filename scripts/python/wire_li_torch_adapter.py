#!/usr/bin/env python3
"""Wire the op-src adapter into vLLM's existing indexer schema and rebuild it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-root',type=Path,required=True)
    parser.add_argument('--backup-dir',type=Path,required=True)
    parser.add_argument('--soc',choices=('ascend910b','ascend910_93'),required=True)
    args = parser.parse_args()
    root = args.vllm_root.resolve()
    backup = args.backup_dir.resolve()
    adapter = root/'csrc/attention/lightning_indexer/lightning_indexer_torch_adpt.h'
    assert 'npu_lightning_indexer_tensor_only' in adapter.read_text(), 'Copy the current op-src adapter first'
    binding = root/'csrc/torch_binding.cpp'
    original = binding.read_text()
    old_include = '#include "attention/lightning_indexer_vllm/lightning_indexer_vllm_torch_adpt.h"'
    plain_include = '#include "attention/lightning_indexer/lightning_indexer_torch_adpt.h"'
    new_include = '#define LI_EXPLICIT_OPAPI_ROUTING\n'+plain_include+'\n#undef LI_EXPLICIT_OPAPI_ROUTING'
    old_registration = '&vllm_ascend::npu_lightning_indexer);'
    new_registration = '&vllm_ascend::npu_lightning_indexer_tensor_only);'
    if original.count(old_include) == 1 and original.count(old_registration) == 1:
        changed = original.replace(old_include,new_include).replace(old_registration,new_registration)
    elif original.count(new_include) == 1 and original.count(new_registration) == 1:
        changed = original
    elif original.count(plain_include) == 1 and original.count(new_registration) == 1:
        changed = original.replace(plain_include,new_include)
    else:
        raise RuntimeError('Unexpected vLLM indexer binding; inspect its schema before adapting it')
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(binding,backup/'torch_binding.cpp')
    for path in (root/'vllm_ascend').glob('*.so'):
        shutil.copy2(path,backup/path.name)
    binding.write_text(changed)
    # This vLLM setup.py uses COMPILE_CUSTOM_KERNELS for both ACLNN packages and
    # the Torch extension. Keep it enabled, and skip only the already completed
    # package-build command in a saved, inspectable setup snapshot.
    setup = root/'setup.py'
    setup_source = setup.read_text()
    anchor = '            self.run_command("build_aclnn")'
    assert setup_source.count(anchor) == 1, 'Unexpected vLLM setup.py build hook'
    setup_source = setup_source.replace(anchor,'            print("Using the explicitly built ACLNN package")')
    (backup/'setup-extension-only.py').write_text(setup_source)
    bootstrap = backup/'build_extension_only.py'
    context = repr({'__name__':'__main__','__file__':str(setup)})
    bootstrap.write_text('import sys\nfrom pathlib import Path\n' +
        f'sys.path.insert(0, {str(root)!r})\n' +
        f'sys.argv = [{str(setup)!r}, "build_ext", "--inplace"]\n' +
        f'code = Path({str(backup / "setup-extension-only.py")!r}).read_text()\n' +
        f'exec(compile(code, {str(setup)!r}, "exec"), {context})\n')
    command = [sys.executable,str(bootstrap)]
    environment = dict(os.environ)
    environment['COMPILE_CUSTOM_KERNELS'] = '1'
    environment['MAX_JOBS'] = '8'
    environment['SOC_VERSION'] = args.soc
    environment['CMAKE_BUILD_TYPE'] = 'Release'
    (backup/'adapter-build.json').write_text(json.dumps(dict(
        command=command,compile_custom_kernels=1,max_jobs=8,soc=args.soc,
        original_binding_sha256=hashlib.sha256(original.encode()).hexdigest(),
        patched_binding_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest()),indent=2))
    subprocess.run(command,cwd=root,env=environment,check=True)
    extensions = list((root/'vllm_ascend').glob('vllm_ascend_C*.so'))
    assert len(extensions) == 1, 'Expected the rebuilt extension in the vLLM checkout'
    extension = extensions[0]
    previous = backup/extension.name
    if changed != original and previous.exists():
        assert extension.read_bytes() != previous.read_bytes(), 'The extension was not rebuilt'
    print('WIRED_INDEXER_ADAPTER',binding,flush=True)


if __name__ == '__main__':
    main()
