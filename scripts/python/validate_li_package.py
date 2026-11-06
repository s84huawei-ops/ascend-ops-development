#!/usr/bin/env python3
"""Validate a source-built ACLNN package against an isolated native process.

The native worker never sees ASCEND_CUSTOM_OPP_PATH. The custom worker loads the
exact supplied op-api library and checks its exported symbol. This prevents a
custom registration from accidentally replacing both sides of the comparison.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def custom_call(torch, vendor, query, key, weights, lengths, output):
    libraries = list((vendor / 'op_api/lib').glob('libcust_opapi.so'))
    if len(libraries) != 1:
        raise RuntimeError('Expected exactly one custom op-api library')
    library_path = libraries[0].resolve()
    symbols = subprocess.check_output(['nm','-D','--defined-only',str(library_path)],text=True)
    assert any(line.split()[-1] == 'aclnnLightningIndexerGetWorkspaceSize'
               for line in symbols.splitlines() if line.split()), 'Package does not define the requested API'
    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.c_void_p
    integer = ctypes.c_int64
    create = library.aclCreateTensor
    create.argtypes = [ctypes.POINTER(integer),ctypes.c_uint64,ctypes.c_int,
                       ctypes.POINTER(integer),integer,ctypes.c_int,
                       ctypes.POINTER(integer),ctypes.c_uint64,pointer]
    create.restype = pointer
    destroy = library.aclDestroyTensor
    destroy.argtypes = [pointer]
    destroy.restype = ctypes.c_int
    descriptors = []
    arrays = []
    try:
        for tensor in (query,key,weights,lengths,lengths,output):
            shape = (integer*tensor.ndim)(*tensor.shape)
            stride = (integer*tensor.ndim)(*tensor.stride())
            arrays.extend((shape,stride))
            dtype = 27 if tensor.dtype == torch.bfloat16 else 3
            descriptor = create(shape,tensor.ndim,dtype,stride,0,2,shape,tensor.ndim,tensor.data_ptr())
            assert descriptor, 'aclCreateTensor failed'
            descriptors.append(descriptor)
        get_workspace = library.aclnnLightningIndexerGetWorkspaceSize
        get_workspace.argtypes = ([pointer]*6+[ctypes.c_char_p]*2+[integer]*4+
                                  [ctypes.c_bool]+[pointer]*2+
                                  [ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(pointer)])
        get_workspace.restype = ctypes.c_int
        workspace_size, executor = ctypes.c_uint64(), pointer()
        q,k,w,lq,lk,out = descriptors
        rc = get_workspace(q,k,w,lq,lk,None,b'BSND',b'BSND',2048,3,
                           2**63-1,2**63-1,False,out,None,
                           ctypes.byref(workspace_size),ctypes.byref(executor))
        assert rc == 0, f'Custom GetWorkspaceSize returned {rc}'
        workspace = torch.empty(workspace_size.value,dtype=torch.uint8,device=query.device)
        execute = library.aclnnLightningIndexer
        execute.argtypes = [pointer,ctypes.c_uint64,pointer,pointer]
        execute.restype = ctypes.c_int
        stream = torch.npu.current_stream(query.device).npu_stream
        rc = execute(workspace.data_ptr() if workspace_size.value else None,
                     workspace_size,executor,stream)
        assert rc == 0, f'Custom execution returned {rc}'
        torch.npu.synchronize(query.device)
        return dict(library=str(library_path),library_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest(),
                    workspace_bytes=workspace_size.value)
    finally:
        for descriptor in descriptors:
            destroy(descriptor)


def worker(args):
    import torch
    import torch_npu
    assert os.environ.get('TASK_QUEUE_ENABLE') == '0'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    torch.npu.set_option({'ACL_OP_COMPILER_CACHE_MODE':'disable'})
    device = torch.device(f'npu:{args.device}')
    torch.manual_seed(args.seed)
    query = (torch.randn(1,args.sequence,64,128)*0.25).bfloat16().to(device)
    key = (torch.randn(1,args.sequence,1,128)*0.25).bfloat16().to(device)
    weights = (torch.randn(1,args.sequence,64)*0.25).bfloat16().to(device)
    lengths = torch.tensor([args.sequence],dtype=torch.int32,device=device)
    metadata = dict(device=args.device,device_name=torch.npu.get_device_properties(args.device).name,
                    seed=args.seed,sequence=args.sequence,torch_version=torch.__version__,
                    torch_npu_version=torch_npu.__version__)
    if args._worker == 'native':
        assert 'ASCEND_CUSTOM_OPP_PATH' not in os.environ
        output = torch_npu.npu_lightning_indexer(
            query,key,weights,actual_seq_lengths_query=lengths,actual_seq_lengths_key=lengths,
            block_table=None,layout_query='BSND',layout_key='BSND',sparse_count=2048,
            sparse_mode=3,pre_tokens=2**63-1,next_tokens=2**63-1,return_value=False)[0]
        torch.npu.synchronize(device)
        torch.save(output.cpu(),args.output/'native-indices.pt')
        (args.output/'native.json').write_text(json.dumps(metadata,indent=2))
        print('NATIVE_REFERENCE_SAVED',flush=True)
    else:
        output = torch.empty(1,args.sequence,1,2048,dtype=torch.int32,device=device)
        if args.backend == 'aclnn':
            metadata.update(custom_call(torch,args.vendor_dir,query,key,weights,lengths,output))
        else:
            from vllm_ascend.utils import enable_custom_op
            assert enable_custom_op(), 'vLLM custom operators are unavailable'
            operation = torch.ops._C_ascend.npu_lightning_indexer
            output = operation(query,key,weights,actual_seq_lengths_query=lengths,
                actual_seq_lengths_key=lengths,block_table=None,layout_query='BSND',
                layout_key='BSND',sparse_count=2048,sparse_mode=3)
            assert isinstance(output,torch.Tensor), 'The vLLM tensor-only interface changed'
            torch.npu.synchronize(device)
            metadata['schema'] = str(operation._schemas)
        metadata['backend'] = args.backend
        expected = torch.load(args.output/'native-indices.pt',map_location='cpu',weights_only=True)
        assert torch.equal(output.cpu(),expected), 'Installed package differs from native'
        metadata.update(exact_indices=True,complete=True)
        (args.output/'custom.json').write_text(json.dumps(metadata,indent=2))
        print('CORRECT_INSTALLED_PACKAGE exact indices',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor-dir',type=Path,required=True)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--sequence',type=int,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=('aclnn','vllm'),default='aclnn')
    parser.add_argument('--_worker',choices=('native','custom'),help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.vendor_dir = args.vendor_dir.resolve()
    args.output = args.output.resolve()
    if args._worker:
        worker(args)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    for label in ('native','custom'):
        env = dict(os.environ)
        env['TASK_QUEUE_ENABLE'] = '0'
        env.pop('LD_PRELOAD',None)
        env.pop('ASCEND_CUSTOM_OPP_PATH',None)
        if label == 'custom':
            env['ASCEND_CUSTOM_OPP_PATH'] = str(args.vendor_dir)
            env['LD_LIBRARY_PATH'] = str(args.vendor_dir/'op_api/lib')+':'+env.get('LD_LIBRARY_PATH','')
        command = [sys.executable,str(Path(__file__).resolve()),'--vendor-dir',str(args.vendor_dir),
                   '--device',str(args.device),'--sequence',str(args.sequence),'--seed',str(args.seed),
                   '--output',str(args.output),'--_worker',label]
        command.extend(['--backend',args.backend])
        with (args.output/(label+'.log')).open('x') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('RESULT',args.output/'custom.json',flush=True)


if __name__ == '__main__':
    main()
