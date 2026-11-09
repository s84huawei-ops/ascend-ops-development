#!/usr/bin/env python3
"""A/B device test of two source-built ELFs using the independent study adapter.

This test is for Ascend910B2. It does not install or validate a production custom
op package. The ordinary CANN frontend supplies tiling and the captured ABI;
both ELFs use identical owned buffers, streams, and submission paths.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Baseline identity.json')
    parser.add_argument('--candidate', type=Path, required=True, help='Candidate identity.json')
    parser.add_argument('--study-common', type=Path, required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--warmups', type=int, required=True)
    parser.add_argument('--repeats', type=int, required=True)
    parser.add_argument('--rounds', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matrix', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite results')
    adapter = args.study_common.resolve() / 'build/libstudy_direct.so'
    assert os.environ.get('LD_PRELOAD') == str(adapter), 'Preload the selected study adapter'
    assert os.environ.get('TASK_QUEUE_ENABLE') == '0'
    assert args.warmups >= 12 and args.repeats >= 8 and args.rounds >= 2
    ctypes.CDLL(None).study_arm
    os.environ.pop('LD_PRELOAD')
    import torch
    import torch_npu
    sys.path.insert(0, str(args.study_common.resolve()))
    from direct import Direct
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.npu.set_device(args.device)
    assert torch.npu.get_device_properties(args.device).name == 'Ascend910B2'
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    torch.npu.set_option({'ACL_OP_COMPILER_CACHE_MODE': 'disable'})
    device = torch.device(f'npu:{args.device}')
    identities = {}
    for label in ('baseline', 'candidate'):
        identities[label] = json.loads(getattr(args, label).read_text())
        elf = Path(identities[label]['elf'])
        assert hashlib.sha256(elf.read_bytes()).hexdigest() == identities[label]['sha256']
        assert identities[label]['soc'] == 'Ascend910B2'
    direct = Direct(identities['candidate']['elf'], vector_core=False,
                    reference_elf=identities['baseline']['elf'])
    workspace = torch.empty(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    stream = torch.npu.current_stream(args.device).npu_stream
    report = dict(seed=args.seed, device=args.device, identities=identities,
                  adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),
                  method='NPU events, alternating baseline/candidate direct ELF launches; identical owned workspace',
                  cases=[], exact_indices=True, exact_values=True)
    # Fields: query length, KV length, heads, query dtype, weights dtype,
    # layout, top-k, return-values, zero weights. Include cache/fallback paths,
    # split-core boundaries, odd query tails, paged KV, and tied ordering.
    cases = [(16384,16384,64,'bf16','bf16','BSND',2048,False,False)]
    if args.matrix:
        cases += [
            (7,513,64,'bf16','bf16','BSND',2048,True,False),
            (9,2049,64,'bf16','bf16','BSND',2048,False,False),
            (17,4096,64,'bf16','bf16','BSND',2048,False,True),
            (257,4096,64,'bf16','bf16','BSND',2048,True,True),
            (513,8193,64,'bf16','bf16','BSND',2048,True,False),
            (257,4096,64,'bf16','bf16','TND_BATCH2',2048,True,True),
            (65,4097,64,'bf16','bf16','TND',2048,True,False),
            (16,4097,64,'bf16','bf16','PA_BSND',2048,False,False),
            (2,4096,64,'bf16','bf16','BSND',2048,True,False),
            (8,4096,32,'bf16','bf16','BSND',1024,True,False),
            (9,513,64,'bf16','bf16','BSND',1024,True,False),
            (9,4097,64,'bf16','bf16','BSND',4096,True,False),
            (9,1024,64,'fp16','fp16','BSND',2048,True,False),
            (9,2049,64,'bf16','float','BSND',2048,True,False),
        ]
    torch.manual_seed(args.seed)
    for case_index, (qlen,klen,heads,qdtype,wdtype,layout,topk,values,zero) in enumerate(cases):
        dtype = torch.bfloat16 if qdtype == 'bf16' else torch.float16
        weight_dtype = {'bf16':torch.bfloat16,'fp16':torch.float16,'float':torch.float32}[wdtype]
        batch = 2 if layout == 'TND_BATCH2' else 1
        query = (torch.randn(batch,qlen,heads,128)*0.25).to(dtype).to(device)
        key = (torch.randn(batch,klen,1,128)*0.25).to(dtype).to(device)
        weights = (torch.randn(batch,qlen,heads)*0.25).to(weight_dtype).to(device)
        if zero:
            weights.zero_()
        block_table = None
        query_layout = 'BSND'
        key_layout = 'BSND'
        if layout in ('TND','TND_BATCH2'):
            query = query.reshape(batch*qlen,heads,128)
            key = key.reshape(batch*klen,1,128)
            weights = weights.reshape(batch*qlen,heads)
            query_layout = key_layout = 'TND'
        elif layout == 'PA_BSND':
            pages = (klen + 127) // 128
            padded = torch.zeros(1,pages*128,1,128,dtype=dtype,device=device)
            padded[:,:klen].copy_(key)
            key = padded.reshape(pages,128,1,128)
            block_table = torch.arange(pages,dtype=torch.int32,device=device).reshape(1,-1)
            key_layout = 'PA_BSND'
        qlengths = torch.arange(1,batch+1,dtype=torch.int32,device=device)*qlen
        klengths = torch.arange(1,batch+1,dtype=torch.int32,device=device)*klen
        direct.arm(query.data_ptr())
        output = torch_npu.npu_lightning_indexer(
            query,key,weights,actual_seq_lengths_query=qlengths,actual_seq_lengths_key=klengths,
            block_table=block_table,layout_query=query_layout,layout_key=key_layout,
            sparse_count=topk,sparse_mode=3,pre_tokens=2**63-1,next_tokens=2**63-1,
            return_value=values)
        torch.npu.synchronize(args.device)
        expected_indices = output[0].clone()
        expected_values = output[1].clone() if values else None
        meta, raw = direct.captured()
        assert meta['args_size'] == 192 and meta['placeholder_count'] == 1, meta
        assert struct.unpack_from('<Q',raw,8)[0] == query.data_ptr(), 'Unexpected query ABI'
        assert struct.unpack_from('<Q',raw,56)[0] == output[0].data_ptr(), 'Unexpected output ABI'
        direct.prepare(key=meta['key'],blocks=meta['blocks'],args_size=192,placeholder_count=1)
        old_workspace = struct.unpack_from('<Q',raw,72)[0]
        assert old_workspace
        direct.patch(72,old_workspace,workspace.data_ptr())
        for label, launch in (('baseline',direct.launch_reference),('candidate',direct.launch)):
            # A missing output store must not inherit the native/previous lane's
            # correct result. Give each checked launch identical clean scratch.
            output[0].fill_(torch.iinfo(torch.int32).min)
            if values:
                output[1].fill_(float('nan'))
            workspace.zero_()
            torch.npu.synchronize(args.device)
            launch(stream)
            torch.npu.synchronize(args.device)
            assert torch.equal(output[0],expected_indices), f'{label}: indices differ for case {case_index}'
            if values:
                assert torch.equal(output[1],expected_values), f'{label}: values differ for case {case_index}'
        case = dict(batch=batch,query_length=qlen,key_length=klen,heads=heads,query_dtype=qdtype,
                    weights_dtype=wdtype,layout=layout,topk=topk,return_values=values,
                    zero_weights=zero,key=meta['key'],blocks=meta['blocks'],exact=True)
        print('CORRECT',case_index,json.dumps(case),flush=True)
        if case_index == 0:
            samples = {'baseline':[], 'candidate':[]}
            for rnd in range(args.rounds):
                order = ('baseline','candidate') if rnd % 2 == 0 else ('candidate','baseline')
                for label in order:
                    launch = direct.launch_reference if label == 'baseline' else direct.launch
                    for _ in range(args.warmups):
                        launch(stream)
                    torch.npu.synchronize(args.device)
                    events = []
                    for _ in range(args.repeats):
                        start,end = torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                        start.record(); launch(stream); end.record(); events.append((start,end))
                    torch.npu.synchronize(args.device)
                    times = [start.elapsed_time(end) for start,end in events]
                    samples[label].append(times)
                    print('DEVICE',label,rnd,statistics.median(times),flush=True)
            case['samples_ms'] = samples
            case['median_ms'] = {label:statistics.median(t for row in rows for t in row)
                                 for label,rows in samples.items()}
            case['speedup'] = case['median_ms']['baseline']/case['median_ms']['candidate']
            output[0].fill_(torch.iinfo(torch.int32).min)
            workspace.zero_()
            torch.npu.synchronize(args.device)
            direct.launch(stream)
            torch.npu.synchronize(args.device)
            assert torch.equal(output[0],expected_indices), 'Candidate changed after timing'
        report['cases'].append(case)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
        del query,key,weights,output,expected_indices,expected_values
    report['complete'] = True
    args.output.write_text(json.dumps(report,indent=2))
    print('RESULT',args.output,flush=True)


if __name__ == '__main__':
    main()
