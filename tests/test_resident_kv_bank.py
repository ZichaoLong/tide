"""Unique KV bank and actual raw decoder against an independent list recurrence."""
import ctypes
from pathlib import Path
import subprocess
import pytest
import torch
from test_resident_block_sequence import Fixture


@pytest.fixture(scope="module")
def bank_kernel(tmp_path_factory):
    root = Path(__file__).resolve().parents[1] / "tools/device_online"
    work = tmp_path_factory.mktemp("kv_bank")
    source, lib = work / "adapter.cpp", work / "adapter.so"
    source.write_text('#define TIDE_KERNEL_CPU\n'+''.join(f'#include "tide_block_kv_{name}.cpp"\n'
                      for name in ("bank_plan","bank_decode","bank_numerical_vjp","bank_snapshot_vjp"))+
                      'extern "C" void reset(){tide_cuda::worker=0;tide_cuda::workers=1;tide_cuda::reset_arena();}\n')
    subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-I"+str(root/"cuda"),
                    "-I"+str(root/"ascendc"), "-I"+str(Path(torch.__file__).parent/"include"),
                    str(source), "-o", str(lib)], check=True)
    dll = ctypes.CDLL(str(lib))
    dll.tide_block_kv_bank_plan.argtypes = [ctypes.c_void_p]*18+[ctypes.c_int64]*10
    dll.tide_block_kv_bank_decode.argtypes = [ctypes.c_void_p]*11+[ctypes.c_int64]*6
    dll.tide_block_kv_bank_numerical_vjp.argtypes = [ctypes.c_void_p]*9+[ctypes.c_int64]*7
    dll.tide_block_kv_bank_snapshot_vjp.argtypes = [ctypes.c_void_p]*11+[ctypes.c_int64]*10
    return dll


class Bank:
    def __init__(self, fixture, bound=False):
        f = self.f = fixture
        self.bound=bound
        o, k = f.owners, f.capacity+(0 if bound else f.length)
        self.entries=k
        self.keys, self.values = torch.empty(o, k, 2), torch.empty(o, k, 2)
        self.summaries, self.weights = torch.empty(o, k, 1), torch.empty(o, k)
        self.stamps = torch.empty(o, k, dtype=torch.int64)
        self.valid = torch.empty(o, k, dtype=torch.bool)
        self.written = torch.empty(f.rows, dtype=torch.bool)
        self.old_map = torch.empty(f.rows, f.capacity, dtype=torch.int64)
        self.proposal_map = torch.empty_like(self.old_map)

    def run(self, kernel):
        f = self.f
        kernel.reset()
        arrays = (f.fields, f.initial, f.initial_i, f.write, f.write_i, f.times,
                  self.keys, self.values, self.summaries, self.weights, self.stamps,
                  self.valid, self.written, self.old_map, self.proposal_map, f.old_i, f.proposal_i, f.error)
        kernel.tide_block_kv_bank_plan(*(x.data_ptr() for x in arrays), f.batch, f.length,
            f.nodes, f.capacity, 2, 1, f.width, f.write_width, f.horizon, int(self.bound))

    def decode(self, kernel, mapping, metadata, length):
        f = self.f
        kernel.reset()
        output = torch.empty(f.owners*length, f.width)
        arrays = (f.fields, f.initial, f.initial_i, self.keys, self.values, self.summaries,
                  self.weights, mapping, metadata, output, f.error)
        kernel.tide_block_kv_bank_decode(*(x.data_ptr() for x in arrays), f.batch, length,
                                         f.nodes, f.capacity, self.entries, f.width)
        return output


def test_bank_views_decode_all_three_raw_snapshots(bank_kernel):
    f = Fixture(); bank = Bank(f); bank.run(bank_kernel)
    assert not f.error.item()
    final_ids = torch.tensor([(b*f.length+f.length-1)*f.nodes+n
                              for b in range(f.batch) for n in range(f.nodes)])
    views = ((bank.old_map, f.old_i, f.length), (bank.proposal_map, f.proposal_i, f.length),
             (bank.proposal_map[final_ids], f.proposal_i[final_ids], 1))
    for (mapping, metadata, length), expected in zip(views, f.reference()):
        actual = bank.decode(bank_kernel, mapping, metadata, length)
        for row, state in enumerate(expected):
            values, integers = f.pack(state)
            assert torch.equal(actual[row], values)
            assert torch.equal(metadata[row], integers)
    assert not f.error.item()
    assert bank.keys.numel() == f.owners*(f.capacity+f.length)*2
    assert bank.written.tolist() == f.write_i[:, 1].bool().tolist()


def test_no_writes_preserve_opaque_raw_padding(bank_kernel):
    f = Fixture(); f.write_i.zero_(); f.write_i[:, 0] = -1
    f.initial[1, 3:] = float("nan")
    bank = Bank(f); bank.run(bank_kernel)
    value = bank.decode(bank_kernel, bank.proposal_map, f.proposal_i, f.length)
    expected = f.initial.reshape(f.batch, 1, f.nodes, f.width).expand(-1, f.length, -1, -1).reshape(f.rows, f.width)
    assert not f.error.item()
    torch.testing.assert_close(value, expected, equal_nan=True)


def test_bound_read_zero_bank_has_no_projected_writes(bank_kernel):
    f=Fixture();bank=Bank(f,bound=True);bank.run(bank_kernel)
    value=bank.decode(bank_kernel,bank.old_map,f.old_i,f.length)
    expected=f.initial.reshape(f.batch,1,f.nodes,f.width).expand(-1,f.length,-1,-1).reshape(f.rows,f.width)
    torch.testing.assert_close(value,expected)
    assert not f.error.item() and not bank.written.any()
    assert bank.keys.size(1)==f.capacity


@pytest.mark.parametrize("family",["old","proposal","final","all","none","zero"])
def test_shared_bank_raw_transpose_and_causal_evicted_roots(bank_kernel,family):
    f=Fixture();bank=Bank(f);bank.run(bank_kernel);refs=f.reference()
    final_ids=torch.tensor([(b*f.length+f.length-1)*f.nodes+n for b in range(f.batch) for n in range(f.nodes)])
    mappings=(bank.old_map,bank.proposal_map,bank.proposal_map[final_ids])
    metadata=(f.old_i,f.proposal_i,f.proposal_i[final_ids]);generator=torch.Generator().manual_seed(57)
    initial=torch.zeros_like(f.initial);writes=torch.zeros_like(f.write)
    initial_flags=torch.zeros(f.owners,5,dtype=torch.bool);write_flags=torch.zeros(f.rows,5,dtype=torch.bool)
    outputs=[];cotangents=[]
    for name,states,mapping,meta,prefix in zip(("old","proposal","final"),refs,mappings,metadata,(0,1,f.length)):
        bar=torch.full((len(states),f.width),float("nan"));flags=torch.zeros(len(states),5,dtype=torch.bool)
        for row,state in enumerate(states):
            for field,((offset,_,_,_),value) in enumerate(zip(f.fields.tolist(),state[3])):
                if family in (name,"all","zero") and (row+field)%3!=1:
                    g=torch.randn(value.shape,generator=generator) if family!="zero" else torch.zeros_like(value)
                    flags[row,field]=True;bar[row,offset:offset+value.numel()]=g.flatten();outputs.append(value);cotangents.append(g)
        arrays=(f.fields,bank.written,mapping,meta,bar,flags,initial,initial_flags,writes,write_flags,f.error)
        bank_kernel.tide_block_kv_bank_snapshot_vjp(*(x.data_ptr() for x in arrays),f.batch,f.length,f.nodes,f.capacity,
            bank.entries,f.width,f.write_width,len(states)//f.owners,prefix,1)
    assert not f.error.item()
    leaves=[x for state in f.initial_fields+f.write_fields for x in state]
    expected=torch.autograd.grad(outputs,leaves,cotangents,allow_unused=True) if outputs else [None]*len(leaves)
    position=0
    for kind,states,values,flags in (("initial",f.initial_fields,initial,initial_flags),("write",f.write_fields,writes,write_flags)):
        for row,state in enumerate(states):
            for field,((off,wo,_,_),value) in enumerate(zip(f.fields.tolist(),state)):
                gradient=expected[position];position+=1;offset=off if kind=="initial" else wo
                assert bool(flags[row,field])==(gradient is not None),(kind,row,field)
                actual=values[row,offset:offset+value.numel()].reshape(value.shape)
                torch.testing.assert_close(actual,gradient if gradient is not None else torch.zeros_like(value))


def test_unique_bank_numerical_transpose_is_owner_separated(bank_kernel):
    f=Fixture();bank=Bank(f);bank.run(bank_kernel)
    bars=[torch.arange(x.numel()).reshape(x.shape).float()*.003 for x in (bank.keys,bank.values,bank.summaries,bank.weights)]
    initial=torch.zeros_like(f.initial);writes=torch.zeros_like(f.write)
    arrays=(f.fields,bank.valid,*bars,initial,writes,f.error)
    bank_kernel.tide_block_kv_bank_numerical_vjp(*(x.data_ptr() for x in arrays),f.batch,f.length,f.nodes,f.capacity,bank.entries,f.width,f.write_width)
    assert not f.error.item()
    for owner in range(f.owners):
        sample,node=divmod(owner,f.nodes)
        for entry in range(bank.entries):
            for field,bar in enumerate(bars,1):
                off,wo,width,_=f.fields[field].tolist()
                if entry<f.capacity:actual=initial[owner,off+entry*width:off+(entry+1)*width]
                else:actual=writes[(sample*f.length+entry-f.capacity)*f.nodes+node,wo:wo+width]
                want=bar[owner,entry].reshape(-1) if bank.valid[owner,entry] else torch.zeros(width)
                torch.testing.assert_close(actual,want)


@pytest.mark.parametrize("fault,code", [("capacity",79),("clock",78),("overflow",78),("raw",78)])
def test_bank_rejects_real_state_faults(bank_kernel, fault, code):
    f = Fixture(horizon=0 if fault=="capacity" else 7)
    if fault == "clock": f.times[0,0,0] = f.initial_i[0,0]
    if fault == "overflow": f.initial_i[0,1] = 2**63-1
    if fault == "raw": f.initial[0,18] = -1
    bank = Bank(f); bank.run(bank_kernel)
    assert f.error.item() == code
