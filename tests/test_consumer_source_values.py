"""Exact named initial parameters, independent of graph execution or Torch."""
from pathlib import Path
import random
import subprocess


def test_fused_initializer_matches_integer_definition(tmp_path):
    source=Path(__file__).with_name('consumer_source_probe.cpp')
    binary=tmp_path/'source-probe'
    subprocess.run(['c++','-std=c++17','-O3',str(source),'-o',str(binary)],check=True)
    modulus=2**31-1
    names=['embedding','head','edge/2207/projection','node/479/fiber_qkv']
    seeds=[0,7,modulus-1,modulus,2**63-1]
    indices=[0,1,65535,65536,modulus-1,modulus,modulus+1,2*modulus,2**63-1]
    cases=[(name,seed,i) for name in names for seed in seeds for i in indices]
    rng=random.Random(31)
    cases.extend((names[i%4],rng.randrange(2**63),rng.randrange(2**63)) for i in range(200))
    wanted=[]
    for name,seed,index in cases:
        key=seed%modulus
        for byte in name.encode('ascii'):
            key=(key*131+byte)%modulus
        value=(index+key)%modulus
        for _ in range(3):
            value=(value*1103515245+12345)%modulus
        wanted.append((value%65536-32768)*2**-20)
    result=subprocess.check_output([str(binary)],input=''.join(f'{n} {s} {i}\n' for n,s,i in cases),text=True,timeout=10)
    assert list(map(float,result.splitlines()))==wanted
