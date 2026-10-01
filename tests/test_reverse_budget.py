"""CPU-only reservation invariants, including full-width geometry without allocation."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_reverse_budget_boundaries_and_monotonicity(tmp_path):
    source = tmp_path / "check.cpp"
    source.write_text(r'''
#include "reverse_budget.h"
#include "reverse_statistics.h"
#include <cassert>
using namespace tide::device_online;
int main() {
  int successes=0, refusals=0, shrink=0;
  for(bool half:{false,true})for(int64_t w:{4,32,128,2048})for(int64_t maximum:{1,16,64}) {
    int64_t last=0;
    for(int64_t budget=4096;budget<=128*1024*1024;budget*=2) {
      try {
        auto p=plan_fiber_reverse(32,8,w,4,128,8,16,maximum,half,budget);
        assert(p.owner_rows>=last && p.owner_rows<=32 && p.owner_rows<=maximum);
        assert(p.query_rows>=1 && p.query_rows<=maximum && p.query_rows<=8*p.owner_rows);
        assert(p.key_rows>=1 && p.key_rows<=64 && p.tensor_bytes<=budget);
        auto local=plan_fiber_vjp(p.owner_rows,8,w,4,128,8,p.query_rows,p.key_rows,half,budget/2);
        assert(local.query_rows==p.query_rows && local.key_rows==p.key_rows);
        last=p.owner_rows;++successes;shrink+=p.owner_rows<std::min<int64_t>(32,maximum);
      } catch(const std::invalid_argument&) { assert(last==0);++refusals; }
    }
  }
  assert(successes>0 && refusals>0 && shrink>0);
  for(bool half:{false,true})for(int64_t kvheads:{1,4}) {
    auto large=plan_event_reverse(32,32,4,kvheads,128,16,half,16*1024*1024);
    assert(large.owner_rows==16 && large.key_rows==64);
    auto small=plan_event_reverse(32,32,4,kvheads,128,16,half,512*1024);
    assert(small.owner_rows<large.owner_rows && small.tensor_bytes<=small.budget);
  }
  for(int which:{0,1,2}) {
    bool refused=false;
    try {
      if(which==0)plan_fiber_reverse(32,8,2048,4,128,8,16,16,false,1);
      if(which==1)plan_event_reverse(32,32,4,3,128,16,false,1<<20);
      if(which==2)plan_fiber_vjp(4,129,32,4,128,8,16,64,false,1<<20);
    } catch(const std::invalid_argument&) {refused=true;}
    assert(refused);
  }
  assert(reverse_budget::largest(INT64_MAX,[](int64_t x){return x<=INT64_MAX-2;})==INT64_MAX-2);
  ReverseStatistics a,b;
  record_reverse_plan(a,{4,2,64,16,100,200},"fiber");
  record_reverse_plan(b,{2,1,8,16,50,100},"fiber");merge_reverse_statistics(a,b);
  assert(a.at("reverse_fiber_groups")==2 && a.at("reverse_fiber_owner_rows_min")==2
    && a.at("reverse_fiber_owner_rows_max")==4 && a.at("reverse_fiber_tensor_bytes")==150);
}
''')
    binary = tmp_path / "check"
    subprocess.run(["c++", "-std=c++17", "-O2", "-I"+str(ROOT/"tools/device_online"),
                    str(source), "-o", str(binary)], check=True, timeout=30)
    subprocess.run([str(binary)], check=True, timeout=10)
