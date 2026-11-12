#include "optimizer_storage.h"
#include <iostream>
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F>void refuses(F&& f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}
  require(caught,"optimizer storage accepted invalid extent or insufficient capacity");}
}
int main(){try{
  using namespace tide::device_online;
  // Independent inventory: master/slots, live+next counters, 64-byte tile
  // errors, 4 int64 fields per owner, tile prefix, ten floats/three flags per group.
  const int64_t base=17*4+3*4+3*32+3*64+3*4*8+4*8+2*(10*4+3*8);
  int cases=0;
  for(bool adam:{false,true})for(bool extra:{false,true}){
    auto p=optimizer_storage_plan(17,3,3,2,adam,extra,extra);
    const auto expected=base+16*4*(adam?(extra?3:2):(extra?1:0));
    require(p.bytes==expected,"slot-specific optimizer inventory differs");
    require_resident_bytes(p.bytes,p.bytes,"optimizer");
    refuses([&]{require_resident_bytes(p.bytes,p.bytes-1,"optimizer");});++cases;
  }
  require(optimizer_storage_plan(1,0,0,0,true,false,false).bytes==152,"empty-layout sentinels omitted");
  refuses([]{optimizer_storage_plan(0,0,0,0,false,false,false);});
  refuses([]{optimizer_storage_plan(1,-1,0,0,false,false,false);});
  refuses([]{optimizer_storage_plan(1,0,-1,0,false,false,false);});
  refuses([]{optimizer_storage_plan(1,0,0,-1,false,false,false);});
  refuses([]{optimizer_storage_plan(INT64_MAX,1,1,1,true,false,false);});
  refuses([]{optimizer_storage_plan(1,INT64_MAX,0,0,false,false,false);});
  refuses([]{optimizer_storage_plan(1,0,INT64_MAX,0,false,false,false);});
  refuses([]{optimizer_storage_plan(1,0,0,INT64_MAX,false,false,false);});
  ResidentStorageBytes live;int master,slots,aliases;
  live.add(&master,4096);live.add(&master,4096);live.add(&slots,8192);live.add(&aliases,4096);
  require(live.bytes()==16384,"storage views double-counted or equal-valued owners merged");
  const auto tensor_total=live.bytes();live.add(&slots,8192);require(live.bytes()==tensor_total,"borrowed current-program slot charged twice");
  const auto total=resident_bytes_add(live.bytes(),resident_bytes_add(128,256));
  require(total==16768,"simultaneously live workspace arenas not summed");
  require_resident_bytes(total,total,"total");refuses([&]{require_resident_bytes(total,total-1,"total");});
  refuses([]{resident_bytes_add(INT64_MAX,1);});refuses([]{resident_bytes_multiply(INT64_MAX,4);});
  constexpr int64_t elements=14904826,owners=5809,allowance=402653184;
  const auto old_layout=elements*4+owners;
  require(old_layout>allowance/8,"LH176 original partition witness lost");
  // Any distribution into these nonempty canonical owners fits this tile bound;
  // this proves allocation arithmetic, not real LH device/whole-model admission.
  const auto tiles=(elements+255)/256+owners-1;
  const auto sgd=optimizer_storage_plan(elements,owners,tiles,1,false,false,false);
  const auto adam=optimizer_storage_plan(elements,owners,tiles,1,true,false,false);
  const auto amsgrad=optimizer_storage_plan(elements,owners,tiles,1,true,false,true);
  require(sgd.bytes<allowance&&adam.bytes<allowance&&amsgrad.bytes<allowance,"real slot inventories exceed witness budget");
  std::cout<<"optimizer-storage: passed variants="<<cases<<" empty=true aliases=true refusal=true old_layout="<<old_layout
    <<" old_allowance="<<allowance/8<<" sgd_bound="<<sgd.bytes<<" adamw_bound="<<adam.bytes
    <<" amsgrad_bound="<<amsgrad.bytes<<" optimizer_limit="<<allowance<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
