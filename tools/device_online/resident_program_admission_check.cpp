// CPU-only admission-order witness. No device program or runtime is loaded.
#include "resident_program_admission.h"
#include "resident_storage_budget.h"
#include <iostream>
using namespace tide::device_online;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct Inventory {int64_t tensors=10,workspace=4;int admissions=0;bool stop=false,arena=false,recorded=false,executed=false;};
struct Program {
  Inventory& used;int64_t lowering_bytes=0;
  void finish(){used.recorded=true;used.tensors+=lowering_bytes;}
  void run(){used.executed=true;}
};
struct Services {
  Inventory& used;
  void append_stop(){used.stop=true;used.tensors+=2;used.workspace=8;}
  void finish(){used.arena=true;}
};
void execute(Inventory& used,int64_t partition,int64_t total,int64_t lowering,bool services){
  Program program{used,lowering};Services calls{used};
  finish_resident_program(program,services?&calls:nullptr,[&]{
    ++used.admissions;require_resident_bytes(used.tensors,partition,"test tensor");
    require_resident_bytes(resident_bytes_add(used.tensors,used.workspace),total,"test total");
  });
  program.run();
}
void refusal(int64_t partition,int64_t total,int64_t lowering,bool after){
  Inventory used;bool refused=false;
  try{execute(used,partition,total,lowering,true);}catch(const std::invalid_argument&){refused=true;}
  require(refused&&used.stop&&!used.executed,"refusal executed a prefix or missed stop storage");
  require(used.arena==after&&used.recorded==after,"pre-admission allocated or recorded a rejected group");
  require(used.admissions==(after?2:1),"pre/post admission count differs");
}
}
int main(){
  refusal(11,100,0,false); // Stop control storage pushes the tensor partition over.
  refusal(100,19,0,false); // Known tensors fit; planned shared arena does not.
  refusal(100,20,1,true); // Lowering adds one byte after the exact pre-limit.
  Inventory exact;execute(exact,12,20,0,true);
  require(exact.admissions==2&&exact.arena&&exact.executed,"exact group limit did not execute");
  Inventory standalone;execute(standalone,13,17,3,false);
  require(standalone.admissions==2&&!standalone.stop&&!standalone.arena&&standalone.executed,
    "single-program pre/post admission differs");
  std::cout<<"resident-program-admission: passed stop_storage=true planned_workspace=true post_lowering=true exact_limit=true no_services=true\n";
}
