#pragma once
namespace tide::device_online {
// Stop commands can add retained control tensors/workspace. Admit the complete
// planned group before CANN service arena allocation/control recording, then repeat after
// finish because lowering may create additional storage (CUDA branch tables).
template<class Program,class Services,class Admission>
void finish_resident_program(Program& program,Services* services,Admission&& admit){
  if(services)services->append_stop();
  admit();
  if(services)services->finish();
  program.finish();
  admit();
}
}
