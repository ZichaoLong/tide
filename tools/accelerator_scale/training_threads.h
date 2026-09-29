#pragma once
#include <ATen/Parallel.h>
#if defined(__unix__)
#include <dlfcn.h>
#endif

namespace accelerator_scale {
struct CpuThreadCounts { int aten=1, openblas=0; };
// Forward workers have joined before this scope. OpenBLAS is controlled
// separately because ATen's OpenMP setting need not reach its pthread pool.
class CpuPhaseThreads {
 public:
  explicit CpuPhaseThreads(int count):enabled_(count>0) {
    if(!enabled_)return;
#if defined(__unix__)
    get_=reinterpret_cast<int(*)()>(dlsym(RTLD_DEFAULT,"openblas_get_num_threads"));
    set_=reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT,"openblas_set_num_threads"));
#endif
    old_.aten=at::get_num_threads();old_.openblas=get_&&set_?get_():0;
    at::set_num_threads(count);if(old_.openblas)set_(count);
    current_={at::get_num_threads(),old_.openblas?get_():0};
  }
  ~CpuPhaseThreads() {
    if(enabled_){at::set_num_threads(old_.aten);if(old_.openblas)set_(old_.openblas);}
  }
  CpuThreadCounts current() const {return current_;}
  CpuPhaseThreads(const CpuPhaseThreads&)=delete;
  CpuPhaseThreads& operator=(const CpuPhaseThreads&)=delete;
 private:
  bool enabled_;
  int(*get_)()=nullptr;
  void(*set_)(int)=nullptr;
  CpuThreadCounts old_,current_;
};
}  // namespace accelerator_scale
