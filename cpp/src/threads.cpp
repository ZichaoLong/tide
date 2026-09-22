#include "portable_torch/threads.hpp"
#include <ATen/Parallel.h>
#include <dlfcn.h>

namespace portable_torch {
namespace {
double count(const char* name) {
  auto get = reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, name));
  return get ? get() : -1;
}
}
std::map<std::string, double> thread_metrics() {
  return {{"runtime/aten_threads", at::get_num_threads()},
          {"runtime/interop_threads", at::get_num_interop_threads()},
          {"runtime/openblas_reported_threads", count("openblas_get_num_threads")},
          {"runtime/mkl_reported_threads", count("MKL_Get_Max_Threads")},
          {"runtime/omp_max_threads", count("omp_get_max_threads")}};
}
std::string blas_description() {
  auto get = reinterpret_cast<const char* (*)()>(dlsym(RTLD_DEFAULT, "openblas_get_config"));
  return get ? std::string(get()) : "OpenBLAS introspection unavailable";
}
} // namespace portable_torch
