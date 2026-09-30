#include "cann_api.h"
#include <dlfcn.h>
#include <stdexcept>

namespace tide::device_online {
namespace {
void* library(const char* name) {
  auto handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (!handle) throw std::runtime_error(std::string("CANN control dependency unavailable: ")+name+": "+dlerror());
  return handle;
}
}
void* CannApi::find(void* library, const std::string& name) {
  dlerror(); auto address = dlsym(library, name.c_str()); const auto error = dlerror();
  if (error || !address) throw std::runtime_error("CANN control API unavailable: " + name);
  return address;
}
void CannApi::check(int status, const char* operation) {
  if (status) throw std::runtime_error(std::string(operation)+" failed with CANN status "+std::to_string(status));
}
CannApi::CannApi() {
  try {
    runtime = library("libascendcl.so");
    operators = library("libopapi.so"); metadata = library("libnnopbase.so");
#define LOAD(field, name) field = symbol<decltype(field)>(runtime, name)
    LOAD(create_stream, "aclrtCreateStreamWithConfig"); LOAD(destroy_stream, "aclrtDestroyStream");
    LOAD(sync_stream, "aclrtSynchronizeStreamWithTimeout");
    LOAD(begin_model, "aclmdlRIBuildBegin"); LOAD(bind_stream, "aclmdlRIBindStream");
    LOAD(end_task, "aclmdlRIEndTask"); LOAD(end_model, "aclmdlRIBuildEnd");
    LOAD(unbind_stream, "aclmdlRIUnbindStream"); LOAD(execute, "aclmdlRIExecuteAsync");
    LOAD(destroy_model, "aclmdlRIDestroy");
    LOAD(create_label, "aclrtCreateLabel"); LOAD(set_label, "aclrtSetLabel");
    LOAD(destroy_label, "aclrtDestroyLabel"); LOAD(create_list, "aclrtCreateLabelList");
    LOAD(destroy_list, "aclrtDestroyLabelList"); LOAD(jump, "aclrtSwitchLabelByIndex");
#undef LOAD
    create_tensor = symbol<decltype(create_tensor)>(metadata, "aclCreateTensor");
    destroy_tensor = symbol<decltype(destroy_tensor)>(metadata, "aclDestroyTensor");
    create_scalar = symbol<decltype(create_scalar)>(metadata, "aclCreateScalar");
    destroy_scalar = symbol<decltype(destroy_scalar)>(metadata, "aclDestroyScalar");
    create_int_array = symbol<decltype(create_int_array)>(metadata, "aclCreateIntArray");
    destroy_int_array = symbol<decltype(destroy_int_array)>(metadata, "aclDestroyIntArray");
  } catch (...) { release(); throw; }
}
void CannApi::release() noexcept {
  if (metadata) dlclose(metadata);
  if (operators) dlclose(operators);
  if (runtime) dlclose(runtime);
  metadata = operators = runtime = nullptr;
}
CannApi::~CannApi() { release(); }
}  // namespace tide::device_online
