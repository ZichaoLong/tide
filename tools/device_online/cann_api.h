#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace tide::device_online {
// Backend-local ABI boundary for optional CANN 9 public runtime-model APIs.
// Exact names/signatures come from acl_rt.h, acl_meta.h and aclnn_*.h.
// Loading alone is not a feature claim; the executable check must run on device.
struct CannApi {
  void *runtime = nullptr, *operators = nullptr, *metadata = nullptr;
  CannApi();
  ~CannApi();
  CannApi(const CannApi&) = delete;
  CannApi& operator=(const CannApi&) = delete;
  static void check(int status, const char* operation);
  template<class F> static F symbol(void* library, const std::string& name) {
    return reinterpret_cast<F>(find(library, name));
  }
  int (*create_stream)(void**, uint32_t, uint32_t);
  int (*destroy_stream)(void*);
  int (*sync_stream)(void*, int32_t);
  int (*begin_model)(void**, uint32_t);
  int (*bind_stream)(void*, void*, uint32_t);
  int (*end_task)(void*, void*);
  int (*end_model)(void*, void*);
  int (*unbind_stream)(void*, void*);
  int (*execute)(void*, void*);
  int (*destroy_model)(void*);
  int (*create_label)(void**);
  int (*set_label)(void*, void*);
  int (*destroy_label)(void*);
  int (*create_list)(void**, size_t, void**);
  int (*destroy_list)(void*);
  int (*jump)(void*, uint32_t, void*, void*);
  void* (*create_tensor)(const int64_t*, uint64_t, int, const int64_t*, int64_t,
                         int, const int64_t*, uint64_t, void*);
  int (*destroy_tensor)(const void*);
  void* (*create_scalar)(void*, int);
  int (*destroy_scalar)(const void*);
 private:
  static void* find(void*, const std::string&);
  void release() noexcept;
};
}  // namespace tide::device_online
