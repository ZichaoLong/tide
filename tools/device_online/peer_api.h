#pragma once
#include <cstddef>
#include <cstdint>

namespace tide::device_online {
// Optional CANN9 acl_rt.h signatures. Kept outside the portable graph core and
// independent of either TorchNPU runtime owner's headers.
struct PeerApi {
  void* runtime=nullptr;
  PeerApi();
  ~PeerApi();
  PeerApi(const PeerApi&)=delete;
  PeerApi& operator=(const PeerApi&)=delete;
  int (*can)(int32_t*,int32_t,int32_t);
  int (*enable)(int32_t,uint32_t);
  int (*copy)(void*,size_t,const void*,size_t,int,void*);
  int (*create)(void**,uint64_t);
  int (*destroy)(void*);
  int (*export_key)(void*,char*,size_t,uint64_t);
  int (*whitelist)(void*,int32_t*,size_t);
  int (*import_key)(void**,const char*,uint64_t);
  int (*record)(void*,void*);
  int (*wait_reset)(void*,void*,uint32_t);
};
} // namespace tide::device_online
