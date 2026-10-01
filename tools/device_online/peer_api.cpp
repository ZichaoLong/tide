#include "peer_api.h"
#include "cann_api.h"
#include <dlfcn.h>
#include <stdexcept>

namespace tide::device_online {
PeerApi::PeerApi() {
  runtime=dlopen("libacl_rt.so",RTLD_NOW|RTLD_LOCAL);
  if(!runtime)throw std::runtime_error("CANN peer runtime unavailable");
  try {
#define LOAD(field,name) field=CannApi::symbol<decltype(field)>(runtime,name)
    LOAD(can,"aclrtDeviceCanAccessPeer");LOAD(enable,"aclrtDeviceEnablePeerAccess");
    LOAD(copy,"aclrtMemcpyAsync");LOAD(create,"aclrtCreateNotify");LOAD(destroy,"aclrtDestroyNotify");
    LOAD(export_key,"aclrtNotifyGetExportKey");LOAD(whitelist,"aclrtNotifySetImportPid");
    LOAD(import_key,"aclrtNotifyImportByKey");LOAD(record,"aclrtRecordNotify");LOAD(wait_reset,"aclrtWaitAndResetNotify");
#undef LOAD
  } catch(...) {dlclose(runtime);runtime=nullptr;throw;}
}
PeerApi::~PeerApi(){if(runtime)dlclose(runtime);}
} // namespace tide::device_online
