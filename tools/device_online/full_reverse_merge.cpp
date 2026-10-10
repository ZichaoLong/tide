#include "full_reverse_merge.h"
#include "device_backend.h"
#include "device_launch_tide_full_reverse_merge.h"
namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
void append_full_reverse_merge(DeviceProgram& p,const at::Tensor& destinations,const at::Tensor& ids,
    const FullVjp& partial,const FullVjp& output,const at::Tensor& source_error,const at::Tensor& error) {
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_full_reverse_merge)(1,stream,
    ptr(destinations),ptr(ids),ptr(partial.content_connected),ptr(partial.comparison_connected),ptr(partial.parameter_connected),
    ptr(partial.chunks),ptr(source_error),ptr(output.content_connected),ptr(output.comparison_connected),
    ptr(output.parameter_connected),ptr(output.chunks),ptr(error),destinations.numel(),ids.numel(),output.parameter_connected.numel()),
    "merge Full owner reverse metadata");},
    {destinations,ids,partial.content_connected,partial.comparison_connected,partial.parameter_connected,partial.chunks,source_error,
     output.content_connected,output.comparison_connected,output.parameter_connected,output.chunks,error});
}
}
