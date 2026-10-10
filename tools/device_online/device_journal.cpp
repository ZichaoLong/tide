#include "device_journal.h"
#include "device_backend.h"
#include "device_launch_tide_journal.h"
#include <limits>
#include <stdexcept>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void stage(DeviceProgram& p,const DeviceJournal& j,const JournalProposal& in,const at::Tensor& error,bool commit) {
  auto metadata=j.meta,values=j.values,count=j.count;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_journal)(1,stream,ptr(in.meta),ptr(in.values),
    ptr(in.count),ptr(metadata),ptr(values),ptr(count),ptr(error),metadata.size(0),in.meta.size(0),metadata.size(1),
    values.size(1),int64_t(commit)),"device journal transaction");},{in.meta,in.values,in.count,metadata,values,count,error});
}
}
DeviceJournal::DeviceJournal(int64_t capacity,int64_t columns,int64_t width,at::Device device) {
  if(capacity<1||columns<1||width<1||capacity>std::numeric_limits<int64_t>::max()/8/std::max(columns,width))
    throw std::invalid_argument("invalid journal dimensions");
  auto opts=at::TensorOptions().device(device);
  meta=at::zeros({capacity,columns},opts.dtype(at::kLong));values=at::zeros({capacity,width},opts.dtype(at::kFloat));
  count=at::zeros({1},opts.dtype(at::kLong));
}
JournalProposal DeviceJournal::propose(DeviceProgram& p,const at::Tensor& m,const at::Tensor& v,
                                      const at::Tensor& n,const at::Tensor& error) const {
  if(m.dim()!=2||v.dim()!=2||m.size(0)!=v.size(0)||m.size(1)!=meta.size(1)||v.size(1)!=values.size(1)
      ||m.scalar_type()!=at::kLong||(v.scalar_type()!=at::kFloat&&v.scalar_type()!=at::kHalf)||n.sizes()!=at::IntArrayRef{1}||n.scalar_type()!=at::kLong
      ||m.device()!=meta.device()||v.device()!=meta.device()||n.device()!=meta.device()
      ||!m.is_contiguous()||!v.is_contiguous()||!n.is_contiguous()
      ||m.is_alias_of(meta)||v.is_alias_of(values))throw std::invalid_argument("invalid journal input");
  // Diagnostics and future adjoints retain exact stored half values widened
  // to FP32. This bulk cast is part of the device program, not a host loop.
  auto data=v;
  if(v.scalar_type()==at::kHalf){data=at::empty(v.sizes(),values.options());p.cast(v,data);}
  JournalProposal in{m,data,n};stage(p,*this,in,error,false);return in;
}
void DeviceJournal::commit(DeviceProgram& p,const JournalProposal& in,const at::Tensor& error) const {stage(p,*this,in,error,true);}
} // namespace tide::device_online
