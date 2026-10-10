#include "state_shard_pack.h"
#include "device_backend.h"
#include "device_launch_tide_state_shard_pack.h"
#include "device_launch_tide_state_shard_selection.h"
#include <stdexcept>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
StateShardBatch append_state_shard_pack(DeviceProgram& p,const ReadyBatch& ready,const ContentBatch& content,
    const at::Tensor& mapping,int64_t local_nodes,int64_t capacity,const at::Tensor& error) {
  const auto rows=ready.fibers.size(0),width=content.content.size(1),nodes=mapping.numel();
  if(capacity<1||capacity>rows||local_nodes<0||mapping.scalar_type()!=at::kLong
      ||!mapping.is_contiguous()||mapping.device()!=error.device())
    throw std::invalid_argument("invalid state shard packing bounds/map");
  const auto longs=mapping.options(),opts=content.content.options();
  StateShardBatch out;
  out.ready.atoms={at::zeros({capacity,6},longs),at::empty({capacity,width},opts),at::zeros({capacity},opts.dtype(at::kBool))};
  out.ready.fibers=at::zeros({capacity,4},longs);out.ready.fiber_offsets=at::zeros({capacity+1},longs);
  out.ready.counts=at::zeros({3},longs);out.ready.branch=at::zeros_like(error);
  out.fiber_rows=at::empty({capacity},longs);out.atom_rows=at::empty_like(out.fiber_rows);out.destinations=at::empty_like(out.fiber_rows);
  out.content={at::empty({capacity,width},opts),at::zeros({capacity},opts.dtype(at::kFloat)),at::empty({capacity,width},opts)};
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_state_shard_pack)(1,stream,
    ptr(ready.fibers),ptr(ready.fiber_offsets),ptr(ready.counts),ptr(ready.atoms.coordinates),ptr(mapping),
    ptr(out.ready.fibers),ptr(out.ready.fiber_offsets),ptr(out.ready.counts),ptr(out.ready.atoms.coordinates),ptr(out.ready.atoms.valid),
    ptr(out.fiber_rows),ptr(out.atom_rows),ptr(out.destinations),ptr(out.ready.branch),ptr(error),rows,capacity,nodes,local_nodes),
    "pack complete state owner fibers");},
    {ready.fibers,ready.fiber_offsets,ready.counts,ready.atoms.coordinates,mapping,out.ready.fibers,out.ready.fiber_offsets,
     out.ready.counts,out.ready.atoms.coordinates,out.ready.atoms.valid,out.fiber_rows,out.atom_rows,out.destinations,out.ready.branch,error});
  auto gather=[&](const at::Tensor& input,const at::Tensor& indices,const at::Tensor& output) {
    auto padded=at::zeros({rows+1,width},input.options());
    p.copy(padded.narrow(0,0,rows),input);p.index_select(padded,0,indices,output);
  };
  gather(ready.atoms.values,out.atom_rows,out.ready.atoms.values);
  gather(content.weighted,out.atom_rows,out.content.weighted);
  gather(content.content,out.fiber_rows,out.content.content);
  return out;
}
SelectionProposal append_state_shard_selection(DeviceProgram& p,const StateShardBatch& shard,
    const SelectionProposal& global,const at::Tensor& error) {
  const auto rows=shard.ready.fibers.size(0),global_rows=global.active.numel();
  SelectionProposal out{{},at::zeros({rows},global.active.options()),at::zeros({rows},global.controls.options()),shard.ready.branch};
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_state_shard_selection)(1,stream,
    ptr(shard.fiber_rows),ptr(shard.ready.counts),ptr(global.active),ptr(global.controls),ptr(out.active),ptr(out.controls),ptr(error),
    rows,global_rows),"pack global selection for state owner");},
    {shard.fiber_rows,shard.ready.counts,global.active,global.controls,out.active,out.controls,error});
  return out;
}
void append_state_shard_scatter(DeviceProgram& p,const StateShardBatch& shard,const at::Tensor& source,const at::Tensor& destination) {
  if(source.size(0)!=shard.destinations.numel()||source.scalar_type()==at::kBool)
    throw std::invalid_argument("state shard scatter needs numeric rows; Bool metadata has a fused path");
  p.index_copy(destination,0,shard.destinations,source);
}
} // namespace tide::device_online
