#include "owner_stream.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_owner_stream.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
int64_t address(const at::Tensor& x) {
  if(!x.defined())return 0;
  const auto value=reinterpret_cast<uint64_t>(x.data_ptr());int64_t out;std::memcpy(&out,&value,sizeof(out));return out;
}
void buffer(const at::Tensor& x,at::Device d,at::ScalarType dtype,int64_t elements) {
  if(!x.defined()||x.device()!=d||x.scalar_type()!=dtype||x.numel()!=elements||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid streamed owner buffer");
}
void sticky(CannProgram& p,const at::Tensor& from,const at::Tensor& to) {
  auto zero=at::zeros_like(from),same=at::empty({1},from.options().dtype(at::kBool)),index=at::empty_like(from);
  p.equal(from,zero,same);p.cast_index(same,index);auto bad=p.label(),done=p.label();
  p.branch(index,{bad,done});p.mark(bad);p.copy(to,from);p.mark(done);
}
}
int64_t owner_stream_metadata_bytes(int64_t fields,int64_t writes,bool peer) {
  const long double bytes=80.L*fields+56.L*writes+(peer?2.L:1.L)*fields+4096;
  if(fields<1||writes<fields||bytes>std::numeric_limits<int64_t>::max())
    throw std::invalid_argument("owner stream metadata extent overflow");
  return static_cast<int64_t>(bytes);
}
int64_t owner_stream_capacity(int64_t total,int64_t fields,int64_t writes,bool peer,int64_t budget) {
  const auto metadata=owner_stream_metadata_bytes(fields,writes,peer);
  if(total<1||budget<1||metadata>budget-(peer?8:4))throw std::invalid_argument("owner stream metadata/one element exceeds budget");
  return std::min<int64_t>({total,16LL*1024*1024,(budget-metadata)/(peer?8:4)});
}
OwnerStream append_owner_stream(CannProgram& source,CannProgram& destination,
    const std::vector<OwnerStreamField>& fields,const at::Tensor& source_error,const at::Tensor& destination_error,
    int64_t budget,bool accumulate,const OwnerStreamPackets& shared) {
  if(at::GradMode::is_enabled()||fields.empty()||!source_error.defined()||!destination_error.defined()||budget<1)
    throw std::invalid_argument("owner stream requires bounded no-grad fields");
  const auto from=source_error.device(),to=destination_error.device();const bool peer=from!=to;
  if(from.type()!=c10::DeviceType::PrivateUse1||to.type()!=from.type()||from.index()<0||to.index()<0)
    throw std::invalid_argument("owner stream requires explicit NPUs");
  buffer(source_error,from,at::kInt,1);buffer(destination_error,to,at::kInt,1);
  if(!peer && &source!=&destination)throw std::invalid_argument("local owner stream requires one program");
  std::vector<int64_t> send,receive,groups,writes;int64_t total=0;
  std::vector<at::Tensor> keep_source{source_error},keep_target{destination_error};
  for(const auto& field:fields) {
    const auto& x=field.source.values;
    if(!x.defined())throw std::invalid_argument("undefined streamed owner source");
    const auto size=x.numel();
    if(size<1||size>std::numeric_limits<int64_t>::max()-total||field.destinations.empty())
      throw std::invalid_argument("owner stream extent overflow/empty destination");
    buffer(x,from,at::kFloat,size);buffer(field.source.connected,from,at::kBool,1);
    if(accumulate)buffer(field.connected,to,at::kBool,1);
    else if(field.connected.defined())throw std::invalid_argument("publication has no gradient connection target");
    send.insert(send.end(),{total,size,address(x),address(field.source.connected)});
    receive.insert(receive.end(),{total,size,0,0});groups.push_back(writes.size()/7);
    keep_source.push_back(x);keep_source.push_back(field.source.connected);
    if(field.connected.defined())keep_target.push_back(field.connected);
    for(const auto& dest:field.destinations) {
      const auto& y=dest.values;const auto payload=dest.payload_dtype;
      if(!y.defined()||y.device()!=to||y.numel()!=size||y.requires_grad()
          ||(y.scalar_type()!=at::kFloat&&y.scalar_type()!=at::kHalf)||(payload!=at::kFloat&&payload!=at::kHalf)
          ||(payload==at::kFloat&&y.scalar_type()==at::kHalf)
          ||(!y.is_contiguous()&&(y.dim()!=2||y.stride(1)!=1||y.stride(0)<y.size(1)))
          ||(accumulate&&(y.scalar_type()!=at::kFloat||payload!=at::kFloat)))
        throw std::invalid_argument("invalid streamed owner destination");
      const int64_t rows=y.is_contiguous()?1:y.size(0),cols=size/rows,stride=y.is_contiguous()?cols:y.stride(0);
      writes.insert(writes.end(),{address(y),rows,cols,stride,y.scalar_type()==at::kHalf,payload==at::kHalf,address(field.connected)});
      keep_target.push_back(y);
    }
    groups.push_back(writes.size()/7);total+=size;
  }
  // Bound BOTH endpoints plus descriptor/counter allocations before upload.
  const auto metadata=owner_stream_metadata_bytes(fields.size(),writes.size()/7,peer);
  const auto cap=owner_stream_capacity(total,fields.size(),writes.size()/7,peer,budget);
  const bool reusable=shared.send.defined();
  if(shared.receive.defined()!=(reusable&&peer))throw std::invalid_argument("invalid shared owner packet pair");
  if(reusable) {
    buffer(shared.send,from,at::kFloat,shared.send.numel());
    if(shared.send.dim()!=1||shared.send.numel()<cap)throw std::invalid_argument("shared owner send packet is too small");
    if(peer) {
      buffer(shared.receive,to,at::kFloat,shared.receive.numel());
      if(shared.receive.dim()!=1||shared.receive.numel()<cap)throw std::invalid_argument("shared owner receive packet is too small");
    }
  }
  auto floats=at::TensorOptions().device(from).dtype(at::kFloat),longs=floats.dtype(at::kLong);
  auto make=[](const std::vector<int64_t>& v,at::Device d){return at::tensor(v,at::kLong).to(d);};
  auto sd=make(send,from),td=make(receive,to),group=make(groups,to),write=make(writes,to);
  auto packet=reusable?shared.send.narrow(0,0,cap):at::zeros({cap},floats);
  auto on=at::zeros({int64_t(fields.size())},floats.dtype(at::kBool));
  auto cursor=at::zeros({1},longs),branch=at::zeros({1},source_error.options());
  auto received=peer?(reusable?shared.receive.narrow(0,0,cap):at::zeros({cap},floats.device(to))):packet;
  auto flags=peer?at::zeros(on.sizes(),on.options().device(to)):on;
  auto position=peer?at::zeros({1},longs.device(to)):cursor;
  auto more=peer?at::zeros_like(destination_error):branch;
  auto status=peer?at::zeros_like(destination_error):source_error;
  OwnerStream out;out.capacity=cap;out.iterations=total/cap+(total%cap!=0);
  out.reserved_bytes=metadata+(reusable?0:(peer?8:4)*cap);
  if(peer)out.peer=std::make_unique<PeerExchange>(PeerExchange::Fields{{packet,received},{on,flags},
    {cursor,position},{branch,more},{source_error,status}},budget);
  source.zero(cursor);auto increment=at::full({1},cap,longs);
  auto emit=[&](CannProgram& p,int64_t mode) {
    const bool pack=mode==0;auto desc=pack?sd:td,values=pack?packet:received,live=pack?on:flags,at=pack?cursor:position;
    auto next=pack?branch:more,error=pack?source_error:destination_error;
    const auto count=int64_t(fields.size());
    auto keep=pack?keep_source:keep_target;keep.insert(keep.end(),{desc,values,live,at,next,error});
    if(!pack)keep.insert(keep.end(),{group,write});
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_owner_stream)(32,stream,
      ptr(desc),pack?ptr(desc):ptr(group),pack?ptr(desc):ptr(write),ptr(values),ptr(live),ptr(at),ptr(next),ptr(error),
      count,total,cap,mode),"stream packed canonical owner values");},keep);
  };
  auto head=source.label(),advance=source.label(),done=source.label();source.mark(head);emit(source,0);
  if(peer)out.peer->append_send(source);
  else {sticky(source,source_error,destination_error);emit(source,accumulate?1:2);}
  // Increment only when a further complete cursor position exists. In
  // particular the last partial packet must not overflow an int64 cursor.
  source.branch(branch,{done,advance});source.mark(advance);
  source.add(cursor,increment);source.branch(branch,{done,head});source.mark(done);
  if(peer) {
    auto head=destination.label(),done=destination.label();destination.mark(head);
    out.peer->append_receive(destination);sticky(destination,status,destination_error);emit(destination,accumulate?1:2);
    destination.branch(more,{done,head});destination.mark(done);
  }
  return out;
}
} // namespace tide::device_online
