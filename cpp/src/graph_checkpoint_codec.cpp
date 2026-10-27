// Native graph checkpoint v1: exact raw fields, never effective-state projection.
#include "graph_checkpoint_internal.h"
#include "checkpoint_binary.h"
#include <string_view>
namespace tide::checkpoint_detail {
namespace {
constexpr std::string_view magic="TIDEGCK1";
void count(Writer& w,size_t n){if(n>kMaxCollection)fail("graph collection is too large");w.u64(n);}
uint64_t count(Reader& r){auto n=r.u64();if(n>kMaxCollection)fail("graph collection is too large");return n;}
void owner(Writer& w,const Owner& o){w.i64(o.first);w.i64(o.second);}
Owner owner(Reader& r){auto b=r.i64(),n=r.i64();return {b,n};}
void tensors(Writer& w,const std::map<std::string,Tensor>& ts){
  count(w,ts.size());for(const auto& [name,t]:ts){w.string(name);w.tensor(t,true);}
}
std::map<std::string,Tensor> tensors(Reader& r){
  std::map<std::string,Tensor> ts;auto n=count(r);
  for(uint64_t i=0;i<n;++i){auto name=r.string();auto t=r.tensor(true);
    if(!ts.emplace(name,std::move(t)).second)fail("duplicate graph tensor field");}
  return ts;
}
void continuation(Writer& w,const Continuation& q){
  w.string(q.identity);w.i64(q.batch_size);w.i64(q.cut);
  count(w,q.states.size());
  for(const auto& [o,s]:q.states){owner(w,o);w.i64(s.last_time);w.i64(s.observations);w.tensor(s.value);tensors(w,s.slots);}
  count(w,q.history.size());
  for(const auto& [o,h]:q.history){
    owner(w,o);w.i64(h.last_time);count(w,h.scalars.size());
    for(const auto& [name,value]:h.scalars){w.string(name);w.i64(value);}
    count(w,h.node_maps.size());
    for(const auto& [name,m]:h.node_maps){w.string(name);count(w,m.size());for(const auto& [n,v]:m){w.i64(n);w.i64(v);}}
    tensors(w,h.tensors);
  }
  count(w,q.pending.size());
  for(const auto& a:q.pending){for(auto v:{a.batch,a.node,a.time,a.kind,a.source,a.position})w.i64(v);w.tensor(a.value);}
  count(w,q.ledger.size());for(const auto& [o,last]:q.ledger){owner(w,o);owner(w,last);}
}
Continuation continuation(Reader& r){
  Continuation q;q.identity=r.string();q.batch_size=r.i64();q.cut=r.i64();
  auto n=count(r);
  for(uint64_t i=0;i<n;++i){auto o=owner(r);State s;s.last_time=r.signed64();s.observations=r.i64();
    s.value=r.tensor();s.slots=tensors(r);if(!q.states.emplace(o,std::move(s)).second)fail("duplicate state owner");}
  n=count(r);
  for(uint64_t i=0;i<n;++i){auto o=owner(r);History h;h.last_time=r.signed64();auto size=count(r);
    for(uint64_t j=0;j<size;++j){auto name=r.string();auto value=r.signed64();
      if(!h.scalars.emplace(name,value).second)fail("duplicate history scalar");}
    size=count(r);
    for(uint64_t j=0;j<size;++j){auto name=r.string();std::map<Index,Index> m;auto entries=count(r);
      for(uint64_t k=0;k<entries;++k){auto node=r.i64(),value=r.signed64();if(!m.emplace(node,value).second)fail("duplicate history node");}
      if(!h.node_maps.emplace(name,std::move(m)).second)fail("duplicate history map");}
    h.tensors=tensors(r);if(!q.history.emplace(o,std::move(h)).second)fail("duplicate history owner");}
  n=count(r);q.pending.reserve(n);
  for(uint64_t i=0;i<n;++i){Atom a;a.batch=r.i64();a.node=r.i64();a.time=r.i64();a.kind=r.i64();a.source=r.i64();a.position=r.i64();a.value=r.tensor();q.pending.push_back(std::move(a));}
  n=count(r);
  for(uint64_t i=0;i<n;++i){auto o=owner(r),last=owner(r);if(!q.ledger.emplace(o,last).second)fail("duplicate ledger owner");}
  return q;
}
}
std::vector<uint8_t> encode_graph(const GraphRecord& record){
  if(!little_endian())fail("big-endian graph checkpoints are unsupported");
  Writer w;w.raw(reinterpret_cast<const uint8_t*>(magic.data()),magic.size());w.u32(1);w.u8(1);
  auto parameters=encode(record.parameters);w.u64(parameters.size());w.raw(parameters.data(),parameters.size());
  continuation(w,record.continuation);w.u64(checksum(w.bytes.data(),w.bytes.size()));return std::move(w.bytes);
}
GraphRecord decode_graph(const std::vector<uint8_t>& file){
  if(!little_endian()||file.size()<21)fail("unsupported host or truncated graph checkpoint");
  const auto limit=file.size()-8;Reader tail{file,file.size(),limit};
  if(checksum(file.data(),limit)!=tail.u64())fail("graph checkpoint checksum mismatch");
  Reader r{file,limit};for(auto c:magic)if(r.u8()!=static_cast<uint8_t>(c))fail("graph checkpoint magic mismatch");
  if(r.u32()!=1||r.u8()!=1)fail("unsupported graph checkpoint schema/byte order");
  auto n=r.u64();if(n>r.limit-r.offset)fail("truncated parameter record");
  std::vector<uint8_t> blob(file.begin()+r.offset,file.begin()+r.offset+n);r.offset+=n;
  GraphRecord record;record.parameters=decode(blob);record.continuation=continuation(r);
  if(r.offset!=limit)fail("trailing graph checkpoint payload");return record;
}
}
