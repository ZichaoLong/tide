#include "packed_event_attention.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace tide::device_online {
namespace {
std::map<std::pair<int64_t,int64_t>,int64_t> geometries(const ContentProfile& p) {
  std::map<std::pair<int64_t,int64_t>,int64_t> result;
  for(const auto& n:p.graph.nodes)if(!n.identity&&n.memory=="attention")++result[{n.query_heads,n.kv_heads}];
  return result;
}
std::pair<long double,long double> footprint(const ContentProfile& p,const Continuation& q,const ContentLimits& l) {
  long double fixed=0,row=0;const auto w=p.width;
  for(const auto& [heads,count]:geometries(p)) {
    const auto kv=w/heads.first*heads.second;
    fixed+=24.L*(q.batch_size*static_cast<long double>(count)*l.kv_rows+1)*(2.L*kv+1)
      +64.L*(count+1.L)*w*w+128.L*(l.queue+1.L)*(w+8.L)
      +(l.diagnostics?24.L*l.kv_trace_rows*(2.L*kv+8):0);
    row+=96.L*w*w+256.L*w+256+32.L*l.kv_rows*w+24.L*heads.first*l.kv_rows;
  }
  return {fixed,row};
}
}
long double PackedEventAttention::minimum_bytes(const ContentProfile& p,const Continuation& q,const ContentLimits& l) {
  const auto [fixed,row]=footprint(p,q,l);return fixed+row;
}
PackedEventAttention::PackedEventAttention(const ContentProfile& p,const Continuation& q,at::Device d,const ContentLimits& l,int64_t budget)
    :rows_(l.queue),width_(p.width) {
  if(l.kv_rows<1||l.attention_chunk_rows<1||budget<1||(l.diagnostics&&l.kv_trace_rows<1))
    throw std::invalid_argument("invalid event attention cache limits");
  const auto [fixed,row]=footprint(p,q,l);
  if(row<=0||fixed+row>budget)throw std::invalid_argument("event attention cache and one query row exceed workspace budget");
  chunk_=static_cast<int64_t>(std::min<long double>({static_cast<long double>(rows_),
    static_cast<long double>(l.attention_chunk_rows),(budget-fixed)/row}));
  reserved_=static_cast<int64_t>(fixed+row*chunk_);
  for(const auto& [heads,_]:geometries(p))groups_.push_back(std::make_unique<EventAttentionGroup>(p,q,d,l,heads.first,heads.second,chunk_));
}
EventAttentionGroup::EventAttentionGroup(const ContentProfile& p,const Continuation& q,at::Device device,
    const ContentLimits& l,int64_t h,int64_t kh,int64_t c)
    :nodes(p.graph.nodes.size()),width(p.width),query_heads(h),kv_heads(kh),head_width(width/h),
     kv_width(kh*head_width),parameters(0),rows(l.queue),capacity(l.kv_rows),chunk(c) {
  std::vector<at::Tensor> weights,outputs;std::vector<int64_t> windows_cpu,settings;
  for(int64_t n=0;n<nodes;++n) {
    const auto& node=p.graph.nodes[n];const auto& region=p.graph.regions[node.region];
    const bool match=!node.identity&&node.memory=="attention"&&node.query_heads==h&&node.kv_heads==kh;
    node_map.push_back(match?parameters++:-1);adopt_all.push_back(region.observe_all);clear.push_back(node.clear);
    if(!match)continue;
    const auto& w=p.model.nodes[n];
    weights.push_back(at::cat({w.extra.at("attn_q"),w.extra.at("attn_k"),w.extra.at("attn_v")},1));
    outputs.push_back(w.extra.at("attn_out"));windows_cpu.push_back(node.window);
    settings.insert(settings.end(),{int64_t(region.observe_all),int64_t(node.clear)});
  }
  owners=q.batch_size*parameters;
  auto opts=at::TensorOptions().dtype(at::kFloat),longs=opts.dtype(at::kLong);
  weights.push_back(at::zeros({width,width+2*kv_width},opts));outputs.push_back(at::zeros({width,width},opts));
  qkv=at::stack(weights).to(device);projection=at::stack(outputs).to(device);
  mapping=at::tensor(node_map,longs).to(device);windows=at::tensor(windows_cpu,longs).to(device);
  config=at::tensor(settings,longs).reshape({parameters,2}).to(device);
  auto key=at::zeros({owners*capacity+1,kv_width},opts),value=at::zeros_like(key),lengths=at::zeros({owners},longs);
  for(const auto& [owner,s]:q.states) {
    const auto [b,n]=owner;if(node_map[n]<0)continue;const auto id=b*parameters+node_map[n],size=s.slots.at("key").size(0);
    if(size>capacity)throw std::invalid_argument("initial event attention cache exceeds kv_rows");
    key.narrow(0,id*capacity,size).copy_(s.slots.at("key").reshape({size,kv_width}));
    value.narrow(0,id*capacity,size).copy_(s.slots.at("value").reshape({size,kv_width}));lengths[id].fill_(size);
  }
  live={key.to(device),value.to(device),lengths.to(device)};
  chunks=at::zeros({1},longs.device(device));peak=lengths.max().reshape({1}).to(device);
  if(l.diagnostics)journal=std::make_unique<DeviceJournal>(l.kv_trace_rows,5,2*kv_width,device);
}
void PackedEventAttention::reset_window(){for(auto& g:groups_){g->chunks.zero_();if(g->journal)g->journal->count.zero_();}}
// Boundary-only statistics. These downloads never drive the next device stage.
at::Tensor PackedEventAttention::chunks() const {auto sum=at::zeros({1},at::kLong);for(const auto& g:groups_)sum+=g->chunks.cpu();return sum;}
at::Tensor PackedEventAttention::peak() const {auto peak=at::zeros({1},at::kLong);for(const auto& g:groups_)peak=at::maximum(peak,g->peak.cpu());return peak;}
void PackedEventAttention::export_states(Continuation& q) const {for(const auto& g:groups_)g->export_states(q);}
void PackedEventAttention::export_trace(std::vector<Event>& events) const {for(const auto& g:groups_)g->export_trace(events);}
void EventAttentionGroup::export_states(Continuation& q) const {
  auto k=live.key.cpu(),v=live.value.cpu(),lengths=live.lengths.cpu();
  for(auto& [owner,s]:q.states) {
    const auto [b,n]=owner;if(node_map[n]<0)continue;const auto id=b*parameters+node_map[n],size=lengths[id].item<int64_t>();
    s.slots={{"key",k.narrow(0,id*capacity,size).reshape({size,kv_heads,head_width}).clone()},
      {"value",v.narrow(0,id*capacity,size).reshape({size,kv_heads,head_width}).clone()}};
  }
}
void EventAttentionGroup::export_trace(std::vector<Event>& events) const {
  if(!journal)return;
  auto m=journal->meta.cpu(),v=journal->values.cpu();auto c=m.accessor<int64_t,2>();
  std::map<std::array<int64_t,4>,std::vector<int64_t>> rows;
  for(int64_t i=0;i<journal->count.cpu().item<int64_t>();++i)rows[{c[i][0],c[i][1],c[i][2],c[i][3]}].push_back(i);
  for(auto& e:events)if(node_map[e.node]>=0) {
    for(int kind=0;kind<2;++kind) {
      auto& ids=rows[{e.batch,e.node,e.time,kind}];
      std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return c[a][4]<c[b][4];});
      auto data=ids.empty()?at::empty({0,2*kv_width},at::kFloat):v.index_select(0,at::tensor(ids,at::kLong));
      auto& s=kind?e.proposed_state:e.old;const auto size=int64_t(ids.size());
      s.slots={{"key",data.narrow(1,0,kv_width).reshape({size,kv_heads,head_width}).clone()},
        {"value",data.narrow(1,kv_width,kv_width).reshape({size,kv_heads,head_width}).clone()}};
    }
    e.comparison_state.slots=(adopt_all[e.node]||e.active)?e.proposed_state.slots:e.old.slots;
    e.next_state.slots=e.comparison_state.slots;
    if(clear[e.node]&&e.active)for(auto& [_,x]:e.next_state.slots)x=x.slice(0,0,0).clone();
  }
}
} // namespace tide::device_online
