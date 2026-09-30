#include "packed_fiber_attention.h"
#include "tiled_attention.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace tide::device_online {
PackedFiberAttention::PackedFiberAttention(const ContentProfile& profile,const Continuation& q,
    at::Device device,const ContentLimits& limits,int64_t budget)
    :nodes_(profile.graph.nodes.size()),width_(profile.width),parameters_(0),rows_(limits.queue),
     capacity_(limits.kv_rows),max_ticks_(limits.max_repeat_ticks) {
  std::vector<at::Tensor> qkv,bias,out,ob,decay;
  std::vector<int64_t> heads,config;
  for(int64_t n=0;n<nodes_;++n) {
    const auto& node=profile.graph.nodes[n];
    const bool enabled=!node.identity&&is_fiber_attention_profile(node.memory);
    node_map_.push_back(enabled?parameters_++:-1);node_heads_.push_back(node.query_heads);
    const auto& region=profile.graph.regions[node.region];
    adopt_all_.push_back(region.observe_all);clear_.push_back(node.clear);
    if(!enabled)continue;
    const auto& w=profile.model.nodes[n];
    qkv.push_back(w.extra.at("fiber_qkv"));bias.push_back(w.extra.at("fiber_qkv_bias"));
    out.push_back(w.extra.at("fiber_out"));ob.push_back(w.extra.at("fiber_out_bias"));
    decay.push_back(w.extra.at("fiber_decay"));heads.push_back(node.query_heads);
    config.insert(config.end(),{int64_t(region.observe_all),int64_t(node.clear)});
  }
  if(!parameters_||capacity_<1||limits.attention_chunk_rows<1||limits.attention_key_rows<1||budget<1
      ||(limits.diagnostics&&limits.kv_trace_rows<1))throw std::invalid_argument("invalid fiber cache limits");
  owners_=q.batch_size*parameters_;
  head_groups_=heads;std::sort(head_groups_.begin(),head_groups_.end());
  head_groups_.erase(std::unique(head_groups_.begin(),head_groups_.end()),head_groups_.end());
  const auto max_heads=*std::max_element(heads.begin(),heads.end());
  // Include both cache arenas, independent padding, row work, diagnostic copies,
  // all head-group pack/score buffers and immutable parameter tables.
  const bool pooled=std::any_of(profile.graph.nodes.begin(),profile.graph.nodes.end(),[](const Node& n){
    return !n.identity&&is_fiber_attention_profile(n.memory)&&n.memory!="lh-fiber-attention-sum-repeat-v1";});
  const auto pool_fixed=pooled?PackedFiberPool::reserved_bytes(profile,rows_,0):0.L;
  const auto pool_row=pooled?PackedFiberPool::reserved_bytes(profile,rows_,1)-pool_fixed:0.L;
  const long double fixed=pool_fixed+24.L*(owners_*static_cast<long double>(capacity_)+1)*(2.L*width_+1)
    +48.L*(parameters_+1.L)*width_*width_+128.L*(rows_+1.L)*(width_+8.L)
    +(limits.diagnostics?24.L*limits.kv_trace_rows*(2.L*width_+8):0);
  const long double row_base=pool_row+96.L*width_*width_+256.L*width_+256+head_groups_.size()*256.L*width_;
  const long double row_key=head_groups_.size()*(32.L*width_+48.L*max_heads);
  const auto tiles=plan_attention_tiles(fixed,row_base,row_key,budget,rows_,limits.attention_chunk_rows,capacity_,limits.attention_key_rows);
  chunk_=tiles.queries;key_rows_=tiles.keys;reserved_=tiles.reserved;
  if(pooled)pool_=std::make_unique<PackedFiberPool>(profile,device,rows_,chunk_);
  auto opts=at::TensorOptions().dtype(at::kFloat);auto longs=opts.dtype(at::kLong);
  qkv.push_back(at::zeros({width_,3*width_},opts));bias.push_back(at::zeros({3*width_},opts));
  out.push_back(at::zeros({width_,width_},opts));ob.push_back(at::zeros({width_},opts));
  qkv_=at::stack(qkv).to(device);qkv_bias_=at::stack(bias).to(device);
  projection_=at::stack(out).to(device);projection_bias_=at::stack(ob).to(device);
  decay_=at::stack(decay).to(device);mapping_=at::tensor(node_map_,longs).to(device);
  heads_=at::tensor(heads,longs).to(device);config_=at::tensor(config,longs).reshape({parameters_,2}).to(device);
  const auto total=owners_*capacity_+1; // Independent zero sentinel, never a real cache row.
  auto key=at::zeros({total,width_},opts),value=at::zeros_like(key),log_bias=at::zeros({total},opts);
  auto lengths=at::zeros({owners_},longs);
  for(const auto& [owner,state]:q.states) {
    const auto [b,n]=owner;if(node_map_[n]<0)continue;const auto id=b*parameters_+node_map_[n];
    const auto size=state.slots.at("key").size(0);
    if(size>capacity_)throw std::invalid_argument("initial fiber cache exceeds kv_rows");
    key.narrow(0,id*capacity_,size).copy_(state.slots.at("key").reshape({size,width_}));
    value.narrow(0,id*capacity_,size).copy_(state.slots.at("value").reshape({size,width_}));
    log_bias.narrow(0,id*capacity_,size).copy_(state.slots.at("log_bias"));lengths[id].fill_(size);
  }
  cache_={key.to(device),value.to(device),log_bias.to(device),lengths.to(device)};
  chunks_=at::zeros({1},longs.device(device));peak_=lengths.max().reshape({1}).to(device);
  key_work_=at::zeros({3},longs.device(device));
  if(limits.diagnostics)journal_=std::make_unique<DeviceJournal>(limits.kv_trace_rows,5,2*width_+1,device);
}
void PackedFiberAttention::reset_window(){chunks_.zero_();key_work_.zero_();if(journal_)journal_->count.zero_();}
void PackedFiberAttention::export_states(Continuation& q) const {
  auto k=cache_.key.cpu(),v=cache_.value.cpu(),b=cache_.bias.cpu(),lengths=cache_.lengths.cpu();
  for(auto& [owner,s]:q.states) {
    const auto [sample,node]=owner;if(node_map_[node]<0)continue;
    const auto id=sample*parameters_+node_map_[node],length=lengths[id].item<int64_t>(),heads=node_heads_[node];
    s.slots={{"key",k.narrow(0,id*capacity_,length).reshape({length,heads,width_/heads}).clone()},
      {"value",v.narrow(0,id*capacity_,length).reshape({length,heads,width_/heads}).clone()},
      {"log_bias",b.narrow(0,id*capacity_,length).clone()}};
  }
}
void PackedFiberAttention::export_trace(std::vector<Event>& events) const {
  if(!journal_)return;
  auto m=journal_->meta.cpu(),v=journal_->values.cpu();auto c=m.accessor<int64_t,2>();
  std::map<std::array<int64_t,4>,std::vector<int64_t>> rows;
  for(int64_t i=0;i<journal_->count.cpu().item<int64_t>();++i)rows[{c[i][0],c[i][1],c[i][2],c[i][3]}].push_back(i);
  for(auto& e:events)if(node_map_[e.node]>=0) {
    const auto h=node_heads_[e.node];
    for(int kind=0;kind<2;++kind) {
      auto& ids=rows[{e.batch,e.node,e.time,kind}];
      std::sort(ids.begin(),ids.end(),[&](auto a,auto b){return c[a][4]<c[b][4];});
      auto data=ids.empty()?at::empty({0,2*width_+1},at::kFloat):v.index_select(0,at::tensor(ids,at::kLong));
      auto& state=kind?e.proposed_state:e.old;const auto count=int64_t(ids.size());
      state.slots={{"key",data.narrow(1,0,width_).reshape({count,h,width_/h}).clone()},
        {"value",data.narrow(1,width_,width_).reshape({count,h,width_/h}).clone()},
        {"log_bias",data.select(1,2*width_).clone()}};
    }
    e.comparison_state.slots=(adopt_all_[e.node]||e.active)?e.proposed_state.slots:e.old.slots;
    e.next_state.slots=e.comparison_state.slots;
    if(clear_[e.node]&&e.active)for(auto& [_,x]:e.next_state.slots)x=x.slice(0,0,0).clone();
  }
}
} // namespace tide::device_online
