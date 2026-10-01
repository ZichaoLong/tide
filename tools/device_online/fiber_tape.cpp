#include "packed_fiber_attention.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
int64_t PackedFiberAttention::tape_bytes() const {
  if(!journal_)throw std::logic_error("fiber tape requires actual journals");
  const int64_t samples=owners_/parameters_,bytes=cache_.key.element_size();
  long double size=journal_->meta.nbytes()+journal_->values.nbytes()+journal_->count.nbytes();
  for(const auto h:head_groups_) {
    int64_t ps=0,slots=1;for(int64_t n=0;n<nodes_;++n)if(node_map_[n]>=0&&node_heads_[n]==h){++ps;slots=std::max(slots,source_lengths_[n]);}
    if(pool_)slots=pool_->weights().size(1);
    // Include metadata and transient gather indices. Sharing the journal among
    // groups is counted only once. This bound also covers sentinel bank rows.
    size+=bytes*(static_cast<long double>(samples)*ps*capacity_*(2.L*width_+1)+
      (ps+1.L)*(4.L*width_*width_+4.L*width_)+ps)+4.L*(ps+1)*slots+64.L*(nodes_+ps+samples*ps+1);
  }
  if(size>std::numeric_limits<int64_t>::max())throw std::invalid_argument("fiber tape extent overflow");
  return int64_t(size);
}
std::vector<FiberAttentionTape> PackedFiberAttention::tape() const {
  if(!journal_)throw std::logic_error("fiber reverse tape requires actual KV journals");
  const auto device=cache_.key.device();const int64_t samples=owners_/parameters_;
  std::vector<FiberAttentionTape> out;
  for(const auto h:head_groups_) {
    FiberAttentionTape g;auto& a=g.cache;std::vector<int64_t> ids,mapping(nodes_,-1),lengths;
    for(int64_t n=0;n<nodes_;++n)if(node_map_[n]>=0&&node_heads_[n]==h) {
      mapping[n]=ids.size();ids.push_back(node_map_[n]);a.nodes.push_back(n);lengths.push_back(source_lengths_[n]);
    }
    const int64_t ps=ids.size();auto index=at::tensor(ids,at::kLong).to(device);
    auto owners=(at::arange(samples,index.options()).unsqueeze(1)*parameters_+index.unsqueeze(0)).reshape({-1});
    a.samples=samples;a.heads=a.kv_heads=h;a.width=width_;a.capacity=capacity_;
    a.mapping=at::tensor(mapping,at::kLong).to(device);a.windows=at::zeros({ps},index.options());
    a.config=config_.index_select(0,index);a.metadata=journal_->meta;a.values=journal_->values;a.count=journal_->count;
    auto key=cache_.key.narrow(0,0,owners_*capacity_).reshape({owners_,capacity_,width_});
    auto value=cache_.value.narrow(0,0,owners_*capacity_).reshape({owners_,capacity_,width_});
    a.key=key.index_select(0,owners).reshape({samples*ps,capacity_,h,width_/h});
    a.value=value.index_select(0,owners).reshape(a.key.sizes());a.lengths=cache_.lengths.index_select(0,owners);
    g.bias=cache_.bias.narrow(0,0,owners_*capacity_).reshape({owners_,capacity_}).index_select(0,owners);
    g.decay=decay_.index_select(0,index);
    g.pool_kinds=pool_?pool_->kinds().index_select(0,index):at::zeros({ps},index.options());
    g.pool_lengths=at::tensor(lengths,at::kLong).to(device);
    ids.push_back(parameters_);index=at::tensor(ids,at::kLong).to(device);
    a.qkv=qkv_.index_select(0,index);g.qkv_bias=qkv_bias_.index_select(0,index);
    a.projection=projection_.index_select(0,index);g.projection_bias=projection_bias_.index_select(0,index);
    const int64_t slots=std::max<int64_t>(1,*std::max_element(lengths.begin(),lengths.end()));
    g.pool_weights=pool_?pool_->weights().index_select(0,index):at::zeros({ps+1,slots},cache_.key.options().dtype(at::kFloat));
    out.push_back(std::move(g));
  }
  return out;
}
FiberParameterBanks PackedFiberAttention::banks() const {
  FiberParameterBanks out;
  for(int64_t n=0;n<nodes_;++n)if(node_map_[n]>=0)out.nodes.push_back(n);
  out.qkv=qkv_;out.qkv_bias=qkv_bias_;out.projection=projection_;out.projection_bias=projection_bias_;
  out.decay=decay_;if(pool_)out.pool=pool_->weights();return out;
}
std::vector<int64_t> fiber_parameter_offsets(const Graph& g,int64_t width) {
  return fiber_parameter_offsets(g.nodes,g.source_counts,width);
}
std::vector<int64_t> fiber_parameter_offsets(const std::vector<Node>& nodes,const std::vector<int64_t>& counts,int64_t width) {
  if(counts.size()!=nodes.size())throw std::invalid_argument("fiber source-domain extent differs from node projection");
  std::vector<int64_t> out;int64_t offset=0;
  for(size_t n=0;n<nodes.size();++n) {
    out.push_back(offset);const auto& node=nodes[n];
    if(node.identity||!is_fiber_attention_profile(node.memory))continue;
    if(width<1||node.query_heads<1||width%node.query_heads||node.kv_heads!=node.query_heads)
      throw std::invalid_argument("invalid fiber parameter geometry");
    const bool learned=node.memory!="lh-fiber-attention-sum-repeat-v1"&&node.memory!="lh-fiber-attention-mean-repeat-v1";
    if(counts[n]<0)throw std::invalid_argument("negative fiber source-domain extent");
    const long double next=offset+4.L*width*width+4.L*width+1+(learned?counts[n]:0);
    if(next>std::numeric_limits<int64_t>::max())throw std::invalid_argument("fiber parameter extent overflow");
    offset=static_cast<int64_t>(next);
  }
  out.push_back(offset);return out;
}
} // namespace tide::device_online
