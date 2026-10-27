#include "weighted_packed.h"
#include <limits>

namespace tide::weighted::packed {
using detail::require;
Writes::Writes(const std::vector<State>& old,const PackedSequence& batch){
  batch.validate();require(old.size()==batch.owners.size(),"packed initial-state count mismatch");
  auto mass=batch.contents.select(1,0);require((mass>=0).all().item<bool>(),"negative aggregate received mass");
  auto mask=(mass>0).to(at::kCPU).contiguous();auto p=mask.data_ptr<bool>();
  std::vector<Index> stamps;
  for(Index j=0;j<mass.numel();++j){positive.push_back(p[j]);if(p[j]){ids.push_back(j);stamps.push_back(batch.times[j]);}}
  auto ints=batch.contents.options().dtype(at::kLong);
  index=at::tensor(ids,ints);times=at::tensor(stamps,ints);
  receive=mass.index_select(0,index).clamp_max(1);input=batch.contents.slice(1,1).index_select(0,index);
  for(size_t i=0;i<old.size();++i){
    Index n=0;
    for(Index j=batch.offsets[i];j<batch.offsets[i+1];++j)if(positive[j]){
      require(batch.times[j]>old[i].last_time,"packed write times must strictly increase");++n;
    }
    require(n<=std::numeric_limits<Index>::max()-old[i].observations,"packed observation overflow");
    if(n)groups[n].push_back(i);offsets.push_back(offsets.back()+n);
  }
}
Tensor Writes::group_index(Index length,const std::vector<size_t>& owners) const{
  std::vector<Index> rows;
  for(Index j=0;j<length;++j)for(auto i:owners)rows.push_back(offsets[i]+j);
  return at::tensor(rows,times.options()).reshape({length,static_cast<Index>(owners.size())});
}
Tensor Writes::gaps(const std::vector<State>& old) const{
  std::vector<Tensor> deltas;
  for(size_t i=0;i<old.size();++i){auto a=offsets[i],b=offsets[i+1];if(a==b)continue;
    auto now=times.slice(0,a,b),prior=at::cat({old[i].slots.at("time").reshape({1}),now.slice(0,0,b-a-1)});
    deltas.push_back((now-prior).clamp_min(0));
  }
  return at::cat(deltas);  // Exact integer subtraction precedes conversion.
}
PackedStates Writes::decode(const std::vector<State>& old,const PackedSequence& batch,
    const std::vector<std::map<std::string,Tensor>>& slots,Index calls) const{
  PackedStates result;result.calls=calls;result.max_batch=old.size();
  for(size_t i=0;i<old.size();++i){State state=old[i];Index count=0;
    result.max_length=std::max(result.max_length,batch.offsets[i+1]-batch.offsets[i]);
    for(Index j=batch.offsets[i];j<batch.offsets[i+1];++j){
      if(positive[j]){state={old[i].value,batch.times[j],old[i].observations+count+1,slots.at(offsets[i]+count)};++count;}
      result.states.push_back(state);
    }
  }
  return result;
}
PackedSequence independent(const std::vector<State>& old,const Tensor& h,const std::vector<Index>& times,const ContentViews& views){
  PackedSequence batch;batch.contents=h;batch.times=times;batch.views=views;
  for(size_t i=0;i<old.size();++i){batch.offsets.push_back(i+1);batch.owners.emplace_back(i,0);}
  return batch;
}
Tensor matrix_scan(Tensor a,Tensor b,const Tensor& initial){
  auto n=b.size(0);
  for(Index stride=1;stride<n;stride*=2){
    b=at::cat({b.slice(0,0,stride),at::matmul(b.slice(0,0,n-stride),a.slice(0,stride))+b.slice(0,stride)});
    a=at::cat({a.slice(0,0,stride),at::matmul(a.slice(0,0,n-stride),a.slice(0,stride))});
  }
  return at::matmul(initial,a)+b;
}
}
