#include "return_packing.h"
#include <map>
#include <tuple>
namespace weighted_benchmark::return_copy {
namespace {
at::Tensor scalar(const at::Tensor& x){return x.detach().cpu().clone();}
}
void group(const std::vector<Row>& rows){
  if(rows.empty())return;
  if(rows.size()==1){for(auto* output:rows[0].outputs)*output=scalar(rows[0].value);return;}
  std::vector<at::Tensor> inputs;for(const auto& row:rows)inputs.push_back(row.value.detach().reshape({-1}));
  const auto flat=at::cat(inputs).cpu();int64_t offset=0;
  for(const auto& row:rows){
    const auto decoded=flat.narrow(0,offset,row.value.numel()).reshape(row.value.sizes());
    for(auto* output:row.outputs)*output=decoded.clone();
    offset+=row.value.numel();
  }
}
void tensors(const std::vector<at::Tensor*>& fields,bool packed){
  using Key=std::tuple<int,int,int>;std::map<Key,std::vector<Row>> groups;
  std::map<Key,std::map<const void*,size_t>> aliases;
  for(auto* output:fields){
    const auto value=*output;if(!value.defined())continue;
    const bool eligible=packed&&!value.device().is_cpu()&&value.layout()==at::kStrided
      &&!value.is_quantized()&&value.is_contiguous()&&!value.is_conj()&&!value.is_neg()
      &&value.numel()>0&&value.nbytes()<=kPackBytes;
    if(!eligible){*output=scalar(value);continue;}
    const Key key{int(value.device().type()),int(value.device().index()),int(value.scalar_type())};
    auto& rows=groups[key];auto& ids=aliases[key];const auto id=value.unsafeGetTensorImpl();
    auto found=ids.find(id);
    if(found==ids.end()){ids[id]=rows.size();rows.push_back({value,{output}});}
    else rows[found->second].outputs.push_back(output);
  }
  for(const auto& [key,rows]:groups){
    std::vector<Row> chunk;int64_t bytes=0;
    for(const auto& row:rows){
      const auto cost=row.value.nbytes();
      if(!chunk.empty()&&bytes+cost>kPackBytes){group(chunk);chunk.clear();bytes=0;}
      chunk.push_back(row);bytes+=cost;
    }
    group(chunk);
  }
}
}
