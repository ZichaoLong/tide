#include "weighted_kv_preparation.h"
#include "weighted_row_linear.h"
#include "weighted_detail.h"
#include <algorithm>
#include <sstream>
namespace tide::weighted {
std::vector<std::map<std::string,Tensor>> prepare_private_kv(
    const std::vector<const NodeWeights*>& rows,const std::map<std::string,Tensor>& shared,
    std::map<std::string,Index>& stats){
  std::map<std::string,std::vector<size_t>> groups;
  for(size_t i=0;i<rows.size();++i){
    std::ostringstream key;
    for(const auto& name:{"kernel.adapter_v","kernel.adapter_u","kernel.summary"}){
      const auto& p=detail::weight(*rows[i],name);key<<p.sizes()<<p.device()<<int(p.scalar_type());
    }
    groups[key.str()].push_back(i);
  }
  std::vector<std::map<std::string,Tensor>> result(rows.size(),shared);
  for(const auto& [key,ids]:groups){
    if(ids.size()<2)continue;
    std::vector<Tensor> vs,us,ss,normal(ids.size(),shared.at("normal"));
    for(auto i:ids){vs.push_back(detail::weight(*rows[i],"kernel.adapter_v"));
      us.push_back(detail::weight(*rows[i],"kernel.adapter_u"));ss.push_back(detail::weight(*rows[i],"kernel.summary"));}
    auto low=linear_rows(normal,vs),values=linear_rows(low,us),summaries=linear_rows(normal,ss);
    for(size_t j=0;j<ids.size();++j)result[ids[j]]={{"values",shared.at("base")+values[j]},{"summaries",summaries[j]}};
    stats["state_private_projection_calls"]+=3;stats["state_private_projection_rows"]+=3*ids.size();
    stats["max_state_private_projection_batch"]=std::max(stats["max_state_private_projection_batch"],Index(ids.size()));
  }
  return result;
}

std::vector<std::map<std::string,Tensor>> prepare_independent_kv(
    const std::vector<const NodeWeights*>& rows,const std::vector<Tensor>& contents,
    const std::vector<double>& eps,std::map<std::string,Index>& stats,Index budget){
  detail::require(rows.size()==contents.size()&&rows.size()==eps.size()&&budget>0,
                  "invalid independent KV preparation rows/budget");
  std::vector<std::map<std::string,Tensor>> result(rows.size());
  std::map<std::string,std::vector<size_t>> groups;
  for(size_t i=0;i<rows.size();++i){
    if(contents[i][0].item<double>()<=0)continue;
    std::ostringstream key;
    for(const auto* name:{"kernel.base","kernel.adapter_v","kernel.adapter_u","kernel.summary"}){
      const auto& p=detail::weight(*rows[i],name);key<<p.sizes()<<p.device()<<int(p.scalar_type());
    }
    groups[key.str()].push_back(i);
  }
  std::vector<std::vector<size_t>> ordered;
  for(auto& [key,ids]:groups)ordered.push_back(std::move(ids));
  std::sort(ordered.begin(),ordered.end(),[](const auto& a,const auto& b){return a.front()<b.front();});
  for(const auto& ids:ordered){
    Index row_bytes=0;
    for(const auto* name:{"kernel.base","kernel.adapter_v","kernel.adapter_u","kernel.summary"}){
      const auto& p=detail::weight(*rows[ids[0]],name);row_bytes+=4*p.numel()*p.element_size();
    }
    const auto limit=budget/std::max<Index>(1,row_bytes);
    if(limit<2)continue;
    for(size_t start=0;start<ids.size();start+=limit){
      const auto count=std::min<size_t>(limit,ids.size()-start);
      if(count<2)continue;
      std::vector<Tensor> normal,bs,vs,us,ss;
      for(size_t j=0;j<count;++j){
        auto i=ids[start+j];normal.push_back(normalize(contents[i].slice(0,1),eps[i]));
        bs.push_back(detail::weight(*rows[i],"kernel.base"));vs.push_back(detail::weight(*rows[i],"kernel.adapter_v"));
        us.push_back(detail::weight(*rows[i],"kernel.adapter_u"));ss.push_back(detail::weight(*rows[i],"kernel.summary"));
      }
      auto base=linear_rows(normal,bs),low=linear_rows(normal,vs);
      auto values=linear_rows(low,us),summaries=linear_rows(normal,ss);
      for(size_t j=0;j<count;++j)result[ids[start+j]]={{"values",base[j]+values[j]},{"summaries",summaries[j]}};
      stats["state_independent_projection_calls"]+=4;stats["state_independent_projection_rows"]+=4*count;
      stats["max_state_independent_projection_batch"]=std::max(stats["max_state_independent_projection_batch"],Index(count));
    }
  }
  return result;
}
}
