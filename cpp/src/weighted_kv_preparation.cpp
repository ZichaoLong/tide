#include "weighted_kv_preparation.h"
#include "weighted_row_linear.h"
#include "weighted_detail.h"
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
}
