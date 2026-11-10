#include "weighted_kv_preparation.h"
#include "weighted_preparation_rows.h"
#include "weighted_detail.h"
#include <algorithm>
#include <sstream>

namespace tide::weighted {
namespace {
using Groups=std::map<std::string,std::vector<size_t>>;
std::string handle(const Tensor& t){auto p=reinterpret_cast<std::uintptr_t>(t.unsafeGetTensorImpl());
  return std::string(reinterpret_cast<const char*>(&p),sizeof(p));}
std::string private_key(const NodeWeights& w){std::string key;for(auto name:{"kernel.adapter_v","kernel.adapter_u","kernel.summary"})
  key+=handle(detail::weight(w,name));return key;}
std::vector<std::vector<size_t>> ordered(const Groups& groups){std::vector<std::vector<size_t>> rows;
  for(const auto& [key,ids]:groups)rows.push_back(ids);
  std::sort(rows.begin(),rows.end(),[](const auto& a,const auto& b){return a.front()<b.front();});return rows;}
}
std::vector<std::map<std::string,Tensor>> prepare_block_kv(const std::vector<const NodeWeights*>& rows,
    const std::vector<Tensor>& contents,const std::vector<double>& eps,std::map<std::string,Index>& stats){
  detail::require(rows.size()==contents.size()&&rows.size()==eps.size(),"invalid block KV preparation rows");
  std::map<std::string,size_t> repeats;size_t repeated=0;
  for(auto* row:rows)repeated=std::max(repeated,++repeats[private_key(*row)]);
  if(repeated<2)return {}; // Keep the existing independent-node/single-event path.
  Groups metadata,common,privates;std::vector<bool> positive(rows.size());
  for(size_t i=0;i<rows.size();++i){std::ostringstream key;key<<contents[i].device()<<int(contents[i].scalar_type());metadata[key.str()].push_back(i);}
  for(const auto& ids:ordered(metadata)){
    std::vector<Tensor> mass;for(auto i:ids)mass.push_back(contents[i][0]);
    auto flags=(at::stack(mass).detach()>0).to(at::kCPU);const auto* data=flags.const_data_ptr<bool>();
    for(size_t j=0;j<ids.size();++j)positive[ids[j]]=data[j];
  }
  for(size_t i=0;i<rows.size();++i)if(positive[i]){
    auto key=handle(detail::weight(*rows[i],"kernel.base"));key.append(reinterpret_cast<const char*>(&eps[i]),sizeof(double));
    common[key].push_back(i);privates[private_key(*rows[i])].push_back(i);
  }
  std::vector<std::pair<Tensor,Tensor>> operands(rows.size());
  std::vector<std::map<std::string,Tensor>> result(rows.size());
  constexpr size_t budget=8*1024*1024;
  for(const auto& ids:ordered(common)){
    Groups identity;for(auto i:ids)identity[handle(contents[i])].push_back(i);auto chunks=ordered(identity);
    auto base=detail::weight(*rows[ids[0]],"kernel.base");size_t d=base.size(1);
    size_t limit=std::max<size_t>(1,budget/(4*(3*d+1)*base.element_size()));
    for(size_t start=0;start<chunks.size();start+=limit){size_t end=std::min(chunks.size(),start+limit);
      std::vector<std::pair<Tensor,Tensor>> input;for(size_t j=start;j<end;++j)input.push_back({contents[chunks[j][0]],{}});
      auto values=prepare_kv_rows(true,{base},input,eps[ids[0]]);
      ++stats["state_block_base_projection_calls"];
      stats["max_state_block_base_batch"]=std::max(stats["max_state_block_base_batch"],Index(end-start));
      for(size_t j=start;j<end;++j){for(auto i:chunks[j])operands[i]=values[j-start];
        if(chunks[j].size()>1){++stats["state_preparation_calls"];stats["state_preparation_events"]+=chunks[j].size();stats["state_preparation_reused_events"]+=chunks[j].size()-1;}}
    }
  }
  for(const auto& ids:ordered(privates)){
    const auto& w=*rows[ids[0]];auto v=detail::weight(w,"kernel.adapter_v"),u=detail::weight(w,"kernel.adapter_u"),s=detail::weight(w,"kernel.summary");
    size_t limit=std::max<size_t>(1,budget/(4*(5*v.size(1)+s.size(0)+v.size(0))*v.element_size()));
    for(size_t start=0;start<ids.size();start+=limit){size_t end=std::min(ids.size(),start+limit);
      std::vector<std::pair<Tensor,Tensor>> input;for(size_t j=start;j<end;++j)input.push_back(operands[ids[j]]);
      auto values=prepare_kv_rows(false,{v,u,s},input);
      for(size_t j=start;j<end;++j)result[ids[j]]={{"values",values[j-start].first},{"summaries",values[j-start].second}};
      stats["state_private_projection_calls"]+=3;stats["state_private_projection_rows"]+=3*(end-start);
      for(auto name:{"max_state_private_projection_batch","max_state_independent_projection_batch"})
        stats[name]=std::max(stats[name],Index(end-start));
    }
  }
  stats["state_block_preparation_events"]+=rows.size();return result;
}
}
