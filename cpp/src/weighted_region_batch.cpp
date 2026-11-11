#include "weighted_private_rows.h"
#include "tide/operator_profile.h"
#include <ATen/record_function.h>
// Allocation §2–6: complete query/context/normalization/allocation batches.
#include "weighted_region_batch.h"
#include "tide/weighted_columns.h"
#include "weighted_detail.h"
#include "tide/allocation_failure.h"
#include "region_metadata.h"
#include <sstream>

namespace tide::weighted::batch {
std::vector<Selection> regions(const RegionWeights& w,const std::vector<RegionInput>& requests,const RegionalConfig& c,
                               std::vector<std::string>* dependencies,std::optional<at::Device> selection){
  std::vector<Selection> result(requests.size());std::map<std::string,std::vector<size_t>> groups;
  if(dependencies)dependencies->resize(requests.size());
  for(size_t i=0;i<requests.size();++i){
    detail::require(!requests[i].candidates.empty(),"region batch requires nonempty complete candidates");
    const auto& first=requests[i].candidates[0].fields.at("w");std::ostringstream key;
    key<<requests[i].candidates.size()<<first.device()<<int(first.scalar_type());groups[key.str()].push_back(i);
  }
  for(const auto& [key,ids]:groups){
    std::map<std::string,Tensor> fields;
    for(const auto& [name,unused]:requests[ids[0]].candidates[0].fields){
      std::vector<Tensor> rows;
      for(size_t j=0;j<requests[ids[0]].candidates.size();++j){
        std::vector<Tensor> column;for(auto i:ids)column.push_back(requests[i].candidates[j].fields.at(name));
        rows.push_back(private_rows::join(column,true));
      }
      fields[name]=at::stack(rows,1);
    }
    auto recv=fields.at("w"),caps=fields.at("mu"),scores=fields.at("score");Tensor nodes;
    if(c.selection_mode=="regional_total_zero"||c.hard()){
      // One actual int64 upload for the complete sorted-sending group. Other
      // sending functions never consume node IDs and need no device copy.
      std::vector<Index> values;values.reserve(ids.size()*requests[ids[0]].candidates.size());
      for(auto i:ids)for(const auto& d:requests[i].candidates)values.push_back(d.node);
      nodes=at::tensor(values,recv.options().dtype(at::kLong)).reshape(scores.sizes());
    }
    RegionColumns column;
    try{column=region_columns(w,fields,c,nodes,selection);}
    catch(AllocationFailure& e){
      e.row=ids.at(e.row);const auto& request=requests[e.row];e.time=request.time;e.members=request.layout.members.vec();
      for(const auto& candidate:request.candidates)e.nodes.push_back(candidate.node);
      throw;
    }
    auto rows=column.controls.unbind();
    if(dependencies){auto names=region_column_dependencies(column);
      for(size_t j=0;j<ids.size();++j)(*dependencies)[ids[j]]=names[j];}
    const bool grouped=ids.size()>1&&!rows[0].device().is_cpu();
    std::vector<bool> support;size_t offset=0;
    if(grouped){
      std::vector<Tensor> controls;for(const auto& row:rows)for(const auto& value:row.unbind())controls.push_back(value);
      support=region_metadata::flags(controls,true);
    }
    for(size_t j=0;j<ids.size();++j){
      auto i=ids[j];auto& out=result[i];out.history.last_time=requests[i].time;
      for(size_t k=0;k<requests[i].candidates.size();++k){
        auto node=requests[i].candidates[k].node;out.controls[node]=rows[j][k];
        if(grouped?support[offset++]:rows[j][k].item<double>()>0)out.active.insert(node);
      }
    }
  }
  return result;
}
}
