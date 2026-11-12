#include "block_resident_internal.h"
#include "block_kv_bank.h"
#include "weighted_storage_budget.h"
#include <set>
namespace tide::device_online {
std::map<std::string,BlockField> BlockResidentSession::observe(ResidentToken token,const std::vector<std::string>& names){
  auto& s=*impl_;s.check();const Impl::Window* window=nullptr;
  for(const auto& w:s.windows)if(w.view.token.session==token.session&&w.view.token.index==token.index&&w.view.token.generation==token.generation)window=&w;
  if(!window)throw std::invalid_argument("block observation token is not a retained owner window");
  std::set<std::string> requested(names.begin(),names.end());std::map<std::string,BlockField> out;
  for(const auto& name:requested){const auto found=window->view.fields.find(name);if(found==window->view.fields.end())throw std::invalid_argument("unknown block observation: "+name);
    out.emplace(name,found->second);}
  bool deferred=false;for(const auto& [name,f]:out)deferred=deferred||!f.values.defined();if(!deferred)return out;
  const auto& tape=*window->actual.tape;const auto device=tape.positions.device();
  DeviceProgram program(device);program.limit_workspace(s.limits.workspace_bytes);auto error=at::zeros({1},tape.positions.options().dtype(at::kInt));
  std::vector<at::Tensor> borrowed;for(const auto& r:tape.regions){auto tensors=block_kv_bank_tensors(*r.local->sequence.bank);borrowed.insert(borrowed.end(),tensors.begin(),tensors.end());}
  const WeightedStorageIncrement storage(program,std::move(borrowed));
  auto remaining=[&](){const auto used=storage.bytes();if(used>=s.limits.forward_bytes)throw std::invalid_argument("raw observation tensor budget exceeded");return s.limits.forward_bytes-used;};
  for(size_t r=0;r<tape.regions.size();++r){const auto& sequence=tape.regions[r].local->sequence;const auto prefix="region."+std::to_string(r)+".";
    for(bool proposal:{false,true}){const auto name=prefix+(proposal?"proposal.":"old.");bool needed=false;
      for(const auto& f:sequence.layout.fields())needed=needed||requested.count(name+f.name);
      if(!needed)continue;
      const auto raw=append_block_kv_bank_decode(program,block_kv_snapshot(sequence,proposal),error,remaining());
      for(const auto& f:sequence.layout.fields()){const auto key=name+f.name;if(!requested.count(key))continue;auto& field=out.at(key);
        const auto width=f.width*(f.ragged?sequence.layout.capacity():1);field.values=raw.floating.narrow(1,f.offset,width).reshape(field.shape);}
    }
  }
  auto admit=[&](){const auto existing=s.owner->storage_usage();if(storage.bytes()>s.limits.forward_bytes||
      static_cast<long double>(existing.tensor_bytes)+existing.workspace_bytes+storage.bytes()+program.workspace_bytes()>s.limits.total_bytes)
      throw std::invalid_argument("raw observation exceeds the resident owner budget");};
  admit();program.finish();admit();
  try{program.run(s.limits.timeout_ms);
    const auto code=error.cpu().item<int>();if(code)throw std::runtime_error("raw observation refused, code="+std::to_string(code));
    program.close();
  }catch(...){s.failed=true;throw;}
  for(const auto& [name,f]:out)if(!f.values.defined())throw std::logic_error("unmaterialized block observation: "+name);
  return out;
}
}
