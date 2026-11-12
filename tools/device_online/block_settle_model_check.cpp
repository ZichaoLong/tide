#include "block_settle_model.h"
#include "block_parameter_views.h"
#include "block_resident_fixture.h"
#include <ATen/core/grad_mode.h>
#include <iostream>
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void check(bool query){
  at::NoGradGuard guard;auto f=block_public_check::fixture(query);
  auto lower=lower_block_settle_model(f.graph,f.model,f.initial,8,1LL<<28);
  require(lower.graph.graph().edges.size()==5&&lower.graph.graph().outputs.size()==3,"Model lowering lost physical identity");
  require(lower.registry.alias_partitions()==f.model.parameters(false).alias_partitions(),"Model lowering changed canonical aliases");
  require(lower.bindings[1].edge_owners[0]==lower.bindings[1].edge_owners[1],"same SrcClass must share parameters without deleting atoms");
  for(size_t r=0;r<lower.bindings.size();++r){const auto& b=lower.bindings[r];
    require(b.memory.memory_read==(query?"qry":"sum"),"Model lowering changed Memory function");
    for(const auto& role:{"read.query","read.key","read.value"})
      require(bool(lower.roles[r].count(role))==query,"Model lowering fabricated or omitted query parameter roles");
    require(b.parameters.message.read.query.query.defined()==query,"sum Model has a synthetic query bank");
    require(lower.roles[r].count("full.query"),"sum Read must not remove Full Attention parameters");
    require(b.parameters.source_factors.has_value(),"Model lowering dropped original source factors");
    for(const auto& [role,bank]:block_kv_local_parameters(b.parameters)){
      const auto& names=lower.roles[r].at(role);require(bank.size(0)==int64_t(names.size()),"Model bank owner count differs");
      for(size_t j=0;j<names.size();++j){const auto original=lower.registry.value(names[j]);
        require(at::equal(block_parameter_view(role,bank[j],original.sizes()),original),"Model bank changed original owner values");
        if(role.rfind("source.",0)==0&&original.numel()<bank[j].numel()){
          auto residue=bank[j].clone();block_parameter_view(role,residue,original.sizes()).zero_();
          require(residue.eq(0).all().item<bool>(),"padded source bank has nonzero synthetic parameters");}}
    }
    const WeightedRawLayout layout(b.memory,8);const auto actual=decode_weighted_raw_state(layout,lower.initial[r]);
    if(r==0){require(actual[0].last_time==1&&actual[0].observations==1,"Model lowering lost nonempty actual raw state");
      require(actual[1].last_time==-1&&actual[2].last_time==-1,"Model lowering fabricated missing states");}
  }
  require(lower.bindings[1].parameters.source.a.size(1)==4&&lower.bindings[1].parameters.source_factors->a_left.size(2)==2,
    "Model source padding does not cover distinct hidden/rank geometry");
  auto wrong=f.model;wrong.input_scale[0]=at::full({},.5,at::kFloat);bool refused=false;
  try{lower_block_settle_model(f.graph,wrong,f.initial,8,1LL<<28);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"nonunit physical transport was silently replaced");
}
}
int main(){try{for(bool query:{true,false})check(query);std::cout<<"block-settle-model: passed memory_read=qry,sum original_parameters=true source_factors=true heterogeneous_hidden_rank=true raw_import=true\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
