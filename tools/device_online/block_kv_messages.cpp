#include "block_kv_messages.h"
#include "block_read_selection_internal.h"

namespace tide::device_online {
BlockKVMessageTape append_block_kv_messages(DeviceProgram& p,const SettleBlockPlan& layout,
    const weighted::KVConfig& kv,const weighted::RegionalConfig& regional,const BlockKVMessageInput& x,
    const BlockKVMessageParameters& w,const at::Tensor& error,int64_t budget,bool projection){
  using namespace block_selection_detail;
  kv.validate();regional.validate();
  if(!x.read.present.defined()||x.read.present.dim()!=2||(!x.proposal_bank&&(!x.proposal.floating.defined()||x.proposal.floating.dim()!=2))||
     !x.proposal.integer.defined()||x.proposal.integer.dim()!=2||x.proposal.integer.size(1)<=4)
    throw std::invalid_argument("KV message block requires separate actual Read/proposal snapshots");
  const auto frames=x.read.present.size(0),nodes=x.read.present.size(1),rows=x.read.present.numel();
  if(frames<1||nodes<1||(!x.proposal_bank&&x.proposal.floating.size(0)!=rows)||x.proposal.integer.size(0)!=rows)
    throw std::invalid_argument("KV message proposal rows differ");
  const auto owners=x.full_owners.defined()?x.full_owners:x.read.node_owners;
  check(owners,error,at::kLong,{frames,nodes});
  auto inputs=borrowed(x.read,w.read,error);inputs.insert(inputs.end(),{x.proposal.floating,x.proposal.integer,owners,
    w.full.query,w.full.residual.out,w.full.residual.gate,w.full.residual.up,w.full.residual.down});
  if(x.proposal_bank){auto a=block_kv_bank_tensors(*x.proposal_bank->bank);inputs.insert(inputs.end(),a.begin(),a.end());}
  const WeightedStorageIncrement storage(p,std::move(inputs));
  auto remaining=[&](){const auto bytes=storage.bytes();admit(bytes,budget);return budget-bytes;};
  auto selection=append_block_read_selection(p,layout,kv,regional,x.read,w.read,error,remaining());
  WeightedKVFullTape full{{WeightedRawLayout(kv,x.proposal.integer.size(1)-4)}};std::optional<BlockKVFullTape> bank_full;
  if(x.proposal_bank){bank_full=append_block_kv_full(p,kv,*x.proposal_bank,selection.node_times,x.read.content.reshape({rows,kv.width}),
      owners.reshape({rows}),selection.active.reshape({rows}),w.full,error,remaining());
    full.output=bank_full->output;full.window.present=selection.active.reshape({rows});
  }else full=append_weighted_kv_full(p,kv,x.proposal,selection.node_times,x.read.content.reshape({rows,kv.width}),
      owners.reshape({rows}),selection.active.reshape({rows}),w.full,error,remaining());
  // The binding algebra owns the actual values consumed by reverse. No control
  // is multiplied into G here. Inactive Full rows are zeros, with no message.
  auto columns=std::make_shared<WeightedRowGraph>(p,selection.owners,w.read.describe.at("summary.matrix").size(0),error,
    remaining(),std::vector<at::Tensor>{selection.controls,full.output});
  const auto c=columns->input(selection.controls.reshape({rows,1}),true),g=columns->input(full.output);
  std::optional<size_t> cg;if(projection)cg=columns->product(c,g);
  BlockMessageColumns messages{columns->value(c).reshape({frames,nodes}),columns->value(g).reshape({frames,nodes,kv.width}),
    full.window.present.reshape({frames,nodes}),selection.node_times.reshape({frames,nodes}),
    cg?columns->value(*cg).reshape({frames,nodes,kv.width}):at::Tensor{}};
  admit(storage.bytes(),budget);
  return {std::move(selection),std::move(full),std::move(columns),c,g,cg,std::move(messages),std::move(bank_full)};
}
}
