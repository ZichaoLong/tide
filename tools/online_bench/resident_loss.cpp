#include "resident_consumer.h"
#include <cmath>
#include <stdexcept>
namespace tide_flow {
ConsumerLoss head_loss(const tide::ResidentWindow& window,const Tensor& head,const Packet& p,Index denominator,bool backward,const HeadBudget& plan) {
  auto indices=at::nonzero(window.valid).reshape({-1});ConsumerLoss out;out.count=indices.numel();
  if(!out.count)return out;
  Tensor master;
  if(backward) {
    master=head.to(at::kFloat);out.root=at::zeros(window.values.sizes(),master.options());
    out.head_gradient=at::zeros_like(master);
  }
  for(Index begin=0;begin<out.count;begin+=plan.rows) {
    auto selected=indices.narrow(0,begin,std::min(plan.rows,out.count-begin));
    auto coordinates=window.coordinates.index_select(0,selected),rows=window.values.index_select(0,selected);
    auto labels=((at::floor_divide(coordinates.select(1,2),p.stride)+1)*7+coordinates.select(1,0)*3).remainder(p.vocab);
    auto logp=at::log_softmax(at::matmul(rows,head.t()).to(at::kFloat),1);
    auto value=-logp.gather(1,labels.unsqueeze(1)).sum()/double(denominator);
    out.value=out.value.defined()?out.value+value:value;++out.chunks;
    if(!backward)continue;
    auto gradient=logp.exp();gradient.scatter_add_(1,labels.unsqueeze(1),-at::ones({selected.numel(),1},gradient.options()));
    gradient.div_(double(denominator));out.root.index_copy_(0,selected,at::matmul(gradient,master));
    out.head_gradient.add_(at::matmul(gradient.t(),rows.to(at::kFloat)));
  }
  return out;
}
Tensor embedding_gradient(const tide::ResidentGradients& gradient,const Tensor& embedding) {
  Tensor output;
  for(const auto& b:gradient.boundaries) {
    auto index=at::nonzero(b.valid&b.connected&(b.coordinates.select(1,3)==0)).reshape({-1});
    if(!index.numel())continue;
    auto coord=b.coordinates.index_select(0,index);
    auto ids=(coord.select(1,5)*7+coord.select(1,0)*3).remainder(embedding.size(0));
    if(!output.defined())output=at::zeros_like(embedding,embedding.options().dtype(at::kFloat));
    output.index_add_(0,ids,b.values.index_select(0,index));
  }
  return output;
}
ConsumerOptimizer::ConsumerOptimizer(Tensor embedding,Tensor head,std::string kind)
    :payloads_{std::move(embedding),std::move(head)},kind_(std::move(kind)) {
  if(kind_!="sgd"&&kind_!="adamw")throw std::invalid_argument("unknown consumer optimizer");
  for(size_t i=0;i<2;++i)states_[i].master=payloads_[i].detach().to(at::kFloat).clone();
}
void ConsumerOptimizer::prepare(const Tensor& embedding,const Tensor& head) {
  if(ready_)throw std::logic_error("uncommitted consumer proposal");
  std::array<Tensor,2> gradients{embedding,head};std::vector<Tensor> flags;
  for(size_t i=0;i<2;++i) {
    const auto& g=gradients[i];const auto& old=states_[i];auto& next=proposal_[i];next={};rounded_[i]=Tensor();
    if(!g.defined())continue;
    if(g.scalar_type()!=at::kFloat||g.sizes()!=old.master.sizes()||g.device()!=old.master.device())
      throw std::invalid_argument("consumer gradient requires matching FP32 matrix");
    flags.push_back(at::isfinite(g).all());flags.push_back(at::isfinite(old.master).all());next.step=old.step+1;
    if(kind_=="sgd") {
      auto direction=g+.001*old.master;next.first=old.step?.25*old.first+direction:direction;
      next.master=old.master-.0001*next.first;
    } else {
      next.first=old.step?.9*old.first+.1*g:.1*g;
      next.second=old.step?.999*old.second+.001*g.square():.001*g.square();
      next.master=old.master*(1-.0001*.001)-(.0001/(1-std::pow(.9,next.step)))*next.first/
        (next.second.sqrt()/std::sqrt(1-std::pow(.999,next.step))+1e-6);
    }
    rounded_[i]=next.master.to(payloads_[i].scalar_type());
    for(const auto& t:{next.master,rounded_[i],next.first,next.second})if(t.defined())flags.push_back(at::isfinite(t).all());
  }
  if(!flags.empty()&&!at::stack(flags).all().item<bool>())throw std::runtime_error("nonfinite consumer proposal; optimizer not applied");
  ready_=true;
}
void ConsumerOptimizer::commit() {
  if(!ready_)throw std::logic_error("consumer commit requires finite proposal");
  for(size_t i=0;i<2;++i)if(proposal_[i].master.defined()) {payloads_[i].copy_(rounded_[i]);states_[i]=std::move(proposal_[i]);rounded_[i]=Tensor();}
  ready_=false;
}
} // namespace tide_flow
