#include "bounded_check.h"
#include "graph_replay.h"
#include "peer_transport.h"
#include <iostream>

namespace accelerator_scale::bounded {
void check_peer(at::Device device,at::ScalarType dtype) {
  const at::Device other(device.type(),device.index()+1);GraphReplay replay({device,other});
  at::AutoGradMode grad(true);
  auto cpu=at::arange(32,at::TensorOptions().dtype(at::kFloat))*.01;
  auto x=cpu.to(device,dtype).set_requires_grad(true);
  auto unused=at::ones_like(x).to(other).set_requires_grad(true),zero=at::ones_like(x).set_requires_grad(true);
  auto host_keys=at::arange(32,at::TensorOptions().dtype(at::kLong))+(int64_t(1)<<55);
  auto integers=host_keys.to(device);
  Tensor loss,returned_keys,returned_scalar,returned_mask,returned_small,stress;std::vector<Tensor> gradients;
  auto execute=[&] {
    auto a=replay_peer_copy(x.reshape({4,8}).transpose(0,1),other).transpose(0,1).reshape({32})*3;
    auto b=replay_peer_copy(a,device)+x*2;
    auto result=replay_peer_copy(b,other);
    loss=result.to(at::kFloat).square().mean()+replay_peer_copy(zero,other).to(at::kFloat).sum()*0;
    gradients=*replay_peer_vjp(loss,{x,unused,zero},at::ones_like(loss),false);
    at::NoGradGuard no_grad;
    returned_keys=replay_peer_copy(replay_peer_copy(integers.reshape({4,8}).transpose(0,1),other)+17,device);
    returned_scalar=replay_peer_copy(replay_peer_copy(integers[3],other)+1,device);
    returned_mask=replay_peer_copy(replay_peer_copy((integers.remainder(2)==0).slice(0,1,3),other),device);
    returned_small=replay_peer_copy(replay_peer_copy(x.slice(0,3,5),other)*2,device);
    // Temporary storage is deliberately released/reused after every pull.
    // Multiple replays must consume/reset the same bounded notification pair.
    stress=integers;
    for(Index i=0;i<64;++i)stress=replay_peer_copy(replay_peer_copy(stress+1,other)+2,device);
  };
  auto check=[&](Index trial) {
    auto input=(cpu+trial*.02).to(dtype);auto expected=input.to(at::kFloat)*5;
    const double rtol=dtype==at::kHalf?2e-2:1e-5,atol=dtype==at::kHalf?1e-3:1e-6;
    close(loss,expected.square().mean(),rtol,atol,"peer analytic loss");
    close(gradients[0],(expected*10/32).to(dtype),rtol,atol,"peer analytic VJP");
    if(gradients[1].defined() || !gradients[2].defined() || gradients[2].count_nonzero().item<Index>())
      throw std::runtime_error("peer isolated None/zero VJP");
    auto host=host_keys+trial;
    if(!at::equal(returned_keys.cpu(),(host+17).reshape({4,8}).transpose(0,1))
        || !at::equal(returned_scalar.cpu(),host[3]+1) || !at::equal(returned_mask.cpu(),(host.remainder(2)==0).slice(0,1,3)))
      throw std::runtime_error("peer exact int64/bool/offset transfer");
    close(returned_small,input.slice(0,3,5)*2,rtol,atol,"peer small offset payload");
    if(!at::equal(stress.cpu(),host+64*3))throw std::runtime_error("peer reusable buffer/notification ordering");
    const auto inventory=replay.peer_inventory();
    if(inventory.first!=4 || inventory.second!=0)throw std::runtime_error("peer resources grew with transfer count");
  };
  execute();replay.synchronize();check(0);replay.capture(execute);
  for(Index trial=0;trial<3;++trial) {
    {at::NoGradGuard no_grad;x.copy_((cpu+trial*.02).to(dtype));integers.copy_(host_keys+trial);}replay.synchronize();
    replay.replay();replay.synchronize();check(trial);
    std::cout<<"PASS peer roundtrip strided/scalar/int64/bool and analytic VJP None/zero trial="<<trial<<'\n'<<std::flush;
  }
}
}  // namespace accelerator_scale::bounded
