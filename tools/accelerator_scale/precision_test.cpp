#include "precision.h"
#include <tide/device.h>
#include <tide/isolated_linear.h>
#include <iostream>
#include <stdexcept>

using namespace accelerator_scale;
void require(bool condition,const char* message) { if(!condition)throw std::runtime_error(message); }
int main() {
  try {
    at::set_num_threads(1);
    pdg_scale::Fixture f;
    for(double value:{1.,2.,3.})f.owners.push_back(at::full({1},value,at::TensorOptions().dtype(at::kHalf)).set_requires_grad(true));
    f.embedding=f.owners.front();
    require(!supported_payload(f.embedding) && supported_kernel_payload(f.embedding),"public payload boundary");
    TrainingConfig c;c.optimizer="sgd";c.learning_rate=.1;c.loss_scale=128.;
    c.backward_threads=3;c.optimizer_threads=2;
    auto duplicated=f;duplicated.owners.push_back(f.owners.front());
    bool duplicate_rejected=false;
    try {TrainingOwners bad(duplicated,c);}catch(const std::invalid_argument&){duplicate_rejected=true;}
    require(duplicate_rejected,"duplicated FP16 owner must not acquire two masters");
    TrainingOwners owners(f,c);owners.zero_grad();
    owners.backward(f.owners[0].to(at::kFloat).sum()*2.+f.owners[2].to(at::kFloat).sum()*0.);
    owners.step();
    require(owners.backward_threads().aten==3 && owners.optimizer_threads().aten==2
      && at::get_num_threads()==1,"phase ATen pools must apply then restore");
    if(owners.backward_threads().openblas)
      require(owners.backward_threads().openblas==3 && owners.optimizer_threads().openblas==2,
        "phase OpenBLAS pools must follow their independent budgets");
    const auto& m=owners.masters();
    require(std::abs(m[0].item<double>()-.799)<1e-6,"FP32 master gradient unscale/SGD formula");
    require(m[1].item<double>()==2. && !m[1].grad().defined(),"None owner was updated");
    require(std::abs(m[2].item<double>()-2.997)<1e-6 && m[2].grad().defined(),"connected zero must decay");
    require(owners.optimizer().state().size()==2,"optimizer absence inventory");
    for(size_t i:{size_t(0),size_t(2)}) {
      require(m[i].scalar_type()==at::kFloat,"master dtype");
      require(at::equal(f.owners[i],m[i].to(at::kHalf)),"payload publication");
    }
    owners.zero_grad();
    require(!f.owners[0].grad().defined() && !m[0].grad().defined(),"zero_grad clears both owners");
    owners.backward(f.owners[0].to(at::kFloat).sum()*1e10);
    auto before=m[0].clone();bool rejected=false;
    try { owners.step(); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected && at::equal(before,m[0]),"overflow must fail before mutation");
    auto weight=at::eye(2,at::TensorOptions().dtype(at::kHalf)).set_requires_grad(true);
    auto x=at::tensor({1.,2.}).to(at::kHalf).set_requires_grad(true);
    auto y=at::tensor({3.,4.}).to(at::kHalf).set_requires_grad(true);
    auto values=isolated_linear({x,y},weight);values[0].sum().backward();
    require(at::equal(values[0],x) && x.grad().defined() && !y.grad().defined(),"FP16 isolated linear values/None");
    std::cout<<"FP16 master/SGD/unscale/None/zero/overflow/isolated linear: passed\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
