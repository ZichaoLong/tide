// Literal CPU adjoints test the diagnostic machinery, not graph qualification.
#include "root_values.h"
#include "../../cpp/bench/comparison.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <iostream>
using namespace weighted_benchmark;
namespace {
void require(bool value,const char* what){if(!value)throw std::runtime_error(what);}
void number(const at::Tensor& value,double expected){
  require(value.defined()&&std::abs(value.item<double>()-expected)<1e-10,"literal independent VJP differs");
}
}
int main(){
  try{
    at::AutoGradMode grad(true);const auto options=at::TensorOptions().dtype(at::kDouble);
    auto p=at::tensor({2.},options).set_requires_grad(true);
    auto x=at::tensor({3.},options).set_requires_grad(true);
    auto unused=at::tensor({11.},options).set_requires_grad(true);
    RootDiagnostic d;d.leaves={{"parameter.shared",p},{"input",x},{"disconnected",unused}};
    d.result.outputs.push_back({0,0,0,2*p+p+x}); // Three contributions to one owner.
    tide::Event event;event.control=3*p;event.descriptor_fields["query"]=4*x;d.result.trace.push_back(event);
    d.result.continuation.states[{0,0}].value=p+x;
    d.result.continuation.history[{0,0}].tensors["float"]=p;
    d.result.continuation.history[{0,0}].tensors["int64"]=at::tensor({7},at::kLong);
    d.result.continuation.pending.push_back({0,0,1,1,0,0,x});
    const auto roots=diagnostic_roots(d);require(roots.size()==4,"root categories missing");
    const auto output=diagnostic_vjp(roots.at("output"),d.leaves,false);
    number(output.at("parameter.shared"),37.8);number(output.at("input"),12.6);
    require(!output.at("disconnected").defined(),"disconnected root became zero");
    const auto zero=diagnostic_vjp(roots.at("output"),d.leaves,true);
    number(zero.at("parameter.shared"),0);number(zero.at("input"),0);
    require(!zero.at("disconnected").defined(),"zero cotangent connected unused root");
    const auto control=diagnostic_vjp(roots.at("control"),d.leaves,false);
    number(control.at("parameter.shared"),25.2);require(!control.at("input").defined(),"control reached unrelated root");
    const auto descriptor=diagnostic_vjp(roots.at("descriptor"),d.leaves,false);
    number(descriptor.at("input"),67.2);require(!descriptor.at("parameter.shared").defined(),"descriptor reached unrelated root");
    const auto all=diagnostic_vjp(roots.at("all"),d.leaves,false);
    number(all.at("parameter.shared"),72.8);number(all.at("input"),91.);
    tide_bench::check_finite(all,"literal VJP");
    auto changed=zero;changed["disconnected"]=at::zeros_like(unused);
    require(tide_bench::collect_tensors(changed,zero,at::kDouble,"presence").discrete==1,"None/zero not classified discrete");
    RootDiagnostic empty;empty.leaves=d.leaves;
    const auto absent=diagnostic_vjp(diagnostic_roots(empty).at("descriptor"),empty.leaves,false);
    for(const auto& [_,value]:absent)require(!value.defined(),"empty root unexpectedly connected");
    std::cout<<"literal CPU independent-root checks passed\n";return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
