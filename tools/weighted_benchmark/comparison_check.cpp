// Literal CPU checks; no solver, scheduler or device qualification is inferred.
#include "../../cpp/bench/streaming.h"
#include <limits>
#include <iostream>
using namespace tide_bench;
namespace {
void require(bool value,const char* what){if(!value)throw std::runtime_error(what);}
template<class F> void failure(DifferenceKind kind,F test) {
  try{test();}catch(const ComparisonFailure& error){require(error.difference.kind==kind,"wrong comparison class");return;}
  throw std::runtime_error("expected typed comparison failure");
}
tide::Result result() {
  tide::Result out;out.continuation.identity="same-graph";out.continuation.batch_size=1;
  out.continuation.states[{0,0}].value=at::tensor({1.},at::kFloat);
  out.continuation.states[{0,0}].slots["time"]=at::tensor({std::numeric_limits<int64_t>::max()},at::kLong);
  return out;
}
}
int main() {
  try {
    auto a=result(),b=result();b.stats["different_schedule"]=999;compare(a,b,true,at::kFloat);
    a.continuation.states.at({0,0}).value=at::tensor({2.},at::kFloat);
    b.continuation.states.at({0,0}).slots["time"]=at::tensor({std::numeric_limits<int64_t>::max()-1},at::kLong);
    const auto report=collect_compare(a,b,true,at::kFloat,{},-1.,-1.,1);
    require(report.numeric==1&&report.discrete==1&&report.differences.size()==1,"collection stopped before discrete mismatch");
    failure(DifferenceKind::numeric,[&]{compare(a,b,true,at::kFloat);});
    b=result();a=result();b.continuation.identity="different-graph";
    require(collect_compare(a,b,true,at::kFloat).discrete==1,"different graph identity accepted");
    b=result();b.continuation.states.at({0,0}).value=at::zeros({2},at::kFloat);
    failure(DifferenceKind::structure,[&]{collect_compare(a,b,true,at::kFloat);});
    b=result();b.continuation.states[{0,1}].value=at::tensor({std::numeric_limits<double>::infinity()});
    failure(DifferenceKind::nonfinite,[&]{collect_compare(a,b,true,at::kFloat);});
    const std::map<std::string,at::Tensor> zero{{"g",at::zeros({})}},none{{"g",at::Tensor{}}};
    require(collect_tensors(zero,none,at::kFloat,"gradient").discrete==1,"None and connected zero merged");
    failure(DifferenceKind::structure,[&]{collect_tensors({{"p",at::ones({1},at::kHalf)}},
      {{"p",at::ones({1},at::kFloat)}},at::kHalf,"parameter");});
    ComparisonReport sum;sum.limit=1;sum.merge(report);sum.merge(report);
    require(sum.numeric==2&&sum.discrete==2&&sum.differences.size()==1,"report merge lost counts or cap");
    std::cout<<"typed complete comparison checks passed\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
