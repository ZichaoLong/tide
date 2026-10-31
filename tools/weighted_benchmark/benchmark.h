#pragma once
// Static benchmark input codec only. No reference trace enters a candidate.
#include "tide/weighted.h"
#include "tide/allocation_failure.h"
#include <nlohmann/json.hpp>
#include <filesystem>
namespace weighted_benchmark {
using Json=nlohmann::json;
using namespace tide;
struct Fixture {Graph graph;Model model;std::vector<External> inputs;Index span,samples,positions;};
Fixture fixture(const Json&,at::ScalarType dtype=at::kFloat);
std::vector<Tensor> tensors(const Json&,at::ScalarType dtype=at::kFloat);
std::vector<External> inputs(const Fixture&,Index step);
Continuation detached(Continuation);
Result detached(Result);
Tensor loss(const Result&,bool allow_empty=false);
Json work(const Result&);
void write(const std::filesystem::path&,const Json&);
Json allocation_failure(const tide::weighted::AllocationFailure&);
struct Step {Result result;double loss;};
// Final actual eager SGD state. Undefined gradients/slots stay undefined.
struct TrainingState {std::map<std::string,Tensor> gradients,momentum,masters;};
void compare_tensors(const std::map<std::string,Tensor>&,const std::map<std::string,Tensor>&,
                     at::ScalarType,const std::string&,bool exact=false);
class Resident {
 public:
  virtual ~Resident()=default;
  virtual Step step(Index)=0;
  virtual std::map<std::string,Tensor> parameters()const=0;
  virtual TrainingState training_state()const=0;
  virtual Continuation checkpoint()const=0;
  virtual void save(const std::filesystem::path&,int64_t max_file_bytes)const=0;
  virtual void restore(const std::filesystem::path&,int64_t max_file_bytes)=0;
  virtual void close()=0;
};
#ifdef TIDE_WEIGHTED_RESIDENT
std::unique_ptr<Resident> resident(const Json&,at::Device,bool training,bool prefill,at::ScalarType,
                                   const std::vector<at::Device>& peers);
#endif
}
