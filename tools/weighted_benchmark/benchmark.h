#pragma once
// Static benchmark input codec only. No reference trace enters a candidate.
#include "tide/weighted.h"
#include <nlohmann/json.hpp>
#include <filesystem>
namespace weighted_benchmark {
using Json=nlohmann::json;
using namespace tide;
struct Fixture {Graph graph;Model model;std::vector<External> inputs;Index span,samples,positions;};
Fixture fixture(const Json&);
std::vector<External> inputs(const Fixture&,Index step);
Continuation detached(Continuation);
Result detached(Result);
Tensor loss(const Result&);
Json work(const Result&);
void write(const std::filesystem::path&,const Json&);
struct Step {Result result;double loss;};
class Resident {
 public:
  virtual ~Resident()=default;
  virtual Step step(Index)=0;
  virtual std::map<std::string,Tensor> parameters()const=0;
  virtual void close()=0;
};
#ifdef TIDE_WEIGHTED_RESIDENT
std::unique_ptr<Resident> resident(const Json&,at::Device,bool training,bool prefill);
#endif
}
