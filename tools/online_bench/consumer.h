#pragma once
#include <portable_torch/runtime.hpp>
#include <tide/placement.h>
#include <tide/settle.h>
#include <tide/optimizer.h>
#include <chrono>
#include <sstream>

namespace tide_flow {
using tide::Index;
using tide::Tensor;
using Clock=std::chrono::steady_clock;
inline double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
struct Packet {
  std::string sha,memory,topology;
  Index width=0,batch=0,tokens=0,vocab=0,seed=0,stride=0,budget=0;
  bool clear=false;
  tide::Graph body;
  std::vector<Index> ranks;
  Index parameters() const;
};
struct Config {
  portable_torch::RuntimeOptions runtime;
  std::string packet,family="",schedule="",optimizer="sgd";
  tide::ExecutionPlacement placement;
  bool training=false,diagnostics=false;
  Index steps=3,warmup=1,windows=2,threads=1,workers=1;
  bool packed_sources=false,batch_next=false;
  Index parameter_budget=1024LL*1024*1024;
  Index head_workspace_bytes=4LL*1024*1024*1024;
  Index device_memory_bytes=0;
  Index devices=1;
  std::string owner_policy="locality",chunk_policy="conservative";
  std::map<std::string,Index> resident_limits;
};
struct Fixture {
  tide::Graph graph;
  tide::Model model;
  Tensor embedding,head;
  std::unique_ptr<tide::SettleGraph> settle;
  tide::ParameterRegistry parameters;
};
Config parse(int,char**);
Packet read_packet(const std::string&);
Tensor source_values(const std::string&,const std::vector<Index>&,Index);
Fixture fixture(const Packet&,const Config&,at::Device);
std::string run(const Packet&,const Config&,at::Device,std::ostream* diagnostics);
std::string run_resident(const Packet&,const Config&,at::Device,std::ostream* diagnostics);
std::string quoted(const std::string&);
void tensor_json(std::ostream&,const Tensor&);
void window_json(std::ostream&,Index,const tide::Result&);
void parameters_json(std::ostream&,Index,const tide::ParameterRegistry&,bool gradients);
void atomic_text(const std::string&,const std::string&);
} // namespace tide_flow
