#include "resident_checkpoint.h"
#include "tide/graph_checkpoint.h"
#include <ATen/core/grad_mode.h>
#include <torch/serialize.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <system_error>
namespace weighted_benchmark {
namespace {
void sync_path(const std::filesystem::path& path,bool directory=false){
  int fd=open(path.c_str(),O_RDONLY|(directory?O_DIRECTORY:0));
  if(fd<0)throw std::system_error(errno,std::generic_category(),"open checkpoint for fsync");
  const auto status=fsync(fd);const auto error=errno;close(fd);
  if(status)throw std::system_error(error,std::generic_category(),"fsync checkpoint");
}
Json group(const OptimizerGroup& g){
  Json j;
#define FIELD(name) j[#name]=g.name
  FIELD(parameters);FIELD(lr);FIELD(weight_decay);FIELD(momentum);FIELD(dampening);
  FIELD(beta1);FIELD(beta2);FIELD(eps);FIELD(nesterov);FIELD(amsgrad);FIELD(maximize);
#undef FIELD
  return j;
}
OptimizerGroup group(const Json& j){
  OptimizerGroup g;
#define FIELD(name) g.name=j.at(#name).get<decltype(g.name)>()
  FIELD(parameters);FIELD(lr);FIELD(weight_decay);FIELD(momentum);FIELD(dampening);
  FIELD(beta1);FIELD(beta2);FIELD(eps);FIELD(nesterov);FIELD(amsgrad);FIELD(maximize);
#undef FIELD
  return g;
}
}
void save_resident(const std::filesystem::path& path,Fixture data,const ResidentTrainingCheckpoint& c){
  at::NoGradGuard guard;
  if(std::filesystem::exists(path))throw std::invalid_argument("resident checkpoint destination exists");
  const auto staging=path.string()+".partial";
  if(!std::filesystem::create_directory(staging))throw std::invalid_argument("resident checkpoint staging exists");
  // data is a separately decoded material owner, never the live session's model.
  const auto registry=data.model.parameters(false);
  if(registry.alias_partitions()!=c.aliases)throw std::invalid_argument("checkpoint aliases differ");
  for(const auto& [name,tensor]:c.parameters)registry.value(name).copy_(tensor);
  GraphCheckpoint::save(std::filesystem::path(staging)/"graph.pt",data.graph,data.model,c.continuation);
  torch::serialize::OutputArchive archive;
#define FIELD(name) archive.write(#name,c.state.name,false)
  FIELD(values);FIELD(first);FIELD(second);FIELD(maximum);FIELD(steps);FIELD(corrections);
#undef FIELD
  archive.save_to(staging+"/optimizer.pt");sync_path(staging+"/optimizer.pt");
  Json groups=Json::array();for(const auto& g:c.groups)groups.push_back(group(g));
  write(std::filesystem::path(staging)/"metadata.json",Json{
    {"codec","tide-weighted-benchmark-resume-v1"},{"schema",c.schema},{"generation",c.generation},{"next_token",c.next_token},
    {"identity",data.graph.identity},{"mode",c.mode},{"zeta",c.zeta},{"aliases",c.aliases},{"trainable",c.trainable},
    {"optimizer",c.optimizer==ResidentOptimizerKind::sgd?"sgd":"adamw"},{"groups",groups},{"offsets",c.offsets}});
  sync_path(staging,true);std::filesystem::rename(staging,path);sync_path(path.parent_path(),true);
}
ResidentTrainingCheckpoint load_resident(const std::filesystem::path& path,Fixture& data){
  at::NoGradGuard guard;std::ifstream stream(path/"metadata.json");if(!stream)throw std::invalid_argument("missing resident checkpoint metadata");
  auto j=Json::parse(stream);if(j.at("codec")!="tide-weighted-benchmark-resume-v1"||j.at("identity")!=data.graph.identity)
    throw std::invalid_argument("resident checkpoint codec/identity differs");
  ResidentTrainingCheckpoint c;
#define FIELD(name) c.name=j.at(#name).get<decltype(c.name)>()
  FIELD(schema);FIELD(generation);FIELD(next_token);FIELD(mode);FIELD(zeta);FIELD(aliases);FIELD(trainable);FIELD(offsets);
#undef FIELD
  if(j.at("optimizer")!="sgd"&&j.at("optimizer")!="adamw")throw std::invalid_argument("unknown checkpoint optimizer");
  c.optimizer=j.at("optimizer")=="sgd"?ResidentOptimizerKind::sgd:ResidentOptimizerKind::adamw;
  for(const auto& g:j.at("groups"))c.groups.push_back(group(g));
  torch::serialize::InputArchive archive;archive.load_from((path/"optimizer.pt").string(),at::Device(at::kCPU));
#define FIELD(name) archive.read(#name,c.state.name,false)
  FIELD(values);FIELD(first);FIELD(second);FIELD(maximum);FIELD(steps);FIELD(corrections);
#undef FIELD
  c.continuation=GraphCheckpoint::load(path/"graph.pt",data.graph,data.model);
  for(const auto& p:data.model.parameters(false).owners())c.parameters[p.canonical]=p.value.detach().clone();
  if(data.model.parameters(false).alias_partitions()!=c.aliases)throw std::invalid_argument("restored aliases differ");
  return c;
}
}
