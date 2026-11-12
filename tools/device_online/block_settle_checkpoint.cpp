#include "block_settle_session.h"
#include <torch/serialize.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
namespace tide::device_online {
namespace {
void names(std::ostream& out,const std::vector<std::string>& xs){out<<xs.size()<<' ';for(const auto& x:xs)out<<std::quoted(x)<<' ';}
size_t count(std::istream& in){int64_t n=-1;in>>n;if(!in||n<0||n>1000000)throw std::invalid_argument("invalid block checkpoint list extent");return n;}
std::vector<std::string> names(std::istream& in){std::vector<std::string> out(count(in));for(auto& x:out)in>>std::quoted(x);return out;}
void sync(const std::filesystem::path& path,bool directory=false){const int fd=::open(path.c_str(),O_RDONLY|(directory?O_DIRECTORY:0));
  if(fd<0)throw std::runtime_error("cannot open block checkpoint for fsync");const int result=::fsync(fd);::close(fd);
  if(result)throw std::runtime_error("block checkpoint fsync failed");}
}
void save_block_settle_checkpoint(const BlockSettleCheckpoint& c,const std::string& path,int64_t max_bytes){
  if(max_bytes<1||c.schema!=1||c.generation<0||c.next_token<0||c.state.size()!=c.bound_initial.size())throw std::invalid_argument("invalid block checkpoint record/budget");
  std::ostringstream meta;meta<<std::setprecision(17)<<c.schema<<' '<<c.generation<<' '<<c.next_token<<' '<<int(c.optimizer)<<' '<<std::quoted(c.identity)<<' ';
  meta<<c.aliases.size()<<' ';for(const auto& xs:c.aliases)names(meta,xs);meta<<c.parameters.size()<<' ';
  torch::serialize::OutputArchive archive;size_t i=0;int64_t remaining=max_bytes;
  auto tensor=[&](const std::string& name,const at::Tensor& x){if(!x.defined()||!x.device().is_cpu()||x.requires_grad())throw std::invalid_argument("block checkpoint must contain detached CPU tensors");
    if(x.nbytes()>uint64_t(remaining))throw std::invalid_argument("block checkpoint tensor bytes exceed budget");remaining-=x.nbytes();archive.write(name,x.contiguous());};
  for(const auto& [name,x]:c.parameters){meta<<std::quoted(name)<<' ';tensor("parameter_"+std::to_string(i++),x);}
  meta<<c.groups.size()<<' ';for(const auto& group:c.groups){names(meta,group.parameters);
    for(auto x:{group.lr,group.weight_decay,group.momentum,group.dampening,group.beta1,group.beta2,group.eps})meta<<x<<' ';
    meta<<group.nesterov<<' '<<group.amsgrad<<' '<<group.maximize<<' ';}
  const std::vector<std::pair<std::string,at::Tensor>> slots{{"values",c.slots.values},{"first",c.slots.first},{"second",c.slots.second},
    {"maximum",c.slots.maximum},{"steps",c.slots.steps},{"corrections",c.slots.corrections},{"last_positions",c.last_positions}};
  for(const auto& [name,x]:slots)tensor(name,x);meta<<c.state.size()<<' ';
  for(size_t r=0;r<c.state.size();++r){const auto suffix=std::to_string(r);tensor("state_f_"+suffix,c.state[r].floating);tensor("state_i_"+suffix,c.state[r].integer);
    tensor("bound_f_"+suffix,c.bound_initial[r].floating);tensor("bound_i_"+suffix,c.bound_initial[r].integer);}
  const auto text=meta.str();if(text.size()>16*1024*1024)throw std::invalid_argument("block checkpoint metadata exceeds capacity");
  archive.write("metadata",at::from_blob(const_cast<char*>(text.data()),{int64_t(text.size())},at::kByte).clone());
  const std::filesystem::path destination(path),parent=destination.has_parent_path()?destination.parent_path():std::filesystem::path(".");
  const auto staging=destination.string()+".tmp."+std::to_string(::getpid());
  try{archive.save_to(staging);if(std::filesystem::file_size(staging)>uint64_t(max_bytes))throw std::invalid_argument("block checkpoint archive exceeds budget");
    sync(staging);std::filesystem::rename(staging,destination);sync(parent,true);}catch(...){std::error_code ignored;std::filesystem::remove(staging,ignored);throw;}
}
BlockSettleCheckpoint load_block_settle_checkpoint(const std::string& path,int64_t max_bytes){
  if(max_bytes<1||std::filesystem::file_size(path)>uint64_t(max_bytes))throw std::invalid_argument("block checkpoint file exceeds load budget");
  torch::serialize::InputArchive archive;archive.load_from(path,at::kCPU);at::Tensor data;archive.read("metadata",data);
  if(data.scalar_type()!=at::kByte||data.dim()!=1||data.numel()>16*1024*1024||!data.is_contiguous())throw std::invalid_argument("invalid block checkpoint metadata");
  std::istringstream meta(std::string(reinterpret_cast<const char*>(data.const_data_ptr()),data.numel()));BlockSettleCheckpoint c;int kind=-1;
  meta>>c.schema>>c.generation>>c.next_token>>kind>>std::quoted(c.identity);
  if(c.schema!=1||c.generation<0||c.next_token<0||(kind!=int(DeviceOptimizerKind::sgd)&&kind!=int(DeviceOptimizerKind::adamw)))throw std::invalid_argument("invalid block checkpoint schema/progress");
  c.optimizer=DeviceOptimizerKind(kind);c.aliases.resize(count(meta));for(auto& xs:c.aliases)xs=names(meta);
  const auto parameters=count(meta);for(size_t i=0;i<parameters;++i){std::string name;meta>>std::quoted(name);at::Tensor x;archive.read("parameter_"+std::to_string(i),x);
    if(!c.parameters.emplace(name,x.detach()).second)throw std::invalid_argument("duplicate block checkpoint parameter");}
  c.groups.resize(count(meta));for(auto& group:c.groups){group.parameters=names(meta);
    meta>>group.lr>>group.weight_decay>>group.momentum>>group.dampening>>group.beta1>>group.beta2>>group.eps>>group.nesterov>>group.amsgrad>>group.maximize;}
  for(const auto& [name,x]:std::vector<std::pair<std::string,at::Tensor*>>{{"values",&c.slots.values},{"first",&c.slots.first},{"second",&c.slots.second},
      {"maximum",&c.slots.maximum},{"steps",&c.slots.steps},{"corrections",&c.slots.corrections},{"last_positions",&c.last_positions}})archive.read(name,*x);
  c.state.resize(count(meta));c.bound_initial.resize(c.state.size());for(size_t r=0;r<c.state.size();++r){const auto suffix=std::to_string(r);
    archive.read("state_f_"+suffix,c.state[r].floating);archive.read("state_i_"+suffix,c.state[r].integer);
    archive.read("bound_f_"+suffix,c.bound_initial[r].floating);archive.read("bound_i_"+suffix,c.bound_initial[r].integer);}
  if(!meta)throw std::invalid_argument("truncated block checkpoint metadata");
  meta>>std::ws;if(!meta.eof())throw std::invalid_argument("malformed block checkpoint metadata");
  return c;
}
}
