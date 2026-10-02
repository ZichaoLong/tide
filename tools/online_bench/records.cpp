#include "consumer.h"
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <iomanip>
#include <cerrno>
#include <stdexcept>

namespace tide_flow {
std::string quoted(const std::string& value) {
  std::ostringstream out;out<<'"';
  for(unsigned char c:value) {
    if(c=='"'||c=='\\')out<<'\\'<<char(c);
    else if(c<32)out<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c)<<std::dec;
    else out<<char(c);
  }
  out<<'"';return out.str();
}
void tensor_json(std::ostream& out,const Tensor& value) {
  if(!value.defined()){out<<"null";return;}
  if(value.numel()>100000)throw std::invalid_argument("diagnostic tensor exceeds 100000 elements; disable diagnostics for scale runs");
  auto cpu=value.detach().to(at::kCPU).to(at::kDouble).contiguous().reshape({-1});
  if(!at::isfinite(cpu).all().item<bool>())throw std::runtime_error("nonfinite diagnostic tensor");
  out<<"{\"shape\":[";bool first=true;for(auto s:value.sizes()){if(!first)out<<',';first=false;out<<s;}
  out<<"],\"values\":[";const auto* x=cpu.data_ptr<double>();
  for(Index i=0;i<cpu.numel();++i){if(i)out<<',';out<<std::setprecision(17)<<x[i];}out<<"]}";
}
void window_json(std::ostream& out,Index step,const tide::Result& r,Index sample_begin,Index logical_batch) {
  out<<"{\"kind\":\"window\",\"step\":"<<step<<",\"cut\":"<<r.continuation.cut;
  if(logical_batch)out<<",\"sample_range\":["<<sample_begin<<','<<sample_begin+r.continuation.batch_size<<','<<logical_batch<<']';
  out<<",\"outputs\":[";
  bool first=true;for(const auto& x:r.outputs){if(!first)out<<',';first=false;
    out<<"["<<x.batch+sample_begin<<','<<x.time<<','<<x.port<<',';tensor_json(out,x.value);out<<']';}
  out<<"],\"states\":[";first=true;
  for(const auto& [key,s]:r.continuation.states) {
    if(!first)out<<',';first=false;
    out<<'['<<key.first+sample_begin<<','<<key.second<<','<<s.last_time<<','<<s.observations<<',';tensor_json(out,s.value);out<<",{";
    bool slot=true;for(const auto& [name,value]:s.slots){if(!slot)out<<',';slot=false;out<<quoted(name)<<':';tensor_json(out,value);}out<<"}]";
  }
  out<<"],\"history\":[";first=true;
  for(const auto& [key,h]:r.continuation.history) {
    if(!first)out<<',';first=false;out<<'['<<key.first+sample_begin<<','<<key.second<<','<<h.last_time<<",{";
    bool table=true;for(const auto& [name,counts]:h.node_maps) {
      if(!table)out<<',';table=false;out<<quoted(name)<<": [";bool cell=true;
      for(const auto& [node,count]:counts){if(!cell)out<<',';cell=false;out<<'['<<node<<','<<count<<']';}out<<']';
    }out<<"}]";
  }
  out<<"],\"pending\":[";first=true;for(const auto& a:r.continuation.pending){if(!first)out<<',';first=false;
    out<<'['<<a.batch+sample_begin<<','<<a.node<<','<<a.time<<','<<a.kind<<','<<a.source<<','<<a.position<<',';tensor_json(out,a.value);out<<']';}
  out<<"],\"events\":[";first=true;for(const auto& x:r.trace){if(!first)out<<',';first=false;
    out<<'['<<x.batch+sample_begin<<','<<x.node<<','<<x.time<<','<<(x.active?"true":"false")<<']';}
  out<<"],\"ledger\":[";first=true;for(const auto& [key,v]:r.continuation.ledger){if(!first)out<<',';first=false;
    out<<'['<<key.first+sample_begin<<','<<key.second<<','<<v.first<<','<<v.second<<']';}
  out<<"]}\n";
}
void parameters_json(std::ostream& out,Index step,const tide::ParameterRegistry& registry,bool gradients) {
  out<<"{\"kind\":"<<quoted(gradients?"gradients":"updated")<<",\"step\":"<<step<<",\"parameters\":{";
  bool first=true;for(const auto& owner:registry.owners()){if(!first)out<<',';first=false;out<<quoted(owner.canonical)<<':';
    tensor_json(out,gradients?owner.value.grad():owner.value);}
  out<<"}}\n";
}
void atomic_text(const std::string& path,const std::string& text) {
  const auto tmp=path+".tmp";int fd=::open(tmp.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);
  if(fd<0)throw std::runtime_error("cannot create record staging file");
  try {
    size_t done=0;while(done<text.size()){
      auto n=::write(fd,text.data()+done,text.size()-done);if(n<0&&errno==EINTR)continue;
      if(n<=0)throw std::runtime_error("record write failed");done+=size_t(n);
    }
    if(::fsync(fd))throw std::runtime_error("record fsync failed");::close(fd);fd=-1;
    std::filesystem::rename(tmp,path);
    fd=::open(std::filesystem::path(path).parent_path().c_str(),O_RDONLY|O_DIRECTORY);
    if(fd<0||::fsync(fd))throw std::runtime_error("record directory fsync failed");::close(fd);fd=-1;
  }catch(...){if(fd>=0)::close(fd);std::filesystem::remove(tmp);throw;}
}
} // namespace tide_flow
