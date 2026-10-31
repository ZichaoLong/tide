#include "run_checkpoint.h"
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <system_error>
#include <limits>
#include <algorithm>
namespace weighted_benchmark {
namespace {
void sync_directory(const std::filesystem::path& path) {
  const int fd=open(path.c_str(),O_RDONLY|O_DIRECTORY);
  if(fd<0)throw std::system_error(errno,std::generic_category(),"open checkpoint directory");
  const auto status=fsync(fd),error=errno;close(fd);
  if(status)throw std::system_error(error,std::generic_category(),"fsync checkpoint directory");
}
Index cut(const Json& metadata) {
  const auto& value=metadata.at("next_window");const auto span=metadata.at("identity").at("span").get<Index>();
  if(!value.is_number_integer()||value.is_boolean()||value.get<Index>()<0||span<1||
     value.get<Index>()>std::numeric_limits<Index>::max()/span)throw std::invalid_argument("invalid checkpoint window/cut");
  return value.get<Index>()*span;
}
std::filesystem::path generation(const std::filesystem::path& directory) {
  std::ifstream pointer(directory/"latest.json");
  if(!pointer)return directory; // Direct v1 bundles remain readable.
  const auto index=Json::parse(pointer);const auto name=index.at("generation").get<std::string>();
  if(index.at("schema")!="tide-weighted-checkpoint-pointer-v1"||name.rfind("window-",0)!=0||name.size()==7||
     !std::all_of(name.begin()+7,name.end(),[](char c){return c>='0'&&c<='9';}))
    throw std::invalid_argument("invalid checkpoint latest generation");
  return directory/name;
}
Json manifest(const std::filesystem::path& selected,const Json& identity) {
  std::ifstream stream(selected/"manifest.json");if(!stream)throw std::invalid_argument("resume requires a complete checkpoint generation manifest");
  auto metadata=Json::parse(stream);
  if(metadata.at("schema")!="tide-weighted-run-checkpoint-v1"||metadata.at("identity")!=identity||metadata.at("state")!="state.pt")
    throw std::invalid_argument("resume model/workload/precision/task/codec identity differs");
  cut(metadata);
  if(!std::filesystem::exists(selected/"state.pt"))throw std::invalid_argument("resume checkpoint payload is missing");
  return metadata;
}
}
Json checkpoint_manifest(const std::filesystem::path& directory,const Json& identity) {
  return manifest(generation(directory),identity);
}
void save_run_checkpoint(const std::filesystem::path& directory,Resident& owner,const Json& identity,Index next,int64_t maximum) {
  std::filesystem::path previous;
  if(std::filesystem::exists(directory)) {
    previous=generation(directory);manifest(previous,identity);
    if(previous==directory)throw std::invalid_argument("cannot replace a direct checkpoint bundle with generations");
  }else if(!std::filesystem::create_directory(directory))throw std::invalid_argument("cannot create checkpoint root");
  const auto name="window-"+std::to_string(next);const auto destination=directory/name;
  if(std::filesystem::exists(destination))throw std::invalid_argument("run checkpoint generation exists");
  const auto staging=directory/(name+".partial");
  if(!std::filesystem::create_directory(staging))throw std::invalid_argument("run checkpoint staging exists");
  Json metadata={{"schema","tide-weighted-run-checkpoint-v1"},{"identity",identity},{"next_window",next},{"state","state.pt"}};
  if(cut(metadata)!=owner.cut())throw std::logic_error("saved checkpoint next_window differs from live cut");
  owner.save(staging/"state.pt",maximum);write(staging/"manifest.json",metadata);
  sync_directory(staging);std::filesystem::rename(staging,destination);sync_directory(directory);
  write(directory/"latest.json",{{"schema","tide-weighted-checkpoint-pointer-v1"},{"generation",name}});
  if(!previous.empty())std::filesystem::remove_all(previous); // Only this writer's prior complete generation.
  sync_directory(directory);sync_directory(directory.parent_path());
}
Index restore_run_checkpoint(const std::filesystem::path& directory,Resident& owner,const Json& identity,int64_t maximum) {
  const auto selected=generation(directory);const auto metadata=manifest(selected,identity);owner.restore(selected/"state.pt",maximum);
  if(owner.cut()!=cut(metadata))throw std::invalid_argument("restored continuation cut disagrees with checkpoint next_window");
  return metadata.at("next_window").get<Index>();
}
}  // namespace weighted_benchmark
