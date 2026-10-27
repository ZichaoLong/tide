// Small single-writer result records use fsynced staging and atomic replacement.
#include "benchmark.h"
#include <cerrno>
#include <fcntl.h>
#include <system_error>
#include <unistd.h>
namespace weighted_benchmark {
void write(const std::filesystem::path& path,const Json& data){
  static uint64_t serial=0;
  const auto staging=path.string()+".pending-"+std::to_string(getpid())+"-"+std::to_string(serial++);
  const auto bytes=data.dump(2)+"\n";
  auto fail=[](const char* operation){throw std::system_error(errno,std::generic_category(),operation);};
  int fd=open(staging.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  if(fd<0)fail("create benchmark record staging");
  try{
    size_t offset=0;while(offset<bytes.size()){
      const auto count=::write(fd,bytes.data()+offset,bytes.size()-offset);
      if(count<0){if(errno==EINTR)continue;fail("write benchmark record");}
      if(!count){errno=EIO;fail("short benchmark record write");}offset+=count;
    }
    if(fsync(fd))fail("fsync benchmark record");
    const auto status=close(fd);fd=-1;if(status)fail("close benchmark record");
    if(rename(staging.c_str(),path.c_str()))fail("replace benchmark record");
    const auto parent=path.parent_path().empty()?std::filesystem::path("."):path.parent_path();
    fd=open(parent.c_str(),O_RDONLY|O_DIRECTORY);if(fd<0)fail("open benchmark record directory");
    if(fsync(fd))fail("fsync benchmark record directory");
    const auto dir_status=close(fd);fd=-1;if(dir_status)fail("close benchmark record directory");
  }catch(...){if(fd>=0)close(fd);unlink(staging.c_str());throw;}
}
}
