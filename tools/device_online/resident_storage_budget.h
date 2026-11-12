#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
namespace tide::device_online {
inline int64_t resident_bytes_add(int64_t a,int64_t b){
  if(a<0||b<0||b>std::numeric_limits<int64_t>::max()-a)
    throw std::invalid_argument("resident storage byte count overflows int64");
  return a+b;
}
inline int64_t resident_bytes_multiply(int64_t elements,int64_t width){
  if(elements<0||width<1||elements>std::numeric_limits<int64_t>::max()/width)
    throw std::invalid_argument("resident storage extent overflows int64");
  return elements*width;
}
inline void require_resident_bytes(int64_t actual,int64_t limit,const char* partition){
  if(actual<0||limit<1||actual>limit)throw std::invalid_argument(std::string("resident block ")+partition+
    " budget exceeded: required="+std::to_string(actual)+" available="+std::to_string(limit));
}
// Keys are StorageImpl identities, not TensorImpl identities or view extents.
class ResidentStorageBytes {
 public:
  void add(const void* storage,uint64_t bytes){
    if(!seen_.insert(storage).second)return;
    if(bytes>uint64_t(std::numeric_limits<int64_t>::max()))throw std::invalid_argument("resident storage exceeds int64");
    bytes_=resident_bytes_add(bytes_,static_cast<int64_t>(bytes));
  }
  int64_t bytes()const{return bytes_;}
 private:
  std::unordered_set<const void*> seen_;int64_t bytes_=0;
};
}
