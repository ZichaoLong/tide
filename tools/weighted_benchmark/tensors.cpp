// Bounded static material codec. No reference execution, no Python runtime.
#include "benchmark.h"
#include <openssl/evp.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
namespace weighted_benchmark {
namespace {
struct Hash {
  EVP_MD_CTX* ctx=EVP_MD_CTX_new();
  Hash(){if(!ctx||EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr)!=1)throw std::runtime_error("SHA256 initialization failed");}
  ~Hash(){EVP_MD_CTX_free(ctx);}
  void add(const void* p,size_t n){if(EVP_DigestUpdate(ctx,p,n)!=1)throw std::runtime_error("SHA256 update failed");}
  std::string finish(){std::array<unsigned char,EVP_MAX_MD_SIZE> bytes{};unsigned n=0;
    if(EVP_DigestFinal_ex(ctx,bytes.data(),&n)!=1)throw std::runtime_error("SHA256 finalization failed");
    std::ostringstream out;out<<std::hex<<std::setfill('0');for(unsigned i=0;i<n;++i)out<<std::setw(2)<<unsigned(bytes[i]);return out.str();}
};
int64_t size(const Json& row){
  int64_t n=1;for(auto x:row.at("shape").get<std::vector<int64_t>>()){
    if(x<0||(x&&n>std::numeric_limits<int64_t>::max()/4/x))throw std::invalid_argument("invalid fixture tensor shape/size");n*=x;}
  return n;
}
}
std::vector<Tensor> tensors(const Json& data,at::ScalarType dtype){
  if(dtype!=at::kFloat&&dtype!=at::kDouble)throw std::invalid_argument("benchmark dtype requires float32 or float64");
  const bool binary=data.at("schema")=="tide-weighted-benchmark-fixture-v2";
  std::ifstream file;int64_t offset=0,limit=0;
  if(binary){
    uint16_t one=1;if(*reinterpret_cast<uint8_t*>(&one)!=1||data.at("binary").at("byte_order")!="little")
      throw std::invalid_argument("binary fixture requires little endian");
    const auto& meta=data.at("binary");auto name=std::filesystem::path(meta.at("file").get<std::string>());
    if(name.empty()||name.filename()!=name)throw std::invalid_argument("binary tensor file must be a sibling basename");
    auto path=std::filesystem::path(data.at("__fixture_root").get<std::string>())/name;
    limit=meta.at("nbytes");if(limit<0||std::filesystem::file_size(path)!=static_cast<uint64_t>(limit))
      throw std::invalid_argument("binary fixture byte count mismatch");
    file.open(path,std::ios::binary);if(!file)throw std::invalid_argument("missing fixture tensor binary");
    Hash hash;std::array<char,1024*1024> chunk{};
    while(file){file.read(chunk.data(),chunk.size());hash.add(chunk.data(),file.gcount());}
    if(!file.eof()||hash.finish()!=meta.at("sha256"))throw std::invalid_argument("binary fixture SHA256 mismatch");
    file.clear();file.seekg(0);
  }
  std::vector<Tensor> values;int64_t permanent=0;
  for(const auto& row:data.at("tensors")){
    if(row.at("dtype")!="float32")throw std::invalid_argument("benchmark requires explicit FP32 tensors");
    const int64_t n=size(row),element_bytes=dtype==at::kDouble?8:4;
    if(n>std::numeric_limits<int64_t>::max()/element_bytes)throw std::invalid_argument("fixture precision allocation overflow");
    const int64_t bytes=n*element_bytes;
    if(permanent>std::numeric_limits<int64_t>::max()-bytes)throw std::invalid_argument("fixture allocation overflow");
    permanent+=bytes;
  }
  if(binary&&permanent>data.at("max_materialize_bytes").get<int64_t>())
    throw std::invalid_argument("fixture persistent tensors exceed materialization ceiling");
  for(const auto& row:data.at("tensors")){
    auto shape=row.at("shape").get<std::vector<Index>>();const int64_t n=size(row),bytes=n*4;Tensor x;
    if(!binary){auto v=row.at("values").get<std::vector<float>>();
      if(static_cast<int64_t>(v.size())!=n)throw std::invalid_argument("fixture values/shape mismatch");x=at::tensor(v,at::kFloat).reshape(shape);
    }else if(row.at("encoding")=="zeros"){
      if(row.at("trainable").get<bool>())throw std::invalid_argument("zero codec cannot hide trainable parameters");
      x=at::zeros(shape,at::kFloat);
    }else if(row.at("encoding")=="binary"){
      if(row.at("offset")!=offset||row.at("nbytes")!=bytes||bytes>limit-offset)
        throw std::invalid_argument("binary tensor range/shape mismatch");
      x=at::empty(shape,at::kFloat);file.read(static_cast<char*>(x.data_ptr()),bytes);
      if(file.gcount()!=bytes)throw std::invalid_argument("truncated tensor data");
      Hash hash;hash.add(x.data_ptr(),bytes);if(hash.finish()!=row.at("sha256"))throw std::invalid_argument("tensor SHA256 mismatch");
      offset+=bytes;
    }else throw std::invalid_argument("unknown tensor encoding");
    // Promote each TensorImpl table entry once, before alias installation.
    // The hashed input bytes remain the exact original FP32 material.
    values.push_back(x.to(dtype).set_requires_grad(row.at("trainable").get<bool>()));
  }
  if(binary&&offset!=limit)throw std::invalid_argument("unclaimed tensor binary bytes");
  return values;
}
}
