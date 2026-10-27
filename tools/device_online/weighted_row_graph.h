#pragma once
#include "weighted_projection.h"
#include "weighted_head.h"
#include "weighted_convolution.h"
#include <map>
#include <optional>
#include <string>
namespace tide::device_online {
// Static local algebra, not graph-event scheduling. All rows and parameter
// owners are runtime device operands. Declared one-pass VJP, no autograd replay.
enum class WeightedRowOp {Input,Constant,Projection,Head,Norm,Cat,Slice,Add,Product,Scale,Sum,Sigmoid,Silu,Log1p,Clamp,Exp,Sqrt,Softplus,Divide,Matmul,Convolution};
struct WeightedRowExpression {
  WeightedRowOp kind;at::Tensor value;
  std::vector<size_t> parents;std::vector<at::Tensor> indices;
  double scalar=0;std::string role;
  at::Tensor rstd;
  std::optional<WeightedProjectionTape> projection;
  std::optional<WeightedHeadTape> head;
  std::optional<WeightedConvTape> convolution;
  std::vector<int64_t> dimensions;
};
struct WeightedRowAdjoint {at::Tensor value,connected;};
struct WeightedRowReverse {
  std::vector<WeightedRowAdjoint> rows;
  std::map<std::string,WeightedRowAdjoint> parameters;
};
struct WeightedRowRoot {size_t index;at::Tensor value,connected;};
class WeightedRowGraph {
 public:
  WeightedRowGraph(DeviceProgram&,const at::Tensor& owners,int64_t banks,const at::Tensor& error,int64_t budget);
  size_t input(const at::Tensor&,bool nonnegative=false);
  size_t constant(int64_t width,double value);
  size_t projection(size_t,const WeightedProjectionParameters&,const std::string& role);
  size_t head(size_t,const WeightedHeadParameters&,const std::string& role);
  size_t norm(size_t,double epsilon);
  size_t cat(const std::vector<size_t>&);
  size_t slice(size_t,int64_t start,int64_t count);
  size_t add(size_t,size_t);
  size_t product(size_t,size_t);
  size_t divide(size_t,size_t);
  size_t matmul(size_t,size_t,int64_t m,int64_t k,int64_t n);
  size_t convolution(size_t current,size_t cache,const WeightedRawLayout&,const at::Tensor& integer,
    const at::Tensor& times,const WeightedConvParameters&,int64_t stride,const std::string& role);
  size_t scale(size_t,double factor,double offset=0);
  size_t sum(size_t);
  size_t sigmoid(size_t);
  size_t silu(size_t);
  size_t log1p(size_t);
  size_t exp(size_t);
  size_t sqrt(size_t);
  size_t softplus(size_t);
  size_t clamp_max(size_t,double bound);
  const at::Tensor& value(size_t) const;
  WeightedRowReverse reverse(DeviceProgram&,const std::vector<WeightedRowRoot>&,const at::Tensor& error,int64_t budget) const;
 private:
  size_t push(WeightedRowExpression);
  size_t unary(WeightedRowOp,size_t,double scalar=0);
  DeviceProgram* program_;at::Tensor owners_,error_;int64_t rows_,banks_,budget_,initial_bytes_;long double retained_=0;
  std::vector<WeightedRowExpression> expressions_;
};
}
