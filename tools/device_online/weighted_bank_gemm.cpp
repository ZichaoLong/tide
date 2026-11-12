#include "weighted_bank_gemm.h"
#include "device_launch_tide_weighted_bank_plan.h"
#include "device_launch_tide_weighted_bank_rows.h"
#include "device_launch_tide_weighted_bank_snapshot.h"
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void rows(DeviceProgram& p,const WeightedBankPlan& plan,const at::Tensor& source,const at::Tensor& destination,
    const at::Tensor& error,int64_t bank,int64_t capacity,int64_t width,bool scatter,int64_t code){
  // One worker owns ragged destinations; odd row widths may share cache lines.
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_bank_rows)(1,stream,
    ptr(plan.permutation),ptr(plan.offsets),ptr(source),ptr(destination),ptr(error),bank,capacity,width,scatter,code),
    "banked GEMM stable row transfer");},{plan.permutation,plan.offsets,source,destination,error});
}
template<class Operation> void groups(DeviceProgram& p,const WeightedBankPlan& plan,const Operation& operation){
  const auto capacities=weighted_bank_capacities(plan.rows);auto zero=at::zeros({1},plan.buckets.options());
  for(int64_t bank=0;bank<plan.banks;++bank){
    const auto done=p.label();std::vector<size_t> labels{done};
    for(auto capacity:capacities){(void)capacity;labels.push_back(p.label());}
    p.branch(plan.buckets.narrow(0,bank,1),labels);
    for(size_t k=0;k<capacities.size();++k){p.mark(labels[k+1]);operation(bank,capacities[k]);p.branch(zero,{done});}
    p.mark(done);
  }
}
}
WeightedBankPlan append_weighted_bank_plan(DeviceProgram& p,const at::Tensor& owners,
    const at::Tensor& roots,int64_t banks,const at::Tensor& error,bool reverse){
  const auto n=owners.numel();weighted_bank_shape(n,1,1,banks);
  WeightedBankPlan result{at::zeros_like(owners),at::zeros({banks+1},owners.options()),
    at::zeros({banks},error.options()),at::zeros({n},owners.options().dtype(at::kBool)),
    at::zeros({banks},owners.options().dtype(at::kBool)),n,banks};
  auto cursor=at::zeros({banks},owners.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_bank_plan)(1,stream,
    ptr(owners),ptr(roots),ptr(result.permutation),ptr(result.offsets),ptr(cursor),ptr(result.buckets),
    ptr(result.row_connected),ptr(result.owner_connected),ptr(error),n,banks,reverse),"actual bank stable device partition");},
    {owners,roots,result.permutation,result.offsets,cursor,result.buckets,result.row_connected,result.owner_connected,error});
  return result;
}
void append_weighted_bank_snapshot(DeviceProgram& p,const at::Tensor& connected,
    const at::Tensor& matrix,const at::Tensor& bias,const at::Tensor& saved_matrix,
    const at::Tensor& saved_bias,const at::Tensor& error,int64_t code){
  const auto banks=saved_matrix.size(0),out=saved_matrix.size(1),in=saved_matrix.size(2);
  const bool has_bias=bias.defined();const auto sb=has_bias?bias:error,db=has_bias?saved_bias:error;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_bank_snapshot)(32,stream,
    ptr(connected),ptr(matrix),ptr(sb),ptr(saved_matrix),ptr(db),ptr(error),banks,in,out,has_bias,code),
    "actual bank snapshot and finite guard");},{connected,matrix,sb,saved_matrix,db,error});
}
void append_weighted_bank_gemm(DeviceProgram& p,const WeightedBankPlan& plan,const at::Tensor& input,
    const at::Tensor& matrix,const at::Tensor& bias,const at::Tensor& output,const at::Tensor& error){
  const auto in=input.size(1),out=output.size(1);auto x=at::zeros({plan.rows,in},input.options());
  auto y=at::zeros({plan.rows,out},input.options()),wt=at::zeros({plan.banks,in,out},input.options());
  p.permute(matrix,{0,2,1},wt);p.zero(output);
  groups(p,plan,[&](int64_t bank,int64_t capacity){
    auto packed=x.narrow(0,0,capacity),result=y.narrow(0,0,capacity);
    rows(p,plan,input,packed,error,bank,capacity,in,false,20);
    p.batch_matmul(packed.unsqueeze(0),wt.narrow(0,bank,1),result.unsqueeze(0));
    if(bias.defined())p.add(result,bias.narrow(0,bank,1));
    rows(p,plan,result,output,error,bank,capacity,out,true,20);
  });
}
void append_weighted_bank_gemm_vjp(DeviceProgram& p,const WeightedBankPlan& plan,const at::Tensor& input,
    const at::Tensor& matrix,const at::Tensor& gradient,const at::Tensor& input_gradient,
    const at::Tensor& matrix_gradient,const at::Tensor& bias_gradient,const at::Tensor& error){
  const auto in=input.size(1),out=gradient.size(1);
  auto x=at::zeros({plan.rows,in},input.options()),g=at::zeros({plan.rows,out},input.options());
  auto dx=at::zeros_like(x),gt=at::zeros({plan.rows*out},input.options());
  p.zero(input_gradient);p.zero(matrix_gradient);if(bias_gradient.defined())p.zero(bias_gradient);
  groups(p,plan,[&](int64_t bank,int64_t capacity){
    auto px=x.narrow(0,0,capacity),pg=g.narrow(0,0,capacity),pd=dx.narrow(0,0,capacity);
    auto transposed=gt.narrow(0,0,capacity*out).view({1,out,capacity});
    rows(p,plan,input,px,error,bank,capacity,in,false,23);rows(p,plan,gradient,pg,error,bank,capacity,out,false,23);
    p.permute(pg.unsqueeze(0),{0,2,1},transposed);
    p.batch_matmul(transposed,px.unsqueeze(0),matrix_gradient.narrow(0,bank,1));
    p.batch_matmul(pg.unsqueeze(0),matrix.narrow(0,bank,1),pd.unsqueeze(0));
    if(bias_gradient.defined())p.sum(pg,0,true,bias_gradient.narrow(0,bank,1));
    rows(p,plan,pd,input_gradient,error,bank,capacity,in,true,23);
  });
  append_weighted_bank_snapshot(p,plan.owner_connected,matrix_gradient,bias_gradient,matrix_gradient,bias_gradient,error,23);
}
}
