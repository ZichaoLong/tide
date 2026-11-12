#include "weighted_bank_gemm.h"
#include "device_launch_tide_weighted_bank_plan.h"
#include "device_launch_tide_weighted_bank_rows.h"
#include "device_launch_tide_weighted_bank_snapshot.h"
#include "device_launch_tide_weighted_bank_next.h"
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void rows(DeviceProgram& p,const WeightedBankPlan& plan,const at::Tensor& source,const at::Tensor& destination,
    const at::Tensor& error,const at::Tensor& cursor,int64_t capacity,int64_t width,bool scatter,int64_t code){
  // One worker owns ragged destinations; odd row widths may share cache lines.
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_bank_rows)(1,stream,
    ptr(plan.permutation),ptr(plan.offsets),ptr(cursor),ptr(source),ptr(destination),ptr(error),
    plan.rows,plan.banks,capacity,width,scatter,code),"banked GEMM stable tile transfer");},
    {plan.permutation,plan.offsets,cursor,source,destination,error});
}
template<class Operation> void groups(DeviceProgram& p,const WeightedBankPlan& plan,const at::Tensor& error,const Operation& operation){
  const auto capacity=weighted_bank_tile(plan.rows);
  auto cursor=at::zeros({2},plan.offsets.options()),branch=at::zeros({1},error.options());
  auto next=[&](bool reset){p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_bank_next)(1,stream,
      ptr(plan.offsets),ptr(cursor),ptr(branch),ptr(error),plan.rows,plan.banks,capacity,reset),"device actual-bank tile cursor");},
      {plan.offsets,cursor,branch,error});};
  const auto head=p.label(),done=p.label(),body=p.label();
  next(true);p.mark(head);next(false);p.branch(branch,{done,body});
  p.mark(body);operation(cursor,cursor.narrow(0,0,1),capacity);p.branch(branch,{head});
  p.mark(done);
}
}
WeightedBankPlan append_weighted_bank_plan(DeviceProgram& p,const at::Tensor& owners,
    const at::Tensor& roots,int64_t banks,const at::Tensor& error,bool reverse){
  const auto n=owners.numel();weighted_bank_shape(n,1,1,banks);
  WeightedBankPlan result{at::zeros_like(owners),at::zeros({banks+1},owners.options()),
    at::zeros({n},owners.options().dtype(at::kBool)),
    at::zeros({banks},owners.options().dtype(at::kBool)),n,banks};
  auto cursor=at::zeros({banks},owners.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_bank_plan)(1,stream,
    ptr(owners),ptr(roots),ptr(result.permutation),ptr(result.offsets),ptr(cursor),
    ptr(result.row_connected),ptr(result.owner_connected),ptr(error),n,banks,reverse),"actual bank stable device partition");},
    {owners,roots,result.permutation,result.offsets,cursor,result.row_connected,result.owner_connected,error});
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
  const auto in=input.size(1),out=output.size(1),tile=weighted_bank_tile(plan.rows);
  auto x=at::zeros({tile,in},input.options());
  auto y=at::zeros({tile,out},input.options()),wt=at::zeros({plan.banks,in,out},input.options());
  auto bank_matrix=at::zeros({1,in,out},input.options()),bank_bias=bias.defined()?at::zeros({1,out},input.options()):at::Tensor{};
  p.permute(matrix,{0,2,1},wt);p.zero(output);
  groups(p,plan,error,[&](const at::Tensor& cursor,const at::Tensor& bank,int64_t capacity){
    rows(p,plan,input,x,error,cursor,capacity,in,false,20);
    p.index_select(wt,0,bank,bank_matrix);
    p.batch_matmul(x.unsqueeze(0),bank_matrix,y.unsqueeze(0));
    if(bias.defined()){p.index_select(bias,0,bank,bank_bias);p.add(y,bank_bias);}
    rows(p,plan,y,output,error,cursor,capacity,out,true,20);
  });
}
void append_weighted_bank_gemm_vjp(DeviceProgram& p,const WeightedBankPlan& plan,const at::Tensor& input,
    const at::Tensor& matrix,const at::Tensor& gradient,const at::Tensor& input_gradient,
    const at::Tensor& matrix_gradient,const at::Tensor& bias_gradient,const at::Tensor& error){
  const auto in=input.size(1),out=gradient.size(1),tile=weighted_bank_tile(plan.rows);
  auto x=at::zeros({tile,in},input.options()),g=at::zeros({tile,out},input.options());
  auto dx=at::zeros_like(x),gt=at::zeros({1,out,tile},input.options());
  auto bank_matrix=at::zeros({1,out,in},input.options()),dw=at::zeros_like(bank_matrix);
  auto db=bias_gradient.defined()?at::zeros({1,out},input.options()):at::Tensor{};
  auto previous_bias=bias_gradient.defined()?at::zeros_like(db):at::Tensor{};
  p.zero(input_gradient);p.zero(matrix_gradient);if(bias_gradient.defined())p.zero(bias_gradient);
  groups(p,plan,error,[&](const at::Tensor& cursor,const at::Tensor& bank,int64_t capacity){
    rows(p,plan,input,x,error,cursor,capacity,in,false,23);rows(p,plan,gradient,g,error,cursor,capacity,out,false,23);
    p.permute(g.unsqueeze(0),{0,2,1},gt);p.batch_matmul(gt,x.unsqueeze(0),dw);
    // Serial stable tiles have disjoint dX rows, but share a parameter bank.
    // Reuse bank_matrix for the partial sum before loading the saved weight.
    p.index_select(matrix_gradient,0,bank,bank_matrix);p.add(bank_matrix,dw);p.index_copy(matrix_gradient,0,bank,bank_matrix);
    p.index_select(matrix,0,bank,bank_matrix);p.batch_matmul(g.unsqueeze(0),bank_matrix,dx.unsqueeze(0));
    if(bias_gradient.defined()){p.sum(g,0,true,db);p.index_select(bias_gradient,0,bank,previous_bias);
      p.add(previous_bias,db);p.index_copy(bias_gradient,0,bank,previous_bias);}
    rows(p,plan,dx,input_gradient,error,cursor,capacity,in,true,23);
  });
  append_weighted_bank_snapshot(p,plan.owner_connected,matrix_gradient,bias_gradient,matrix_gradient,bias_gradient,error,23);
}
}
