#pragma once
#include "packed_lh_full.h"
#include "lh_precision_check.h"
#include "portable_torch/runtime.hpp"
#include "tide/lh_full.h"
#include <limits>

namespace tide::device_online::test {
namespace lh_component_detail {
inline void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
}
inline std::vector<std::string> lh_profiles() {
  std::vector<std::string> result;
  for(const std::string a:{"relu","silu","identity"})for(const std::string n:{"identity","rms","layer"})
    result.push_back("lh-"+a+"-"+n+"-v1");
  return result;
}
inline Index lh_component(at::Device device,LhPrecision& precision,at::ScalarType dtype=at::kFloat) {
  using lh_component_detail::require;
  Index cases=0;const Index nodes=11,rows=35;const auto names=lh_profiles();
  const auto opts=at::TensorOptions().device(device).dtype(dtype);
  std::vector<Index> kinds{1,2,3,4,5,6,7,8,9,0,2};
  for(Index width:{1,7,33,257})for(Index chunk:{1,4}) {
    auto weight=at::arange(nodes*width,at::kFloat).reshape({nodes,width}).remainder(5)/8+.75;weight=weight.to(dtype);
    auto bias=at::arange(nodes*width,at::kFloat).reshape({nodes,width}).remainder(3)/16;bias=bias.to(dtype);
    weight[10].fill_(std::numeric_limits<float>::quiet_NaN());bias[10].fill_(std::numeric_limits<float>::quiet_NaN());
    PackedLhFull full(kinds,weight,bias,device,rows,chunk,16*1024*1024);
    ActionBatch input{at::zeros({rows,4},opts.dtype(at::kLong)),at::zeros({rows,width},opts),at::zeros({rows},opts.dtype(at::kBool))};
    auto comparison=at::zeros_like(input.values),error=at::zeros({1},opts.dtype(at::kInt)),chunks=at::zeros({1},opts.dtype(at::kLong));
    CannProgram program(device);auto result=full.append_stage(program,input,comparison,error,chunks);program.finish();
    for(Index round=0;round<5;++round) {
      auto coords=at::zeros({rows,4},at::kLong),mask=at::zeros({rows},at::kBool);
      auto h=at::arange(rows*width,at::kFloat).reshape({rows,width})/128,state=at::sin(h)*.5;
      if(round==3)state.fill_(.25); // zero variance, including width one
      if(round==4)state=at::sin(h*128)*.5; // well-conditioned mixed-sign rows
      h=h.to(dtype);state=state.to(dtype);
      auto expected=h.clone();std::vector<Index> totals(10);
      for(Index i=0;i<rows;++i) {
        const Index n=i%nodes;const bool active=round!=0&&n!=10&&(round!=2||i%2==0);
        mask[i].fill_(active);coords[i][1].fill_(active?n:-77);coords[i][2].fill_((Index(1)<<55)+i);
        if(active&&kinds[n]) {
          NodeWeights w;w.bias=at::zeros({width},dtype);w.full_kind=names[kinds[n]-1];
          w.extra={{"lh_norm_weight",weight[n]},{"lh_norm_bias",bias[n]}};
          expected[i].copy_(lh_full_fresh(w,state[i]));++totals[kinds[n]];
        }
        if(!active){state[i].fill_(std::numeric_limits<float>::quiet_NaN());h[i].fill_(std::numeric_limits<float>::quiet_NaN());}
      }
      input.coordinates.copy_(coords);input.values.copy_(h);input.valid.copy_(mask);comparison.copy_(state);chunks.zero_();
      portable_torch::synchronize(device);program.run();
      Index wanted_chunks=0;for(Index total:totals)wanted_chunks+=(total+chunk-1)/chunk;
      require(error.cpu().item<int>()==0,"LH Full refused valid selected rows");
      require(chunks.cpu().item<Index>()==wanted_chunks,"LH Full actual chunk count");
      require(result.values.scalar_type()==dtype,"LH Full changed output precision");
      const auto actual=result.values.cpu();
      for(Index i=0;i<rows;++i)if(mask[i].item<bool>()) {
        const Index node=i%nodes;
        if(kinds[node])precision.check(kinds[node],state[i],weight[node],bias[node],expected[i],actual[i]);
        else require(at::equal(actual[i],expected[i]),"non-LH Full changed");
      }
      if(round==4)require(at::allclose(actual.index({mask}),expected.index({mask}),dtype==at::kHalf?3e-3:1e-5,dtype==at::kHalf?2e-3:1e-6),"well-conditioned LH Full parity");
      ++cases;
    }
    program.close();
  }
  return cases;
}
} // namespace tide::device_online::test
