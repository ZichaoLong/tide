#pragma once
#include "weighted_allocation_math.h"
namespace tide_weighted_reverse_device {
using I=int64_t;
// Actual journal ancestry, never an input-specific precomputed execution order.
// One serialized stage routes all physical atom uses, including parallel edges.
// Event buffers: [stage,node,sample] and [stage,node,sample,payload].
// Boundary buffers contain exact left-pending and current external identities.
__aicore__ inline void route(__gm__ I* coordinates,__gm__ uint8_t* valid,__gm__ I* indices,__gm__ I* lengths,
    __gm__ float* gradients,__gm__ uint8_t* roots,__gm__ I* edge_sources,
    __gm__ I* event_times,__gm__ uint8_t* event_present,__gm__ float* event_bars,__gm__ uint8_t* event_roots,
    __gm__ I* boundary_coordinates,__gm__ uint8_t* boundary_present,__gm__ float* boundary_bars,__gm__ uint8_t* boundary_roots,
    __gm__ I* current_stage,__gm__ I* begin_cut,__gm__ int32_t* error,
    I rows,I slots,I atoms,I width,I edges,I stages,I nodes,I samples,I boundary){
  if(error[0])return;const I stage=current_stage[0],begin=begin_cut[0];
  if(stage<0||stage>stages||begin<0){error[0]=41;return;}
  for(I r=0;r<rows;++r){const I count=lengths[r];if(count<0||count>slots){error[0]=41;return;}
    for(I j=0;j<count;++j){const I local=r*slots+j;if(!roots[local])continue;
      const I atom=indices[local];if(atom<0||atom>=atoms||!valid[atom]){error[0]=41;return;}
      auto meta=coordinates+6*atom;const I sample=meta[0],kind=meta[3],physical=meta[4],position=meta[5];
      if(sample<0||sample>=samples||meta[1]<0||meta[1]>=nodes||meta[2]<0||position<0||(kind!=0&&kind!=1)){
        error[0]=41;return;}
      I target=-1;bool event=false;
      if(kind==1){if(physical<0||physical>=edges||position>=meta[2]){error[0]=41;return;}
        if(position>=begin){event=true;const I source=edge_sources[physical];
          if(source<0||source>=nodes){error[0]=41;return;}
          for(I t=0;t<stage;++t){const I q=(t*nodes+source)*samples+sample;
            if(event_present[q]&&event_times[q]==position){if(target>=0){error[0]=41;return;}target=q;}}
        }
      }else if(physical<0){error[0]=41;return;}
      if(!event)for(I q=0;q<boundary;++q)if(boundary_present[q]){bool same=true;
        for(I k=0;k<6;++k)if(meta[k]!=boundary_coordinates[6*q+k])same=false;
        if(same){if(target>=0){error[0]=41;return;}target=q;}}
      if(target<0){error[0]=41;return;}
      auto destination=event?event_bars:boundary_bars;auto connected=event?event_roots:boundary_roots;
      for(I k=0;k<width;++k){const float value=gradients[local*width+k],sum=(connected[target]?destination[target*width+k]:0.f)+value;
        if(!tide_weighted_device::finite(value)||!tide_weighted_device::finite(sum)){error[0]=37;return;}
        destination[target*width+k]=sum;}
      connected[target]=1;
    }
  }
}
}
