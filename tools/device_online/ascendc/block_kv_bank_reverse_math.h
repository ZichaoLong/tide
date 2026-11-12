#pragma once
#include "weighted_allocation_math.h"
namespace tide_block_kv_bank_reverse_device {
using namespace tide_weighted_device;
__aicore__ inline void numerical(__gm__ I* fields,__gm__ uint8_t* valid,
    __gm__ float* keys,__gm__ float* values,__gm__ float* summaries,__gm__ float* weights,
    __gm__ float* initial,__gm__ float* writes,__gm__ int32_t* error,
    I batch,I length,I nodes,I capacity,I entries,I width,I write_width){
  if(error[0])return;
  for(I b=0;b<batch;++b)for(I n=0;n<nodes;++n){const I owner=b*nodes+n;
    for(I e=0;e<entries;++e){const I index=owner*entries+e;if(!valid[index])continue;
      const I row=(b*length+e-capacity)*nodes+n;
      for(I f=1;f<5;++f){const I off=fields[4*f],wo=fields[4*f+1],size=fields[4*f+2];
        auto source=f==1?keys:f==2?values:f==3?summaries:weights;
        for(I j=0;j<size;++j){const float g=source[index*size+j];if(!finite(g)){error[0]=87;return;}
          if(e<capacity)initial[owner*width+off+e*size+j]+=g;else writes[row*write_width+wo+j]+=g;}
      }
    }
  }
}
// A raw field's causal dependencies include every positive write in its
// prefix, even if membership/window/zero coefficients remove numerical use.
__aicore__ inline void snapshot(__gm__ I* fields,__gm__ uint8_t* written,__gm__ I* mapping,__gm__ I* meta,
    __gm__ float* bar,__gm__ uint8_t* roots,__gm__ float* initial,__gm__ uint8_t* initial_roots,
    __gm__ float* writes,__gm__ uint8_t* write_roots,__gm__ int32_t* error,
    I batch,I length,I nodes,I capacity,I entries,I width,I write_width,I queries,I prefix,I raw){
  if(error[0])return;
  for(I b=0;b<batch;++b)for(I n=0;n<nodes;++n){const I owner=b*nodes+n;
    for(I f=0;f<5;++f){I limit=-1;const I off=fields[4*f],wo=fields[4*f+1],size=fields[4*f+2];
      for(I q=0;q<queries;++q){const I row=(b*queries+q)*nodes+n;if(!roots[row*5+f])continue;
        initial_roots[owner*5+f]=1;const I end=q+prefix;if(end<0||end>length){error[0]=87;return;}if(end>limit)limit=end;
        if(!raw)continue;const I count=meta[row*(capacity+4)+2];if(count<0||count>capacity){error[0]=87;return;}
        for(I i=0;i<(f?count:1);++i){const I source=f?mapping[row*capacity+i]:owner*entries;
          if(source<owner*entries||source>=(owner+1)*entries){error[0]=87;return;}const I e=source-owner*entries;
          for(I j=0;j<size;++j){const float g=bar[row*width+off+i*size+j];if(!finite(g)){error[0]=87;return;}
            __gm__ float* target;if(!f||e<capacity)target=initial+owner*width+off+(f?e*size:0)+j;
            else target=writes+((b*length+e-capacity)*nodes+n)*write_width+wo+j;
            *target+=g;if(!finite(*target)){error[0]=87;return;}
          }
        }
      }
      if(f&&limit>=0)for(I q=0;q<limit;++q){const I row=(b*length+q)*nodes+n;if(written[row])write_roots[row*5+f]=1;}
    }
  }
}
}
