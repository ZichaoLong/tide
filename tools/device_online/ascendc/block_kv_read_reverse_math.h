#pragma once
#include "weighted_allocation_math.h"
namespace tide_block_kv_read_reverse_device {
using namespace tide_weighted_device;
// Pool flags are [channels,O,L,3]. Structural field roots remain [B,L,N,5].
__aicore__ inline void roots(__gm__ uint8_t* flags,__gm__ float* weights,__gm__ uint8_t* present,
    __gm__ uint8_t* snapshot,__gm__ uint8_t* query,__gm__ uint8_t* keys,__gm__ uint8_t* values,
    __gm__ int32_t* error,I batch,I length,I nodes,I entries,I channels,I kind){
  if(error[0])return;const I rows=batch*length*nodes;
  for(I i=0;i<batch*nodes*entries;++i){keys[i]=0;values[i]=0;}
  for(I b=0;b<batch;++b)for(I n=0;n<nodes;++n)for(I q=0;q<length;++q){const I owner=b*nodes+n,a=owner*length+q,r=(b*length+q)*nodes+n;
    bool weight=false,key=false,value=false;for(I c=0;c<channels;++c){const I f=(c*rows+a)*3;weight=weight||flags[f];value=value||flags[f+1];key=key||flags[f+2];}
    query[r]=kind&&key;for(I f=0;f<5;++f)snapshot[r*5+f]=0;
    if(present[a]){snapshot[r*5+1]=kind&&key;snapshot[r*5+2]=kind&&value;snapshot[r*5+3]=!kind&&value;snapshot[r*5+4]=weight;}
    if(kind)for(I k=0;k<entries;++k)if(weights[(rows+a)*entries+k]>0){const I i=owner*entries+k;keys[i]=keys[i]||key;values[i]=values[i]||value;}
  }
}
__aicore__ inline void weights(__gm__ float* bars,__gm__ float* decay,__gm__ uint8_t* visible,
    __gm__ float* output,__gm__ int32_t* error,I owners,I queries,I entries){
  if(error[0])return;const I extent=owners*queries*entries;
  for(I o=0;o<owners;++o)for(I k=0;k<entries;++k){float g=0;
    for(I q=0;q<queries;++q){const I a=(o*queries+q)*entries+k;if(!visible[a])continue;
      for(I c=0;c<2;++c){const I j=c*extent+a;const float v=bars[j]*decay[j];if(!finite(v)){error[0]=86;return;}g+=v;}}
    if(!finite(g)){error[0]=86;return;}output[o*entries+k]=g;
  }
}
}
