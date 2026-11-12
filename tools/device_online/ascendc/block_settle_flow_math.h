#pragma once
#include "weighted_allocation_math.h"
namespace tide_block_settle_flow_device {
using namespace tide_weighted_device;
__aicore__ inline void fibers(__gm__ I* atoms,__gm__ I* counts,__gm__ I* bank,
    __gm__ I* positions,__gm__ float* external,__gm__ uint8_t* external_live,
    __gm__ float* messages,__gm__ uint8_t* message_live,__gm__ float* payload,
    __gm__ I* lengths,__gm__ I* source_owners,__gm__ I* node_owners,__gm__ I* times,
    __gm__ I* mapping,__gm__ int32_t* error,
    I batch,I length,I nodes,I capacity,I width,I input_ports,I all_nodes,I rank,I stride){
  if(error[0])return;
  for(I b=0;b<batch;++b)for(I l=0;l<length;++l){const I frame=b*length+l,position=positions[frame];
    if(position<0||stride<1||rank<0||position>(I(0x7fffffffffffffffLL)-rank)/stride||
       (l&&position<=positions[frame-1])){error[0]=79;return;}
    for(I n=0;n<nodes;++n){const I row=frame*nodes+n;lengths[row]=0;node_owners[row]=bank[n];times[row]=position*stride+rank;
      if(counts[n]<0||counts[n]>capacity||bank[n]<0){error[0]=79;return;}
      for(I a=0;a<capacity;++a){const I dst=row*capacity+a;source_owners[dst]=-1;mapping[3*dst]=mapping[3*dst+1]=mapping[3*dst+2]=-1;
        for(I k=0;k<width;++k)payload[dst*width+k]=0;}
      for(I a=0;a<counts[n];++a){const I slot=(n*capacity+a)*3,kind=atoms[slot],id=atoms[slot+1],owner=atoms[slot+2];
        if((kind!=0&&kind!=1)||id<0||id>=(kind?all_nodes:input_ports)||owner<0){error[0]=79;return;}
        const I source=frame*(kind?all_nodes:input_ports)+id;
        if(!(kind?message_live[source]:external_live[source]))continue;
        const I dst=row*capacity+lengths[row]++;source_owners[dst]=owner;mapping[3*dst]=kind;mapping[3*dst+1]=source;mapping[3*dst+2]=a;
        for(I k=0;k<width;++k)payload[dst*width+k]=kind?messages[source*width+k]:external[source*width+k];
      }
    }
  }
}
__aicore__ inline void emit(__gm__ uint8_t* active,__gm__ uint8_t* emitted,__gm__ int32_t* error,I rows){
  if(error[0])return;emitted[0]=0;for(I r=0;r<rows;++r)if(active[r]){emitted[0]=1;break;}
}
__aicore__ inline void publish(__gm__ I* nodes,__gm__ float* strength,__gm__ float* content,
    __gm__ uint8_t* present,__gm__ I* times,__gm__ float* output,__gm__ uint8_t* live,
    __gm__ I* clock,__gm__ int32_t* error,I frames,I count,I all_nodes,I d){
  if(error[0])return;
  for(I f=0;f<frames;++f)for(I v=0;v<count;++v){const I src=f*count+v,dst=f*all_nodes+nodes[v];
    if(nodes[v]<0||nodes[v]>=all_nodes){error[0]=79;return;}
    live[dst]=present[src];clock[dst]=times[src];output[dst*(d+1)]=strength[src];
    for(I k=0;k<d;++k)output[dst*(d+1)+1+k]=content[src*d+k];
  }
}
__aicore__ inline void ports(__gm__ I* source,__gm__ I* delay,__gm__ float* nodes,
    __gm__ uint8_t* present,__gm__ I* times,__gm__ float* output,__gm__ uint8_t* live,
    __gm__ I* clock,__gm__ int32_t* error,I frames,I count,I all_nodes,I width){
  if(error[0])return;
  for(I f=0;f<frames;++f)for(I p=0;p<count;++p){const I node=source[p],src=f*all_nodes+node,dst=f*count+p;
    if(node<0||node>=all_nodes||delay[p]<0){error[0]=79;return;}
    live[dst]=present[src];clock[dst]=0;
    for(I k=0;k<width;++k)output[dst*width+k]=present[src]?nodes[src*width+k]:0;
    if(present[src]){
      if(times[src]<0||times[src]>I(0x7fffffffffffffffLL)-delay[p]){error[0]=79;return;}
      clock[dst]=times[src]+delay[p];
    }
  }
}
}
