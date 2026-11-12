#pragma once
#include "weighted_allocation_math.h"
namespace tide_block_flow_reverse_device {
using namespace tide_weighted_device;
__aicore__ inline void ports(__gm__ I* sources,__gm__ uint8_t* present,
    __gm__ float* mass,__gm__ uint8_t* mass_roots,__gm__ float* content,__gm__ uint8_t* content_roots,
    __gm__ float* output,__gm__ uint8_t* connected,__gm__ int32_t* error,I frames,I ports,I nodes,I width){
  if(error[0])return;
  for(I f=0;f<frames;++f)for(I p=0;p<ports;++p){const I row=f*ports+p,node=sources[p],target=f*nodes+node;
    if(node<0||node>=nodes){error[0]=80;return;}if(!present[row])continue;
    if(mass_roots[row]){const float x=mass[row],g=output[target*width]+x;if(!finite(x)||!finite(g)){error[0]=80;return;}
      output[target*width]=g;connected[2*target]=1;}
    if(content_roots[row]){connected[2*target+1]=1;for(I d=1;d<width;++d){
      const float x=content[row*(width-1)+d-1],g=output[target*width+d]+x;
      if(!finite(x)||!finite(g)){error[0]=80;return;}output[target*width+d]=g;}}
  }
}
__aicore__ inline void gather(__gm__ I* nodes,__gm__ uint8_t* present,__gm__ float* incoming,__gm__ uint8_t* flags,
    __gm__ float* direct_mass,__gm__ uint8_t* direct_mass_roots,__gm__ float* direct_content,__gm__ uint8_t* direct_content_roots,
    __gm__ float* mass,__gm__ uint8_t* mass_roots,__gm__ float* content,__gm__ uint8_t* content_roots,
    __gm__ int32_t* error,I frames,I count,I all_nodes,I width){
  if(error[0])return;
  for(I f=0;f<frames;++f)for(I n=0;n<count;++n){const I row=f*count+n,source=f*all_nodes+nodes[n];
    if(nodes[n]<0||nodes[n]>=all_nodes){error[0]=80;return;}
    mass[row]=0;mass_roots[row]=content_roots[row]=0;for(I d=1;d<width;++d)content[row*(width-1)+d-1]=0;
    if(!present[row])continue;
    for(I role=0;role<2;++role){const bool a=flags[2*source+role],b=role?direct_content_roots[row]:direct_mass_roots[row];
      if(role)content_roots[row]=a||b;else mass_roots[row]=a||b;
      for(I d=role?1:0;d<(role?width:1);++d){const float x=a?incoming[source*width+d]:0;
        const float y=b?(role?direct_content[row*(width-1)+d-1]:direct_mass[row]):0;
        if(!finite(x)||!finite(y)||!finite(x+y)){error[0]=80;return;}
        if(role)content[row*(width-1)+d-1]=x+y;else mass[row]=x+y;
      }
    }
  }
}
__aicore__ inline void fibers(__gm__ I* mapping,__gm__ I* lengths,__gm__ float* gradient,
    __gm__ uint8_t* roots,__gm__ float* external,__gm__ uint8_t* external_roots,
    __gm__ float* messages,__gm__ uint8_t* message_roots,__gm__ int32_t* error,
    I rows,I capacity,I width,I external_rows,I message_rows){
  if(error[0])return;
  for(I r=0;r<rows;++r){if(lengths[r]<0||lengths[r]>capacity){error[0]=80;return;}
    for(I a=0;a<lengths[r];++a){const I atom=r*capacity+a;if(!roots[atom])continue;
      const I kind=mapping[3*atom],target=mapping[3*atom+1];
      if((kind!=0&&kind!=1)||target<0||target>=(kind?message_rows:external_rows)){error[0]=80;return;}
      if(kind)message_roots[2*target]=message_roots[2*target+1]=1;else external_roots[target]=1;
      for(I d=0;d<width;++d){auto out=kind?messages:external;const float x=gradient[atom*width+d],g=out[target*width+d]+x;
        if(!finite(x)||!finite(g)){error[0]=80;return;}out[target*width+d]=g;}
    }
  }
}
__aicore__ inline void choose(__gm__ I* journal,__gm__ I* ledger,__gm__ I* cursor,
    __gm__ int32_t* branch,__gm__ int32_t* error,I regions,I command){
  branch[0]=0;if(error[0])return;
  if(ledger[2]!=regions||ledger[0]<1){error[0]=80;return;}
  if(command==0){cursor[0]=regions-1;return;}
  if(cursor[0]<0)return;if(cursor[0]>=regions){error[0]=80;return;}
  const I slot=journal[3*cursor[0]];if(slot<0||slot>=regions){error[0]=80;return;}
  branch[0]=int32_t(slot+1);--cursor[0];
}
}
