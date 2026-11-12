#pragma once
#include "kernel_operator.h"
namespace tide_block_selection_device {
using I=int64_t;
__aicore__ inline void plan(__gm__ uint8_t* present,__gm__ I* times,__gm__ I* source_owners,
    __gm__ I* offsets,__gm__ I* members,__gm__ I* owners,__gm__ uint8_t* saved_present,
    __gm__ I* saved_times,__gm__ I* order,__gm__ I* inverse,__gm__ I* ids,__gm__ I* region_times,
    __gm__ I* lengths,__gm__ I* node_region_row,__gm__ int32_t* error,I frames,I nodes,I regions,I width){
  if(error[0])return;
  if(offsets[0]!=0||offsets[regions]!=nodes){error[0]=70;return;}
  for(I r=0;r<regions;++r)if(offsets[r]<0||offsets[r]>=offsets[r+1]||offsets[r+1]>nodes||offsets[r+1]-offsets[r]>width){error[0]=70;return;}
  for(I i=0;i<nodes;++i){if(members[i]<0||members[i]>=nodes){error[0]=70;return;}
    for(I j=0;j<i;++j)if(members[j]==members[i]){error[0]=70;return;}}
  for(I i=0;i<frames*nodes;++i){
    if(present[i]&&(times[i]<0||source_owners[i]<0)){error[0]=70;return;}
    owners[i]=present[i]?source_owners[i]:-1;saved_present[i]=present[i];saved_times[i]=present[i]?times[i]:0;
    inverse[i]=frames*regions*width;
  }
  for(I f=0;f<frames;++f)for(I r=0;r<regions;++r){
    const I row=f*regions+r;I count=0,time=-1;
    for(I j=0;j<width;++j){order[row*width+j]=frames*nodes;ids[row*width+j]=0;}
    for(I k=offsets[r];k<offsets[r+1];++k){
      const I v=members[k],input=f*nodes+v;node_region_row[input]=row;
      if(!present[input])continue;
      if(time>=0&&time!=times[input]){error[0]=70;return;}time=times[input];
      order[row*width+count]=input;ids[row*width+count]=v;inverse[input]=row*width+count;++count;
    }
    lengths[row]=count;region_times[row]=count?time:0;
  }
}
__aicore__ inline void roots(__gm__ I* order,__gm__ float* bars,__gm__ uint8_t* connected,
    __gm__ float* packed,__gm__ uint8_t* row_roots,__gm__ int32_t* error,I rows,I width,I input_rows){
  if(error[0])return;
  for(I r=0;r<rows;++r){row_roots[r]=0;
    for(I j=0;j<width;++j){const I slot=r*width+j,source=order[slot];packed[slot]=0;
      if(source<0||source>input_rows){error[0]=70;return;}
      if(source<input_rows&&connected[source]){const float x=bars[source];
        if(x!=x||x>3.4028234e38f||x< -3.4028234e38f){error[0]=74;return;}
        packed[slot]=x;row_roots[r]=1;
      }
    }
  }
}
__aicore__ inline void flags(__gm__ uint8_t* present,__gm__ I* mapping,
    __gm__ uint8_t* region_roots,__gm__ uint8_t* output,__gm__ int32_t* error,I rows,I regions){
  if(error[0])return;
  for(I i=0;i<rows;++i){const I source=mapping[i];if(source<0||source>=regions){error[0]=70;return;}
    for(I field=0;field<6;++field)output[6*i+field]=present[i]&&region_roots[6*source+field];}
}
__aicore__ inline void merge(__gm__ uint8_t* a,__gm__ uint8_t* b,__gm__ uint8_t* out,
    __gm__ int32_t* error,I rows){if(error[0])return;for(I i=0;i<rows;++i)out[i]=a[i]||b[i];}
__aicore__ inline void gate(__gm__ I* tasks,__gm__ I* selected,__gm__ I* metadata,
    __gm__ uint8_t* active,__gm__ uint8_t* emitted,__gm__ int32_t* error,I slots,I capacity,I frames,I nodes){
  if(error[0])return;const I count=metadata[0];if(count<0||count>capacity||metadata[1]!=2||frames<1||nodes<1){error[0]=70;return;}
  for(I i=0;i<capacity;++i)emitted[i]=0;
  for(I i=0;i<count;++i){const I task=selected[i];if(task<0||task>=slots){error[0]=70;return;}
    const I node=tasks[5*task+2];if(tasks[5*task]!=2||node<0||node>=nodes){error[0]=70;return;}
    for(I f=0;f<frames;++f)if(active[f*nodes+node])emitted[i]=1;
  }
}
}
