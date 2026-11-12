#include "kernel_operator.h"
extern "C" __global__ __aicore__ void tide_block_flow_boundary(GM_ADDR nodes,GM_ADDR present,GM_ADDR times,
    GM_ADDR node_seen,GM_ADDR region_times,GM_ADDR input_counts,GM_ADDR input_times,GM_ADDR error,
    int64_t command,int64_t batch,int64_t length,int64_t count,int64_t all_nodes,int64_t region,int64_t regions,int64_t stride){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;using I=int64_t;
  AscendC::GlobalTensor<int32_t> cache;cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto e=(__gm__ int32_t*)error;auto ids=(__gm__ I*)nodes;auto live=(__gm__ uint8_t*)present;auto clock=(__gm__ I*)times;
  auto seen=(__gm__ uint8_t*)node_seen;auto history=(__gm__ I*)region_times;auto occurrences=(__gm__ I*)input_counts;auto input_clock=(__gm__ I*)input_times;
  if(!e[0])for(I b=0;b<batch&&!e[0];++b)for(I l=0;l<length&&!e[0];++l)for(I n=0;n<count;++n){
    const I row=(b*length+l)*count+n;if(!live[row])continue;
    if(command==0){const I target=b*count+n,position=clock[b*length+l],rank=ids[n];
      if(position<0||rank<0||stride<1||position>(I(0x7fffffffffffffffLL)-rank)/stride||occurrences[target]<-1||occurrences[target]==I(0x7fffffffffffffffLL)){
        e[0]=81;break;}
      const I time=position*stride+rank;if(time<=input_clock[target]){e[0]=81;break;}
      ++occurrences[target];input_clock[target]=time;
    }else if(command==1){const I node=ids[n],time=clock[row];
      if(node<0||node>=all_nodes||region<0||region>=regions||time<0||time<history[b*regions+region]){e[0]=81;break;}
      seen[b*all_nodes+node]=1;history[b*regions+region]=time;
    }else{e[0]=81;break;}
  }
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
