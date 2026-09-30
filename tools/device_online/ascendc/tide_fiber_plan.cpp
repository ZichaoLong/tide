#include "kernel_operator.h"
#include "state_clock.h"
namespace {using I=int64_t;}
// events: [ready fiber, cache owner, parameter owner, old K, proposed K,
//          first query row, exact decay ticks]. tokens: [source row,event,K row,param].
extern "C" __global__ __aicore__ void tide_fiber_plan(GM_ADDR fibers,GM_ADDR offsets,GM_ADDR lengths,
    GM_ADDR atoms,GM_ADDR sources,GM_ADDR mapping,GM_ADDR cache_lengths,GM_ADDR clocks,GM_ADDR policy,GM_ADDR config,
    GM_ADDR events,GM_ADDR tokens,GM_ADDR counts,GM_ADDR error,int64_t nodes,int64_t parameters,
    int64_t capacity,int64_t rows,int64_t inputs,int64_t max_ticks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto f=(__gm__ I*)fibers,o=(__gm__ I*)offsets,c=(__gm__ I*)counts;
  auto e=(__gm__ I*)events,t=(__gm__ I*)tokens,a=(__gm__ I*)atoms;
  c[0]=c[1]=0;
  I observations=0;
  const I nf=((__gm__ I*)lengths)[1];
  for(I i=0;i<nf&&status[0]==0;++i) {
    const I n=f[i*4+1],b=f[i*4],parameter=((__gm__ I*)mapping)[n];if(parameter<0)continue;
    const I owner=b*parameters+parameter,added=o[i+1]-o[i];
    // Ready packing orders (sample,node,time), so each owner's prefix is
    // contiguous even when other memory profiles are omitted from this table.
    const bool next=c[1]>0&&e[(c[1]-1)*7+1]==owner;
    const I old=next?e[(c[1]-1)*7+4]:((__gm__ I*)cache_lengths)[owner];
    if(old<0||old>capacity||added<1||added>capacity-old){status[0]=11;break;}
    auto cfg=(__gm__ I*)config;
    if(next&&(!cfg[parameter*2]||cfg[parameter*2+1])){status[0]=2;break;}
    const I key=b*nodes+n,last=next?f[e[(c[1]-1)*7]*4+2]:((__gm__ I*)clocks)[key*2],time=f[i*4+2];
    if(!next)observations=((__gm__ I*)clocks)[key*2+1];
    if(last< -1||last>=time||observations<0){status[0]=2;break;}
    if(observations==I(0x7fffffffffffffff)){status[0]=5;break;}
    I local,local_old;
    if(!tide_device::local_time(time,(__gm__ I*)policy,n,local)
        ||!tide_device::local_time(last,(__gm__ I*)policy,n,local_old)){status[0]=9;break;}
    const uint64_t ticks=(uint64_t(local)+1)-uint64_t(local_old+1);
    if(old&&ticks>uint64_t(max_ticks)){status[0]=8;break;}
    if(c[0]+added>rows||c[1]>=rows){status[0]=2;break;}
    const I event=c[1]++,start=c[0];auto row=e+event*7;
    row[0]=i;row[1]=owner;row[2]=parameter;row[3]=old;row[4]=old+added;row[5]=start;row[6]=old?I(ticks):0;
    // Stable local-slot order. Physical atoms and Aggregate diagnostics keep
    // their independent canonical order; only the attention query view changes.
    for(I j=o[i];j<o[i+1];++j) {
      const I source=a[j*6+3]==0?a[j*6+4]:inputs+a[j*6+4];
      const I slot=((__gm__ I*)sources)[source*2+1];I pos=c[0];
      while(pos>start) {
        const I before=t[(pos-1)*4],s=a[before*6+3]==0?a[before*6+4]:inputs+a[before*6+4];
        if(((__gm__ I*)sources)[s*2+1]<=slot)break;
        t[pos*4]=before;--pos;
      }
      t[pos*4]=j;++c[0];
    }
    for(I j=start;j<c[0];++j){t[j*4+1]=event;t[j*4+2]=old+j-start;t[j*4+3]=parameter;}
    ++observations;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
