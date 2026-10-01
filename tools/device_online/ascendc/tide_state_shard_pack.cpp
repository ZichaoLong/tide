#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_state_shard_pack(GM_ADDR fibers,GM_ADDR offsets,GM_ADDR counts,GM_ADDR atoms,GM_ADDR mapping,
    GM_ADDR local_fibers,GM_ADDR local_offsets,GM_ADDR local_counts,GM_ADDR local_atoms,GM_ADDR local_valid,
    GM_ADDR fiber_rows,GM_ADDR atom_rows,GM_ADDR destinations,GM_ADDR branch,GM_ADDR error,
    int64_t rows,int64_t capacity,int64_t nodes,int64_t local_nodes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)fibers);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto f=(__gm__ I*)fibers,o=(__gm__ I*)offsets,c=(__gm__ I*)counts,a=(__gm__ I*)atoms,map=(__gm__ I*)mapping;
  auto lf=(__gm__ I*)local_fibers,lo=(__gm__ I*)local_offsets,lc=(__gm__ I*)local_counts,la=(__gm__ I*)local_atoms;
  auto fr=(__gm__ I*)fiber_rows,ar=(__gm__ I*)atom_rows,dest=(__gm__ I*)destinations;
  auto valid=(__gm__ uint8_t*)local_valid;auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;
  go[0]=0;lc[0]=lc[1]=lc[2]=0;
  for(I i=0;i<capacity;++i){fr[i]=ar[i]=rows;dest[i]=rows+i;valid[i]=0;lo[i]=0;
    for(I j=0;j<4;++j)lf[i*4+j]=0;for(I j=0;j<6;++j)la[i*6+j]=0;}
  lo[capacity]=0;
  const I nf=c[1],na=c[0];I used_f=0,used_a=0;
  if(status[0]==0&&(nf<0||nf>rows||na<0||na>rows||o[0]!=0))status[0]=2;
  for(I i=0;i<nf&&status[0]==0;++i) {
    const I n=f[i*4+1],begin=o[i],end=o[i+1];
    if(n<0||n>=nodes||begin<0||end<=begin||end>na){status[0]=2;break;}
    const I local=map[n];if(local==-1)continue;
    if(local<0||local>=local_nodes){status[0]=2;break;}
    if(used_f>=capacity||end-begin>capacity-used_a){status[0]=1;break;}
    for(I j=0;j<4;++j)lf[used_f*4+j]=f[i*4+j];lf[used_f*4+1]=local;
    fr[used_f]=dest[used_f]=i;lo[used_f]=used_a;
    for(I j=begin;j<end;++j) {
      if(a[j*6]!=f[i*4]||a[j*6+1]!=n||a[j*6+2]!=f[i*4+2]){status[0]=2;break;}
      for(I k=0;k<6;++k)la[used_a*6+k]=a[j*6+k];la[used_a*6+1]=local;
      ar[used_a]=j;valid[used_a]=1;++used_a;
    }
    lo[++used_f]=used_a;
  }
  if(status[0]==0&&o[nf]!=na)status[0]=2;
  if(status[0]==0){lc[0]=used_a;lc[1]=used_f;go[0]=used_f>0;
    for(I i=used_f+1;i<=capacity;++i)lo[i]=used_a;
  }else {
    // Failed proposals expose no partial fibers or dangerous gather indices.
    for(I i=0;i<capacity;++i){fr[i]=ar[i]=rows;dest[i]=rows+i;valid[i]=0;}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
