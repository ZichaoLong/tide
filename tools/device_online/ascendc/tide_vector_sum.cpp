#include "kernel_operator.h"
namespace {
using I=int64_t;
constexpr uint32_t tile=256;
class VectorSum {
 public:
  __aicore__ inline void run(GM_ADDR values,GM_ADDR offsets,GM_ADDR lengths,GM_ADDR keys,GM_ADDR order,
      GM_ADDR scales,GM_ADDR content,GM_ADDR weighted,GM_ADDR error,I width) {
    AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)keys);
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
    if(((__gm__ int32_t*)error)[0])return;
    auto o=(__gm__ I*)offsets,k=(__gm__ I*)keys,ordered=(__gm__ I*)order;auto w=(__gm__ float*)scales;
    const I count=((__gm__ I*)lengths)[1],tiles=(width+tile-1)/tile;
    AscendC::GlobalTensor<float> x,h,z;
    x.SetGlobalBuffer((__gm__ float*)values);h.SetGlobalBuffer((__gm__ float*)content);z.SetGlobalBuffer((__gm__ float*)weighted);
    pipe.InitBuffer(input,1,tile*sizeof(float));pipe.InitBuffer(product,1,tile*sizeof(float));pipe.InitBuffer(sum,1,tile*sizeof(float));
    for(I task=AscendC::GetBlockIdx();task<count*tiles;task+=AscendC::GetBlockNum()) {
      const I row=task/tiles,start=(task%tiles)*tile;
      const uint32_t size=width-start<tile?width-start:tile;
      const AscendC::DataCopyExtParams copy{1,size*uint32_t(sizeof(float)),0,0,0};
      const AscendC::DataCopyPadExtParams<float> padding{true,0,uint8_t((8-size%8)%8),0};
      auto total=sum.AllocTensor<float>();
      for(I pos=o[row];pos<o[row+1];++pos) {
        const I a=ordered[pos];
        auto v=input.AllocTensor<float>();AscendC::DataCopyPad(v,x[a*width+start],copy,padding);
        input.EnQue(v);v=input.DeQue<float>();auto p=product.AllocTensor<float>();
        const float scale=w[k[a]];
        AscendC::Muls(p,v,scale,size);
        AscendC::PipeBarrier<PIPE_V>();
        if(pos==o[row])AscendC::Muls(total,v,scale,size);
        else AscendC::Add(total,total,p,size);
        AscendC::PipeBarrier<PIPE_V>();
        product.EnQue(p);p=product.DeQue<float>();
        AscendC::DataCopyPad(z[a*width+start],p,copy);
        product.FreeTensor(p);input.FreeTensor(v);
      }
      sum.EnQue(total);total=sum.DeQue<float>();AscendC::DataCopyPad(h[row*width+start],total,copy);sum.FreeTensor(total);
    }
  }
 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> product,sum;
};
}
extern "C" __global__ __aicore__ void tide_vector_sum(GM_ADDR values,GM_ADDR offsets,
    GM_ADDR lengths,GM_ADDR keys,GM_ADDR order,GM_ADDR scales,GM_ADDR content,GM_ADDR weighted,GM_ADDR error,int64_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  VectorSum op;op.run(values,offsets,lengths,keys,order,scales,content,weighted,error,width);
}
