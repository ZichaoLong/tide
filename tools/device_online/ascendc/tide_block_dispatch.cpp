#include "block_dispatch_math.h"
extern "C" __global__ __aicore__ void tide_block_dispatch(
    GM_ADDR tasks, GM_ADDR edges, GM_ADDR valid, GM_ADDR seeds, GM_ADDR epoch,
    GM_ADDR state, GM_ADDR proposal, GM_ADDR selected, GM_ADDR emitted,
    GM_ADDR meta, GM_ADDR ledger, GM_ADDR journal, GM_ADDR branch, GM_ADDR error,
    int64_t command, int64_t slots, int64_t edge_count, int64_t groups,
    int64_t rows, int64_t journal_capacity) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (AscendC::GetBlockIdx() != 0) return;
  using namespace tide_block_dispatch_device;
  AscendC::GlobalTensor<int32_t> cache; cache.SetGlobalBuffer((__gm__ int32_t*)error);
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto t=(__gm__ I*)tasks, e=(__gm__ I*)edges, generation=(__gm__ I*)epoch;
  auto s=(__gm__ I*)state, p=(__gm__ I*)proposal, ids=(__gm__ I*)selected;
  auto m=(__gm__ I*)meta, l=(__gm__ I*)ledger, j=(__gm__ I*)journal;
  auto b=(__gm__ int32_t*)branch, err=(__gm__ int32_t*)error;
  if(command==0) reset(t,e,(__gm__ uint8_t*)valid,(__gm__ uint8_t*)seeds,generation,s,l,err,slots,edge_count,groups);
  else if(command==1) choose(t,e,generation,s,p,ids,m,l,b,err,slots,edge_count,groups,rows,journal_capacity);
  else if(command==2) commit(t,e,generation,s,p,ids,(__gm__ uint8_t*)emitted,m,l,j,b,err,slots,edge_count,rows,journal_capacity);
  else if(!err[0]) err[0]=Invalid;
  AscendC::DataCacheCleanAndInvalid<int32_t,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
