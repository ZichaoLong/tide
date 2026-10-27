#include "weighted_parameter_reduce.h"
extern "C" __global__ __aicore__ void tide_weighted_parameter_reduce(GM_ADDR owners,GM_ADDR refs,
    GM_ADDR tiles,GM_ADDR output,GM_ADDR connected,GM_ADDR row_errors,GM_ADDR error,int64_t count,int64_t tasks){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  tide_weighted_parameter_device::reduce((__gm__ int64_t*)owners,(__gm__ int64_t*)refs,(__gm__ int64_t*)tiles,
    (__gm__ float*)output,(__gm__ uint8_t*)connected,(__gm__ int32_t*)row_errors,(__gm__ int32_t*)error,count,tasks);
}
