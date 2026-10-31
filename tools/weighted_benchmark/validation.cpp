#include "validation.h"
namespace weighted_benchmark {
namespace {
double rtol(at::ScalarType dtype){return dtype==at::kDouble?1e-8:dtype==at::kHalf?2e-2:1e-5;}
double atol(at::ScalarType dtype){return dtype==at::kDouble?1e-10:dtype==at::kHalf?1e-3:1e-6;}
}
Json comparison_json(const tide_bench::ComparisonReport& report) {
  Json rows=Json::array();for(const auto& row:report.differences)rows.push_back({
    {"kind",tide_bench::difference_name(row.kind)},{"path",row.path},{"message",row.message}});
  return {{"schema","tide-comparison-v1"},{"finite","passed"},
    {"numeric",report.discrete?"not_comparable":report.numeric?"failed":"passed"},
    {"discrete",report.discrete?"failed":"passed"},{"strict_passed",report.passed()},
    {"mismatch_counts",{{"numeric",report.numeric},{"discrete",report.discrete}}},{"differences",rows},
    {"differences_truncated",report.numeric+report.discrete>report.differences.size()}};
}
void compare_step(tide_bench::ComparisonReport& total,const Step& actual,const Step& expected,at::ScalarType dtype,Index window) {
  auto report=tide_bench::collect_compare(actual.result,expected.result,true,dtype,{},rtol(dtype),atol(dtype));
  auto options=at::TensorOptions().dtype(at::kDouble);
  report.merge(tide_bench::collect_tensors({{"loss",at::scalar_tensor(actual.loss,options)}},
    {{"loss",at::scalar_tensor(expected.loss,options)}},dtype,"objective",rtol(dtype),atol(dtype)));
  for(auto& row:report.differences)row.path="window["+std::to_string(window)+"]."+row.path;
  total.merge(report);
}
void compare_parameters(tide_bench::ComparisonReport& total,const std::map<std::string,Tensor>& actual,
                        const std::map<std::string,Tensor>& expected,at::ScalarType dtype,const std::string& name) {
  total.merge(tide_bench::collect_tensors(actual,expected,dtype,name,rtol(dtype),atol(dtype)));
}
void compare_training(tide_bench::ComparisonReport& total,const TrainingState& actual,const TrainingState& expected,at::ScalarType dtype) {
  compare_parameters(total,actual.gradients,expected.gradients,dtype,"final parameter gradient");
  compare_parameters(total,actual.momentum,expected.momentum,dtype,"final SGD momentum");
  compare_parameters(total,actual.masters,expected.masters,dtype,"final FP32 master");
}
}  // namespace weighted_benchmark
