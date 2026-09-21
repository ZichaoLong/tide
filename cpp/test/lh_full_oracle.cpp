// Actual ModuleUtils activation/norm/signaling factories from the LH snapshot.
#include "ModuleUtils.h"
#include "portable_torch/runtime.hpp"
#include "tide/full.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <stdexcept>

namespace {
using tide::Index;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(const at::Tensor& a, const at::Tensor& b, const char* message) {
  const bool fp64 = a.scalar_type() == at::kDouble;
  require(a.sizes() == b.sizes() && at::allclose(a, b, fp64 ? 1e-8 : 1e-5, fp64 ? 1e-10 : 1e-6), message);
}
void check(const at::TensorOptions& opts, const std::string& act, const std::string& norm,
           Index width, Index slots, bool bias) {
  const nlohmann::json cfg{{"actfn", act}, {"norm", norm}, {"signalling", "linear"}, {"bias", bias}};
  auto activation = act == "identity" ? torch::nn::AnyModule(std::make_shared<torch::nn::IdentityImpl>())
                                      : ConfigurableModule::gen_actfn_anymodule(cfg);
  auto normalization = ConfigurableModule::gen_norm_anymodule(width, cfg);
  normalization.ptr()->to(opts.dtype().toScalarType());
  ConfigurableModule::SubModules signaling;
  ConfigurableModule::gen_signalling_modulelist(std::vector<Index>{slots}, width, width, cfg, signaling);
  signaling.ml->to(opts.dtype().toScalarType());
  const auto params = normalization.ptr()->named_parameters();
  if (params.contains("weight")) params["weight"].copy_(.7+at::arange(width, opts)/10);
  if (params.contains("bias")) params["bias"].copy_(at::arange(width, opts)/20-.1);
  const auto signals = signaling.anymodules[0].ptr()->named_parameters();
  signals["weight"].copy_(at::sin(at::arange(slots*width*width, opts).reshape({slots*width, width})*.17));
  if (bias) signals["bias"].copy_(at::arange(slots*width, opts)/13-.2);
  tide::Node node{0}; node.full = "lh-"+act+"-"+norm+"-v1"; node.emission = "slot_affine";
  tide::NodeWeights w{at::zeros({width}, opts), at::zeros({width, width}, opts),
                      at::zeros({width}, opts), at::zeros({width}, opts)};
  w.full_kind = node.full;
  if (norm != "identity") w.extra["lh_norm_weight"] = params["weight"].clone();
  if (norm == "layer") w.extra["lh_norm_bias"] = params["bias"].clone();
  for (Index slot = 0; slot < slots; ++slot) {
    w.extra["emit_w_"+std::to_string(slot)] = signals["weight"].slice(0, slot*width, (slot+1)*width).t().clone();
    w.extra["emit_b_"+std::to_string(slot)] = bias ? signals["bias"].slice(0, slot*width, (slot+1)*width).clone()
                                                               : at::zeros({width}, opts);
  }
  const Index rows = 7;
  auto comparisons = at::sin(at::arange(rows*width, opts).reshape({rows, width})*.31)-.2;
  comparisons[0].zero_(); comparisons[1].fill_(-2); comparisons[2].fill_(3); comparisons[3] *= 100;
  auto expected = normalization.forward<at::Tensor>(activation.forward<at::Tensor>(comparisons));
  auto outputs = signaling.anymodules[0].forward<at::Tensor>(expected).reshape({rows, slots, width});
  auto kernel = tide::make_full_kernel(node); kernel->validate_weights(w, slots);
  std::vector<tide::State> states;
  for (Index row = 0; row < rows; ++row) states.push_back({comparisons[row]});
  std::vector<tide::FullInput> inputs;
  for (Index row = 0; row < rows; ++row)
    inputs.push_back({&states[row], row*3, {at::full({width}, .123+row/10., opts)}, at::scalar_tensor(.3, opts)});
  tide::Options options; auto packed = kernel->batch(w, inputs, slots, options);
  for (Index row = 0; row < rows; ++row) {
    auto scalar = kernel->step(w, inputs[row], slots, options);
    for (const auto* result : {&scalar, &packed[row]}) {
      close(result->value, expected[row], "LH activation/norm Full value mismatch");
      require(result->emitted.size() == static_cast<size_t>(slots), "LH signaling slot count mismatch");
      for (Index slot = 0; slot < slots; ++slot) {
        require(result->emitted[slot].slot == slot, "LH signaling slot ordering mismatch");
        close(result->emitted[slot].value, outputs[row][slot], "LH per-edge signaling mismatch");
      }
    }
  }
}
}  // namespace
int main(int argc, char** argv) {
  try {
    const auto args = portable_torch::parse_cli(argc, argv);
    if (args.help) { portable_torch::print_usage(std::cout, argv[0]); return 0; }
    const auto device = portable_torch::resolve_device(args);
    if (!device.is_cpu() || (args.dtype != at::kDouble && args.dtype != at::kFloat))
      throw std::invalid_argument("LH Full oracle requires CPU FP64/FP32");
    at::set_num_threads(1); at::NoGradGuard guard;
    const auto opts = at::TensorOptions().dtype(args.dtype).device(device);
    for (const auto& act : {"relu", "silu", "identity"}) for (const auto& norm : {"identity", "rms", "layer"})
      for (Index width : {1, 3, 5}) for (Index slots : {1, 3}) for (bool bias : {false, true})
        check(opts, act, norm, width, slots, bias);
    std::cout << "original-LH-full: passed; 108 configurations, 756 rows; 72 activation + 36 norm-only; per-edge signaling; scalar/packed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
