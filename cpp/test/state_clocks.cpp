#include "tide/clocked_kernel.h"
#include "tide/lazy_add.h"
#include <stdexcept>

void check_state_clocks(const at::TensorOptions& options) {
  using namespace tide;
  StateClock clock{4, 3, 1};
  NodeWeights w{at::zeros({2}, options), at::zeros({2, 2}, options), at::zeros({2}, options), at::ones({2}, options)};
  w.extra["add_retention"] = at::full({}, .5, options);
  w.kernel = with_state_clock(make_add_repeat_kernel(), clock);
  if (with_state_clock(w.kernel, clock) != w.kernel || kernel_clock(w.kernel) != clock)
    throw std::runtime_error("clock wrapper was duplicated");
  bool rejected = false;
  try { with_state_clock(w.kernel, StateClock{}); } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("incompatible shared clock accepted");
  State old{at::ones({2}, options)};
  auto h = at::full({2}, .2, options);
  auto first = w.kernel->step(w, old, {h, {}, {}}, 3);
  auto check = [&](const Tensor& value, double expected) {
    const auto fp64 = value.scalar_type() == at::kDouble;
    if (!at::allclose(value, at::full_like(value, expected), fp64 ? 1e-8 : 1e-5, fp64 ? 1e-10 : 1e-6))
      throw std::runtime_error("clocked Add value mismatch");
  };
  check(first.value, .7); check(decode_add_repeat(w, first, 8), .35);
  auto bare = w; bare.kernel.reset();
  check(decode_add_repeat(bare, first, 8, clock), .35);
  auto reset = w.kernel->reset(first);
  if (reset.last_time != 3 || reset.observations != 1) throw std::runtime_error("clock reset lost metadata");
  check(reset.value, 0);
  auto batch = w.kernel->batch(w, {first, first}, at::stack({h, h}), {7, 11}, {{h, {}, {}}, {h, {}, {}}});
  check(batch[0].value, .55); check(batch[1].value, .375);
  if (batch[0].last_time != 7 || batch[1].last_time != 11) throw std::runtime_error("clock batch metadata mismatch");
  PackedSequence sequence{at::stack({h, h, h}), {0, 2, 3}, {{0, 0}, {1, 0}}, {3, 11, 7},
                          {{h, {}, {}}, {h, {}, {}}, {h, {}, {}}}};
  auto packed = w.kernel->packed_sequence(w, {old, old}, sequence);
  if (packed.calls != 2 || packed.scalar_steps != 3 || packed.states.size() != 3
      || packed.states[1].last_time != 11 || packed.states[2].last_time != 7)
    throw std::runtime_error("clock wrapper changed packed capabilities or metadata");
  check(packed.states[0].value, .7); check(packed.states[1].value, .375); check(packed.states[2].value, .45);
}
