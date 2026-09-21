#include "tide/clock.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tide {
void StateClock::validate() const {
  if (period <= 0 || first < 0 || first >= period || count <= 0 || count > period-first)
    throw std::invalid_argument("invalid periodic state clock");
}
std::int64_t StateClock::to_local(std::int64_t time) const {
  validate();
  if (time < -1) throw std::invalid_argument("invalid state clock coordinate");
  if (time == -1) return -1;
  const auto phase = time % period;
  if (phase < first || phase-first >= count) throw std::invalid_argument("event is outside the state clock phases");
  // Result <= time, so neither product nor sum can overflow.
  return (time/period)*count + (phase-first);
}
std::int64_t StateClock::to_global(std::int64_t time) const {
  validate();
  if (time < -1) throw std::invalid_argument("invalid state clock coordinate");
  if (time == -1) return -1;
  const auto cycle = time/count, phase = first + time%count;
  if (cycle > (std::numeric_limits<std::int64_t>::max()-phase)/period)
    throw std::invalid_argument("state clock inverse overflow");
  return cycle*period + phase;
}
std::int64_t StateClock::cut(std::int64_t cut) const {
  validate();
  if (cut < 0) throw std::invalid_argument("invalid state clock coordinate");
  return (cut/period)*count + std::clamp(cut%period-first, std::int64_t{0}, count);
}
}  // namespace tide
