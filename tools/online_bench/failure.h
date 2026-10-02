#pragma once
#include <stdexcept>
#include <string>
#include <utility>

namespace tide_flow {
// A failed run can still have useful measurements. Keep the failure exit status
// and its complete record together instead of replacing evidence with a string.
struct RecordedFailure:std::runtime_error {
  std::string record;
  RecordedFailure(const std::string& message,std::string value)
      :std::runtime_error(message),record(std::move(value)){}
};
} // namespace tide_flow
