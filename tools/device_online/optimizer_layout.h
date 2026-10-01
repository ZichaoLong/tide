#pragma once
namespace tide_device {
enum OptimizerOption {LR,WD,MOM,DAMP,B1,B1C,B2,B2C,EPS,DECAY,OPTION_COUNT};
enum OptimizerFlag {NESTEROV,AMSGRAD,MAXIMIZE,FLAG_COUNT};
enum OptimizerOwner {OFFSET,SIZE,GROUP,PAYLOAD_HALF,OWNER_FIELDS};
constexpr int optimizer_finite_error=20;
constexpr int optimizer_step_error=21;
} // namespace tide_device
