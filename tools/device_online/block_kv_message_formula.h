#pragma once
#include "block_kv_messages.h"
#include "block_selection_formula.h"

namespace tide::device_online::block_message_check {
using Fields=std::map<std::string,at::Tensor>;
struct Fixture {
  block_selection_check::Fixture selection;
  WeightedRawLayout proposal_layout;
  std::vector<State> proposal_states;
  WeightedRawState proposal;
  at::Tensor full_owners;
  WeightedKVFullParameters full;
  ParameterRegistry registry;
  BlockKVParameterRoles roles;
  Fixture();
};
Fields parameter_roles(const BlockKVMessageParameters&);
BlockKVMessageInput device_input(const Fixture&,at::Device);
BlockKVMessageParameters device_parameters(const Fixture&,at::Device);
struct Formula {
  block_selection_check::Formula selection;
  std::vector<at::Tensor> leaves;
  std::vector<block_selection_check::FormulaLabel> labels;
  std::vector<at::Tensor> full,message_strength,message_content,projection;
  at::Tensor presence;
};
Formula formula(const Fixture&);
std::string parameter_name(const block_selection_check::FormulaLabel&);
}
