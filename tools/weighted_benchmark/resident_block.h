#pragma once
#include "resident_policy.h"
namespace weighted_benchmark {
std::unique_ptr<Resident> resident_block(const Json&,Fixture,BlockSelection,at::Device,bool training,const std::string& returns);
}
