#pragma once
#include "region.h"
#include <filesystem>
namespace tide_native {
Tensors read_bundle(const std::filesystem::path& root);
void write_bundle(const std::filesystem::path& root, const Tensors& values);
void atomic_text(const std::filesystem::path& path, const std::string& text);
void publish_staged(const std::filesystem::path& stage, const std::filesystem::path& path);
std::string json_string(const std::string& text);
}  // namespace tide_native
