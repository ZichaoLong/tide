# The official standalone package exports variables rather than an imported target.
find_package(Torch_npu REQUIRED CONFIG)
if(NOT TORCH_NPU_INCLUDE_DIRS OR NOT TORCH_NPU_LIBRARIES)
  message(FATAL_ERROR "Standalone Torch_npu package must expose headers and libraries")
endif()
if(NOT TARGET TideGraph::NpuSDK)
  add_library(TideGraph::NpuSDK INTERFACE IMPORTED)
  set_target_properties(TideGraph::NpuSDK PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TORCH_NPU_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${TORCH_NPU_LIBRARIES}")
endif()
