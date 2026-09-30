set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Active CANN toolkit")
if(NOT SOC_VERSION)
  message(FATAL_ERROR "Pass -DSOC_VERSION=... for the actual target; no implicit architecture")
endif()
find_file(TIDE_ASCENDC_CMAKE ascendc.cmake PATHS
  "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/ascendc_kernel_cmake"
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake" NO_DEFAULT_PATH REQUIRED)
include("${TIDE_ASCENDC_CMAKE}")
# The CANN generator embeds this target name in C++ symbols.
ascendc_library(tide_closure_kernel STATIC tide_closure.cpp)
ascendc_library(tide_queue_kernel STATIC tide_queue_propose.cpp)
ascendc_library(tide_broadcast_kernel STATIC tide_broadcast_route.cpp)
ascendc_library(tide_ready_kernel STATIC tide_ready_pack.cpp)
ascendc_library(tide_selector_kernel STATIC tide_frame_select.cpp)
