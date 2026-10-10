execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_kernels.py"
  --source "${CMAKE_CURRENT_SOURCE_DIR}/../ascendc" --output "${CMAKE_CURRENT_BINARY_DIR}/../launch"
  --backend cuda COMMAND_ERROR_IS_FATAL ANY)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../ascendc")
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
include("${CMAKE_CURRENT_BINARY_DIR}/../launch/targets.cmake")
