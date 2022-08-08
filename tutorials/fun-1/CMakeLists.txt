include_directories(${CURREENT_SOURCE_DIR})
include_directories(${CURREENT_SOURCE_DIR}/src)

tutorials_build_and_install(
  integration
  ${CMAKE_CURRENT_SOURCE_DIR}/integration.cpp)

set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

tutorials_copy_and_install("*.petsc" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")

add_test(
  integration
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/integration
  -file_name ${CMAKE_CURRENT_BINARY_DIR}/cube.h5m -test -log_no_color)
