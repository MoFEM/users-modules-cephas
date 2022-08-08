include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

tutorials_build_and_install(
  magnetostatic ${CMAKE_CURRENT_SOURCE_DIR}/magnetostatic.cpp)

add_test(magnetostatic
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
  ${CMAKE_CURRENT_BINARY_DIR}/magnetostatic
  -my_file ${CMAKE_CURRENT_SOURCE_DIR}/magnetic_coil_2parts.h5m -my_order 1 
  -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type mumps 
  -ksp_monitor -my_max_post_proc_ref_level 0 -regression_test 1)

set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

tutorials_copy_and_install("*.jou" "${permissions_default}")
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("README.md" "${permissions_default}")


