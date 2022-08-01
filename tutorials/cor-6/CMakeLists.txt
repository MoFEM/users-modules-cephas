#
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

tutorials_build_and_install(simple_elasticity simple_elasticity.cpp)

add_test(simple_elasticity
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
  ${CMAKE_CURRENT_BINARY_DIR}/simple_elasticity
  -file_name ${CMAKE_CURRENT_SOURCE_DIR}/simple_elasticity.h5m -order 3
  -ksp_type gmres -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor
  -test 1)

 cm_export_file("simple_elasticity.h5m" export_files_simple_elasticity)

set(permissions_default 
 OWNER_WRITE 
 OWNER_READ
 GROUP_READ)

set(permissions_execute 
 OWNER_WRITE 
 OWNER_READ
 GROUP_READ
 OWNER_EXECUTE 
 GROUP_EXECUTE)

 tutorials_copy_and_install("*.msh" "${permissions_default}")
 tutorials_copy_and_install("*.cfg" "${permissions_default}")
 # tutorials_copy_and_install("*.jou" "${permissions_default}")
 tutorials_copy_and_install("*.cub" "${permissions_default}")
 tutorials_copy_and_install("*.h5m" "${permissions_default}")
 tutorials_copy_and_install("*.petsc" "${permissions_default}")
 tutorials_copy_and_install("*.sh" "${permissions_execute}")
 tutorials_copy_and_install("*.geo" "${permissions_execute}")