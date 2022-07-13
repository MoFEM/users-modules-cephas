#
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

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
tutorials_copy_and_install("README" "${permissions_execute}")

tutorials_build_and_install(elasticity_mixed_formulation
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_mixed_formulation.cpp)

#testing
add_test(elasticity_mixed_formulation
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 ${CMAKE_CURRENT_BINARY_DIR}/elasticity_mixed_formulation
  -my_file  ${CMAKE_CURRENT_SOURCE_DIR}/LShape_incompressible.cub
  -ksp_type gmres -pc_type lu 
  -pc_factor_mat_solver_type mumps
  )
