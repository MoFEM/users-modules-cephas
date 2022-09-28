include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../scl-1/src)

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

tutorials_copy_and_install("*.jou" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.petsc" "${permissions_default}")
tutorials_copy_and_install("*.sh" "${permissions_execute}")
tutorials_copy_and_install("*.md" "${permissions_default}")
tutorials_copy_and_install("*.ipynb" "${permissions_default}")

tutorials_build_and_install(
  poisson_2d_dis_galerkin ${CMAKE_CURRENT_SOURCE_DIR}/poisson_2d_dis_galerkin.cpp)

add_test(
  poisson_2d_dis_galerkin
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 3
  poisson_2d_dis_galerkin
  -file_name mesh_dense_p3.h5m
  -order 4 
  -penalty 1
  -phi -1 
  -nitsche 1 
  -log_sl inform
  -is_test)
