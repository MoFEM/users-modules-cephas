# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vec-0/src)

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
tutorials_copy_and_install("README.md" "${permissions_execute}")
tutorials_copy_and_install("*.ipynb" "${permissions_execute}")
tutorials_copy_and_install("*.md" "${permissions_execute}")

if(SLEPC_FOUND)

  tutorials_build_and_install(
    eigen_elastic_3d ${CMAKE_CURRENT_SOURCE_DIR}/eigen_elastic.cpp)
  set_target_properties(
    eigen_elastic_3d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3")

  tutorials_build_and_install(
    eigen_elastic_2d ${CMAKE_CURRENT_SOURCE_DIR}/eigen_elastic.cpp) 
  set_target_properties(
    eigen_elastic_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")

  add_test(
    eigen_elastic_3d
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
    ${CMAKE_CURRENT_BINARY_DIR}/eigen_elastic_3d
    -eps_ncv 300 
    -eps_krylovschur_restart 0.1
    -file_name fork-2-0_2parts.h5m 
    -order 1
    -log_no_color -test)

endif(SLEPC_FOUND)