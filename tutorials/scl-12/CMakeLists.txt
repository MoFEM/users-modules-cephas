include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(
  ${UM_SOURCE_DIR}/tutorials/cor-2to5/src
)

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
tutorials_copy_and_install("*.petsc" "${permissions_default}")
tutorials_copy_and_install("*.sh" "${permissions_execute}")
tutorials_copy_and_install("*.md" "${permissions_default}")

tutorials_build_and_install(
  electrostatic_2d_homogeneous ${CMAKE_CURRENT_SOURCE_DIR}/electrostatic_2d_homogeneous.cpp)
tutorials_build_and_install(
  electrostatic_3d_homogeneous ${CMAKE_CURRENT_SOURCE_DIR}/electrostatic_3d_homogeneous.cpp)


    if(SLEPC_FOUND)

    tutorials_build_and_install(
      electrostatics_3d ${CMAKE_CURRENT_SOURCE_DIR}/electrostatics.cpp)
    set_target_properties(
      electrostatics_3d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3")
  
    tutorials_build_and_install(
      electrostatics_2d ${CMAKE_CURRENT_SOURCE_DIR}/electrostatics.cpp) 
    set_target_properties(
      electrostatics_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")
  
    add_test(
      electrostatics_2d
      ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
      ${CMAKE_CURRENT_BINARY_DIR}/electrostatics_2d
      -eps_ncv 300 
      -eps_krylovschur_restart 0.1
      -file_name fork-2-0_2parts.h5m 
      -order 1
      -log_no_color -test)
  
  endif(SLEPC_FOUND)
  


# add_test(
#   electrostatichomogeneous
#   ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
#   ${CMAKE_CURRENT_BINARY_DIR}/electrostatichomogeneous
#   -file_name squareGD.h5m 
#   -order 1
#   -log_no_color -test)
  