include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vec-0/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vec-2/src)

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
tutorials_copy_and_install("*.txt" "${permissions_execute}")

tutorials_build_and_install(
  nonlinear_dynamic_elastic_3d ${CMAKE_CURRENT_SOURCE_DIR}/nonlinear_dynamic_elastic.cpp)
tutorials_build_and_install(
    nonlinear_dynamic_elastic_2d ${CMAKE_CURRENT_SOURCE_DIR}/nonlinear_dynamic_elastic.cpp)
    
set_target_properties(
  nonlinear_dynamic_elastic_3d 
  PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3")

set_target_properties(
  nonlinear_dynamic_elastic_2d 
  PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")

add_test(
  nonlinear_dynamic_elastic
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/nonlinear_dynamic_elastic_2d
  -file_name part.h5m -order 1 -ts_max_time 0.3 -ts_dt 0.1 
  -time_vector_file body_force.txt -log_no_color -test)