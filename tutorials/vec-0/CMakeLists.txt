include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

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

tutorials_copy_and_install("../vec-2/beam_2D.cub" "${permissions_default}")
tutorials_copy_and_install("../vec-2/beam_3D.cub" "${permissions_default}")

tutorials_build_and_install(
  elastic_2d ${CMAKE_CURRENT_SOURCE_DIR}/elastic.cpp)
set_target_properties(
  elastic_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")

# tutorials_build_and_install(
#   elastic_3d ${CMAKE_CURRENT_SOURCE_DIR}/elastic.cpp)
# set_target_properties(
#   elastic_3d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3")

add_test(
  elastic
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/elastic_2d
  -file_name part.h5m -order 3 -log_no_color)