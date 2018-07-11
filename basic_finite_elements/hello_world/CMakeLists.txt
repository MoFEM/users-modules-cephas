include_directories(
  ${UM_SOURCE_DIR}/basic_finite_elements/hello_world
)

bfe_add_executable(hello_world
  ${CMAKE_CURRENT_SOURCE_DIR}/hello_world.cpp
)
target_link_libraries(hello_world
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  mofem_cblas
  ${MoFEM_PROJECT_LIBS}
)

add_test(hello_world
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
  ${CMAKE_CURRENT_BINARY_DIR}/hello_world
)

cm_export_file("mesh.h5m" export_files_hello_world)
