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
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("*.md" "${permissions_execute}")

tutorials_build_and_install(
  plot_base_2d ${CMAKE_CURRENT_SOURCE_DIR}/plot_base.cpp)
set_target_properties(
  plot_base_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")

# Add 3d when is tested
tutorials_build_and_install(
 plot_base_3d ${CMAKE_CURRENT_SOURCE_DIR}/plot_base.cpp)
set_target_properties(
 plot_base_3d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3") 

add_test(
    plot_base_2d_h1_ainsworth
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
    ${CMAKE_CURRENT_BINARY_DIR}/plot_base_2d
    -base ainsworth -space h1 -order 3 -log_no_color)