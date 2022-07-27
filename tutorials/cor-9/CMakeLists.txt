tutorials_build_and_install(
  reaction_diffusion 
  ${CMAKE_CURRENT_SOURCE_DIR}/reaction_diffusion.cpp)

add_test(reaction_diffusion
   ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
   ${CMAKE_CURRENT_BINARY_DIR}/reaction_diffusion
   -file_name mesh_quad_2part.h5m
   -ts_max_time 0.4)

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
 tutorials_copy_and_install("*.cub" "${permissions_default}")
 tutorials_copy_and_install("*.h5m" "${permissions_default}")
 tutorials_copy_and_install("*.sh" "${permissions_execute}")
 tutorials_copy_and_install("*.geo" "${permissions_execute}")
 tutorials_copy_and_install("README" "${permissions_execute}")
 tutorials_copy_and_install("param_file.petsc" "${permissions_execute}")
 
    