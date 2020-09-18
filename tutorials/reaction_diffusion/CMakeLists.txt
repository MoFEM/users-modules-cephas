# MoFEM is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# MoFEM is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MoFEM. If not, see <http://www.gnu.org/licenses/>

tutorials_build_and_install(
  reaction_diffusion 
  ${CMAKE_CURRENT_SOURCE_DIR}/reaction_diffusion.cpp)

add_test(reaction_diffusion
   ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
   ${CMAKE_CURRENT_BINARY_DIR}/reaction_diffusion
   -file_name mesh_quad_2part.h5m
   -ts_final_time 0.4)

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
 
    