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

bfe_build_and_install(
  reaction_diffusion_equation 
  ${CMAKE_CURRENT_SOURCE_DIR}/reaction_diffusion_equation.cpp)

bfe_build_and_install(
  reac_diff_mixed 
  ${CMAKE_CURRENT_SOURCE_DIR}/reac_diff_mixed.cpp)

 cm_export_file("param_file.petsc" export_reaction_diffusion_equation)
 cm_export_file("README" export_reaction_diffusion_equation)
 cm_export_file("mesh.jou" export_reaction_diffusion_equation)
 cm_export_file("mesh.cub" export_reaction_diffusion_equation)
 cm_export_file("mesh_quad.cub" export_reaction_diffusion_equation)
 cm_export_file("mesh_quad_2part.h5m" export_reaction_diffusion_equation)

 install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/param_file.petsc
  DESTINATION ${BFE_MOD_DIR})
 install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mesh.jou
   DESTINATION ${BFE_MOD_DIR})
 install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mesh.cub
   DESTINATION ${BFE_MOD_DIR})
 install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mesh_quad_2part.h5m
   DESTINATION ${BFE_MOD_DIR})

add_test(reaction_diffusion_equation
   ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
   ${CMAKE_CURRENT_BINARY_DIR}/reaction_diffusion_equation
   -file_name mesh_quad_2part.h5m
   -ts_final_time 0.4)
    