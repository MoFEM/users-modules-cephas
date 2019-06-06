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
  level_set ${CMAKE_CURRENT_SOURCE_DIR}/reaction_diffusion_equation.cpp)

 cm_export_file(".petscrc" export_reaction_diffusion_equation)
 cm_export_file("README" export_reaction_diffusion_equation)
 cm_export_file("mesh.jou" export_reaction_diffusion_equation)
 cm_export_file("mesh.cub" export_reaction_diffusion_equation)
 install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/.petscrc 
  DESTINATION ${BFE_MOD_DIR})
 install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mesh.jou
   DESTINATION ${BFE_MOD_DIR})
 install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/mesh.h5m
   DESTINATION ${BFE_MOD_DIR})
    
