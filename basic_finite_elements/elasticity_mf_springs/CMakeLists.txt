#
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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

bfe_build_and_install(elasticity_mf_springs
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_mf_springs.cpp
  )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/bar_springs.cub
  ${CMAKE_CURRENT_BINARY_DIR}/
  @ONLY)  


#testing
add_test(elasticity_mf_springs
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 ${CMAKE_CURRENT_BINARY_DIR}/elasticity_mf_springs
  -my_file ${CMAKE_CURRENT_SOURCE_DIR}/bar_springs.cub
  -ksp_type gmres -pc_type lu 
  -pc_factor_mat_solver_package mumps
  )