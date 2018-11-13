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

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)

bfe_build_and_install(elasticity
  ${CMAKE_CURRENT_SOURCE_DIR}/elasticity.cpp)

#testing
add_test(ElasticityTest
 ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
 ${CMAKE_CURRENT_BINARY_DIR}/elasticity
 -my_file ${CMAKE_CURRENT_SOURCE_DIR}/LShape.h5m
 -ksp_type cg -pc_type asm -sub_pc_type lu
 -ksp_monitor -log_view)

add_test(MultiGridViaOrderSolver
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
  ${CMAKE_CURRENT_BINARY_DIR}/elasticity
  -my_file ${CMAKE_CURRENT_SOURCE_DIR}/LShape2Parts.h5m 
  -my_is_partitioned
  -ksp_type fgmres -ksp_monitor
  -my_order 4
  -pc_type mg
  -mofem_mg_verbose 1 -mofem_mg_coarse_order 2 -mofem_mg_levels 3
  -mg_coarse_ksp_type preonly -mg_coarse_pc_type bjacobi
  -pc_mg_smoothup 10 -pc_mg_smoothdown 10
  -pc_mg_type multiplicative
  -log_view)

