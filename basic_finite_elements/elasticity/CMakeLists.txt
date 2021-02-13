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

 set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

bfe_copy_and_install("*.cfg" "${permissions_default}")
bfe_copy_and_install("*.h5m" "${permissions_default}")
bfe_copy_and_install("*.med" "${permissions_default}")
bfe_copy_and_install("*.geo" "${permissions_default}")
bfe_copy_and_install("*.msh" "${permissions_default}")

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/README 
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

bfe_build_and_install(elasticity
  ${CMAKE_CURRENT_SOURCE_DIR}/elasticity.cpp)

#testing
add_test(ElasticityTest
 ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
 ${CMAKE_CURRENT_BINARY_DIR}/elasticity
 -my_file ${CMAKE_CURRENT_SOURCE_DIR}/LShape.h5m
 -ksp_type cg -pc_type asm -sub_pc_type lu
 -ksp_monitor -log_view -is_atom_test 1)

add_test(ElasticityDirichletOnPartition
 ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
 ${CMAKE_CURRENT_BINARY_DIR}/elasticity
 -my_file ${CMAKE_CURRENT_SOURCE_DIR}/LShape2PartsDirich.h5m
 -ksp_type cg -pc_type asm -sub_pc_type lu
 -ksp_monitor -log_view -my_is_partitioned -is_atom_test 2)

add_test(ElasticityRotationTest
 ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
 ${CMAKE_CURRENT_BINARY_DIR}/elasticity
 -my_file ${CMAKE_CURRENT_SOURCE_DIR}/LShapeRot.cub
 -ksp_type cg -pc_type asm -sub_pc_type lu
 -ksp_monitor -log_view -my_order 2 -is_atom_test 3)

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

add_test(Spring
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
  ${CMAKE_CURRENT_BINARY_DIR}/elasticity
  -my_file ${CMAKE_CURRENT_SOURCE_DIR}/spring_LShape.h5m
  -my_order 2 
  -ksp_type cg 
  -pc_type asm 
  -sub_pc_type lu 
  -ksp_monitor)

  add_test(SimpleRod
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
  ${CMAKE_CURRENT_BINARY_DIR}/elasticity
  -my_file ${CMAKE_CURRENT_SOURCE_DIR}/simple_rod_beam.h5m
  -my_order 1 
  -my_is_calculating_frequency 1
  -ksp_type cg 
  -pc_type asm 
  -sub_pc_type lu 
  -ksp_monitor)

  add_test(AnaliticalStrain
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
    ${CMAKE_CURRENT_BINARY_DIR}/elasticity
    -my_file ${CMAKE_CURRENT_SOURCE_DIR}/free_beam.h5m 
    -my_order 4 -ksp_type cg -pc_type lu 
    -ksp_monitor -is_atom_test 4
  )

  add_test(NormalSpringForDismplacements
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/elasticity
  -my_file ${CMAKE_CURRENT_SOURCE_DIR}/test_for_springs_normal.cub 
  -my_order 1 -ksp_type cg -pc_type lu 
  -ksp_monitor -is_atom_test 5
)

add_test(NormalSpringForDismplacements
${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
${CMAKE_CURRENT_BINARY_DIR}/elasticity
-my_file ${CMAKE_CURRENT_SOURCE_DIR}/test_for_springs_tangent.cub 
-my_order 2 -ksp_type cg -pc_type lu 
-ksp_monitor -is_atom_test 6
)