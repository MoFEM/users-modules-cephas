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
  ${UM_SOURCE_DIR}/basic_finite_elements/poisson
)
include_directories(
  ${UM_SOURCE_DIR}/basic_finite_elements/poisson/src
)

bfe_build_and_install(analytical_poisson
  ${CMAKE_CURRENT_SOURCE_DIR}/analytical_poisson.cpp)

bfe_build_and_install(analytical_poisson_field_split
  ${CMAKE_CURRENT_SOURCE_DIR}/analytical_poisson_field_split.cpp)

bfe_build_and_install(analytical_nonlinear_poisson
  ${CMAKE_CURRENT_SOURCE_DIR}/analytical_nonlinear_poisson.cpp)

cm_export_file(".petscrc" export_files_poisson)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/.petscrc 
  DESTINATION ${BFE_MOD_DIR})
  
add_test(analytical_poisson
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/analytical_poisson
  -file_name ${CMAKE_CURRENT_SOURCE_DIR}/cube_2part.h5m -order 3
  -test 1)

add_test(analytical_poisson_field_split
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/analytical_poisson_field_split
  -file_name ${CMAKE_CURRENT_SOURCE_DIR}/cube_2part.h5m -order 3 -pc_type fieldsplit
  -test 1)

add_test(analytical_nonlinear_poisson
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/analytical_nonlinear_poisson
  -file_name ${CMAKE_CURRENT_SOURCE_DIR}/cube_2part.h5m -order 4
  -snes_monitor -snes_converged_reason
  -snes_type newtonls -snes_linesearch_type basic -test 1)
