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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../elasticity/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elasticity)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

if(ADOL-C_LIBRARY)

  bfe_build_and_install(arc_length_interface
    ${CMAKE_CURRENT_SOURCE_DIR}/arc_length_interface.cpp)

  file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README DESTINATION ${BFE_MOD_DIR})

  cm_export_file("PeelTestHalfCrack.h5m" export_files_cohesive_interface)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/PeelTestHalfCrack.h5m DESTINATION ${BFE_MOD_DIR})

  cm_export_file("PeelTest.h5m" export_files_cohesive_interface)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/PeelTest.h5m DESTINATION ${BFE_MOD_DIR})


  #testing
  add_test(ArcLengthInterfaceTest
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 ${CMAKE_CURRENT_BINARY_DIR}/arc_length_interface
    -my_file ${CMAKE_CURRENT_SOURCE_DIR}/PeelTestHalfCrack.h5m -my_order 1
    -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_package mumps
    -ksp_monitor -ksp_atol 1e-10 -ksp_rtol 1e-10
    -snes_monitor -snes_type newtonls -snes_linesearch_type l2 -snes_linesearch_monitor
    -snes_max_it 16 -snes_atol 1e-8 -snes_rtol 1e-8 -my_sr 1e-1 -my_ms 3 -snes_converged_reason
    -log_view)

endif(ADOL-C_LIBRARY)
