# MIT License
#
# Copyright (c) 2022 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../elasticity/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elasticity)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)  

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/PeelTest.h5m
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/PeelTest.h5m 
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/PeelTestHalfCrack.h5m
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
  
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/PeelTestHalfCrack.h5m 
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

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
    -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_type mumps
    -ksp_monitor -ksp_atol 1e-10 -ksp_rtol 1e-10
    -snes_monitor -snes_type newtonls -snes_linesearch_type l2 -snes_linesearch_monitor
    -snes_max_it 16 -snes_atol 1e-8 -snes_rtol 1e-8 -my_sr 1e-1 -my_ms 3 -snes_converged_reason
    -log_view)

endif(ADOL-C_LIBRARY)
