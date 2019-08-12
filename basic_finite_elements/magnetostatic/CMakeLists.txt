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

bfe_build_and_install(
  magnetostatic ${CMAKE_CURRENT_SOURCE_DIR}/magnetostatic.cpp)
bfe_build_and_install(
  lorentz_force ${CMAKE_CURRENT_SOURCE_DIR}/lorentz_force.cpp)

file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README DESTINATION ${BFE_MOD_DIR})
  
file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/.petscrc
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README DESTINATION ${BFE_MOD_DIR})

cm_export_file("magnetic_coil.cub" export_files_magnetostatic)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/magnetic_coil.cub DESTINATION ${BFE_MOD_DIR})

add_test(MagnetostaticTest
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
  ${CMAKE_CURRENT_BINARY_DIR}/magnetostatic
  -my_file ${CMAKE_CURRENT_SOURCE_DIR}/magnetic_coil.cub -my_order 1 
  -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_package mumps 
  -ksp_monitor -my_max_post_proc_ref_level 0 -regression_test 1)

cm_export_file("magnetic_coil.jou" export_files_magenotstatic)
cm_export_file("magnetic_coil.cub" export_files_magenotstatic)
cm_export_file("README" export_files_magenotstatic)
cm_export_file(".petscrc" export_files_magenotstatic)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/.petscrc
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/magnetic_coil.cub
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/magnetic_coil.jou
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)  
