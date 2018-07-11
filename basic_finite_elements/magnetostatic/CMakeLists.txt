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

file(
  COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
)
cm_export_file("magnetic_coil.cub" export_files_magneto_static)
cm_export_file("magnetic_coil.jou" export_files_magneto_static)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# For convinience source files from MoFEM project are used. If user is
# implementing independent application it will be using instead project
# directories.

bfe_add_executable(
  magnetostatic ${CMAKE_CURRENT_SOURCE_DIR}/magnetostatic.cpp
)
target_link_libraries(magnetostatic
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  mofem_cblas
  ${MoFEM_PROJECT_LIBS}
)

add_test(MagnetostaticTest
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
  ${CMAKE_CURRENT_BINARY_DIR}/magnetostatic
  -my_file magnetic_coil.cub -my_order 1 
  -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_package mumps 
  -ksp_monitor -my_max_post_proc_ref_level 0 -regression_test 1
)