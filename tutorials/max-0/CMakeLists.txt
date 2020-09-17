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

tutorials_build_and_install(
  magnetostatic ${CMAKE_CURRENT_SOURCE_DIR}/magnetostatic.cpp)

add_test(magnetostatic
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
  ${CMAKE_CURRENT_BINARY_DIR}/magnetostatic
  -my_file ${CMAKE_CURRENT_SOURCE_DIR}/magnetic_coil_2parts.h5m -my_order 1 
  -ksp_type fgmres -pc_type lu -pc_factor_mat_solver_package mumps 
  -ksp_monitor -my_max_post_proc_ref_level 0 -regression_test 1)

set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

tutorials_copy_and_install("*.jou" "${permissions_default}")
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("README" "${permissions_default}")


