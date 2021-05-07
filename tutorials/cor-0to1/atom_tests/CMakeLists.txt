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

# add test for mix transport finite element
tutorials_add_executable(
  mix_transport ${CMAKE_CURRENT_SOURCE_DIR}/mix_transport.cpp)
target_link_libraries(mix_transport
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS})

add_test(
  basic_mix_transport_atom
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
  ${CMAKE_CURRENT_BINARY_DIR}/mix_transport
  -my_file ${CMAKE_CURRENT_SOURCE_DIR}/simple_thermal_problem.h5m
  -ksp_type fgmres -pc_type bjacobi -ksp_monitor -ksp_atol 0 -ksp_rtol 1e-12
  -my_max_post_proc_ref_level 0 -my_order 0)
