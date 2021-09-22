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

set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

set(permissions_execute 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ
  OWNER_EXECUTE 
  GROUP_EXECUTE)

tutorials_copy_and_install("*.msh" "${permissions_default}")
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("*.md" "${permissions_execute}")

tutorials_build_and_install(
  plot_base_2d ${CMAKE_CURRENT_SOURCE_DIR}/plot_base.cpp)
set_target_properties(
  plot_base_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")

# Add 3d when is tested
tutorials_build_and_install(
 plot_base_3d ${CMAKE_CURRENT_SOURCE_DIR}/plot_base.cpp)
set_target_properties(
 plot_base_3d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3") 

add_test(
    plot_base_2d_h1_ainsworth
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
    ${CMAKE_CURRENT_BINARY_DIR}/plot_base_2d
    -base ainsworth -space h1 -order 3 -log_no_color)