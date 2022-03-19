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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../scl-1/src)

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

tutorials_copy_and_install("*.jou" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.petsc" "${permissions_default}")
tutorials_copy_and_install("*.sh" "${permissions_execute}")
tutorials_copy_and_install("*.md" "${permissions_default}")
tutorials_copy_and_install("*.ipynb" "${permissions_default}")

tutorials_build_and_install(
  poisson_2d_dis_galerkin ${CMAKE_CURRENT_SOURCE_DIR}/poisson_2d_dis_galerkin.cpp)

  add_test(
    poisson_2d_dis_galerkin
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
    poisson_2d_dis_galerkin
    -file_name mesh2d.cub 
    -order 4 
    -penalty 1
    -phi -1 
    -nitsche 1 
    -log_sl inform
    -is_test)