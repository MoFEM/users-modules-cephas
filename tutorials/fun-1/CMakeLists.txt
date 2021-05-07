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

include_directories(${CURREENT_SOURCE_DIR})
include_directories(${CURREENT_SOURCE_DIR}/src)

tutorials_build_and_install(
  integration
  ${CMAKE_CURRENT_SOURCE_DIR}/integration.cpp)

set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

tutorials_copy_and_install("*.petsc" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")

add_test(
  integration
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/integration
  -file_name ${CMAKE_CURRENT_BINARY_DIR}/cube.h5m -test -log_no_color)
