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

include_directories(simple_elasticity
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_elasticity.cpp)

bfe_build_and_install(simple_elasticity
  ${UM_SOURCE_DIR}/basic_finite_elements/simple_elasticity/simple_elasticity.cpp)

add_test(simple_elasticity
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/simple_elasticity
  -file_name ${CMAKE_CURRENT_SOURCE_DIR}/simple_elasticity.h5m -order 3
  -test 1)

