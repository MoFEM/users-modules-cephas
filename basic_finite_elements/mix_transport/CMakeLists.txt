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

# Add tests
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/atom_tests)

bfe_build_and_install(
  h_adaptive_transport ${CMAKE_CURRENT_SOURCE_DIR}/h_adaptive_transport.cpp)

if(ADOL-C_LIBRARY)

  bfe_build_and_install(
    unsaturated_transport ${CMAKE_CURRENT_SOURCE_DIR}/unsaturated_transport.cpp)

  add_test(
    unsaturated_flow ${CMAKE_CURRENT_BINARY_DIR}/run_uf.sh
    soil_impermeable.cub unsaturated.cfg 2 0.0001 0.0003)

endif(ADOL-C_LIBRARY)

