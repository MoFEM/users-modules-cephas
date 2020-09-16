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
tutorials_copy_and_install("*.cfg" "${permissions_default}")
tutorials_copy_and_install("*.jou" "${permissions_default}")
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("*.sh" "${permissions_execute}")
tutorials_copy_and_install("README" "${permissions_execute}")

tutorials_build_and_install(
  h_adaptive_transport ${CMAKE_CURRENT_SOURCE_DIR}/h_adaptive_transport.cpp)

if(ADOL-C_LIBRARY)

  tutorials_build_and_install(
    unsaturated_transport ${CMAKE_CURRENT_SOURCE_DIR}/unsaturated_transport.cpp)

  add_test(
    unsaturated_flow ${CMAKE_CURRENT_BINARY_DIR}/run_uf.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/soil_impermeable.h5m
    ${CMAKE_CURRENT_SOURCE_DIR}/unsaturated.cfg 2 0.0001 0.0003)

endif(ADOL-C_LIBRARY)

