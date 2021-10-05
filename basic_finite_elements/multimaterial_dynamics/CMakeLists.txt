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

set(permissions_default 
 OWNER_WRITE 
 OWNER_READ
 GROUP_READ)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)

bfe_copy_and_install("*.cfg" "${permissions_default}")
bfe_copy_and_install("*.h5m" "${permissions_default}")


file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/README 
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

bfe_build_and_install(multimaterial_dynamics
  ${CMAKE_CURRENT_SOURCE_DIR}/multimaterial_dynamics.cpp)


