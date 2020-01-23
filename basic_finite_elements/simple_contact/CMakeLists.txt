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


include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party)
include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party/phg-quadrule)
include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party/cblas)
include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../elasticity/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elasticity)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)  

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

bfe_build_and_install(simple_contact
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_contact.cpp)
