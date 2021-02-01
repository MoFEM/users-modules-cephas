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

include_directories(${PROJECT_SOURCE_DIR}/basic_finite_elements/lessons)
include_directories(${PROJECT_SOURCE_DIR}/basic_finite_elements/lessons/src)
include_directories(${PROJECT_SOURCE_DIR}/tutorials/vec-0/src)

bfe_build_and_install(
    basic_poisson
    ${CMAKE_CURRENT_SOURCE_DIR}/lesson3_poisson.cpp)  

bfe_build_and_install(
      basic_radiation
      ${CMAKE_CURRENT_SOURCE_DIR}/lesson6_radiation.cpp)    

bfe_build_and_install(
  basic_plastic
  ${CMAKE_CURRENT_SOURCE_DIR}/lesson7_plastic.cpp)

set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

bfe_copy_and_install(".petscrc" "${permissions_default}")
bfe_copy_and_install("*.h5m" "${permissions_default}")

