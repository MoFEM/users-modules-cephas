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

include_directories(${PROJECT_SOURCE_DIR}/basic_finite_elements/basic)
include_directories(${PROJECT_SOURCE_DIR}/basic_finite_elements/basic/src)

bfe_build_and_install(
  second_moment_of_inertia
  ${CMAKE_CURRENT_SOURCE_DIR}/second_moment_of_inertia.cpp)
