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

include_directories(${PROJECT_SOURCE_DIR}/basic_finite_elements/src/impl)

# Users modules library, common for all programs
add_library(
  users_modules 
  ${PROJECT_SOURCE_DIR}/basic_finite_elements/src/impl/All.cpp)
target_link_libraries(users_modules PUBLIC ${MoFEM_PROJECT_LIBS})
set_target_properties(users_modules PROPERTIES VERSION ${PROJECT_VERSION})

install(
  TARGETS users_modules 
  DESTINATION lib/basic_finite_elements 
  EXPORT users_modules_targets)
install(
  EXPORT users_modules_targets
  FILE users_modules_targets.cmake
  DESTINATION lib/basic_finite_elements
)

