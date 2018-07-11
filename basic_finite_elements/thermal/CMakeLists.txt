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

file(
  COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
)

# For convinience source files from MoFEM project are used. If user is
# implementing independent application it will be using instead project
# directories.

bfe_add_executable(thermal_steady ${CMAKE_CURRENT_SOURCE_DIR}/thermal_steady.cpp)
target_link_libraries(thermal_steady
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  mofem_cblas
  ${MoFEM_PROJECT_LIBS}
)

if(${GROUND_SURFACE_TEMPERATURE})
  include_directories("${UM_SOURCE_DIR}/ground_surface_temperature/src")
  include_directories("${UM_SOURCE_DIR}/ground_surface_temperature/third_party")
  bfe_add_executable(thermal_unseady
    ${CMAKE_CURRENT_BINARY_DIR}/thermal_unsteady.cpp
    ${UM_SOURCE_DIR}/ground_surface_temperature/third_party/spa.c
  )
else(${GROUND_SURFACE_TEMPERATURE})
  bfe_add_executable(
    thermal_unsteady ${CMAKE_CURRENT_SOURCE_DIR}/thermal_unsteady.cpp
  )
endif()

target_link_libraries(thermal_unsteady
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  mofem_cblas
  ${MoFEM_PROJECT_LIBS}
)
