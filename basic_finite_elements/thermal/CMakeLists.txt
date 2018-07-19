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

bfe_build_and_install(thermal_steady ${CMAKE_CURRENT_SOURCE_DIR}/thermal_steady.cpp)

if(${GROUND_SURFACE_TEMPERATURE})
  # FIXME: That will not always work if is installed in external directory
  include_directories("${UM_SOURCE_DIR}/ground_surface_temperature/src")
  include_directories("${UM_SOURCE_DIR}/ground_surface_temperature/third_party")
  bfe_build_and_install(thermal_unseady
    ${CMAKE_CURRENT_BINARY_DIR}/thermal_unsteady.cpp
    ${UM_SOURCE_DIR}/ground_surface_temperature/third_party/spa.c)
else(${GROUND_SURFACE_TEMPERATURE})
  bfe_build_and_install(
    thermal_unsteady ${CMAKE_CURRENT_SOURCE_DIR}/thermal_unsteady.cpp)
endif()
