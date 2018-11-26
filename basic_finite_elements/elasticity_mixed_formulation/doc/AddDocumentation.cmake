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

# copy dox/figures to html directory created by doxygen
add_custom_target(elasticity_mixed_formulation
  ${CMAKE_COMMAND} -E copy_directory
  ${PROJECT_SOURCE_DIR}/users_modules/basic_finite_elements/elasticity_mixed_formulation/doc/figures ${PROJECT_BINARY_DIR}/html
)
add_dependencies(doc elasticity_mixed_formulation)