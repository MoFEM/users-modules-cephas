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


#copy figures form users UM documentation
add_custom_target(doxygen_copy_figures_from_user_modules
  ${CMAKE_COMMAND} -E copy_directory
  ${PROJECT_SOURCE_DIR}/users_modules/doc/figures ${PROJECT_BINARY_DIR}/html
)
add_dependencies(doc doxygen_copy_figures_from_user_modules)

#copy pdfs form users UM documentation
add_custom_target(doxygen_copy_pdfs_from_user_modules
  ${CMAKE_COMMAND} -E copy_directory
  ${PROJECT_SOURCE_DIR}/users_modules/doc/pdfs ${PROJECT_BINARY_DIR}/html
)
add_dependencies(doc doxygen_copy_pdfs_from_user_modules)
