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

set(BFE_DIRS 
  elasticity
  nonlinear_elasticity
  thermal
  cohesive_interface)

function(bfe_install bfe_mod_dir)
  install(
    DIRECTORY
    ${PROJECT_SOURCE_DIR}/basic_finite_elements/${bfe_mod_dir} 
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}
    FILES_MATCHING
    PATTERN "README"
    PATTERN "*.h5m"
    PATTERN "*.cub"
    PATTERN "*.jou"
    PATTERN "*.msh"
    PATTERN "*.cfg"
    PATTERN "*.geo"
    PATTERN "src" EXCLUDE
    PATTERN "doc" EXCLUDE
    PATTERN "atom_tests" EXCLUDE)
endfunction(bfe_install)

function(bfe_build_and_install target source)
  bfe_add_executable(${target} ${source})
  target_link_libraries(${target}
    users_modules
    mofem_finite_elements
    mofem_interfaces
    mofem_multi_indices
    mofem_petsc
    mofem_approx
    mofem_third_party
    ${MoFEM_PROJECT_LIBS})
  install(TARGETS ${target} DESTINATION ${BFE_MOD_DIR})
endfunction(bfe_build_and_install)

function(bfe_copy_and_install extension file_permissions)
  file(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/${extension})
  foreach(file ${files})
    file(
      COPY ${file}
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
      FILE_PERMISSIONS ${file_permissions})     
    install(FILES ${file}
      DESTINATION ${BFE_MOD_DIR}
      PERMISSIONS ${file_permissions})
  endforeach(file)
endfunction()

if(MOFEM_UM_BUILD_TESTS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/basic_finite_elements/atom_tests 
  ${PROJECT_BINARY_DIR}/basic_finite_elements/atom_tests)
endif(MOFEM_UM_BUILD_TESTS)

option(BUILD_BASIC_FINITE_ELEMENTS
  "Compile basic finite elements" ON)
message(STATUS "Build basic finite elements ${BUILD_BASIC_FINITE_ELEMENTS}")
  
if(BUILD_BASIC_FINITE_ELEMENTS) 

foreach(LOOP_DIR ${BFE_DIRS})
  set(BFE_MOD_DIR ${LOOP_DIR})
  add_subdirectory(${LOOP_DIR})
  bfe_install(${LOOP_DIR})
endforeach(LOOP_DIR)

endif(BUILD_BASIC_FINITE_ELEMENTS)