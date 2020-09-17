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

set(TUTORIALS_INSTALL_DIR "tutorials") 

function(tutorials_add_executable target source)
  if(MoFEM_PRECOMPILED_HEADRES)
    set_source_files_properties(${source} PROPERTIES COMPILE_FLAGS "-include ${PERCOMPILED_HEADER}")
  endif(MoFEM_PRECOMPILED_HEADRES)
  add_executable(${target} ${source})
  if(MoFEM_PRECOMPILED_HEADRES)
    add_dependencies(${target} BasicFiniteElements.hpp.pch_copy)
  endif(MoFEM_PRECOMPILED_HEADRES)
endfunction(tutorials_add_executable)

function(tutorials_install tutorials_mod_dir)
  install(
    DIRECTORY
    ${TUTORIALS_INSTALL_DIR} 
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/${TUTORIALS_MOD_DIR}
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
endfunction(tutorials_install)



function(tutorials_build_and_install target source)
  tutorials_add_executable(${target} ${source})
  target_link_libraries(${target}
    users_modules
    mofem_finite_elements
    mofem_interfaces
    mofem_multi_indices
    mofem_petsc
    mofem_approx
    mofem_third_party
    mofem_cblas
    ${MoFEM_PROJECT_LIBS})
  install(TARGETS ${target} DESTINATION ${TUTORIALS_INSTALL_DIR}/${TUTORIALS_MOD_DIR})
endfunction(tutorials_build_and_install)

function(tutorials_copy_and_install extension file_permissions)
  file(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/${extension})
  foreach(file ${files})
    file(
      COPY ${file}
      DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
      FILE_PERMISSIONS ${file_permissions})     
    install(FILES ${file}
      DESTINATION ${TUTORIALS_INSTALL_DIR}/${TUTORIALS_MOD_DIR}
      PERMISSIONS ${file_permissions})
  endforeach(file)
endfunction()

set(TUTORIALS_DIRS 
  fundamentals
  hello_world
  scl-0
  scl-1
  scl-2
  scl-3
  scl-4
  scl-5
  scl-6
  scl-7
  clx-0
  max-0
  max-1
  mix_transport
  simple_elasticity
)

foreach(LOOP_DIR ${TUTORIALS_DIRS})
  set(TUTORIALS_MOD_DIR ${LOOP_DIR})
  add_subdirectory(${LOOP_DIR})
  tutorials_install(${LOOP_DIR})
endforeach(LOOP_DIR)