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

include_directories(${PROJECT_SOURCE_DIR}/basic_finite_elements/src)
include_directories(${PROJECT_SOURCE_DIR}/basic_finite_elements/src/impl)

if(MoFEM_PRECOMPILED_HEADRES)

  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(OUT_PCH_SUFFIX "pch")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(OUT_PCH_SUFFIX "gch")
  endif()

  # BasicFiniteElements.hpp
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/basic_finite_elements/src/BasicFiniteElements.hpp
    PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "-x c++-header"
  )
  add_library(
    BasicFiniteElements.hpp.pch
    OBJECT
    ${PROJECT_SOURCE_DIR}/basic_finite_elements/src/BasicFiniteElements.hpp
  )
  add_custom_target(
    BasicFiniteElements.hpp.pch_copy
    ${CMAKE_COMMAND} -E copy_if_different
    ${PROJECT_BINARY_DIR}/CMakeFiles/BasicFiniteElements.hpp.pch.dir/basic_finite_elements/src/BasicFiniteElements.hpp.o
    ${PROJECT_BINARY_DIR}/basic_finite_elements/src/precompile/BasicFiniteElements.hpp.${OUT_PCH_SUFFIX}
    COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${PROJECT_SOURCE_DIR}/basic_finite_elements/src/BasicFiniteElements.hpp
    ${PROJECT_BINARY_DIR}/basic_finite_elements/src/precompile/BasicFiniteElements.hpp
    COMMENT
    "Copy precompiled BasicFiniteElements.hpp header"
  )
  add_dependencies(BasicFiniteElements.hpp.pch_copy BasicFiniteElements.hpp.pch)

endif(MoFEM_PRECOMPILED_HEADRES)

set(PERCOMPILED_HEADER ${PROJECT_BINARY_DIR}/basic_finite_elements/src/precompile/BasicFiniteElements.hpp)

function(bfe_add_executable target source)
  if(MoFEM_PRECOMPILED_HEADRES)
    set_source_files_properties(${source} PROPERTIES COMPILE_FLAGS "-include ${PERCOMPILED_HEADER}")
  endif(MoFEM_PRECOMPILED_HEADRES)
  add_executable(${target} ${source})
  if(MoFEM_PRECOMPILED_HEADRES)
    add_dependencies(${target} BasicFiniteElements.hpp.pch_copy)
  endif(MoFEM_PRECOMPILED_HEADRES)
endfunction(bfe_add_executable)

set(UM_LIB_SOURCES
  ${UM_LIB_SOURCES} ${PROJECT_SOURCE_DIR}/basic_finite_elements/src/impl/All.cpp
)

add_subdirectory(${PROJECT_SOURCE_DIR}/basic_finite_elements)
