# MIT License
#
# Copyright (c) 2022 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include_directories(${PROJECT_SOURCE_DIR}/basic_finite_elements/src)

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

set(PERCOMPILED_HEADER 
  ${PROJECT_BINARY_DIR}/basic_finite_elements/src/precompile/BasicFiniteElements.hpp)

function(bfe_add_executable target source)
  if(MoFEM_PRECOMPILED_HEADRES)
    set_source_files_properties(${source} PROPERTIES COMPILE_FLAGS "-include ${PERCOMPILED_HEADER}")
  endif(MoFEM_PRECOMPILED_HEADRES)
  add_executable(${target} ${source})
  if(MoFEM_PRECOMPILED_HEADRES)
    add_dependencies(${target} BasicFiniteElements.hpp.pch_copy)
  endif(MoFEM_PRECOMPILED_HEADRES)
endfunction(bfe_add_executable)


