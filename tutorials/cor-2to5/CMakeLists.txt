#
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

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

tutorials_build_and_install(analytical_poisson
  ${CMAKE_CURRENT_SOURCE_DIR}/analytical_poisson.cpp)

tutorials_build_and_install(analytical_poisson_field_split
  ${CMAKE_CURRENT_SOURCE_DIR}/analytical_poisson_field_split.cpp)

tutorials_build_and_install(analytical_nonlinear_poisson
  ${CMAKE_CURRENT_SOURCE_DIR}/analytical_nonlinear_poisson.cpp)

set(permissions_default 
 OWNER_WRITE 
 OWNER_READ
 GROUP_READ)

set(permissions_execute 
 OWNER_WRITE 
 OWNER_READ
 GROUP_READ
 OWNER_EXECUTE 
 GROUP_EXECUTE)

 tutorials_copy_and_install("*.msh" "${permissions_default}")
 tutorials_copy_and_install("*.cfg" "${permissions_default}")
 # tutorials_copy_and_install("*.jou" "${permissions_default}")
 tutorials_copy_and_install("*.cub" "${permissions_default}")
 tutorials_copy_and_install("*.h5m" "${permissions_default}")
 tutorials_copy_and_install(".petscrc" "${permissions_default}")
 tutorials_copy_and_install("*.sh" "${permissions_execute}")
 tutorials_copy_and_install("*.geo" "${permissions_execute}")
 
add_test(analytical_poisson
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/analytical_poisson
  -file_name ${CMAKE_CURRENT_SOURCE_DIR}/cube_2part.h5m -order 3
  -test 1)

add_test(analytical_poisson_field_split
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/analytical_poisson_field_split
  -file_name ${CMAKE_CURRENT_SOURCE_DIR}/cube_2part.h5m -order 3 
  -pc_type fieldsplit
  -test 1)

add_test(analytical_nonlinear_poisson
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
  ${CMAKE_CURRENT_BINARY_DIR}/analytical_nonlinear_poisson
  -file_name ${CMAKE_CURRENT_SOURCE_DIR}/cube_2part.h5m -order 4
  -snes_monitor -snes_converged_reason
  -snes_type newtonls -snes_linesearch_type basic -test 1)
