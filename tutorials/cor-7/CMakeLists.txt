#
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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

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
tutorials_copy_and_install("*.petsc" "${permissions_default}")
tutorials_copy_and_install("*.sh" "${permissions_execute}")
tutorials_copy_and_install("*.geo" "${permissions_execute}")
tutorials_copy_and_install("README" "${permissions_execute}")

tutorials_build_and_install(elasticity_mixed_formulation
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_mixed_formulation.cpp)

#testing
add_test(elasticity_mixed_formulation
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 ${CMAKE_CURRENT_BINARY_DIR}/elasticity_mixed_formulation
  -my_file  ${CMAKE_CURRENT_SOURCE_DIR}/LShape_incompressible.cub
  -ksp_type gmres -pc_type lu 
  -pc_factor_mat_solver_package mumps
  )
