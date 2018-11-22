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

include_directories(
  ${UM_SOURCE_DIR}/basic_finite_elements/nonlinear_elastic_materials/src
)

include_directories(
  ${UM_SOURCE_DIR}/basic_finite_elements/elasticity_mixed_formulation/src
)

if(ADOL-C_LIBRARY)


  bfe_build_and_install(elasticity_mixed_formulation
    ${CMAKE_CURRENT_SOURCE_DIR}/elasticity_mixed_formulation.cpp
  )

  target_link_libraries(elasticity_mixed_formulation
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
  
  cm_export_file("LShape.h5m" export_files_mix)

  file(
    COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/README
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/
  )

  #testing
  add_test(MixFormulationTest
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 ${CMAKE_CURRENT_BINARY_DIR}/elasticity_mixed_formulation
    -my_file LShape_incompressible.cub
    -ksp_type gmres -pc_type lu 
    -pc_factor_mat_solver_package mumps
  )

endif(ADOL-C_LIBRARY)
