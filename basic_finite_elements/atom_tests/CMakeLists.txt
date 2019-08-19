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
  ${PROJECT_SOURCE_DIR}/basic_finite_elements/nonlinear_elastic_materials/src
)

# body forces test
bfe_add_executable(body_force_atom ${CMAKE_CURRENT_SOURCE_DIR}/body_force_atom_test.cpp)
target_link_libraries(body_force_atom
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

cm_export_file("elasticity_atom_test_body_force.h5m" export_atom_tests_basic_fe)
add_test(basic_body_force_atom_test
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/body_force_atom -my_file elasticity_atom_test_body_force.h5m
)
add_test(basic_forces_and_sources_body_force_atom_test_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/body_force_atom_test.txt
  ${CMAKE_CURRENT_BINARY_DIR}/body_force_atom_test.txt
)

# neumann forces
cm_export_file("elasticity_atom_test_01X_pressure.h5m" export_atom_tests_basic_fe)

bfe_add_executable(neumann_forces_atom ${CMAKE_CURRENT_SOURCE_DIR}/neumann_forces.cpp)
target_link_libraries(neumann_forces_atom
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS})

cm_export_file("elasticity_atom_test_01X.h5m" export_atom_tests_basic_fe)
cm_export_file("sphere.h5m" export_atom_tests_basic_fe)

add_test(basic_neumann_elasticity_atom_test_01X_pressure
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/neumann_forces_atom -my_file elasticity_atom_test_01X_pressure.h5m
)
add_test(basic_neumann_elasticity_atom_test_01X_pressure_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/forces_and_sources_elasticity_atom_test_01X_pressure.cub.txt
  ${CMAKE_CURRENT_BINARY_DIR}/forces_and_sources_elasticity_atom_test_01X_pressure.h5m.txt
)
add_test(basic_neumann_elasticity_atom_test_01X
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/neumann_forces_atom -my_file elasticity_atom_test_01X.h5m
)
add_test(basic_neumann_elasticity_atom_test_01X_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/forces_and_sources_elasticity_atom_test_01X.cub.txt
  ${CMAKE_CURRENT_BINARY_DIR}/forces_and_sources_elasticity_atom_test_01X.h5m.txt
)
add_test(basic_neumann_sphre_pressure_test
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/neumann_forces_atom -my_file sphere.h5m
)
add_test(basic_neumann_sphre_pressure_test_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/forces_and_sources_sphere.cub.txt
  ${CMAKE_CURRENT_BINARY_DIR}/forces_and_sources_sphere.h5m.txt
)

# add example test for forces_and_sources_fluid_pressure_element_atom
cm_export_file("fluid_presssure_cub_test.h5m" export_atom_tests_basic_fe)
bfe_add_executable(fluid_pressure_element_atom ${CMAKE_CURRENT_SOURCE_DIR}/fluid_pressure_element.cpp)
target_link_libraries(fluid_pressure_element_atom
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

add_test(basic_fluid_pressure_element_atom_test
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/fluid_pressure_element_atom -my_file fluid_presssure_cub_test.h5m
)

# add example test for temperature element
cm_export_file("simple_thermal_problem.h5m" export_atom_tests_basic_fe)
bfe_add_executable(thermal_elem_atom ${CMAKE_CURRENT_SOURCE_DIR}/thermal_elem.cpp)
target_link_libraries(thermal_elem_atom
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

add_test(
  basic_thermal_elem_atom_test ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/thermal_elem_atom
  -my_file simple_thermal_problem.h5m -ksp_type fgmres -pc_type bjacobi -ksp_atol 0 -ksp_rtol 1e-12 -ksp_monitor
)

# add test test for temerature unsteady element
bfe_add_executable(thermal_elem_unsteady_atom ${CMAKE_CURRENT_SOURCE_DIR}/thermal_elem_unsteady.cpp)
target_link_libraries(thermal_elem_unsteady_atom
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

add_test(
  basic_thermal_elem_unsteady_atom_test
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/thermal_elem_unsteady_atom
  -my_file simple_thermal_problem.h5m
  -ksp_type fgmres -pc_type bjacobi -ksp_atol 0 -ksp_rtol 1e-12
  -snes_type newtonls -snes_linesearch_type basic
  -snes_max_it 100 -snes_atol 1e-8 -snes_rtol 1e-8
  -ts_monitor -ts_type beuler -ts_dt 1 -ts_final_time 2  -ksp_monitor -snes_max_it 3 -snes_monitor
)

# add theraml stress eleme
cm_export_file("elasticity_atom_test_thermal_expansion.h5m" export_atom_tests_basic_fe)
bfe_add_executable(thermal_stress_elem ${CMAKE_CURRENT_SOURCE_DIR}/thermal_stress_elem.cpp)
target_link_libraries(thermal_stress_elem
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

add_test(
  basic_thermal_stress_elem_atom_test
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/thermal_stress_elem
  -my_file elasticity_atom_test_thermal_expansion.h5m
)

cm_export_file("sphere.jou" export_analytical_bc_atom_tests)
cm_export_file("sphere.cub" export_analytical_bc_atom_tests)

# Analytical dirihlet boundary condition

add_executable(thermal_with_analytical_bc
${CMAKE_CURRENT_SOURCE_DIR}/thermal_with_analytical_bc.cpp
)

target_link_libraries(thermal_with_analytical_bc
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

add_test(thermal_with_analytical_bc_atom
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
  ${CMAKE_CURRENT_BINARY_DIR}/thermal_with_analytical_bc
  -my_file sphere.cub
  -ksp_type fgmres -pc_type bjacobi -sub_pc_type lu -ksp_rtol 0 -ksp_atol 1e-12
  -ksp_monitor
)

# Testing field approximation
bfe_add_executable(
  field_approximation ${CMAKE_CURRENT_SOURCE_DIR}/field_approximation.cpp
)
target_link_libraries(field_approximation
  users_modules
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

add_test(
  basic_field_approximation_test
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/field_approximation
  -my_file elasticity_atom_test_01X_pressure.h5m
  -ksp_monitor -ksp_type fgmres -pc_type bjacobi -ksp_atol 0 -ksp_rtol 1e-12
)
add_test(
  basic_field_approximation_test_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/field_approximation.txt
  ${CMAKE_CURRENT_BINARY_DIR}/field_approximation.txt
)

if(ADOL-C_LIBRARY)

  cm_export_file("LShape.h5m" export_atom_tests_basic_fe)

  bfe_add_executable(
    nonlinear_elastic_basic_test
    ${CMAKE_CURRENT_SOURCE_DIR}/nonlinear_elastic.cpp
  )
  target_link_libraries(nonlinear_elastic_basic_test
    users_modules
    mofem_finite_elements
    mofem_interfaces
    mofem_multi_indices
    mofem_petsc
    mofem_approx
    mofem_cblas
    mofem_third_party
    ${MoFEM_PROJECT_LIBS}
  )
  add_test(
    basic_nonlinear_elastic_basic_test ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
    ${CMAKE_CURRENT_BINARY_DIR}/nonlinear_elastic_basic_test -my_file LShape.h5m
  )

  bfe_add_executable(
    nonlinear_elastic_test_jacobian
    ${CMAKE_CURRENT_SOURCE_DIR}/nonlinear_elastic_test_jacobian.cpp
  )
  target_link_libraries(
    nonlinear_elastic_test_jacobian
    users_modules
    mofem_finite_elements
    mofem_interfaces
    mofem_multi_indices
    mofem_petsc
    mofem_approx
    mofem_cblas
    mofem_third_party
    ${MoFEM_PROJECT_LIBS}
  )
  add_test(
    basic_nonlinear_elastic_jacobian_test ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
    ${CMAKE_CURRENT_BINARY_DIR}/nonlinear_elastic_test_jacobian -my_file LShape.h5m
  )
  add_test(
    basic_nonlinear_hooke_mat_test ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
    ${CMAKE_CURRENT_BINARY_DIR}/nonlinear_elastic_test_jacobian
    -my_file LShape.h5m -mat HOOKE
  )
  add_test(
    basic_nonlinear_neohookean_mat_test ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
    ${CMAKE_CURRENT_BINARY_DIR}/nonlinear_elastic_test_jacobian
    -my_file LShape.h5m -mat NEOHOOKEAN
  )

endif(ADOL-C_LIBRARY)

# kelving voigt damper
if(ADOL-C_LIBRARY)

  bfe_add_executable(damper_jacobian_test
    ${CMAKE_CURRENT_SOURCE_DIR}/damper_jacobian_test.cpp
  )

  target_link_libraries(damper_jacobian_test
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

  add_test(
    basic_damper_jacobian_atom_test
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/damper_jacobian_test
    -my_file fluid_presssure_cub_test.h5m
  )

endif(ADOL-C_LIBRARY)


# Convective element

if(ADOL-C_LIBRARY)

  bfe_add_executable(convective_matrix
    ${CMAKE_CURRENT_SOURCE_DIR}/convective_matrix.cpp
  )

  target_link_libraries(convective_matrix
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

  cm_export_file("LShape.cub" export_atom_tests_basic_fe)
  add_test(
    basic_convective_matrix
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 ${CMAKE_CURRENT_BINARY_DIR}/convective_matrix -my_file LShape.cub
  )

endif(ADOL-C_LIBRARY)

# Elastic element

bfe_add_executable(testing_jacobian_of_hook_element
  ${CMAKE_CURRENT_SOURCE_DIR}/testing_jacobian_of_hook_element.cpp
)

target_link_libraries(testing_jacobian_of_hook_element
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

cm_export_file("mesh_for_testing_elastic_jacobian.h5m" export_atom_tests_basic_fe)
add_test(
  testing_jacobian_of_hook_element
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/testing_jacobian_of_hook_element
  -file_name mesh_for_testing_elastic_jacobian.h5m
  )

add_test(
  testing_jacobian_of_hook_element_ale
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/testing_jacobian_of_hook_element
  -file_name mesh_for_testing_elastic_jacobian.h5m -ale
)


bfe_add_executable(testing_jacobian_of_hook_scaled_with_density_element
  ${CMAKE_CURRENT_SOURCE_DIR}/testing_jacobian_of_hook_scaled_with_density_element.cpp
)

target_link_libraries(testing_jacobian_of_hook_scaled_with_density_element
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

add_test(
  testing_jacobian_of_hook_scaled_with_density_element
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/testing_jacobian_of_hook_scaled_with_density_element
  -file_name mesh_for_testing_elastic_jacobian.h5m 
  )

add_test(
  testing_jacobian_of_hook_scaled_with_density_element_ale
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/testing_jacobian_of_hook_scaled_with_density_element
  -file_name mesh_for_testing_elastic_jacobian.h5m -ale
  )

bfe_add_executable(testing_jacobian_of_navier_stokes_element
  ${CMAKE_CURRENT_SOURCE_DIR}/testing_jacobian_of_navier_stokes_element.cpp
)

target_link_libraries(testing_jacobian_of_navier_stokes_element
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
cm_export_file("testing_jacobian_of_navier_stokes_element.h5m" export_atom_tests_basic_fe)