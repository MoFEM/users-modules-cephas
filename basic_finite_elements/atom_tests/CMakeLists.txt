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
cm_export_file("elasticity_atom_test_body_force.h5m" export_atom_tests_basic_fe)

bfe_add_executable(body_force_atom_test ${CMAKE_CURRENT_SOURCE_DIR}/body_force_atom_test.cpp)
target_link_libraries(body_force_atom_test
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS})
add_test(
  basic_body_force_atom_test
  ${CMAKE_COMMAND}
  -DMPI_RUN=${MoFEM_MPI_RUN}
  -DMPI_FLAGS=${MPI_RUN_FLAGS}
  -DMPI_NP=1
  -DPROG=${CMAKE_CURRENT_BINARY_DIR}/body_force_atom_test
  -DFILE=elasticity_atom_test_body_force.h5m
  -DLOG1=body_force_atom_test.txt
  -DLOG2=body_force_atom_test.txt
  -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
  -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/blessed_files
  -P ${MoFEM_INSTALL_DIR}/include/cmake/RunAndComp.cmake)

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

add_test(
  basic_neumann_elasticity_atom_test_01X_pressure
  ${CMAKE_COMMAND}
  -DMPI_RUN=${MoFEM_MPI_RUN}
  -DMPI_FLAGS=${MPI_RUN_FLAGS}
  -DMPI_NP=1
  -DPROG=${CMAKE_CURRENT_BINARY_DIR}/neumann_forces_atom
  -DFILE=elasticity_atom_test_01X_pressure.h5m
  -DLOG1=forces_and_sources_elasticity_atom_test_01X_pressure.h5m.txt
  -DLOG2=forces_and_sources_elasticity_atom_test_01X_pressure.h5m.txt
  -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
  -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/blessed_files
  -P ${MoFEM_INSTALL_DIR}/include/cmake/RunAndComp.cmake)

add_test(
  basic_neumann_elasticity_atom_test_01X_pressure
  ${CMAKE_COMMAND}
  -DMPI_RUN=${MoFEM_MPI_RUN}
  -DMPI_FLAGS=${MPI_RUN_FLAGS}
  -DMPI_NP=1
  -DPROG=${CMAKE_CURRENT_BINARY_DIR}/neumann_forces_atom
  -DFILE=elasticity_atom_test_01X.h5m
  -DLOG1=forces_and_sources_elasticity_atom_test_01X.h5m.txt
  -DLOG2=forces_and_sources_elasticity_atom_test_01X.h5m.txt
  -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
  -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/blessed_files
  -P ${MoFEM_INSTALL_DIR}/include/cmake/RunAndComp.cmake)

add_test(
  basic_neumann_elasticity_atom_test_01X_pressure
  ${CMAKE_COMMAND}
  -DMPI_RUN=${MoFEM_MPI_RUN}
  -DMPI_FLAGS=${MPI_RUN_FLAGS}
  -DMPI_NP=1
  -DPROG=${CMAKE_CURRENT_BINARY_DIR}/neumann_forces_atom
  -DFILE=elasticity_atom_test_01X.h5m
  -DLOG1=forces_and_sources_elasticity_atom_test_01X.h5m.txt
  -DLOG2=forces_and_sources_elasticity_atom_test_01X.h5m.txt
  -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
  -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/blessed_files
  -P ${MoFEM_INSTALL_DIR}/include/cmake/RunAndComp.cmake)

add_test(
  basic_neumann_elasticity_atom_test_01X_pressure
  ${CMAKE_COMMAND}
  -DMPI_RUN=${MoFEM_MPI_RUN}
  -DMPI_FLAGS=${MPI_RUN_FLAGS}
  -DMPI_NP=1
  -DPROG=${CMAKE_CURRENT_BINARY_DIR}/neumann_forces_atom
  -DFILE=sphere.h5m
  -DLOG1=forces_and_sources_sphere.h5m.txt
  -DLOG2=forces_and_sources_sphere.h5m.txt
  -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
  -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/blessed_files
  -P ${MoFEM_INSTALL_DIR}/include/cmake/RunAndComp.cmake)

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
  ${MoFEM_PROJECT_LIBS})

add_test(
  basic_field_approximation_test
  ${CMAKE_COMMAND}
  -DMPI_RUN=${MoFEM_MPI_RUN}
  -DMPI_FLAGS=${MPI_RUN_FLAGS}
  -DMPI_NP=1
  -DPROG=${CMAKE_CURRENT_BINARY_DIR}/field_approximation
  -DFILE=elasticity_atom_test_01X_pressure.h5m
  -DLOG1=field_approximation.txt
  -DLOG2=field_approximation.txt
  -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
  -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/blessed_files
  -P ${MoFEM_INSTALL_DIR}/include/cmake/RunAndComp.cmake)

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

# Surface pressure element (ALE)

bfe_add_executable(testing_jacobian_of_surface_pressure_element
${CMAKE_CURRENT_SOURCE_DIR}/testing_jacobian_of_surface_pressure_element.cpp
)

target_link_libraries(testing_jacobian_of_surface_pressure_element
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

cm_export_file("mesh_for_testing_surface_pressure_element.h5m" export_atom_tests_basic_fe)
add_test(
  testing_jacobian_of_surface_pressure_element_ale
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/testing_jacobian_of_surface_pressure_element
  -file_name mesh_for_testing_surface_pressure_element.h5m
)

cm_export_file("2_tets_springs.cub" export_atom_tests_basic_fe)
add_test(
  testing_jacobian_of_spring_element_ale
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/testing_jacobian_of_surface_pressure_element
  -file_name 2_tets_springs.cub -snes_test_jacobian_display -snes_test_jacobian_display
)

# SIMPLE CONTACT

bfe_add_executable(test_jacobian_of_simple_contact_element
${CMAKE_CURRENT_SOURCE_DIR}/test_jacobian_of_simple_contact_element.cpp
)

target_link_libraries(test_jacobian_of_simple_contact_element
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

cm_export_file("2tets_contact.cub" export_atom_tests_basic_fe)
add_test(
  test_jacobian_of_simple_contact_element
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/test_jacobian_of_simple_contact_element
  -my_file 2tets_contact.cub -my_cn_value 1e2 -my_order 3 -my_order_lambda 3
)
add_test(
  test_jacobian_of_simple_contact_element_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/test_jacobian_of_simple_contact_element
  -my_file 2tets_contact.cub -my_cn_value 1e2 -my_order 3 -my_order_lambda 3 -alm_flag 1
)
add_test(
  test_jacobian_of_simple_contact_element_convect_integration_pts
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/test_jacobian_of_simple_contact_element
  -my_file 2tets_contact.cub -my_cn_value 1e2 -my_order 3 -my_order_lambda 3 
  -my_convect 1
)
add_test(
  test_jacobian_of_simple_contact_element_ale
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/test_jacobian_of_simple_contact_element
  -my_file 2tets_contact.cub -my_cn_value 1e2 -my_order 4 -my_order_lambda 3 
  -test_ale 1
)