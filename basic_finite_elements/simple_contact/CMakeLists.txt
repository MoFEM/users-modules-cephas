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

set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party)
include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party/phg-quadrule)
include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party/cblas)
include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../elasticity/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elasticity)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)


bfe_build_and_install(simple_contact
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_contact.cpp)

bfe_build_and_install(simple_contact_thermal
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_contact_thermal.cpp)

bfe_copy_and_install("README.md" "${permissions_default}")

file(
  COPY examples DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

install(
  DIRECTORY examples DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  
add_test(
  test_simple_contact_8cube
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 1
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/8cube_contact.cub 
  -my_order 2
  -my_cn_value 1e3
)

add_test(
  test_simple_contact_4seasons
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 2
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/4seasons_contact.cub
  -my_order 2
  -my_cn_value 1e3
  -snes_monitor
)

add_test(
  test_simple_contact_Tinterface
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 3
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/Tinterface_contact.cub
  -my_order 2
  -my_cn_value 1e3
  -snes_monitor
)

add_test(
  test_simple_contact_punch_top_and_mid
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 4
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/punch_top_and_mid.cub
  -my_order 2
  -my_cn_value 1e3
  -snes_monitor
)

add_test(
  test_simple_contact_punch_top_only
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 5
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/punch_top_only.cub
  -my_order 2
  -my_cn_value 1e3
  -snes_monitor
)

add_test(
  test_simple_contact_plane_axisymmetric
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 6
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/plane_axi_contact.cub
  -my_order 2
  -my_cn_value 1e3
  -snes_monitor
)

add_test(
  test_simple_contact_arc_3surfaces
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 7
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/arc_3surf_contact.cub
  -my_order 2
  -my_cn_value 1e3
  -snes_monitor
)

add_test(
  test_simple_contact_smiling_face
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 8
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/smiling_face_contact.cub
  -my_order 2
  -my_cn_value 1e-1
  -snes_monitor
)

add_test(
  test_simple_contact_smiling_face_convective
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 9
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/smiling_face_contact.cub
  -my_order 2
  -my_cn_value 1e-1
  -my_convect 1
  -snes_monitor
)

add_test(
  test_simple_contact_wave_2d
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 10
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/plane_strain_contact.cub
  -my_order 1
  -my_order_contact 2
  -my_ho_levels_num 2
  -my_cn_value 1e2
  -my_alm_flag 0
  -my_wave_surf 1
  -my_wave_surf_block_id 1
  -my_wave_dim 2
  -my_wave_length 2.0
  -my_wave_ampl 0.01
  -my_mesh_height 1.0
  -snes_monitor
  -snes_divergence_tolerance 0
)

add_test(
  test_simple_contact_8cube_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 1
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/8cube_contact.cub 
  -my_order 2
  -my_cn_value 1e3
  -my_alm_flag 1
)

add_test(
  test_simple_contact_4seasons_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 2
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/4seasons_contact.cub
  -my_order 2
  -my_cn_value 1e3
  -my_alm_flag 1
  -snes_monitor
)

add_test(
  test_simple_contact_Tinterface_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 3
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/Tinterface_contact.cub
  -my_order 2
  -my_cn_value 1e3
  -my_alm_flag 1
  -snes_monitor
)

add_test(
  test_simple_contact_punch_top_and_mid_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 4
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/punch_top_and_mid.cub
  -my_order 2
  -my_cn_value 1e3
  -my_alm_flag 1
  -snes_monitor
)

add_test(
  test_simple_contact_punch_top_only_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 5
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/punch_top_only.cub
  -my_order 2
  -my_cn_value 1e3
  -my_alm_flag 1
  -snes_monitor
)

add_test(
  test_simple_contact_plane_axisymmetric_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 6
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/plane_axi_contact.cub
  -my_order 2
  -my_cn_value 1e3
  -my_alm_flag 1
  -snes_monitor
)

add_test(
  test_simple_contact_arc_3surfaces_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 7
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/arc_3surf_contact.cub
  -my_order 2
  -my_cn_value 1e-3
  -my_alm_flag 1
  -snes_monitor
)

add_test(
  test_simple_contact_smiling_face_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 8
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/smiling_face_contact.cub
  -my_order 2
  -my_cn_value 1e-1
  -my_alm_flag 1
  -snes_monitor
)

add_test(
  test_simple_contact_wave_2d_alm
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_test_num 11
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/plane_strain_contact.cub
  -my_order 1
  -my_order_contact 2
  -my_ho_levels_num 2
  -my_cn_value 1e2
  -my_alm_flag 1
  -my_wave_surf 1
  -my_wave_surf_block_id 1
  -my_wave_dim 2
  -my_wave_length 2.0
  -my_wave_ampl 0.01
  -my_mesh_height 1.0
  -snes_monitor
  -snes_divergence_tolerance 0
)

add_test(
  test_hooke_internal_stress_1
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact_thermal
  -my_test_num 1
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/hooke_internal_stress_1.cub
  -my_order 2
  -my_output_mesh_file med_stress.h5m
  -my_analytical_input 1
  -my_save_mean_stress 1
  -my_thermal_expansion_coef 1e-3
  -snes_monitor
)

add_test(
  test_hooke_internal_stress_2
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact_thermal
  -my_test_num 2
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/hooke_internal_stress_2.cub
  -my_order 2
  -my_output_mesh_file med_stress.h5m
  -my_analytical_input 1
  -my_save_mean_stress 1
  -my_thermal_expansion_coef 1e-3
  -snes_monitor
)

add_test(
  test_two_bars_contact_analytical_input
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact_thermal
  -my_test_num 3
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/two_bars_contact_thermal.cub
  -my_order 2
  -my_cn_value 1e2
  -my_alm_flag 0
  -my_output_mesh_file two_bars_contact_stress.h5m
  -my_analytical_input 1
  -my_thermal_expansion_coef 0.2
  -my_save_mean_stress 0
  -my_deform_flat 1
  -my_mesh_height 1.0
  -my_flat_shift 0.1
  -my_wave_surf 0
  -my_scale_factor 1.0
  -my_delete_prisms 0
  -my_eigen_pos_flag 0
  -snes_monitor
)

add_test(
  test_two_bars_contact_eigen_stress
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact_thermal
  -my_test_num 3
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/two_bars_contact_stress.h5m
  -my_order 2
  -my_cn_value 1e2
  -my_alm_flag 0
  -my_output_mesh_file med_stress.h5m
  -my_analytical_input 0
  -my_stress_tag_name INTERNAL_STRESS
  -my_save_mean_stress 1
  -my_deform_flat 0
  -my_wave_surf 0
  -my_scale_factor 1.0
  -my_delete_prisms 1
  -my_eigen_pos_flag 0
  -snes_monitor
)

add_test(
  test_two_bars_contact_actual_stress_eigen_pos
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact_thermal
  -my_test_num 3
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/two_bars_contact_stress.h5m
  -my_order 2
  -my_cn_value 1e4
  -my_alm_flag 0
  -my_output_mesh_file med_stress.h5m
  -my_analytical_input 0
  -my_stress_tag_name ACTUAL_STRESS
  -my_save_mean_stress 1
  -my_deform_flat 0
  -my_wave_surf 0
  -my_scale_factor 1.0
  -my_delete_prisms 1
  -my_eigen_pos_flag 1
  -snes_monitor
)