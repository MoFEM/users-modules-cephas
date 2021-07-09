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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)

if(ADOL-C_LIBRARY)
  bfe_build_and_install(arc_length_nonlinear_elasticity
    ${CMAKE_CURRENT_SOURCE_DIR}/arc_length_nonlinear_elasticity.cpp)

  set(permissions_default 
    OWNER_WRITE 
    OWNER_READ
    GROUP_READ)
  
  bfe_copy_and_install("README" "${permissions_default}")
  bfe_copy_and_install("*.h5m" "${permissions_default}")
  bfe_copy_and_install("*.txt" "${permissions_default}")
  bfe_copy_and_install("*.ipynb" "${permissions_execute}")
  bfe_copy_and_install("*.md" "${permissions_execute}")

  file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/do_vtk.sh.in
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE)
  configure_file(
    ${CMAKE_CURRENT_BINARY_DIR}/do_vtk.sh.in
    ${CMAKE_CURRENT_BINARY_DIR}/do_vtk.sh
    @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/do_vtk.sh 
    DESTINATION ${BFE_MOD_DIR} 
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/do_vtk.sh 
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE)

  file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  install(
    DIRECTORY examples DESTINATION ${BFE_MOD_DIR})

  #testing
  add_test(ArcLengthNonLinearElasticityTest
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
    ${CMAKE_CURRENT_BINARY_DIR}/arc_length_nonlinear_elasticity
    -my_file ${CMAKE_CURRENT_SOURCE_DIR}/dome_thin_symm.h5m -ksp_type fgmres
    -pc_type asm -sub_pc_type lu
    -ksp_atol 1e-10 -ksp_rtol 1e-10
    -snes_monitor
    -snes_type newtonls -snes_linesearch_type basic -snes_linesearch_monitor
    -snes_max_it 16 -snes_atol 1e-8 -snes_rtol 1e-8
    -my_sr 1e-6 -my_ms 3 -snes_converged_reason
    -log_view)

  bfe_build_and_install(nonlinear_dynamics
    ${CMAKE_CURRENT_SOURCE_DIR}/nonlinear_dynamics.cpp)

  add_test(NonLinearDynamics
    ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
    ${CMAKE_CURRENT_BINARY_DIR}/nonlinear_dynamics 
    -my_file ${CMAKE_CURRENT_SOURCE_DIR}/LShape.h5m
    -ksp_type fgmres -pc_type asm -sub_pc_type lu
    -ksp_atol 1e-10 -ksp_rtol 1e-10 -snes_monitor
    -snes_type newtonls -snes_linesearch_type basic
    -snes_max_it 100 -snes_atol 1e-7 -snes_rtol 1e-7
    -ts_monitor -ts_type alpha -ts_dt 0.001 -ts_max_time 0.002
    -my_time_data_file ${CMAKE_CURRENT_SOURCE_DIR}/data.txt -my_output_prt 1
    -my_max_post_proc_ref_level 0 -my_disp_order 3 -my_vel_order 3
  )

  if(SLEPC_FOUND)

    bfe_build_and_install(stability ${CMAKE_CURRENT_SOURCE_DIR}/stability.cpp)
  
    add_test(EigenProblem
      ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2 
      ${CMAKE_CURRENT_BINARY_DIR}/stability
      -my_file ${CMAKE_CURRENT_SOURCE_DIR}/rod.h5m
      -ksp_type gmres -pc_type lu -pc_factor_mat_solver_type mumps 
      -ksp_monitor -ksp_atol 1e-12 -ksp_rtol 1e-12
      -st_ksp_type preonly -st_pc_type cholesky -st_pc_factor_mat_solver_type mumps
      -eps_monitor -eps_tol 1e-12
      -eps_nev 2 -eps_ncv 10  -eps_smalest_real -eps_gen_no_hermitian
      -my_max_post_proc_ref_level 0 -st_ksp_atol 1e-12 -st_ksp_rtol 1e-12 -my_order 2)

  endif(SLEPC_FOUND)

endif(ADOL-C_LIBRARY)

