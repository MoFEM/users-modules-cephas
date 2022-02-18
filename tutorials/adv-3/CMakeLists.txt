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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../adv-0/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vec-0/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vec-2/src)

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
tutorials_copy_and_install("*.png" "${permissions_execute}")
tutorials_copy_and_install("*.md" "${permissions_execute}")
tutorials_copy_and_install("*.cfg" "${permissions_execute}")
tutorials_copy_and_install("*.jou" "${permissions_execute}")
tutorials_copy_and_install("meshes/*" "${permissions_execute}")

tutorials_build_and_install(
  ale_plastic_2d ${CMAKE_CURRENT_SOURCE_DIR}/ale_plastic.cpp)

set_target_properties(
  ale_plastic_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")

# set(DEBUG false)
# if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
#   set(DEBUG true)
# endif ()

# tutorials_build_and_install(
#   thermo_plastic_3d ${CMAKE_CURRENT_SOURCE_DIR}/plastic.cpp)
  
# set_target_properties(
#   thermo_plastic_3d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3")

# add_test(
#   plastic_3d_necking
#   ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 2
#   plastic_3d 
#   -file_name necking_3D_p2.h5m
#   -ts_dt 0.02
#   -ts_max_time 1.2
#   -ts_max_steps 4
#   -ts_type theta
#   -snes_atol 1e-9
#   -order 2
#   -large_strains 1
#   -scale 1
#   -hardening_viscous 0
#   -cn 1 -ksp_monitor 
#   -log_no_color)