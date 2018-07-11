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

if(MED_LIBRARY)
  add_executable(read_med ${CMAKE_CURRENT_SOURCE_DIR}/read_med.cpp)
  target_link_libraries(read_med
    mofem_finite_elements
    mofem_interfaces
    mofem_multi_indices
    mofem_petsc
    mofem_approx
    mofem_cblas
    mofem_third_party
    ${MoFEM_PROJECT_LIBS}
  )
endif(MED_LIBRARY)

add_executable(mofem_part ${CMAKE_CURRENT_SOURCE_DIR}/mofem_part.cpp)
target_link_libraries(mofem_part
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

add_executable(split_sideset ${CMAKE_CURRENT_SOURCE_DIR}/split_sideset.cpp)
target_link_libraries(split_sideset
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

add_executable(mesh_cut ${CMAKE_CURRENT_SOURCE_DIR}/mesh_cut.cpp)
target_link_libraries(mesh_cut
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

add_executable(field_to_vertices ${CMAKE_CURRENT_SOURCE_DIR}/field_to_vertices.cpp)
target_link_libraries(field_to_vertices
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

add_executable(delete_ho_nodes ${CMAKE_CURRENT_SOURCE_DIR}/delete_ho_nodes.cpp)
target_link_libraries(delete_ho_nodes
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_cblas
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)

if(ADOL-C_LIBRARY)

  include_directories(
    ${PROJECT_SOURCE_DIR}/basic_finite_elements/nonlinear_elastic_materials/src
  )

  add_executable(mesh_smoothing ${CMAKE_CURRENT_SOURCE_DIR}/mesh_smoothing.cpp)
  target_link_libraries(mesh_smoothing
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

endif(ADOL-C_LIBRARY)