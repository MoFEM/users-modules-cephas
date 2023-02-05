#

if(MED_LIBRARY)
  add_executable(read_med ${CMAKE_CURRENT_SOURCE_DIR}/read_med.cpp)
  target_link_libraries(read_med
    mofem_finite_elements
    mofem_interfaces
    mofem_multi_indices
    mofem_petsc
    mofem_approx
    mofem_third_party
    ${MoFEM_PROJECT_LIBS}
  )
  install(TARGETS read_med DESTINATION bin)
endif(MED_LIBRARY)

add_executable(add_meshsets ${CMAKE_CURRENT_SOURCE_DIR}/add_meshsets.cpp)
target_link_libraries(add_meshsets
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS add_meshsets DESTINATION bin)

add_executable(mofem_part ${CMAKE_CURRENT_SOURCE_DIR}/mofem_part.cpp)
target_link_libraries(mofem_part
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS mofem_part DESTINATION bin)

add_executable(split_sideset ${CMAKE_CURRENT_SOURCE_DIR}/split_sideset.cpp)
target_link_libraries(split_sideset
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS split_sideset DESTINATION bin)

add_executable(mesh_cut ${CMAKE_CURRENT_SOURCE_DIR}/mesh_cut.cpp)
target_link_libraries(mesh_cut
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS mesh_cut DESTINATION bin)

add_executable(field_to_vertices ${CMAKE_CURRENT_SOURCE_DIR}/field_to_vertices.cpp)
target_link_libraries(field_to_vertices
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS field_to_vertices DESTINATION bin)

add_executable(remove_mofem_meshsets
  ${CMAKE_CURRENT_SOURCE_DIR}/remove_mofem_meshsets.cpp)
target_link_libraries(remove_mofem_meshsets
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS remove_mofem_meshsets DESTINATION bin)

add_executable(delete_ho_nodes ${CMAKE_CURRENT_SOURCE_DIR}/delete_ho_nodes.cpp)
target_link_libraries(delete_ho_nodes
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS delete_ho_nodes DESTINATION bin)

add_executable(meshset_to_vtk ${CMAKE_CURRENT_SOURCE_DIR}/meshset_to_vtk.cpp)
target_link_libraries(meshset_to_vtk
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS meshset_to_vtk DESTINATION bin)

add_executable(uniform_mesh_refinement 
  ${CMAKE_CURRENT_SOURCE_DIR}/uniform_mesh_refinement.cpp)
target_link_libraries(uniform_mesh_refinement
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS uniform_mesh_refinement DESTINATION bin)

add_executable(extrude_prisms ${CMAKE_CURRENT_SOURCE_DIR}/extrude_prisms.cpp)
target_link_libraries(extrude_prisms
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS extrude_prisms DESTINATION bin)

add_executable(get_skin ${CMAKE_CURRENT_SOURCE_DIR}/get_skin.cpp)
target_link_libraries(get_skin
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS get_skin DESTINATION bin)

add_executable(fix_file ${CMAKE_CURRENT_SOURCE_DIR}/fix_file.cpp)
target_link_libraries(fix_file
  mofem_finite_elements
  mofem_interfaces
  mofem_multi_indices
  mofem_petsc
  mofem_approx
  mofem_third_party
  ${MoFEM_PROJECT_LIBS}
)
install(TARGETS fix_file DESTINATION bin)

cm_export_file("convert.py" export_tools)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/convert.py
  DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
  GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

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
    mofem_third_party
    ${MoFEM_PROJECT_LIBS}
  )
  install(TARGETS mesh_smoothing DESTINATION bin)

endif(ADOL-C_LIBRARY)