# copy dox/figures to html directory created by doxygen

add_custom_target(mesh_generation_msh_2
  ${CMAKE_COMMAND} -E copy_directory
  ${ADD_DOC_DIRECTORY}/figures ${PROJECT_BINARY_DIR}/html)
add_dependencies(doc mesh_generation_msh_2)