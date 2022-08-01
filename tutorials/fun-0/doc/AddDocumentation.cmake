# copy dox/figures to html directory created by doxygen
add_custom_target(hellow_world_docs
  ${CMAKE_COMMAND} -E copy_directory
  ${ADD_DOC_DIRECTORY}/figures ${PROJECT_BINARY_DIR}/html)
add_dependencies(doc hellow_world_docs)