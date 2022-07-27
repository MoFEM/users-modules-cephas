# copy dox/figures to html directory created by doxygen
add_custom_target(complex_scalar
  ${CMAKE_COMMAND} -E copy_directory
  ${ADD_DOC_DIRECTORY}/figures ${PROJECT_BINARY_DIR}/html)
add_dependencies(doc complex_scalar)