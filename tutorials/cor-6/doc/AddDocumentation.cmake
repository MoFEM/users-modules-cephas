#

# copy dox/figures to html directory created by doxygen
add_custom_target(simple_elasticity
  ${CMAKE_COMMAND} -E copy_directory
  ${ADD_DOC_DIRECTORY}/figures ${PROJECT_BINARY_DIR}/html)
add_dependencies(doc simple_elasticity)