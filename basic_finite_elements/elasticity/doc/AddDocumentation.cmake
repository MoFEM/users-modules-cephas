#

# copy dox/figures to html directory created by doxygen
add_custom_target(using_Gmsh
  ${CMAKE_COMMAND} -E copy_directory
  ${PROJECT_SOURCE_DIR}/users_modules/basic_finite_elements/elasticity/doc/figures ${PROJECT_BINARY_DIR}/html
)
add_dependencies(doc using_Gmsh)
