# copy dox/figures to html directory created by doxygen
add_custom_target(reaction_diffusion_imp
  ${CMAKE_COMMAND} -E copy_directory
  ${ADD_DOC_DIRECTORY}/figures ${PROJECT_BINARY_DIR}/html)
add_dependencies(doc reaction_diffusion_imp)