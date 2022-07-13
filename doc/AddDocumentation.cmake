#


#copy figures form users UM documentation
add_custom_target(doxygen_copy_figures_from_user_modules
  ${CMAKE_COMMAND} -E copy_directory
  ${ADD_DOC_DIRECTORY}/figures ${PROJECT_BINARY_DIR}/html)
add_dependencies(doc doxygen_copy_figures_from_user_modules)

#copy pdfs form users UM documentation
add_custom_target(doxygen_copy_pdfs_from_user_modules
  ${CMAKE_COMMAND} -E copy_directory
  ${ADD_DOC_DIRECTORY}/pdfs ${PROJECT_BINARY_DIR}/html)
add_dependencies(doc doxygen_copy_pdfs_from_user_modules)
