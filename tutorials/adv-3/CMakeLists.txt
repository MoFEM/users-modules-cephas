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
  level_set_2d ${CMAKE_CURRENT_SOURCE_DIR}/level_set.cpp)
set_target_properties(
  level_set_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")

add_definitions(-Wfatal-errors)