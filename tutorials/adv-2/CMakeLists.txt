include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../adv-0/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vec-0/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vec-2/src)

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
  thermo_elastic_2d ${CMAKE_CURRENT_SOURCE_DIR}/thermo_elastic.cpp)
set_target_properties(
  thermo_elastic_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")

tutorials_build_and_install(
  thermo_elastic_3d ${CMAKE_CURRENT_SOURCE_DIR}/thermo_elastic.cpp)
set_target_properties(
  thermo_elastic_3d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3")