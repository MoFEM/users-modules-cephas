include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(
  ${UM_SOURCE_DIR}/tutorials/cor-2to5/src
)

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

tutorials_copy_and_install("*.jou" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("*.petsc" "${permissions_default}")
tutorials_copy_and_install("*.sh" "${permissions_execute}")
tutorials_copy_and_install("*.md" "${permissions_default}")

tutorials_build_and_install(
  electrostatic_2d_homogeneous ${CMAKE_CURRENT_SOURCE_DIR}/electrostatic_2d_homogeneous.cpp)
tutorials_build_and_install(
  electrostatic_3d_homogeneous ${CMAKE_CURRENT_SOURCE_DIR}/electrostatic_3d_homogeneous.cpp)
