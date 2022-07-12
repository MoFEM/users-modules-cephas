include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../max-0/src)

tutorials_build_and_install(
  lorentz_force ${CMAKE_CURRENT_SOURCE_DIR}/lorentz_force.cpp)

set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

tutorials_copy_and_install("*.jou" "${permissions_default}")
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("README.md" "${permissions_default}")


