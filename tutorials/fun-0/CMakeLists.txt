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

tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("README" "${permissions_execute}")

tutorials_build_and_install(
  hello_world ${CMAKE_CURRENT_SOURCE_DIR}/hello_world.cpp)

add_test(hello_world
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
  ${CMAKE_CURRENT_BINARY_DIR}/hello_world)