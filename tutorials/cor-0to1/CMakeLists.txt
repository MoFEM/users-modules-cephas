include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add tests
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/atom_tests)

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
tutorials_copy_and_install("*.jou" "${permissions_default}")
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("*.sh" "${permissions_execute}")
tutorials_copy_and_install("README" "${permissions_execute}")

tutorials_build_and_install(
  h_adaptive_transport ${CMAKE_CURRENT_SOURCE_DIR}/h_adaptive_transport.cpp)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/run_uf.sh.in"
  "${CMAKE_CURRENT_BINARY_DIR}/run_uf.sh"
  @ONLY
)

install(FILES 
  ${CMAKE_CURRENT_BINARY_DIR}/run_uf.sh 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TUTORIALS_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE)


if(ADOL-C_LIBRARY)

  tutorials_build_and_install(
    unsaturated_transport ${CMAKE_CURRENT_SOURCE_DIR}/unsaturated_transport.cpp)

  add_test(
    unsaturated_flow ${CMAKE_CURRENT_BINARY_DIR}/run_uf.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/soil_impermeable.h5m
    ${CMAKE_CURRENT_SOURCE_DIR}/unsaturated.cfg 2 0.0001 0.0003)

endif(ADOL-C_LIBRARY)

