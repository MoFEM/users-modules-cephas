# MoFEM is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# MoFEM is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MoFEM. If not, see <http://www.gnu.org/licenses/>

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add tests
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/atom_tests)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/l-shape.msh
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)    
file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/bc.cfg
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/.petscrc
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
  
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/l-shape.msh
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bc.cfg
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/.petscrc
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)  

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)


bfe_build_and_install(
  h_adaptive_transport ${CMAKE_CURRENT_SOURCE_DIR}/h_adaptive_transport.cpp)

file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README DESTINATION ${BFE_MOD_DIR})


if(ADOL-C_LIBRARY)

  bfe_build_and_install(
    unsaturated_transport ${CMAKE_CURRENT_SOURCE_DIR}/unsaturated_transport.cpp)

  file(
    COPY ${CMAKE_CURRENT_SOURCE_DIR}/run_uf.sh
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE)    
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/run_uf.sh 
    DESTINATION ${BFE_MOD_DIR}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_EXECUTE)

  add_test(
    unsaturated_flow ${CMAKE_CURRENT_BINARY_DIR}/run_uf.sh
    ${CMAKE_CURRENT_SOURCE_DIR}/soil_impermeable.cub 
    ${CMAKE_CURRENT_SOURCE_DIR}/unsaturated.cfg 2 0.0001 0.0003)

endif(ADOL-C_LIBRARY)

