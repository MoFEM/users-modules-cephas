#

bfe_build_and_install(thermal_steady ${CMAKE_CURRENT_SOURCE_DIR}/thermal_steady.cpp)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/simple_thermal_problem.h5m
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/simple_thermal_problem.h5m
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/heat_exchange.h5m
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/heat_exchange.h5m
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)


cm_export_file("heat_exchange.h5m" export_files_thermal)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/heat_exchange.h5m DESTINATION ${BFE_MOD_DIR})

cm_export_file("simple_thermal_problem.h5m" export_files_thermal)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/simple_thermal_problem.h5m DESTINATION ${BFE_MOD_DIR})
  
  
if(${GROUND_SURFACE_TEMPERATURE})
  # FIXME: That will not always work if is installed in external directory
  include_directories("${PROJECT_SOURCE_DIR}/ground_surface_temperature/src")
  include_directories("${PROJECT_SOURCE_DIR}/ground_surface_temperature/third_party")
  bfe_build_and_install(thermal_unsteady
    ${CMAKE_CURRENT_BINARY_DIR}/thermal_unsteady.cpp
    ${PROJECT_SOURCE_DIR}/ground_surface_temperature/third_party/spa.c)
else(${GROUND_SURFACE_TEMPERATURE})
  bfe_build_and_install(
    thermal_unsteady ${CMAKE_CURRENT_SOURCE_DIR}/thermal_unsteady.cpp)
endif()
