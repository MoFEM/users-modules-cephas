#

include_directories(${PROJECT_SOURCE_DIR}/basic_finite_elements/src/impl)

# Users modules library, common for all programs
add_library(
  users_modules 
  ${PROJECT_SOURCE_DIR}/basic_finite_elements/src/impl/All.cpp)
target_link_libraries(users_modules PUBLIC ${MoFEM_PROJECT_LIBS})
set_target_properties(users_modules PROPERTIES VERSION ${PROJECT_VERSION})

install(
  TARGETS users_modules 
  DESTINATION lib/basic_finite_elements 
  EXPORT users_modules_targets)
install(
  EXPORT users_modules_targets
  FILE users_modules_targets.cmake
  DESTINATION lib/basic_finite_elements
)

