#

# Load precompiled headers to the speed-up compilation process.
include(cmake/PrecompileHeaders.cmake)

# Get user module commit id and pass to cpp for logging
include(${MoFEM_INSTALL_DIR}/include/cmake/GetGitRevisionSimple.cmake)

# If the basic UMs are already loaded target to um library. Otherwise, build a
# basic user modules library.
if(EXTERNAL_MODULES_BUILD)

  find_package(
     UM REQUIRED HINTS ${MOFEM_DIR} ${PROJECT_BINARY_DIR} $ENV{MOFEM_DIR})  

  if(UM_INSTALL_PREFIX)
    include(
      ${UM_INSTALL_PREFIX}/lib/basic_finite_elements/users_modules_targets.cmake)
  else(UM_INSTALL_PREFIX)
    include(
      ${MoFEM_INSTALL_DIR}/lib/basic_finite_elements/users_modules_targets.cmake)
  endif(UM_INSTALL_PREFIX)

else(EXTERNAL_MODULES_BUILD)

  get_git_hash(${PROJECT_SOURCE_DIR} GIT_UM_SHA1)
  get_git_tag(${PROJECT_SOURCE_DIR} "v${UM_FALLBACK_VERSION}" GIT_VERSION)
  get_git_version(
    ${GIT_VERSION} UM_VERSION_MAJOR UM_VERSION_MINOR UM_VERSION_BUILD)

  # Find modules in a users modules source directory. If file
  # InstalledAddModule.cmake is found in the directory, it is recognised that
  # directory is module and name of directory become module name.
  file(
    GLOB_RECURSE INSTLLED_MODULES
    FOLLOW_SYMLINKS
    ${PROJECT_SOURCE_DIR}/?*/InstalledAddModule.cmake)
  # Compile tools
  add_subdirectory(${PROJECT_SOURCE_DIR}/tools)
endif(EXTERNAL_MODULES_BUILD)

message(STATUS "MoFEM version ${MoFEM_VERSION}")
set(UM_VERSON 
  ${UM_VERSION_MAJOR}.${UM_VERSION_MINOR}.${UM_VERSION_BUILD} 
  CACHE STRING "UM version" FORCE)
message(STATUS "User Module commit id: " ${GIT_UM_SHA1})
message(STATUS "UM GIT_TAG ${GIT_VERSION}")
message(STATUS "UM Version v${UM_VERSON}")

add_definitions(-DUM_VERSION_MAJOR="${UM_VERSION_MAJOR}")
add_definitions(-DUM_VERSION_MINOR="${UM_VERSION_MINOR}")
add_definitions(-DUM_VERSION_BUILD="${UM_VERSION_BUILD}")
add_definitions(-DGIT_UM_SHA1_NAME="${GIT_UM_SHA1}")

include(cmake/CheckMoFEMVersion.cmake)

# Find modules in an external directory. If file InstalledAddModule.cmake if
# found in the directory, it is recognised that director is module and name of
# directory become module name.
if(EXTERNAL_MODULE_SOURCE_DIRS)
  foreach(LOOP_DIR ${EXTERNAL_MODULE_SOURCE_DIRS})
    message(STATUS "Search module directory: " ${LOOP_DIR})
    file(
      GLOB_RECURSE EXTERNAL_INSTLLED_MODULES
      FOLLOW_SYMLINKS
      ${LOOP_DIR}/*InstalledAddModule.cmake)
    message(STATUS "Found: " ${EXTERNAL_INSTLLED_MODULES})
    set(INSTLLED_MODULES ${INSTLLED_MODULES} ${EXTERNAL_INSTLLED_MODULES})
  endforeach(LOOP_DIR)
endif(EXTERNAL_MODULE_SOURCE_DIRS)

# Install modules && git pull for all users modules
add_custom_target(
  update_users_modules
  COMMENT "Update all modules ..." VERBATIM)
add_custom_target(
  checkout_develop
  COMMENT "Checkout develop branch ..." VERBATIM)
add_custom_target(
  checkout_master
  COMMENT "Checkout master branch ..." VERBATIM)

# Recognise that module is installed
foreach(LOOP_MODULE ${INSTLLED_MODULES})
  string(REGEX REPLACE
    "/+InstalledAddModule.cmake" ""
    MODULE_DIRECTORY ${LOOP_MODULE})
  string(REGEX REPLACE
    ".*/+" ""
    MODULE_NAME ${MODULE_DIRECTORY})
  string(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME}) 
  string(TOUPPER ${MODULE_NAME} MODULE_NAME)
  message(STATUS "Add definitions to the compiler command -DWITH_MODULE_${MODULE_NAME}")
  add_definitions(-DWITH_MODULE_${MODULE_NAME})
endforeach(LOOP_MODULE)

# Add custom tags for all modules
foreach(LOOP_MODULE ${INSTLLED_MODULES})
  # message(STATUS "${LOOP_MODULE}")
  string(REGEX REPLACE
    "/+InstalledAddModule.cmake" ""
    MODULE_DIRECTORY ${LOOP_MODULE})
  message(STATUS "Add module ... ${MODULE_DIRECTORY}")
  string(REGEX REPLACE
    ".*/+" ""
    MODULE_NAME ${MODULE_DIRECTORY}
  )
  if(
    NOT ${MODULE_NAME} MATCHES "basic_finite_elements" 
    AND 
    NOT ${MODULE_NAME} MATCHES "tutorials")
    #message(STATUS "Add custom targets for ${MODULE_NAME}")
    add_custom_target(
      update_${MODULE_NAME}
      COMMAND ${GIT_EXECUTABLE} pull
      WORKING_DIRECTORY ${MODULE_DIRECTORY}
      COMMENT "Update module ... ${MODULE_NAME}" VERBATIM)
    add_dependencies(update_users_modules update_${MODULE_NAME})
    add_custom_target(
      checkout_develop_${MODULE_NAME}
      COMMAND ${GIT_EXECUTABLE} checkout develop
      WORKING_DIRECTORY ${MODULE_DIRECTORY}
      COMMENT "Checkout develop baranch for module ${MODULE_NAME}" VERBATIM)
    add_dependencies(checkout_develop checkout_develop_${MODULE_NAME})
    add_custom_target(
      checkout_master_${MODULE_NAME}
      COMMAND ${GIT_EXECUTABLE} checkout master
      WORKING_DIRECTORY ${MODULE_DIRECTORY}
      COMMENT "Checkout master baranch for module ${MODULE_NAME}" VERBATIM)
    add_dependencies(checkout_master checkout_master_${MODULE_NAME})
  endif(
    NOT ${MODULE_NAME} MATCHES "basic_finite_elements" 
    AND
    NOT ${MODULE_NAME} MATCHES "tutorials")
  # include module
  include(${LOOP_MODULE})
endforeach(LOOP_MODULE)

if(NOT EXTERNAL_MODULES_BUILD)

  # Generate config files
  configure_file(
    ${PROJECT_SOURCE_DIR}/UMConfig-version.cmake.in
    ${PROJECT_BINARY_DIR}/UMConfig-version.cmake)
  configure_file(
    ${PROJECT_SOURCE_DIR}/UMConfig.cmake.in
    ${PROJECT_BINARY_DIR}/UMConfig.cmake)
  # Install project files
  install(
    FILES ${PROJECT_BINARY_DIR}/UMConfig.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX})
  install(
    FILES ${PROJECT_BINARY_DIR}/UMConfig-version.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX})

endif(NOT EXTERNAL_MODULES_BUILD)