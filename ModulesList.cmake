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

# List of sources for user_modules libaries.
set(UM_LIB_SOURCES "")

# Those are default sub-modules
include(${UM_SOURCE_DIR}/basic_finite_elements/AddModule.cmake)

# Users modules library, common for all programs
add_library(users_modules ${UM_LIB_SOURCES})

# Download some known modules (Obsolete)
include(cmake/WithModules.cmake)

file(
  GLOB_RECURSE INSTLLED_MODULES
  FOLLOW_SYMLINKS
  ${PROJECT_SOURCE_DIR}/?*/InstalledAddModule.cmake
)

if(EXTERNAL_MODULE_SOURCE_DIRS)
  foreach(LOOP_DIR ${EXTERNAL_MODULE_SOURCE_DIRS})
    message(STATUS "Search module directory: " ${LOOP_DIR})
    file(
      GLOB_RECURSE EXTERNAL_INSTLLED_MODULES
      FOLLOW_SYMLINKS
      ${LOOP_DIR}/?*/InstalledAddModule.cmake
    )
    message(STATUS "Found: " ${EXTERNAL_INSTLLED_MODULES})
    set(INSTLLED_MODULES ${INSTLLED_MODULES} ${EXTERNAL_INSTLLED_MODULES})
  endforeach(LOOP_DIR)
endif(EXTERNAL_MODULE_SOURCE_DIRS)

# Install modules && git pull for all users modules
add_custom_target(
  update_users_modules
  COMMENT "Update all modules ..." VERBATIM
)
add_custom_target(
  checkout_CDashTesting
  COMMENT "Checkout CDashTesting branch ..." VERBATIM
)
add_custom_target(
  checkout_develop
  COMMENT "Checkout develop branch ..." VERBATIM
)
add_custom_target(
  checkout_master
  COMMENT "Checkout master branch ..." VERBATIM
)

# Recognise that module is installed
foreach(LOOP_MODULE ${INSTLLED_MODULES})
  string(REGEX REPLACE
    "/+InstalledAddModule.cmake" ""
    MODULE_DIRECTORY ${LOOP_MODULE}
  )
  string(REGEX REPLACE
    ".*/+" ""
    MODULE_NAME ${MODULE_DIRECTORY}
  )
  string(TOUPPER ${MODULE_NAME} MODULE_NAME)
  message(STATUS "Add definitions to the compiler command -DWITH_MODULE_${MODULE_NAME}")
  add_definitions(-DWITH_MODULE_${MODULE_NAME})
endforeach(LOOP_MODULE)

# Add custom tags for all modules
foreach(LOOP_MODULE ${INSTLLED_MODULES})
  # message(STATUS "${LOOP_MODULE}")
  string(REGEX REPLACE
    "/+InstalledAddModule.cmake" ""
    MODULE_DIRECTORY ${LOOP_MODULE}
  )
  message(STATUS "Add module ... ${MODULE_DIRECTORY}")
  string(REGEX REPLACE
    ".*/+" ""
    MODULE_NAME ${MODULE_DIRECTORY}
  )
  message(STATUS "Add custom targets for ${MODULE_NAME}")
  add_custom_target(
    update_${MODULE_NAME}
    COMMAND ${GIT_EXECUTABLE} pull
    WORKING_DIRECTORY ${MODULE_DIRECTORY}
    COMMENT "Update module ... ${MODULE_NAME}" VERBATIM
  )
  add_dependencies(update_users_modules update_${MODULE_NAME})
  add_custom_target(
    checkout_CDashTesting_${MODULE_NAME}
    COMMAND ${GIT_EXECUTABLE} checkout CDashTesting
    WORKING_DIRECTORY ${MODULE_DIRECTORY}
    COMMENT "Checkout CDashTesting baranch for module ${MODULE_NAME}" VERBATIM
  )
  add_dependencies(checkout_CDashTesting checkout_CDashTesting_${MODULE_NAME})
  add_custom_target(
    checkout_develop_${MODULE_NAME}
    COMMAND ${GIT_EXECUTABLE} checkout develop
    WORKING_DIRECTORY ${MODULE_DIRECTORY}
    COMMENT "Checkout develop baranch for module ${MODULE_NAME}" VERBATIM
  )
  add_dependencies(checkout_develop checkout_develop_${MODULE_NAME})
  add_custom_target(
    checkout_master_${MODULE_NAME}
    COMMAND ${GIT_EXECUTABLE} checkout master
    WORKING_DIRECTORY ${MODULE_DIRECTORY}
    COMMENT "Checkout master baranch for module ${MODULE_NAME}" VERBATIM
  )
  add_dependencies(checkout_master checkout_master_${MODULE_NAME})
  # include module
  include(${LOOP_MODULE})
endforeach(LOOP_MODULE)
