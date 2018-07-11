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
# FIXME: This has to find solution as a install with sub-modules
include(${UM_SOURCE_DIR}/basic_finite_elements/AddModule.cmake)

# Users modules library, common for all programs
add_library(users_modules ${UM_LIB_SOURCES})

# Those modules could be downloaded with MoFEM.
# FIXME: Do not delete this, it is  used with code testing. That will be renoved
# for testing with docker container.
if(WITH_MODULE_OBSOLETE)
  if(NOT EXISTS ${UM_SOURCE_DIR}/obsolete)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_obsolete.git obsolete
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/obsolete)
endif(WITH_MODULE_OBSOLETE)

if(WITH_MODULE_HOMOGENISATION)
  if(NOT EXISTS ${UM_SOURCE_DIR}/homogenisation)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_homogenisation homogenisation
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/homogenisation)
endif(WITH_MODULE_HOMOGENISATION)

if(WITH_MODULE_BONE_REMODELLING)
  if(NOT EXISTS ${UM_SOURCE_DIR}/bone_remodelling)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_bone_remodelling.git bone_remodelling
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/bone_remodelling)
endif(WITH_MODULE_BONE_REMODELLING)

 if(WITH_MODULE_MWLS_APPROX)
  if(NOT EXISTS ${UM_SOURCE_DIR}/mwls_approx)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone git clone https://bitbucket.org/karol41/mofem_um_mwls.git mwls_approx
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/mwls_approx)
endif(WITH_MODULE_MWLS_APPROX)

if(WITH_MODULE_FRACTURE_MECHANICS)
  if(NOT EXISTS ${UM_SOURCE_DIR}/fracture_mechanics)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_fracture_mechanics fracture_mechanics
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/fracture_mechanics)
endif(WITH_MODULE_FRACTURE_MECHANICS)

if(WITH_MODULE_GELS)
  if(NOT EXISTS ${UM_SOURCE_DIR}/gels)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_gels gels
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/gels)
endif(WITH_MODULE_GELS)

if(WITH_MODULE_STRAIN_PLASTICITY)
  if(NOT EXISTS ${UM_SOURCE_DIR}/small_strain_plasticity)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_small_strain_plasticity small_strain_plasticity
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/small_strain_plasticity)
endif(WITH_MODULE_STRAIN_PLASTICITY)

if(WITH_MODULE_SOLID_SHELL_PRISM_ELEMENT)
  if(NOT EXISTS ${UM_SOURCE_DIR}/solid_shell_prism_element)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_solid_shell_prism_element solid_shell_prism_element
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/solid_shell_prism_element)
endif(WITH_MODULE_SOLID_SHELL_PRISM_ELEMENT)

if(WITH_MODULE_MINIMAL_SURFACE_EQUATION)
  if(NOT EXISTS ${UM_SOURCE_DIR}/minimal_surface_equation)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_minimal_surface_equation minimal_surface_equation
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/minimal_surface_equation)
endif(WITH_MODULE_MINIMAL_SURFACE_EQUATION)

if(WITH_MODULE_HELMHOLTZ)
  if(NOT EXISTS ${UM_SOURCE_DIR}/helmholtz)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone https://bitbucket.org/likask/mofem_um_helmholtz helmholtz
      WORKING_DIRECTORY ${UM_SOURCE_DIR}
    )
  endif(NOT EXISTS ${UM_SOURCE_DIR}/helmholtz)
endif(WITH_MODULE_HELMHOLTZ)

file(
  GLOB_RECURSE INSTLLED_MODULES
  FOLLOW_SYMLINKS
  ?*/InstalledAddModule.cmake
)

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
# add_custom_target(
#   merge_CDashTesting
#   COMMENT "Make merge CDashTesting branch ..." VERBATIM
# )
add_custom_target(
  get_modules_contributors_init
  COMMAND echo "Modules contributors" > ${PROJECT_SOURCE_DIR}/doc/contributors_list_modules
  COMMENT "Init file for modules contributors list ..." VERBATIM
)
add_custom_target(
  get_modules_contributors
  COMMENT "Get module contributors ..." VERBATIM
)
add_dependencies(get_modules_contributors get_modules_contributors_init)

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
  # add_custom_target(
  #   merge_CDashTesting_${MODULE_NAME}
  #   COMMAND ${GIT_EXECUTABLE} merge --ff CDashTesting
  #   WORKING_DIRECTORY ${MODULE_DIRECTORY}
  #   COMMENT "Make merge CDashTesting branch for module ${MODULE_NAME}" VERBATIM
  # )
  # add_dependencies(merge_CDashTesting merge_CDashTesting_${MODULE_NAME})
  # get contributors
  add_custom_target(
    get_module_contributors_${MODULE_NAME}
    COMMAND echo >> ${PROJECT_SOURCE_DIR}/doc/contributors_list_modules
    COMMAND echo ${MODULE_NAME} >> ${PROJECT_SOURCE_DIR}/doc/contributors_list_modules
    COMMAND ${GIT_EXECUTABLE} shortlog -s -e -n | sed "s/\@/ at /g" >> ${PROJECT_SOURCE_DIR}/doc/contributors_list_modules
    WORKING_DIRECTORY ${MODULE_DIRECTORY}
    COMMENT "Get module contributors ${MODULE_NAME}" VERBATIM
  )
  add_dependencies(get_module_contributors_${MODULE_NAME} get_modules_contributors_init)
  add_dependencies(get_modules_contributors get_module_contributors_${MODULE_NAME})
  # include module
  include(${LOOP_MODULE})
endforeach(LOOP_MODULE)
