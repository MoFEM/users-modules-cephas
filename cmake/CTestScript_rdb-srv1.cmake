set(
  CTEST_BUILD_OPTIONS
  "-DSTAND_ALLONE_USERS_MODULES=1 -DWITH_MODULE_OBSOLETE=1 -DWITH_MODULE_HOMOGENISATION=1 -DWITH_MODULE_FRACTURE_MECHANICS=1 -DWITH_MODULE_GELS=1 -DWITH_MODULE_STRAIN_PLASTICITY=1 -DWITH_MODULE_SOLID_SHELL_PRISM_ELEMENT=1 -DWITH_MODULE_MINIMAL_SURFACE_EQUATION=1 -DWITH_METAIO=1  -DWITH_MODULE_BONE_REMODELLING=1 /home/lukasz/tmp/cephas_users_modules/users_modules"
)

set(CTEST_SITE "rdb-srv1")
set(CTEST_BUILD_NAME "Linux-mpicxx")

if(NOT DASHBOARDTEST)
  set(DASHBOARDTEST "Continuous")
endif(NOT DASHBOARDTEST)

# modules check out
exec_program(
  ${CTEST_GIT_COMMAND}
  "${CTEST_SOURCE_DIRECTORY}/users_modules/fracture_mechanics"
  ARGS checkout CDashTesting
)
exec_program(
  ${CTEST_GIT_COMMAND}
  "${CTEST_SOURCE_DIRECTORY}/users_modules/obsolete"
  ARGS checkout CDashTesting
)
exec_program(
  ${CTEST_GIT_COMMAND}
  "${CTEST_SOURCE_DIRECTORY}/users_modules/homogenisation"
  ARGS checkout CDashTesting
)
exec_program(
  ${CTEST_GIT_COMMAND}
  "${CTEST_SOURCE_DIRECTORY}/users_modules/gels"
  ARGS checkout CDashTesting
)
exec_program(
  ${CTEST_GIT_COMMAND}
  "${CTEST_SOURCE_DIRECTORY}/users_modules/small_strain_plasticity"
  ARGS checkout CDashTesting
)
exec_program(
  ${CTEST_GIT_COMMAND}
  "${CTEST_SOURCE_DIRECTORY}/users_modules/solid_shell_prism_element"
  ARGS checkout CDashTesting
)
exec_program(
  ${CTEST_GIT_COMMAND}
  "${CTEST_SOURCE_DIRECTORY}/users_modules/minimal_surface_equation"
  ARGS checkout CDashTesting
)
exec_program(
  ${CTEST_GIT_COMMAND}
  "${CTEST_SOURCE_DIRECTORY}/users_modules/bone_remodelling"
  ARGS checkout CDashTesting
)

# modules - moisture_transport
if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/users_modules/moisture_transport")
  exec_program(
    ${CTEST_GIT_COMMAND}
    "${CTEST_SOURCE_DIRECTORY}/users_modules"
    ARGS clone https://likask@bitbucket.org/likask/mofem_um_moisture_transport.git
    "${CTEST_SOURCE_DIRECTORY}/users_modules/moisture_transport"
  )
  exec_program(
    ${CTEST_GIT_COMMAND}
    "${CTEST_SOURCE_DIRECTORY}/users_modules/moisture_transport"
    ARGS checkout CDashTesting
  )
else(EXISTS "${CTEST_SOURCE_DIRECTORY}/users_modules/moisture_transport")
  exec_program(
    ${CTEST_GIT_COMMAND}
    "${CTEST_SOURCE_DIRECTORY}/users_modules/moisture_transport"
    ARGS pull
  )
endif()

# modules - ground_surface_temperature
if(NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/users_modules/ground_surface_temperature")
  exec_program(
    ${CTEST_GIT_COMMAND}
    "${CTEST_SOURCE_DIRECTORY}/users_modules"
    ARGS clone https://likask@bitbucket.org/likask/mofem_um_ground_surface_temperature.git
    "${CTEST_SOURCE_DIRECTORY}/users_modules/ground_surface_temperature"
  )
  exec_program(
    ${CTEST_GIT_COMMAND}
    "${CTEST_SOURCE_DIRECTORY}/users_modules/ground_surface_temperature"
    ARGS checkout CDashTesting
  )
else(EXISTS "${CTEST_SOURCE_DIRECTORY}/users_modules/ground_surface_temperature")
  exec_program(
    ${CTEST_GIT_COMMAND}
    "${CTEST_SOURCE_DIRECTORY}/users_modules/ground_surface_temperature"
    ARGS pull
  )
endif()

set(CTEST_SOURCE_DIRECTORY "/home/lukasz/tmp/cephas_users_modules/users_modules")
set(CTEST_BINARY_DIRECTORY "/home/lukasz/tmp/cephas_users_modules/build")

include(CTestScript.cmake)
