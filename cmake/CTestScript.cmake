set(CTEST_PROJECT_NAME "MoFEM-UsersModules")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
set(CTEST_BUILD_CONFIGURATION "Debug")

ctest_empty_binary_directory(${CTEST_BINARY_DIRECTORY})

find_program(CTEST_COVERAGE_COMMAND NAMES gcov)

set(CTEST_CONFIGURE_COMMAND "\"${CMAKE_COMMAND}\" -DCMAKE_BUILD_TYPE:STRING=${CTEST_BUILD_CONFIGURATION}")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} ${CTEST_BUILD_OPTIONS} -DWITHCOVERAGE=ON")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"-G${CTEST_CMAKE_GENERATOR}\"")
set(CTEST_CONFIGURE_COMMAND "${CTEST_CONFIGURE_COMMAND} \"${CTEST_SOURCE_DIRECTORY}\"")

#Ctest time outr
set(CTEST_TEST_TIMEOUT 1200)

# Perform the CDashTesting
ctest_start(${DASHBOARDTEST})

set(CTEST_CUSTOM_MEMCHECK_IGNORE
  ${CTEST_CUSTOM_MEMCHECK_IGNORE}
  #compare
)

ctest_configure()
ctest_build(TARGET checkout_CDashTesting)
ctest_build(TARGET update_users_modules)
ctest_build()
ctest_test()
if(CTEST_COVERAGE_COMMAND)
  ctest_coverage(QUIET)
endif(CTEST_COVERAGE_COMMAND)
ctest_submit()
