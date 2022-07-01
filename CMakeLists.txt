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

cmake_minimum_required (VERSION 3.1.3)

project(MoFEM-UserModules 
  VERSION 0.13.1
  DESCRIPTION "Finite element framework" 
  LANGUAGES C CXX)

# Set fallback version. Fallback version is version of users moudles which is used
# as default version if version from the git tags can not be extracted. For example 
# that is a case when someone install mofem from tarball.
set(UM_FALLBACK_VERSION "0.13.0" CACHE STRING "Fallback version" FORCE)
message(STATUS "Falback version v${UM_FALLBACK_VERSION}")

#search include directories before system directories
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)

# options
option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(NOT_PRECOMPILED_HEADRES "Do not use precompiled headers" OFF)
option(STAND_ALLONE_USERS_MODULES "If if ON copy files, otherwise make link" OFF)
option(EXTERNAL_MODULES_BUILD "If is in build modules only from external folders" OFF)
option(MOFEM_UM_BUILD_TESTS "If is ON enable testing" ON)

# set MPI runtime flags when not set form command line but set in core library
if(NOT MPI_RUN_FLAGS)
  if(MoFEM_MPI_RUN_FLAGS)
    set(MPI_RUN_FLAGS ${MoFEM_MPI_RUN_FLAGS})
  endif(MoFEM_MPI_RUN_FLAGS)
endif(NOT MPI_RUN_FLAGS)

# This is obsolete option should not be used, is set for back compatibility
# with some old D_LIBCPP_DISABLE_DEPRECATION_WARNINGSmodules. You should replace UM_SOURCE_DIR by PROJECT_SOURCE_DIR
# when you spot it in your code.
set(UM_SOURCE_DIR ${PROJECT_SOURCE_DIR})

if(NOT_PRECOMPILED_HEADRES) 
  set(MoFEM_PRECOMPILED_HEADRES OFF CACHE FORCE)
endif(NOT_PRECOMPILED_HEADRES)

# Find MoFEM core package 
message(STATUS "MOFEM directory ${MOFEM_DIR}")
find_package(
  MoFEM REQUIRED HINTS ${MOFEM_DIR} ../ ../../ $ENV{MOFEM_DIR})  
add_definitions(
  -DMoFEM_VERSION_MAJOR=${MoFEM_VERSION_MAJOR} 
  -DMoFEM_VERSION_MINOR=${MoFEM_VERSION_MINOR} 
  -DMoFEM_VERSION_BUILD=${MoFEM_VERSION_BUILD})

add_definitions(${MoFEM_DEFINE_RESTRICT})
add_definitions("-D_LIBCPP_DISABLE_DEPRECATION_WARNINGS")

# Find mbconvert
find_program(
  MBCONVERT NAMES mbconvert
  PATH 
  ${MoFEM_PETSC_DIR}/${MoFEM_PETSC_ARCH}/bin
)

include(ExternalProject)
include(${MoFEM_INSTALL_DIR}/lib/cmake/mofem/ExportFile.cmake)

set(CMAKE_C_COMPILER ${MoFEM_CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MoFEM_CMAKE_CXX_COMPILER})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add options unique to compilation with Spack
include(${MoFEM_INSTALL_DIR}/lib/cmake/mofem/WithSpack.cmake)

#add coverage
option(WITHCOVERAGE "Add gcc coverage compile tags" OFF)
if(WITHCOVERAGE)
  set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -fprofile-arcs -ftest-coverage")
  add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
  set(CMAKE_EXE_LINKER_FLAGS "${GCC_COVERAGE_COMPILE_FLAGS}")
endif(WITHCOVERAGE)

include_directories(
  ${MoFEM_PROJECT_INCLUDE_DIRECTORIES}
  ${MoFEM_PETSC_DIR}/include
  ${MoFEM_PETSC_DIR}/${PETSC_ARCH}/include
  ${MoFEM_PETSC_DIR}/${MoFEM_PETSC_ARCH}/include
  ${MoFEM_PETSCVAR_DIR}/include
  ${MOAB_INCLUDE_DIR})

add_definitions(${MOAB_DEFINITIONS})

include_directories(${MoFEM_INSTALL_DIR}/include)
include_directories(${MoFEM_INSTALL_DIR}/external/include)
include_directories(${MoFEM_INSTALL_DIR}/include/ftensor)
include_directories(${MoFEM_INSTALL_DIR}/include/ftensor/FTensor)
include_directories(${MoFEM_INSTALL_DIR}/include/approximation)
include_directories(${MoFEM_INSTALL_DIR}/include/approximation/c)
include_directories(${MoFEM_INSTALL_DIR}/include/multi_indices)
include_directories(${MoFEM_INSTALL_DIR}/include/interfaces)
include_directories(${MoFEM_INSTALL_DIR}/include/petsc)
include_directories(${MoFEM_INSTALL_DIR}/include/finite_elements)
include_directories(${MoFEM_INSTALL_DIR}/third_party)
include_directories(${MoFEM_INSTALL_DIR}/third_party/cblas)

# boost dir
if(MoFEM_BOOST_INCLUDE_DIR)
  include_directories(${MoFEM_BOOST_INCLUDE_DIR})
endif(MoFEM_BOOST_INCLUDE_DIR)
# cblas 
if(MoFEM_CBLAS_INCLUDE_DIR) 
  include_directories(${MoFEM_CBLAS_INCLUDE_DIR})
endif(MoFEM_CBLAS_INCLUDE_DIR)
add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)

link_directories(${MoFEM_INSTALL_DIR}/lib)

if(MOFEM_UM_BUILD_TESTS)
  enable_testing()
  include(CTest)
endif(MOFEM_UM_BUILD_TESTS)

if(SLEPC_FOUND)
  include_directories("${SLEPC_INCLUDE_DIRS}")
  add_definitions(-DWITH_SLEPC)
  message(STATUS "With SLEPC")
elseif(SLEPC_LIBRARY OR SLEPC_LINK_LIBRARIES)
  # This is depracated adde for back compatibility
  include_directories("${SLEPC_DIR}/${PETSC_ARCH}/include")
  include_directories("${SLEPC_DIR}/include")
  set(SLEPC_FOUND TRUE)
  add_definitions(-DWITH_SLEPC)
  message(STATUS "With SLEPC")
endif()

if(TETGEN_DIR)
  include(${MoFEM_INSTALL_DIR}/include/cmake/FindTetGen.cmake)
  message(STATUS "With TetGen")
endif(TETGEN_DIR)

if(ADOL-C_DIR)
  include(${MoFEM_INSTALL_DIR}/include/cmake/FindADOL-C.cmake)
  message(STATUS "With Adol-C")
endif(ADOL-C_DIR)

if(MED_DIR)
  include(${MoFEM_INSTALL_DIR}/include/cmake/FindMed.cmake)
  message(STATUS "With Med")
endif(MED_DIR)

set(OPTIONAL_PROJECT_LIBS
  ${TETGEN_LIBRARY}
  ${ADOL-C_LIBRARY}
  ${MED_LIBRARY}
)
set(MoFEM_PROJECT_LIBS 
  ${OPTIONAL_PROJECT_LIBS} ${MoFEM_SYSTEM_PROJECT_LIBS})
message(STATUS ${MoFEM_PROJECT_LIBS})

# Add modules
include(ModulesList.cmake)
