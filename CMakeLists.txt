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

project(MoFEM-UserModules C CXX)

#search include directories before system directories
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)

# options
option(NOT_PRECOMPILED_HEADRES "Do not use procompiled headers" OFF)
option(STAND_ALLONE_USERS_MODULES "If if ON copy files, othrewise make link" OFF)
option(EXTERNAL_MODULES_BUILD "If is in build modules only from external folders" OFF)
option(MOFEM_UM_BUILD_TETS "If is ON enbale testing" ON)

# This is obsolete option should not be used, is set for back comaptibility
# with some old modules. You should replace UM_SOURCE_DIR by PROJECT_SOURCE_DIR
# when you spot it in your code.
set(UM_SOURCE_DIR ${PROJECT_SOURCE_DIR})

if(NOT_PRECOMPILED_HEADRES) 
  set(MoFEM_PRECOMPILED_HEADRES OFF CACHE FORCE)
endif(NOT_PRECOMPILED_HEADRES)

include(cmake/GetGitRevisionDescription.cmake)

if(STAND_ALLONE_USERS_MODULES)
  find_package(
    MoFEM REQUIRED HINTS ${MOFEM_DIR} ../ ../../ $ENV{MOFEM_DIR}
  )  
else(STAND_ALLONE_USERS_MODULES)
  git_describe(GIT_VERSION "--tags")
  if(GIT_VERSION)
    string(
      REGEX 
      REPLACE "^v([0-9]+)\\..*" "\\1" GIT_MoFEM_VERSION_MAJOR "${GIT_VERSION}"
    )
    string(
      REGEX 
      REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" GIT_MoFEM_VERSION_MINOR "${GIT_VERSION}"
    )
    string(
      REGEX 
      REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" GIT_MoFEM_VERSION_BUILD "${GIT_VERSION}"
    )
    find_package(
      MoFEM 
      ${GIT_MoFEM_VERSION_MAJOR}.${GIT_MoFEM_VERSION_MINOR}.${GIT_MoFEM_VERSION_MINOR}
      REQUIRED HINTS ${MOFEM_DIR} ../ ../../ $ENV{MOFEM_DIR}
    )  
  else(GIT_VERSION)
    find_package(
      MoFEM REQUIRED HINTS ${MOFEM_DIR} ../ ../../ $ENV{MOFEM_DIR}
    )  
  endif(GIT_VERSION)
endif(STAND_ALLONE_USERS_MODULES)
include(cmake/CheckMoFEMVersion.cmake)

message(STATUS "MoFEM version ${MoFEM_VERSION}")
add_definitions(${MoFEM_DEFINE_RESTRICT})

find_package(Git)
find_program(
  MBCONVERT NAMES mbconvert
  PATH 
  ${MoFEM_PETSC_DIR}/${MoFEM_PETSC_ARCH}/bin
)

include(ExternalProject)
include(${MoFEM_INSTALL_DIR}/lib/cmake/mofem/ExportFile.cmake)

set(CMAKE_C_COMPILER ${MoFEM_CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MoFEM_CMAKE_CXX_COMPILER})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add options unique to compiliation with Spack
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
)

add_definitions(${MOAB_DEFINITIONS})

include_directories(${MoFEM_INSTALL_DIR}/include)
include_directories(${MoFEM_INSTALL_DIR}/external/include)
if(STAND_ALLONE_USERS_MODULES)
  include_directories(${MoFEM_INSTALL_DIR}/third_party)
  include_directories(${MoFEM_INSTALL_DIR}/third_party/cblas)
  include_directories(${MoFEM_INSTALL_DIR}/include/ftensor)
  include_directories(${MoFEM_INSTALL_DIR}/include/approximation)
  include_directories(${MoFEM_INSTALL_DIR}/include/approximation/c)
  include_directories(${MoFEM_INSTALL_DIR}/include/multi_indices)
  include_directories(${MoFEM_INSTALL_DIR}/include/interfaces)
  include_directories(${MoFEM_INSTALL_DIR}/include/petsc)
  include_directories(${MoFEM_INSTALL_DIR}/include/finite_elements)
else(STAND_ALLONE_USERS_MODULES)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/include)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party/cblas)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/src/ftensor/src)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/src/approximation)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/src/approximation/c)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/src/multi_indices)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/src/interfaces)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/src/petsc)
  include_directories(${MoFEM_PROJECT_SOURCE_DIR}/src/finite_elements)
endif(STAND_ALLONE_USERS_MODULES)

# boost dir
if(MoFEM_BOOST_INCLUDE_DIR)
  include_directories(${MoFEM_BOOST_INCLUDE_DIR})
endif(MoFEM_BOOST_INCLUDE_DIR)
# cblas 
if(MoFEM_CBLAS_INCLUDE_DIR) 
  include_directories(${MoFEM_CBLAS_INCLUDE_DIR})
endif(MoFEM_CBLAS_INCLUDE_DIR)

link_directories(${MoFEM_INSTALL_DIR}/lib)

if(MOFEM_UM_BUILD_TETS)
  enable_testing()
  include(CTest)
endif(MOFEM_UM_BUILD_TETS)

if(SLEPC_LIBRARY)
  include_directories("${SLEPC_DIR}/${PETSC_ARCH}/include")
  include_directories("${SLEPC_DIR}/include")
  add_definitions( -DWITH_SLEPC_DIR )
  message(STATUS "WITH SLEPC")
endif(SLEPC_LIBRARY)

if(TETGEN_DIR)
  include(cmake/FindTetGen.cmake)
endif(TETGEN_DIR)

if(ADOL-C_DIR)
  include(cmake/FindADOL-C.cmake)
endif(ADOL-C_DIR)

if(MED_DIR)
  include(cmake/FindMed.cmake)
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
