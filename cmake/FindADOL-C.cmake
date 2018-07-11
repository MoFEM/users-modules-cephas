# - Try to find ADOL-C
# Once done this will define
#
#  ADOL-C_DIR - directory in which ADOL-C resides

if(ADOL-C_DIR)

  find_library(
    ADOL-C_LIBRARY
    NAMES adolc
    PATHS 
    ${PROJECT_BINARY_DIR}/external/lib
    ${PROJECT_BINARY_DIR}/external/lib64
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  find_path(ADOL-C_HEADER
    NAMES adolc/adolc.h
    PATHS 
    ${PROJECT_BINARY_DIR}/external/include
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  if(ADOL-C_LIBRARY AND ADOL-C_HEADER)
    set(ADOL-C_DIR ${PROJECT_BINARY_DIR}/external)
    find_library(
      COLPACK_LIBLARY 
      NAMES Colpack 
      PATHS ${ADOL-C_DIR}/lib ${ADOL-C_DIR}/lib64 /usr/local/lib)
    if(COLPACK_LIBLARY)
      set(ADOL-C_LIBRARY ${ADOL-C_LIBRARY} ${COLPACK_LIBLARY})
    endif(COLPACK_LIBLARY)
    include_directories(${ADOL-C_HEADER})
    add_definitions(-DWITH_ADOL_C)
    message(STATUS ${ADOL-C_LIBRARY})
    message(STATUS ${ADOL-C_HEADER})
  else(ADOL-C_LIBRARY AND ADOL-C_HEADER)
    find_library(ADOL-C_LIBRARY NAMES adolc PATHS ${ADOL-C_DIR}/lib ${ADOL-C_DIR}/lib64)
    find_path(ADOL-C_HEADER NAMES adolc/adolc.h PATHS ${ADOL-C_DIR}/include)
    if(ADOL-C_LIBRARY AND ADOL-C_HEADER)
      find_library(
        COLPACK_LIBLARY 
        NAMES Colpack 
        PATHS ${ADOL-C_DIR}/lib ${ADOL-C_DIR}/lib64 /usr/local/lib)
      if(COLPACK_LIBLARY)
        set(ADOL-C_LIBRARY ${ADOL-C_LIBRARY} ${COLPACK_LIBLARY})
      endif(COLPACK_LIBLARY)
      include_directories(${ADOL-C_HEADER})
      add_definitions(-DWITH_ADOL_C)
    endif(ADOL-C_LIBRARY AND ADOL-C_HEADER)
    message(STATUS ${ADOL-C_LIBRARY})
    message(STATUS ${ADOL-C_HEADER})
  endif(ADOL-C_LIBRARY AND ADOL-C_HEADER)

endif(ADOL-C_DIR)