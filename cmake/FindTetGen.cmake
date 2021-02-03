if(TETGEN_DIR)

  find_library(
    TETGEN_LIBRARY 
    NAMES tet 
    PATHS ${PROJECT_BINARY_DIR}/external/lib
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  find_path(TETGEN_HEADER
    NAMES tetgen.h
    PATHS ${PROJECT_BINARY_DIR}/external/include
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  if(TETGEN_LIBRARY AND TETGEN_HEADER)
    # Check if not installed with mofem in external when stand alone
    message(STATUS ${TETGEN_LIBRARY})
    message(STATUS ${TETGEN_HEADER})
    include_directories(${TETGEN_HEADER})
    add_definitions(-DWITH_TETGEN)
  else(TETGEN_LIBRARY AND TETGEN_HEADER)
    # Check TETGEN_DIR
    find_library(
      TETGEN_LIBRARY 
      NAMES tet 
      PATHS ${TETGEN_DIR}/lib
      NO_DEFAULT_PATH
      NO_CMAKE_ENVIRONMENT_PATH
      NO_CMAKE_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      NO_CMAKE_SYSTEM_PATH
      CMAKE_FIND_ROOT_PATH_BOTH
    )
    find_path(TETGEN_HEADER
      NAMES tetgen.h
      PATHS ${TETGEN_DIR}/include
      NO_DEFAULT_PATH
      NO_CMAKE_ENVIRONMENT_PATH
      NO_CMAKE_PATH
      NO_SYSTEM_ENVIRONMENT_PATH
      NO_CMAKE_SYSTEM_PATH
      CMAKE_FIND_ROOT_PATH_BOTH
    )
    message(STATUS ${TETGEN_LIBRARY})
    message(STATUS ${TETGEN_HEADER})
    if(TETGEN_LIBRARY AND TETGEN_HEADER)
      include_directories(${TETGEN_HEADER})
      add_definitions(-DWITH_TETGEN)
    endif(TETGEN_LIBRARY AND TETGEN_HEADER)
  endif(TETGEN_LIBRARY AND TETGEN_HEADER)

endif(TETGEN_DIR)