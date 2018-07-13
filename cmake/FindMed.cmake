# - Try to find MED

if(MED_DIR)

  find_library(
    MED_LIBRARY 
    NAMES med medC
    PATHS ${PROJECT_BINARY_DIR}/external/lib
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  find_path(MED_HEADER
    NAMES med.h
    PATHS ${PROJECT_BINARY_DIR}/external/include
    NO_DEFAULT_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_SYSTEM_PATH
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  if(MED_LIBRARY AND MED_HEADER)
    # Check if not installed with mofem in external when stand alone
    message(STATUS ${MED_LIBRARY})
    message(STATUS ${MED_HEADER})
    include_directories(${MED_HEADER})
    add_definitions(-DWITH_MED)
  else(MED_LIBRARY AND MED_HEADER)
    # Check MED_DIR
    find_library(
      MED_LIBRARY 
      NAMES med medC
      PATHS ${MED_DIR}/lib
    )
    find_path(MED_HEADER
      NAMES med.h
      PATHS ${MED_DIR}/include
    )
    message(STATUS ${MED_LIBRARY})
    message(STATUS ${MED_HEADER})
    if(MED_LIBRARY AND MED_HEADER)
      include_directories(${MED_HEADER})
      add_definitions(-DWITH_MED)
    endif(MED_LIBRARY AND MED_HEADER)
  endif(MED_LIBRARY AND MED_HEADER)

endif(MED_DIR)