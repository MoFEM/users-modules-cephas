
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vec-0/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vec-2/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../adv-0/src)

set(permissions_default 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ)

set(permissions_execute 
  OWNER_WRITE 
  OWNER_READ
  GROUP_READ
  OWNER_EXECUTE 
  GROUP_EXECUTE)

tutorials_copy_and_install("*.msh" "${permissions_default}")
tutorials_copy_and_install("*.cfg" "${permissions_default}")
# tutorials_copy_and_install("*.jou" "${permissions_default}")
tutorials_copy_and_install("*.cub" "${permissions_default}")
tutorials_copy_and_install("*.h5m" "${permissions_default}")
tutorials_copy_and_install("*.petsc" "${permissions_default}")
tutorials_copy_and_install("*.sh" "${permissions_execute}")
tutorials_copy_and_install("*.geo" "${permissions_execute}")
tutorials_copy_and_install("*.py" "${permissions_execute}")

if(Python_LIBRARY AND Boost_PYTHON_LIBRARY_RELEASE) 

  include_directories(${_Python_INCLUDE_DIR})

  tutorials_build_and_install(
    contact_2d ${CMAKE_CURRENT_SOURCE_DIR}/contact.cpp)
  tutorials_build_and_install(
    contact_3d ${CMAKE_CURRENT_SOURCE_DIR}/contact.cpp)

  set_target_properties(
    contact_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2 -DPYTHON_SFD") 
  set_target_properties(
    contact_3d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3 -DPYTHON_SFD") 

  target_link_libraries(contact_2d 
    ${Python_LIBRARY} ${Boost_LIBRARIES} ${Boost_PYTHON_LIBRARY_RELEASE})
  target_link_libraries(contact_3d 
    ${Python_LIBRARY} ${Boost_LIBRARIES} ${Boost_PYTHON_LIBRARY_RELEASE})

else (Python_LIBRARY AND Boost_PYTHON_LIBRARY_RELEASE)

  tutorials_build_and_install(
    contact_2d ${CMAKE_CURRENT_SOURCE_DIR}/contact.cpp)
  tutorials_build_and_install(
    contact_3d ${CMAKE_CURRENT_SOURCE_DIR}/contact.cpp)

  set_target_properties(
    contact_2d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=2")
  set_target_properties(
    contact_3d PROPERTIES COMPILE_FLAGS "-DEXECUTABLE_DIMENSION=3")  

endif(Python_LIBRARY AND Boost_PYTHON_LIBRARY_RELEASE)

add_test(
  contact_2d
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1
  contact_2d
  -file_name ${CMAKE_CURRENT_BINARY_DIR}/sphere_2d.h5m
  -ts_type beuler
  -ts_max_time 1 -ts_dt 0.2 -log_no_color)
  