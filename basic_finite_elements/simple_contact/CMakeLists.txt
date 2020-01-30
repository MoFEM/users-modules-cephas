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


include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party)
include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party/phg-quadrule)
include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party/cblas)
include_directories(${MoFEM_PROJECT_SOURCE_DIR}/third_party)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../elasticity/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elasticity)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nonlinear_elastic_materials/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE)  

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README
  DESTINATION ${BFE_MOD_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE)

bfe_build_and_install(simple_contact
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_contact.cpp)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
    
install(
  DIRECTORY examples
  DESTINATION ${BFE_MOD_DIR}
)
  
add_test(
  test_simple_contact_8cube
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_is_test
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/8cube_contact.cub 
  -my_order 2
  -my_cn_value 1e3
)
add_test(
  test_simple_contact_8cube_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/test_simple_contact_8cube.txt
  ${CMAKE_CURRENT_BINARY_DIR}/test_simple_contact.txt
)

add_test(
  test_simple_contact_4seasons
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_is_test
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/4seasons_contact.cub
  -my_order 2
  -my_cn_value 1e3
)
add_test(
  test_simple_contact_4seasons_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/test_simple_contact_4seasons.txt
  ${CMAKE_CURRENT_BINARY_DIR}/test_simple_contact.txt
)

add_test(
  test_simple_contact_Tinterface
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_is_test
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/Tinterface_contact.cub
  -my_order 2
  -my_cn_value 1e3
)
add_test(
  test_simple_contact_Tinterface_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/test_simple_contact_Tinterface.txt
  ${CMAKE_CURRENT_BINARY_DIR}/test_simple_contact.txt
)

add_test(
  test_simple_contact_punch_top_and_mid
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_is_test
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/punch_top_and_mid.cub
  -my_order 2
  -my_cn_value 1e3
)
add_test(
  test_simple_contact_punch_top_and_mid_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/test_simple_contact_punch_top_and_mid.txt
  ${CMAKE_CURRENT_BINARY_DIR}/test_simple_contact.txt
)

add_test(
  test_simple_contact_punch_top_only
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_is_test
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/punch_top_only.cub
  -my_order 2
  -my_cn_value 1e3
)
add_test(
  test_simple_contact_punch_top_only_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/test_simple_contact_punch_top_only.txt
  ${CMAKE_CURRENT_BINARY_DIR}/test_simple_contact.txt
)

add_test(
  test_simple_contact_plane_axisymmetric
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_is_test
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/plane_axi_contact.cub
  -my_order 2
  -my_cn_value 1e3
)
add_test(
  test_simple_contact_plane_axisymmetric_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/test_simple_contact_plane_axi.txt
  ${CMAKE_CURRENT_BINARY_DIR}/test_simple_contact.txt
)

add_test(
  test_simple_contact_arc_3surfaces
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_is_test
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/arc_3surf_contact.cub
  -my_order 2
  -my_cn_value 1e3
)
add_test(
  test_simple_contact_arc_3surfaces_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/test_simple_contact_arc_3surf.txt
  ${CMAKE_CURRENT_BINARY_DIR}/test_simple_contact.txt
)

add_test(
  test_simple_contact_smiling_face
  ${MoFEM_MPI_RUN} ${MPI_RUN_FLAGS} -np 1 
  ${CMAKE_CURRENT_BINARY_DIR}/simple_contact 
  -my_is_test
  -my_file ${CMAKE_CURRENT_BINARY_DIR}/examples/smiling_face_contact.cub
  -my_order 2
  -my_cn_value 1e-1
)
add_test(
  test_simple_contact_smiling_face_compare ${CMAKE_COMMAND} -E compare_files
  ${CMAKE_CURRENT_SOURCE_DIR}/blessed_files/test_simple_contact_smiling_face.txt
  ${CMAKE_CURRENT_BINARY_DIR}/test_simple_contact.txt
)
