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

set(BFE_DIRS 
  elasticity
  simple_elasticity
  elasticity_mixed_formulation
  nonlinear_elasticity
  mix_transport
  thermal
  cohesive_interface
  magnetostatic
  poisson
  hello_world
  hierarchical_approximation
  reaction_diffusion_equation
  mixed_chemo_mechanics)

function(bfe_install bfe_mod_dir)
  install(
    DIRECTORY
    ${PROJECT_SOURCE_DIR}/basic_finite_elements/${bfe_mod_dir} 
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}
    FILES_MATCHING
    PATTERN "README"
    PATTERN "*.h5m"
    PATTERN "*.cub"
    PATTERN "*.jou"
    PATTERN "*.msh"
    PATTERN "*.cfg"
    PATTERN "*.geo"
    PATTERN "src" EXCLUDE
    PATTERN "doc" EXCLUDE
    PATTERN "atom_tests" EXCLUDE)
endfunction(bfe_install)

function(bfe_build_and_install target source)
  bfe_add_executable(${target} ${source})
  target_link_libraries(${target}
    users_modules
    mofem_finite_elements
    mofem_interfaces
    mofem_multi_indices
    mofem_petsc
    mofem_approx
    mofem_third_party
    mofem_cblas
    ${MoFEM_PROJECT_LIBS})
  install(TARGETS ${target} DESTINATION ${BFE_MOD_DIR})
endfunction(bfe_build_and_install)

if(MOFEM_UM_BUILD_TESTS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/basic_finite_elements/atom_tests 
  ${PROJECT_BINARY_DIR}/basic_finite_elements/atom_tests)
endif(MOFEM_UM_BUILD_TESTS)

foreach(LOOP_DIR ${BFE_DIRS})
  set(BFE_MOD_DIR ${LOOP_DIR})
  add_subdirectory(${LOOP_DIR})
  bfe_install(${LOOP_DIR})
endforeach(LOOP_DIR)

