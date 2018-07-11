/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * MoFEM is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */

#include <BasicFiniteElements.hpp>
using namespace MoFEM;

namespace bio = boost::iostreams;
using bio::tee_device;
using bio::stream;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc,&argv,(char *)0,help);

  try {

    moab::Core mb_instance;
    moab::Interface& moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    ierr = PetscOptionsGetString(PETSC_NULL,PETSC_NULL,"-my_file",mesh_file_name,255,&flg); CHKERRG(ierr);
    if(flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF,1,"*** ERROR -my_file (MESH FILE NEEDED)");
    }

    ParallelComm* pcomm = ParallelComm::get_pcomm(&moab,MYPCOMM_INDEX);
    if(pcomm == NULL) pcomm =  new ParallelComm(&moab,PETSC_COMM_WORLD);

    const char *option;
    option = "";//"PARALLEL=BCAST;";//;DEBUG_IO";
    BARRIER_PCOMM_RANK_START(pcomm)
    rval = moab.load_file(mesh_file_name, 0, option); CHKERRG(rval);
    BARRIER_PCOMM_RANK_END(pcomm)

    //Create MoFEM (Joseph) database
    MoFEM::Core core(moab);
    MoFEM::Interface& m_field = core;

    //set entitities bit level
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    rval = moab.create_meshset(MESHSET_SET,meshset_level0); CHKERRG(rval);
    ierr = m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(0,3,bit_level0); CHKERRG(ierr);

    //Fields
    ierr = m_field.add_field("TEMP",H1,AINSWORTH_LEGENDRE_BASE,1); CHKERRG(ierr);

    //Problem
    ierr = m_field.add_problem("TEST_PROBLEM"); CHKERRG(ierr);

    //set refinement level for problem
    ierr = m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM",bit_level0); CHKERRG(ierr);

    //meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    //add entities to field
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"TEMP"); CHKERRG(ierr);

    //set app. order
    //see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes (Mark Ainsworth & Joe Coyle)
    int order = 4;
    ierr = m_field.set_field_order(root_set,MBTET,"TEMP",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"TEMP",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"TEMP",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"TEMP",1); CHKERRG(ierr);

    ThermalElement thermal_elements(m_field);
    ierr = thermal_elements.addThermalElements("TEMP"); CHKERRG(ierr);
    ierr = thermal_elements.addThermalFluxElement("TEMP"); CHKERRG(ierr);

    ierr = m_field.modify_problem_add_finite_element("TEST_PROBLEM","THERMAL_FE"); CHKERRG(ierr);
    ierr = m_field.modify_problem_add_finite_element("TEST_PROBLEM","THERMAL_FLUX_FE"); CHKERRG(ierr);

    /****/
    //build database
    //build field
    ierr = m_field.build_fields(); CHKERRG(ierr);
    //build finite elemnts
    ierr = m_field.build_finite_elements(); CHKERRG(ierr);
    //build adjacencies
    ierr = m_field.build_adjacencies(bit_level0); CHKERRG(ierr);

    /****/
    //mesh partitioning
    //build problem
    ProblemsManager *prb_mng_ptr;
    ierr = m_field.getInterface(prb_mng_ptr); CHKERRG(ierr);
    ierr = prb_mng_ptr->buildProblem("TEST_PROBLEM",true); CHKERRG(ierr);
    //partition
    ierr = prb_mng_ptr->partitionSimpleProblem("TEST_PROBLEM"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("TEST_PROBLEM"); CHKERRG(ierr);
    //what are ghost nodes, see Petsc Manual
    ierr = prb_mng_ptr->partitionGhostDofs("TEST_PROBLEM"); CHKERRG(ierr);

    Vec F;
    ierr = m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",ROW,&F); CHKERRG(ierr);
    Vec T;
    ierr = VecDuplicate(F,&T); CHKERRG(ierr);
    Mat A;
    ierr = m_field.MatCreateMPIAIJWithArrays("TEST_PROBLEM",&A); CHKERRG(ierr);

    DirichletTemperatureBc my_dirichlet_bc(m_field,"TEMP",A,T,F);
    ierr = thermal_elements.setThermalFiniteElementRhsOperators("TEMP",F); CHKERRG(ierr);
    ierr = thermal_elements.setThermalFiniteElementLhsOperators("TEMP",A); CHKERRG(ierr);
    ierr = thermal_elements.setThermalFluxFiniteElementRhsOperators("TEMP",F); CHKERRG(ierr);

    ierr = VecZeroEntries(T); CHKERRG(ierr);
    ierr = VecZeroEntries(F); CHKERRG(ierr);
    ierr = MatZeroEntries(A); CHKERRG(ierr);

    //preproc
    ierr = m_field.problem_basic_method_preProcess("TEST_PROBLEM",my_dirichlet_bc); CHKERRG(ierr);
    ierr = m_field.getInterface<VecManager>()->setGlobalGhostVector("TEST_PROBLEM",ROW,T,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);

    ierr = m_field.loop_finite_elements("TEST_PROBLEM","THERMAL_FE",thermal_elements.getLoopFeRhs()); CHKERRG(ierr);
    ierr = m_field.loop_finite_elements("TEST_PROBLEM","THERMAL_FE",thermal_elements.getLoopFeLhs()); CHKERRG(ierr);
    ierr = m_field.loop_finite_elements("TEST_PROBLEM","THERMAL_FLUX_FE",thermal_elements.getLoopFeFlux()); CHKERRG(ierr);

    //postproc
    ierr = m_field.problem_basic_method_postProcess("TEST_PROBLEM",my_dirichlet_bc); CHKERRG(ierr);

    ierr = VecGhostUpdateBegin(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecAssemblyBegin(F); CHKERRG(ierr);
    ierr = VecAssemblyEnd(F); CHKERRG(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);

    ierr = VecScale(F,-1); CHKERRG(ierr);

    //Solver
    KSP solver;
    ierr = KSPCreate(PETSC_COMM_WORLD,&solver); CHKERRG(ierr);
    ierr = KSPSetOperators(solver,A,A); CHKERRG(ierr);
    ierr = KSPSetFromOptions(solver); CHKERRG(ierr);
    ierr = KSPSetUp(solver); CHKERRG(ierr);

    ierr = KSPSolve(solver,F,T); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(T,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(T,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

    ierr = m_field.problem_basic_method_preProcess("TEST_PROBLEM",my_dirichlet_bc); CHKERRG(ierr);

    //Save data on mesh
    ierr = m_field.getInterface<VecManager>()->setGlobalGhostVector("TEST_PROBLEM",ROW,T,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    //ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD); CHKERRG(ierr);

    // PetscViewer viewer;
    // PetscViewerASCIIOpen(PETSC_COMM_WORLD,"thermal_elem.txt",&viewer);
    // ierr = VecChop(T,1e-4); CHKERRG(ierr);
    // ierr = VecView(T,viewer); CHKERRG(ierr);
    // ierr = PetscViewerDestroy(&viewer); CHKERRG(ierr);

    double sum = 0;
    ierr = VecSum(F,&sum); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sum  = %9.8f\n",sum); CHKERRG(ierr);
    double fnorm;
    ierr = VecNorm(F,NORM_2,&fnorm); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"fnorm  = %9.8e\n",fnorm); CHKERRG(ierr);
    if(fabs(sum+0.59583333)>1e-7) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }
    if(fabs(fnorm-2.32872499e-01)>1e-6) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }


    /*PostProcVertexMethod ent_method(moab,"TEMP");
    ierr = m_field.loop_dofs("TEST_PROBLEM","TEMP",ROW,ent_method); CHKERRG(ierr);
    if(pcomm->rank()==0) {
    EntityHandle out_meshset;
    rval = moab.create_meshset(MESHSET_SET,out_meshset); CHKERRG(rval);
    ierr = m_field.get_problem_finite_elements_entities("TEST_PROBLEM","THERMAL_FE",out_meshset); CHKERRG(ierr);
    rval = moab.write_file("out.vtk","VTK","",&out_meshset,1); CHKERRG(rval);
    rval = moab.delete_entities(&out_meshset,1); CHKERRG(rval);
    */

    //Matrix View
    //MatView(A,PETSC_VIEWER_DRAW_WORLD);//PETSC_VIEWER_STDOUT_WORLD);
    //ierr = VecView(T,PETSC_VIEWER_STDOUT_WORLD); CHKERRG(ierr);
    //std::string wait;
    //std::cin >> wait;

    ierr = MatDestroy(&A); CHKERRG(ierr);
    ierr = VecDestroy(&F); CHKERRG(ierr);
    ierr = VecDestroy(&T); CHKERRG(ierr);
    ierr = KSPDestroy(&solver); CHKERRG(ierr);

  } catch (MoFEMException const &e) {
    SETERRQ(PETSC_COMM_SELF,e.errorCode,e.errorMessage);
  }

  MoFEM::Core::Finalize();

  return 0;

}
