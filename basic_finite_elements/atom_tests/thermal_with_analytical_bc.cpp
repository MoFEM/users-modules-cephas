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

#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <fstream>
#include <iostream>

namespace bio = boost::iostreams;
using bio::tee_device;
using bio::stream;

#include <MoFEM.hpp>
using namespace MoFEM;

#include <MethodForForceScaling.hpp>
#include <DirichletBC.hpp>
#include <PostProcOnRefMesh.hpp>
#include <ThermalElement.hpp>

#include <Projection10NodeCoordsOnField.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <AnalyticalDirichlet.hpp>

#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <fstream>
#include <iostream>

static int debug = 1;
static char help[] = "...\n\n";

struct AnalyticalFunction {

  std::vector<VectorDouble > val;

  std::vector<VectorDouble >& operator()(double x,double y,double z) {
    val.resize(1);
    val[0].resize(1);
    (val[0])[0] = pow(x,1);
    return val;
  }
};

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
    ierr = m_field.add_problem("BC_PROBLEM"); CHKERRG(ierr);

    //set refinement level for problem
    ierr = m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM",bit_level0); CHKERRG(ierr);
    ierr = m_field.modify_problem_ref_level_add_bit("BC_PROBLEM",bit_level0); CHKERRG(ierr);

    //meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    ierr = m_field.add_field("MESH_NODE_POSITIONS",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"MESH_NODE_POSITIONS"); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTET,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTRI,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBEDGE,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBVERTEX,"MESH_NODE_POSITIONS",1); CHKERRG(ierr);

    //add entities to field
    ierr = m_field.add_ents_to_field_by_type(root_set,MBTET,"TEMP"); CHKERRG(ierr);

    //set app. order
    //see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes (Mark Ainsworth & Joe Coyle)
    PetscInt order;
    ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-my_order",&order,&flg); CHKERRG(ierr);
    if(flg != PETSC_TRUE) {
      order = 2;
    }
    ierr = m_field.set_field_order(root_set,MBTET,"TEMP",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBTRI,"TEMP",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBEDGE,"TEMP",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(root_set,MBVERTEX,"TEMP",1); CHKERRG(ierr);

    ThermalElement thermal_elements(m_field);
    ierr = thermal_elements.addThermalElements("TEMP"); CHKERRG(ierr);
    ierr = m_field.modify_problem_add_finite_element("TEST_PROBLEM","THERMAL_FE"); CHKERRG(ierr);

    Range bc_tris;
    for(_IT_CUBITMESHSETS_BY_NAME_FOR_LOOP_(m_field,"ANALYTICAL_BC",it)) {
      rval = moab.get_entities_by_type(it->getMeshset(),MBTRI,bc_tris,true); CHKERRG(rval);
    }

    AnalyticalDirichletBC analytical_bc(m_field);
    ierr = analytical_bc.setFiniteElement(m_field,"BC_FE","TEMP",bc_tris); CHKERRG(ierr);
    ierr = m_field.modify_problem_add_finite_element("BC_PROBLEM","BC_FE"); CHKERRG(ierr);

    /****/
    //build database
    //build field
    ierr = m_field.build_fields(); CHKERRG(ierr);
    //build finite elemnts
    ierr = m_field.build_finite_elements(); CHKERRG(ierr);
    //build adjacencies
    ierr = m_field.build_adjacencies(bit_level0); CHKERRG(ierr);
    //build problem
    ProblemsManager *prb_mng_ptr;
    ierr = m_field.getInterface(prb_mng_ptr); CHKERRG(ierr);
    ierr = prb_mng_ptr->buildProblem("TEST_PROBLEM",true); CHKERRG(ierr);
    ierr = prb_mng_ptr->buildProblem("BC_PROBLEM",true); CHKERRG(ierr);

    Projection10NodeCoordsOnField ent_method_material(m_field,"MESH_NODE_POSITIONS");
    ierr = m_field.loop_dofs("MESH_NODE_POSITIONS",ent_method_material); CHKERRG(ierr);

    /****/
    //mesh partitioning
    //partition
    ierr = prb_mng_ptr->partitionSimpleProblem("TEST_PROBLEM"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("TEST_PROBLEM"); CHKERRG(ierr);
    //what are ghost nodes, see Petsc Manual
    ierr = prb_mng_ptr->partitionGhostDofs("TEST_PROBLEM"); CHKERRG(ierr);

    ierr = prb_mng_ptr->partitionSimpleProblem("BC_PROBLEM"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("BC_PROBLEM"); CHKERRG(ierr);
    //what are ghost nodes, see Petsc Manual
    ierr = prb_mng_ptr->partitionGhostDofs("BC_PROBLEM"); CHKERRG(ierr);

    Vec F;
    ierr = m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",ROW,&F); CHKERRG(ierr);
    Vec T;
    ierr = VecDuplicate(F,&T); CHKERRG(ierr);
    Mat A;
    ierr = m_field.MatCreateMPIAIJWithArrays("TEST_PROBLEM",&A); CHKERRG(ierr);

    ierr = thermal_elements.setThermalFiniteElementRhsOperators("TEMP",F); CHKERRG(ierr);
    ierr = thermal_elements.setThermalFiniteElementLhsOperators("TEMP",A); CHKERRG(ierr);
    ierr = thermal_elements.setThermalFluxFiniteElementRhsOperators("TEMP",F); CHKERRG(ierr);

    ierr = VecZeroEntries(T); CHKERRG(ierr);
    ierr = VecZeroEntries(F); CHKERRG(ierr);
    ierr = MatZeroEntries(A); CHKERRG(ierr);

    //analytical Dirichlet bc
    AnalyticalDirichletBC::DirichletBC analytical_ditihlet_bc(m_field,"TEMP",A,T,F);

    //solve for ditihlet bc dofs
    ierr = analytical_bc.setUpProblem(m_field,"BC_PROBLEM"); CHKERRG(ierr);

    boost::shared_ptr<AnalyticalFunction> testing_function = boost::shared_ptr<AnalyticalFunction>(new AnalyticalFunction);

    ierr = analytical_bc.setApproxOps(m_field,"TEMP",testing_function,0); CHKERRG(ierr);
    ierr = analytical_bc.solveProblem(m_field,"BC_PROBLEM","BC_FE",analytical_ditihlet_bc); CHKERRG(ierr);

    ierr = analytical_bc.destroyProblem(); CHKERRG(ierr);

    //preproc
    ierr = m_field.problem_basic_method_preProcess("TEST_PROBLEM",analytical_ditihlet_bc); CHKERRG(ierr);
    //ierr = m_field.getInterface<VecManager>()->setGlobalGhostVector("TEST_PROBLEM",ROW,T,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);

    ierr = m_field.loop_finite_elements("TEST_PROBLEM","THERMAL_FE",thermal_elements.getLoopFeRhs()); CHKERRG(ierr);
    ierr = m_field.loop_finite_elements("TEST_PROBLEM","THERMAL_FE",thermal_elements.getLoopFeLhs()); CHKERRG(ierr);
    if(m_field.check_finite_element("THERMAL_FLUX_FE"))
    ierr = m_field.loop_finite_elements("TEST_PROBLEM","THERMAL_FLUX_FE",thermal_elements.getLoopFeFlux()); CHKERRG(ierr);

    //postproc
    ierr = m_field.problem_basic_method_postProcess("TEST_PROBLEM",analytical_ditihlet_bc); CHKERRG(ierr);

    ierr = VecGhostUpdateBegin(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = VecAssemblyBegin(F); CHKERRG(ierr);
    ierr = VecAssemblyEnd(F); CHKERRG(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);

    ierr = VecScale(F,-1); CHKERRG(ierr);
    //std::string wait;
    //std::cout << "\n matrix is coming = \n" << std::endl;
    //ierr = MatView(A,PETSC_VIEWER_DRAW_WORLD);
    //std::cin >> wait;

    //Solver
    KSP solver;
    ierr = KSPCreate(PETSC_COMM_WORLD,&solver); CHKERRG(ierr);
    ierr = KSPSetOperators(solver,A,A); CHKERRG(ierr);
    ierr = KSPSetFromOptions(solver); CHKERRG(ierr);
    ierr = KSPSetUp(solver); CHKERRG(ierr);

    ierr = KSPSolve(solver,F,T); CHKERRG(ierr);
    ierr = VecGhostUpdateBegin(T,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
    ierr = VecGhostUpdateEnd(T,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

    //Save data on mesh
    //ierr = m_field.problem_basic_method_preProcess("TEST_PROBLEM",analytical_ditihlet_bc); CHKERRG(ierr);
    ierr = m_field.getInterface<VecManager>()->setGlobalGhostVector("TEST_PROBLEM",ROW,T,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = m_field.getInterface<VecManager>()->setLocalGhostVector("TEST_PROBLEM",ROW,T,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

    //ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD); CHKERRG(ierr);

    //ierr = VecView(T,PETSC_VIEWER_STDOUT_WORLD); CHKERRG(ierr);

    PetscReal pointwisenorm;
    ierr = VecMax(T,NULL,&pointwisenorm);
    std::cout << "\n The Global Pointwise Norm of error for this problem is : " << pointwisenorm << std::endl;

    // PetscViewer viewer;
    // PetscViewerASCIIOpen(PETSC_COMM_WORLD,"thermal_with_analytical_bc.txt",&viewer);
    // ierr = VecChop(T,1e-4); CHKERRG(ierr);
    // ierr = VecView(T,viewer); CHKERRG(ierr);
    // ierr = PetscViewerDestroy(&viewer); CHKERRG(ierr);


    double sum = 0;
    ierr = VecSum(T,&sum); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sum  = %9.8e\n",sum); CHKERRG(ierr);
    double fnorm;
    ierr = VecNorm(T,NORM_2,&fnorm); CHKERRG(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"fnorm  = %9.8e\n",fnorm); CHKERRG(ierr);
    if(fabs(sum+6.46079983e-01)>1e-7) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }
    if(fabs(fnorm-4.26080052e+00)>1e-6) {
      SETERRQ(PETSC_COMM_WORLD,MOFEM_ATOM_TEST_INVALID,"Failed to pass test");
    }


    if(debug) {

      PostProcVolumeOnRefinedMesh post_proc(m_field);
      ierr = post_proc.generateReferenceElementMesh(); CHKERRG(ierr);
      ierr = post_proc.addFieldValuesPostProc("TEMP"); CHKERRG(ierr);
      ierr = post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS"); CHKERRG(ierr);
      ierr = post_proc.addFieldValuesGradientPostProc("TEMP"); CHKERRG(ierr);
      ierr = m_field.loop_finite_elements("TEST_PROBLEM","THERMAL_FE",post_proc); CHKERRG(ierr);
      ierr = post_proc.writeFile("out.h5m"); CHKERRG(ierr);

    }

    ierr = MatDestroy(&A); CHKERRG(ierr);
    ierr = VecDestroy(&F); CHKERRG(ierr);
    ierr = VecDestroy(&T); CHKERRG(ierr);
    ierr = KSPDestroy(&solver); CHKERRG(ierr);


  } 
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;

}
