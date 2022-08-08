

#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <fstream>
#include <iostream>

namespace bio = boost::iostreams;
using bio::stream;
using bio::tee_device;

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

  std::vector<VectorDouble> val;

  std::vector<VectorDouble> &operator()(double x, double y, double z) {
    val.resize(1);
    val[0].resize(1);
    (val[0])[0] = pow(x, 1);
    return val;
  }
};

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                 mesh_file_name, 255, &flg);
    if (flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    const char *option;
    option = ""; 
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create MoFEM (Joseph) database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // set entitities bit level
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_level0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    // Fields
    CHKERR m_field.add_field("TEMP", H1, AINSWORTH_LEGENDRE_BASE, 1);

    // Problem
    CHKERR m_field.add_problem("TEST_PROBLEM");
    CHKERR m_field.add_problem("BC_PROBLEM");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("TEST_PROBLEM", bit_level0);
    CHKERR m_field.modify_problem_ref_level_add_bit("BC_PROBLEM", bit_level0);

    // meshset consisting all entities in mesh
    EntityHandle root_set = moab.get_root_set();
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3);
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET,
                                             "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    // add entities to field
    CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET, "TEMP");

    // set app. order
    // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
    // (Mark Ainsworth & Joe Coyle)
    PetscInt order;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &order,
                              &flg);
    if (flg != PETSC_TRUE) {
      order = 2;
    }
    CHKERR m_field.set_field_order(root_set, MBTET, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBTRI, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBEDGE, "TEMP", order);
    CHKERR m_field.set_field_order(root_set, MBVERTEX, "TEMP", 1);

    ThermalElement thermal_elements(m_field);
    CHKERR thermal_elements.addThermalElements("TEMP");
    CHKERR m_field.modify_problem_add_finite_element("TEST_PROBLEM",
                                                     "THERMAL_FE");

    Range bc_tris;
    for (_IT_CUBITMESHSETS_BY_NAME_FOR_LOOP_(m_field, "ANALYTICAL_BC", it)) {
      CHKERR moab.get_entities_by_type(it->getMeshset(), MBTRI, bc_tris, true);
    }

    AnalyticalDirichletBC analytical_bc(m_field);
    CHKERR analytical_bc.setFiniteElement(m_field, "BC_FE", "TEMP", bc_tris);
    CHKERR m_field.modify_problem_add_finite_element("BC_PROBLEM", "BC_FE");

    /****/
    // build database
    // build field
    CHKERR m_field.build_fields();
    // build finite elemnts
    CHKERR m_field.build_finite_elements();
    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);
    // build problem
    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    CHKERR prb_mng_ptr->buildProblem("TEST_PROBLEM", true);
    CHKERR prb_mng_ptr->buildProblem("BC_PROBLEM", true);

    Projection10NodeCoordsOnField ent_method_material(m_field,
                                                      "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);

    /****/
    // mesh partitioning
    // partition
    CHKERR prb_mng_ptr->partitionSimpleProblem("TEST_PROBLEM");
    CHKERR prb_mng_ptr->partitionFiniteElements("TEST_PROBLEM");
    // what are ghost nodes, see Petsc Manual
    CHKERR prb_mng_ptr->partitionGhostDofs("TEST_PROBLEM");

    CHKERR prb_mng_ptr->partitionSimpleProblem("BC_PROBLEM");
    CHKERR prb_mng_ptr->partitionFiniteElements("BC_PROBLEM");
    // what are ghost nodes, see Petsc Manual
    CHKERR prb_mng_ptr->partitionGhostDofs("BC_PROBLEM");

    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost("TEST_PROBLEM",
                                                              ROW, &F);
    Vec T;
    CHKERR VecDuplicate(F, &T);
    Mat A;
    CHKERR m_field.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("TEST_PROBLEM", &A);

    CHKERR addHOOpsVol("MESH_NODE_POSITIONS", thermal_elements.getLoopFeRhs(),
                    true, false, false, false);
    CHKERR addHOOpsVol("MESH_NODE_POSITIONS", thermal_elements.getLoopFeLhs(),
                    true, false, false, false);

    CHKERR thermal_elements.setThermalFiniteElementRhsOperators("TEMP", F);
    CHKERR thermal_elements.setThermalFiniteElementLhsOperators("TEMP", A);
    CHKERR thermal_elements.setThermalFluxFiniteElementRhsOperators("TEMP", F);

    CHKERR VecZeroEntries(T);
    CHKERR VecZeroEntries(F);
    CHKERR MatZeroEntries(A);

    // analytical Dirichlet bc
    AnalyticalDirichletBC::DirichletBC analytical_dirichlet_bc(m_field, "TEMP",
                                                               A, T, F);

    // solve for dirichlet bc dofs
    CHKERR analytical_bc.setUpProblem(m_field, "BC_PROBLEM");

    boost::shared_ptr<AnalyticalFunction> testing_function =
        boost::shared_ptr<AnalyticalFunction>(new AnalyticalFunction);

    CHKERR analytical_bc.setApproxOps(m_field, "TEMP", testing_function, 0);
    CHKERR analytical_bc.solveProblem(m_field, "BC_PROBLEM", "BC_FE",
                                      analytical_dirichlet_bc);

    CHKERR analytical_bc.destroyProblem();

    // preproc
    CHKERR m_field.problem_basic_method_preProcess("TEST_PROBLEM",
                                                   analytical_dirichlet_bc);

    CHKERR m_field.loop_finite_elements("TEST_PROBLEM", "THERMAL_FE",
                                        thermal_elements.getLoopFeRhs());
    CHKERR m_field.loop_finite_elements("TEST_PROBLEM", "THERMAL_FE",
                                        thermal_elements.getLoopFeLhs());
    if (m_field.check_finite_element("THERMAL_FLUX_FE"))
      CHKERR m_field.loop_finite_elements("TEST_PROBLEM", "THERMAL_FLUX_FE",
                                          thermal_elements.getLoopFeFlux());

    // postproc
    CHKERR m_field.problem_basic_method_postProcess("TEST_PROBLEM",
                                                    analytical_dirichlet_bc);

    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);
    CHKERR MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    CHKERR VecScale(F, -1);
    // std::string wait;
    // std::cout << "\n matrix is coming = \n" << std::endl;
    // CHKERR MatView(A,PETSC_VIEWER_DRAW_WORLD);
    // std::cin >> wait;

    // Solver
    KSP solver;
    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);
    CHKERR KSPSetOperators(solver, A, A);
    CHKERR KSPSetFromOptions(solver);
    CHKERR KSPSetUp(solver);

    CHKERR KSPSolve(solver, F, T);
    CHKERR VecGhostUpdateBegin(T, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(T, INSERT_VALUES, SCATTER_FORWARD);

    // Save data on mesh
    // CHKERR
    // m_field.problem_basic_method_preProcess("TEST_PROBLEM",analytical_dirichlet_bc);
    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "TEST_PROBLEM", ROW, T, ADD_VALUES, SCATTER_REVERSE);
    CHKERR m_field.getInterface<VecManager>()->setLocalGhostVector(
        "TEST_PROBLEM", ROW, T, INSERT_VALUES, SCATTER_FORWARD);

    // CHKERR VecView(F,PETSC_VIEWER_STDOUT_WORLD);

    // CHKERR VecView(T,PETSC_VIEWER_STDOUT_WORLD);

    PetscReal pointwisenorm;
    CHKERR VecMax(T, NULL, &pointwisenorm);
    std::cout << "\n The Global Pointwise Norm of error for this problem is : "
              << pointwisenorm << std::endl;

    // PetscViewer viewer;
    // PetscViewerASCIIOpen(PETSC_COMM_WORLD,"thermal_with_analytical_bc.txt",&viewer);
    // CHKERR VecChop(T,1e-4);
    // CHKERR VecView(T,viewer);
    // CHKERR PetscViewerDestroy(&viewer);

    double sum = 0;
    CHKERR VecSum(T, &sum);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "sum  = %9.8e\n", sum);
    double fnorm;
    CHKERR VecNorm(T, NORM_2, &fnorm);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "fnorm  = %9.8e\n", fnorm);
    if (fabs(sum + 6.46079983e-01) > 1e-7) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }
    if (fabs(fnorm - 4.26080052e+00) > 1e-6) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID, "Failed to pass test");
    }

    if (debug) {

      PostProcVolumeOnRefinedMesh post_proc(m_field);
      CHKERR post_proc.generateReferenceElementMesh();
      CHKERR addHOOpsVol("MESH_NODE_POSITIONS", post_proc, true, false, false,
                      false);
      CHKERR post_proc.addFieldValuesPostProc("TEMP");
      CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
      CHKERR post_proc.addFieldValuesGradientPostProc("TEMP");
      CHKERR m_field.loop_finite_elements("TEST_PROBLEM", "THERMAL_FE",
                                          post_proc);
      CHKERR post_proc.writeFile("out.h5m");
    }

    CHKERR MatDestroy(&A);
    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&T);
    CHKERR KSPDestroy(&solver);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
