/** \file elasticity_mf_springs.cpp
 * \example elasticity_mf_springs.cpp
 *
 * \brief Main implementation of U-P (mixed) finite element with springs.
 *
 * */
/* MoFEM is free software: you can redistribute it and/or modify it under
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
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *
 * */
#include <BasicFiniteElements.hpp>
#include <ElasticityMixedFormulationSprings.hpp>

using namespace boost::numeric;
using namespace MoFEM;
using namespace std;

static char help[] = "-my_block_config set block data\n"
                     "\n";

int main(int argc, char *argv[]) {

  // Initialise MoFEM
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  // Create mesh database
  moab::Core mb_instance;              // create database
  moab::Interface &moab = mb_instance; // create interface to database

  // Create moab communicator
  // Create separate MOAB communicator, it is duplicate of PETSc communicator.
  // NOTE That this should eliminate potential communication problems between
  // MOAB and PETSC functions.
  MPI_Comm moab_comm_world;
  MPI_Comm_dup(PETSC_COMM_WORLD, &moab_comm_world);
  ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
  if (pcomm == NULL)
    pcomm = new ParallelComm(&moab, moab_comm_world);

  try {
    // Get command line options
    char mesh_file_name[255];
    PetscBool flg_file;
    int order_p = 2; // default approximation order_p
    int order_u = 3; // default approximation order_u
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool flg_test = PETSC_FALSE; // true check if error is numerical error

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Mixed elastic problem",
                             "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    // Set approximation order
    CHKERR PetscOptionsInt("-my_order_p", "approximation order_p", "", order_p,
                           &order_p, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_order_u", "approximation order_u", "", order_u,
                           &order_u, PETSC_NULL);

    CHKERR PetscOptionsBool("-is_partitioned", "is_partitioned?", "",
                            is_partitioned, &is_partitioned, PETSC_NULL);

    // Set testing (used by CTest)
    CHKERR PetscOptionsBool("-test", "if true is ctest", "", flg_test,
                            &flg_test, PETSC_NULL);
    ierr = PetscOptionsEnd();

    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    // Read whole mesh or part of it if partitioned
    if (is_partitioned == PETSC_TRUE) {
      // This is a case of distributed mesh and algebra. In this case each
      // processor keeps only one part of the problem.
      const char *option;
      option = "PARALLEL=READ_PART;"
               "PARALLEL_RESOLVE_SHARED_ENTS;"
               "PARTITION=PARALLEL_PARTITION;";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    } else {
      // In this case, we have distributed algebra, i.e. assembly of vectors and
      // matrices is in parallel, but whole mesh is stored on all processors.
      // Solver and matrix scales well, however problem set-up of problem is
      // not fully parallel.
      const char *option;
      option = "";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    }

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // Print boundary conditions and material parameters
    MeshsetsManager *meshsets_mng_ptr;
    CHKERR m_field.getInterface(meshsets_mng_ptr);
    CHKERR meshsets_mng_ptr->printDisplacementSet();
    CHKERR meshsets_mng_ptr->printForceSet();
    CHKERR meshsets_mng_ptr->printMaterialsSet();

    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    // **** ADD FIELDS **** //
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    CHKERR m_field.add_field("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "U");
    CHKERR m_field.set_field_order(0, MBVERTEX, "U", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "U", order_u);
    CHKERR m_field.set_field_order(0, MBTRI, "U", order_u);
    CHKERR m_field.set_field_order(0, MBTET, "U", order_u);

    CHKERR m_field.add_field("P", H1, AINSWORTH_LEGENDRE_BASE, 1);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "P");
    CHKERR m_field.set_field_order(0, MBVERTEX, "P", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "P", order_p);
    CHKERR m_field.set_field_order(0, MBTRI, "P", order_p);
    CHKERR m_field.set_field_order(0, MBTET, "P", order_p);

    CHKERR m_field.build_fields();

    // CHKERR m_field.getInterface<FieldBlas>()->setField(
    //     0, MBVERTEX, "P"); // initial p = 0 everywhere
    {
      Projection10NodeCoordsOnField ent_method_material(m_field,
                                                        "MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);
    }
    // setup elements for loading
    CHKERR MetaNeummanForces::addNeumannBCElements(m_field, "U");
    CHKERR MetaNodalForces::addElement(m_field, "U");
    CHKERR MetaEdgeForces::addElement(m_field, "U");

    // **** ADD ELEMENTS **** //

    // Add finite element (this defines element, declaration comes later)
    CHKERR m_field.add_finite_element("ELASTIC");
    CHKERR m_field.modify_finite_element_add_field_row("ELASTIC", "U");
    CHKERR m_field.modify_finite_element_add_field_col("ELASTIC", "U");
    CHKERR m_field.modify_finite_element_add_field_data("ELASTIC", "U");

    CHKERR m_field.modify_finite_element_add_field_row("ELASTIC", "P");
    CHKERR m_field.modify_finite_element_add_field_col("ELASTIC", "P");
    CHKERR m_field.modify_finite_element_add_field_data("ELASTIC", "P");
    CHKERR m_field.modify_finite_element_add_field_data("ELASTIC",
                                                        "MESH_NODE_POSITIONS");

    // Add spring element, just depends on displacement "U"
    CHKERR m_field.add_finite_element("SPRING");
    CHKERR m_field.modify_finite_element_add_field_row("SPRING", "U");
    CHKERR m_field.modify_finite_element_add_field_col("SPRING", "U");
    CHKERR m_field.modify_finite_element_add_field_data("SPRING", "U");

    // Add entities to that element
    CHKERR m_field.add_ents_to_finite_element_by_type(0, MBTET, "ELASTIC");

    // Add entities to spring element,
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {
        CHKERR m_field.add_ents_to_finite_element_by_type(bit->getMeshset(),
                                                          MBTRI, "SPRING");
      } 
    }

    // build finite elements
    CHKERR m_field.build_finite_elements();
    // build adjacencies between elements and degrees of freedom
    CHKERR m_field.build_adjacencies(bit_level0);

    // **** BUILD DM **** //
    DM dm;
    DMType dm_name = "DM_ELASTIC_MIX";
    // Register DM problem
    CHKERR DMRegister_MoFEM(dm_name);
    CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
    CHKERR DMSetType(dm, dm_name);
    // Create DM instance
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, dm_name, bit_level0);
    // Configure DM form line command options (DM itself, solvers,
    // pre-conditioners, ... )
    CHKERR DMSetFromOptions(dm);

    // Add elements to DM
    CHKERR DMMoFEMAddElement(dm, "ELASTIC");
    CHKERR DMMoFEMAddElement(dm, "SPRING");
    if (m_field.check_finite_element("PRESSURE_FE"))
      CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
    if (m_field.check_finite_element("FORCE_FE"))
      CHKERR DMMoFEMAddElement(dm, "FORCE_FE");
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);

    // setup the DM
    CHKERR DMSetUp(dm);

    DataAtIntegrationPts commonData(m_field);
    CHKERR commonData.getParameters();

    // Create new instances
    boost::shared_ptr<FEMethod> nullFE;
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> feLhs(
        new VolumeElementForcesAndSourcesCore(m_field));
    // *** Springs: Create new instances of face elements for Springs
    boost::shared_ptr<FaceElementForcesAndSourcesCore> feSpringLhs(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<FaceElementForcesAndSourcesCore> feSpringRhs(
        new FaceElementForcesAndSourcesCore(m_field));

    PostProcVolumeOnRefinedMesh post_proc(m_field);

    // Push operators to instances
    // loop over blocks
    for (auto &sitElastic : commonData.mapElastic) {
      feLhs->getOpPtrVector().push_back(
          new OpAssembleP(commonData, sitElastic.second));
      feLhs->getOpPtrVector().push_back(
          new OpAssembleK(commonData, sitElastic.second));
      feLhs->getOpPtrVector().push_back(
          new OpAssembleG(commonData, sitElastic.second));

      post_proc.getOpPtrVector().push_back(
          new OpCalculateScalarFieldValues("P", commonData.pPtr));
      post_proc.getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>("U",
                                                   commonData.gradDispPtr));
      post_proc.getOpPtrVector().push_back(
          new OpPostProcStress(post_proc.postProcMesh, post_proc.mapGaussPts,
                               commonData, sitElastic.second));
    }

    for (auto &sitSpring : commonData.mapSpring) {
      feSpringLhs->getOpPtrVector().push_back(
          new OpSpringKs(commonData, sitSpring.second));
          
      feSpringRhs->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<3>("U", commonData.xAtPts));
      feSpringRhs->getOpPtrVector().push_back(
          new OpSpringFs(commonData, sitSpring.second));
    }

    Mat Aij;      // Stiffness matrix
    Vec d, F_ext; // Vector of DOFs and the RHS

    {
      CHKERR DMCreateGlobalVector_MoFEM(dm, &d);
      CHKERR VecZeroEntries(d);
      CHKERR VecDuplicate(d, &F_ext);
      CHKERR DMCreateMatrix_MoFEM(dm, &Aij);
      CHKERR VecGhostUpdateBegin(d, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(d, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR DMoFEMMeshToLocalVector(dm, d, INSERT_VALUES, SCATTER_REVERSE);
      CHKERR MatZeroEntries(Aij);
    }

    // Assign global matrix/vector
    feLhs->ksp_B = Aij;
    feSpringLhs->ksp_B = Aij;
    feSpringRhs->ksp_f = F_ext;

    boost::shared_ptr<DirichletDisplacementBc> dirichlet_bc_ptr(
        new DirichletDisplacementBc(m_field, "U", Aij, d, F_ext));
    dirichlet_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
    dirichlet_bc_ptr->ts_ctx = FEMethod::CTX_TSNONE;
    CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", feLhs);
    CHKERR DMoFEMLoopFiniteElements(dm, "SPRING", feSpringLhs);
    CHKERR DMoFEMLoopFiniteElements(dm, "SPRING", feSpringRhs);

    // Assemble pressure and traction forces.
    boost::ptr_map<std::string, NeummanForcesSurface> neumann_forces;
    CHKERR MetaNeummanForces::setMomentumFluxOperators(m_field, neumann_forces,
                                                       F_ext, "U");

    {
      boost::ptr_map<std::string, NeummanForcesSurface>::iterator mit =
          neumann_forces.begin();
      for (; mit != neumann_forces.end(); mit++) {
        CHKERR DMoFEMLoopFiniteElements(dm, mit->first.c_str(),
                                        &mit->second->getLoopFe());
      }
    }
    // Assemble forces applied to nodes
    boost::ptr_map<std::string, NodalForce> nodal_forces;
    CHKERR MetaNodalForces::setOperators(m_field, nodal_forces, F_ext, "U");

    {
      boost::ptr_map<std::string, NodalForce>::iterator fit =
          nodal_forces.begin();
      for (; fit != nodal_forces.end(); fit++) {
        CHKERR DMoFEMLoopFiniteElements(dm, fit->first.c_str(),
                                        &fit->second->getLoopFe());
      }
    }
    // Assemble edge forces
    boost::ptr_map<std::string, EdgeForce> edge_forces;
    CHKERR MetaEdgeForces::setOperators(m_field, edge_forces, F_ext, "U");

    {
      boost::ptr_map<std::string, EdgeForce>::iterator fit =
          edge_forces.begin();
      for (; fit != edge_forces.end(); fit++) {
        CHKERR DMoFEMLoopFiniteElements(dm, fit->first.c_str(),
                                        &fit->second->getLoopFe());
      }
    }

    CHKERR DMoFEMPostProcessFiniteElements(dm, dirichlet_bc_ptr.get());
    CHKERR DMMoFEMKSPSetComputeOperators(dm, "ELASTIC", feLhs, nullFE, nullFE);
    // CHKERR DMMoFEMKSPSetComputeOperators(dm, "SPRING", feSpringLhs, nullFE,
                                        //  nullFE);  //TODO: Need for spring?
    CHKERR VecAssemblyBegin(F_ext);
    CHKERR VecAssemblyEnd(F_ext);

    // **** SOLVE **** //

    KSP solver;

    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);
    CHKERR KSPSetFromOptions(solver);
    CHKERR KSPSetOperators(solver, Aij, Aij);
    CHKERR KSPSetUp(solver);
    CHKERR KSPSolve(solver, F_ext, d);

    // VecView(F_ext, PETSC_VIEWER_STDOUT_WORLD);
    // CHKERR VecView(d, PETSC_VIEWER_STDOUT_WORLD);    // Print out the
    // results
    CHKERR DMoFEMMeshToGlobalVector(dm, d, INSERT_VALUES, SCATTER_REVERSE);

    // Save data on mesh
    CHKERR post_proc.generateReferenceElementMesh();
    CHKERR post_proc.addFieldValuesPostProc("U");
    CHKERR post_proc.addFieldValuesPostProc("P");

    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);
    // CHKERR DMoFEMLoopFiniteElements(dm, "SPRING", &post_proc);
    PetscPrintf(PETSC_COMM_WORLD, "Output file: %s\n", "out.h5m");
    CHKERR post_proc.postProcMesh.write_file("out.h5m", "MOAB",
                                             "PARALLEL=WRITE_PART");

    CHKERR MatDestroy(&Aij);
    CHKERR VecDestroy(&d);
    CHKERR VecDestroy(&F_ext);
    CHKERR DMDestroy(&dm);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();
  return 0;
}