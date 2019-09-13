/** \file navier_stokes.cpp
 * \example navier_stokes.cpp
 *
 * \brief Main implementation of a finite element for the Stokes equation
 * 
 */

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

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Mix STOKES problem",
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
      // This is a case of distributed mesh and algebra. In that case each
      // processor keeps only part of the problem.
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
    

    CHKERR m_field.add_field("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR m_field.add_field("P", H1, AINSWORTH_LEGENDRE_BASE, 1);
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3);

    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "U");
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "P");
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");

    CHKERR m_field.set_field_order(0, MBVERTEX, "U", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "U", order_u);
    CHKERR m_field.set_field_order(0, MBTRI, "U", order_u);
    CHKERR m_field.set_field_order(0, MBTET, "U", order_u);

    CHKERR m_field.set_field_order(0, MBVERTEX, "P", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "P", order_p);
    CHKERR m_field.set_field_order(0, MBTRI, "P", order_p);
    CHKERR m_field.set_field_order(0, MBTET, "P", order_p);

    // Set 2nd order of approximation of geometry
    auto setting_second_order_geometry = [&m_field]() {
      MoFEMFunctionBegin;
      // Setting geometry order everywhere
      Range tets, edges;
      CHKERR m_field.get_moab().get_entities_by_type(0, MBTET, tets);
      CHKERR m_field.get_moab().get_adjacencies(tets, 1, false, edges,
                                                moab::Interface::UNION);

      CHKERR m_field.set_field_order(edges, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);
      MoFEMFunctionReturn(0);
    };
    CHKERR setting_second_order_geometry();

    CHKERR m_field.build_fields();

    // CHKERR m_field.getInterface<FieldBlas>()->setField(
    //     0, MBVERTEX, "P"); // initial p = 0 everywhere
    Projection10NodeCoordsOnField ent_method_material(m_field,
                                                        "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);
    // setup elements for loading
    CHKERR MetaNeummanForces::addNeumannBCElements(m_field, "U");
    CHKERR MetaNodalForces::addElement(m_field, "U");
    CHKERR MetaEdgeForces::addElement(m_field, "U");

    // **** ADD ELEMENTS **** //

    // Add finite element (this defines element, declaration comes later)
    CHKERR m_field.add_finite_element("STOKES");
    CHKERR m_field.modify_finite_element_add_field_row("STOKES", "U");
    CHKERR m_field.modify_finite_element_add_field_col("STOKES", "U");
    CHKERR m_field.modify_finite_element_add_field_data("STOKES", "U");

    CHKERR m_field.modify_finite_element_add_field_row("STOKES", "P");
    CHKERR m_field.modify_finite_element_add_field_col("STOKES", "P");
    CHKERR m_field.modify_finite_element_add_field_data("STOKES", "P");
    CHKERR m_field.modify_finite_element_add_field_data("STOKES",
                                                        "MESH_NODE_POSITIONS");

    // Add entities to that element
    CHKERR m_field.add_ents_to_finite_element_by_type(0, MBTET, "STOKES");
    // build finite elements
    CHKERR m_field.build_finite_elements();
    // build adjacencies between elements and degrees of freedom
    CHKERR m_field.build_adjacencies(bit_level0);

    // **** BUILD DM **** //
    DM dm;
    DMType dm_name = "DM_STOKES";
    // Register DM problem
    CHKERR DMRegister_MoFEM(dm_name);
    CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
    CHKERR DMSetType(dm, dm_name);
    // Create DM instance
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, dm_name, bit_level0);
    // Configure DM form line command options (DM itself, solvers,
    // pre-conditioners, ... )
    CHKERR DMSetFromOptions(dm);
    // Add elements to dm (only one here)
    CHKERR DMMoFEMAddElement(dm, "STOKES");
    if (m_field.check_finite_element("PRESSURE_FE"))
      CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
    //if (m_field.check_finite_element("FORCE_FE"))
    //  CHKERR DMMoFEMAddElement(dm, "FORCE_FE");
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // setup the DM
    CHKERR DMSetUp(dm);

    //NavierStokesElement::DataAtIntegrationPts commonData(m_field);
    //CHKERR commonData->getParameters();

    boost::shared_ptr<NavierStokesElement::CommonData> commonData =
        boost::make_shared<NavierStokesElement::CommonData>();

    boost::shared_ptr<FEMethod> nullFE;
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> feLhs(
        new VolumeElementForcesAndSourcesCore(m_field));
    //boost::shared_ptr<VolumeElementForcesAndSourcesCore> feRhs(
    //    new VolumeElementForcesAndSourcesCore(m_field));

    feLhs->getRuleHook = NavierStokesElement::VolRule();
    //feRhs->getRuleHook = FaceRule();

    //CHKERR NavierStokesElement::setOperators(feLhs, feRhs, commonData);

    PostProcVolumeOnRefinedMesh post_proc(m_field);

    //std::map<int, NavierStokesElement::BlockData> setOfBlocksData;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 5, "FLUID") == 0) {
        const int id = bit->getMeshsetId();
        CHKERR m_field.get_moab().get_entities_by_type(
            bit->getMeshset(), MBTET, commonData->setOfBlocksData[id].tEts, true);

        std::vector<double> attributes;
        bit->getAttributes(attributes);
        if (attributes.size() != 2) {
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                   "should be 2 attributes but is %d", attributes.size());
        }
        commonData->setOfBlocksData[id].iD = id;
        commonData->setOfBlocksData[id].fluidViscosity = attributes[0];
        commonData->setOfBlocksData[id].fluidDensity = attributes[1];
      }
    }

    // loop over blocks
    for (auto &sit : commonData->setOfBlocksData) {

      // feLhs->getOpPtrVector().push_back(
      //     new OpAssembleP(commonData, sit.second));
      feLhs->getOpPtrVector().push_back(
          new NavierStokesElement::OpAssembleLhsDiagNonLin(
              "U", "U", commonData, sit.second));
      feLhs->getOpPtrVector().push_back(
          new NavierStokesElement::OpAssembleLhsOffDiag(
              "U", "P", commonData, sit.second));

      post_proc.getOpPtrVector().push_back(
          new OpCalculateScalarFieldValues("P", commonData->pPtr));
      post_proc.getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>("U",
                                                   commonData->gradDispPtr));
      //post_proc.getOpPtrVector().push_back(
      //    new OpPostProcStress(post_proc.postProcMesh, post_proc.mapGaussPts,
      //                         commonData, sit.second));
    }

    Mat Aij;      // Stiffness matrix
    Vec D, F, D0; // Vector of DOFs and the RHS

    {
      CHKERR DMCreateGlobalVector(dm, &D);
      CHKERR VecDuplicate(D, &D0);
      CHKERR VecDuplicate(D, &F);
      CHKERR DMCreateMatrix(dm, &Aij);
      CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);

      CHKERR VecZeroEntries(F);
      CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecZeroEntries(D);
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
      CHKERR MatZeroEntries(Aij);
    }

    // Assign global matrix/vector
    feLhs->ksp_B = Aij;
    feLhs->ksp_f = F;

    boost::shared_ptr<DirichletDisplacementBc> dirichlet_bc_ptr(
        new DirichletDisplacementBc(m_field, "U", Aij, D0, F));
    dirichlet_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
    dirichlet_bc_ptr->ts_ctx = FEMethod::CTX_TSNONE;

    CHKERR VecZeroEntries(D0);
    CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D0, INSERT_VALUES, SCATTER_REVERSE);

    CHKERR  DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());

    CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D0, INSERT_VALUES, SCATTER_REVERSE);
    
    CHKERR DMoFEMLoopFiniteElements(dm, "STOKES", feLhs);

    // Assemble pressure and traction forces.
    boost::ptr_map<std::string, NeummanForcesSurface> neumann_forces;
    CHKERR MetaNeummanForces::setMomentumFluxOperators(m_field, neumann_forces,
                                                       F, "U");

    {
      boost::ptr_map<std::string, NeummanForcesSurface>::iterator mit =
          neumann_forces.begin();
      for (; mit != neumann_forces.end(); mit++) {
        //CHKERR 
        DMoFEMLoopFiniteElements(dm, mit->first.c_str(),
                                        &mit->second->getLoopFe());
      }
    }
    // Assemble forces applied to nodes
    // boost::ptr_map<std::string, NodalForce> nodal_forces;
    // CHKERR MetaNodalForces::setOperators(m_field, nodal_forces, F, "U");

    // {
    //   boost::ptr_map<std::string, NodalForce>::iterator fit =
    //       nodal_forces.begin();
    //   for (; fit != nodal_forces.end(); fit++) {
    //     CHKERR DMoFEMLoopFiniteElements(dm, fit->first.c_str(),
    //                                     &fit->second->getLoopFe());
    //   }
    // }
    // Assemble edge forces
    // boost::ptr_map<std::string, EdgeForce> edge_forces;
    // CHKERR MetaEdgeForces::setOperators(m_field, edge_forces, F, "U");

    // {
    //   boost::ptr_map<std::string, EdgeForce>::iterator fit =
    //       edge_forces.begin();
    //   for (; fit != edge_forces.end(); fit++) {
    //     CHKERR DMoFEMLoopFiniteElements(dm, fit->first.c_str(),
    //                                     &fit->second->getLoopFe());
    //   }
    // }

    CHKERR DMoFEMPostProcessFiniteElements(dm, dirichlet_bc_ptr.get());
    CHKERR DMMoFEMKSPSetComputeOperators(dm, "STOKES", feLhs, nullFE, nullFE);
    
    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);
    CHKERR VecScale(F, -1);

    // **** SOLVE **** //

    KSP solver;

    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);
    CHKERR KSPSetFromOptions(solver);
    CHKERR KSPSetOperators(solver, Aij, Aij);

    //CHKERR KSPSetInitialGuessKnoll(solver, PETSC_FALSE);
    //CHKERR KSPSetInitialGuessNonzero(solver, PETSC_TRUE);
    CHKERR KSPSetUp(solver);

    CHKERR KSPSolve(solver, F, D);

    CHKERR VecAXPY(D, 1., D0);

    // VecView(F, PETSC_VIEWER_STDOUT_WORLD);
    // CHKERR VecView(d, PETSC_VIEWER_STDOUT_WORLD);    // Print out the results
    
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    // Save data on mesh
    CHKERR post_proc.generateReferenceElementMesh();
    CHKERR post_proc.addFieldValuesPostProc("U");
    CHKERR post_proc.addFieldValuesPostProc("P");
    CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");

    CHKERR DMoFEMLoopFiniteElements(dm, "STOKES", &post_proc);
    PetscPrintf(PETSC_COMM_WORLD, "Output file: %s\n", "out.h5m");
    CHKERR post_proc.postProcMesh.write_file("out.h5m", "MOAB",
                                             "PARALLEL=WRITE_PART");

    CHKERR MatDestroy(&Aij);
    CHKERR VecDestroy(&D);
    CHKERR VecDestroy(&D0);
    CHKERR VecDestroy(&F);
    CHKERR DMDestroy(&dm);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();
  return 0;
}