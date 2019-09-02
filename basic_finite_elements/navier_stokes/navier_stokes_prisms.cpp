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
#include <NavierStokesElement.hpp>

using namespace boost::numeric;
using namespace MoFEM;
using namespace std;

static char help[] = "-my_block_config set block data\n"
                     "\n";

double NavierStokesElement::LoadScale::lambda = 1;

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
    char load_file_name[255];

    PetscBool flg_mesh_file;
    PetscBool flg_load_file;
    int order_p = 2; // default approximation order_p
    int order_u = 3; // default approximation order_u
    int nb_ho_levels = 0;

    int nbSubSteps = 1;   // default number of steps
    double lAmbda0 = 1.0; // default step size
    // int initD = 0.0;            // initial phase_field value
    double stepRed = 0.2; // stepsize reduction while diverging to 20%
    int maxDivStep = 100; // maximumim number of diverged steps
    int outPutStep = 1;   // how often post processing data is saved to h5m file

    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool flg_test = PETSC_FALSE; // true check if error is numerical error

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "NAVIER_STOKES problem",
                             "none");

    CHKERR PetscOptionsString("-file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_mesh_file);
    // Set approximation order
    CHKERR PetscOptionsInt("-order_p", "approximation order_p", "", order_p,
                           &order_p, PETSC_NULL);
    CHKERR PetscOptionsInt("-order_u", "approximation order_u", "", order_u,
                           &order_u, PETSC_NULL);
    CHKERR PetscOptionsInt("-ho_levels", "number of ho levels", "",
                           nb_ho_levels, &nb_ho_levels, PETSC_NULL);

    CHKERR PetscOptionsInt("-output_prt",
                           "frequncy how often results are dumped on hard disk",
                           "", 1, &outPutStep, NULL);

    CHKERR PetscOptionsInt("-steps", "number of steps", "", nbSubSteps,
                           &nbSubSteps, PETSC_NULL);

    CHKERR PetscOptionsInt("-steps_max_div", "number of steps", "", maxDivStep,
                           &maxDivStep, PETSC_NULL);

    CHKERR PetscOptionsScalar("-lambda", "lambda", "", lAmbda0, &lAmbda0,
                              PETSC_NULL);

    CHKERR PetscOptionsString("-load_table", "load history file name", "",
                              "load_table.txt", load_file_name, 255,
                              &flg_load_file);

    CHKERR PetscOptionsScalar("-step_red", "step reductcion while diverge", "",
                              stepRed, &stepRed, PETSC_NULL);

    CHKERR PetscOptionsBool("-is_partitioned", "is_partitioned?", "",
                            is_partitioned, &is_partitioned, PETSC_NULL);

    // Set testing (used by CTest)
    CHKERR PetscOptionsBool("-test", "if true is ctest", "", flg_test,
                            &flg_test, PETSC_NULL);
    ierr = PetscOptionsEnd();

    if (flg_mesh_file != PETSC_TRUE) {
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
      // snes and matrix scales well, however problem set-up of problem is
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
    // CHKERR meshsets_mng_ptr->printDisplacementSet();
    // CHKERR meshsets_mng_ptr->printForceSet();
    // CHKERR meshsets_mng_ptr->printMaterialsSet();

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

    /* for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 5, "SOLID") == 0) {
        Range ents;
        CHKERR m_field.get_moab().get_entities_by_type(
            bit->getMeshset(), MBTRI, ents, true);

        std::vector<Range> levels(nb_ho_levels);
        for (int ll = 0; ll!= nb_ho_levels; ll++) {
          Range verts;
          CHKERR m_field.get_moab().get_connectivity(ents, verts, true);
          for (auto d : {1,2,3}) {
            CHKERR m_field.get_moab().get_adjacencies(verts, d, false, ents,
    moab::Interface::UNION);
          }
          levels[ll] = subtract(ents, ents.subset_by_type(MBVERTEX));
        }

        for (int ll = nb_ho_levels-1; ll>=1; ll--) {
          levels[ll] = subtract(levels[ll], levels[ll-1]);
        }

        int add_order = 1;
        for (int ll = nb_ho_levels-1; ll>=0; ll--) {
          CHKERR m_field.set_field_order(levels[ll], "U", order_u + add_order);
          CHKERR m_field.set_field_order(levels[ll], "P", order_p + add_order);
          ++add_order;
        }

      }
    } */

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
    CHKERR m_field.add_finite_element("NAVIER_STOKES");
    CHKERR m_field.modify_finite_element_add_field_row("NAVIER_STOKES", "U");
    CHKERR m_field.modify_finite_element_add_field_col("NAVIER_STOKES", "U");
    CHKERR m_field.modify_finite_element_add_field_data("NAVIER_STOKES", "U");

    CHKERR m_field.modify_finite_element_add_field_row("NAVIER_STOKES", "P");
    CHKERR m_field.modify_finite_element_add_field_col("NAVIER_STOKES", "P");
    CHKERR m_field.modify_finite_element_add_field_data("NAVIER_STOKES", "P");
    CHKERR m_field.modify_finite_element_add_field_data("NAVIER_STOKES",
                                                        "MESH_NODE_POSITIONS");

    /* CHKERR m_field.add_finite_element("DRAG");
    CHKERR m_field.modify_finite_element_add_field_row("DRAG", "U");
    CHKERR m_field.modify_finite_element_add_field_col("DRAG", "U");
    CHKERR m_field.modify_finite_element_add_field_data("DRAG", "U");

    CHKERR m_field.modify_finite_element_add_field_row("DRAG", "P");
    CHKERR m_field.modify_finite_element_add_field_col("DRAG", "P");
    CHKERR m_field.modify_finite_element_add_field_data("DRAG", "P");
    CHKERR m_field.modify_finite_element_add_field_data("DRAG",
    "MESH_NODE_POSITIONS");
 */

    // Add entities to that element
    CHKERR m_field.add_ents_to_finite_element_by_type(0, MBTET,
                                                      "NAVIER_STOKES");

    /*     Range solid_faces;
        for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
          cout << bit->getName() << endl;
          if (bit->getName().compare(0, 5, "SOLID") == 0) {
            CHKERR m_field.get_moab().get_entities_by_type(
                bit->getMeshset(), MBTRI, solid_faces, true);
          }
        }
        CHKERR m_field.add_ents_to_finite_element_by_type(solid_faces, MBTRI,
       "DRAG");
        */

    // build finite elements
    CHKERR m_field.build_finite_elements();
    // build adjacencies between elements and degrees of freedom
    CHKERR m_field.build_adjacencies(bit_level0);

    // **** BUILD DM **** //
    DM dm;
    DMType dm_name = "DM_NAVIER_STOKES";
    // Register DM problem
    CHKERR DMRegister_MoFEM(dm_name);
    CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
    CHKERR DMSetType(dm, dm_name);
    // Create DM instance
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, dm_name, bit_level0);
    // Configure DM form line command options (DM itself, sness,
    // pre-conditioners, ... )
    CHKERR DMSetFromOptions(dm);
    // Add elements to dm (only one here)
    CHKERR DMMoFEMAddElement(dm, "NAVIER_STOKES");
    // CHKERR DMMoFEMAddElement(dm, "DRAG");

    if (m_field.check_finite_element("PRESSURE_FE"))
      CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
    // if (m_field.check_finite_element("FORCE_FE"))
    //  CHKERR DMMoFEMAddElement(dm, "FORCE_FE");
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // setup the DM
    CHKERR DMSetUp(dm);

    boost::shared_ptr<FEMethod> nullFE;
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> feLhs(
        new VolumeElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> feRhs(
        new VolumeElementForcesAndSourcesCore(m_field));

    feLhs->getRuleHook = NavierStokesElement::VolRule();
    feRhs->getRuleHook = NavierStokesElement::VolRule();
    // feRhs->getRuleHook = FaceRule();

    // boost::shared_ptr<FaceElementForcesAndSourcesCore> dragFe(
    //   new FaceElementForcesAndSourcesCore(m_field));
    // boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideDragFe(
    //   new VolumeElementForcesAndSourcesCoreOnSide(m_field));

    // dragFe->getRuleHook = NavierStokesElement::FaceRule();

    boost::shared_ptr<NavierStokesElement::CommonData> commonData =
        boost::make_shared<NavierStokesElement::CommonData>();
    // CHKERR commonData->getParameters();

    // dragFe->getOpPtrVector().push_back(new OpCalcViscosity());

    // std::map<int, NavierStokesElement::BlockData> setOfBlocksData;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 5, "FLUID") == 0) {
        const int id = bit->getMeshsetId();
        CHKERR m_field.get_moab().get_entities_by_type(
            bit->getMeshset(), MBTET, commonData->setOfBlocksData[id].tEts,
            true);

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

    // for (auto &sit : commonData->setOfBlocksData) {
    //   sideDragFe->getOpPtrVector().push_back(new
    //   OpCalculateVectorFieldGradient<3, 3>("U",
    //                                              commonData->gradDispPtr));
    //   dragFe->getOpPtrVector().push_back(
    //     new OpCalculateInvJacForFace(commonData->invJac));
    //   dragFe->getOpPtrVector().push_back(new
    //   OpSetInvJacH1ForFace(commonData->invJac));
    //   dragFe->getOpPtrVector().push_back(
    //     new OpCalculateScalarFieldValues("P", commonData->pPtr));
    //   dragFe->getOpPtrVector().push_back(
    //     new NavierStokesElement::OpCalcPressureDrag(commonData, sit.second));
    //   dragFe->getOpPtrVector().push_back(
    //     new NavierStokesElement::OpCalcViscousDrag(sideDragFe, commonData,
    //     sit.second));
    // }

    CHKERR NavierStokesElement::setOperators(feLhs, feRhs, "U", "P",
                                             commonData);

    Mat Aij;  // Stiffness matrix
    Vec D, F; //, D0; // Vector of DOFs and the RHS

    {
      CHKERR DMCreateGlobalVector(dm, &D);
      // CHKERR VecDuplicate(D, &D0);
      CHKERR VecDuplicate(D, &F);
      CHKERR DMCreateMatrix(dm, &Aij);
      CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);

      CHKERR VecZeroEntries(F);
      CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
      // CHKERR VecZeroEntries(D);
      // CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      // CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      // CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
      CHKERR MatZeroEntries(Aij);
    }

    // STANDARD DIRICHLET BC
    boost::shared_ptr<DirichletDisplacementBc> dirichlet_bc_ptr(
        new DirichletDisplacementBc(m_field, "U", Aij, D, F));
    if (flg_load_file == PETSC_TRUE) {
      dirichlet_bc_ptr->methodsOp.push_back(
          new TimeForceScale("-load_table", false));
    }
    dirichlet_bc_ptr->methodsOp.push_back(new NavierStokesElement::LoadScale());
    dirichlet_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
    dirichlet_bc_ptr->snes_x = D;

    // VELOCITY DIRICHLET BC
    boost::shared_ptr<DirichletVelocityBc> dirichlet_vel_bc_ptr(
        new DirichletVelocityBc(m_field, "U", Aij, D, F));
    if (flg_load_file == PETSC_TRUE) {
      dirichlet_vel_bc_ptr->methodsOp.push_back(
          new TimeForceScale("-load_table", false));
    }
    dirichlet_vel_bc_ptr->methodsOp.push_back(
        new NavierStokesElement::LoadScale());
    dirichlet_vel_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
    dirichlet_vel_bc_ptr->snes_x = D;

    // PRESSURE DIRICHLET BC
    boost::shared_ptr<DirichletPressureBc> dirichlet_pres_bc_ptr(
        new DirichletPressureBc(m_field, "P", Aij, D, F));
    dirichlet_pres_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
    dirichlet_pres_bc_ptr->snes_x = D;

    CHKERR VecZeroEntries(D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_vel_bc_ptr.get());
    CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
    CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_pres_bc_ptr.get());

    CHKERR VecAssemblyBegin(D);
    CHKERR VecAssemblyEnd(D);
    // CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR DMoFEMMeshToLocalVector(dm, D0, INSERT_VALUES, SCATTER_REVERSE);

    // CHKERR DMoFEMLoopFiniteElements(dm, "NAVIER_STOKES", feLhs);

    // Assemble pressure and traction forces.
    boost::ptr_map<std::string, NeummanForcesSurface> neumann_forces;
    CHKERR MetaNeummanForces::setMomentumFluxOperators(m_field, neumann_forces,
                                                       NULL, "U");
    {
      boost::ptr_map<std::string, NeummanForcesSurface>::iterator mit =
          neumann_forces.begin();
      for (; mit != neumann_forces.end(); mit++) {
        CHKERR DMMoFEMSNESSetFunction(dm, mit->first.c_str(),
                                      &mit->second->getLoopFe(), NULL, NULL);
      }
    }

    CHKERR DMMoFEMSNESSetFunction(dm, "NAVIER_STOKES", feRhs, nullFE, nullFE);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, nullFE, nullFE,
                                  dirichlet_vel_bc_ptr);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, nullFE, nullFE,
                                  dirichlet_bc_ptr);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, nullFE, nullFE,
                                  dirichlet_pres_bc_ptr);

    // Set operators for SNES snes
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, nullFE,
                                  dirichlet_vel_bc_ptr, nullFE);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, nullFE, dirichlet_bc_ptr,
                                  nullFE);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, nullFE,
                                  dirichlet_pres_bc_ptr, nullFE);
    CHKERR DMMoFEMSNESSetJacobian(dm, "NAVIER_STOKES", feLhs, nullFE, nullFE);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, nullFE, nullFE,
                                  dirichlet_vel_bc_ptr);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, nullFE, nullFE,
                                  dirichlet_bc_ptr);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, nullFE, nullFE,
                                  dirichlet_pres_bc_ptr);

    // Set calculation of the right hand side vector for SNES snes

    // CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    // CHKERR VecAssemblyBegin(F);
    // CHKERR VecAssemblyEnd(F);
    // CHKERR VecScale(F, -1);

    // **** SOLVE **** //

    // START

    SNES snes;

    SnesCtx *snes_ctx;
    // create snes nonlinear solver
    {
      CHKERR SNESCreate(PETSC_COMM_WORLD, &snes);
      CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
      CHKERR SNESSetFunction(snes, F, SnesRhs, snes_ctx);
      CHKERR SNESSetJacobian(snes, Aij, Aij, SnesMat, snes_ctx);
      CHKERR SNESSetFromOptions(snes);
    }

    VectorDouble3 totalDrag(3);

    SNESConvergedReason snes_reason;
    int number_of_diverges = 0;
    int desired_iteration_number = 5; // TODO: improve adaptivity
    double step_size = lAmbda0 / nbSubSteps;

    // VectorDouble reactions(3);

    boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProcPtr;

    if (flg_load_file == PETSC_TRUE)
      NavierStokesElement::LoadScale::lambda = lAmbda0;
    else
      NavierStokesElement::LoadScale::lambda = 0;

    for (int ss = 0; ss < nbSubSteps; ++ss) {
      //
      // for testing loading //TODO: Implement load history config file
      // if(ss < nbSubSteps/3)
      //   LoadScale::lambda += lambda0 / nbSubSteps;
      // else
      //   LoadScale::lambda -= lambda0 / nbSubSteps;

      // CHKERR VecCopy(D, D0);
      // CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
      // CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);

      if (flg_load_file == PETSC_TRUE) {
        dirichlet_vel_bc_ptr->ts_t = ss;
        dirichlet_bc_ptr->ts_t = ss;
      } else
        NavierStokesElement::LoadScale::lambda += step_size;

      CHKERR PetscPrintf(PETSC_COMM_WORLD, "Step: %d | Lambda: %6.4e  \n", ss,
                         NavierStokesElement::LoadScale::lambda);

      dirichlet_vel_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
      dirichlet_vel_bc_ptr->snes_x = D;
      dirichlet_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
      dirichlet_bc_ptr->snes_x = D;
      dirichlet_pres_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
      dirichlet_pres_bc_ptr->snes_x = D;

      CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_vel_bc_ptr.get());
      CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
      CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_pres_bc_ptr.get());
      CHKERR VecAssemblyBegin(D);
      CHKERR VecAssemblyEnd(D);
      CHKERR SNESSolve(snes, PETSC_NULL, D);

      CHKERR SNESGetConvergedReason(snes, &snes_reason);
      int its;
      CHKERR SNESGetIterationNumber(snes, &its);
      // CHKERR PetscPrintf(PETSC_COMM_WORLD,
      //                  "%s Number of nonlinear iterations = %D\n",
      //                    SNESConvergedReasons[snes_reason], its);
      // adaptivity

      if (snes_reason < 0) {
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Nonlinear solver diverged!\n");
        // if (isAtomTest)
        //  SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
        //          "atom test diverged!");
        if (number_of_diverges < maxDivStep) {
          // CHKERR VecCopy(D0, D);
          NavierStokesElement::LoadScale::lambda -= 2 * step_size;
          NavierStokesElement::LoadScale::lambda += stepRed * (step_size);
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "Reducing step... \n");
          number_of_diverges++;
          ss--;
          continue;

        } else {
          break;
        }
      }
      // ADAPTIVE STEPPING
      // const double frac = (double)desired_iteration_number / its;
      // step_size *= sqrt(frac);

      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

      // CHKERR VecAYPX(D0, -1., D);
      // CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
      // CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);

      // CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", feUpdate);
      if (ss % outPutStep == 0) {
        // for postprocessing:
        if (!postProcPtr) {

          postProcPtr =
              boost::make_shared<PostProcVolumeOnRefinedMesh>(m_field);
          CHKERR postProcPtr->generateReferenceElementMesh();
          CHKERR postProcPtr->addFieldValuesPostProc("U");
          CHKERR postProcPtr->addFieldValuesPostProc("P");
          CHKERR postProcPtr->addFieldValuesPostProc("MESH_NODE_POSITIONS");

          // postProcPtr->getOpPtrVector().push_back(
          //     new OpCalculateScalarFieldValues("PHASE_FIELD",
          //     commonData->cPtr));
          // postProcPtr->getOpPtrVector().push_back(
          //     new OpCalculateVectorFieldGradient<3, 3>("U",
          //                                             commonData->gradDispPtr));
          // for (auto &sit : commonData->setOfBlocksData) {
          //   postProcPtr->getOpPtrVector().push_back(new OpPostProcStress(
          //       postProcPtr->postProcMesh, postProcPtr->mapGaussPts,
          //       commonData, sit.second));
          // }

          // loop over blocks
          // for (auto &sit : commonData->setOfBlocksData) {

          //   // postProcPtr->getOpPtrVector().push_back(
          //   //    new OpCalculateScalarFieldValues("P", commonData->pPtr));
          //   postProcPtr->getOpPtrVector().push_back(
          //       new OpCalculateVectorFieldGradient<3, 3>(
          //           "U", commonData->gradDispPtr));
          //   postProcPtr->getOpPtrVector().push_back(
          //       new NavierStokesElement::OpPostProcVorticity(
          //           postProcPtr->postProcMesh, postProcPtr->mapGaussPts,
          //           commonData, sit.second));
          // }
        }

        CHKERR DMoFEMLoopFiniteElements(dm, "NAVIER_STOKES", postProcPtr);
        string out_file_name;
        std::ostringstream stm;
        stm << "out_" << ss << ".h5m";
        out_file_name = stm.str();
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
                           out_file_name.c_str());
        CHKERR postProcPtr->postProcMesh.write_file(
            out_file_name.c_str(), "MOAB", "PARALLEL=WRITE_PART");

        // commonData->pressureDrag = commonData->viscousDrag =
        // commonData->totalDrag = 0.0;
        // for (int dd = 0; dd != 3; dd++) {
        //   commonData->pressureDrag[dd] = commonData->viscousDrag[dd] = 0.0;
        // }
        // CHKERR DMoFEMLoopFiniteElements(dm, "DRAG", dragFe);
        // for (int dd = 0; dd != 3; dd++) {
        //   totalDrag[dd] =
        //       commonData->pressureDrag[dd] + commonData->viscousDrag[dd];
        // }
        // /// SUM ALL WITH MPI
        // cout << "pressure drag: " << commonData->pressureDrag << endl;
        // cout << "viscous drag: " << commonData->viscousDrag << endl;
        // cout << "total drag: " << totalDrag << endl;
      }
    }

    // CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    // CHKERR DMoFEMLoopFiniteElements(dm, "NAVIER_STOKES", &post_proc);
    // PetscPrintf(PETSC_COMM_WORLD, "Output file: %s\n", "out.h5m");
    // CHKERR post_proc.postProcMesh.write_file("out.h5m", "MOAB",
    //                                          "PARALLEL=WRITE_PART");

    CHKERR SNESDestroy(&snes);
    CHKERR MatDestroy(&Aij);
    CHKERR VecDestroy(&D);
    // CHKERR VecDestroy(&D0);
    CHKERR VecDestroy(&F);
    CHKERR DMDestroy(&dm);

    // MoFEMFunctionReturn(0);

    // // STOP

    // // Create solver and link it to DM
    // SNES solver;
    // CHKERR SNESCreate(PETSC_COMM_WORLD, &solver);
    // CHKERR SNESSetFromOptions(solver);
    // CHKERR SNESSetDM(solver, dm);
    // // Set-up solver, is type of solver and pre-conditioners
    // CHKERR SNESSetUp(solver);
    // // At solution process, KSP solver using DM creates matrices, Calculate
    // // values of the left han d side and the right hand side vector. then
    // // solves system of equations. Results are stored in vector D.
    // CHKERR SNESSolve(solver, F, D);

    // SnesCtx *snes_ctx;
    // CHKERR DMMoFEMGetSnesCtx (dm, &snes_ctx);
    // CHKERR SnesRhs(solver, D, F, snes_ctx);
    // //VecView(F, PETSC_VIEWER_STDOUT_WORLD);

    // //CHKERR VecAXPY(D, 1., D0);

    // // VecView(F, PETSC_VIEWER_STDOUT_WORLD);
    // // CHKERR VecView(d, PETSC_VIEWER_STDOUT_WORLD);    // Print out the
    // results

    // CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    // // Save data on mesh
    // CHKERR post_proc.generateReferenceElementMesh();
    // CHKERR post_proc.addFieldValuesPostProc("U");
    // CHKERR post_proc.addFieldValuesPostProc("P");
    // CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");

    // CHKERR DMoFEMLoopFiniteElements(dm, "NAVIER_STOKES", &post_proc);
    // PetscPrintf(PETSC_COMM_WORLD, "Output file: %s\n", "out.h5m");
    // CHKERR post_proc.postProcMesh.write_file("out.h5m", "MOAB",
    //                                          "PARALLEL=WRITE_PART");

    // CHKERR MatDestroy(&Aij);
    // CHKERR VecDestroy(&D);
    // //CHKERR VecDestroy(&D0);
    // CHKERR VecDestroy(&F);
    // CHKERR DMDestroy(&dm);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();
  return 0;
}