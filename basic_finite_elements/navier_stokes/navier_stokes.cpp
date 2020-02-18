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
#include <NavierStokesProblem.hpp>

using namespace boost::numeric;
using namespace MoFEM;
using namespace std;

static char help[] = "-my_block_config set block data\n"
                     "\n";

double NavierStokesElement::LoadScale::lambda = 1;

int main(int argc, char *argv[]) {

  const char param_file[] = "param_file.petsc";
  // Initialise MoFEM
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

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

    int desired_iteration_number = 6;

    int nbSubSteps = 1;   // default number of steps
    double lambda0 = 0;   // initial lambda
    double lambda1 = 1.0; // multiplier
    double stepRed = 0.5; // stepsize reduction while diverging
    int maxDivStep = 10;  // maximum number of diverged steps
    int outPutStep = 1;   // how often post processing data is saved to h5m file
    int restartStep = 1;  // how often post restart data is saved to h5m file

    double length_scale = 0;

    double adapt_step_exp = 0.5;

    double step_size = 0.0;

    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool flg_test = PETSC_FALSE;

    PetscBool stokes_flow = PETSC_FALSE;
    PetscBool discont_pressure = PETSC_FALSE;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "NAVIER_STOKES problem",
                             "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_mesh_file);
    // Set approximation order
    CHKERR PetscOptionsInt("-my_order_p", "approximation order_p", "", order_p,
                           &order_p, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_order_u", "approximation order_u", "", order_u,
                           &order_u, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_ho_levels", "number of ho levels", "",
                           nb_ho_levels, &nb_ho_levels, PETSC_NULL);

    CHKERR PetscOptionsInt(
        "-output_step", "frequency how often results are dumped on hard disk",
        "", 1, &outPutStep, NULL);
    CHKERR PetscOptionsInt("-my_restart_freq",
                           "frequency how often restart file is written", "", 1,
                           &restartStep, NULL);

    CHKERR PetscOptionsInt("-my_step_num", "number of steps", "", nbSubSteps,
                           &nbSubSteps, PETSC_NULL);

    CHKERR PetscOptionsInt("-my_desired_it_num", "desired number of iterations",
                           "", desired_iteration_number,
                           &desired_iteration_number, PETSC_NULL);

    CHKERR PetscOptionsInt("-my_step_max_div", "number of steps", "",
                           maxDivStep, &maxDivStep, PETSC_NULL);

    CHKERR PetscOptionsScalar("-my_lambda", "lambda", "", lambda1, &lambda1,
                              PETSC_NULL);
    CHKERR PetscOptionsScalar("-my_lambda_init", "initial lambda", "", lambda0,
                              &lambda0, PETSC_NULL);
    CHKERR PetscOptionsScalar("-my_lambda_step", "initial step in lambda", "",
                              step_size, &step_size, PETSC_NULL);

    CHKERR PetscOptionsScalar("-my_adapt_step_exp", "adaptive step exponent",
                              "", adapt_step_exp, &adapt_step_exp, PETSC_NULL);
    CHKERR PetscOptionsScalar("-my_length_scale", "length scale", "",
                              length_scale, &length_scale, PETSC_NULL);

    CHKERR PetscOptionsString("-my_load_table", "load history file name", "",
                              "load_table.txt", load_file_name, 255,
                              &flg_load_file);

    CHKERR PetscOptionsScalar("-my_step_red", "step reduction when diverge", "",
                              stepRed, &stepRed, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned", "is_partitioned?", "",
                            is_partitioned, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_stokes_flow", "stokes flow", "", stokes_flow,
                            &stokes_flow, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_discont_pressure", "discontinuous pressure",
                            "", discont_pressure, &discont_pressure,
                            PETSC_NULL);

    // Set testing (used by CTest)
    CHKERR PetscOptionsBool("-my_test", "if true is ctest", "", flg_test,
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

    bool is_restart = string(mesh_file_name).compare(0, 8, "restart_") == 0;

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // Print boundary conditions and material parameters
    MeshsetsManager *meshsets_mng_ptr;
    CHKERR m_field.getInterface(meshsets_mng_ptr);
    CHKERR meshsets_mng_ptr->printDisplacementSet();
    CHKERR meshsets_mng_ptr->printPressureSet();
    // CHKERR meshsets_mng_ptr->printForceSet();
    // CHKERR meshsets_mng_ptr->printMaterialsSet();

    bool mesh_has_tets = false;
    bool mesh_has_prisms = false;
    int nb_tets = 0;
    int nb_prisms = 0;
    CHKERR moab.get_number_entities_by_type(0, MBTET, nb_tets, true);
    CHKERR moab.get_number_entities_by_type(0, MBPRISM, nb_prisms, true);
    mesh_has_tets = nb_tets > 0;
    mesh_has_prisms = nb_prisms > 0;

    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    if (!is_restart) {
      // **** ADD FIELDS **** //
      CHKERR m_field.add_field("VELOCITY", H1, AINSWORTH_LEGENDRE_BASE, 3);
      if (!discont_pressure)
        CHKERR m_field.add_field("PRESSURE", H1, AINSWORTH_LEGENDRE_BASE, 1);
      else
        CHKERR m_field.add_field("PRESSURE", L2, AINSWORTH_LEGENDRE_BASE, 1);
      CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1,
                               AINSWORTH_LEGENDRE_BASE, 3);

      CHKERR m_field.add_ents_to_field_by_dim(0, 3, "VELOCITY");
      CHKERR m_field.add_ents_to_field_by_dim(0, 3, "PRESSURE");
      CHKERR m_field.add_ents_to_field_by_dim(0, 3, "MESH_NODE_POSITIONS");
    }

    CHKERR m_field.set_field_order(0, MBVERTEX, "VELOCITY", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "VELOCITY", order_u);
    CHKERR m_field.set_field_order(0, MBTRI, "VELOCITY", order_u);
    CHKERR m_field.set_field_order(0, MBTET, "VELOCITY", order_u);
    CHKERR m_field.set_field_order(0, MBQUAD, "VELOCITY", order_u);
    CHKERR m_field.set_field_order(0, MBPRISM, "VELOCITY", order_u);

    CHKERR m_field.set_field_order(0, MBVERTEX, "PRESSURE", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "PRESSURE", order_p);
    CHKERR m_field.set_field_order(0, MBTRI, "PRESSURE", order_p);
    CHKERR m_field.set_field_order(0, MBTET, "PRESSURE", order_p);
    CHKERR m_field.set_field_order(0, MBQUAD, "PRESSURE", order_p);
    CHKERR m_field.set_field_order(0, MBPRISM, "PRESSURE", order_p);

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 5, "INT_SOLID") == 0) {
        Range ents;
        CHKERR m_field.get_moab().get_entities_by_type(bit->getMeshset(), MBTRI,
                                                       ents, true);

        std::vector<Range> levels(nb_ho_levels);
        for (int ll = 0; ll != nb_ho_levels; ll++) {
          Range verts;
          CHKERR m_field.get_moab().get_connectivity(ents, verts, true);
          for (auto d : {1, 2, 3}) {
            CHKERR m_field.get_moab().get_adjacencies(verts, d, false, ents,
                                                      moab::Interface::UNION);
          }
          levels[ll] = subtract(ents, ents.subset_by_type(MBVERTEX));
        }

        for (int ll = nb_ho_levels - 1; ll >= 1; ll--) {
          levels[ll] = subtract(levels[ll], levels[ll - 1]);
        }

        int add_order = 1;
        for (int ll = nb_ho_levels - 1; ll >= 0; ll--) {
          CHKERR m_field.set_field_order(levels[ll], "VELOCITY",
                                         order_u + add_order);
          CHKERR m_field.set_field_order(levels[ll], "PRESSURE",
                                         order_p + add_order);
          ++add_order;
        }
      }
    }

    // Set 2nd order of approximation of geometry
    auto setting_second_order_geometry = [&m_field]() {
      MoFEMFunctionBegin;
      // Setting geometry order everywhere
      Range tets, edges;
      CHKERR m_field.get_moab().get_entities_by_dimension(0, 3, tets);
      CHKERR m_field.get_moab().get_adjacencies(tets, 1, false, edges,
                                                moab::Interface::UNION);

      CHKERR m_field.set_field_order(edges, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);
      MoFEMFunctionReturn(0);
    };
    CHKERR setting_second_order_geometry();
    // CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.build_fields();

    Projection10NodeCoordsOnField ent_method_material(m_field,
                                                      "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);
    // setup elements for loading
    CHKERR MetaNeummanForces::addNeumannBCElements(m_field, "VELOCITY");
    CHKERR MetaNodalForces::addElement(m_field, "VELOCITY");
    CHKERR MetaEdgeForces::addElement(m_field, "VELOCITY");

    // **** ADD ELEMENTS **** //

    if (!is_restart) {
      // Add finite element (this defines element, declaration comes later)
      CHKERR m_field.add_finite_element("NAVIER_STOKES");
      CHKERR m_field.modify_finite_element_add_field_row("NAVIER_STOKES",
                                                         "VELOCITY");
      CHKERR m_field.modify_finite_element_add_field_col("NAVIER_STOKES",
                                                         "VELOCITY");
      CHKERR m_field.modify_finite_element_add_field_data("NAVIER_STOKES",
                                                          "VELOCITY");

      CHKERR m_field.modify_finite_element_add_field_row("NAVIER_STOKES",
                                                         "PRESSURE");
      CHKERR m_field.modify_finite_element_add_field_col("NAVIER_STOKES",
                                                         "PRESSURE");
      CHKERR m_field.modify_finite_element_add_field_data("NAVIER_STOKES",
                                                          "PRESSURE");
      CHKERR m_field.modify_finite_element_add_field_data(
          "NAVIER_STOKES", "MESH_NODE_POSITIONS");

      CHKERR m_field.add_finite_element("DRAG");
      CHKERR m_field.modify_finite_element_add_field_row("DRAG", "VELOCITY");
      CHKERR m_field.modify_finite_element_add_field_col("DRAG", "VELOCITY");
      CHKERR m_field.modify_finite_element_add_field_data("DRAG", "VELOCITY");

      CHKERR m_field.modify_finite_element_add_field_row("DRAG", "PRESSURE");
      CHKERR m_field.modify_finite_element_add_field_col("DRAG", "PRESSURE");
      CHKERR m_field.modify_finite_element_add_field_data("DRAG", "PRESSURE");
      CHKERR m_field.modify_finite_element_add_field_data(
          "DRAG", "MESH_NODE_POSITIONS");
      // Add entities to that element
      CHKERR m_field.add_ents_to_finite_element_by_dim(0, 3, "NAVIER_STOKES");
    }

    boost::shared_ptr<NavierStokesElement::CommonData> commonData =
        boost::make_shared<NavierStokesElement::CommonData>();

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 9, "MAT_FLUID") == 0) {
        const int id = bit->getMeshsetId();
        CHKERR m_field.get_moab().get_entities_by_dimension(
            bit->getMeshset(), 3, commonData->setOfBlocksData[id].eNts, true);

        std::vector<double> attributes;
        bit->getAttributes(attributes);
        if (attributes.size() < 2) {
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                   "should be at least 2 attributes but is %d",
                   attributes.size());
        }
        commonData->setOfBlocksData[id].iD = id;
        commonData->setOfBlocksData[id].fluidViscosity = attributes[0];
        commonData->setOfBlocksData[id].fluidDensity = attributes[1];
      }
    }

    Range solid_faces;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 5, "INT_SOLID") == 0) {
        Range tets, tet;
        const int id = bit->getMeshsetId();
        CHKERR m_field.get_moab().get_entities_by_type(
            bit->getMeshset(), MBTRI, commonData->setOfFacesData[id].eNts,
            true);
        solid_faces.merge(commonData->setOfFacesData[id].eNts);
        CHKERR moab.get_adjacencies(commonData->setOfFacesData[id].eNts, 3,
                                    true, tets, moab::Interface::UNION);
        tet = Range(tets.front(), tets.front());
        for (auto &bit : commonData->setOfBlocksData) {
          if (bit.second.eNts.contains(tet)) {
            commonData->setOfFacesData[id].fluidViscosity =
                bit.second.fluidViscosity;
            commonData->setOfFacesData[id].fluidDensity =
                bit.second.fluidDensity;
            commonData->setOfFacesData[id].iD = id;
            break;
          }
        }
        if (commonData->setOfFacesData[id].fluidViscosity < 0) {
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                  "Cannot find a fluid block adjacent to a given solid face");
        }
      }
    }

    EntityHandle tree_root;
    AdaptiveKDTree myTree(&moab);
    BoundBox box;

    if (!solid_faces.empty()) {
      // solid_faces.print();
      CHKERR m_field.add_ents_to_finite_element_by_type(solid_faces, MBTRI,
                                                        "DRAG");
      CHKERR myTree.build_tree(solid_faces, &tree_root);
    } else {
      Range tets;
      CHKERR m_field.get_moab().get_entities_by_dimension(0, 3, tets);
      CHKERR myTree.build_tree(tets, &tree_root);
    }

    if (fabs(length_scale) < 1e-12) {
      // get the overall bounding box corners
      CHKERR myTree.get_bounding_box(box, &tree_root);
      length_scale =
          max(max(box.bMax[0] - box.bMin[0], box.bMax[1] - box.bMin[1]),
              box.bMax[2] - box.bMin[2]);
    }

    CHKERR m_field.getInterface<FieldBlas>()->fieldScale((1.0 / length_scale),
                                                         "MESH_NODE_POSITIONS");
    double rho;
    double mu;
    double U;
    double L = length_scale;
    double Re;
    double P;

    auto scale_problem = [&](double U, double L, double P) {
      MoFEMFunctionBegin;
      CHKERR m_field.getInterface<FieldBlas>()->fieldScale(
          L, "MESH_NODE_POSITIONS");
      CHKERR m_field.getInterface<FieldBlas>()->fieldScale(U, "VELOCITY");
      CHKERR m_field.getInterface<FieldBlas>()->fieldScale(P, "PRESSURE");
      MoFEMFunctionReturn(0);
    };

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
    CHKERR DMMoFEMAddElement(dm, "DRAG");

    if (m_field.check_finite_element("PRESSURE_FE"))
      CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
    // if (m_field.check_finite_element("FORCE_FE"))
    //  CHKERR DMMoFEMAddElement(dm, "FORCE_FE");
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // setup the DM
    CHKERR DMSetUp(dm);

    boost::shared_ptr<FEMethod> null_fe;
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_lhs_ptr(
        new VolumeElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_rhs_ptr(
        new VolumeElementForcesAndSourcesCore(m_field));

    fe_lhs_ptr->getRuleHook = NavierStokesElement::VolRule();
    fe_rhs_ptr->getRuleHook = NavierStokesElement::VolRule();

    boost::shared_ptr<FatPrism> prism_fe_lhs_ptr(new FatPrism(m_field));
    boost::shared_ptr<FatPrism> prism_fe_rhs_ptr(new FatPrism(m_field));

    boost::shared_ptr<FaceElementForcesAndSourcesCore> drag_fe_ptr(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_drag_fe_ptr(
        new VolumeElementForcesAndSourcesCoreOnSide(m_field));

    drag_fe_ptr->getRuleHook = NavierStokesElement::FaceRule();

    if (!solid_faces.empty()) {
      NavierStokesElement::setCalcDragOperators(drag_fe_ptr, side_drag_fe_ptr,
                                                "NAVIER_STOKES", "VELOCITY",
                                                "PRESSURE", commonData);
    }

    if (mesh_has_tets) {
      if (stokes_flow) {
        CHKERR NavierStokesElement::setStokesOperators(
            fe_rhs_ptr, fe_lhs_ptr, "VELOCITY", "PRESSURE", commonData);
      } else {
        CHKERR NavierStokesElement::setNavierStokesOperators(
            fe_rhs_ptr, fe_lhs_ptr, "VELOCITY", "PRESSURE", commonData);
      }
    }
    if (mesh_has_prisms) {
      if (stokes_flow) {
        CHKERR NavierStokesElement::setStokesOperators(
            prism_fe_rhs_ptr, prism_fe_lhs_ptr, "VELOCITY", "PRESSURE",
            commonData, MBPRISM);
      } else {
        CHKERR NavierStokesElement::setNavierStokesOperators(
            prism_fe_rhs_ptr, prism_fe_lhs_ptr, "VELOCITY", "PRESSURE",
            commonData, MBPRISM);
      }
    }

    Mat Aij;      // Stiffness matrix
    Vec D, D0, F; //, D0; // Vector of DOFs and the RHS

    CHKERR DMCreateGlobalVector(dm, &D);
    CHKERR VecDuplicate(D, &D0);
    CHKERR VecDuplicate(D, &F);
    CHKERR DMCreateMatrix(dm, &Aij);
    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);

    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);

    if (is_restart) {
      CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);
    } else {
      CHKERR VecZeroEntries(D);
    }
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR VecZeroEntries(D0);
    CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR MatZeroEntries(Aij);

    // CHKERR VecView(D, PETSC_VIEWER_STDOUT_WORLD);

    // STANDARD DIRICHLET BC
    boost::shared_ptr<DirichletDisplacementBc> dirichlet_bc_ptr(
        new DirichletDisplacementBc(m_field, "VELOCITY", Aij, D, F));
    if (flg_load_file == PETSC_TRUE) {
      dirichlet_bc_ptr->methodsOp.push_back(
          new TimeForceScale("-load_table", false));
    }
    dirichlet_bc_ptr->methodsOp.push_back(new NavierStokesElement::LoadScale());
    // dirichlet_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
    dirichlet_bc_ptr->snes_x = D;

    // VELOCITY DIRICHLET BC
    boost::shared_ptr<DirichletVelocityBc> dirichlet_vel_bc_ptr(
        new DirichletVelocityBc(m_field, "VELOCITY", Aij, D, F));
    if (flg_load_file == PETSC_TRUE) {
      dirichlet_vel_bc_ptr->methodsOp.push_back(
          new TimeForceScale("-load_table", false));
    }
    dirichlet_vel_bc_ptr->methodsOp.push_back(
        new NavierStokesElement::LoadScale());
    // dirichlet_vel_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
    dirichlet_vel_bc_ptr->snes_x = D;

    // PRESSURE DIRICHLET BC
    boost::shared_ptr<DirichletSetFieldFromBlock> dirichlet_pres_bc_ptr(
        new DirichletSetFieldFromBlock(m_field, "PRESSURE", "PRESS", Aij, D,
                                       F));
    // dirichlet_pres_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
    dirichlet_pres_bc_ptr->snes_x = D;

    // CHKERR DMoFEMPreProcessFiniteElements(dm,
    // dirichlet_vel_bc_ptr.get()); CHKERR
    // DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get()); CHKERR
    // DMoFEMPreProcessFiniteElements(dm, dirichlet_pres_bc_ptr.get());

    // CHKERR VecAssemblyBegin(D);
    // CHKERR VecAssemblyEnd(D);

    // CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES,
    // SCATTER_REVERSE);

    // Assemble pressure and traction forces.
    boost::ptr_map<std::string, NeummanForcesSurface> neumann_forces;
    CHKERR MetaNeummanForces::setMomentumFluxOperators(m_field, neumann_forces,
                                                       NULL, "VELOCITY");
    {
      boost::ptr_map<std::string, NeummanForcesSurface>::iterator mit =
          neumann_forces.begin();
      for (; mit != neumann_forces.end(); mit++) {
        mit->second->methodsOp.push_back(new NavierStokesElement::LoadScale());
        CHKERR DMMoFEMSNESSetFunction(dm, mit->first.c_str(),
                                      &mit->second->getLoopFe(), NULL, NULL);
      }
    }

    CHKERR DMMoFEMSNESSetFunction(dm, "NAVIER_STOKES", fe_rhs_ptr, null_fe,
                                  null_fe);
    CHKERR DMMoFEMSNESSetFunction(dm, "NAVIER_STOKES", prism_fe_rhs_ptr,
                                  null_fe, null_fe);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, null_fe, null_fe,
                                  dirichlet_vel_bc_ptr);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, null_fe, null_fe,
                                  dirichlet_bc_ptr);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, null_fe, null_fe,
                                  dirichlet_pres_bc_ptr);

    // Set operators for SNES snes
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, null_fe,
                                  dirichlet_vel_bc_ptr, null_fe);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, null_fe, dirichlet_bc_ptr,
                                  null_fe);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, null_fe,
                                  dirichlet_pres_bc_ptr, null_fe);
    CHKERR DMMoFEMSNESSetJacobian(dm, "NAVIER_STOKES", fe_lhs_ptr, null_fe,
                                  null_fe);
    CHKERR DMMoFEMSNESSetJacobian(dm, "NAVIER_STOKES", prism_fe_lhs_ptr,
                                  null_fe, null_fe);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, null_fe, null_fe,
                                  dirichlet_vel_bc_ptr);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, null_fe, null_fe,
                                  dirichlet_bc_ptr);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, null_fe, null_fe,
                                  dirichlet_pres_bc_ptr);

    // **** SOLVE **** //

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

    Vec G;
    CHKERR VecCreateMPI(PETSC_COMM_WORLD, 3, 3, &G);
    VectorDouble3 vecGlobalDrag = VectorDouble3(3);

    auto compute_global_drag = [&G](VectorDouble3 &loc, VectorDouble3 &res) {
      MoFEMFunctionBegin;

      CHKERR VecZeroEntries(G);
      CHKERR VecAssemblyBegin(G);
      CHKERR VecAssemblyEnd(G);

      int ind[3] = {0, 1, 2};
      CHKERR VecSetValues(G, 3, ind, loc.data().begin(), ADD_VALUES);
      CHKERR VecAssemblyBegin(G);
      CHKERR VecAssemblyEnd(G);

      CHKERR VecGetValues(G, 3, ind, res.data().begin());

      MoFEMFunctionReturn(0);
    };

    SNESConvergedReason snes_reason;
    int number_of_diverges = 0;

    if (fabs(step_size) < 1e-14)
      step_size = lambda1 / nbSubSteps;

    boost::shared_ptr<PostProcVolumeOnRefinedMesh> post_proc_ptr;
    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_drag_ptr;

    boost::shared_ptr<PostProcFatPrismOnRefinedMesh> prism_post_proc_ptr;

    // for postprocessing:

    post_proc_ptr = boost::make_shared<PostProcVolumeOnRefinedMesh>(m_field);

    if (mesh_has_tets) {
      CHKERR post_proc_ptr->generateReferenceElementMesh();
      CHKERR post_proc_ptr->addFieldValuesPostProc("VELOCITY");
      CHKERR post_proc_ptr->addFieldValuesPostProc("PRESSURE");
      CHKERR post_proc_ptr->addFieldValuesPostProc("MESH_NODE_POSITIONS");
      CHKERR post_proc_ptr->addFieldValuesGradientPostProc("VELOCITY");

      // loop over blocks
      for (auto &sit : commonData->setOfBlocksData) {

        post_proc_ptr->getOpPtrVector().push_back(
            new OpCalculateVectorFieldGradient<3, 3>("VELOCITY",
                                                     commonData->gradDispPtr));
        post_proc_ptr->getOpPtrVector().push_back(
            new NavierStokesElement::OpPostProcVorticity(
                post_proc_ptr->postProcMesh, post_proc_ptr->mapGaussPts,
                commonData, sit.second));
      }
    }

    prism_post_proc_ptr =
        boost::make_shared<PostProcFatPrismOnRefinedMesh>(m_field);

    if (mesh_has_prisms) {
      CHKERR prism_post_proc_ptr->generateReferenceElementMesh();
      CHKERR prism_post_proc_ptr->addFieldValuesPostProc("VELOCITY");
      CHKERR prism_post_proc_ptr->addFieldValuesPostProc("PRESSURE");
      CHKERR prism_post_proc_ptr->addFieldValuesPostProc("MESH_NODE_POSITIONS");
      CHKERR prism_post_proc_ptr->addFieldValuesGradientPostProc("VELOCITY");

      boost::shared_ptr<MatrixDouble> inv_jac_ptr(new MatrixDouble);
      prism_post_proc_ptr->getOpPtrVector().push_back(
          new OpCalculateInvJacForFatPrism(inv_jac_ptr));
      prism_post_proc_ptr->getOpPtrVector().push_back(
          new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
    }

    if (!solid_faces.empty()) {

      post_proc_drag_ptr =
          boost::make_shared<PostProcFaceOnRefinedMesh>(m_field);
      CHKERR post_proc_drag_ptr->generateReferenceElementMesh();

      CHKERR NavierStokesElement::setPostProcDragOperators(
          post_proc_drag_ptr, side_drag_fe_ptr, "NAVIER_STOKES", "VELOCITY",
          "PRESSURE", commonData);
    }

    double lambda = lambda0;
    double frac = 0;
    // if (flg_load_file == PETSC_TRUE)
    //   NavierStokesElement::LoadScale::lambda = lambda0;
    // else
    //   NavierStokesElement::LoadScale::lambda = 0;

    // CHKERR VecView(D, PETSC_VIEWER_STDOUT_WORLD);

    int ss = 0;
    if (is_restart) {
      string str(mesh_file_name);
      stringstream sstream(str.substr(8));
      sstream >> ss;
      ss++;
    }

    CHKERR PetscPrintf(PETSC_COMM_WORLD, "L: %6.4e\n", L);

    while (lambda < lambda1 - 1e-12) {

      if (flg_load_file == PETSC_TRUE) {
        dirichlet_vel_bc_ptr->ts_t = ss;
        dirichlet_bc_ptr->ts_t = ss;
      } else
        lambda += step_size;

      for (auto &bit : commonData->setOfBlocksData) {
        rho = bit.second.fluidDensity;
        mu = bit.second.fluidViscosity;
        U = lambda;
        Re = rho * U * L / mu;
        bit.second.dimScales.length = L;
        bit.second.dimScales.velocity = U;
        bit.second.dimScales.reNumber = Re;
        if (stokes_flow) {
          P = mu * U / L;
          bit.second.dimScales.pressure = P;
          bit.second.inertiaCoef = 0;
          bit.second.viscousCoef = 1.0;
        } else {
          P = rho * U * U;
          bit.second.dimScales.pressure = P;
          bit.second.inertiaCoef = 1.0;
          bit.second.viscousCoef = 1.0 / Re;
        }
        NavierStokesElement::LoadScale::lambda /= P;
      }

      CHKERR PetscPrintf(PETSC_COMM_WORLD,
                         "Step: %d | Size: %6.4e | U: %6.4e | Re: %6.4e \n", ss,
                         step_size, U, Re);

      CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_vel_bc_ptr.get());
      CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
      CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_pres_bc_ptr.get());
      CHKERR VecAssemblyBegin(D);
      CHKERR VecAssemblyEnd(D);

      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

      CHKERR SNESSolve(snes, PETSC_NULL, D);

      CHKERR SNESGetConvergedReason(snes, &snes_reason);
      int its;
      CHKERR SNESGetIterationNumber(snes, &its);

      if (snes_reason < 0) {

        if (number_of_diverges < maxDivStep) {
          lambda -= step_size;
          step_size *= stepRed;

          CHKERR PetscPrintf(PETSC_COMM_WORLD, "Reducing step... \n");
          number_of_diverges++;

          CHKERR VecCopy(D0, D);
          CHKERR VecAssemblyBegin(D);
          CHKERR VecAssemblyEnd(D);
          CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
          CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

          continue;

        } else {
          break;
        }
      }

      number_of_diverges = 0;

      // ADAPTIVE STEPPING
      if (its > 0)
        frac = (double)desired_iteration_number / its;
      else
        frac = (double)desired_iteration_number / (its + 1);
      step_size *= pow(frac, adapt_step_exp);

      CHKERR VecCopy(D, D0);

      CHKERR VecAssemblyBegin(D0);
      CHKERR VecAssemblyEnd(D0);
      CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);

      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

      if (ss % outPutStep == 0) {

        CHKERR scale_problem(U, L, P);

        CHKERR DMoFEMLoopFiniteElements(dm, "NAVIER_STOKES", post_proc_ptr);
        CHKERR DMoFEMLoopFiniteElements(dm, "NAVIER_STOKES",
                                        prism_post_proc_ptr);

        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Write output file ...\n");

        if (mesh_has_tets)
          CHKERR post_proc_ptr->postProcMesh.write_file("out.h5m", "MOAB",
                                                        "PARALLEL=WRITE_PART");
        if (mesh_has_prisms)
          CHKERR prism_post_proc_ptr->postProcMesh.write_file(
              "out_prism.h5m", "MOAB", "PARALLEL=WRITE_PART");

        if (!solid_faces.empty()) {

          CHKERR DMoFEMLoopFiniteElements(dm, "DRAG", post_proc_drag_ptr);
          string out_file_name;
          std::ostringstream stm;
          stm << "out_drag_" << ss << ".h5m";
          out_file_name = stm.str();
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
                             out_file_name.c_str());
          CHKERR post_proc_drag_ptr->postProcMesh.write_file(
              out_file_name.c_str(), "MOAB", "PARALLEL=WRITE_PART");

          commonData->pressureDragForce.clear();
          commonData->viscousDragForce.clear();
          commonData->totalDragForce.clear();
          CHKERR DMoFEMLoopFiniteElements(dm, "DRAG", drag_fe_ptr);

          compute_global_drag(commonData->pressureDragForce, vecGlobalDrag);
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "pressure drag: (%g, %g, %g)\n",
                             vecGlobalDrag[0], vecGlobalDrag[1],
                             vecGlobalDrag[2]);

          compute_global_drag(commonData->viscousDragForce, vecGlobalDrag);
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "viscous drag: (%g, %g, %g)\n",
                             vecGlobalDrag[0], vecGlobalDrag[1],
                             vecGlobalDrag[2]);

          compute_global_drag(commonData->totalDragForce, vecGlobalDrag);
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "total drag: (%g, %g, %g)\n",
                             vecGlobalDrag[0], vecGlobalDrag[1],
                             vecGlobalDrag[2]);
        }

        CHKERR scale_problem(1.0 / U, 1.0 / L, 1.0 / P);
      }

      if (ss % restartStep == 0) {
        if (m_field.get_comm_rank() == 0) {
          const std::string file_name =
              "restart_" + boost::lexical_cast<std::string>(ss) + ".h5m";
          CHKERR m_field.get_moab().write_mesh(file_name.c_str());
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "restart file %s\n",
                             file_name.c_str());
        }
      }

      // CHKERR VecView(D, PETSC_VIEWER_STDOUT_WORLD);

      ss++;
    }

    CHKERR SNESDestroy(&snes);
    CHKERR MatDestroy(&Aij);
    CHKERR VecDestroy(&D);
    CHKERR VecDestroy(&D0);
    CHKERR VecDestroy(&F);
    CHKERR DMDestroy(&dm);

    CHKERR VecDestroy(&G);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();
  return 0;
}