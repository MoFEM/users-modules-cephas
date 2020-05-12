/** \file rough_contact.cpp
 * \example rough_contact.cpp
 *
 * Implementation of simple contact (matching meshes)
 * between a solid with a rough surface and a rigid flat
 *
 **/

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
 */

#include <BasicFiniteElements.hpp>
#include <Hooke.hpp>

using namespace std;
using namespace MoFEM;

static char help[] = "\n";
double SimpleContactProblem::LoadScale::lAmbda = 1;

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_package mumps \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_max_it 10 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-my_order 1 \n"
                                 "-my_order_lambda 1 \n"
                                 "-my_cn_value 1. \n"
                                 "-my_is_newton_cotes 0 \n"
                                 "-my_is_test 0 \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  // Initialize MoFEM
  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  // Create mesh database
  moab::Core mb_instance;              // create database
  moab::Interface &moab = mb_instance; // create interface to database

  try {
    PetscBool flg_file;

    char mesh_file_name[255];
    PetscInt order = 1;
    PetscInt order_lambda = 1;
    PetscReal r_value = 1.;
    PetscReal cn_value = -1;
    PetscInt nb_sub_steps = 1; // default number of steps
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool is_newton_cotes = PETSC_FALSE;
    PetscBool is_test = PETSC_FALSE;
    PetscBool convect_pts = PETSC_FALSE;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order", "default approximation order", "", 1,
                           &order, PETSC_NULL);
    CHKERR PetscOptionsInt(
        "-my_order_lambda",
        "default approximation order of Lagrange multipliers", "", 1,
        &order_lambda, PETSC_NULL);

    CHKERR PetscOptionsInt("-my_step_num", "number of steps", "", nb_sub_steps,
                           &nb_sub_steps, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_cn_value", "default regularisation cn value",
                            "", 1., &cn_value, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_newton_cotes",
                            "set if Newton-Cotes quadrature rules are used", "",
                            PETSC_FALSE, &is_newton_cotes, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_is_test", "set if run as test", "",
                            PETSC_FALSE, &is_test, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_convect", "set to convect integration pts", "",
                            PETSC_FALSE, &convect_pts, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    // Check if mesh file was provided
    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    if (is_partitioned == PETSC_TRUE) {
      // Read mesh to MOAB
      const char *option;
      option = "PARALLEL=BCAST_DELETE;"
               "PARALLEL_RESOLVE_SHARED_ENTS;"
               "PARTITION=PARALLEL_PARTITION;";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    } else {
      const char *option;
      option = "";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    }

    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    MeshsetsManager *mmanager_ptr;
    CHKERR m_field.getInterface(mmanager_ptr);

    PrismsFromSurfaceInterface *prisms_from_surface_interface;
    CHKERR m_field.getInterface(prisms_from_surface_interface);

    auto set_prism_layer_thickness = [&](const Range &prisms, int nb_layers) {
      MoFEMFunctionBegin;
      Range nodes_f4;
      int ff = 4;
      for (Range::iterator pit = prisms.begin(); pit != prisms.end(); pit++) {
        EntityHandle face;
        CHKERR moab.side_element(*pit, 2, ff, face);
        const EntityHandle *conn;
        int number_nodes = 0;
        CHKERR moab.get_connectivity(face, conn, number_nodes, false);
        nodes_f4.insert(&conn[0], &conn[number_nodes]);
      }
      double coords[3], director[3];
      for (Range::iterator nit = nodes_f4.begin(); nit != nodes_f4.end();
           nit++) {
        CHKERR moab.get_coords(&*nit, 1, coords);
        if (coords[2] > 0.0) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Solid's surface is on the wrong side of the plane z = 0");
        }
        director[0] = 0.0;
        director[1] = 0.0;
        director[2] = -coords[2] / (double)nb_layers;
        cblas_daxpy(3, 1, director, 1, coords, 1);
        CHKERR moab.set_coords(&*nit, 1, coords);
      }
      MoFEMFunctionReturn(0);
    };

    Range contact_prisms, master_tris, slave_tris;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 11, "INT_CONTACT") == 0) {
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Insert %s (id: %d)\n",
                           bit->getName().c_str(), bit->getMeshsetId());
        Range tris;
        CHKERR moab.get_entities_by_dimension(bit->getMeshset(), 2, tris, true);
        master_tris.merge(tris);
      }
    }

    bool swap_nodes = true;
    CHKERR prisms_from_surface_interface->createPrisms(master_tris, swap_nodes,
                                                       contact_prisms);
    CHKERR set_prism_layer_thickness(contact_prisms, 1);

    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);
    CHKERR prisms_from_surface_interface->seedPrismsEntities(contact_prisms,
                                                             bit_level0);

    Range faces;
    CHKERR moab.get_adjacencies(contact_prisms, 2, true, faces,
                                moab::Interface::UNION);
    faces = faces.subset_by_type(MBTRI);
    slave_tris = subtract(faces, master_tris);

    CHKERR mmanager_ptr->addMeshset(BLOCKSET, 10, "RIGID_FLAT");
    CHKERR mmanager_ptr->addEntitiesToMeshset(BLOCKSET, 10, slave_tris);

    CHKERR mmanager_ptr->setMeshsetFromFile("bc.cfg");

    CHKERR mmanager_ptr->printDisplacementSet();
    CHKERR mmanager_ptr->printMaterialsSet();
    CHKERR mmanager_ptr->printPressureSet();

    EntityHandle rootset = moab.get_root_set();
    CHKERR moab.write_file("tets_and_prisms.vtk", "VTK", "", &rootset, 1);

    CHKERR m_field.add_field("SPATIAL_POSITION", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("LAGMULT", H1, AINSWORTH_LEGENDRE_BASE, 1,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
  
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    // Declare problem add entities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");
    CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                             "SPATIAL_POSITION");
    CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_POSITION", 1);

    CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI, "LAGMULT");
    CHKERR m_field.set_field_order(0, MBTRI, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBEDGE, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBVERTEX, "LAGMULT", 1);

    // build field
    CHKERR m_field.build_fields();

    // Projection on "x" field
    {
      Projection10NodeCoordsOnField ent_method(m_field, "SPATIAL_POSITION");
      CHKERR m_field.loop_dofs("SPATIAL_POSITION", ent_method);
    }
    // Projection on "X" field
    {
      Projection10NodeCoordsOnField ent_method(m_field, "MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);
    }

    // Add elastic element
    boost::shared_ptr<Hooke<adouble>> hooke_adouble_ptr(new Hooke<adouble>());
    boost::shared_ptr<Hooke<double>> hooke_double_ptr(new Hooke<double>());
    NonlinearElasticElement elastic(m_field, 2);
    CHKERR elastic.setBlocks(hooke_double_ptr, hooke_adouble_ptr);
    CHKERR elastic.addElement("ELASTIC", "SPATIAL_POSITION");

    CHKERR elastic.setOperators("SPATIAL_POSITION", "MESH_NODE_POSITIONS",
                                false, false);

    auto make_contact_element = [&]() {
      return boost::make_shared<SimpleContactProblem::SimpleContactElement>(
          m_field);
    };

    auto make_convective_master_element = [&]() {
      return boost::make_shared<
          SimpleContactProblem::ConvectMasterContactElement>(
          m_field, "SPATIAL_POSITION", "MESH_NODE_POSITIONS");
    };

    auto make_convective_slave_element = [&]() {
      return boost::make_shared<
          SimpleContactProblem::ConvectSlaveContactElement>(
          m_field, "SPATIAL_POSITION", "MESH_NODE_POSITIONS");
    };

    auto make_contact_common_data = [&]() {
      return boost::make_shared<SimpleContactProblem::CommonDataSimpleContact>(
          m_field);
    };

    auto get_contact_rhs = [&](auto contact_problem, auto make_element) {
      auto fe_rhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactOperatorsRhs(fe_rhs_simple_contact,
                                              common_data_simple_contact,
                                              "SPATIAL_POSITION", "LAGMULT");
      return fe_rhs_simple_contact;
    };

    auto get_master_traction_rhs = [&](auto contact_problem,
                                       auto make_element) {
      auto fe_rhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setMasterForceOperatorsRhs(
          fe_rhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT");
      return fe_rhs_simple_contact;
    };

    auto get_master_traction_lhs = [&](auto contact_problem,
                                       auto make_element) {
      auto fe_lhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setMasterForceOperatorsLhs(
          fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT");
      return fe_lhs_simple_contact;
    };

    auto get_master_contact_lhs = [&](auto contact_problem, auto make_element) {
      auto fe_lhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactOperatorsLhs(fe_lhs_simple_contact,
                                              common_data_simple_contact,
                                              "SPATIAL_POSITION", "LAGMULT");
      return fe_lhs_simple_contact;
    };

    auto contact_problem = boost::make_shared<SimpleContactProblem>(
        m_field, cn_value, is_newton_cotes);

    // add fields to the global matrix by adding the element
    contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                       "LAGMULT", contact_prisms);
    contact_problem->addPostProcContactElement(
        "CONTACT_POST_PROC", "SPATIAL_POSITION", "LAGMULT",
        "MESH_NODE_POSITIONS", slave_tris);

    CHKERR MetaNeumannForces::addNeumannBCElements(m_field, "SPATIAL_POSITION");

    // Add spring boundary condition applied on surfaces.
    CHKERR MetaSpringBC::addSpringElements(m_field, "SPATIAL_POSITION",
                                           "MESH_NODE_POSITIONS");

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);

    // define problems
    CHKERR m_field.add_problem("CONTACT_PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("CONTACT_PROB", bit_level0);

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    SmartPetscObj<DM> dm;
    dm = createSmartDM(m_field.get_comm(), dm_name);

    // create dm instance
    CHKERR DMSetType(dm, dm_name);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "CONTACT_PROB", bit_level0);
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "CONTACT_ELEM");
    CHKERR DMMoFEMAddElement(dm, "ELASTIC");
    CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
    CHKERR DMMoFEMAddElement(dm, "SPRING");
    CHKERR DMMoFEMAddElement(dm, "CONTACT_POST_PROC");

    CHKERR DMSetUp(dm);

    // Vector of DOFs and the RHS
    auto D = smartCreateDMVector(dm);
    auto F = smartVectorDuplicate(D);

    // Stiffness matrix
    auto Aij = smartCreateDMMatrix(dm);

    CHKERR VecZeroEntries(D);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);
    CHKERR MatZeroEntries(Aij);

    // Dirichlet BC
    boost::shared_ptr<FEMethod> dirichlet_bc_ptr =
        boost::shared_ptr<FEMethod>(new DirichletSpatialPositionsBc(
            m_field, "SPATIAL_POSITION", Aij, D, F));

    dirichlet_bc_ptr->snes_ctx = SnesMethod::CTX_SNESNONE;
    dirichlet_bc_ptr->snes_x = D;

    // Assemble pressure and traction forces
    boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
    CHKERR MetaNeumannForces::setMomentumFluxOperators(
        m_field, neumann_forces, NULL, "SPATIAL_POSITION");

    boost::ptr_map<std::string, NeumannForcesSurface>::iterator mit =
        neumann_forces.begin();
    for (; mit != neumann_forces.end(); mit++) {
      mit->second->methodsOp.push_back(new SimpleContactProblem::LoadScale());
      CHKERR DMMoFEMSNESSetFunction(dm, mit->first.c_str(),
                                    &mit->second->getLoopFe(), NULL, NULL);
    }

    // Implementation of spring element
    // Create new instances of face elements for springs
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr(
        new FaceElementForcesAndSourcesCore(m_field));

    CHKERR MetaSpringBC::setSpringOperators(
        m_field, fe_spring_lhs_ptr, fe_spring_rhs_ptr, "SPATIAL_POSITION",
        "MESH_NODE_POSITIONS");

    CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL,
                                  dirichlet_bc_ptr.get(), NULL);
    if (convect_pts == PETSC_TRUE) {
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_contact_rhs(contact_problem, make_convective_master_element),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_master_traction_rhs(contact_problem,
                                  make_convective_slave_element),
          PETSC_NULL, PETSC_NULL);
    } else {
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_contact_rhs(contact_problem, make_contact_element), PETSC_NULL,
          PETSC_NULL);
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_master_traction_rhs(contact_problem, make_contact_element),
          PETSC_NULL, PETSC_NULL);
    }
    CHKERR DMMoFEMSNESSetFunction(dm, "ELASTIC", &elastic.getLoopFeRhs(),
                                  PETSC_NULL, PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, "SPRING", fe_spring_rhs_ptr, PETSC_NULL,
                                  PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL, NULL,
                                  dirichlet_bc_ptr.get());

    boost::shared_ptr<FEMethod> fe_null;
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, dirichlet_bc_ptr,
                                  fe_null);
    if (convect_pts == PETSC_TRUE) {
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_contact_lhs(contact_problem,
                                 make_convective_master_element),
          NULL, NULL);
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_traction_lhs(contact_problem,
                                  make_convective_slave_element),
          NULL, NULL);
    } else {
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_contact_lhs(contact_problem, make_contact_element), NULL,
          NULL);
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_traction_lhs(contact_problem, make_contact_element), NULL,
          NULL);
    }
    CHKERR DMMoFEMSNESSetJacobian(dm, "ELASTIC", &elastic.getLoopFeLhs(), NULL,
                                  NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, "SPRING", fe_spring_lhs_ptr, NULL, NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, fe_null,
                                  dirichlet_bc_ptr);

    if (is_test == PETSC_TRUE) {
      char testing_options[] = "-ksp_type fgmres "
                               "-pc_type lu "
                               "-pc_factor_mat_solver_package mumps "
                               "-snes_type newtonls "
                               "-snes_linesearch_type basic "
                               "-snes_max_it 10 "
                               "-snes_atol 1e-8 "
                               "-snes_rtol 1e-8 ";
      CHKERR PetscOptionsInsertString(NULL, testing_options);
    }

    auto snes = MoFEM::createSNES(m_field.get_comm());
    CHKERR SNESSetDM(snes, dm);
    SNESConvergedReason snes_reason;
    SnesCtx *snes_ctx;
    // create snes nonlinear solver
    {
      CHKERR SNESSetDM(snes, dm);
      CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
      CHKERR SNESSetFunction(snes, F, SnesRhs, snes_ctx);
      CHKERR SNESSetJacobian(snes, Aij, Aij, SnesMat, snes_ctx);
      CHKERR SNESSetFromOptions(snes);
    }

    PostProcVolumeOnRefinedMesh post_proc(m_field);
    // Add operators to the elements, starting with some generic
    CHKERR post_proc.generateReferenceElementMesh();
    CHKERR post_proc.addFieldValuesPostProc("SPATIAL_POSITION");
    CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
    CHKERR post_proc.addFieldValuesGradientPostProc("SPATIAL_POSITION");

    std::map<int, NonlinearElasticElement::BlockData>::iterator sit =
        elastic.setOfBlocks.begin();
    for (; sit != elastic.setOfBlocks.end(); sit++) {
      post_proc.getOpPtrVector().push_back(new PostProcStress(
          post_proc.postProcMesh, post_proc.mapGaussPts, "SPATIAL_POSITION",
          sit->second, post_proc.commonData));
    }

    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_contact_ptr(
        new PostProcFaceOnRefinedMesh(m_field));

    CHKERR post_proc_contact_ptr->generateReferenceElementMesh();
    CHKERR post_proc_contact_ptr->addFieldValuesPostProc("LAGMULT");
    CHKERR post_proc_contact_ptr->addFieldValuesPostProc("SPATIAL_POSITION");
    CHKERR post_proc_contact_ptr->addFieldValuesPostProc("MESH_NODE_POSITIONS");

    for (int ss = 0; ss != nb_sub_steps; ++ss) {

      SimpleContactProblem::LoadScale::lAmbda = (ss + 1.0) / nb_sub_steps;
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "\nLoad scale: %6.4e\n",
                         SimpleContactProblem::LoadScale::lAmbda);

      CHKERR SNESSolve(snes, PETSC_NULL, D);

      CHKERR SNESGetConvergedReason(snes, &snes_reason);

      int its;
      CHKERR SNESGetIterationNumber(snes, &its);
      CHKERR PetscPrintf(PETSC_COMM_WORLD,
                         "number of Newton iterations = %D\n", its);

      // save on mesh
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

      PetscPrintf(PETSC_COMM_WORLD, "Loop post proc\n");
      CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);

      elastic.getLoopFeEnergy().snes_ctx = SnesMethod::CTX_SNESNONE;
      elastic.getLoopFeEnergy().eNergy = 0;
      PetscPrintf(PETSC_COMM_WORLD, "Loop energy\n");
      CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC",
                                      &elastic.getLoopFeEnergy());
      // Print elastic energy
      PetscPrintf(PETSC_COMM_WORLD, "Elastic energy %6.4e\n",
                  elastic.getLoopFeEnergy().eNergy);

      {
        string out_file_name;
        std::ostringstream stm;
        stm << "out_" << ss << ".h5m";
        out_file_name = stm.str();
        CHKERR
        PetscPrintf(PETSC_COMM_WORLD, "out file %s\n", out_file_name.c_str());
        CHKERR post_proc.postProcMesh.write_file(out_file_name.c_str(), "MOAB",
                                                 "PARALLEL=WRITE_PART");
      }

      // moab_instance
      moab::Core mb_post;                   // create database
      moab::Interface &moab_proc = mb_post; // create interface to database

      auto common_data_simple_contact = make_contact_common_data();

      boost::shared_ptr<SimpleContactProblem::SimpleContactElement>
          fe_post_proc_simple_contact;
      if (convect_pts == PETSC_TRUE) {
        fe_post_proc_simple_contact = make_convective_master_element();
      } else {
        fe_post_proc_simple_contact = make_contact_element();
      }

      contact_problem->setContactOperatorsForPostProc(
          fe_post_proc_simple_contact, common_data_simple_contact, m_field,
          "SPATIAL_POSITION", "LAGMULT", mb_post);

      mb_post.delete_mesh();

      if (is_test == PETSC_TRUE) {
        std::ofstream ofs(
            (std ::string("test_simple_contact") + ".txt").c_str());

        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new SimpleContactProblem::OpMakeTestTextFile(
                m_field, "SPATIAL_POSITION", common_data_simple_contact, ofs));

        CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_ELEM",
                                        fe_post_proc_simple_contact);

        ofs << "Elastic energy: " << elastic.getLoopFeEnergy().eNergy << endl;
        ofs.flush();
        ofs.close();
      } else {
        CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_ELEM",
                                        fe_post_proc_simple_contact);
      }

      {
        string out_file_name;
        std::ostringstream strm;
        strm << "out_contact_integ_pts_" << ss << ".h5m";
        out_file_name = strm.str();
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
                           out_file_name.c_str());
        CHKERR mb_post.write_file(out_file_name.c_str(), "MOAB",
                                  "PARALLEL=WRITE_PART");
      }

      // boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_contact_ptr(
      //     new PostProcFaceOnRefinedMesh(m_field));

      // CHKERR post_proc_contact_ptr->generateReferenceElementMesh();

      // auto common_post_proc_data_simple_contact = make_contact_common_data();

      // CHKERR contact_problem->setPostProcContactOperators(
      //     post_proc_contact_ptr, "SPATIAL_POSITION", "LAGMULT",
      //     common_post_proc_data_simple_contact);

      CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_POST_PROC",
                                      post_proc_contact_ptr);

      {
        string out_file_name;
        std::ostringstream stm;
        stm << "out_contact_" << ss << ".h5m";
        out_file_name = stm.str();
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
                           out_file_name.c_str());
        CHKERR post_proc_contact_ptr->postProcMesh.write_file(
            out_file_name.c_str(), "MOAB", "PARALLEL=WRITE_PART");
      }
    }
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}