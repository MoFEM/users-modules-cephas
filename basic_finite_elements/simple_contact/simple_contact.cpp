/** \file simple_contact.cpp
 * \example simple_contact.cpp
 *
 * Implementation of simple contact between surfaces with matching meshes
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
int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_max_it 10 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-my_order 2 \n"
                                 "-my_order_lambda 1 \n"
                                 "-my_cn_value 1. \n"
                                 "-my_test_num 0 \n" 
                                 "-my_alm_flag 0 \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  enum contact_tests {
    EIGHT_CUBE = 1,
    FOUR_SEASONS = 2,
    T_INTERFACE = 3,
    PUNCH_TOP_AND_MID = 4,
    PUNCH_TOP_ONLY = 5,
    PLANE_AXI = 6,
    ARC_THREE_SURF = 7,
    SMILING_FACE = 8,
    SMILING_FACE_CONVECT = 9,
    LAST_TEST
  };

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
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool is_newton_cotes = PETSC_FALSE;
    PetscInt test_num = 0;
    PetscBool convect_pts = PETSC_FALSE;
    PetscBool out_integ_pts = PETSC_FALSE;
    PetscBool alm_flag = PETSC_FALSE;

    PetscBool is_friction = PETSC_FALSE;
    PetscReal cn_tangent_value = 1.;
    PetscReal mu_tangent = 0.1;
    PetscInt order_tangent_lambda = 1;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order", "default approximation order", "", 1,
                           &order, PETSC_NULL);
    CHKERR PetscOptionsInt(
        "-my_order_lambda",
        "default approximation order of Lagrange multipliers", "", 1,
        &order_lambda, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_cn_value", "default regularisation cn value",
                            "", 1., &cn_value, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_newton_cotes",
                            "set if Newton-Cotes quadrature rules are used", "",
                            PETSC_FALSE, &is_newton_cotes, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_test_num", "test number", "", 0, &test_num,
                           PETSC_NULL);
    CHKERR PetscOptionsBool("-my_convect", "set to convect integration pts", "",
                            PETSC_FALSE, &convect_pts, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_out_integ_pts",
                            "output data at contact integration points", "",
                            PETSC_FALSE, &out_integ_pts, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_alm_flag", "set to convect integration pts",
                            "", PETSC_FALSE, &alm_flag, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_cn_tangent_value",
                            "default regularisation cn value", "", 1.,
                            &cn_tangent_value, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_mu_tangent", "default regularisation cn value",
                            "", 1., &mu_tangent, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_friction",
                            "set if mesh is friction (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_friction, PETSC_NULL);

    CHKERR PetscOptionsInt("-my_order_tangent_lambda",
                           "approximation order of Lagrange multipliers", "", 1,
                           &order_tangent_lambda, PETSC_NULL);

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
    CHKERR mmanager_ptr->printDisplacementSet();
    CHKERR mmanager_ptr->printForceSet();
    // print block sets with materials
    CHKERR mmanager_ptr->printMaterialsSet();

    auto add_prism_interface = [&](Range &contact_prisms, Range &master_tris,
                                   Range &slave_tris,
                                   std::vector<BitRefLevel> &bit_levels) {
      MoFEMFunctionBegin;
      PrismInterface *interface;
      CHKERR m_field.getInterface(interface);

      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, cit)) {
        if (cit->getName().compare(0, 11, "INT_CONTACT") == 0) {
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "Insert %s (id: %d)\n",
                             cit->getName().c_str(), cit->getMeshsetId());
          EntityHandle cubit_meshset = cit->getMeshset();

          // get tet entities from back bit_level
          EntityHandle ref_level_meshset;
          CHKERR moab.create_meshset(MESHSET_SET | MESHSET_TRACK_OWNER,
                                     ref_level_meshset);
          CHKERR m_field.getInterface<BitRefManager>()
              ->getEntitiesByTypeAndRefLevel(bit_levels.back(),
                                             BitRefLevel().set(), MBTET,
                                             ref_level_meshset);
          CHKERR m_field.getInterface<BitRefManager>()
              ->getEntitiesByTypeAndRefLevel(bit_levels.back(),
                                             BitRefLevel().set(), MBPRISM,
                                             ref_level_meshset);

          // get faces and tets to split
          CHKERR interface->getSides(cubit_meshset, bit_levels.back(), true, 0);
          // set new bit level
          bit_levels.push_back(BitRefLevel().set(bit_levels.size()));
          // split faces and tets
          CHKERR interface->splitSides(ref_level_meshset, bit_levels.back(),
                                       cubit_meshset, true, true, 0);
          // clean meshsets
          CHKERR moab.delete_entities(&ref_level_meshset, 1);

          // update cubit meshsets
          for (_IT_CUBITMESHSETS_FOR_LOOP_(m_field, ciit)) {
            EntityHandle cubit_meshset = ciit->meshset;
            CHKERR m_field.getInterface<BitRefManager>()
                ->updateMeshsetByEntitiesChildren(
                    cubit_meshset, bit_levels.back(), cubit_meshset, MBVERTEX,
                    true);
            CHKERR m_field.getInterface<BitRefManager>()
                ->updateMeshsetByEntitiesChildren(cubit_meshset,
                                                  bit_levels.back(),
                                                  cubit_meshset, MBEDGE, true);
            CHKERR m_field.getInterface<BitRefManager>()
                ->updateMeshsetByEntitiesChildren(cubit_meshset,
                                                  bit_levels.back(),
                                                  cubit_meshset, MBTRI, true);
            CHKERR m_field.getInterface<BitRefManager>()
                ->updateMeshsetByEntitiesChildren(cubit_meshset,
                                                  bit_levels.back(),
                                                  cubit_meshset, MBTET, true);
          }

          CHKERR m_field.getInterface<BitRefManager>()->shiftRightBitRef(1);
          bit_levels.pop_back();
        }
      }

      EntityHandle meshset_prisms;
      CHKERR moab.create_meshset(MESHSET_SET, meshset_prisms);
      CHKERR m_field.getInterface<BitRefManager>()
          ->getEntitiesByTypeAndRefLevel(bit_levels.back(), BitRefLevel().set(),
                                         MBPRISM, meshset_prisms);
      CHKERR moab.get_entities_by_handle(meshset_prisms, contact_prisms);
      CHKERR moab.delete_entities(&meshset_prisms, 1);

      EntityHandle tri;
      for (Range::iterator pit = contact_prisms.begin();
           pit != contact_prisms.end(); pit++) {
        CHKERR moab.side_element(*pit, 2, 3, tri);
        master_tris.insert(tri);
        CHKERR moab.side_element(*pit, 2, 4, tri);
        slave_tris.insert(tri);
      }

      MoFEMFunctionReturn(0);
    };

    Range contact_prisms, master_tris, slave_tris;
    std::vector<BitRefLevel> bit_levels;

    bit_levels.push_back(BitRefLevel().set(0));
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_levels.back());

    CHKERR add_prism_interface(contact_prisms, master_tris, slave_tris,
                               bit_levels);

    CHKERR m_field.add_field("SPATIAL_POSITION", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("LAGMULT", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    // Declare problem add entities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");
    CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_POSITION", 1);

    CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI, "LAGMULT");
    CHKERR m_field.set_field_order(0, MBTRI, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBEDGE, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBVERTEX, "LAGMULT", 1);

    if (is_friction) {

      CHKERR m_field.add_field("PREVIOUS_CONV_SPAT_POS", H1,
                               AINSWORTH_LEGENDRE_BASE, 3, MB_TAG_SPARSE,
                               MF_ZERO);

      CHKERR m_field.add_ents_to_field_by_type(0, MBTET,
                                               "PREVIOUS_CONV_SPAT_POS");

      CHKERR m_field.set_field_order(0, MBTET, "PREVIOUS_CONV_SPAT_POS", order);
      CHKERR m_field.set_field_order(0, MBTRI, "PREVIOUS_CONV_SPAT_POS", order);
      CHKERR m_field.set_field_order(0, MBEDGE, "PREVIOUS_CONV_SPAT_POS",
                                     order);
      CHKERR m_field.set_field_order(0, MBVERTEX, "PREVIOUS_CONV_SPAT_POS", 1);

      // CHKERR m_field.add_field("TANGENT_LAGMULT", H1, AINSWORTH_LEGENDRE_BASE,
      //                          2, MB_TAG_SPARSE, MF_ZERO);
      // CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
      //                                          "TANGENT_LAGMULT");
      // CHKERR m_field.set_field_order(0, MBTRI, "TANGENT_LAGMULT",
      //                                order_tangent_lambda);
      // CHKERR m_field.set_field_order(0, MBEDGE, "TANGENT_LAGMULT",
      //                                order_tangent_lambda);
      // CHKERR m_field.set_field_order(0, MBVERTEX, "TANGENT_LAGMULT", 1);
    }

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

    if (is_friction) {
      Projection10NodeCoordsOnField ent_method(m_field,
                                               "PREVIOUS_CONV_SPAT_POS");
      CHKERR m_field.loop_dofs("PREVIOUS_CONV_SPAT_POS", ent_method);
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

    // auto get_augmented_rhs = [&](auto contact_problem, auto make_element) {
    //   auto fe_rhs_extended_contact = make_element();
    //   auto common_data_simple_contact = make_contact_common_data();
    //   contact_problem->setContactAugmentedOperatorsRhs(
    //       fe_rhs_extended_contact, common_data_simple_contact,
    //       "SPATIAL_POSITION", "LAGMULT");
    //   return fe_rhs_extended_contact;
    // };

    // auto get_augmented_contact_lhs = [&](auto contact_problem,
    //                                      auto make_element) {
    //   auto fe_lhs_extended_contact = make_element();
    //   auto common_data_simple_contact = make_contact_common_data();
    //   contact_problem->setContactAugmentedOperatorsLhs(
    //       fe_lhs_extended_contact, common_data_simple_contact,
    //       "SPATIAL_POSITION", "LAGMULT");
    //   return fe_lhs_extended_contact;
    // };

    auto get_augmented_rhs = [&](auto contact_problem, auto make_element,
                                 auto contact_state_ptr) {
      auto fe_rhs_extended_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_state_ptr->contactStateVec =
          common_data_simple_contact->gaussPtsStateVec;
      contact_problem->setContactAugmentedOperatorsRhs(
          fe_rhs_extended_contact, common_data_simple_contact,
          "SPATIAL_POSITION", "LAGMULT");
      return fe_rhs_extended_contact;
    };

    auto get_augmented_contact_lhs = [&](auto contact_problem,
                                         auto make_element) {
      auto fe_lhs_extended_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactAugmentedOperatorsLhs(
          fe_lhs_extended_contact, common_data_simple_contact,
          "SPATIAL_POSITION", "LAGMULT");
      return fe_lhs_extended_contact;
    };

    auto get_tangent_augmented_rhs = [&](auto contact_problem,
                                         auto make_element) {
      auto fe_rhs_extended_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactFrictionAugmentedOperatorsRhs(
          fe_rhs_extended_contact, common_data_simple_contact,
          "SPATIAL_POSITION", "LAGMULT", "TANGENT_LAGMULT",
          "PREVIOUS_CONV_SPAT_POS");
      return fe_rhs_extended_contact;
    };

    auto get_augmented_friction_lhs = [&](auto contact_problem,
                                          auto make_element) {
      auto fe_lhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactFrictionAugmentedOperatorsLhs(
          fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", "TANGENT_LAGMULT", "PREVIOUS_CONV_SPAT_POS");
      return fe_lhs_simple_contact;
    };

    auto get_contact_problem = [&]() {
      if (!is_friction) {
        auto d_contact_problem = boost::make_shared<SimpleContactProblem>(
            m_field, cn_value, is_newton_cotes);
        return d_contact_problem;
      } else {
        auto d_contact_problem = boost::make_shared<SimpleContactProblem>(
            m_field, cn_value, cn_tangent_value, mu_tangent, is_newton_cotes);
        return d_contact_problem;
      }
    };

    auto contact_problem = get_contact_problem();

    if (is_friction) {
      contact_problem->addContactFrictionElement(
          "CONTACT_ELEM", "SPATIAL_POSITION", "LAGMULT",
          "PREVIOUS_CONV_SPAT_POS", "TANGENT_LAGMULT", contact_prisms);
      // contact_problem->addPostProcContactFrictionElement(
      //     "CONTACT_POST_PROC", "SPATIAL_POSITION", "LAGMULT", "TANGENT_LAGMULT",
      //     "MESH_NODE_POSITIONS", slave_tris);

    } else {
      // add fields to the global matrix by adding the element
      contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                         "LAGMULT", contact_prisms);
    //   contact_problem->addPostProcContactElement(
    //       "CONTACT_POST_PROC", "SPATIAL_POSITION", "LAGMULT",
    //       "MESH_NODE_POSITIONS", slave_tris);
    }

      CHKERR MetaNeumannForces::addNeumannBCElements(m_field,
                                                     "SPATIAL_POSITION");

      // Add spring boundary condition applied on surfaces.
      CHKERR MetaSpringBC::addSpringElements(m_field, "SPATIAL_POSITION",
                                             "MESH_NODE_POSITIONS");

      // build finite elemnts
      CHKERR m_field.build_finite_elements();

      // build adjacencies
      CHKERR m_field.build_adjacencies(bit_levels.back());

      // define problems
      CHKERR m_field.add_problem("CONTACT_PROB");

      // set refinement level for problem
      CHKERR m_field.modify_problem_ref_level_add_bit("CONTACT_PROB",
                                                      bit_levels.back());

      DMType dm_name = "DMMOFEM";
      CHKERR DMRegister_MoFEM(dm_name);

      SmartPetscObj<DM> dm;
      dm = createSmartDM(m_field.get_comm(), dm_name);

      // create dm instance
      CHKERR DMSetType(dm, dm_name);

      // set dm datastruture which created mofem datastructures
      CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "CONTACT_PROB",
                                bit_levels.back());
      CHKERR DMSetFromOptions(dm);
      CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
      // add elements to dm
      CHKERR DMMoFEMAddElement(dm, "CONTACT_ELEM");
      CHKERR DMMoFEMAddElement(dm, "ELASTIC");
      CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
      CHKERR DMMoFEMAddElement(dm, "SPRING");
      // CHKERR DMMoFEMAddElement(dm, "CONTACT_POST_PROC");

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

      boost::shared_ptr<SimpleContactProblem::PrintContactState>
          contact_state_ptr(
              new SimpleContactProblem::PrintContactState(m_field));

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
      CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL,
                                    contact_state_ptr.get(), NULL);

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
        if (alm_flag) {
          CHKERR DMMoFEMSNESSetFunction(dm, "CONTACT_ELEM",
                                        get_augmented_rhs(contact_problem,
                                                          make_contact_element,
                                                          contact_state_ptr),
                                        PETSC_NULL, PETSC_NULL);

          if (is_friction)
            CHKERR DMMoFEMSNESSetFunction(
                dm, "CONTACT_ELEM",
                get_tangent_augmented_rhs(contact_problem,
                                          make_contact_element),
                PETSC_NULL, PETSC_NULL);

        } else {
          CHKERR DMMoFEMSNESSetFunction(
              dm, "CONTACT_ELEM",
              get_contact_rhs(contact_problem, make_contact_element),
              PETSC_NULL, PETSC_NULL);
          CHKERR DMMoFEMSNESSetFunction(
              dm, "CONTACT_ELEM",
              get_master_traction_rhs(contact_problem, make_contact_element),
              PETSC_NULL, PETSC_NULL);
        }
      }

      CHKERR DMMoFEMSNESSetFunction(dm, "ELASTIC", &elastic.getLoopFeRhs(),
                                    PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetFunction(dm, "SPRING", fe_spring_rhs_ptr, PETSC_NULL,
                                    PETSC_NULL);
      CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL, NULL,
                                    dirichlet_bc_ptr.get());
      CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL, NULL,
                                    contact_state_ptr.get());

      boost::shared_ptr<FEMethod> fe_null;
      CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null,
                                    dirichlet_bc_ptr, fe_null);
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
        if (alm_flag) {
          CHKERR DMMoFEMSNESSetJacobian(
              dm, "CONTACT_ELEM",
              get_augmented_contact_lhs(contact_problem, make_contact_element),
              NULL, NULL);
          if (is_friction) {

            cerr << "\n\nget_augmented_friction_lhs\n\n";
            CHKERR
            DMMoFEMSNESSetJacobian(dm, "CONTACT_ELEM",
                                   get_augmented_friction_lhs(
                                       contact_problem, make_contact_element),
                                   NULL, NULL);
          }
        } else {
          CHKERR DMMoFEMSNESSetJacobian(
              dm, "CONTACT_ELEM",
              get_master_contact_lhs(contact_problem, make_contact_element),
              NULL, NULL);
          CHKERR DMMoFEMSNESSetJacobian(
              dm, "CONTACT_ELEM",
              get_master_traction_lhs(contact_problem, make_contact_element),
              NULL, NULL);
        }
      }
    CHKERR DMMoFEMSNESSetJacobian(dm, "ELASTIC", &elastic.getLoopFeLhs(), NULL,
                                  NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, "SPRING", fe_spring_lhs_ptr, NULL, NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, fe_null,
                                  dirichlet_bc_ptr);

    if (test_num) {
      char testing_options[] = "-ksp_type fgmres "
                               "-pc_type lu "
                               "-pc_factor_mat_solver_type mumps "
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

    CHKERR SNESSolve(snes, PETSC_NULL, D);

    CHKERR SNESGetConvergedReason(snes, &snes_reason);

    int its;
    CHKERR SNESGetIterationNumber(snes, &its);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "number of Newton iterations = %D\n\n",
                       its);

    // save on mesh
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    PetscPrintf(PETSC_COMM_WORLD, "Loop post proc\n");
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);

    elastic.getLoopFeEnergy().snes_ctx = SnesMethod::CTX_SNESNONE;
    elastic.getLoopFeEnergy().eNergy = 0;
    PetscPrintf(PETSC_COMM_WORLD, "Loop energy\n");
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &elastic.getLoopFeEnergy());
    // Print elastic energy
    PetscPrintf(PETSC_COMM_WORLD, "Elastic energy %16.9f\n",
                elastic.getLoopFeEnergy().eNergy);

    {
      string out_file_name;
      std::ostringstream stm;
      stm << "out"
          << ".h5m";
      out_file_name = stm.str();
      CHKERR
      PetscPrintf(PETSC_COMM_WORLD, "Write file %s\n", out_file_name.c_str());
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
        "SPATIAL_POSITION", "LAGMULT", mb_post, true, alm_flag, is_friction);

    mb_post.delete_mesh();

    CHKERR VecZeroEntries(common_data_simple_contact->gaussPtsStateVec);
    CHKERR VecZeroEntries(common_data_simple_contact->contactAreaVec);

    CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_ELEM",
                                    fe_post_proc_simple_contact);

    std::array<double, 2> nb_gauss_pts;
    std::array<double, 2> contact_area;

    auto get_contact_data = [&](auto vec, std::array<double, 2> &data) {
      MoFEMFunctionBegin;
      CHKERR VecAssemblyBegin(vec);
      CHKERR VecAssemblyEnd(vec);
      const double *array;
      CHKERR VecGetArrayRead(vec, &array);
      if (m_field.get_comm_rank() == 0) {
        for (int i : {0, 1})
          data[i] = array[i];
      }
      CHKERR VecRestoreArrayRead(vec, &array);
      MoFEMFunctionReturn(0);
    };

    CHKERR get_contact_data(common_data_simple_contact->gaussPtsStateVec,
                            nb_gauss_pts);
    CHKERR get_contact_data(common_data_simple_contact->contactAreaVec,
                            contact_area);

    if (m_field.get_comm_rank() == 0) {
      PetscPrintf(PETSC_COMM_SELF, "Active gauss pts: %d out of %d\n",
                  (int)nb_gauss_pts[0], (int)nb_gauss_pts[1]);

      PetscPrintf(PETSC_COMM_SELF,
                  "Active contact area: %16.9f out of %16.9f\n",
                  contact_area[0], contact_area[1]);
    }

    if (test_num) {
      double expected_energy, expected_contact_area;
      int expected_nb_gauss_pts;
      constexpr double eps = 1e-8;
      switch (test_num) {
      case EIGHT_CUBE:
        expected_energy = 3.0e-04;
        expected_contact_area = 3.0;
        expected_nb_gauss_pts = 576;
        break;
      case FOUR_SEASONS:
        expected_energy = 1.2e-01;
        expected_contact_area = 106.799036701;
        expected_nb_gauss_pts = 672;
        break;
      case T_INTERFACE:
        expected_energy = 3.0e-04;
        expected_contact_area = 1.75;
        expected_nb_gauss_pts = 336;
        break;
      case PUNCH_TOP_AND_MID:
        expected_energy = 3.125e-04;
        expected_contact_area = 0.25;
        expected_nb_gauss_pts = 84;
        break;
      case PUNCH_TOP_ONLY:
        expected_energy = 0.000096432;
        expected_contact_area = 0.25;
        expected_nb_gauss_pts = 336;
        break;
      case PLANE_AXI:
        expected_energy = 0.000438889;
        expected_contact_area = 0.784409608;
        expected_nb_gauss_pts = 300;
        break;
      case ARC_THREE_SURF:
        expected_energy = 0.002573411;
        expected_contact_area = 2.831455633;
        expected_nb_gauss_pts = 228;
        break;
      case SMILING_FACE:
        expected_energy = 0.000733553;
        expected_contact_area = 3.0;
        expected_nb_gauss_pts = 144;
        break;
      case SMILING_FACE_CONVECT:
        expected_energy = 0.000733621;
        expected_contact_area = 3.0;
        expected_nb_gauss_pts = 144;
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                 "Unknown test number %d", test_num);
      }
      if (std::abs(elastic.getLoopFeEnergy().eNergy - expected_energy) > eps) {
        SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                 "Wrong energy %6.4e != %6.4e", expected_energy,
                 elastic.getLoopFeEnergy().eNergy);
      }
      if (m_field.get_comm_rank() == 0) {
        if ((int)nb_gauss_pts[0] != expected_nb_gauss_pts) {
          SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                   "Wrong number of active gauss pts %d != %d",
                   expected_nb_gauss_pts, (int)nb_gauss_pts[0]);
        }
        if (std::abs(contact_area[0] - expected_contact_area) > eps) {
          SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                   "Wrong active contact area %6.4e != %6.4e",
                   expected_contact_area, contact_area[0]);
        }
      }
    }

    if (out_integ_pts) {
      string out_file_name;
      std::ostringstream strm;
      strm << "out_contact_integ_pts"
           << ".h5m";
      out_file_name = strm.str();
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "Write file %s\n",
                         out_file_name.c_str());
      CHKERR mb_post.write_file(out_file_name.c_str(), "MOAB",
                                "PARALLEL=WRITE_PART");
    }

    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_contact_ptr(
        new PostProcFaceOnRefinedMesh(m_field));

    CHKERR post_proc_contact_ptr->generateReferenceElementMesh();
    CHKERR post_proc_contact_ptr->addFieldValuesPostProc("LAGMULT");
    CHKERR post_proc_contact_ptr->addFieldValuesPostProc("SPATIAL_POSITION");
    CHKERR post_proc_contact_ptr->addFieldValuesPostProc("MESH_NODE_POSITIONS");

    // if (is_friction)
    //   CHKERR post_proc_contact_ptr->addFieldValuesPostProc("TANGENT_LAGMULT");

    CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_POST_PROC",
                                    post_proc_contact_ptr);

    {
      string out_file_name;
      std::ostringstream stm;
      stm << "out_contact"
          << ".h5m";
      out_file_name = stm.str();
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "Write file %s\n",
                         out_file_name.c_str());
      CHKERR post_proc_contact_ptr->postProcMesh.write_file(
          out_file_name.c_str(), "MOAB", "PARALLEL=WRITE_PART");
    }
    }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}