/** \file test_jacobian_of_simple_contact_element.cpp
 * \example test_jacobian_of_simple_contact_element.cpp
 *
 * Testing implementation of simple contact element (for contact between
 * surfaces with matching meshes) by verifying its tangent matrix
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

using namespace std;
using namespace MoFEM;

static char help[] = "\n";
int main(int argc, char *argv[]) {

  // Initialize MoFEM
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  // Create mesh database
  moab::Core mb_instance;              // create database
  moab::Interface &moab = mb_instance; // create interface to database

  try {
    PetscBool flg_file;
    char mesh_file_name[255];
    PetscInt order = 1;
    PetscInt order_lambda = 1;
    PetscReal r_value = 1.;
    PetscReal cn_value = 1.;
    PetscBool is_newton_cotes = PETSC_FALSE;
    PetscBool test_jacobian = PETSC_FALSE;
    PetscBool convect_pts = PETSC_FALSE;
    PetscBool test_ale = PETSC_FALSE;
    PetscBool alm_flag = PETSC_FALSE;
    PetscBool eigen_pos_flag = PETSC_FALSE;
    PetscBool use_reference_coordinates = PETSC_TRUE;

    PetscBool is_friction = PETSC_FALSE;
    PetscReal cn_tangent_value = 1.;
    PetscReal mu_tangent = 0.1;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test_jacobian", &test_jacobian,
                               PETSC_NULL);

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order",
                           "approximation order for spatial positions", "", 1,
                           &order, PETSC_NULL);
    CHKERR PetscOptionsInt(
        "-my_order_lambda",
        "default approximation order of Lagrange multipliers", "", 1,
        &order_lambda, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_cn_value", "default regularisation cn value",
                            "", 1., &cn_value, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_newton_cotes",
                            "set if Newton-Cotes integration rules are used",
                            "", PETSC_FALSE, &is_newton_cotes, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_convect", "set to convect integration pts", "",
                            PETSC_FALSE, &convect_pts, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_alm_flag", "set to convect integration pts",
                            "", PETSC_FALSE, &alm_flag, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_eigen_pos_flag",
                            "if set use eigen spatial positions are taken into "
                            "account for predeformed configuration",
                            "", PETSC_FALSE, &eigen_pos_flag, PETSC_NULL);

    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test_ale", &test_ale,
                               PETSC_NULL);

    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-my_use_reference_coordinates",
                               &use_reference_coordinates, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_cn_tangent_value",
                            "default regularisation cn value", "", 1.,
                            &cn_tangent_value, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_mu_tangent", "default regularisation cn value",
                            "", 1., &mu_tangent, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_friction",
                            "set if mesh is friction (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_friction, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    // Check if mesh file was provided
    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    // Read mesh to MOAB
    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

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
          CHKERR moab.create_meshset(MESHSET_SET, ref_level_meshset);
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

    // Declare problem
    // add entities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");
    CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_POSITION", 1);

    CHKERR m_field.add_field("LAGMULT", H1, AINSWORTH_LEGENDRE_BASE, 1,
                             MB_TAG_SPARSE, MF_ZERO);

    if (is_friction) {
      cerr << "FRICTION !! \n";
      CHKERR m_field.add_field("TANGENT_LAGMULT", H1, AINSWORTH_LEGENDRE_BASE, 2,
                               MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI, "TANGENT_LAGMULT");
    CHKERR m_field.set_field_order(0, MBTRI, "TANGENT_LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBEDGE, "TANGENT_LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBVERTEX, "TANGENT_LAGMULT", 1);

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

    }

    CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI, "LAGMULT");
    CHKERR m_field.set_field_order(0, MBTRI, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBEDGE, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBVERTEX, "LAGMULT", 1);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    if (eigen_pos_flag) {
      CHKERR m_field.add_field("EIGEN_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                               3, MB_TAG_SPARSE, MF_ZERO);
      CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "EIGEN_POSITIONS");
      CHKERR m_field.set_field_order(0, MBTET, "EIGEN_POSITIONS", order);
      CHKERR m_field.set_field_order(0, MBTRI, "EIGEN_POSITIONS", order);
      CHKERR m_field.set_field_order(0, MBEDGE, "EIGEN_POSITIONS", order);
      CHKERR m_field.set_field_order(0, MBVERTEX, "EIGEN_POSITIONS", 1);
    }

    // build field
    CHKERR m_field.build_fields();

    PetscRandom rctx;
    PetscRandomCreate(PETSC_COMM_WORLD, &rctx);

    auto set_coord = [&](VectorAdaptor &&field_data, double *x, double *y,
                         double *z) {
      MoFEMFunctionBegin;
      double value;
      double scale = 0.5;
      PetscRandomGetValue(rctx, &value);
      field_data[0] = (*x) + (value - 0.5) * scale;
      PetscRandomGetValue(rctx, &value);
      field_data[1] = (*y) + (value - 0.5) * scale;
      PetscRandomGetValue(rctx, &value);
      field_data[2] = (*z) + (value - 0.5) * scale;
      MoFEMFunctionReturn(0);
    };

    auto set_pressure = [&](VectorAdaptor &&field_data, double *x, double *y,
                            double *z) {
      MoFEMFunctionBegin;
      double value;
      double scale = 1.0;
      PetscRandomGetValueReal(rctx, &value);
      field_data[0] = value * scale;
      MoFEMFunctionReturn(0);
    };

    auto set_normal = [&](VectorAdaptor &&field_data, double *x, double *y,
                            double *z) {
      MoFEMFunctionBegin;
      double value;
      double scale = 1.e0;
      // PetscRandomGetValue(rctx, &value);
      // field_data[0] =  value * scale;
      // PetscRandomGetValue(rctx, &value);
      // field_data[1] =  value  * scale;
      // PetscRandomGetValue(rctx, &value);
      // field_data[2] = value * scale;

      PetscRandomGetValue(rctx, &value);
      field_data[0] = -10;
      MoFEMFunctionReturn(0);
    };


    auto set_friction = [&](VectorAdaptor &&field_data, double *x, double *y,
                            double *z) {
      MoFEMFunctionBegin;
      double value;
      double scale = 1.e0;
      // PetscRandomGetValue(rctx, &value);
      // field_data[0] =  value * scale;
      // PetscRandomGetValue(rctx, &value);
      // field_data[1] =  value  * scale;
      // PetscRandomGetValue(rctx, &value);
      // field_data[2] = value * scale;

      PetscRandomGetValue(rctx, &value);
      field_data[0] = +1;
      PetscRandomGetValue(rctx, &value);
      field_data[1] = +1;
      MoFEMFunctionReturn(0);
    };

      int count = 0;
        auto set_gap_friction = [&](VectorAdaptor &&field_data, double *x, double *y,
                            double *z) {
      MoFEMFunctionBegin;
      double value;
      double scale = 1.e0;
      // PetscRandomGetValue(rctx, &value);
      // field_data[0] =  value * scale;
      // PetscRandomGetValue(rctx, &value);
      // field_data[1] =  value  * scale;
      // PetscRandomGetValue(rctx, &value);
      // field_data[2] = value * scale;
      if(count > 3){
      MoFEMFunctionReturnHot(0);
      }else {
      count++;
      }
      cerr << "x " << *x <<"\n";
      cerr << "y " << *y <<"\n";
      cerr << "z " << *z <<"\n";

      
      PetscRandomGetValue(rctx, &value);
      field_data[0] = -1;
      PetscRandomGetValue(rctx, &value);
      field_data[1] = -1;
      PetscRandomGetValue(rctx, &value);
      field_data[2] = +1;
      MoFEMFunctionReturn(0);
    };

    if (is_friction) {
      CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(set_normal,
                                                              "LAGMULT");
      CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(set_friction,
                                                              "TANGENT_LAGMULT");
      // CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(
      //     set_gap_friction, "SPATIAL_POSITION");
    } else {
      CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(
          set_coord, "SPATIAL_POSITION");

      CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(set_pressure,
                                                              "LAGMULT");
    }

    if (eigen_pos_flag) {
      CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(
          set_coord, "SPATIAL_POSITION");
    }

    if (test_ale == PETSC_TRUE) {
      CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(
          set_coord, "MESH_NODE_POSITIONS");
    } else {
      // MESH_NODE_POSITIONS
      {
        Projection10NodeCoordsOnField ent_method(m_field, "MESH_NODE_POSITIONS");
        CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);
      }
    }

    if (is_friction) {
      Projection10NodeCoordsOnField ent_method(m_field,
                                               "PREVIOUS_CONV_SPAT_POS");
      CHKERR m_field.loop_dofs("PREVIOUS_CONV_SPAT_POS", ent_method);
    }

    PetscRandomDestroy(&rctx);

    auto cn_value_ptr = boost::make_shared<double>(cn_value);
    
    // auto contact_problem = boost::make_shared<SimpleContactProblem>(
    //     m_field, cn_value_ptr, is_newton_cotes);

    auto get_contact_problem = [&]() {
      if (!is_friction) {
        auto d_contact_problem = boost::make_shared<SimpleContactProblem>(
            m_field, cn_value_ptr, is_newton_cotes);
        return d_contact_problem;
      } else {
        auto d_contact_problem = boost::make_shared<SimpleContactProblem>(
            m_field, cn_value, cn_tangent_value, mu_tangent, is_newton_cotes);
        return d_contact_problem;
      }
    };

    auto contact_problem = get_contact_problem();

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

    auto get_contact_rhs = [&](auto contact_problem, auto make_element,
                               bool is_alm = false) {
      auto fe_rhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactOperatorsRhs(
          fe_rhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", is_alm, eigen_pos_flag, "EIGEN_POSITIONS",
          use_reference_coordinates);
      return fe_rhs_simple_contact;
    };

    auto get_master_contact_lhs = [&](auto contact_problem, auto make_element,
                                      bool is_alm = false) {
      auto fe_lhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactOperatorsLhs(
          fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", is_alm, eigen_pos_flag, "EIGEN_POSITIONS",
          use_reference_coordinates);
      return fe_lhs_simple_contact;
    };

    auto get_master_traction_rhs = [&](auto contact_problem, auto make_element,
                                       bool alm_flag = false) {
      auto fe_rhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setMasterForceOperatorsRhs(
          fe_rhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", alm_flag, eigen_pos_flag, "EIGEN_POSITIONS",
          use_reference_coordinates);
      return fe_rhs_simple_contact;
    };

    auto get_friction_contact_rhs = [&](auto contact_problem,
                                        auto make_element) {
      auto fe_rhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactFrictionAugmentedOperatorsRhs(
          fe_rhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", "TANGENT_LAGMULT", "PREVIOUS_CONV_SPAT_POS");
      return fe_rhs_simple_contact;
    };

    auto get_master_traction_lhs = [&](auto contact_problem, auto make_element,
                                       bool alm_flag = false) {
      auto fe_lhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setMasterForceOperatorsLhs(
          fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", alm_flag, eigen_pos_flag, "EIGEN_POSITIONS",
          use_reference_coordinates);
      return fe_lhs_simple_contact;
    };

    auto get_contact_material_rhs = [&](auto contact_problem, auto make_element,
                                        Range &ale_nodes) {
      auto fe_rhs_simple_contact_ale_material = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      common_data_simple_contact->forcesOnlyOnEntitiesRow.clear();
      common_data_simple_contact->forcesOnlyOnEntitiesRow = ale_nodes;
      contact_problem->setContactOperatorsRhsALEMaterial(
          fe_rhs_simple_contact_ale_material, common_data_simple_contact,
          "SPATIAL_POSITION", "MESH_NODE_POSITIONS", "LAGMULT", "MATERIAL");
      return fe_rhs_simple_contact_ale_material;
    };

    auto get_simple_contact_ale_lhs = [&](auto contact_problem,
                                          auto make_element) {
      auto fe_lhs_simple_contact_ale = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactOperatorsLhsALE(
          fe_lhs_simple_contact_ale, common_data_simple_contact,
          "SPATIAL_POSITION", "MESH_NODE_POSITIONS", "LAGMULT", eigen_pos_flag,
          "EIGEN_POSITIONS");
      return fe_lhs_simple_contact_ale;
    };

    auto get_simple_contact_ale_material_lhs =
        [&](auto contact_problem, auto make_element, Range &ale_nodes) {
          auto fe_lhs_simple_contact_material_ale = make_element();
          auto common_data_simple_contact = make_contact_common_data();
          common_data_simple_contact->forcesOnlyOnEntitiesRow.clear();
          common_data_simple_contact->forcesOnlyOnEntitiesRow = ale_nodes;
          contact_problem->setContactOperatorsLhsALEMaterial(
              fe_lhs_simple_contact_material_ale, common_data_simple_contact,
              "SPATIAL_POSITION", "MESH_NODE_POSITIONS", "LAGMULT", "MATERIAL");
          return fe_lhs_simple_contact_material_ale;
        };

    auto get_simple_friction_contact = [&](auto contact_problem,
                                           auto make_element) {
      auto fe_lhs_simple_contact_ale = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactFrictionAugmentedOperatorsLhs(
          fe_lhs_simple_contact_ale, common_data_simple_contact,
          "SPATIAL_POSITION", "LAGMULT", "TANGENT_LAGMULT", "PREVIOUS_CONV_SPAT_POS");
      return fe_lhs_simple_contact_ale;
    };

    // add fields to the global matrix by adding the element
    if (is_friction){
      contact_problem->addContactFrictionElement(
          "CONTACT_ELEM", "SPATIAL_POSITION", "LAGMULT", "TANGENT_LAGMULT",
          "PREVIOUS_CONV_SPAT_POS", contact_prisms);
    } else {
    if (!eigen_pos_flag)
      contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                         "LAGMULT", contact_prisms);
    else
      contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                         "LAGMULT", contact_prisms,
                                         eigen_pos_flag, "EIGEN_POSITIONS");
    }

    Range all_tets;
    if (test_ale == PETSC_TRUE) {
      if (!eigen_pos_flag)
        contact_problem->addContactElementALE(
            "ALE_CONTACT_ELEM", "SPATIAL_POSITION", "MESH_NODE_POSITIONS",
            "LAGMULT", contact_prisms);
      else
        contact_problem->addContactElementALE(
            "ALE_CONTACT_ELEM", "SPATIAL_POSITION", "MESH_NODE_POSITIONS",
            "LAGMULT", contact_prisms, eigen_pos_flag, "EIGEN_POSITIONS");

      Range faces;
      CHKERR moab.get_adjacencies(contact_prisms, 2, false, faces,
                                  moab::Interface::UNION);
      Range tris = faces.subset_by_type(MBTRI);

      CHKERR moab.get_adjacencies(tris, 3, false, all_tets,
                                  moab::Interface::UNION);

      // Add finite elements
      CHKERR m_field.add_finite_element("MATERIAL", MF_ZERO);
      CHKERR m_field.modify_finite_element_add_field_row("MATERIAL",
                                                         "SPATIAL_POSITION");
      CHKERR m_field.modify_finite_element_add_field_col("MATERIAL",
                                                         "SPATIAL_POSITION");
      CHKERR m_field.modify_finite_element_add_field_row("MATERIAL",
                                                         "MESH_NODE_POSITIONS");
      CHKERR m_field.modify_finite_element_add_field_col("MATERIAL",
                                                         "MESH_NODE_POSITIONS");
      CHKERR m_field.modify_finite_element_add_field_data("MATERIAL",
                                                          "SPATIAL_POSITION");
      CHKERR m_field.modify_finite_element_add_field_data(
          "MATERIAL", "MESH_NODE_POSITIONS");
      CHKERR m_field.add_ents_to_finite_element_by_type(all_tets, MBTET,
                                                        "MATERIAL");
      CHKERR m_field.build_finite_elements("MATERIAL", &all_tets);
    }

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

    // create dm instance
    SmartPetscObj<DM> dm;
    dm = createSmartDM(m_field.get_comm(), dm_name);
    CHKERR DMSetType(dm, dm_name);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "CONTACT_PROB", bit_levels.back());
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, PETSC_FALSE);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "CONTACT_ELEM");

    if (test_ale == PETSC_TRUE) {
      CHKERR DMMoFEMAddElement(dm, "ALE_CONTACT_ELEM");
      CHKERR DMMoFEMAddElement(dm, "MATERIAL");
    }

    CHKERR DMSetUp(dm);

    // Vector of DOFs and the RHS
    auto D = smartCreateDMVector(dm);
    auto F = smartVectorDuplicate(D);

    // Stiffness matrix
    auto A = smartCreateDMMatrix(dm);

    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR MatSetOption(A, MAT_SPD, PETSC_TRUE);
    CHKERR MatZeroEntries(A);

    auto fdA = smartMatDuplicate(A, MAT_COPY_VALUES);

    if (convect_pts == PETSC_TRUE) {
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_contact_rhs(contact_problem, make_convective_master_element),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_contact_lhs(contact_problem,
                                 make_convective_master_element),
          NULL, NULL);
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_master_traction_rhs(contact_problem,
                                  make_convective_slave_element),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_traction_lhs(contact_problem,
                                  make_convective_slave_element),
          NULL, NULL);
    } else {

      if (is_friction) {
        CHKERR DMMoFEMSNESSetFunction(
            dm, "CONTACT_ELEM",
            get_friction_contact_rhs(contact_problem, make_contact_element),
            PETSC_NULL, PETSC_NULL);

        CHKERR DMMoFEMSNESSetJacobian(
            dm, "CONTACT_ELEM",
            get_simple_friction_contact(contact_problem, make_contact_element),
            PETSC_NULL, PETSC_NULL);

        CHKERR DMMoFEMSNESSetFunction(
            dm, "CONTACT_ELEM",
            get_contact_rhs(contact_problem, make_contact_element, true),
            PETSC_NULL, PETSC_NULL);
        CHKERR DMMoFEMSNESSetFunction(
            dm, "CONTACT_ELEM",
            get_master_traction_rhs(contact_problem, make_contact_element,
                                    true),
            PETSC_NULL, PETSC_NULL);
        CHKERR DMMoFEMSNESSetJacobian(
            dm, "CONTACT_ELEM",
            get_master_contact_lhs(contact_problem, make_contact_element,
                                   true),
            PETSC_NULL, PETSC_NULL);
        CHKERR DMMoFEMSNESSetJacobian(
            dm, "CONTACT_ELEM",
            get_master_traction_lhs(contact_problem, make_contact_element,
                                    true),
            PETSC_NULL, PETSC_NULL);

      } else {

        CHKERR DMMoFEMSNESSetFunction(
            dm, "CONTACT_ELEM",
            get_contact_rhs(contact_problem, make_contact_element, alm_flag),
            PETSC_NULL, PETSC_NULL);
        CHKERR DMMoFEMSNESSetFunction(
            dm, "CONTACT_ELEM",
            get_master_traction_rhs(contact_problem, make_contact_element,
                                    alm_flag),
            PETSC_NULL, PETSC_NULL);
        CHKERR DMMoFEMSNESSetJacobian(
            dm, "CONTACT_ELEM",
            get_master_contact_lhs(contact_problem, make_contact_element,
                                   alm_flag),
            PETSC_NULL, PETSC_NULL);
        CHKERR DMMoFEMSNESSetJacobian(
            dm, "CONTACT_ELEM",
            get_master_traction_lhs(contact_problem, make_contact_element,
                                    alm_flag),
            PETSC_NULL, PETSC_NULL);
      }
    }

    if (test_ale == PETSC_TRUE) {
      Range nodes;
      CHKERR moab.get_connectivity(all_tets, nodes, false);

      CHKERR DMMoFEMSNESSetFunction(
          dm, "ALE_CONTACT_ELEM",
          get_contact_material_rhs(contact_problem, make_contact_element,
                                   nodes),
          PETSC_NULL, PETSC_NULL);

      CHKERR DMMoFEMSNESSetJacobian(
          dm, "ALE_CONTACT_ELEM",
          get_simple_contact_ale_lhs(contact_problem, make_contact_element),
          NULL, NULL);

      CHKERR DMMoFEMSNESSetJacobian(
          dm, "ALE_CONTACT_ELEM",
          get_simple_contact_ale_material_lhs(contact_problem,
                                              make_contact_element, nodes),
          NULL, NULL);
    }

    if (test_jacobian == PETSC_TRUE) {
      char testing_options[] =
          "-snes_test_jacobian -snes_test_jacobian_display "
          "-snes_no_convergence_test -snes_atol 0 -snes_rtol 0 "
          "-snes_max_it "
          "1 ";
      CHKERR PetscOptionsInsertString(NULL, testing_options);
    } else {
      char testing_options[] = "-snes_no_convergence_test -snes_atol 0 "
                               "-snes_rtol 0 "
                               "-snes_max_it 1 ";
      CHKERR PetscOptionsInsertString(NULL, testing_options);
    }

    auto snes = MoFEM::createSNES(m_field.get_comm());
    SNESConvergedReason snes_reason;
    SnesCtx *snes_ctx;

    // create snes nonlinear solver
    {
      CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
      CHKERR SNESSetFunction(snes, F, SnesRhs, snes_ctx);
      CHKERR SNESSetJacobian(snes, A, A, SnesMat, snes_ctx);
      CHKERR SNESSetFromOptions(snes);
    }

    CHKERR SNESSolve(snes, PETSC_NULL, D);

    if (test_jacobian == PETSC_FALSE) {
      double nrm_A0;
      CHKERR MatNorm(A, NORM_INFINITY, &nrm_A0);

      char testing_options_fd[] = "-snes_fd";
      CHKERR PetscOptionsInsertString(NULL, testing_options_fd);

      CHKERR SNESSetFunction(snes, F, SnesRhs, snes_ctx);
      CHKERR SNESSetJacobian(snes, fdA, fdA, SnesMat, snes_ctx);
      CHKERR SNESSetFromOptions(snes);

      CHKERR SNESSolve(snes, NULL, D);
      CHKERR MatAXPY(A, -1, fdA, SUBSET_NONZERO_PATTERN);

      double nrm_A;
      CHKERR MatNorm(A, NORM_INFINITY, &nrm_A);
      PetscPrintf(PETSC_COMM_WORLD, "Matrix norms %3.4e %3.4e\n", nrm_A,
                  nrm_A / nrm_A0);
      nrm_A /= nrm_A0;

      constexpr double tol = 1e-6;
      if (nrm_A > tol) {
        SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                "Difference between hand-calculated tangent matrix and finite "
                "difference matrix is too big");
      }
    }
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}