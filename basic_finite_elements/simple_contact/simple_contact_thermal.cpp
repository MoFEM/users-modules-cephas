/** \file simple_contact_thermal.cpp
 * \example simple_contact_thermal.cpp
 *
 * Implementation of simple contact between surfaces with matching meshes
 * taking into account internal stress resulting from the thermal expansion
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

#include <HookeInternalStressElement.hpp>

using namespace std;
using namespace MoFEM;

static char help[] = "\n";
double SimpleContactProblem::LoadScale::lAmbda = 1;
using BlockData = NonlinearElasticElement::BlockData;
using VolSideFe = VolumeElementForcesAndSourcesCoreOnSide;

struct VolRule {
  int operator()(int, int, int order) const { return 2 * (order - 1); }
};

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_divergence_tolerance 0 \n"
                                 "-snes_max_it 50 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-ksp_monitor \n"
                                 "-snes_converged_reason \n"
                                 "-my_order 1 \n"
                                 "-my_order_lambda 1 \n"
                                 "-my_order_contact 2 \n"
                                 "-my_ho_levels_num 1 \n"
                                 "-my_step_num 1 \n"
                                 "-my_cn_value 1. \n"
                                 "-my_r_value 1. \n"
                                 "-my_alm_flag 0 \n"
                                 "-my_eigen_pos_flag 0 \n";

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
    PetscBool flg_file_out;

    char mesh_file_name[255];
    char output_mesh_name[255];
    PetscInt order = 1;
    PetscInt order_contact = 1;
    PetscInt nb_ho_levels = 0;
    PetscInt order_lambda = 1;
    PetscReal r_value = 1.;
    PetscReal cn_value = -1;
    PetscInt nb_sub_steps = 1;
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool is_newton_cotes = PETSC_FALSE;
    PetscInt test_num = 0;
    PetscBool convect_pts = PETSC_FALSE;
    PetscBool print_contact_state = PETSC_FALSE;
    PetscBool alm_flag = PETSC_FALSE;
    PetscBool eigen_pos_flag = PETSC_FALSE;

    PetscReal thermal_expansion_coef = 1e-5;
    PetscReal init_temp = 250.0;
    PetscReal scale_factor = 1.0;
    PetscBool analytical_input = PETSC_TRUE;
    char stress_tag_name[255];
    PetscBool flg_tag_name;
    PetscBool save_mean_stress = PETSC_TRUE;
    PetscBool ignore_contact = PETSC_FALSE;
    PetscBool ignore_pressure = PETSC_FALSE;

    PetscBool deform_flat_flag = PETSC_FALSE;
    PetscReal flat_shift = 1.0;
    PetscReal mesh_height = 1.0;

    PetscBool wave_surf_flag = PETSC_FALSE;
    PetscInt wave_dim = 2;
    PetscReal wave_length = 1.0;
    PetscReal wave_ampl = 0.01;

    PetscBool delete_prisms = PETSC_FALSE;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    CHKERR PetscOptionsString("-my_output_mesh_file", "output mesh file name",
                              "", "mesh.h5m", output_mesh_name, 255,
                              &flg_file_out);

    CHKERR PetscOptionsInt("-my_order",
                           "approximation order of spatial positions", "", 1,
                           &order, PETSC_NULL);
    CHKERR PetscOptionsInt(
        "-my_order_contact",
        "approximation order of spatial positions in contact interface", "", 1,
        &order_contact, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_ho_levels_num", "number of higher order levels",
                           "", 0, &nb_ho_levels, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_order_lambda",
                           "approximation order of Lagrange multipliers", "", 1,
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
    CHKERR PetscOptionsInt("-my_test_num", "test number", "", 0, &test_num,
                           PETSC_NULL);
    CHKERR PetscOptionsBool("-my_convect", "set to convect integration pts", "",
                            PETSC_FALSE, &convect_pts, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_print_contact_state",
                            "output number of active gp at every iteration", "",
                            PETSC_FALSE, &print_contact_state, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_alm_flag",
                            "if set use ALM, if not use C-function", "",
                            PETSC_FALSE, &alm_flag, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_eigen_pos_flag",
                            "if set use eigen spatial positions are taken into "
                            "account for predeformed configuration",
                            "", PETSC_FALSE, &eigen_pos_flag, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_scale_factor", "scale factor", "",
                            scale_factor, &scale_factor, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_ignore_contact", "if set true, ignore contact",
                            "", PETSC_FALSE, &ignore_contact, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_ignore_pressure",
                            "if set true, ignore pressure", "", PETSC_FALSE,
                            &ignore_pressure, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_analytical_input",
                            "if set true, use analytical strain", "",
                            PETSC_TRUE, &analytical_input, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_save_mean_stress",
                            "if set true, save mean stress", "", PETSC_TRUE,
                            &save_mean_stress, PETSC_NULL);
    CHKERR PetscOptionsString(
        "-my_stress_tag_name", "stress tag name file name", "",
        "INTERNAL_STRESS", stress_tag_name, 255, &flg_tag_name);
    CHKERR PetscOptionsReal(
        "-my_thermal_expansion_coef", "thermal expansion coef ", "",
        thermal_expansion_coef, &thermal_expansion_coef, PETSC_NULL);
    CHKERR PetscOptionsReal("-my_init_temp", "init_temp ", "", init_temp,
                            &init_temp, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_mesh_height",
                            "vertical dimension of the mesh ", "", mesh_height,
                            &mesh_height, PETSC_NULL);
    CHKERR PetscOptionsBool("-my_deform_flat", "if set true, deform flat", "",
                            PETSC_FALSE, &deform_flat_flag, PETSC_NULL);
    CHKERR PetscOptionsReal("-my_flat_shift", "flat shift ", "", flat_shift,
                            &flat_shift, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_wave_surf",
                            "if set true, make one of the surfaces wavy", "",
                            PETSC_FALSE, &wave_surf_flag, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_wave_dim", "dimension (2 or 3)", "", wave_dim,
                           &wave_dim, PETSC_NULL);
    CHKERR PetscOptionsReal("-my_wave_length", "profile wavelength", "",
                            wave_length, &wave_length, PETSC_NULL);
    CHKERR PetscOptionsReal("-my_wave_ampl", "profile amplitude", "", wave_ampl,
                            &wave_ampl, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_delete_prisms", "if set true, delete prisms",
                            "", PETSC_FALSE, &delete_prisms, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    // Check if mesh file was provided
    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    if (is_partitioned == PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "Partitioned mesh is not supported");
    }

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    std::vector<BitRefLevel> bit_levels;
    bit_levels.push_back(BitRefLevel().set(0));
    auto bit_ref_manager = m_field.getInterface<BitRefManager>();
    CHKERR bit_ref_manager->setBitRefLevelByDim(0, 3, bit_levels.back());

    auto add_prism_interface = [&](std::vector<BitRefLevel> &bit_levels) {
      MoFEMFunctionBegin;
      auto prism_interface = m_field.getInterface<PrismInterface>();

      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, cit)) {
        if (cit->getName().compare(0, 11, "INT_CONTACT") == 0) {
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "Insert %s (id: %d)\n",
                             cit->getName().c_str(), cit->getMeshsetId());
          EntityHandle cubit_meshset = cit->getMeshset();

          // get tet entities from back bit_level
          EntityHandle ref_level_meshset;
          CHKERR moab.create_meshset(MESHSET_SET, ref_level_meshset);
          CHKERR bit_ref_manager->getEntitiesByTypeAndRefLevel(
              bit_levels.back(), BitRefLevel().set(), MBTET, ref_level_meshset);
          CHKERR bit_ref_manager->getEntitiesByTypeAndRefLevel(
              bit_levels.back(), BitRefLevel().set(), MBPRISM,
              ref_level_meshset);

          // get faces and tets to split
          CHKERR prism_interface->getSides(cubit_meshset, bit_levels.back(),
                                           true, 0);
          // set new bit level
          bit_levels.push_back(BitRefLevel().set(bit_levels.size()));
          // split faces and tets
          CHKERR prism_interface->splitSides(ref_level_meshset,
                                             bit_levels.back(), cubit_meshset,
                                             true, true, 0);
          // clean meshsets
          CHKERR moab.delete_entities(&ref_level_meshset, 1);

          // update cubit meshsets
          for (_IT_CUBITMESHSETS_FOR_LOOP_(m_field, ciit)) {
            EntityHandle cubit_meshset = ciit->meshset;
            CHKERR bit_ref_manager->updateMeshsetByEntitiesChildren(
                cubit_meshset, bit_levels.back(), cubit_meshset, MBVERTEX,
                true);
            CHKERR bit_ref_manager->updateMeshsetByEntitiesChildren(
                cubit_meshset, bit_levels.back(), cubit_meshset, MBEDGE, true);
            CHKERR bit_ref_manager->updateMeshsetByEntitiesChildren(
                cubit_meshset, bit_levels.back(), cubit_meshset, MBTRI, true);
            CHKERR bit_ref_manager->updateMeshsetByEntitiesChildren(
                cubit_meshset, bit_levels.back(), cubit_meshset, MBTET, true);
          }

          CHKERR bit_ref_manager->shiftRightBitRef(1);
          bit_levels.pop_back();
        }
      }

      MoFEMFunctionReturn(0);
    };

    auto find_contact_prisms = [&](std::vector<BitRefLevel> &bit_levels,
                                   Range &contact_prisms, Range &master_tris,
                                   Range &slave_tris) {
      MoFEMFunctionBegin;

      EntityHandle meshset_prisms;
      CHKERR moab.create_meshset(MESHSET_SET, meshset_prisms);
      CHKERR bit_ref_manager->getEntitiesByTypeAndRefLevel(
          bit_levels.back(), BitRefLevel().set(), MBPRISM, meshset_prisms);
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

    if (!ignore_contact) {
      if (analytical_input) {
        CHKERR add_prism_interface(bit_levels);
      }
      CHKERR find_contact_prisms(bit_levels, contact_prisms, master_tris,
                                 slave_tris);
    }

    auto deform_flat_surface = [&](int block_id, double shift, double height) {
      MoFEMFunctionBegin;
      Range all_tets, all_nodes;
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
        if (bit->getName().compare(0, 11, "MAT_ELASTIC") == 0) {
          const int id = bit->getMeshsetId();
          Range tets;
          if (id == block_id) {
            CHKERR m_field.get_moab().get_entities_by_dimension(
                bit->getMeshset(), 3, tets, true);
            all_tets.merge(tets);
          }
        }
      }
      CHKERR m_field.get_moab().get_connectivity(all_tets, all_nodes);
      double coords[3];
      for (Range::iterator nit = all_nodes.begin(); nit != all_nodes.end();
           nit++) {
        CHKERR moab.get_coords(&*nit, 1, coords);
        double x = coords[0];
        double y = coords[1];
        double z = coords[2];
        coords[2] -= shift;
        CHKERR moab.set_coords(&*nit, 1, coords);
      }
      MoFEMFunctionReturn(0);
    };

    auto make_wavy_surface = [&](int block_id, int dim, double lambda,
                                 double delta, double height) {
      MoFEMFunctionBegin;
      Range all_tets, all_nodes;
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
        if (bit->getName().compare(0, 11, "MAT_ELASTIC") == 0) {
          const int id = bit->getMeshsetId();
          Range tets;
          if (id == block_id) {
            CHKERR m_field.get_moab().get_entities_by_dimension(
                bit->getMeshset(), 3, tets, true);
            all_tets.merge(tets);
          }
        }
      }
      CHKERR m_field.get_moab().get_connectivity(all_tets, all_nodes);
      double coords[3];
      for (Range::iterator nit = all_nodes.begin(); nit != all_nodes.end();
           nit++) {
        CHKERR moab.get_coords(&*nit, 1, coords);
        double x = coords[0];
        double y = coords[1];
        double z = coords[2];
        double coef = (height + z) / height;
        switch (dim) {
        case 2:
          coords[2] -= coef * delta * (1. - cos(2. * M_PI * x / lambda));
          break;
        case 3:
          coords[2] -=
              coef * delta *
              (1. - cos(2. * M_PI * x / lambda) * cos(2. * M_PI * y / lambda));
          break;
        default:
          SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                   "Wrong dimension = %d", dim);
        }

        CHKERR moab.set_coords(&*nit, 1, coords);
      }
      MoFEMFunctionReturn(0);
    };

    if (deform_flat_flag && analytical_input) {
      CHKERR deform_flat_surface(1, flat_shift, mesh_height);
      CHKERR deform_flat_surface(2, -flat_shift, mesh_height);
    }

    if (wave_surf_flag && analytical_input) {
      CHKERR make_wavy_surface(1, wave_dim, wave_length, wave_ampl,
                               mesh_height);
      // CHKERR make_wavy_surface(2, wave_dim, wave_length, -wave_ampl,
      //                          mesh_height);
    }

    CHKERR m_field.add_field("SPATIAL_POSITION", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);
    if (!eigen_pos_flag) {
      CHKERR m_field.add_field("EIGEN_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                               3, MB_TAG_SPARSE, MF_ZERO);
    }

    CHKERR m_field.add_field("LAGMULT", H1, AINSWORTH_LEGENDRE_BASE, 1,
                             MB_TAG_SPARSE, MF_ZERO);

    // Declare problem add entities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");
    CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_POSITION", 1);

    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    if (!eigen_pos_flag) {
      CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "EIGEN_POSITIONS");
      CHKERR m_field.set_field_order(0, MBTET, "EIGEN_POSITIONS", order);
      CHKERR m_field.set_field_order(0, MBTRI, "EIGEN_POSITIONS", order);
      CHKERR m_field.set_field_order(0, MBEDGE, "EIGEN_POSITIONS", order);
      CHKERR m_field.set_field_order(0, MBVERTEX, "EIGEN_POSITIONS", 1);
    }

    CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI, "LAGMULT");
    CHKERR m_field.set_field_order(0, MBTRI, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBEDGE, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBVERTEX, "LAGMULT", 1);

    auto set_contact_order = [&](Range &contact_prisms, int order_contact,
                                 int nb_ho_levels) {
      MoFEMFunctionBegin;
      Range contact_tris, contact_edges;
      CHKERR moab.get_adjacencies(contact_prisms, 2, false, contact_tris,
                                  moab::Interface::UNION);
      contact_tris = contact_tris.subset_by_type(MBTRI);
      CHKERR moab.get_adjacencies(contact_tris, 1, false, contact_edges,
                                  moab::Interface::UNION);
      Range ho_ents;
      ho_ents.merge(contact_tris);
      ho_ents.merge(contact_edges);
      for (int ll = 0; ll < nb_ho_levels; ll++) {
        Range ents, verts, tets;
        CHKERR moab.get_connectivity(ho_ents, verts, true);
        CHKERR moab.get_adjacencies(verts, 3, false, tets,
                                    moab::Interface::UNION);
        tets = tets.subset_by_type(MBTET);
        for (auto d : {1, 2}) {
          CHKERR moab.get_adjacencies(tets, d, false, ents,
                                      moab::Interface::UNION);
        }
        ho_ents = unite(ho_ents, ents);
        ho_ents = unite(ho_ents, tets);
      }

      CHKERR m_field.set_field_order(ho_ents, "SPATIAL_POSITION",
                                     order_contact);

      MoFEMFunctionReturn(0);
    };

    if (!ignore_contact && order_contact > order) {
      CHKERR set_contact_order(contact_prisms, order_contact, nb_ho_levels);
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

    Range slave_verts;
    CHKERR moab.get_connectivity(slave_tris, slave_verts, true);
    CHKERR m_field.getInterface<FieldBlas>()->setField(0.0, MBVERTEX,
                                                       slave_verts, "LAGMULT");

    // Add elastic element
    boost::shared_ptr<std::map<int, BlockData>> block_sets_ptr =
        boost::make_shared<std::map<int, BlockData>>();
    CHKERR HookeElement::setBlocks(m_field, block_sets_ptr);

    boost::shared_ptr<ForcesAndSourcesCore> fe_elastic_lhs_ptr(
        new VolumeElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> fe_elastic_rhs_ptr(
        new VolumeElementForcesAndSourcesCore(m_field));
    fe_elastic_lhs_ptr->getRuleHook = VolRule();
    fe_elastic_rhs_ptr->getRuleHook = VolRule();

    CHKERR HookeElement::addElasticElement(m_field, block_sets_ptr, "ELASTIC",
                                           "SPATIAL_POSITION",
                                           "MESH_NODE_POSITIONS", false);

    auto data_hooke_element_at_pts =
        boost::make_shared<HookeInternalStressElement::DataAtIntegrationPts>();
    CHKERR HookeElement::setOperators(fe_elastic_lhs_ptr, fe_elastic_rhs_ptr,
                                      block_sets_ptr, "SPATIAL_POSITION",
                                      "MESH_NODE_POSITIONS", false, false,
                                      MBTET, data_hooke_element_at_pts);
    auto thermal_strain =
        [&thermal_expansion_coef, &init_temp, &test_num](
            FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> &t_coords) {
          FTensor::Tensor2_symmetric<double, 3> t_thermal_strain;
          FTensor::Index<'i', 3> i;
          FTensor::Index<'j', 3> j;

          constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
          double temp;
          t_thermal_strain(i, j) = 0.0;

          switch (test_num) {
          case 0:
            // Put here analytical formula which may depend on coordinates
            temp = init_temp + 1.0;
            t_thermal_strain(i, j) =
                thermal_expansion_coef * (temp - init_temp) * t_kd(i, j);
            break;
          case 1:
          case 2:
            t_thermal_strain(i, j) = -thermal_expansion_coef * t_kd(i, j);
            break;
          case 3:
            t_thermal_strain(2, 2) = thermal_expansion_coef;
            break;
          case 4:
            t_thermal_strain(i, j) = thermal_expansion_coef * t_kd(i, j);
            break;
          default:
            break;
          }
          return t_thermal_strain;
        };

    if (analytical_input) {
      fe_elastic_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpAnalyticalInternalStrain_dx<0>(
              "SPATIAL_POSITION", data_hooke_element_at_pts, thermal_strain));
      fe_elastic_rhs_ptr->getOpPtrVector().push_back(
          new HookeInternalStressElement::OpGetAnalyticalInternalStress<0>(
              "SPATIAL_POSITION", "SPATIAL_POSITION", data_hooke_element_at_pts,
              thermal_strain));
    } else {
      fe_elastic_rhs_ptr->getOpPtrVector().push_back(
          new HookeInternalStressElement::OpGetInternalStress(
              "SPATIAL_POSITION", "SPATIAL_POSITION", data_hooke_element_at_pts,
              moab, stress_tag_name));
      fe_elastic_rhs_ptr->getOpPtrVector().push_back(
          new HookeInternalStressElement::OpInternalStrain_dx(
              "SPATIAL_POSITION", data_hooke_element_at_pts));
    }

    fe_elastic_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            "SPATIAL_POSITION", data_hooke_element_at_pts->hMat));
    fe_elastic_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            "MESH_NODE_POSITIONS", data_hooke_element_at_pts->HMat));
    fe_elastic_rhs_ptr->getOpPtrVector().push_back(
        new HookeInternalStressElement::OpSaveStress(
            "SPATIAL_POSITION", "SPATIAL_POSITION", data_hooke_element_at_pts,
            *block_sets_ptr.get(), moab, scale_factor, save_mean_stress, false,
            false));

    Range all_tets;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 11, "MAT_ELASTIC") == 0) {
        Range tets;
        CHKERR moab.get_entities_by_dimension(bit->getMeshset(), 3, tets, true);
        all_tets.merge(tets);
      }
    }
    Skinner skinner(&moab);
    Range skin_tris;
    CHKERR skinner.find_skin(0, all_tets, false, skin_tris);

    CHKERR m_field.add_finite_element("SKIN", MF_ZERO);
    CHKERR m_field.add_ents_to_finite_element_by_type(skin_tris, MBTRI, "SKIN");
    CHKERR m_field.modify_finite_element_add_field_row("SKIN",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_col("SKIN",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_data("SKIN",
                                                        "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_data("SKIN",
                                                        "MESH_NODE_POSITIONS");
    CHKERR m_field.modify_finite_element_add_field_data("SKIN",
                                                        "EIGEN_POSITIONS");

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
      if (print_contact_state) {
        fe_rhs_simple_contact->contactStateVec =
            common_data_simple_contact->gaussPtsStateVec;
      }
      contact_problem->setContactOperatorsRhs(
          fe_rhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", is_alm, eigen_pos_flag, "EIGEN_POSITIONS");
      return fe_rhs_simple_contact;
    };

    auto get_master_traction_rhs = [&](auto contact_problem, auto make_element,
                                       bool is_alm = false) {
      auto fe_rhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setMasterForceOperatorsRhs(
          fe_rhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", is_alm, eigen_pos_flag, "EIGEN_POSITIONS");
      return fe_rhs_simple_contact;
    };

    auto get_master_traction_lhs = [&](auto contact_problem, auto make_element,
                                       bool is_alm = false) {
      auto fe_lhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setMasterForceOperatorsLhs(
          fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", is_alm, eigen_pos_flag, "EIGEN_POSITIONS");
      return fe_lhs_simple_contact;
    };

    auto get_contact_lhs = [&](auto contact_problem, auto make_element,
                               bool is_alm = false) {
      auto fe_lhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactOperatorsLhs(
          fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", is_alm, eigen_pos_flag, "EIGEN_POSITIONS");
      return fe_lhs_simple_contact;
    };

    auto cn_value_ptr = boost::make_shared<double>(cn_value);
    auto contact_problem = boost::make_shared<SimpleContactProblem>(
        m_field, cn_value_ptr, is_newton_cotes);

    // add fields to the global matrix by adding the element
    if (!eigen_pos_flag)
      contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                         "LAGMULT", contact_prisms);
    else
      contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                         "LAGMULT", contact_prisms,
                                         eigen_pos_flag, "EIGEN_POSITIONS");

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
    CHKERR m_field.build_adjacencies(bit_levels.back());

    // define problems
    CHKERR m_field.add_problem("CONTACT_PROB", MF_ZERO);

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
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "CONTACT_PROB", bit_levels.back());
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "CONTACT_ELEM");
    CHKERR DMMoFEMAddElement(dm, "ELASTIC");
    CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
    CHKERR DMMoFEMAddElement(dm, "SPRING");
    CHKERR DMMoFEMAddElement(dm, "CONTACT_POST_PROC");
    CHKERR DMMoFEMAddElement(dm, "SKIN");

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

    fe_elastic_rhs_ptr->snes_f = F;
    fe_elastic_lhs_ptr->snes_B = Aij;

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
          get_contact_rhs(contact_problem, make_contact_element, alm_flag),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_master_traction_rhs(contact_problem, make_contact_element,
                                  alm_flag),
          PETSC_NULL, PETSC_NULL);
    }

    CHKERR DMMoFEMSNESSetFunction(dm, "ELASTIC", fe_elastic_rhs_ptr, PETSC_NULL,
                                  PETSC_NULL);

    CHKERR DMMoFEMSNESSetFunction(dm, "SPRING", fe_spring_rhs_ptr, PETSC_NULL,
                                  PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, PETSC_NULL, PETSC_NULL,
                                  dirichlet_bc_ptr.get());

    boost::shared_ptr<FEMethod> fe_null;
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, dirichlet_bc_ptr,
                                  fe_null);
    if (convect_pts == PETSC_TRUE) {
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_contact_lhs(contact_problem, make_convective_master_element),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_traction_lhs(contact_problem,
                                  make_convective_slave_element),
          PETSC_NULL, PETSC_NULL);
    } else {
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_contact_lhs(contact_problem, make_contact_element, alm_flag),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_traction_lhs(contact_problem, make_contact_element,
                                  alm_flag),
          PETSC_NULL, PETSC_NULL);
    }
    CHKERR DMMoFEMSNESSetJacobian(dm, "ELASTIC", fe_elastic_lhs_ptr, PETSC_NULL,
                                  PETSC_NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, "SPRING", fe_spring_lhs_ptr, PETSC_NULL,
                                  PETSC_NULL);
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
      CHKERR PetscOptionsInsertString(PETSC_NULL, testing_options);
    }

    auto snes = MoFEM::createSNES(m_field.get_comm());
    CHKERR SNESSetDM(snes, dm);
    SnesCtx *snes_ctx;
    // create snes nonlinear solver
    {
      CHKERR SNESSetDM(snes, dm);
      CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
      CHKERR SNESSetFunction(snes, F, SnesRhs, snes_ctx);
      CHKERR SNESSetJacobian(snes, Aij, Aij, SnesMat, snes_ctx);
      CHKERR SNESSetFromOptions(snes);
    }

    /// Post proc on the skin
    PostProcFaceOnRefinedMesh post_proc_skin(m_field);
    CHKERR post_proc_skin.generateReferenceElementMesh();
    CHKERR post_proc_skin.addFieldValuesPostProc("SPATIAL_POSITION");
    CHKERR post_proc_skin.addFieldValuesPostProc("MESH_NODE_POSITIONS");
    CHKERR post_proc_skin.addFieldValuesPostProc("EIGEN_POSITIONS");

    struct OpGetFieldGradientValuesOnSkin
        : public FaceElementForcesAndSourcesCore::UserDataOperator {

      const std::string feVolName;
      boost::shared_ptr<VolSideFe> sideOpFe;

      OpGetFieldGradientValuesOnSkin(const std::string field_name,
                                     const std::string vol_fe_name,
                                     boost::shared_ptr<VolSideFe> side_fe)
          : FaceElementForcesAndSourcesCore::UserDataOperator(
                field_name, UserDataOperator::OPROW),
            feVolName(vol_fe_name), sideOpFe(side_fe) {}

      MoFEMErrorCode doWork(int side, EntityType type,
                            DataForcesAndSourcesCore::EntData &data) {
        MoFEMFunctionBegin;
        if (type != MBVERTEX)
          MoFEMFunctionReturnHot(0);
        CHKERR loopSideVolumes(feVolName, *sideOpFe);
        MoFEMFunctionReturn(0);
      }
    };

    boost::shared_ptr<VolSideFe> my_vol_side_fe_ptr =
        boost::make_shared<VolSideFe>(m_field);
    my_vol_side_fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            "SPATIAL_POSITION", data_hooke_element_at_pts->hMat));
    my_vol_side_fe_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(
            "MESH_NODE_POSITIONS", data_hooke_element_at_pts->HMat));

    post_proc_skin.getOpPtrVector().push_back(
        new OpGetFieldGradientValuesOnSkin("SPATIAL_POSITION", "ELASTIC",
                                           my_vol_side_fe_ptr));
    post_proc_skin.getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(
            "SPATIAL_POSITION", data_hooke_element_at_pts->spatPosMat));
    post_proc_skin.getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(
            "MESH_NODE_POSITIONS", data_hooke_element_at_pts->meshNodePosMat));

    post_proc_skin.getOpPtrVector().push_back(
        new HookeInternalStressElement::OpPostProcHookeElement<
            FaceElementForcesAndSourcesCore>(
            "SPATIAL_POSITION", data_hooke_element_at_pts,
            *block_sets_ptr.get(), post_proc_skin.postProcMesh,
            post_proc_skin.mapGaussPts, false, false));

    for (int ss = 0; ss != nb_sub_steps; ++ss) {
      if (!ignore_pressure) {
        SimpleContactProblem::LoadScale::lAmbda = (ss + 1.0) / nb_sub_steps;
      } else {
        SimpleContactProblem::LoadScale::lAmbda = 0;
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Ignoring pressure...\n");
      }
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "Load scale: %6.4e\n",
                         SimpleContactProblem::LoadScale::lAmbda);

      CHKERR SNESSolve(snes, PETSC_NULL, D);

      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    }

    // save on mesh
    CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    Vec v_energy;
    CHKERR HookeElement::calculateEnergy(dm, block_sets_ptr, "SPATIAL_POSITION",
                                         "MESH_NODE_POSITIONS", false, false,
                                         &v_energy);
    const double *eng_ptr;
    CHKERR VecGetArrayRead(v_energy, &eng_ptr);
    // Print elastic energy
    PetscPrintf(PETSC_COMM_WORLD, "Elastic energy: %8.8e\n", *eng_ptr);

    {
      PetscPrintf(PETSC_COMM_WORLD, "Loop post proc on the skin\n");
      CHKERR DMoFEMLoopFiniteElements(dm, "SKIN", &post_proc_skin);
      ostringstream stm;
      string out_file_name;
      stm << "out_skin.h5m";
      out_file_name = stm.str();
      PetscPrintf(PETSC_COMM_WORLD, "Write file %s\n", out_file_name.c_str());
      CHKERR post_proc_skin.writeFile(stm.str());
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

    std::array<double, 2> nb_gauss_pts;
    std::array<double, 2> contact_area;

    if (!ignore_contact) {
      contact_problem->setContactOperatorsForPostProc(
          fe_post_proc_simple_contact, common_data_simple_contact, m_field,
          "SPATIAL_POSITION", "LAGMULT", mb_post, alm_flag, eigen_pos_flag,
          "EIGEN_POSITIONS");

      mb_post.delete_mesh();

      CHKERR VecZeroEntries(common_data_simple_contact->gaussPtsStateVec);
      CHKERR VecZeroEntries(common_data_simple_contact->contactAreaVec);

      CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_ELEM",
                                      fe_post_proc_simple_contact);

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
                    "Active contact area: %8.8f out of %8.8f (%8.8f% %)\n",
                    contact_area[0], contact_area[1],
                    contact_area[0] / contact_area[1] * 100.);
      }

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

    if (!ignore_contact) {
      CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_POST_PROC",
                                      post_proc_contact_ptr);
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

    CHKERR m_field.getInterface<FieldBlas>()->fieldCopy(1., "SPATIAL_POSITION",
                                                        "EIGEN_POSITIONS");

    const int n_parts = m_field.get_comm_size();
    if (m_field.get_comm_rank() == 0) {
      CHKERR DMoFEMLoopFiniteElementsUpAndLowRank(
          dm, "ELASTIC", fe_elastic_rhs_ptr, 0, n_parts);

      if (delete_prisms) {
        Range faces, tris, quads, tris_edges, quads_edges, ents_to_delete;

        CHKERR moab.get_adjacencies(contact_prisms, 2, true, faces,
                                    moab::Interface::UNION);
        tris = faces.subset_by_type(MBTRI);
        quads = faces.subset_by_type(MBQUAD);
        CHKERR moab.get_adjacencies(tris, 1, true, tris_edges,
                                    moab::Interface::UNION);
        CHKERR moab.get_adjacencies(quads, 1, true, quads_edges,
                                    moab::Interface::UNION);

        ents_to_delete.merge(contact_prisms);
        ents_to_delete.merge(quads);
        ents_to_delete.merge(subtract(quads_edges, tris_edges));

        CHKERR moab.delete_entities(ents_to_delete);
      }

      PetscPrintf(PETSC_COMM_WORLD, "Write file %s\n", output_mesh_name);
      CHKERR moab.write_file(output_mesh_name, "MOAB");

      auto get_tag_handle = [&](auto name, auto size) {
        Tag th;
        std::vector<double> def_vals(size, 0.0);
        CHKERR moab.tag_get_handle(name, size, MB_TYPE_DOUBLE, th,
                                   MB_TAG_CREAT | MB_TAG_SPARSE,
                                   def_vals.data());
        return th;
      };

      if (test_num) {
        Range tets;
        CHKERR moab.get_entities_by_dimension(0, 3, tets);
        EntityHandle tet = tets.front();
        std::array<double, 9> internal_stress, actual_stress;
        std::array<double, 9> internal_stress_ref, actual_stress_ref;
        std::array<double, 2> nb_gauss_pts_ref, contact_area_ref;
        switch (test_num) {
        case 1:
          internal_stress_ref = {5., 5., 5., 0., 0., 0., 0., 0., 0.};
          actual_stress_ref = {0., 0., 1., 0., 0., 0., 0., 0., 0.};
          break;
        case 2:
          internal_stress_ref = {5., 5., 5., 0., 0., 0., 0., 0., 0.};
          actual_stress_ref = {0., 5. / 3., 5. / 3., 0., 0., 0., 0., 0., 0.};
          break;
        case 3:
          actual_stress_ref = {0., 0., -100., 0., 0., 0., 0., 0., 0.};
          if (strcmp(stress_tag_name, "INTERNAL_STRESS") == 0)
            internal_stress_ref = {0., 0., -200., 0., 0., 0., 0., 0., 0.};
          else
            internal_stress_ref = {0., 0., -100., 0., 0., 0., 0., 0., 0.};
          break;
        case 4:
          nb_gauss_pts_ref = {96, 192};
          contact_area_ref = {0.125, 0.25};
          break;
        default:
          SETERRQ1(PETSC_COMM_SELF, MOFEM_NOT_FOUND, "Test number %d not found",
                   test_num);
        }

        auto th_internal_stress = get_tag_handle("MED_INTERNAL_STRESS", 9);
        auto th_actual_stress = get_tag_handle("MED_ACTUAL_STRESS", 9);
        CHKERR moab.tag_get_data(th_internal_stress, &tet, 1,
                                 internal_stress.data());
        CHKERR moab.tag_get_data(th_actual_stress, &tet, 1,
                                 actual_stress.data());
        const double eps = 1e-10;
        if (test_num == 4) {
          if (std::abs(nb_gauss_pts_ref[0] - nb_gauss_pts[0]) > eps) {
            SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                     "Wrong number of active contact gauss pts: should be %d "
                     "but is %d",
                     (int)nb_gauss_pts_ref[0], (int)nb_gauss_pts[0]);
          }
          if (std::abs(nb_gauss_pts_ref[1] - nb_gauss_pts[1]) > eps) {
            SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                     "Wrong total number of contact gauss pts: should be %d "
                     "but is %d",
                     (int)nb_gauss_pts_ref[1], (int)nb_gauss_pts[1]);
          }
          if (std::abs(contact_area_ref[0] - contact_area[0]) > eps) {
            SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                     "Wrong active contact area: should be %g "
                     "but is %g",
                     contact_area_ref[0], contact_area[0]);
          }
          if (std::abs(contact_area_ref[1] - contact_area[1]) > eps) {
            SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                     "Wrong potential contact area: should be %g "
                     "but is %g",
                     contact_area_ref[1], contact_area[1]);
          }
        } else {
          if (save_mean_stress) {
            for (int i = 0; i < 9; i++) {
              if (std::abs(internal_stress[i] - internal_stress_ref[i]) > eps) {
                SETERRQ3(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                         "Wrong component %d of internal stress: should be %g "
                         "but is %g",
                         i, internal_stress_ref[i], internal_stress[i]);
              }
              if (std::abs(actual_stress[i] - actual_stress_ref[i]) > eps) {
                SETERRQ3(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                         "Wrong component %d of actual stress: should be %g "
                         "but is %g",
                         i, actual_stress_ref[i], actual_stress[i]);
              }
            }
          }
        }
      }
    }
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}