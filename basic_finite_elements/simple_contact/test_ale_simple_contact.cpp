/** \file test_contact.cpp
 * \example test_contact.cpp

Testing implementation of Hook element by verifying tangent stiffness matrix.
Test like this is an example of how to verify the implementation of Jacobian.

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
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */

#include <BasicFiniteElements.hpp>

#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;
#include <Hooke.hpp>
using namespace boost::numeric;

using namespace MoFEM;

static char help[] = "\n";
int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_package mumps \n"
                                 "-ksp_monitor \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_max_it 100 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-snes_monitor \n"
                                 "-my_order 1 \n"
                                 "-my_order_lambda 1 \n"
                                 "-my_r_value 1. \n"
                                 "-my_cn_value 1e3 \n"
                                 "-test_case_lambda 0 \n"
                                 "-x_x_test_case 0\n"
                                 "-X_test_case 0\n"
                                 "-my_is_ale 0\n"
                                 "-my_is_lag 1\n"
                                 "-my_is_newton_cotes 0 \n"
                                 "-my_omega_value 0\n"
                                 "-my_theta_s_value 0\n"
                                 "-my_hdiv_trace 0";

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
    PetscBool flg_block_config, flg_file;

    char mesh_file_name[255];
    PetscInt order = 1;

    PetscInt order_lambda = 1;
    PetscReal r_value = 1.;
    PetscReal cn_value = -1;
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool is_newton_cotes = PETSC_FALSE;
    PetscBool is_hdiv_trace = PETSC_FALSE;
    PetscBool is_ale = PETSC_TRUE;
    PetscBool is_lag = PETSC_TRUE;
    PetscInt master_move = 0;
    PetscInt test_case_lambda = 0;
    PetscInt test_case_x = 0;
    PetscInt test_case_X = 0;
    PetscReal my_omega_value = 0.5;
    PetscInt my_theta_s_value = 1;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order", "default approximation order", "", 1,
                           &order, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsInt("-my_master_move", "Move master?", "", 0,
                           &master_move, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_r_value", "default regularisation r value", "",
                            1., &r_value, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_cn_value", "default regularisation cn value",
                            "", 1., &cn_value, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_omega_value",
                            "default regularisation cn value", "", 1.,
                            &my_omega_value, PETSC_NULL);

    CHKERR PetscOptionsInt("-test_case_lambda", "test case", "", 0,
                           &test_case_lambda, PETSC_NULL);

    CHKERR PetscOptionsInt("-x_x_test_case", "test case", "", 0, &test_case_x,
                           PETSC_NULL);

    CHKERR PetscOptionsInt("-X_test_case", "test case", "", 0, &test_case_X,
                           PETSC_NULL);

    CHKERR PetscOptionsInt("-my_theta_s_value", "test case", "", 0,
                           &my_theta_s_value, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_newton_cotes",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_newton_cotes, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_hdiv_trace",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_hdiv_trace, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_ale",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_ale, PETSC_NULL);

    CHKERR PetscOptionsBool("-is_lag",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_lag, PETSC_NULL);

    CHKERR PetscOptionsInt(
        "-my_order_lambda",
        "default approximation order of Lagrange multipliers", "", 1,
        &order_lambda, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    // Read parameters from line command
    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    CHKERR DMRegister_MoFEM("DMMOFEM");

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

    auto add_prism_interface = [&](Range &tets, Range &prisms,
                                   Range &master_tris, Range &slave_tris,
                                   EntityHandle &meshset_tets,
                                   EntityHandle &meshset_prisms,
                                   std::vector<BitRefLevel> &bit_levels) {
      MoFEMFunctionBegin;
      PrismInterface *interface;
      CHKERR m_field.getInterface(interface);

      int ll = 1;
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, SIDESET, cit)) {
        if (cit->getName().compare(0, 11, "INT_CONTACT") == 0) {
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "Insert %s (id: %d)\n",
                             cit->getName().c_str(), cit->getMeshsetId());
          EntityHandle cubit_meshset = cit->getMeshset();
          Range tris;
          CHKERR moab.get_entities_by_type(cubit_meshset, MBTRI, tris, true);
          master_tris.merge(tris);

          {
            // get tet entities from back bit_level
            EntityHandle ref_level_meshset = 0;
            CHKERR moab.create_meshset(MESHSET_SET, ref_level_meshset);
            CHKERR m_field.getInterface<BitRefManager>()
                ->getEntitiesByTypeAndRefLevel(bit_levels.back(),
                                               BitRefLevel().set(), MBTET,
                                               ref_level_meshset);
            CHKERR m_field.getInterface<BitRefManager>()
                ->getEntitiesByTypeAndRefLevel(bit_levels.back(),
                                               BitRefLevel().set(), MBPRISM,
                                               ref_level_meshset);
            Range ref_level_tets;
            CHKERR moab.get_entities_by_handle(ref_level_meshset,
                                               ref_level_tets, true);
            // get faces and tets to split
            CHKERR interface->getSides(cubit_meshset, bit_levels.back(), true,
                                       0);
            // set new bit level
            bit_levels.push_back(BitRefLevel().set(ll++));
            // split faces and tets
            CHKERR interface->splitSides(ref_level_meshset, bit_levels.back(),
                                         cubit_meshset, true, true, 0);
            // clean meshsets
            CHKERR moab.delete_entities(&ref_level_meshset, 1);
          }
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
        }
      }

      for (unsigned int ll = 0; ll != bit_levels.size() - 1; ll++) {
        CHKERR m_field.delete_ents_by_bit_ref(bit_levels[ll], bit_levels[ll],
                                              true);
      }
      CHKERR m_field.getInterface<BitRefManager>()->shiftRightBitRef(
          bit_levels.size() - 1);

      CHKERR moab.create_meshset(MESHSET_SET, meshset_tets);
      CHKERR moab.create_meshset(MESHSET_SET, meshset_prisms);

      CHKERR m_field.getInterface<BitRefManager>()
          ->getEntitiesByTypeAndRefLevel(bit_levels[0], BitRefLevel().set(),
                                         MBTET, meshset_tets);
      CHKERR moab.get_entities_by_handle(meshset_tets, tets, true);

      CHKERR m_field.getInterface<BitRefManager>()
          ->getEntitiesByTypeAndRefLevel(bit_levels[0], BitRefLevel().set(),
                                         MBPRISM, meshset_prisms);
      CHKERR moab.get_entities_by_handle(meshset_prisms, prisms);

      Range tris;
      CHKERR moab.get_adjacencies(prisms, 2, false, tris,
                                  moab::Interface::UNION);
      tris = tris.subset_by_type(MBTRI);
      slave_tris = subtract(tris, master_tris);

      MoFEMFunctionReturn(0);
    };

    Range all_tets, contact_prisms, master_tris, slave_tris;
    EntityHandle meshset_tets, meshset_prisms;
    std::vector<BitRefLevel> bit_levels;

    bit_levels.push_back(BitRefLevel().set(0));
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_levels[0]);

    CHKERR add_prism_interface(all_tets, contact_prisms, master_tris,
                               slave_tris, meshset_tets, meshset_prisms,
                               bit_levels);

    cout << "contact_prisms:" << contact_prisms.size() << endl;
    contact_prisms.print();
    cout << "master_tris:" << master_tris.size() << endl;
    master_tris.print();
    cout << "slave_tris:" << slave_tris.size() << endl;
    slave_tris.print();

    CHKERR m_field.add_field("SPATIAL_POSITION", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    // Declare problem
    // add entities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");
    CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_POSITION", 1);

    if (is_lag) {
      cerr << "  IS LAG \n";
      cerr << "  IS LAG \n";
      cerr << "  IS LAG \n";
      cerr << "  IS LAG \n";
      cerr << "  IS LAG \n";
      cerr << "  IS LAG \n";

      if (is_hdiv_trace) {
        CHKERR m_field.add_field("LAGMULT", HDIV, DEMKOWICZ_JACOBI_BASE, 1);
        CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI, "LAGMULT");
        CHKERR m_field.set_field_order(0, MBTRI, "LAGMULT", order_lambda);
      } else {
        CHKERR m_field.add_field("LAGMULT", H1, AINSWORTH_LEGENDRE_BASE, 1,
                                 MB_TAG_SPARSE, MF_ZERO);

        CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI, "LAGMULT");
        CHKERR m_field.set_field_order(0, MBTRI, "LAGMULT", order_lambda);
        CHKERR m_field.set_field_order(0, MBEDGE, "LAGMULT", order_lambda);
        CHKERR m_field.set_field_order(0, MBVERTEX, "LAGMULT", 1);
      }
    }

    // Add these prisim (between master and slave tris) to the mofem database
    // EntityHandle meshset_slave_master_prisms;
    // CHKERR moab.create_meshset(MESHSET_SET, meshset_slave_master_prisms);

    // CHKERR
    // moab.add_entities(meshset_slave_master_prisms, contact_prisms);

    // CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
    //     meshset_slave_master_prisms, 3, bit_levels[0]);

    // CHKERR moab.write_mesh("slave_master_prisms.vtk",
    //                        &meshset_slave_master_prisms, 1);

    DMType dm_name = "CONTACT_PROB";
    CHKERR DMRegister_MoFEM(dm_name);

    // create dm instance
    DM dm;
    CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
    CHKERR DMSetType(dm, dm_name);

    boost::shared_ptr<SimpleContactProblem> contact_problem;
    contact_problem = boost::shared_ptr<SimpleContactProblem>(
        new SimpleContactProblem(m_field, r_value, cn_value, is_newton_cotes));

    contact_problem->omegaValue = my_omega_value;
    contact_problem->thetaSValue = my_theta_s_value;

    Range nodes;
    CHKERR moab.get_adjacencies(master_tris, 0, false, nodes,
                                moab::Interface::UNION);

    nodes.pop_front();
    nodes.pop_back();

    // contact_problem->commonDataSimpleContact->forcesOnlyOnEntitiesRow =
    // nodes;

    //    contact_problem->commonDataSimpleContact->forcesOnlyOnEntitiesCol =
    //    nodes;

    // add fields to the global matrix by adding the element

    boost::shared_ptr<Hooke<adouble>> hooke_adouble_ptr(new Hooke<adouble>());
    boost::shared_ptr<Hooke<double>> hooke_double_ptr(new Hooke<double>());
    NonlinearElasticElement elastic(m_field, 2);

    if (is_ale) {
      contact_problem->addContactElementALE("CONTACT_ELEM", "SPATIAL_POSITION",
                                            "MESH_NODE_POSITIONS", "LAGMULT",
                                            contact_prisms);
    } else {

      // Add elastic element
      CHKERR elastic.setBlocks(hooke_double_ptr, hooke_adouble_ptr);
      CHKERR elastic.addElement("ELASTIC", "SPATIAL_POSITION");

      CHKERR elastic.setOperators("SPATIAL_POSITION", "MESH_NODE_POSITIONS",
                                  false, false);

      // add fields to the global matrix by adding the element
      contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                         "LAGMULT", contact_prisms, is_lag);

      Range solid_faces;
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
        if (bit->getName().compare(0, 12, "MAT_ELASTIC2") == 0) {

          Range tets, tet, tris, tet_first;
          const int id = bit->getMeshsetId();
          CHKERR m_field.get_moab().get_entities_by_type(
              bit->getMeshset(), MBTET, tet_first, true);
          CHKERR moab.get_adjacencies(tet_first, 2, false, tris,
                                      moab::Interface::UNION);
          tris = intersect(tris, master_tris);

          // boost::shared_ptr<NonlinearElasticElement::BlockData> block_data =
          //     boost::make_shared<NonlinearElasticElement::BlockData>();

          NonlinearElasticElement::BlockData *block_data;

          if (tris.size() == 0) {
            cerr << "M 2 with slave\n";
            CHKERR moab.get_adjacencies(tet_first, 2, false, tris,
                                        moab::Interface::UNION);
            tris = intersect(tris, slave_tris);
            block_data = &(contact_problem->commonDataSimpleContact
                               ->setOfSlaveFacesData[1]);
          } else {
            block_data = &(contact_problem->commonDataSimpleContact
                               ->setOfMasterFacesData[1]);
          }
          CHKERR moab.get_adjacencies(tris, 3, false, tets,
                                      moab::Interface::UNION);
          tets = tets.subset_by_type(MBTET);
          block_data->tEts = tets;

          for (auto &bit : elastic.setOfBlocks) {
            if (bit.second.tEts.contains(tets)) {
              // Range edges, nodes, faces;
              // CHKERR moab.get_adjacencies(tets, 2, false, faces,
              //                             moab::Interface::UNION);
              // CHKERR moab.get_adjacencies(tets, 1, false, edges,
              //                             moab::Interface::UNION);
              // CHKERR moab.get_adjacencies(tets, 0, false, nodes,
              //                             moab::Interface::UNION);

              // contact_problem->commonDataSimpleContact->setOfMasterFacesData[1]
              //     .forcesOnlyOnEntitiesRow.insert(slave_tris.begin(),
              //     slave_tris.end());

              block_data->E = bit.second.E;
              block_data->PoissonRatio = bit.second.PoissonRatio;
              block_data->iD = id;
              block_data->materialDoublePtr = bit.second.materialDoublePtr;
              block_data->materialAdoublePtr = bit.second.materialAdoublePtr;
              break;
            }
          }
          if (block_data->E < 0) {
            SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                    "Cannot find a fluid block adjacent to a given solid face");
          }
        }
      }

      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
        if (bit->getName().compare(0, 12, "MAT_ELASTIC1") == 0) {
          Range tets, tet, tris, tet_first;
          const int id = bit->getMeshsetId();

          CHKERR m_field.get_moab().get_entities_by_type(
              bit->getMeshset(), MBTET, tet_first, true);
          CHKERR moab.get_adjacencies(tet_first, 2, false, tris,
                                      moab::Interface::UNION);
          tris = intersect(tris, master_tris);
          NonlinearElasticElement::BlockData *block_data;
          if (tris.size() == 0) {
            cerr << "M 1 with slave\n";
            CHKERR moab.get_adjacencies(tet_first, 2, false, tris,
                                        moab::Interface::UNION);
            tris = intersect(tris, slave_tris);
            block_data = &(contact_problem->commonDataSimpleContact
                               ->setOfSlaveFacesData[1]);
          } else {
            block_data = &(contact_problem->commonDataSimpleContact
                               ->setOfMasterFacesData[1]);
          }
          CHKERR moab.get_adjacencies(tris, 3, false, tets,
                                      moab::Interface::UNION);
          tets = tets.subset_by_type(MBTET);
          block_data->tEts = tets;

          // CHKERR moab.get_adjacencies(slave_tris, 3, true, tets,
          //                             moab::Interface::UNION);
          // tet = Range(tets.front(), tets.front());
          for (auto &bit : elastic.setOfBlocks) {
            if (bit.second.tEts.contains(tet)) {
              block_data->E = bit.second.E;
              block_data->PoissonRatio = bit.second.PoissonRatio;
              block_data->iD = id;
              block_data->materialDoublePtr = bit.second.materialDoublePtr;
              block_data->materialAdoublePtr = bit.second.materialAdoublePtr;
              break;
            }
          }
          if (block_data->E < 0) {
            SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                    "Cannot find a fluid block adjacent to a given solid face");
          }
        }
      }
    }

    CHKERR m_field.add_finite_element("DUMMY_CONTACT_ELEM", MF_ZERO);
    CHKERR m_field.modify_finite_element_add_field_col("DUMMY_CONTACT_ELEM",
                                                       "SPATIAL_POSITION");

    CHKERR m_field.modify_finite_element_add_field_row("DUMMY_CONTACT_ELEM", "SPATIAL_POSITION");
    
    CHKERR m_field.modify_finite_element_add_field_data("DUMMY_CONTACT_ELEM",
                                                       "SPATIAL_POSITION");
    
    CHKERR m_field.add_ents_to_finite_element_by_type(contact_prisms, MBPRISM,
                                                     "DUMMY_CONTACT_ELEM");
    
    contact_problem->addContactElement("DUMMY_CONTACT_ELEM", "SPATIAL_POSITION",
                                       "LAGMULT", contact_prisms, is_lag);

    auto add_adjacencies_for_nitsche_prism =
        [&](moab::Interface &moab, const Field &field,
            const EntFiniteElement &fe, Range &adjacency) -> MoFEMErrorCode {
      MoFEMFunctionBegin;

      if (field.getName() != "SPATIAL_POSITION")
        MoFEMFunctionReturnHot(0);

      CHKERR DefaultElementAdjacency::defaultPrism(moab, field, fe, adjacency);

      // find two tries from prism
      const EntityHandle prism = fe.getEnt();
      Range faces_range;
      CHKERR moab.get_adjacencies(&prism, 1, 2, false, faces_range);

      EntityHandle faces_mesh_set;
      CHKERR moab.create_meshset(MESHSET_SET, faces_mesh_set);

      CHKERR moab.add_entities(faces_mesh_set, faces_range);

      Range tris;
      CHKERR moab.get_entities_by_type(faces_mesh_set, MBTRI, tris, false);

        cerr << "Field name " << field.getName() << "\n";
        cerr << " size of tris!!!  " << tris.size() << "\n";
      
      // find the two tets to be connected
      Range all_rem_entities;

      for (Range::iterator it_tris = tris.begin(); it_tris != tris.end();
           it_tris++) {

        Range init_vertex, init_edge, init_face;

        // CHKERR moab.get_adjacencies(&*it_tris, 1, 0, true, init_vertex);
        // CHKERR moab.get_adjacencies(&*it_tris, 1, 1, true, init_edge);
        // CHKERR moab.get_adjacencies(&*it_tris, 1, 2, true, init_face);

        Range vols_tet_range;
        CHKERR moab.get_adjacencies(&*it_tris, 1, 3, true, vols_tet_range);
        cerr << "volumes passed " << vols_tet_range.size() << " \n";

        EntityHandle vol_mesh_set;
        CHKERR moab.create_meshset(MESHSET_SET, vol_mesh_set);

        CHKERR moab.add_entities(vol_mesh_set, vols_tet_range);

        Range tet;
        CHKERR moab.get_entities_by_type(vol_mesh_set, MBTET, tet, false);
        // adjacency.merge(tet);

        Range faces;
        CHKERR moab.get_adjacencies(tet, 2, false, faces,
                                    moab::Interface::UNION);
        
        // adjacency.merge(faces);

        Range edges;
        CHKERR moab.get_adjacencies(tet, 1, false, edges,
                                    moab::Interface::UNION);

        // adjacency.merge(edges);

        Range vertices;
        CHKERR moab.get_adjacencies(tet, 0, false, vertices,
                                    moab::Interface::UNION);

        // adjacency.merge(vertices);
        all_rem_entities.merge(tet);
        all_rem_entities.merge(faces);
        all_rem_entities.merge(edges);
        all_rem_entities.merge(vertices);

        cerr << " all_rem_entities " << all_rem_entities.size() << "\n";

        // Range rem_vertex, rem_edge, rem_face;
        // rem_vertex = subtract(vertices, init_vertex);
        // rem_edge = subtract(edges, init_edge);
        // rem_face = subtract(faces, init_face);
        // all_rem_entities.merge(rem_vertex);
        // all_rem_entities.merge(rem_edge);
        // all_rem_entities.merge(rem_face);

        // there is an issue with RefElement_PRISM::getSideNumberPtr
        // otherwise
        

        // adjacency.merge(rem_vertex);
        // adjacency.merge(rem_edge);
        // adjacency.merge(rem_face);
      }
      Range sub_ent = subtract(all_rem_entities, adjacency);
      cerr << " sub_ents size " << sub_ent.size() << "\n";
      for (auto e : sub_ent)
        const_cast<SideNumber_multiIndex &>(fe.getSideNumberTable())
            .insert(boost::shared_ptr<SideNumber>(new SideNumber(e, -1, 0, 0)));

      cerr << " Total number of entities !!!  " << adjacency.size() << "\n";
      adjacency.merge(sub_ent);
      cerr << " Total number of entities 2 !!!  " << adjacency.size() << "\n";

      MoFEMFunctionReturn(0);
    };

    // CHKERR m_field.modify_finite_element_adjacency_table(
    //     "DUMMY_CONTACT_ELEM", MBPRISM, add_adjacencies_for_nitsche);

    // build field
    CHKERR m_field.build_fields();

    // Projection on "x" field
    {
      Projection10NodeCoordsOnField ent_method(m_field, "SPATIAL_POSITION");
      CHKERR m_field.loop_dofs("SPATIAL_POSITION", ent_method);
    }
    // MESH_NODE_POSITIONS
    {
      Projection10NodeCoordsOnField ent_method(m_field, "MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);
    }

    Range tets_of_interest;
    for (Range::iterator it_prisms = contact_prisms.begin();
         it_prisms != contact_prisms.end(); ++it_prisms) {

      Range faces;
      CHKERR moab.get_adjacencies(&*it_prisms, 1, 2, true, faces);

      EntityHandle faces_mesh_set;
      CHKERR moab.create_meshset(MESHSET_SET, faces_mesh_set);

      CHKERR moab.add_entities(faces_mesh_set, faces);

      Range tris;
      CHKERR moab.get_entities_by_type(faces_mesh_set, MBTRI, tris, false);
      
      for (Range::iterator it_tris = tris.begin(); it_tris != tris.end();
           ++it_tris){
        Range vols, tet;
        CHKERR moab.get_adjacencies(&*it_tris, 1, 3, true, vols);

        EntityHandle vol_mesh_set;
        CHKERR moab.create_meshset(MESHSET_SET, vol_mesh_set);
        CHKERR moab.add_entities(vol_mesh_set, vols);
        CHKERR moab.get_entities_by_type(vol_mesh_set, MBTET, tet, false);
        tets_of_interest.merge(tet);
      }
    }

    cerr << "tets of interest " << tets_of_interest.size() << "\n";

    if (is_ale) {
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
      // Range current_ents_with_fe;
      // CHKERR m_field.get_finite_element_entities_by_handle("MATERIAL",
      //                                                     current_ents_with_fe);
      // Range ents_to_remove;
      // ents_to_remove = subtract(current_ents_with_fe, all_tets);
      // CHKERR m_field.remove_ents_from_finite_element("MATERIAL",
      // ents_to_remove);
      CHKERR m_field.add_ents_to_finite_element_by_type(all_tets, MBTET,
                                                        "MATERIAL");
      CHKERR m_field.build_finite_elements("MATERIAL", &all_tets);
    }
else{
    // CHKERR m_field.add_finite_element("MATERIAL", MF_ZERO);
    // CHKERR m_field.modify_finite_element_add_field_row("MATERIAL",
    //                                                    "SPATIAL_POSITION");
    // CHKERR m_field.modify_finite_element_add_field_col("MATERIAL",
    //                                                    "SPATIAL_POSITION");

    // CHKERR m_field.modify_finite_element_add_field_data("MATERIAL",
    //                                                     "SPATIAL_POSITION");
    // Range all_contact_tris;
    // all_contact_tris.merge(master_tris);
    // all_contact_tris.merge(slave_tris);

    // // CHKERR m_field.add_ents_to_finite_element_by_type(all_contact_tris, MBTRI,
    // //                                                   "MATERIAL");
    // CHKERR m_field.add_ents_to_finite_element_by_type(tets_of_interest, MBTET,
    //                                                   "MATERIAL");
    // CHKERR m_field.build_finite_elements("MATERIAL", &tets_of_interest);
}

    auto add_adjacencies_for_nitsche_tet =
        [&](moab::Interface &moab, const Field &field,
            const EntFiniteElement &fe, Range &adjacency) -> MoFEMErrorCode {
      MoFEMFunctionBegin;

      if (field.getName() != "SPATIAL_POSITION")
        MoFEMFunctionReturnHot(0);

      CHKERR DefaultElementAdjacency::defaultTet(moab, field, fe, adjacency);

      // find two tris from prism
      const EntityHandle tet = fe.getEnt();

      Range initial_range;

      initial_range.insert(tet);

      Range faces_range;
      CHKERR moab.get_adjacencies(&tet, 1, 2, false, faces_range);

      Range adj_vol_range;
      CHKERR moab.get_adjacencies(faces_range, 3, false, adj_vol_range,
                                  moab::Interface::UNION);

      EntityHandle vol_mesh_set;
      CHKERR moab.create_meshset(MESHSET_SET, vol_mesh_set);

      CHKERR moab.add_entities(vol_mesh_set, adj_vol_range);
      // prism
      Range prism;
      CHKERR moab.get_entities_by_type(vol_mesh_set, MBPRISM, prism, false);

      Range prism_faces_range;
      CHKERR moab.get_adjacencies(prism, 2, false, prism_faces_range,
                                  moab::Interface::UNION);
    
      EntityHandle prism_faces_mesh_set;
      CHKERR moab.create_meshset(MESHSET_SET, prism_faces_mesh_set);

      CHKERR moab.add_entities(prism_faces_mesh_set, prism_faces_range);
    
      //prism tris
      Range prism_tris;
      CHKERR moab.get_entities_by_type(prism_faces_mesh_set, MBTRI, prism_tris,
                                       false);
    
      // find the two tets to be connected
      Range two_tets;

      for (Range::iterator it_tris = prism_tris.begin();
           it_tris != prism_tris.end(); it_tris++) {

        Range vols_tet_range;
        CHKERR moab.get_adjacencies(&*it_tris, 1, 3, true, vols_tet_range);
        EntityHandle vol_mesh_set_from_tri;
        CHKERR moab.create_meshset(MESHSET_SET, vol_mesh_set_from_tri);

        CHKERR moab.add_entities(vol_mesh_set_from_tri, vols_tet_range);

        Range tet_wanted;
        CHKERR moab.get_entities_by_type(vol_mesh_set_from_tri, MBTET, tet_wanted,
                                         false);

        Range faces;
        CHKERR moab.get_adjacencies(tet_wanted, 2, false, faces,
                                    moab::Interface::UNION);

        Range edges;
        CHKERR moab.get_adjacencies(tet_wanted, 1, false, edges,
                                    moab::Interface::UNION);

        Range vertices;
        CHKERR moab.get_adjacencies(tet_wanted, 0, false, vertices,
                                    moab::Interface::UNION);

        Range check_intersect = intersect(tet_wanted, initial_range);

        Range all_need;
        cerr << " out "
             << adjacency.size() << " \n";
        if (check_intersect.size() != 0) {
          cerr << " do not add entities! asdasd adjacencies size" << adjacency.size() << " \n";
          continue;
        }

        
        

        all_need.merge(tet_wanted);
        all_need.merge(faces);
        all_need.merge(edges);
        all_need.merge(vertices);
        cerr << " all needed " << all_need.size() << "\n";
        Range diff = subtract(adjacency, all_need);
        cerr << " all difference " << diff.size() << "\n";

        for (auto e : all_need)
          const_cast<SideNumber_multiIndex &>(fe.getSideNumberTable())
              .insert(
                  boost::shared_ptr<SideNumber>(new SideNumber(e, -1, 0, 0)));
        adjacency.merge(tet_wanted);
        adjacency.merge(faces);
        adjacency.merge(edges);
        adjacency.merge(vertices);
      }

      cerr << " Total number of volumes !!!  " << adjacency.size() << "\n";
      MoFEMFunctionReturn(0);
    };

    auto add_adjacencies_for_nitsche_tri =
        [&](moab::Interface &moab, const Field &field,
            const EntFiniteElement &fe, Range &adjacency) -> MoFEMErrorCode {
      MoFEMFunctionBegin;

      cerr << "WTH WTH WTH\n";

      if (field.getName() != "SPATIAL_POSITION")
        MoFEMFunctionReturnHot(0);

      CHKERR DefaultElementAdjacency::defaultFace(moab, field, fe, adjacency);

          // find two tries from prism
          const EntityHandle tri = fe.getEnt();

      Range initial_range;

      initial_range.insert(tri);

      // cerr << "initial_range " << initial_range << "\n";

      Range vol_range;
      CHKERR moab.get_adjacencies(&tri, 1, 3, false, vol_range);

      EntityHandle vol_mesh_set;
      CHKERR moab.create_meshset(MESHSET_SET, vol_mesh_set);

      CHKERR moab.add_entities(vol_mesh_set, vol_range);

      Range prism;
      CHKERR moab.get_entities_by_type(vol_mesh_set, MBPRISM, prism, false);

      Range prism_faces_range;
      CHKERR moab.get_adjacencies(prism, 2, false, prism_faces_range,
                                  moab::Interface::UNION);

      EntityHandle prism_faces_mesh_set;
      CHKERR moab.create_meshset(MESHSET_SET, prism_faces_mesh_set);

      CHKERR moab.add_entities(prism_faces_mesh_set, prism_faces_range);

      // prism tris
      Range prism_tris;
      CHKERR moab.get_entities_by_type(prism_faces_mesh_set, MBTRI, prism_tris,
                                       false);

      Range two_tets;
      Range all_need;
      for (Range::iterator it_tris = prism_tris.begin();
           it_tris != prism_tris.end(); it_tris++) {

        Range vols_tet_range;
        CHKERR moab.get_adjacencies(&*it_tris, 1, 3, true, vols_tet_range);
        // cerr << " asdasdasdasdasdasdasd " << vols_tet_range.size() << "\n";
        EntityHandle vol_mesh_set_from_tri;
        CHKERR moab.create_meshset(MESHSET_SET, vol_mesh_set_from_tri);

        CHKERR moab.add_entities(vol_mesh_set_from_tri, vols_tet_range);

        Range tet_wanted;
        CHKERR moab.get_entities_by_type(vol_mesh_set_from_tri, MBTET,
                                         tet_wanted, false);

        // cerr << " w00t " << tet_wanted << "\n";

        EntityHandle vol_mesh_set_check;
        CHKERR moab.create_meshset(MESHSET_SET, vol_mesh_set_check);

        CHKERR moab.add_entities(vol_mesh_set_check, tet_wanted);

        // cerr << " wanted tets " << tet_wanted.size()  << "\n";
        // cerr << " vol_mesh_set_check " <<vol_mesh_set_check << "\n";
        // cerr << " tet " << tet << "\n";

        // adjacency.merge(tet_wanted);

        Range faces;
        CHKERR moab.get_adjacencies(tet_wanted, 2, false, faces,
                                    moab::Interface::UNION);

        // adjacency.merge(faces);

        Range edges;
        CHKERR moab.get_adjacencies(tet_wanted, 1, false, edges,
                                    moab::Interface::UNION);

        // adjacency.merge(edges);

        Range vertices;
        CHKERR moab.get_adjacencies(tet_wanted, 0, false, vertices,
                                    moab::Interface::UNION);
        // adjacency.merge(vertices);

        Range check_intersect = intersect(tet_wanted, initial_range);

       

        // adjacency.merge(tet_wanted);
        // adjacency.merge(faces);
        // adjacency.merge(edges);
        // adjacency.merge(vertices);

        all_need.merge(tet_wanted);
        all_need.merge(faces);
        all_need.merge(edges);
        all_need.merge(vertices);      
        
      }
      Range final_range;
      final_range = subtract(all_need, adjacency);

      for (auto e : final_range)
        const_cast<SideNumber_multiIndex &>(fe.getSideNumberTable())
            .insert(boost::shared_ptr<SideNumber>(new SideNumber(e, -1, 0, 0)));

      // adjacency.merge(final_range);
      cerr << " Total number of volumes ads as !!!   !!!  " << adjacency.size()
           << "\n";
      MoFEMFunctionReturn(0);
    };

    CHKERR m_field.modify_finite_element_adjacency_table(
        "DUMMY_CONTACT_ELEM", MBPRISM, add_adjacencies_for_nitsche_prism);

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_levels[0]);

    // define problems
    CHKERR m_field.add_problem("CONTACT_PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("CONTACT_PROB",
                                                    bit_levels[0]);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, dm_name, bit_levels[0]);
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "CONTACT_ELEM");
    // DUMMY_CONTACT_ELEM
    CHKERR DMMoFEMAddElement(dm, "DUMMY_CONTACT_ELEM");
    if (is_ale) {
      CHKERR DMMoFEMAddElement(dm, "MATERIAL");
    } else {
      // CHKERR DMMoFEMAddElement(dm, "MATERIAL");
      CHKERR DMMoFEMAddElement(dm, "ELASTIC");
    }
    CHKERR DMSetUp(dm);

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

    // CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(set_coord,
    //                                                         "SPATIAL_POSITION");
    // CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(
    //     set_coord, "MESH_NODE_POSITIONS");

    PetscRandomDestroy(&rctx);

    {
      Range range_vertices;
      CHKERR m_field.get_moab().get_adjacencies(
          slave_tris, 0, false, range_vertices, moab::Interface::UNION);

      for (Range::iterator it_vertices = range_vertices.begin();
           it_vertices != range_vertices.end(); it_vertices++) {

        for (DofEntityByNameAndEnt::iterator itt =
                 m_field.get_dofs_by_name_and_ent_begin("SPATIAL_POSITION",
                                                        *it_vertices);
             itt != m_field.get_dofs_by_name_and_ent_end("SPATIAL_POSITION",
                                                         *it_vertices);
             ++itt) {

          auto &dof = **itt;

          EntityHandle ent = dof.getEnt();
          int dof_rank = dof.getDofCoeffIdx();
          VectorDouble3 coords(3);

          CHKERR moab.get_coords(&ent, 1, &coords[0]);

          if (dof_rank == 2 /*&& fabs(coords[2]) <= 1e-6*/) {
            printf("Before x: %e\n", dof.getFieldData());

            switch (test_case_x) {

            case 1:
              dof.getFieldData() = coords[2] - 8.;
              break;
            case 2:
              dof.getFieldData() = coords[2] + 1.;
              break;
            }
            printf("Slave After x : %e test case %d\n", dof.getFieldData(),
                   test_case_x);
          }
        }
      }
    }

    {
      Range range_vertices;
      CHKERR m_field.get_moab().get_adjacencies(
          master_tris, 0, false, range_vertices, moab::Interface::UNION);

      for (Range::iterator it_vertices = range_vertices.begin();
           it_vertices != range_vertices.end(); it_vertices++) {

        for (DofEntityByNameAndEnt::iterator itt =
                 m_field.get_dofs_by_name_and_ent_begin("SPATIAL_POSITION",
                                                        *it_vertices);
             itt != m_field.get_dofs_by_name_and_ent_end("SPATIAL_POSITION",
                                                         *it_vertices);
             ++itt) {

          auto &dof = **itt;

          EntityHandle ent = dof.getEnt();
          int dof_rank = dof.getDofCoeffIdx();
          VectorDouble3 coords(3);

          CHKERR moab.get_coords(&ent, 1, &coords[0]);

          if (dof_rank == 2 /*&& fabs(coords[2]) <= 1e-6*/) {
            printf("Before x: %e\n", dof.getFieldData());

            switch (test_case_x) {

            case 1:
              dof.getFieldData() = coords[2] - 8.;
              break;
            case 2:
              dof.getFieldData() = coords[2] + 1.;
              break;
            }
            printf("Master  After x : %e test case %d\n", dof.getFieldData(),
                   test_case_x);
          }
        }
      }
    }

    {
      Range range_vertices;
      CHKERR m_field.get_moab().get_adjacencies(
          slave_tris, 0, false, range_vertices, moab::Interface::UNION);

      for (Range::iterator it_vertices = range_vertices.begin();
           it_vertices != range_vertices.end(); it_vertices++) {

        for (DofEntityByNameAndEnt::iterator itt =
                 m_field.get_dofs_by_name_and_ent_begin("MESH_NODE_POSITIONS",
                                                        *it_vertices);
             itt != m_field.get_dofs_by_name_and_ent_end("MESH_NODE_POSITIONS",
                                                         *it_vertices);
             ++itt) {

          auto &dof = **itt;

          EntityHandle ent = dof.getEnt();
          int dof_rank = dof.getDofCoeffIdx();
          VectorDouble3 coords(3);

          CHKERR moab.get_coords(&ent, 1, &coords[0]);

          if (dof_rank == 2 /*&& fabs(coords[2]) <= 1e-6*/) {
            printf("Before Slave X: %e\n", dof.getFieldData());

            switch (test_case_X) {

            case 1:
              dof.getFieldData() = coords[2] - 8.;
              break;
            case 2:
              dof.getFieldData() = coords[2] + 1.;
              break;
            }
            printf("After Slave X : %e test case %d\n", dof.getFieldData(),
                   test_case_X);
          }
        }
      }
    }

    {
      Range range_vertices;
      CHKERR m_field.get_moab().get_adjacencies(
          master_tris, 0, false, range_vertices, moab::Interface::UNION);

      for (Range::iterator it_vertices = range_vertices.begin();
           it_vertices != range_vertices.end(); it_vertices++) {

        for (DofEntityByNameAndEnt::iterator itt =
                 m_field.get_dofs_by_name_and_ent_begin("MESH_NODE_POSITIONS",
                                                        *it_vertices);
             itt != m_field.get_dofs_by_name_and_ent_end("MESH_NODE_POSITIONS",
                                                         *it_vertices);
             ++itt) {

          auto &dof = **itt;

          EntityHandle ent = dof.getEnt();
          int dof_rank = dof.getDofCoeffIdx();
          VectorDouble3 coords(3);

          CHKERR moab.get_coords(&ent, 1, &coords[0]);

          if (dof_rank == 2 /*&& fabs(coords[2]) <= 1e-6*/) {
            printf("Before Master X: %e\n", dof.getFieldData());

            switch (test_case_X) {

            case 1:
              dof.getFieldData() = coords[2] - 8.;
              break;
            case 2:
              dof.getFieldData() = coords[2] + 1.;
              break;
            }
            printf("After Master X : %e test case %d\n", dof.getFieldData(),
                   test_case_X);
          }
        }
      }
    }

    if (is_lag) {
      Range range_vertices;
      CHKERR m_field.get_moab().get_adjacencies(
          slave_tris, 0, false, range_vertices, moab::Interface::UNION);

      for (Range::iterator it_vertices = range_vertices.begin();
           it_vertices != range_vertices.end(); it_vertices++) {

        for (DofEntityByNameAndEnt::iterator itt =
                 m_field.get_dofs_by_name_and_ent_begin("LAGMULT",
                                                        *it_vertices);
             itt !=
             m_field.get_dofs_by_name_and_ent_end("LAGMULT", *it_vertices);
             ++itt) {

          auto &dof = **itt;

          EntityHandle ent = dof.getEnt();
          int dof_rank = dof.getDofCoeffIdx();

          printf("Before Lambda: %e\n", dof.getFieldData());

          switch (test_case_lambda) {

          case 1:
            dof.getFieldData() = -2.5;
            break;
          case 2:
            dof.getFieldData() = +2.5;
            break;
          }

          printf("After  Lambda: %e\n", dof.getFieldData());
        }
      }
    }

    Vec x, f;
    CHKERR DMCreateGlobalVector_MoFEM(dm, &x);
    CHKERR VecDuplicate(x, &f);

    CHKERR DMoFEMMeshToLocalVector(dm, x, INSERT_VALUES, SCATTER_FORWARD);

    Mat A, fdA;
    CHKERR DMCreateMatrix_MoFEM(dm, &A);
    CHKERR MatZeroEntries(A);

    boost::shared_ptr<SimpleContactProblem::SimpleContactElement>
        fe_rhs_simple_contact =
            boost::make_shared<SimpleContactProblem::SimpleContactElement>(
                m_field);
    boost::shared_ptr<SimpleContactProblem::SimpleContactElement>
        fe_lhs_simple_contact =
            boost::make_shared<SimpleContactProblem::SimpleContactElement>(
                m_field);
    boost::shared_ptr<SimpleContactProblem::CommonDataSimpleContact>
        common_data_simple_contact =
            boost::make_shared<SimpleContactProblem::CommonDataSimpleContact>(
                m_field);
    if (is_ale) {

      contact_problem->setContactOperatorsRhsALE(
          fe_rhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "MESH_NODE_POSITIONS", "LAGMULT", "MATERIAL");

      contact_problem->setContactOperatorsLhsALE(
          fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "MESH_NODE_POSITIONS", "LAGMULT", "MATERIAL");
    } else {
      cerr << " !!!!!!!  !!!!!!!  !!!!!!!  Nitsche !!!!!!! !!!!!!!  !!!!!!! \n";
      contact_problem->setContactNitschePenaltyRhsOperators(
          fe_rhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "MESH_NODE_POSITIONS", "ELASTIC",
          contact_problem->commonDataSimpleContact->setOfMasterFacesData[1],
          contact_problem->commonDataSimpleContact->setOfSlaveFacesData[1]);
      contact_problem->setContactNitschePenaltyLhsOperators(
          fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "MESH_NODE_POSITIONS", "ELASTIC",
          contact_problem->commonDataSimpleContact->setOfMasterFacesData[1],
          contact_problem->commonDataSimpleContact->setOfSlaveFacesData[1], A);
    }

    CHKERR DMMoFEMSNESSetFunction(dm, "CONTACT_ELEM",
                                  fe_rhs_simple_contact.get(), PETSC_NULL,
                                  PETSC_NULL);

    CHKERR DMMoFEMSNESSetJacobian(dm, "CONTACT_ELEM",
                                  fe_lhs_simple_contact.get(), NULL, NULL);

    if (!is_ale) {
      // CHKERR DMMoFEMSNESSetFunction(dm, "ELASTIC", &elastic.getLoopFeRhs(),
      //                               PETSC_NULL, PETSC_NULL);

      // CHKERR DMMoFEMSNESSetJacobian(dm, "ELASTIC", &elastic.getLoopFeLhs(),
      //                               NULL, NULL);
    }

    SNES snes;
    CHKERR SNESCreate(PETSC_COMM_WORLD, &snes);
    MoFEM::SnesCtx *snes_ctx;
    CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
    CHKERR SNESSetFunction(snes, f, SnesRhs, snes_ctx);
    CHKERR SNESSetJacobian(snes, A, A, SnesMat, snes_ctx);
    CHKERR SNESSetFromOptions(snes);
    CHKERR SNESSolve(snes, NULL, x);

    CHKERR MatView(A, PETSC_VIEWER_STDOUT_WORLD);

    double nrm_A0;
    CHKERR MatNorm(A, NORM_INFINITY, &nrm_A0);

    char testing_options_fd[] = "-snes_fd";
    CHKERR PetscOptionsInsertString(NULL, testing_options_fd);

    CHKERR SNESSetFunction(snes, f, SnesRhs, snes_ctx);
    CHKERR SNESSetJacobian(snes, fdA, fdA, SnesMat, snes_ctx);
    CHKERR SNESSetFromOptions(snes);

    CHKERR SNESSolve(snes, NULL, x);
    CHKERR MatAXPY(A, -1, fdA, SUBSET_NONZERO_PATTERN);

    double nrm_A;
    CHKERR MatNorm(A, NORM_INFINITY, &nrm_A);
    PetscPrintf(PETSC_COMM_WORLD, "Matrix norms %3.4e %3.4e\n", nrm_A,
                nrm_A / nrm_A0);
    nrm_A /= nrm_A0;

    const double tol = 1e-5;
    if (nrm_A > tol) {
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
              "Difference between hand-calculated tangent matrix and finite "
              "difference matrix is too big");
    }

    CHKERR VecDestroy(&x);
    CHKERR VecDestroy(&f);
    CHKERR MatDestroy(&A);
    CHKERR MatDestroy(&fdA);
    CHKERR SNESDestroy(&snes);

    // destroy DM
    CHKERR DMDestroy(&dm);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}