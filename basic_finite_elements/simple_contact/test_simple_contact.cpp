/** \file test_simple_contact.cpp
 * \example test_simple_contact.cpp
 *
 * Testing implementation of simple contact element
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
                                 "-my_is_newton_cotes PETSC_FALSE \n"
                                 "-is_lag PETSC_TRUE \n"
                                 "-is_nitsche PETSC_FALSE \n"
                                 "-my_hdiv_trace PETSC_FALSE";

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
    PetscBool is_lag = PETSC_TRUE;
    PetscBool is_nitsche = PETSC_FALSE;
    PetscBool is_hdiv_trace = PETSC_FALSE;
    PetscInt master_move = 0;

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

    CHKERR PetscOptionsBool("-my_is_newton_cotes",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_newton_cotes, PETSC_NULL);

    CHKERR PetscOptionsBool("-is_lag",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_TRUE, &is_lag, PETSC_NULL);

    CHKERR PetscOptionsBool("-is_nitsche",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_nitsche, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_hdiv_trace",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_hdiv_trace, PETSC_NULL);

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

if(is_lag){
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

    // Add elastic element
    boost::shared_ptr<Hooke<adouble>> hooke_adouble_ptr(new Hooke<adouble>());
    boost::shared_ptr<Hooke<double>> hooke_double_ptr(new Hooke<double>());
    NonlinearElasticElement elastic(m_field, 2);
    CHKERR elastic.setBlocks(hooke_double_ptr, hooke_adouble_ptr);
    CHKERR elastic.addElement("ELASTIC", "SPATIAL_POSITION");

    CHKERR elastic.setOperators("SPATIAL_POSITION", "MESH_NODE_POSITIONS",
                                false, false);

    //if (flg_block_config) {
      // try {
      //   ifstream ini_file(block_config_file);

      //   po::variables_map vm;
      //   po::options_description config_file_options;
      //   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      //     std::ostringstream str_order;
      //     str_order << "block_" << it->getMeshsetId() << ".displacement_order";
      //     config_file_options.add_options()(
      //         str_order.str().c_str(),
      //         po::value<int>(&block_data[it->getMeshsetId()].oRder)
      //             ->default_value(order));
      //     std::ostringstream str_cond;
      //     str_cond << "block_" << it->getMeshsetId() << ".young_modulus";
      //     config_file_options.add_options()(
      //         str_cond.str().c_str(),
      //         po::value<double>(&block_data[it->getMeshsetId()].yOung)
      //             ->default_value(-1));
      //     std::ostringstream str_capa;
      //     str_capa << "block_" << it->getMeshsetId() << ".poisson_ratio";
      //     config_file_options.add_options()(
      //         str_capa.str().c_str(),
      //         po::value<double>(&block_data[it->getMeshsetId()].pOisson)
      //             ->default_value(-2));
      //     std::ostringstream str_init_temp;
      //     str_init_temp << "block_" << it->getMeshsetId()
      //                   << ".initial_temperature";
      //     config_file_options.add_options()(
      //         str_init_temp.str().c_str(),
      //         po::value<double>(&block_data[it->getMeshsetId()].initTemp)
      //             ->default_value(0));
      //   }
      //   po::parsed_options parsed =
      //       parse_config_file(ini_file, config_file_options, true);
      //   store(parsed, vm);
      //   po::notify(vm);
      //   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      //     if (block_data[it->getMeshsetId()].oRder == -1)
      //       continue;
      //     if (block_data[it->getMeshsetId()].oRder == order)
      //       continue;
      //     PetscPrintf(PETSC_COMM_WORLD, "Set block %d order to %d\n",
      //                 it->getMeshsetId(), block_data[it->getMeshsetId()].oRder);
      //     Range block_ents;
      //     CHKERR moab.get_entities_by_handle(it->getMeshset(), block_ents,
      //                                        true);

      //     Range ents_to_set_order;
      //     CHKERR moab.get_adjacencies(block_ents, 3, false, ents_to_set_order,
      //                                 moab::Interface::UNION);

      //     ents_to_set_order = ents_to_set_order.subset_by_type(MBTET);
      //     CHKERR moab.get_adjacencies(block_ents, 2, false, ents_to_set_order,
      //                                 moab::Interface::UNION);

      //     CHKERR moab.get_adjacencies(block_ents, 1, false, ents_to_set_order,
      //                                 moab::Interface::UNION);

      //     CHKERR m_field.synchronise_entities(ents_to_set_order);
      //     CHKERR m_field.set_field_order(ents_to_set_order, "SPATIAL_POSITION",
      //                                    block_data[it->getMeshsetId()].oRder);

      //     CHKERR m_field.set_field_order(ents_to_set_order,
      //                                    "PREVIOUS_CONVERGED_SP",
      //                                    block_data[it->getMeshsetId()].oRder);
      //   }
      //   std::vector<std::string> additional_parameters;
      //   additional_parameters =
      //       collect_unrecognized(parsed.options, po::include_positional);
      //   for (std::vector<std::string>::iterator vit =
      //            additional_parameters.begin();
      //        vit != additional_parameters.end(); ++vit) {
      //     CHKERR PetscPrintf(PETSC_COMM_WORLD,
      //                        "** WARNING Unrecognized option %s\n",
      //                        vit->c_str());
      //   }
      // } catch (const std::exception &ex) {
      //   std::ostringstream ss;
      //   ss << ex.what() << std::endl;
      //   SETERRQ(PETSC_COMM_SELF, MOFEM_STD_EXCEPTION_THROW, ss.str().c_str());
      // }
    //}

    boost::shared_ptr<SimpleContactProblem> contact_problem;
    contact_problem = boost::shared_ptr<SimpleContactProblem>(
        new SimpleContactProblem(m_field, r_value, cn_value, is_newton_cotes));

    // add fields to the global matrix by adding the element
    contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                       "LAGMULT", contact_prisms, is_lag);

    CHKERR MetaNeumannForces::addNeumannBCElements(m_field, "SPATIAL_POSITION");

    // Add spring boundary condition applied on surfaces.
    CHKERR MetaSpringBC::addSpringElements(m_field, "SPATIAL_POSITION",
                                           "MESH_NODE_POSITIONS");

    Range solid_faces;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 12, "MAT_ELASTIC2") == 0) {

        Range tets, tet, tris, tet_first;
        const int id = bit->getMeshsetId();
        CHKERR m_field.get_moab().get_entities_by_type(
            bit->getMeshset(), MBTET, tet_first,
            true);
        CHKERR moab.get_adjacencies(tet_first, 2, false, tris,
                                    moab::Interface::UNION);
        tris = intersect(tris, master_tris);

        // boost::shared_ptr<NonlinearElasticElement::BlockData> block_data =
        //     boost::make_shared<NonlinearElasticElement::BlockData>();

        NonlinearElasticElement::BlockData *block_data;

        if (tris.size() == 0) {
          cerr << "M 2 with slave\n";
          tris = intersect(tris, slave_tris);
          block_data = &(contact_problem->commonDataSimpleContact
                  ->setOfSlaveFacesData[1]);
        } else {
          block_data = &(
              contact_problem->commonDataSimpleContact->setOfMasterFacesData[1]);
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
            //     .forcesOnlyOnEntitiesRow.insert(slave_tris.begin(), slave_tris.end());

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

        CHKERR m_field.get_moab().get_entities_by_type(bit->getMeshset(), MBTET,
                                                       tet_first, true);
        CHKERR moab.get_adjacencies(tet_first, 2, false, tris,
                                    moab::Interface::UNION);
        tris = intersect(tris, master_tris);
        NonlinearElasticElement::BlockData *block_data;
        if (tris.size() == 0) {
          cerr << "M 1 with slave\n";
          tris = intersect(tris, slave_tris);
          block_data = &(
              contact_problem->commonDataSimpleContact->setOfSlaveFacesData[1]);
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

    // contact_problem->commonDataSimpleContact->elasticityCommonData
    //     .forcesOnlyOnEntitiesRow();

    // contact_problem->commonDataSimpleContact->elasticityCommonData.setOfBlocks[1].tEts
    // = elastic.setOfBlocks[1].tEts;

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_levels[0]);

    // define problems
    CHKERR m_field.add_problem("CONTACT_PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("CONTACT_PROB",
                                                    bit_levels[0]);

    DMType dm_name = "CONTACT_PROB";
    CHKERR DMRegister_MoFEM(dm_name);

    // create dm instance
    DM dm;
    CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
    CHKERR DMSetType(dm, dm_name);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, dm_name, bit_levels[0]);
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "CONTACT_ELEM");
    CHKERR DMMoFEMAddElement(dm, "ELASTIC");
    CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
    CHKERR DMMoFEMAddElement(dm, "SPRING");

    CHKERR DMSetUp(dm);

    Mat Aij;  // Stiffness matrix
    Vec D, F; //, D0; // Vector of DOFs and the RHS

    CHKERR DMCreateGlobalVector(dm, &D);

    // CHKERR VecZeroEntries(D);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR VecDuplicate(D, &F);
    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR DMCreateMatrix(dm, &Aij);
    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);
    CHKERR MatZeroEntries(Aij);

    // Dirichlet BC
    boost::shared_ptr<FEMethod> dirichlet_bc_ptr =
        boost::shared_ptr<FEMethod>(new DirichletSpatialPositionsBc(
            m_field, "SPATIAL_POSITION", Aij, D, F));
    // (static_cast<DirichletSpatialPositionsBc *>(dirichlet_bc_ptr.get()))
    //     ->methodsOp.push_back(new SimpleContactProblem::LoadScale());

    dirichlet_bc_ptr->snes_ctx = SnesMethod::CTX_SNESNONE;
    dirichlet_bc_ptr->snes_x = D;

    elastic.getLoopFeRhs().snes_f = F;

    boost::shared_ptr<SimpleContactProblem::SimpleContactElement>
        fe_rhs_simple_contact =
            boost::make_shared<SimpleContactProblem::SimpleContactElement>(
                m_field);
    boost::shared_ptr<SimpleContactProblem::SimpleContactElement>
        fe_lhs_simple_contact =
            boost::make_shared<SimpleContactProblem::SimpleContactElement>(
                m_field);
    boost::shared_ptr<SimpleContactProblem::SimpleContactElement>
        fe_post_proc_simple_contact =
            boost::make_shared<SimpleContactProblem::SimpleContactElement>(
                m_field);

    if (is_lag) {
      if (is_hdiv_trace) {

        contact_problem->setContactOperatorsRhsOperatorsHdiv(
            fe_rhs_simple_contact, "SPATIAL_POSITION", "LAGMULT");

        contact_problem->setContactOperatorsLhsOperatorsHdiv(
            fe_lhs_simple_contact, "SPATIAL_POSITION", "LAGMULT", Aij);
      } else {
        contact_problem->setContactOperatorsRhsOperators(
            fe_rhs_simple_contact, "SPATIAL_POSITION", "LAGMULT", "ELASTIC");

        contact_problem->setContactOperatorsLhsOperators(
            fe_lhs_simple_contact, "SPATIAL_POSITION", "LAGMULT", Aij);
      }
} else {
  if (is_nitsche){
    contact_problem->setContactNitschePenaltyRhsOperators(
        fe_rhs_simple_contact, "SPATIAL_POSITION", "MESH_NODE_POSITIONS",
        "ELASTIC");
    contact_problem->setContactNitschePenaltyLhsOperators(
        fe_lhs_simple_contact, "SPATIAL_POSITION", "MESH_NODE_POSITIONS",
        "ELASTIC", Aij);
  }else {
    contact_problem->setContactPenaltyRhsOperators(fe_rhs_simple_contact,
                                                   "SPATIAL_POSITION");

    contact_problem->setContactPenaltyLhsOperators(fe_lhs_simple_contact,
                                                   "SPATIAL_POSITION", Aij);
  }

}

// Assemble pressure and traction forces
boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
CHKERR MetaNeumannForces::setMomentumFluxOperators(m_field, neumann_forces,
                                                   NULL, "SPATIAL_POSITION");

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
    CHKERR DMMoFEMSNESSetFunction(dm, "CONTACT_ELEM", fe_rhs_simple_contact,
                                  PETSC_NULL, PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, "ELASTIC", &elastic.getLoopFeRhs(),
                                  PETSC_NULL, PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, "SPRING", fe_spring_rhs_ptr.get(),
                                  PETSC_NULL, PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL, NULL,
                                  dirichlet_bc_ptr.get());

    boost::shared_ptr<FEMethod> fe_null;
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, dirichlet_bc_ptr,
                                  fe_null);
    CHKERR DMMoFEMSNESSetJacobian(dm, "CONTACT_ELEM", fe_lhs_simple_contact,
                                  NULL, NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, "ELASTIC", &elastic.getLoopFeLhs(), NULL,
                                  NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, "SPRING", fe_spring_lhs_ptr.get(), NULL,
                                  NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, fe_null,
                                  dirichlet_bc_ptr);
    SNES snes;
    SNESConvergedReason snes_reason;
    SnesCtx *snes_ctx;
    // create snes nonlinear solver
    {
      CHKERR SNESCreate(PETSC_COMM_WORLD, &snes);
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

    // CHKERR VecAssemblyBegin(D);
    // CHKERR VecAssemblyEnd(D);

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
    // CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    PetscPrintf(PETSC_COMM_WORLD, "Loop post proc\n");
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);

    elastic.getLoopFeEnergy().snes_ctx = SnesMethod::CTX_SNESNONE;
    elastic.getLoopFeEnergy().eNergy = 0;
    PetscPrintf(PETSC_COMM_WORLD, "Loop energy\n");
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &elastic.getLoopFeEnergy());
    // Print elastic energy
    PetscPrintf(PETSC_COMM_WORLD, "Elastic energy %6.4e\n",
                elastic.getLoopFeEnergy().eNergy);

    string out_file_name;
    std::ostringstream stm;
    stm << "out"
        << ".h5m";
    out_file_name = stm.str();
    CHKERR
    PetscPrintf(PETSC_COMM_WORLD, "out file %s\n", out_file_name.c_str());

    CHKERR post_proc.postProcMesh.write_file(out_file_name.c_str(), "MOAB",
                                             "PARALLEL=WRITE_PART");

    // moab_instance
    moab::Core mb_post;                   // create database
    moab::Interface &moab_proc = mb_post; // create interface to database
    
    if (is_hdiv_trace) {
      contact_problem->setContactOperatorsForPostProcHdiv(
          fe_post_proc_simple_contact, m_field, "SPATIAL_POSITION", "LAGMULT",
          mb_post);
    } else {
      contact_problem->setContactOperatorsForPostProc(
          fe_post_proc_simple_contact, m_field, "SPATIAL_POSITION",
          "MESH_NODE_POSITIONS", "LAGMULT", "ELASTIC", mb_post,
          contact_problem->commonDataSimpleContact->setOfMasterFacesData[1],
          contact_problem->commonDataSimpleContact
              ->setOfSlaveFacesData[1], is_lag);
    }

    mb_post.delete_mesh();
    CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_ELEM",
                                    fe_post_proc_simple_contact);

    std::ostringstream ostrm;

    ostrm << "out_contact"
          << ".h5m";

    out_file_name = ostrm.str();
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
                       out_file_name.c_str());
    CHKERR mb_post.write_file(out_file_name.c_str(), "MOAB",
                              "PARALLEL=WRITE_PART");

    EntityHandle out_meshset_slave_tris;
    EntityHandle out_meshset_master_tris;

    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_slave_tris);
    CHKERR moab.create_meshset(MESHSET_SET, out_meshset_master_tris);

    CHKERR moab.add_entities(out_meshset_slave_tris, slave_tris);
    CHKERR moab.add_entities(out_meshset_master_tris, master_tris);

    CHKERR moab.write_file("out_slave_tris.vtk", "VTK", "",
                           &out_meshset_slave_tris, 1);
    CHKERR moab.write_file("out_master_tris.vtk", "VTK", "",
                           &out_meshset_master_tris, 1);

    CHKERR VecDestroy(&D);
    CHKERR VecDestroy(&F);
    CHKERR MatDestroy(&Aij);
    CHKERR SNESDestroy(&snes);

    // destroy DM
    CHKERR DMDestroy(&dm);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}