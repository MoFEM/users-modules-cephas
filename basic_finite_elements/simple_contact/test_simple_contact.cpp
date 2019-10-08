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

#include <SimpleContact.hpp>

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
                                 "-my_cn_value 1e3 \n";

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
    PetscReal cn_value = 0.;
    PetscBool is_partitioned = PETSC_FALSE;
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

    // // Projection on "x" field
    // {
    // Projection10NodeCoordsOnField ent_method(m_field, "SPATIAL_POSITION");
    // CHKERR m_field.loop_dofs("SPATIAL_POSITION", ent_method);
    // }
    // // CHKERR m_field.getInterface<FieldBlas>()->fieldScale(2, "x");

    // // Project coordinates on "X" field
    // {
    // Projection10NodeCoordsOnField ent_method(m_field, "LAGMULT");
    // CHKERR m_field.loop_dofs("LAGMULT", ent_method);
    // }
    // // CHKERR m_field.getInterface<FieldBlas>()->fieldScale(2, "X");

    auto add_prism_interface = [&](Range &tets, Range &prisms,
                                   Range &slave_tris) {
      MoFEMFunctionBegin;
      PrismInterface *interface;
      CHKERR m_field.getInterface(interface);

      CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
          0, 3, BitRefLevel().set(0));
      std::vector<BitRefLevel> bit_levels;
      bit_levels.push_back(BitRefLevel().set(0));

      Range master_tris;

      int ll = 1;
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, SIDESET, cit)) {
        if (cit->getName().compare(0, 11, "INT_CONTACT") == 0) {
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "Insert $s (id: %d)\n",
                             cit->getName().c_str(), cit->getMeshsetId());
          EntityHandle cubit_meshset = cit->getMeshset();
          Range tris;
          CHKERR moab.get_entities_by_type(cubit_meshset, MBTRI, tris, true);
          master_tris.merge(tris);

          {
            // get tet enties form back bit_level
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
            ref_level_tets.print();
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

      // Range tets_back_bit_level;
      // CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
      //     bit_levels.back(), BitRefLevel().set(), tets_back_bit_level);

      // for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, cit)) {

      //   EntityHandle cubit_meshset = cit->getMeshset();

      //   BlockSetAttributes mydata;
      //   CHKERR cit->getAttributeDataStructure(mydata);
      //   std::cout << mydata << std::endl;

      //   Range tets;
      //   CHKERR moab.get_entities_by_type(cubit_meshset, MBTET, tets, true);
      //   tets = intersect(tets_back_bit_level, tets);
      //   Range nodes;
      //   CHKERR moab.get_connectivity(tets, nodes, true);

      //   for (Range::iterator nit = nodes.begin(); nit != nodes.end(); nit++) {
      //     double coords[3];
      //     CHKERR moab.get_coords(&*nit, 1, coords);
      //     coords[0] += mydata.data.User1;
      //     coords[1] += mydata.data.User2;
      //     coords[2] += mydata.data.User3;
      //     CHKERR moab.set_coords(&*nit, 1, coords);
      //   }
      // }

      EntityHandle meshset_tets;
      EntityHandle meshset_prisms;
      //EntityHandle out_meshset_tets_and_prism;

      CHKERR moab.create_meshset(MESHSET_SET, meshset_tets);
      CHKERR moab.create_meshset(MESHSET_SET, meshset_prisms);
      //CHKERR moab.create_meshset(MESHSET_SET, out_meshset_tets_and_prism);

      CHKERR m_field.getInterface<BitRefManager>()
          ->getEntitiesByTypeAndRefLevel(bit_levels.back(), BitRefLevel().set(),
                                         MBTET, meshset_tets);
      CHKERR moab.get_entities_by_handle(meshset_tets, tets, true);

      CHKERR m_field.getInterface<BitRefManager>()
          ->getEntitiesByTypeAndRefLevel(bit_levels.back(), BitRefLevel().set(),
                                         MBPRISM, meshset_prisms);
      CHKERR moab.get_entities_by_handle(meshset_prisms, prisms);

      //CHKERR moab.add_entities(out_meshset_tets_and_prism, tets);
      //CHKERR moab.add_entities(out_meshset_tets_and_prism, prisms);

      Range tris;
      CHKERR moab.get_adjacencies(prisms, 2, false, tris,
                                  moab::Interface::UNION);
      tris = tris.subset_by_type(MBTRI);
      slave_tris = subtract(tris, master_tris);

      MoFEMFunctionReturn(0);
    };

    // BitRefLevel bit_level0;
    // bit_level0.set(0);
    // CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
    //     0, 3, bit_level0);

    // Range meshset_level0;
    // CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
    //     bit_level0, BitRefLevel().set(), meshset_level0);

    // PetscSynchronizedPrintf(PETSC_COMM_WORLD, "meshset_level0 %d\n",
    //                         meshset_level0.size());
    // PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    // Range range_surf_master, range_surf_slave;
    // for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, SIDESET, it)) {
    //   if (it->getName().compare(0, 6, "Master") == 0) {
    //     CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI,
    //                                                    range_surf_master, true);
    //   }
    // }
    // cout << "range_surf_master = " << range_surf_master.size() << endl;

    // for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, SIDESET, it)) {
    //   if (it->getName().compare(0, 5, "Slave") == 0) {
    //     CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI,
    //                                                    range_surf_slave, true);
    //   }
    // }
    // cout << "range_surf_slave = " << range_surf_slave.size() << endl;

    Range all_tets, contact_prisms, slave_tris;

    CHKERR add_prism_interface(all_tets, contact_prisms, slave_tris);

    cout << "all tets size: " << all_tets.size() << endl;
    all_tets.print();
    cout << "contact prisms size: " << contact_prisms.size() << endl;
    contact_prisms.print();
    cout << "slave tris size: " << slave_tris.size() << endl;
    slave_tris.print();

    CHKERR m_field.add_field("SPATIAL_POSITION", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);

    // Declare problem
    // add entities (by tets) to the field
    // CHKERR m_field.add_ents_to_field_by_type(range_surf_master, MBTRI,
    //                                          "SPATIAL_POSITION");
    // CHKERR m_field.add_ents_to_field_by_type(range_surf_slave, MBTRI,
    //                                          "SPATIAL_POSITION");

    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");
    CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_POSITION", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_POSITION", 1);

    CHKERR m_field.add_field("LAGMULT", H1, AINSWORTH_LEGENDRE_BASE, 1,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                             "LAGMULT");
    CHKERR m_field.set_field_order(0, MBTRI, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBEDGE, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(0, MBVERTEX, "LAGMULT", 1);

    //Range range_slave_master_prisms;

    // create prism
    //@todo : Andrei here we need the conforming prism creations

    //EntityHandle slave_master_1prism;
    //EntityHandle prism_nodes[6];

    // Range::iterator tri_it_master = range_surf_master.begin();

    // int num_nodes_master;
    // const EntityHandle *conn_master = NULL;
    // CHKERR m_field.get_moab().get_connectivity(*tri_it_master, conn_master,
    //                                            num_nodes_master);

    // VectorDouble v_coords_master;
    // v_coords_master.resize(9, false);
    // CHKERR m_field.get_moab().get_coords(conn_master, 3,
    //                                      &*v_coords_master.data().begin());

    // Range::iterator tri_it_slave = range_surf_slave.begin();

    // int num_nodes_slave;
    // const EntityHandle *conn_slave = NULL;
    // CHKERR m_field.get_moab().get_connectivity(*tri_it_slave, conn_slave,
    //                                            num_nodes_slave);

    // VectorDouble v_coords_slave;
    // v_coords_slave.resize(9, false);
    // CHKERR m_field.get_moab().get_coords(conn_slave, 3,
    //                                      &*v_coords_slave.data().begin());

    // for (int ii = 0; ii != 3; ++ii) {
    //   prism_nodes[ii] = conn_master[ii];
    //   prism_nodes[ii + 3] = conn_slave[ii];
    // }

    // CHKERR m_field.get_moab().create_element(MBPRISM, prism_nodes, 6,
    //                                          slave_master_1prism);

    // range_slave_master_prisms.insert(slave_master_1prism);

    // Add these prisim (between master and slave tris) to the mofem database
    // EntityHandle meshset_slave_master_prisms;
    // CHKERR moab.create_meshset(MESHSET_SET, meshset_slave_master_prisms);

    // CHKERR
    // moab.add_entities(meshset_slave_master_prisms, range_slave_master_prisms);

    // CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
    //     meshset_slave_master_prisms, 3, bit_level0);
    // CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
    //     bit_level0, BitRefLevel().set(), meshset_level0);

    // CHKERR moab.write_mesh("slave_master_prisms.vtk",
    //                        &meshset_slave_master_prisms, 1);


    // HERE 

    DMType dm_name = "CONTACT_PROB";
    CHKERR DMRegister_MoFEM(dm_name);

    // create dm instance
    DM dm;
    CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
    CHKERR DMSetType(dm, dm_name);

    boost::shared_ptr<SimpleContactProblem> contact_problem;
    contact_problem = boost::shared_ptr<SimpleContactProblem>(
        new SimpleContactProblem(m_field, r_value, cn_value));

    // ContactProblemSmallDispNoFriction contact_problem(
    //     m_field, contact_commondata_multi_index, r_value, cn_value);

    // add fields to the global matrix by adding the element
    contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                       "LAGMULT", contact_prisms);

    // Forces
    boost::shared_ptr<boost::ptr_map<string, NodalForce>> neumann_forces =
        boost::make_shared<boost::ptr_map<string, NodalForce>>();
    // boost::ptr_map<std::string, NodalForce> neumann_forces;
    string fe_name_str = "FORCE_FE";
    // neumann_forces->insert(fe_name_str, new NodalForce(m_field));

    CHKERR m_field.add_finite_element(fe_name_str);

    CHKERR m_field.modify_finite_element_add_field_row(fe_name_str,
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_col(fe_name_str,
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_data(fe_name_str,
                                                        "SPATIAL_POSITION");

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      Range range_tris, range_vertices;

      CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBVERTEX,
                                                     range_vertices, true);

      CHKERR m_field.add_ents_to_finite_element_by_type(range_vertices,
                                                        MBVERTEX, fe_name_str);
      printf("FORCES TEST !!!! ID %d Size %zu\n", it->getMeshsetId(),
             range_vertices.size());

      printf("Check size %zu\n", range_vertices.size());
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

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    // Its build adjacencies for all elements in database,
    // for practical applications consider to build adjacencies
    // only for refinemnt levels which you use for calculations
    // TODO: build adjacencies only for levels we need
    CHKERR m_field.build_adjacencies(BitRefLevel().set());

    // define problems
    CHKERR m_field.add_problem("CONTACT_PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("CONTACT_PROB", bit_level0);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, dm_name, bit_level0);
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "CONTACT_ELEM");
    CHKERR DMMoFEMAddElement(dm, "FORCE_FE");
    CHKERR DMSetUp(dm);

    Vec x, f;
    CHKERR DMCreateGlobalVector_MoFEM(dm, &x);
    CHKERR VecDuplicate(x, &f);

    CHKERR DMoFEMMeshToLocalVector(dm, x, INSERT_VALUES, SCATTER_FORWARD);

    Mat A, fdA;
    CHKERR DMCreateMatrix_MoFEM(dm, &A);
    CHKERR MatZeroEntries(A);

    contact_problem->setContactOperatorsRhsOperators("SPATIAL_POSITION",
                                                     "LAGMULT");

    contact_problem->setContactOperatorsLhsOperators("SPATIAL_POSITION",
                                                     "LAGMULT", A);

    CHKERR DMMoFEMSNESSetFunction(dm, "CONTACT_ELEM",
                                  contact_problem->feRhsSimpleContact.get(),
                                  PETSC_NULL, PETSC_NULL);

    CHKERR DMMoFEMSNESSetJacobian(dm, "CONTACT_ELEM",
                                  contact_problem->feLhsSimpleContact.get(),
                                  NULL, NULL);

    SNES snes;
    CHKERR SNESCreate(PETSC_COMM_WORLD, &snes);
    MoFEM::SnesCtx *snes_ctx;
    CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
    CHKERR SNESSetFunction(snes, f, SnesRhs, snes_ctx);
    CHKERR SNESSetJacobian(snes, A, A, SnesMat, snes_ctx);
    CHKERR SNESSetFromOptions(snes);
    // CHKERR VecView(f, PETSC_VIEWER_STDOUT_WORLD);
    CHKERR SNESSolve(snes, NULL, x);

    CHKERR MatView(A, PETSC_VIEWER_STDOUT_WORLD);

    // if (test_jacobian == PETSC_FALSE) {
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
    //}

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