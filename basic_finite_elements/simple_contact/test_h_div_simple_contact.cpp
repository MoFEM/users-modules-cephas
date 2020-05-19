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

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config",
                                 "none");

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

    CHKERR PetscOptionsInt("-test_case_lambda", "test case", "", 0,
                           &test_case_lambda, PETSC_NULL);

    CHKERR PetscOptionsInt("-x_x_test_case", "test case", "", 0, &test_case_x,
                           PETSC_NULL);

    CHKERR PetscOptionsInt("-X_test_case", "test case", "", 0, &test_case_X,
                           PETSC_NULL);

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

    CHKERR PetscOptionsBool("-my_is_lag",
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

    Range nodes;
    CHKERR moab.get_adjacencies(master_tris, 0, false, nodes,
                                moab::Interface::UNION);

    nodes.pop_front();
    nodes.pop_back();

    boost::shared_ptr<Hooke<adouble>> hooke_adouble_ptr(new Hooke<adouble>());
    boost::shared_ptr<Hooke<double>> hooke_double_ptr(new Hooke<double>());
    NonlinearElasticElement elastic(m_field, 2);
    CHKERR elastic.setBlocks(hooke_double_ptr, hooke_adouble_ptr);
    CHKERR elastic.addElement("ELASTIC", "SPATIAL_POSITION");

    CHKERR elastic.setOperators("SPATIAL_POSITION", "MESH_NODE_POSITIONS",
                                false, false);

    //contact_problem->commonDataSimpleContact->forcesOnlyOnEntitiesRow = nodes;

//    contact_problem->commonDataSimpleContact->forcesOnlyOnEntitiesCol = nodes;

    // add fields to the global matrix by adding the element

    // boost::shared_ptr<Hooke<adouble>> hooke_adouble_ptr(new Hooke<adouble>());
    // boost::shared_ptr<Hooke<double>> hooke_double_ptr(new Hooke<double>());
    // NonlinearElasticElement elastic(m_field, 2);

    Range slave_tets;

    CHKERR moab.get_adjacencies(slave_tris, 3, false, slave_tets,
                                moab::Interface::UNION);

    slave_tets = slave_tets.subset_by_type(MBTET);
    Range full_slave_tets_tris;
    CHKERR moab.get_adjacencies(slave_tets, 2, false, full_slave_tets_tris,
                                moab::Interface::UNION);
    cerr << "  IS LAG \n";
    cerr << "  IS LAG \n";
    cerr << "  IS LAG \n";
    cerr << "  IS LAG \n";
    cerr << "  IS LAG \n";
    cerr << "  IS LAG \n";
    cerr << "  IS LAG slave tris " << slave_tris.size() << "\n";
    cerr << "  IS LAG tris " << full_slave_tets_tris.size() << "\n";
    cerr << "  IS LAG tets " << slave_tets.size() << "\n";

    CHKERR m_field.add_field("LAGMULT", HDIV, DEMKOWICZ_JACOBI_BASE, 3);

    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "LAGMULT");
    // CHKERR m_field.add_ents_to_field_by_type(0, MBTRI, "LAGMULT");
    CHKERR m_field.set_field_order(0, MBTET, "LAGMULT", 0);
    // CHKERR m_field.set_field_order(0, MBTRI, "LAGMULT", 0);

    // CHKERR m_field.add_ents_to_field_by_type(full_slave_tets_tris, MBTRI,
    //                                          "LAGMULT");

    // CHKERR m_field.add_ents_to_field_by_type(slave_tets, MBTET, "LAGMULT");
    // CHKERR m_field.set_field_order(0, MBTRI, "LAGMULT", order_lambda);
    // CHKERR m_field.set_field_order(0, MBTET, "LAGMULT", order_lambda);

    // CHKERR m_field.set_field_order(slave_tets, "LAGMULT", order_lambda);
    CHKERR m_field.set_field_order(slave_tris, "LAGMULT",
                                   order_lambda);

    

    CHKERR m_field.add_finite_element("HDIVMATERIAL", MF_ZERO);

    CHKERR m_field.modify_finite_element_add_field_row("HDIVMATERIAL",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_col("HDIVMATERIAL",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_data("HDIVMATERIAL",
                                                        "SPATIAL_POSITION");

    CHKERR m_field.modify_finite_element_add_field_row("HDIVMATERIAL",
                                                       "LAGMULT");
    CHKERR m_field.modify_finite_element_add_field_col("HDIVMATERIAL",
                                                       "LAGMULT");
    CHKERR m_field.modify_finite_element_add_field_data("HDIVMATERIAL",
                                                        "LAGMULT");

    CHKERR m_field.modify_finite_element_add_field_data("HDIVMATERIAL",
                                                        "MESH_NODE_POSITIONS");

    CHKERR m_field.add_ents_to_finite_element_by_type(slave_tets, MBTET,
                                                      "HDIVMATERIAL");
    CHKERR m_field.build_finite_elements("HDIVMATERIAL", &slave_tets);

    contact_problem->addContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                       "LAGMULT", contact_prisms);

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
    CHKERR m_field.build_adjacencies(bit_levels[0]);

    // define problems
    CHKERR m_field.add_problem("CONTACT_PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("CONTACT_PROB", bit_levels[0]);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, dm_name, bit_levels[0]);
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "CONTACT_ELEM");
    CHKERR DMMoFEMAddElement(dm, "HDIVMATERIAL");
    CHKERR DMMoFEMAddElement(dm, "ELASTIC");
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
        Range  range_vertices;
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
         itt !=
         m_field.get_dofs_by_name_and_ent_end("SPATIAL_POSITION", *it_vertices);
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
          // dof.getFieldData() = coords[2] - 8.;
          break;
        case 2:
          // dof.getFieldData() = coords[2] + 1.;
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
        printf("After Slave X : %e test case %d\n", dof.getFieldData(), test_case_X);
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
        printf("After Master X : %e test case %d\n", dof.getFieldData(), test_case_X);
      }
    }
  }
}

if(is_lag){
  Range range_vertices;
  CHKERR m_field.get_moab().get_adjacencies(
      slave_tris, 0, false, range_vertices, moab::Interface::UNION);

  for (Range::iterator it_tris = slave_tris.begin();
       it_tris != slave_tris.end(); it_tris++) {

    for (DofEntityByNameAndEnt::iterator itt =
             m_field.get_dofs_by_name_and_ent_begin("LAGMULT", *it_tris);
         itt != m_field.get_dofs_by_name_and_ent_end("LAGMULT", *it_tris);
         ++itt) {

      auto &dof = **itt;

      EntityHandle ent = dof.getEnt();
      int dof_rank = dof.getDofCoeffIdx();

      printf("Before Lambda: %e  and rank %d \n", dof.getFieldData(), dof_rank);

      if (dof_rank != 0){
        switch (test_case_lambda) {

        case 1:
          dof.getFieldData() = -2.5;
          break;
        case 2:
          dof.getFieldData() = +2.5;
          break;
        }
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

    boost::shared_ptr<SimpleContactProblem::SimpleContactElement>
        fe_lhs_simple_contact_2 =
            boost::make_shared<SimpleContactProblem::SimpleContactElement>(
                m_field);

    boost::shared_ptr<SimpleContactProblem::CommonDataSimpleContact>
        common_data_simple_contact =
            boost::make_shared<SimpleContactProblem::CommonDataSimpleContact>(
                m_field);

    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_hdiv_rhs_slave_tet =
        boost::make_shared<VolumeElementForcesAndSourcesCore>(m_field);

    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_hdiv_lhs_slave_tet =
        boost::make_shared<VolumeElementForcesAndSourcesCore>(m_field);

    contact_problem->setContactOperatorsRhsOperatorsHdiv3D(
        fe_rhs_simple_contact, fe_hdiv_rhs_slave_tet, common_data_simple_contact,
        "SPATIAL_POSITION", "LAGMULT", "HDIVMATERIAL");

    contact_problem->setContactOperatorsLhsOperatorsHdiv3D(
        fe_lhs_simple_contact, fe_lhs_simple_contact_2, fe_hdiv_lhs_slave_tet,
        common_data_simple_contact, "SPATIAL_POSITION", "LAGMULT",
        "HDIVMATERIAL");

    // contact_problem->setContactOperatorsLhsOperatorsHdiv(
    //     fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
    //     "LAGMULT", Aij);

    CHKERR DMMoFEMSNESSetFunction(dm, "HDIVMATERIAL",
                                  fe_hdiv_rhs_slave_tet.get(), PETSC_NULL,
                                  PETSC_NULL);

    CHKERR DMMoFEMSNESSetFunction(dm, "CONTACT_ELEM",
                                  fe_rhs_simple_contact.get(), PETSC_NULL,
                                  PETSC_NULL);


    CHKERR DMMoFEMSNESSetJacobian(dm, "CONTACT_ELEM",
                                  fe_lhs_simple_contact.get(), NULL, NULL);

    CHKERR DMMoFEMSNESSetJacobian(dm, "HDIVMATERIAL",
                                  fe_hdiv_lhs_slave_tet.get(), NULL, NULL);

    CHKERR DMMoFEMSNESSetJacobian(dm, "CONTACT_ELEM",
                                  fe_lhs_simple_contact_2.get(), NULL, NULL);
                                  

    // if (!is_ale) {
      CHKERR DMMoFEMSNESSetFunction(dm, "ELASTIC", &elastic.getLoopFeRhs(),
                                    PETSC_NULL, PETSC_NULL);

      CHKERR DMMoFEMSNESSetJacobian(dm, "ELASTIC", &elastic.getLoopFeLhs(),
                                    NULL, NULL);
                                  // }

    
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