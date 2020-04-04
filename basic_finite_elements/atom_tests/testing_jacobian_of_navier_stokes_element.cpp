/** \file test_jac_navier_stokes_elem.cpp
 * \example test_jac_navier_stokes_elem.cpp

Testing implementation of Navier-Stokes element by verifying tangent stiffness
matrix. Test like this is an example of how to verify the implementation of
Jacobian.

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
#include <NavierStokesElement.hpp>

using namespace boost::numeric;
using namespace MoFEM;

static char help[] = "\n";

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

  PetscBool test_jacobian = PETSC_FALSE;

  try {
    // Get command line options
    char mesh_file_name[255];
    PetscBool flg_file;
    int order_p = 1; // default approximation order_p
    int order_u = 2; // default approximation order_u
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool flg_test = PETSC_FALSE; // true check if error is numerical error

    PetscBool only_stokes = PETSC_FALSE;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "TEST_NAVIER_STOKES problem",
                             "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);
    // Set approximation order
    CHKERR PetscOptionsInt("-my_order_p", "approximation order_p", "", order_p,
                           &order_p, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_order_u", "approximation order_u", "", order_u,
                           &order_u, PETSC_NULL);

    CHKERR PetscOptionsBool("-is_partitioned", "is_partitioned?", "",
                            is_partitioned, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsBool("-only_stokes", "only stokes", "", only_stokes,
                            &only_stokes, PETSC_NULL);

    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test_jacobian", &test_jacobian,
                               PETSC_NULL);
    ierr = PetscOptionsEnd();

    if (flg_file != PETSC_TRUE) {
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
    // CHKERR m_field.getInterface(meshsets_mng_ptr);
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

    // Projection10NodeCoordsOnField ent_method_material(m_field,
    //                                                   "MESH_NODE_POSITIONS");
    // CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);

    PetscRandom rctx;
    PetscRandomCreate(PETSC_COMM_WORLD, &rctx);

    auto set_velocity = [&](VectorAdaptor &&field_data, double *x, double *y,
                            double *z) {
      MoFEMFunctionBegin;
      double value;
      double scale = 2.0;
      PetscRandomGetValueReal(rctx, &value);
      field_data[0] = (value - 0.5) * scale;
      PetscRandomGetValueReal(rctx, &value);
      field_data[1] = (value - 0.5) * scale;
      PetscRandomGetValueReal(rctx, &value);
      field_data[2] = (value - 0.5) * scale;
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

    CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(set_velocity, "U");
    CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(set_pressure, "P");
    CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(
        set_coord, "MESH_NODE_POSITIONS");

    PetscRandomDestroy(&rctx);

    // **** ADD ELEMENTS **** //

    // Add finite element (this defines element, declaration comes later)
    CHKERR m_field.add_finite_element("TEST_NAVIER_STOKES");

    CHKERR m_field.modify_finite_element_add_field_row("TEST_NAVIER_STOKES",
                                                       "U");
    CHKERR m_field.modify_finite_element_add_field_col("TEST_NAVIER_STOKES",
                                                       "U");
    CHKERR m_field.modify_finite_element_add_field_data("TEST_NAVIER_STOKES",
                                                        "U");

    CHKERR m_field.modify_finite_element_add_field_row("TEST_NAVIER_STOKES",
                                                       "P");
    CHKERR m_field.modify_finite_element_add_field_col("TEST_NAVIER_STOKES",
                                                       "P");
    CHKERR m_field.modify_finite_element_add_field_data("TEST_NAVIER_STOKES",
                                                        "P");

    CHKERR m_field.modify_finite_element_add_field_data("TEST_NAVIER_STOKES",
                                                        "MESH_NODE_POSITIONS");

    // Add entities to that element
    CHKERR m_field.add_ents_to_finite_element_by_type(0, MBTET,
                                                      "TEST_NAVIER_STOKES");
    // build finite elements
    CHKERR m_field.build_finite_elements();
    // build adjacencies between elements and degrees of freedom
    CHKERR m_field.build_adjacencies(bit_level0);

    // **** BUILD DM **** //
    DM dm;
    DMType dm_name = "DM_TEST_NAVIER_STOKES";
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
    CHKERR DMMoFEMAddElement(dm, "TEST_NAVIER_STOKES");

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

    boost::shared_ptr<NavierStokesElement::CommonData> commonData =
        boost::make_shared<NavierStokesElement::CommonData>(m_field);

    // auto set_scales_for_block = [&](NavierStokesElement::BlockData &block) {
    //   MoFEMFunctionBegin;
    //   EntityHandle tree_root;
    //   AdaptiveKDTree myTree(&moab);
    //   CHKERR myTree.build_tree(block.eNts, &tree_root);

    //   // get the overall bounding box corners
    //   BoundBox box;
    //   CHKERR myTree.get_bounding_box(box, &tree_root);
    //   block.dimScales.length = box.diagonal_length();
    //   block.dimScales.velocity = 1.0;
    //   CHKERR m_field.getInterface<FieldBlas>()->fieldScale(
    //       (1.0/block.dimScales.length), "MESH_NODE_POSITIONS");

    //   block.dimScales.Re = block.fluidDensity * block.dimScales.velocity *
    //                        block.dimScales.length / block.fluidViscosity;
    //   if (only_stokes) {
    //     block.dimScales.pressure = block.fluidViscosity *
    //                                block.dimScales.velocity /
    //                                block.dimScales.length;
    //     block.inertiaCoef = 0.0;
    //     block.viscousCoef = 1.0;

    //   } 
    //   else {
    //     block.dimScales.pressure = block.fluidDensity *
    //                                block.dimScales.velocity *
    //                                block.dimScales.velocity;
    //     block.inertiaCoef = 1.0;
    //     block.viscousCoef = 1.0 / block.dimScales.Re;
    //   }
    //   MoFEMFunctionReturn(0);
    // };

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 5, "FLUID") == 0) {
        const int id = bit->getMeshsetId();
        CHKERR m_field.get_moab().get_entities_by_type(
            bit->getMeshset(), MBTET, commonData->setOfBlocksData[id].eNts,
            true);
        std::vector<double> attributes;
        bit->getAttributes(attributes);
        if (attributes.size() != 2) {
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                   "should be 2 attributes but is %d", attributes.size());
        }
        commonData->setOfBlocksData[id].iD = id;
        commonData->setOfBlocksData[id].fluidViscosity = attributes[0];
        commonData->setOfBlocksData[id].fluidDensity = attributes[1];
        //set_scales_for_block(commonData->setOfBlocksData[id]);
      }
    }

    if (only_stokes) {
      CHKERR NavierStokesElement::setStokesOperators(feRhs, feLhs, "U", "P",
                                                     commonData);
    } else {
      CHKERR NavierStokesElement::setNavierStokesOperators(feRhs, feLhs, "U",
                                                           "P", commonData);
    }

    CHKERR DMMoFEMSNESSetJacobian(dm, "TEST_NAVIER_STOKES", feLhs, nullFE,
                                  nullFE);
    CHKERR DMMoFEMSNESSetFunction(dm, "TEST_NAVIER_STOKES", feRhs, nullFE,
                                  nullFE);

    Vec x, f;
    CHKERR DMCreateGlobalVector(dm, &x);
    CHKERR VecDuplicate(x, &f);
    CHKERR DMoFEMMeshToLocalVector(dm, x, INSERT_VALUES, SCATTER_FORWARD);

    Mat A, fdA;
    CHKERR DMCreateMatrix(dm, &A);
    CHKERR MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &fdA);

    if (test_jacobian == PETSC_TRUE) {
      char testing_options[] =
          "-snes_test_jacobian -snes_test_jacobian_display "
          "-snes_no_convergence_test -snes_atol 0 -snes_rtol 0 -snes_max_it "
          "1 "
          "-pc_type none";
      CHKERR PetscOptionsInsertString(NULL, testing_options);
      } else {
        char testing_options[] = "-snes_no_convergence_test -snes_atol 0 "
                                 "-snes_rtol 0 -snes_max_it 1 -pc_type none";
        CHKERR PetscOptionsInsertString(NULL, testing_options);
      }

      SNES snes;
      CHKERR SNESCreate(PETSC_COMM_WORLD, &snes);
      MoFEM::SnesCtx *snes_ctx;
      CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
      CHKERR SNESSetFunction(snes, f, SnesRhs, snes_ctx);
      CHKERR SNESSetJacobian(snes, A, A, SnesMat, snes_ctx);
      CHKERR SNESSetFromOptions(snes);

      CHKERR SNESSolve(snes, NULL, x);

      // int ierr = VecView(f, PETSC_VIEWER_STDOUT_WORLD);
      // CHKERRG(ierr);

      PetscInt N;
      VecGetSize(f, &N);
      cout << "f size: " << N << endl;

      if (test_jacobian == PETSC_FALSE) {
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
          SETERRQ(
              PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
              "Difference between hand-calculated tangent matrix and finite "
              "difference matrix is too big");
        }
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