/** \file testing_jacobian_of_surface_pressure.cpp
 * \example testing_jacobian_of_surface_pressure.cpp

Testing implementation of surface pressure element (ALE) by verifying tangent
stiffness matrix. Test like this is an example of how to verify the
implementation of Jacobian.

*/



#include <BasicFiniteElements.hpp>

using namespace boost::numeric;
using namespace MoFEM;

static char help[] = "\n";

int main(int argc, char *argv[]) {

  // Initialize MoFEM
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  // Create mesh database
  moab::Core mb_instance;              // create database
  moab::Interface &moab = mb_instance; // create interface to database

  try {
    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    PetscInt order_x = 2;
    PetscInt order_X = 2;
    PetscBool flg = PETSC_TRUE;

    PetscBool test_jacobian = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test_jacobian", &test_jacobian,
                               PETSC_NULL);
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-order_spat", &order_x,
                              &flg);
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-order_mat", &order_X,
                              &flg);

    CHKERR DMRegister_MoFEM("DMMOFEM");

    Simple *si = m_field.getInterface<MoFEM::Simple>();

    CHKERR si->getOptions();
    CHKERR si->loadFile();
    CHKERR si->addDomainField("SPATIAL_POSITION", H1, AINSWORTH_LOBATTO_BASE,
                              3);
    CHKERR si->addBoundaryField("SPATIAL_POSITION", H1, AINSWORTH_LOBATTO_BASE,
                                3);
    CHKERR si->setFieldOrder("SPATIAL_POSITION", order_x);

    CHKERR si->addDomainField("MESH_NODE_POSITIONS", H1,
                              AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR si->addBoundaryField("MESH_NODE_POSITIONS", H1,
                                AINSWORTH_LEGENDRE_BASE, 3);
    CHKERR si->setFieldOrder("MESH_NODE_POSITIONS", order_X);

    Range triangle_springs;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, 9, "SPRING_BC") == 0) {
        CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI,
                                                       triangle_springs, true);
      }
    }

    // Add spring boundary condition applied on surfaces.
    CHKERR MetaSpringBC::addSpringElements(m_field, "SPATIAL_POSITION",
                                           "MESH_NODE_POSITIONS");
    CHKERR MetaSpringBC::addSpringElementsALE(
        m_field, "SPATIAL_POSITION", "MESH_NODE_POSITIONS", triangle_springs);
    si->getOtherFiniteElements().push_back("SPRING");
    si->getOtherFiniteElements().push_back("SPRING_ALE");
    CHKERR si->setUp();

    // create DM
    DM dm;
    CHKERR si->getDM(&dm);

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

    CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(set_coord,
                                                            "SPATIAL_POSITION");
    CHKERR m_field.getInterface<FieldBlas>()->setVertexDofs(
        set_coord, "MESH_NODE_POSITIONS");

    PetscRandomDestroy(&rctx);

    boost::shared_ptr<NeumannForcesSurface> surfacePressure(
        new NeumannForcesSurface(m_field));

    boost::shared_ptr<NeumannForcesSurface::MyTriangleFE> fe_rhs_ptr(
        surfacePressure, &(surfacePressure->getLoopFe()));
    boost::shared_ptr<NeumannForcesSurface::MyTriangleFE> fe_lhs_ptr(
        surfacePressure, &(surfacePressure->getLoopFeLhs()));
    boost::shared_ptr<NeumannForcesSurface::MyTriangleFE> fe_mat_rhs_ptr(
        surfacePressure, &(surfacePressure->getLoopFeMatRhs()));
    boost::shared_ptr<NeumannForcesSurface::MyTriangleFE> fe_mat_lhs_ptr(
        surfacePressure, &(surfacePressure->getLoopFeMatLhs()));

    CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS", *fe_rhs_ptr, false, false);
    CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS", *fe_lhs_ptr, false, false);
    CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS", *fe_mat_rhs_ptr, false, false);
    CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS", *fe_mat_lhs_ptr, false, false);

    Range nodes;
    CHKERR moab.get_entities_by_type(0, MBVERTEX, nodes, false);

    nodes.pop_front();
    nodes.pop_back();

    boost::shared_ptr<NeumannForcesSurface::DataAtIntegrationPts> dataAtPts =
        boost::make_shared<NeumannForcesSurface::DataAtIntegrationPts>();

    dataAtPts->forcesOnlyOnEntitiesRow = nodes;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 8, "PRESSURE") == 0) {
        CHKERR surfacePressure->addPressure("SPATIAL_POSITION", PETSC_NULL,
                                            bit->getMeshsetId(), true, true);
        CHKERR surfacePressure->addPressureAle(
            "SPATIAL_POSITION", "MESH_NODE_POSITIONS", dataAtPts,
            si->getDomainFEName(), PETSC_NULL, PETSC_NULL, bit->getMeshsetId(),
            true, true);
      }
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

    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ale_ptr_dx(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ale_ptr_dX(
        new FaceElementForcesAndSourcesCore(m_field));

    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ale_ptr(
        new FaceElementForcesAndSourcesCore(m_field));

    boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
        data_at_spring_gp =
            boost::make_shared<MetaSpringBC::DataAtIntegrationPtsSprings>(
                m_field);

    Range spring_ale_nodes;
    CHKERR moab.get_connectivity(triangle_springs, spring_ale_nodes, true);

    data_at_spring_gp->forcesOnlyOnEntitiesRow = spring_ale_nodes;

    CHKERR MetaSpringBC::setSpringOperatorsMaterial(
        m_field, fe_spring_lhs_ale_ptr_dx, fe_spring_lhs_ale_ptr_dX,
        fe_spring_rhs_ale_ptr, data_at_spring_gp, "SPATIAL_POSITION",
        "MESH_NODE_POSITIONS", si->getDomainFEName());

    CHKERR DMMoFEMSNESSetJacobian(dm, "SPRING", fe_spring_lhs_ptr, PETSC_NULL,
                                  PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, "SPRING", fe_spring_rhs_ptr, PETSC_NULL,
                                  PETSC_NULL);

    CHKERR DMMoFEMSNESSetJacobian(dm, "SPRING_ALE", fe_spring_lhs_ale_ptr_dx,
                                  PETSC_NULL, PETSC_NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, "SPRING_ALE", fe_spring_lhs_ale_ptr_dX,
                                  PETSC_NULL, PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, "SPRING_ALE", fe_spring_rhs_ale_ptr,
                                  PETSC_NULL, PETSC_NULL);

    CHKERR DMMoFEMSNESSetJacobian(dm, si->getBoundaryFEName(), fe_lhs_ptr,
                                  nullptr, nullptr);
    CHKERR DMMoFEMSNESSetFunction(dm, si->getBoundaryFEName(), fe_rhs_ptr,
                                  nullptr, nullptr);

    CHKERR DMMoFEMSNESSetJacobian(dm, si->getBoundaryFEName(), fe_mat_lhs_ptr,
                                  nullptr, nullptr);
    CHKERR DMMoFEMSNESSetFunction(dm, si->getBoundaryFEName(), fe_mat_rhs_ptr,
                                  nullptr, nullptr);

    Vec x, f;
    CHKERR DMCreateGlobalVector(dm, &x);
    CHKERR VecDuplicate(x, &f);
    CHKERR VecSetOption(f, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
    CHKERR DMoFEMMeshToLocalVector(dm, x, INSERT_VALUES, SCATTER_FORWARD);

    Mat A, fdA;
    CHKERR DMCreateMatrix(dm, &A);
    CHKERR MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &fdA);

    if (test_jacobian == PETSC_TRUE) {
      char testing_options[] =
          "-snes_test_jacobian -snes_test_jacobian_display "
          "-snes_no_convergence_test -snes_atol 0 -snes_rtol 0 -snes_max_it 1 ";
      //"-pc_type none";
      CHKERR PetscOptionsInsertString(NULL, testing_options);
    } else {
      char testing_options[] = "-snes_no_convergence_test -snes_atol 0 "
                               "-snes_rtol 0 "
                               "-snes_max_it 1 ";
      //"-pc_type none";
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

      const double tol = 1e-7;
      if (nrm_A > tol) {
        SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
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