/** \file testing_jacobian_of_surface_pressure.cpp
 * \example testing_jacobian_of_surface_pressure.cpp

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

using namespace boost::numeric;
using namespace MoFEM;

static char help[] = "\n";

/* struct ScaleOp: public MethodForForceScaling {
  //Hassan: This function to read data file (once) and save it in a pair vector
ts

  ScaleOp(
  ):
 {

  }

  //Hassan: this function will loop over data in pair vector ts to find load
  //scale based on ts_t
  MoFEMErrorCode scaleNf(const FEMethod *fe,VectorDouble &Nf) {
    MoFEMFunctionBeginHot;
    NF*=2;
    MoFEMFunctionReturnHot(0);
  }
}; */

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
    PetscInt order_X = 1;
    PetscBool flg = PETSC_TRUE;

    // PetscBool ale = PETSC_FALSE;
    // CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-ale", &ale, PETSC_NULL);
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
    CHKERR si->addDomainField("x", H1, AINSWORTH_LOBATTO_BASE, 3);
    CHKERR si->addBoundaryField("x", H1, AINSWORTH_LOBATTO_BASE, 3);
    CHKERR si->setFieldOrder("x", order_x);

    CHKERR si->addDomainField("X", H1, AINSWORTH_LOBATTO_BASE, 3);
    CHKERR si->addBoundaryField("X", H1, AINSWORTH_LOBATTO_BASE, 3);
    CHKERR si->setFieldOrder("X", order_X);

    CHKERR si->setUp();

    // create DM
    DM dm;
    CHKERR si->getDM(&dm);

    // Projection on "x" field
    {
      Projection10NodeCoordsOnField ent_method(m_field, "x");
      CHKERR m_field.loop_dofs("x", ent_method);
      // CHKERR m_field.getInterface<FieldBlas>()->fieldScale(2, "x");
    }

    // Project coordinates on "X" field
    {
      Projection10NodeCoordsOnField ent_method(m_field, "X");
      CHKERR m_field.loop_dofs("X", ent_method);
      // CHKERR m_field.getInterface<FieldBlas>()->fieldScale(2, "X");
    }

    boost::shared_ptr<NeummanForcesSurface::DataAtIntegrationPts>
        dataAtIntegrationPts =
            boost::make_shared<NeummanForcesSurface::DataAtIntegrationPts>();

    boost::shared_ptr<NeummanForcesSurface::DataAtIntegrationPtsMat>
        data_at_pts =
            boost::make_shared<NeummanForcesSurface::DataAtIntegrationPtsMat>();

    boost::shared_ptr<NeummanForcesSurface> surfacePressure(
        new NeummanForcesSurface(m_field));

    boost::shared_ptr<NeummanForcesSurface::MyTriangleFE> fe_rhs_ptr(
        surfacePressure, &(surfacePressure->getLoopFe()));
    boost::shared_ptr<NeummanForcesSurface::MyTriangleFE> fe_lhs_ptr(
        surfacePressure, &(surfacePressure->getLoopFeLhs()));

    boost::shared_ptr<NeummanForcesSurface::MyTriangleFE> fe_mat_rhs_ptr(
        surfacePressure, &(surfacePressure->getLoopFeMatRhs()));
    boost::shared_ptr<NeummanForcesSurface::MyTriangleFE> fe_mat_lhs_ptr(
        surfacePressure, &(surfacePressure->getLoopFeMatLhs()));

    fe_rhs_ptr->meshPositionsFieldName = "X";
    fe_lhs_ptr->meshPositionsFieldName = "X";
    fe_mat_rhs_ptr->meshPositionsFieldName = "X";
    fe_mat_lhs_ptr->meshPositionsFieldName = "X";
    // fe_rhs_ptr->addToRule = 2;
    // fe_lhs_ptr->addToRule = 2;

    /* for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
               cout << it->getName() << endl;
      CHKERR surfacePressure->addPressure("x", "X", dataAtIntegrationPts,
                                          PETSC_NULL, PETSC_NULL,
                                          it->getMeshsetId(), false, false) ;
    }  */

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe_rhs(
      new VolumeElementForcesAndSourcesCoreOnSide(m_field));
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe_lhs(
      new VolumeElementForcesAndSourcesCoreOnSide(m_field));

    boost::shared_ptr<double> lambda_ptr = boost::make_shared<double>(1.0);

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      cout << bit->getName() << endl;
      if (bit->getName().compare(0, 8, "PRESSURE") == 0) {
         CHKERR surfacePressure->addPressure(
            "x", "X", dataAtIntegrationPts, PETSC_NULL, PETSC_NULL,
            bit->getMeshsetId(), lambda_ptr, true, true); 
        // CHKERR surfacePressure->addPressureMaterial(
        //     "x", "X", data_at_pts, side_fe_rhs, side_fe_lhs,
        //     si->getDomainFEName(), PETSC_NULL, PETSC_NULL, bit->getMeshsetId(),
        //     lambda_ptr, true, true); 
      }
    }

    // CHKERR DMMoFEMSNESSetJacobian(dm, si->getBoundaryFEName(), fe_lhs_ptr,
    //                               nullptr, nullptr);
    // CHKERR DMMoFEMSNESSetFunction(dm, si->getBoundaryFEName(), fe_rhs_ptr,
    //                               nullptr, nullptr); 

    CHKERR DMMoFEMSNESSetJacobian(dm, si->getBoundaryFEName(), fe_mat_lhs_ptr,
                                  nullptr, nullptr);
    CHKERR DMMoFEMSNESSetFunction(dm, si->getBoundaryFEName(), fe_mat_rhs_ptr,
                                  nullptr, nullptr);

    Vec x, f;
    CHKERR DMCreateGlobalVector(dm, &x);
    CHKERR VecDuplicate(x, &f);
    CHKERR DMoFEMMeshToLocalVector(dm, x, INSERT_VALUES, SCATTER_FORWARD);

    // CHKERR VecDuplicate(x, &dx);
    // PetscRandom rctx;
    // PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
    // VecSetRandom(dx, rctx);
    // PetscRandomDestroy(&rctx);
    // CHKERR DMoFEMMeshToGlobalVector(dm, x, INSERT_VALUES, SCATTER_REVERSE);

    Mat A, fdA;
    CHKERR DMCreateMatrix(dm, &A);
    CHKERR MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &fdA);

    if (test_jacobian == PETSC_TRUE) {
      char testing_options[] =
          "-snes_test_jacobian -snes_test_jacobian_display "
          "-snes_no_convergence_test -snes_atol 0 -snes_rtol 0 -snes_max_it 1 "
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
        SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                "Difference between hand-calculated tangent matrix and finite "
                "difference matrix is too big");
      }
    }

    int size;
    VecGetSize(f, &size);
    cout << "f size: " << size << endl;

    int m, n;
    MatGetSize(A, &m, &n);
    cout << "A size: " << m << " " << n << endl;

    // int ierr;
    // cout << "----- Start printting f -----" << endl;
    // ierr = VecView(f, PETSC_VIEWER_STDOUT_WORLD); CHKERRG(ierr);
    // cout << "----- Finish printting f -----" << endl;

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