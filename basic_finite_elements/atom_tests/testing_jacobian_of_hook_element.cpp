/** \file testing_jacobian_of_hook_element.cpp
 * \example testing_jacobian_of_hook_element.cpp

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

    PetscBool ale = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-ale", &ale, PETSC_NULL);
    PetscBool test_jacobian = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test_jacobian", &test_jacobian,
                               PETSC_NULL);

    CHKERR DMRegister_MoFEM("DMMOFEM");

    Simple *si = m_field.getInterface<MoFEM::Simple>();

    CHKERR si->getOptions();
    CHKERR si->loadFile();
    CHKERR si->addDomainField("x", H1, AINSWORTH_LEGENDRE_BASE, 3);
    const int order = 2;
    CHKERR si->setFieldOrder("x", order);

    if (ale == PETSC_TRUE) {
      CHKERR si->addDomainField("X", H1, AINSWORTH_LEGENDRE_BASE, 3);
      CHKERR si->setFieldOrder("X", 2);
    }

    CHKERR si->setUp();

    // create DM
    DM dm;
    CHKERR si->getDM(&dm);

    // Projection on "x" field
    {
      Projection10NodeCoordsOnField ent_method(m_field, "x");
      CHKERR m_field.loop_dofs("x", ent_method);
    //   CHKERR m_field.getInterface<FieldBlas>()->fieldScale(2, "x");
    }

    // Project coordinates on "X" field
    if (ale == PETSC_TRUE) {
      Projection10NodeCoordsOnField ent_method(m_field, "X");
      CHKERR m_field.loop_dofs("X", ent_method);
    //   CHKERR m_field.getInterface<FieldBlas>()->fieldScale(2, "X");
    }

    boost::shared_ptr<ForcesAndSourcesCore> fe_lhs_ptr(
        new VolumeElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> fe_rhs_ptr(
        new VolumeElementForcesAndSourcesCore(m_field));
    struct VolRule {
      int operator()(int, int, int) const { return 2 * (order - 1); }
    };
    fe_lhs_ptr->getRuleHook = VolRule();
    fe_rhs_ptr->getRuleHook = VolRule();

    CHKERR DMMoFEMSNESSetJacobian(dm, si->getDomainFEName(), fe_lhs_ptr,
                                  nullptr, nullptr);
    CHKERR DMMoFEMSNESSetFunction(dm, si->getDomainFEName(), fe_rhs_ptr,
                                  nullptr, nullptr);

    using BlockData = NonlinearElasticElement::BlockData;
    using DataAtIntegrationPts = HookeElement::DataAtIntegrationPts;

    boost::shared_ptr<BlockData> block_data_ptr(new BlockData());
    block_data_ptr->iD = 0;
    block_data_ptr->E = 1;
    block_data_ptr->PoissonRatio = 0.25;
    CHKERR m_field.get_finite_element_entities_by_dimension(
        si->getDomainFEName(), 3, block_data_ptr->tEts);

    boost::shared_ptr<DataAtIntegrationPts> data_at_pts(
        new DataAtIntegrationPts());

    if(ale == PETSC_FALSE) {
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateHomogeneousStiffness<true>(
              "x", "x", block_data_ptr, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpLhs_dx_dx<0>("x", "x", data_at_pts));
    } else {
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>("X", data_at_pts->HMat));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateHomogeneousStiffness<true>(
              "x", "x", block_data_ptr, data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>("x", data_at_pts->hMat));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateStrainAle("x", "x", data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateStress<0>("x", "x", data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpAleLhs_dx_dx<0>("x", "x", data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpAleLhs_dx_dX<0>("x", "X", data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateEnergy("X", "X", data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateEshelbyStress("X", "X", data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpAleLhs_dX_dX<0>("X", "X", data_at_pts));
      fe_lhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpAleLhs_dX_dx<0>("X", "x", data_at_pts));
    }

    if(ale == PETSC_FALSE) {
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>("x", data_at_pts->hMat));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateHomogeneousStiffness<true>(
              "x", "x", block_data_ptr, data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateStrain<false>("x", "x", data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateStress<0>("x", "x", data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpRhs_dx("x", "x", data_at_pts));
    } else {
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>("X", data_at_pts->HMat));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateHomogeneousStiffness<true>(
              "x", "x", block_data_ptr, data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>("x", data_at_pts->hMat));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateStrainAle("x", "x", data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateStress<0>("x", "x", data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpAleRhs_dx("x", "x", data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateEnergy("X", "X", data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpCalculateEshelbyStress("X", "X", data_at_pts));
      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpAleRhs_dX("X", "X", data_at_pts));
    }

    Vec x,f;
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
          "-snes_test_jacobian \ -snes_test_jacobian_display "
          "\ -snes_no_convergence_test \ -snes_atol 0 \ -snes_rtol 0 "
          "\ -snes_max_it 1 \ -pc_type none";
      CHKERR PetscOptionsInsertString(NULL, testing_options);
    } else {
      char testing_options[] =
          "\ -snes_no_convergence_test \ -snes_atol 0 \ -snes_rtol 0 "
          "\ -snes_max_it 1 \ -pc_type none";
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