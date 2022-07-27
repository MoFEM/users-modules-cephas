/** \file testing_jacobian_of_hook_scaled_with_density_element.cpp
 * \example testing_jacobian_of_hook_scaled_with_density_element.cpp

Testing implementation of Hook element by verifying tangent stiffness matrix.
Test like this is an example of how to verify the implementation of Jacobian.

*/



#include <BasicFiniteElements.hpp>

using namespace boost::numeric;
using namespace MoFEM;

static char help[] = "\n";

template <bool ALE>
struct OpGetDensityField : public HookeElement::VolUserDataOperator {

  boost::shared_ptr<MatrixDouble> matCoordsPtr;
  boost::shared_ptr<VectorDouble> rhoAtGaussPtsPtr;
  boost::shared_ptr<MatrixDouble> rhoGradAtGaussPtsPtr;
  OpGetDensityField(const std::string row_field,
                    boost::shared_ptr<MatrixDouble> mat_coords_ptr,
                    boost::shared_ptr<VectorDouble> density_at_pts,
                    boost::shared_ptr<MatrixDouble> rho_grad_at_gauss_pts_ptr)
      : HookeElement::VolUserDataOperator(row_field, OPROW),
        matCoordsPtr(mat_coords_ptr), rhoAtGaussPtsPtr(density_at_pts),
        rhoGradAtGaussPtsPtr(rho_grad_at_gauss_pts_ptr) {}

  MoFEMErrorCode doWork(int row_side, EntityType row_type,
                        HookeElement::EntData &row_data) {
    MoFEMFunctionBegin;
    if (row_type != MBVERTEX)
      MoFEMFunctionReturnHot(0);
    // get number of integration points
    const int nb_integration_pts = getGaussPts().size2();
    rhoAtGaussPtsPtr->resize(nb_integration_pts, false);
    rhoAtGaussPtsPtr->clear();
    rhoGradAtGaussPtsPtr->resize(3, nb_integration_pts, false);
    rhoGradAtGaussPtsPtr->clear();

    auto t_rho = getFTensor0FromVec(*rhoAtGaussPtsPtr);
    auto t_grad_rho = getFTensor1FromMat<3>(*rhoGradAtGaussPtsPtr);

    MatrixDouble coords;
    if (ALE)
      coords = *matCoordsPtr;
    else
      coords = trans(getCoordsAtGaussPts()); // because the size is (nb_gg,3)

    FTensor::Index<'i', 3> i;
    auto t_coords = getFTensor1FromMat<3>(coords);

    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      t_rho = 1 + t_coords(i) * t_coords(i); // density is equal to
                                             // 1+x^2+y^2+z^2 (x,y,z coordines)
      t_grad_rho(i) = 2 * t_coords(i);

      ++t_rho;
      ++t_coords;
      ++t_grad_rho;
    }

    MoFEMFunctionReturn(0);
  }
};

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
      // CHKERR m_field.getInterface<FieldBlas>()->fieldScale(1.5, "x"); 
    }

    // Project coordinates on "X" field
    if (ale == PETSC_TRUE) {
      Projection10NodeCoordsOnField ent_method(m_field, "X");
      CHKERR m_field.loop_dofs("X", ent_method);
    //   CHKERR m_field.getInterface<FieldBlas>()->fieldScale(1.5, "X"); 
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
    boost::shared_ptr<map<int, BlockData>> block_sets_ptr =
        boost::make_shared<map<int, BlockData>>();
    (*block_sets_ptr)[0].iD = 0;
    (*block_sets_ptr)[0].E = 1;
    (*block_sets_ptr)[0].PoissonRatio = 0.25;
    CHKERR m_field.get_finite_element_entities_by_dimension(
        si->getDomainFEName(), 3, (*block_sets_ptr)[0].tEts);

    // Parameters for density
    const double rho_n = 2.0;
    const double rho_0 = 0.5;

    auto my_operators = [&](boost::shared_ptr<ForcesAndSourcesCore> &fe_lhs_ptr,
                            boost::shared_ptr<ForcesAndSourcesCore> &fe_rhs_ptr,
                            boost::shared_ptr<map<int, BlockData>>
                                &block_sets_ptr,
                            const std::string x_field,
                            const std::string X_field, const bool ale,
                            const bool field_disp) {
      MoFEMFunctionBeginHot;

      boost::shared_ptr<HookeElement::DataAtIntegrationPts> data_at_pts(
          new HookeElement::DataAtIntegrationPts());
      boost::shared_ptr<MatrixDouble> mat_coords_ptr =
          boost::make_shared<MatrixDouble>();
      boost::shared_ptr<VectorDouble> rho_at_gauss_pts_ptr =
          boost::make_shared<VectorDouble>();
      boost::shared_ptr<MatrixDouble> rho_grad_at_gauss_pts_ptr =
          boost::make_shared<MatrixDouble>();

      if (fe_lhs_ptr) {
        if (ale == PETSC_FALSE) {
          fe_lhs_ptr->getOpPtrVector().push_back(
              new OpCalculateVectorFieldValues<3>(x_field, mat_coords_ptr));
          fe_lhs_ptr->getOpPtrVector().push_back(new OpGetDensityField<false>(
              x_field, mat_coords_ptr, rho_at_gauss_pts_ptr,
              rho_grad_at_gauss_pts_ptr));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateStiffnessScaledByDensityField(
                  x_field, x_field, block_sets_ptr, data_at_pts,
                  rho_at_gauss_pts_ptr, rho_n, rho_0));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpLhs_dx_dx<1>(x_field, x_field, data_at_pts));
        } else {
          fe_lhs_ptr->getOpPtrVector().push_back(
              new OpCalculateVectorFieldGradient<3, 3>(X_field,
                                                       data_at_pts->HMat));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new OpCalculateVectorFieldValues<3>(X_field, mat_coords_ptr));
          fe_lhs_ptr->getOpPtrVector().push_back(new OpGetDensityField<true>(
              X_field, mat_coords_ptr, rho_at_gauss_pts_ptr,
              rho_grad_at_gauss_pts_ptr));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateStiffnessScaledByDensityField(
                  x_field, x_field, block_sets_ptr, data_at_pts,
                  rho_at_gauss_pts_ptr, rho_n, rho_0));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new OpCalculateVectorFieldGradient<3, 3>(x_field,
                                                       data_at_pts->hMat));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateStrainAle(x_field, x_field,
                                                     data_at_pts));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateStress<1>(x_field, x_field,
                                                     data_at_pts)); // FIXME:
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpAleLhs_dx_dx<1>(x_field, x_field,
                                                  data_at_pts));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpAleLhs_dx_dX<1>(x_field, X_field,
                                                  data_at_pts));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateEnergy(X_field, X_field,
                                                  data_at_pts));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateEshelbyStress(X_field, X_field,
                                                         data_at_pts));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpAleLhs_dX_dX<1>(X_field, X_field,
                                                  data_at_pts));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpAleLhsPre_dX_dx<1>(X_field, x_field,
                                                     data_at_pts));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpAleLhs_dX_dx(X_field, x_field, data_at_pts));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpAleLhsWithDensity_dx_dX(
                  x_field, X_field, data_at_pts, rho_at_gauss_pts_ptr,
                  rho_grad_at_gauss_pts_ptr, rho_n, rho_0));
          fe_lhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpAleLhsWithDensity_dX_dX(
                  X_field, X_field, data_at_pts, rho_at_gauss_pts_ptr,
                  rho_grad_at_gauss_pts_ptr, rho_n, rho_0));
        }
      }

      if (fe_rhs_ptr) {

        if (ale == PETSC_FALSE) {
          fe_rhs_ptr->getOpPtrVector().push_back(
              new OpCalculateVectorFieldGradient<3, 3>(x_field,
                                                       data_at_pts->hMat));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new OpCalculateVectorFieldValues<3>(x_field, mat_coords_ptr));
          fe_rhs_ptr->getOpPtrVector().push_back(new OpGetDensityField<false>(
              x_field, mat_coords_ptr, rho_at_gauss_pts_ptr,
              rho_grad_at_gauss_pts_ptr));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateStiffnessScaledByDensityField(
                  x_field, x_field, block_sets_ptr, data_at_pts,
                  rho_at_gauss_pts_ptr, rho_n, rho_0));
          if (field_disp) {
            fe_rhs_ptr->getOpPtrVector().push_back(
                new HookeElement::OpCalculateStrain<1>(x_field, x_field,
                                                       data_at_pts));
          } else {
            fe_rhs_ptr->getOpPtrVector().push_back(
                new HookeElement::OpCalculateStrain<0>(x_field, x_field,
                                                       data_at_pts));
          }
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateStress<1>(x_field, x_field,
                                                     data_at_pts));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpRhs_dx(x_field, x_field, data_at_pts));
        } else {
          fe_rhs_ptr->getOpPtrVector().push_back(
              new OpCalculateVectorFieldGradient<3, 3>(X_field,
                                                       data_at_pts->HMat));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new OpCalculateVectorFieldValues<3>(X_field, mat_coords_ptr));
          fe_rhs_ptr->getOpPtrVector().push_back(new OpGetDensityField<true>(
              X_field, mat_coords_ptr, rho_at_gauss_pts_ptr,
              rho_grad_at_gauss_pts_ptr));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateStiffnessScaledByDensityField(
                  x_field, x_field, block_sets_ptr, data_at_pts,
                  rho_at_gauss_pts_ptr, rho_n, rho_0));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new OpCalculateVectorFieldGradient<3, 3>(x_field,
                                                       data_at_pts->hMat));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateStrainAle(x_field, x_field,
                                                     data_at_pts));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateStress<1>(x_field, x_field,
                                                     data_at_pts));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpAleRhs_dx(x_field, x_field, data_at_pts));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateEnergy(X_field, X_field,
                                                  data_at_pts));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpCalculateEshelbyStress(X_field, X_field,
                                                         data_at_pts));
          fe_rhs_ptr->getOpPtrVector().push_back(
              new HookeElement::OpAleRhs_dX(X_field, X_field, data_at_pts));
        }
      }
      MoFEMFunctionReturnHot(0);
    };
    CHKERR my_operators(fe_lhs_ptr, fe_rhs_ptr, block_sets_ptr, "x", "X", ale,
                        false);
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