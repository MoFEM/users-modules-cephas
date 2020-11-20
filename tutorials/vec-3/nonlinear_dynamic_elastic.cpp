/**
 * \file dynamic_elastic.cpp
 * \example dynamic_elastic.cpp
 *
 * Plane stress elastic dynamic problem
 *
 */

/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
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

#include <MoFEM.hpp>
#include <MatrixFunction.hpp>
using namespace MoFEM;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = FaceElementForcesAndSourcesCoreBase;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCoreBase;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
};

constexpr int SPACE_DIM = EXECUTABLE_DIMENSION; //< Space dimension of problem, mesh

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = ElementsAndOps<SPACE_DIM>::DomainEleOp;
using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;

using OpK = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;
using OpMass = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, SPACE_DIM>;
using OpInternalForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;
using OpBodyForce = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpSource<1, SPACE_DIM>;
using OpInertiaForce = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesVector<1, SPACE_DIM, 1>;

constexpr double rho = 1;
constexpr double omega = 2.4;
constexpr double young_modulus = 1;
constexpr double poisson_ratio = 0.25;
constexpr double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
constexpr double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

template <int DIM>
inline MoFEMErrorCode
get_eigen_val_and_proj_lapack(const FTensor::Tensor2_symmetric<double, DIM> &X,
                              FTensor::Tensor1<double, DIM> &eig,
                              FTensor::Tensor2<double, DIM, DIM> &eig_vec) {

  MoFEMFunctionBeginHot;
  for (int ii = 0; ii != DIM; ii++)
    for (int jj = 0; jj != DIM; jj++)
      eig_vec(ii, jj) = X(ii, jj);

  int n = DIM, lda = DIM, info, lwork = -1;
  double wkopt;
  info =
      lapack_dsyev('V', 'U', n, &(eig_vec(0, 0)), lda, &eig(0), &wkopt, lwork);
  lwork = (int)wkopt;
  double work[lwork];
  info = lapack_dsyev('V', 'U', n, &(eig_vec(0, 0)), lda, &eig(0), work, lwork);

  MoFEMFunctionReturnHot(0);
}

template <int DIM, bool IS_LHS>
struct OpHenckyStressAndTangent : public DomainEleOp {
  OpHenckyStressAndTangent(const std::string field_name,
                           boost::shared_ptr<MatrixDouble> mat_grad_ptr,
                           boost::shared_ptr<MatrixDouble> mat_d_ptr,
                           boost::shared_ptr<MatrixDouble> mat_stress_ptr,
                           boost::shared_ptr<MatrixDouble> mat_tangent_ptr)
      : DomainEleOp(field_name, DomainEleOp::OPROW), matGradPtr(mat_grad_ptr),
        matDPtr(mat_d_ptr), matStressPtr(mat_stress_ptr),
        matTangentPtr(mat_tangent_ptr) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    FTensor::Index<'i', DIM> i;
    FTensor::Index<'j', DIM> j;
    FTensor::Index<'k', DIM> k;
    FTensor::Index<'l', DIM> l;
    FTensor::Index<'m', DIM> m;
    FTensor::Index<'n', DIM> n;
    FTensor::Index<'o', DIM> o;
    FTensor::Index<'p', DIM> p;

    // auto get_tensor4_from_ddg = [](auto &ddg) {
    //   FTensor::Tensor4<double, DIM, DIM, DIM, DIM> tens4;
    //   for (int ii = 0; ii != DIM; ++ii)
    //     for (int jj = 0; jj != DIM; ++jj)
    //       for (int kk = 0; kk != DIM; ++kk)
    //         for (int ll = 0; ll != DIM; ++ll)
    //           tens4(ii, jj, kk, ll) = ddg(ii, jj, kk, ll);
    //   return tens4;
    // };

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    // const size_t nb_gauss_pts = matGradPtr->size2();
    const size_t nb_gauss_pts = getGaussPts().size2();
    constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
    matTangentPtr->resize(DIM * DIM * DIM * DIM, nb_gauss_pts);
    matStressPtr->resize(DIM * DIM, nb_gauss_pts, false);
    auto dP_dF = getFTensor4FromMat<DIM, DIM, DIM, DIM, 1>(*matTangentPtr);

    auto t_D = getFTensor4DdgFromMat<DIM, DIM, 0>(*matDPtr);
    auto t_grad = getFTensor2FromMat<DIM, DIM>(*matGradPtr);
    auto t_stress = getFTensor2FromMat<DIM, DIM>(*matStressPtr);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      FTensor::Tensor2<double, DIM, DIM> F;
      FTensor::Tensor2_symmetric<double, DIM> C; // right Cauchy Green
      FTensor::Tensor2_symmetric<double, DIM> T; // stress measure
      FTensor::Tensor2_symmetric<double, DIM> S; // 2nd Piola Stress
      FTensor::Tensor2<double, DIM, DIM> P;      // 1st Piola Stress
      FTensor::Tensor1<double, DIM> eig;
      FTensor::Tensor2<double, DIM, DIM> eigen_vec;

      auto f = [](double v) { return 0.5 * log(v); };
      auto d_f = [](double v) { return 0.5 / v; };
      auto dd_f = [](double v) { return -0.5 / (v * v); };

      F(i, j) = t_grad(i, j) + t_kd(i, j);
      C(i, j) = F(k, i) ^ F(k, j);

      CHKERR get_eigen_val_and_proj_lapack<DIM>(C, eig, eigen_vec);

      auto it = unique(&eig(0), &eig(0) + DIM,
                       [](double a, double b) { return abs(a - b) < 1e-8; });

      int nb_uniq = std::distance(&eig(0), it);
      if(nb_uniq == 2)
        string wait2; //here you have to sort eigen values and vectors //FIXME:
      auto logC = EigenMatrix::getMat(eig, eigen_vec, f);

      T(i, j) = t_D(i, j, k, l) * logC(k, l);

      auto dlogC_dC =
          EigenMatrix::getDiffMat(eig, eigen_vec, f, d_f, nb_uniq);
      dlogC_dC(i, j, k, l) *= 2;

      S(k, l) = T(i, j) * dlogC_dC(i, j, k, l);
      P(i, l) = F(i, k) * S(k, l);

      if (IS_LHS) {
        FTensor::Tensor4<double, DIM, DIM, DIM, DIM> dC_dF;
        dC_dF(i, j, k, l) = (t_kd(i, l) * F(k, j)) + (t_kd(j, l) * F(k, i));

        auto TL =
            EigenMatrix::getDiffDiffMat(eig, eigen_vec, f, d_f, dd_f, T, nb_uniq);

        TL(i, j, k, l) *= 4;
        FTensor::Ddg<double, DIM, DIM> P_D_P_plus_TL;
        P_D_P_plus_TL(i, j, k, l) =
            TL(i, j, k, l) +
            (dlogC_dC(i, j, o, p) * t_D(o, p, m, n)) * dlogC_dC(m, n, k, l);
        P_D_P_plus_TL(i, j, k, l) *= 0.5;
        dP_dF(i, j, m, n) = t_kd(i, m) * (t_kd(k, n) * S(k, j));
        dP_dF(i, j, m, n) +=
            F(i, k) * (P_D_P_plus_TL(k, j, o, p) * dC_dF(o, p, m, n));

        ++dP_dF;
      }

      t_stress(i, j) = P(i, j);
      ++t_grad;
      ++t_stress;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> matGradPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;
  boost::shared_ptr<MatrixDouble> matStressPtr;
  boost::shared_ptr<MatrixDouble> matTangentPtr;
};

#include <OpPostProcElastic.hpp>
using namespace Tutorial;

static double *ts_time_ptr;
static double *ts_aa_ptr;

struct Example {

  Example(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEM::Interface &mField;

  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode createCommonData();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();
  MoFEMErrorCode outputResults();
  MoFEMErrorCode checkResults();

  MatrixDouble invJac;
  boost::shared_ptr<MatrixDouble> matGradPtr;
  boost::shared_ptr<MatrixDouble> matStrainPtr;
  boost::shared_ptr<MatrixDouble> matStressPtr;
  boost::shared_ptr<MatrixDouble> matAccelerationPtr;
  boost::shared_ptr<MatrixDouble> matInertiaPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;

  boost::shared_ptr<MatrixDouble> matTangentPtr;
};

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

  auto set_matrial_stiffness = [&]() {
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Index<'j', SPACE_DIM> j;
    FTensor::Index<'k', SPACE_DIM> k;
    FTensor::Index<'l', SPACE_DIM> l;
    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
    MoFEMFunctionBegin;
    constexpr double A =
        (SPACE_DIM == 2) ? 2 * shear_modulus_G /
                               (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                         : 1;
    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*matDPtr);
    t_D(i, j, k, l) = 2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
                      A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                          t_kd(i, j) * t_kd(k, l);
    MoFEMFunctionReturn(0);
  };

  matGradPtr = boost::make_shared<MatrixDouble>();
  matStrainPtr = boost::make_shared<MatrixDouble>();
  matStressPtr = boost::make_shared<MatrixDouble>();
  matAccelerationPtr = boost::make_shared<MatrixDouble>();
  matInertiaPtr = boost::make_shared<MatrixDouble>();
  matDPtr = boost::make_shared<MatrixDouble>();

  matTangentPtr = boost::make_shared<MatrixDouble>();

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  matDPtr->resize(size_symm * size_symm, 1);

  CHKERR set_matrial_stiffness();

  MoFEMFunctionReturn(0);
}
//! [Create common data]

//! [Run problem]
MoFEMErrorCode Example::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR createCommonData();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  CHKERR outputResults();
  CHKERR checkResults();
  MoFEMFunctionReturn(0);
}
//! [Run problem]

//! [Read mesh]
MoFEMErrorCode Example::readMesh() {
  MoFEMFunctionBegin;
  auto simple = mField.getInterface<Simple>();
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();
  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode Example::setupProblem() {
  MoFEMFunctionBegin;
  Simple *simple = mField.getInterface<Simple>();
  // Add field
  CHKERR simple->addDomainField("U", H1, AINSWORTH_BERNSTEIN_BEZIER_BASE,
                                SPACE_DIM);
  int order = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;

  auto fix_disp = [&](const std::string blockset_name) {
    Range fix_ents;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
      if (it->getName().compare(0, blockset_name.length(), blockset_name) ==
          0) {
        CHKERR mField.get_moab().get_entities_by_handle(it->meshset, fix_ents,
                                                        true);
      }
    }
    return fix_ents;
  };

  auto remove_ents = [&](const Range &&ents, const int lo, const int hi) {
    auto prb_mng = mField.getInterface<ProblemsManager>();
    auto simple = mField.getInterface<Simple>();
    MoFEMFunctionBegin;
    Range verts;
    CHKERR mField.get_moab().get_connectivity(ents, verts, true);
    verts.merge(ents);
    if (SPACE_DIM == 3) {
      Range adj;
      CHKERR mField.get_moab().get_adjacencies(ents, 1, false, adj,
                                               moab::Interface::UNION);
      verts.merge(adj);
    };

    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), "U", verts,
                                         lo, hi);
    MoFEMFunctionReturn(0);
  };

  CHKERR remove_ents(fix_disp("FIX_X"), 0, 0);
  CHKERR remove_ents(fix_disp("FIX_Y"), 1, 1);
  CHKERR remove_ents(fix_disp("FIX_Z"), 2, 2);
  CHKERR remove_ents(fix_disp("FIX_ALL"), 0, 3);

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  if (SPACE_DIM == 2) {
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainLhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpCalculateInvJacForFace(invJac));
    pipeline_mng->getOpDomainRhsPipeline().push_back(
        new OpSetInvJacH1ForFace(invJac));
  }

  // Get pointer to U_tt shift in domain element
  auto get_rho = [this](const double, const double, const double) {
    auto *pipeline_mng = mField.getInterface<PipelineManager>();
    auto &fe_domain_lhs = pipeline_mng->getDomainLhsFE();
    return rho * fe_domain_lhs->ts_aa;
  };

  auto get_body_force = [this](const double, const double, const double) {
    auto *pipeline_mng = mField.getInterface<PipelineManager>();
    auto fe_domain_rhs = pipeline_mng->getDomainRhsFE();
    FTensor::Index<'i', SPACE_DIM> i;
    FTensor::Tensor1<double, SPACE_DIM> t_source;
    t_source(i) = 0.;
    t_source(0) = 0.1;
    t_source(1) = 1.;
    const auto time = fe_domain_rhs->ts_t;
    t_source(i) *= sin(time * omega * M_PI);
    return t_source;
  };

  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               matGradPtr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpHenckyStressAndTangent<SPACE_DIM, true>(
          "U", matGradPtr, matDPtr, matStressPtr, matTangentPtr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpK("U", "U", matTangentPtr));
  pipeline_mng->getOpDomainLhsPipeline().push_back(
      new OpMass("U", "U", get_rho));

  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpBodyForce("U", get_body_force));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               matGradPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpHenckyStressAndTangent<SPACE_DIM, false>(
          "U", matGradPtr, matDPtr, matStressPtr, matTangentPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInternalForce("U", matStressPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpCalculateVectorFieldValuesDotDot<SPACE_DIM>("U",
                                                        matAccelerationPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpScaleMatrix("U", rho, matAccelerationPtr, matInertiaPtr));
  pipeline_mng->getOpDomainRhsPipeline().push_back(
      new OpInertiaForce("U", matInertiaPtr));

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  MoFEMFunctionReturn(0);
}
//! [Push operators to pipeline]

/**
 * @brief Monitor solution
 *
 * This functions is called by TS solver at the end of each step. It is used
 * to output results to the hard drive.
 */
struct Monitor : public FEMethod {
  Monitor(SmartPetscObj<DM> dm, boost::shared_ptr<PostProcEle> post_proc)
      : dM(dm), postProc(post_proc){};
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    constexpr int save_every_nth_step = 1;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR postProc->writeFile(
          "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
    }
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProc;
};

//! [Solve]
MoFEMErrorCode Example::solveSystem() {
  MoFEMFunctionBegin;
  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto dm = simple->getDM();
  auto ts = pipeline_mng->createTS2();

  // Setup postprocessing
  auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
  post_proc_fe->generateReferenceElementMesh();
  if (SPACE_DIM == 2) {
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(invJac));
    post_proc_fe->getOpPtrVector().push_back(new OpSetInvJacH1ForFace(invJac));
  }
  post_proc_fe->getOpPtrVector().push_back(
      new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>("U",
                                                               matGradPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpSymmetrizeTensor<SPACE_DIM>("U", matGradPtr, matStrainPtr));
  post_proc_fe->getOpPtrVector().push_back(
      new OpTensorTimesSymmetricTensor<SPACE_DIM, SPACE_DIM>(
          "U", matStrainPtr, matStressPtr, matDPtr));
  post_proc_fe->getOpPtrVector().push_back(new OpPostProcElastic<SPACE_DIM>(
      "U", post_proc_fe->postProcMesh, post_proc_fe->mapGaussPts, matStrainPtr,
      matStressPtr));
  post_proc_fe->addFieldValuesPostProc("U");

  // Add monitor to time solver
  boost::shared_ptr<FEMethod> null_fe;
  auto monitor_ptr = boost::make_shared<Monitor>(dm, post_proc_fe);
  CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                             null_fe, monitor_ptr);

  double ftime = 1;
  CHKERR TSSetMaxTime(ts, ftime);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  auto T = smartCreateDMVector(simple->getDM());
  auto TT = smartVectorDuplicate(T);
  CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                 SCATTER_FORWARD);

  CHKERR TS2SetSolution(ts, T, TT);
  CHKERR TSSetFromOptions(ts);

  CHKERR TSSolve(ts, NULL);
  CHKERR TSGetTime(ts, &ftime);

  PetscInt steps, snesfails, rejects, nonlinits, linits;
  CHKERR TSGetStepNumber(ts, &steps);
  CHKERR TSGetSNESFailures(ts, &snesfails);
  CHKERR TSGetStepRejections(ts, &rejects);
  CHKERR TSGetSNESIterations(ts, &nonlinits);
  CHKERR TSGetKSPIterations(ts, &linits);
  MOFEM_LOG_C("EXAMPLE", Sev::inform,
              "steps %D (%D rejected, %D SNES fails), ftime %g, nonlinits "
              "%D, linits %D\n",
              steps, rejects, snesfails, ftime, nonlinits, linits);

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
  PetscBool test_flg = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-test", &test_flg, PETSC_NULL);
  if (test_flg) {
    auto *simple = mField.getInterface<Simple>();
    auto T = smartCreateDMVector(simple->getDM());
    CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                   SCATTER_FORWARD);
    double nrm2;
    CHKERR VecNorm(T, NORM_2, &nrm2);
    MOFEM_LOG("EXAMPLE", Sev::inform) << "Regression norm " << nrm2;
    constexpr double regression_value = 1.09572;
    if (fabs(nrm2 - regression_value) > 1e-2)
      SETERRQ(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
              "Regression test faileed; wrong norm value.");
  }
  MoFEMFunctionReturn(0);
}
//! [Postprocess results]

//! [Check]
MoFEMErrorCode Example::checkResults() {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}
//! [Check]

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "EXAMPLE"));
  LogManager::setLog("EXAMPLE");
  MOFEM_LOG_TAG("EXAMPLE", "example");

  try {

    //! [Register MoFEM discrete manager in PETSc]
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    //! [Register MoFEM discrete manager in PETSc

    //! [Create MoAB]
    moab::Core mb_instance;              ///< mesh database
    moab::Interface &moab = mb_instance; ///< mesh database interface
    //! [Create MoAB]

    //! [Create MoFEM]
    MoFEM::Core core(moab);           ///< finite element database
    MoFEM::Interface &m_field = core; ///< finite element database insterface
    //! [Create MoFEM]

    //! [Example]
    Example ex(m_field);
    CHKERR ex.runProblem();
    //! [Example]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
