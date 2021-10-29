/**
 * \file shallow_wave.cpp
 * \example shallow_wave.cpp
 *
 * Solving shallow wave equation on manifold
 *
 * The inital conditions are set following this paper \cite scott2016test.
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
using namespace MoFEM;

#include <boost/math/quadrature/gauss_kronrod.hpp>
using namespace boost::math::quadrature;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = FaceElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

constexpr int FE_DIM = 2;

using EntData = DataForcesAndSourcesCore::EntData;
using DomainEle = ElementsAndOps<FE_DIM>::DomainEle;
using DomainEleOp = ElementsAndOps<FE_DIM>::DomainEleOp;
using PostProcEle = ElementsAndOps<FE_DIM>::PostProcEle;

using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;

// Use forms iterators for Grad-Grad term
using OpBaseDivU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMixScalarTimesDiv<3>;
using OpBaseGradH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMixVectorTimesGrad<1, 3, 3>;

// Use forms for Mass term
using OpMassUU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, 3>;
using OpMassHH = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::BiLinearForm<
    GAUSS>::OpMass<1, 1>;

using OpBaseTimesDotH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<1>;
using OpBaseTimesDivU = OpBaseTimesDotH;

using OpSourceU = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpSource<1, 3>;
using OpSourceH = FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
    GAUSS>::OpSource<1, 1>;

using OpConvectiveH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpConvectiveTermRhs<1, 1, 3>;
using OpConvectiveH_dU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpConvectiveTermLhsDu<1, 1, 3>;
using OpConvectiveH_dGradH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpConvectiveTermLhsDy<1, 1, 3>;

constexpr double omega = 7.292 * 1e-5;
constexpr double g = 9.80616;
constexpr double mu = 1e4;
constexpr double h0 = 1e4;

constexpr double h_hat = 120;
constexpr double u_max = 80;
constexpr double phi_0 = M_PI / 7;
constexpr double phi_1 = M_PI / 2 - phi_0;
constexpr double phi_2 = M_PI / 4;
constexpr double alpha = 1. / 3.;
constexpr double beta = 1. / 15.;

constexpr double penalty = 1;

constexpr bool is_implicit_solver = false;

FTensor::Index<'i', 3> i;
FTensor::Index<'j', 3> j;
FTensor::Index<'l', 3> l;
FTensor::Index<'m', 3> m;

struct OpURhs : public AssemblyDomainEleOp {

  OpURhs(const std::string field_name, boost::shared_ptr<MatrixDouble> u_ptr,
         boost::shared_ptr<MatrixDouble> u_dot_ptr,
         boost::shared_ptr<MatrixDouble> grad_u_ptr,
         boost::shared_ptr<MatrixDouble> grad_h_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        uPtr(u_ptr), uDotPtr(u_dot_ptr), uGradPtr(grad_u_ptr),
        hGradPtr(grad_h_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
    if (!is_implicit_solver) {
      uDotPtr->resize(3, nbIntegrationPts, false);
      uDotPtr->clear();
    }
    auto t_dot_u = getFTensor1FromMat<3>(*uDotPtr);
    auto t_u = getFTensor1FromMat<3>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<3, 3>(*uGradPtr);
    auto t_grad_h = getFTensor1FromMat<3>(*hGradPtr);
    auto t_coords = getFTensor1CoordsAtGaussPts();
    auto t_normal = getFTensor1NormalsAtGaussPts();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha = t_w * vol;
      auto t_nf = getFTensor1FromArray<3, 3>(locF);

      FTensor::Tensor1<double, 3> t_rhs;
      FTensor::Tensor2<double, 3, 3> t_rhs_grad;

      const auto a = sqrt(t_coords(i) * t_coords(i));
      const auto sin_fi = t_coords(2) / a;
      const auto f = 2 * omega * sin_fi;

      FTensor::Tensor1<double, 3> t_r;
      t_r(i) = t_normal(i);
      t_r.normalize();

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      FTensor::Tensor2<double, 3, 3> t_P, t_Q;
      t_P(i, j) = t_r(i) * t_r(j);
      t_Q(i, j) = t_kd(i, j) - t_P(i, j);

      FTensor::Tensor2<double, 3, 3> t_A;
      t_A(i, m) = levi_civita(i, j, m) * t_r(j);

      t_rhs(m) = t_Q(m, i) * (t_dot_u(i) + t_grad_u(i, j) * t_u(j) +
                              f * t_A(i, j) * t_u(j) + g * t_grad_h(i));
      t_rhs_grad(m, j) = t_Q(m, i) * (mu * t_grad_u(i, j));

      t_rhs(m) += t_P(m, j) * t_u(j);

      int rr = 0;
      for (; rr != nbRows / 3; ++rr) {
        t_nf(i) += alpha * t_row_base * t_rhs(i);
        t_nf(i) += alpha * t_row_diff_base(j) * t_rhs_grad(i, j);
        ++t_row_base;
        ++t_row_diff_base;
        ++t_nf;
      }
      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
        ++t_row_diff_base;
      }

      ++t_w;
      ++t_u;
      ++t_dot_u;
      ++t_grad_u;
      ++t_grad_h;
      ++t_coords;
      ++t_normal;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> uDotPtr;
  boost::shared_ptr<MatrixDouble> uGradPtr;
  boost::shared_ptr<MatrixDouble> hGradPtr;
};

struct OpULhs_dU : public AssemblyDomainEleOp {

  OpULhs_dU(const std::string field_name_row, const std::string field_name_col,
            boost::shared_ptr<MatrixDouble> u_ptr,
            boost::shared_ptr<MatrixDouble> grad_u_ptr)
      : AssemblyDomainEleOp(field_name_row, field_name_col,
                            AssemblyDomainEleOp::OPROWCOL),
        uPtr(u_ptr), uGradPtr(grad_u_ptr) {
    this->sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<3>();
    auto t_coords = getFTensor1CoordsAtGaussPts();
    auto t_normal = getFTensor1NormalsAtGaussPts();

    auto t_u = getFTensor1FromMat<3>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<3, 3>(*uGradPtr);

    auto get_t_mat = [&](const int rr) {
      return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>{
          &locMat(rr + 0, 0), &locMat(rr + 0, 1), &locMat(rr + 0, 2),

          &locMat(rr + 1, 0), &locMat(rr + 1, 1), &locMat(rr + 1, 2),

          &locMat(rr + 2, 0), &locMat(rr + 2, 1), &locMat(rr + 2, 2)};
    };

    const auto ts_a = getFEMethod()->ts_a;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const auto a = sqrt(t_coords(i) * t_coords(i));
      const auto sin_fi = t_coords(2) / a;
      const auto f = 2 * omega * sin_fi;

      FTensor::Tensor1<double, 3> t_r;
      t_r(i) = t_normal(i);
      t_r.normalize();

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      FTensor::Tensor2<double, 3, 3> t_P, t_Q;
      t_P(i, j) = t_r(i) * t_r(j);
      t_Q(i, j) = t_kd(i, j) - t_P(i, j);

      FTensor::Tensor2<double, 3, 3> t_A;
      t_A(i, m) = levi_civita(i, j, m) * t_r(j);

      FTensor::Tensor2<double, 3, 3> t_rhs_du;
      t_rhs_du(m, j) =
          t_Q(m, i) * (ts_a * t_kd(i, j) + t_grad_u(i, j) + f * t_A(i, j)) +
          t_P(m, j);

      const double alpha = t_w * vol;

      int rr = 0;
      for (; rr != nbRows / 3; rr++) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);
        auto t_mat = get_t_mat(3 * rr);

        for (int cc = 0; cc != nbCols / 3; cc++) {
          t_mat(i, j) += (alpha * t_row_base * t_col_base) * t_rhs_du(i, j);
          t_mat(i, j) += (alpha * mu) * t_Q(i, j) *
                         (t_row_diff_base(m) * t_col_diff_base(m));
          ++t_col_diff_base;
          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
        ++t_row_diff_base;
      }
      for (; rr < nbRowBaseFunctions; ++rr) {
        ++t_row_base;
        ++t_row_diff_base;
      }

      ++t_w;
      ++t_coords;
      ++t_normal;
      ++t_u;
      ++t_grad_u;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> uGradPtr;
};

struct OpULhs_dH : public AssemblyDomainEleOp {

  OpULhs_dH(const std::string field_name_row, const std::string field_name_col)
      : AssemblyDomainEleOp(field_name_row, field_name_col,
                            AssemblyDomainEleOp::OPROWCOL) {
    this->sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    // get element volume
    const double vol = getMeasure();
    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();
    // get base function gradient on rows
    auto t_row_base = row_data.getFTensor0N();
    // normal
    auto t_normal = getFTensor1NormalsAtGaussPts();

    auto get_t_vec = [&](const int rr) {
      return FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3>{
          &locMat(rr + 0, 0),

          &locMat(rr + 1, 0),

          &locMat(rr + 2, 0)};
    };

    // loop over integration points
    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      FTensor::Tensor1<double, 3> t_r;
      t_r(i) = t_normal(i);
      t_r.normalize();

      constexpr auto t_kd = FTensor::Kronecker_Delta<double>();
      FTensor::Tensor2<double, 3, 3> t_P, t_Q;
      t_P(i, j) = t_r(i) * t_r(j);
      t_Q(i, j) = t_kd(i, j) - t_P(i, j);

      const double alpha = t_w * vol;

      int rr = 0;
      for (; rr != nbRows / 3; rr++) {
        auto t_vec = get_t_vec(3 * rr);
        auto t_col_diff_base = col_data.getFTensor1DiffN<3>(gg, 0);
        const double a = alpha * g * t_row_base;

        for (int cc = 0; cc != nbCols; cc++) {
          t_vec(i) += a * (t_Q(i, m) * t_col_diff_base(m));
          ++t_vec;
          ++t_col_diff_base;
        }

        ++t_row_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_normal;
    }

    MoFEMFunctionReturn(0);
  }
};

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

  boost::shared_ptr<FEMethod> domianLhsFEPtr;
  boost::shared_ptr<FEMethod> domianRhsFEPtr;
};

//! [Create common data]
MoFEMErrorCode Example::createCommonData() {
  MoFEMFunctionBegin;

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
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, 3);
  CHKERR simple->addDomainField("H", H1, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addDataField("HO_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE, 3);

  int order = 2;
  CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("H", order);
  CHKERR simple->setFieldOrder("HO_POSITIONS", 4);

  CHKERR simple->setUp();
  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode Example::boundaryCondition() {
  MoFEMFunctionBegin;
  
  PetscBool is_restart = PETSC_FALSE;
  CHKERR PetscOptionsGetBool(PETSC_NULL, "", "-is_restart", &is_restart,
                             PETSC_NULL);

  auto restart_vector = [&]() {
    MoFEMFunctionBegin;
    auto simple = mField.getInterface<Simple>();
    auto dm = simple->getDM();
    MOFEM_LOG("SW", Sev::inform)
        << "reading vector in binary from vector.dat ...";
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_READ,
                          &viewer);
    auto T = smartCreateDMVector(simple->getDM());
    VecLoad(T, viewer);
    CHKERR DMoFEMMeshToLocalVector(dm, T, INSERT_VALUES, SCATTER_REVERSE);
    MoFEMFunctionReturn(0);
  };

  if (is_restart) {

    CHKERR restart_vector();

  } else {

    const double e_n = exp(-4 / pow(phi_1 - phi_0, 2));
    const double u0 = u_max / e_n;

    FTensor::Tensor1<double, 3> t_k{0., 0., 1.};
    FTensor::Tensor2<double, 3, 3> t_A;
    t_A(i, m) = levi_civita(i, j, m) * t_k(j);

    auto get_phi = [&](const double x, const double y, const double z) {
      FTensor::Tensor1<double, 3> t_r{x, y, 0.};
      const double r = sqrt(t_r(i) * t_r(i));
      return atan2(z, r);
    };

    auto init_u_phi = [&](const double phi) {
      if (phi > phi_0 && phi < phi_1) {
        return u0 * exp(1. / ((phi - phi_0) * (phi - phi_1)));
      } else {
        return 0.;
      }
    };

    auto init_u = [&](const double x, const double y, const double z) {
      FTensor::Tensor1<double, 3> t_u{0., 0., 0.};
      const double u_phi = init_u_phi(get_phi(x, y, z));
      if (u_phi > 0) {
        FTensor::Tensor1<double, 3> t_r{x, y, 0.};
        t_r.normalize();
        t_u(i) = ((t_A(i, j) * t_r(j)) * u_phi);
      }
      return t_u;
    };

    auto init_h = [&](const double x, const double y, const double z) {
      const double a = sqrt(x * x + y * y + z * z);

      auto integral = [&](const double fi) {
        const double u_phi = init_u_phi(fi);
        const auto f = 2 * omega * sin(fi);
        return a * u_phi * (f + (tan(fi) / a) * u_phi);
      };

      auto montain = [&](const double lambda, const double fi) {
        if (lambda > -M_PI && lambda < M_PI)
          return h_hat * cos(fi) * exp(-pow(lambda / alpha, 2)) *
                 exp(-pow((phi_2 - fi) / beta, 2));
        else
          return 0.;
      };

      const double fi = get_phi(x, y, z);
      const double lambda = atan2(x, y);

      double h1 = 0;
      if (fi > phi_0)
        h1 = gauss_kronrod<double, 32>::integrate(
            integral, phi_0, fi, 0, std::numeric_limits<float>::epsilon());

      return h0 + (montain(lambda, fi) - (h1 / g));
    };

    auto set_domain_general = [&](auto &pipeline) {
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
      pipeline.push_back(new OpGetHONormalsOnFace("HO_POSITIONS"));
      pipeline.push_back(new OpCalculateHOCoords("HO_POSITIONS"));
      pipeline.push_back(new OpSetHOWeigthsOnFace());
    };

    auto set_domain_rhs = [&](auto &pipeline) {
      pipeline.push_back(new OpSourceU("U", init_u));
      pipeline.push_back(new OpSourceH("H", init_h));
    };

    auto set_domain_lhs = [&](auto &pipeline) {
      pipeline.push_back(
          new OpMassUU("U", "U", [](double, double, double) { return 1; }));
      pipeline.push_back(
          new OpMassHH("H", "H", [](double, double, double) { return 1; }));
    };

    auto post_proc = [&]() {
      MoFEMFunctionBegin;
      auto simple = mField.getInterface<Simple>();
      auto dm = simple->getDM();

      auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
      post_proc_fe->generateReferenceElementMesh();

      auto det_ptr = boost::make_shared<VectorDouble>();
      auto jac_ptr = boost::make_shared<MatrixDouble>();
      auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

      post_proc_fe->getOpPtrVector().push_back(
          new OpGetHONormalsOnFace("HO_POSITIONS"));
      post_proc_fe->getOpPtrVector().push_back(
          new OpCalculateHOCoords("HO_POSITIONS"));
      post_proc_fe->getOpPtrVector().push_back(
          new OpCalculateHOJacForFaceEmbeddedIn3DSpace(jac_ptr));
      post_proc_fe->getOpPtrVector().push_back(
          new OpInvertMatrix<3>(jac_ptr, det_ptr, inv_jac_ptr));
      post_proc_fe->addFieldValuesPostProc("U");
      post_proc_fe->addFieldValuesPostProc("H");
      post_proc_fe->addFieldValuesPostProc("HO_POSITIONS");

      CHKERR DMoFEMLoopFiniteElements(dm, "dFE", post_proc_fe);
      CHKERR post_proc_fe->writeFile("out_init.h5m");

      MoFEMFunctionReturn(0);
    };

    auto solve_init = [&]() {
      MoFEMFunctionBegin;
      auto simple = mField.getInterface<Simple>();
      auto pipeline_mng = mField.getInterface<PipelineManager>();

      auto solver = pipeline_mng->createKSP();
      CHKERR KSPSetFromOptions(solver);
      PC pc;
      CHKERR KSPGetPC(solver, &pc);
      PetscBool is_pcfs = PETSC_FALSE;
      PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);
      if (is_pcfs == PETSC_TRUE) {
        auto bc_mng = mField.getInterface<BcManager>();
        auto name_prb = simple->getProblemName();
        SmartPetscObj<IS> is_u;
        CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
            name_prb, ROW, "U", 0, 3, is_u);
        CHKERR PCFieldSplitSetIS(pc, PETSC_NULL, is_u);
        CHKERR PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE);
      }

      CHKERR KSPSetUp(solver);

      auto dm = simple->getDM();
      auto D = smartCreateDMVector(dm);
      auto F = smartVectorDuplicate(D);

      CHKERR KSPSolve(solver, F, D);
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

      MoFEMFunctionReturn(0);
    };

    auto pipeline_mng = mField.getInterface<PipelineManager>();

    auto integration_rule = [](int, int, int approx_order) {
      return 2 * approx_order + 4;
    };
    CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
    CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

    set_domain_general(pipeline_mng->getOpDomainRhsPipeline());
    set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline());
    set_domain_general(pipeline_mng->getOpDomainLhsPipeline());
    set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline());

    CHKERR solve_init();
    CHKERR post_proc();

    // Clear pipelines
    pipeline_mng->getOpDomainRhsPipeline().clear();
    pipeline_mng->getOpDomainLhsPipeline().clear();
  }

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode Example::assembleSystem() {
  MoFEMFunctionBegin;

  // Push element from reference configuration to current configuration in 3d
  // space
  auto set_domain_general = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpGetHONormalsOnFace("HO_POSITIONS"));
    pipeline.push_back(new OpCalculateHOCoords("HO_POSITIONS"));
    pipeline.push_back(new OpSetHOWeigthsOnFace());
    pipeline.push_back(new OpCalculateHOJacForFaceEmbeddedIn3DSpace(jac_ptr));
    pipeline.push_back(new OpInvertMatrix<3>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacH1ForFaceEmbeddedIn3DSpace(inv_jac_ptr));
  };

  auto set_domain_rhs = [&](auto &pipeline) {
    auto dot_u_ptr = boost::make_shared<MatrixDouble>();
    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto grad_u_ptr = boost::make_shared<MatrixDouble>();
    auto div_u_ptr = boost::make_shared<VectorDouble>();
    auto dot_h_ptr = boost::make_shared<VectorDouble>();
    auto grad_h_ptr = boost::make_shared<MatrixDouble>();

    if (is_implicit_solver) {
      pipeline.push_back(
          new OpCalculateVectorFieldValuesDot<3>("U", dot_u_ptr));
      pipeline.push_back(new OpCalculateScalarFieldValuesDot("H", dot_h_ptr));
    }

    pipeline.push_back(new OpCalculateVectorFieldValues<3>("U", u_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldGradient<3, 3>("U", grad_u_ptr));
    pipeline.push_back(
        new OpCalculateDivergenceVectorFieldValues<3>("U", div_u_ptr));
    pipeline.push_back(new OpCalculateScalarFieldGradient<3>("H", grad_h_ptr));

    if (is_implicit_solver)
      pipeline.push_back(new OpBaseTimesDotH(
          "H", dot_h_ptr, [](double, double, double) { return 1.; }));
    pipeline.push_back(new OpBaseTimesDivU(
        "H", div_u_ptr, [](double, double, double) { return h0; }));
    pipeline.push_back(new OpConvectiveH("H", u_ptr, grad_h_ptr));
    pipeline.push_back(
        new OpURhs("U", u_ptr, dot_u_ptr, grad_u_ptr, grad_h_ptr));
  };

  auto set_domain_lhs = [&](auto &pipeline) {
    auto u_ptr = boost::make_shared<MatrixDouble>();
    auto grad_u_ptr = boost::make_shared<MatrixDouble>();
    auto grad_h_ptr = boost::make_shared<MatrixDouble>();

    pipeline.push_back(new OpCalculateVectorFieldValues<3>("U", u_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldGradient<3, 3>("U", grad_u_ptr));
    pipeline.push_back(new OpCalculateScalarFieldGradient<3>("H", grad_h_ptr));

    pipeline.push_back(new OpMassHH("H", "H", [&](double, double, double) {
      return domianLhsFEPtr->ts_a;
    }));
    pipeline.push_back(new OpBaseDivU(
        "H", "U", []() { return h0; }, false, false));
    pipeline.push_back(
        new OpConvectiveH_dU("H", "U", grad_h_ptr, []() { return 1; }));
    pipeline.push_back(
        new OpConvectiveH_dGradH("H", "H", u_ptr, []() { return 1; }));
    pipeline.push_back(new OpULhs_dU("U", "U", u_ptr, grad_u_ptr));
    pipeline.push_back(new OpULhs_dH("U", "H"));
  };

  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order + 4;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);

  set_domain_general(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_general(pipeline_mng->getOpDomainLhsPipeline());

  set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline());

  domianLhsFEPtr = pipeline_mng->getDomainLhsFE();
  domianRhsFEPtr = pipeline_mng->getDomainRhsFE();

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
    constexpr int save_every_nth_step = 100;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
      CHKERR postProc->writeFile(
          "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      MOFEM_LOG("SW", Sev::verbose)
          << "writing vector in binary to vector.dat ...";
      PetscViewer viewer;
      PetscViewerBinaryOpen(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_WRITE,
                            &viewer);
      VecView(ts_u, viewer);
      PetscViewerDestroy(&viewer);
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

  auto set_initial_step = [&](auto ts) {
    MoFEMFunctionBegin;
    int step = 0;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-step", &step,
                              PETSC_NULL);
    CHKERR TSSetStepNumber(ts, step);
    MoFEMFunctionReturn(0);
  };

  auto assemble_mass_mat = [&](auto M) {
    MoFEMFunctionBegin;
    auto fe = boost::make_shared<DomainEle>(mField);
    CHKERR MatZeroEntries(M);
    fe->getOpPtrVector().push_back(new OpGetHONormalsOnFace("HO_POSITIONS"));
    fe->getOpPtrVector().push_back(new OpCalculateHOCoords("HO_POSITIONS"));
    fe->getOpPtrVector().push_back(new OpSetHOWeigthsOnFace());
    fe->getOpPtrVector().push_back(
        new OpMassUU("U", "U", [&](double, double, double) { return 1; }));
    fe->getOpPtrVector().push_back(
        new OpMassHH("H", "H", [&](double, double, double) { return 1; }));
    fe->B = M;
    CHKERR DMoFEMLoopFiniteElements(dm, "dFE", fe);
    CHKERR MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    MoFEMFunctionReturn(0);
  };

  auto set_fieldsplit_preconditioner_ksp = [&](auto ksp) {
    MoFEMFunctionBeginHot;
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    PetscBool is_pcfs = PETSC_FALSE;
    PetscObjectTypeCompare((PetscObject)pc, PCFIELDSPLIT, &is_pcfs);
    if (is_pcfs == PETSC_TRUE) {
      auto bc_mng = mField.getInterface<BcManager>();
      auto name_prb = simple->getProblemName();
      SmartPetscObj<IS> is_u;
      CHKERR mField.getInterface<ISManager>()->isCreateProblemFieldAndRank(
          name_prb, ROW, "U", 0, 3, is_u);
      CHKERR PCFieldSplitSetIS(pc, PETSC_NULL, is_u);
    }
    MoFEMFunctionReturnHot(0);
  };

  auto set_mass_ksp = [&](auto ksp, auto M) {
    MoFEMFunctionBegin;
    CHKERR KSPSetOperators(ksp, M, M);
    CHKERR KSPSetFromOptions(ksp);
    CHKERR set_fieldsplit_preconditioner_ksp(ksp);
    CHKERR KSPSetUp(ksp);
    MoFEMFunctionReturn(0);
  };

  // Setup postprocessing
  auto get_fe_post_proc = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    post_proc_fe->generateReferenceElementMesh();

    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

    post_proc_fe->getOpPtrVector().push_back(
        new OpGetHONormalsOnFace("HO_POSITIONS"));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOCoords("HO_POSITIONS"));
    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOJacForFaceEmbeddedIn3DSpace(jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<3>(jac_ptr, det_ptr, inv_jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFaceEmbeddedIn3DSpace(inv_jac_ptr));
    post_proc_fe->addFieldValuesPostProc("U");
    post_proc_fe->addFieldValuesPostProc("H");
    post_proc_fe->addFieldValuesGradientPostProc("U");
    post_proc_fe->addFieldValuesGradientPostProc("H");
    post_proc_fe->addFieldValuesPostProc("HO_POSITIONS");
    return post_proc_fe;
  };

  if (is_implicit_solver) {

    auto set_fieldsplit_preconditioner_ts = [&](auto solver) {
      MoFEMFunctionBeginHot;
      SNES snes;
      CHKERR TSGetSNES(solver, &snes);
      KSP ksp;
      CHKERR SNESGetKSP(snes, &ksp);
      CHKERR set_fieldsplit_preconditioner_ksp(ksp);
      MoFEMFunctionReturnHot(0);
    };

    MoFEM::SmartPetscObj<TS> ts;
    ts = pipeline_mng->createTSIM();

    boost::shared_ptr<FEMethod> null_fe;
    auto monitor_ptr = boost::make_shared<Monitor>(dm, get_fe_post_proc());
    CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                               null_fe, monitor_ptr);

    // Add monitor to time solver
    double ftime = 1;
    // CHKERR TSSetMaxTime(ts, ftime);
    CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

    auto T = smartCreateDMVector(simple->getDM());
    CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                   SCATTER_FORWARD);
    CHKERR TSSetSolution(ts, T);
    CHKERR TSSetFromOptions(ts);
    CHKERR set_fieldsplit_preconditioner_ts(ts);
    CHKERR TSSetUp(ts);
    CHKERR set_initial_step(ts);
    CHKERR TSSolve(ts, NULL);
    CHKERR TSGetTime(ts, &ftime);
  } else {

    auto M = smartCreateDMMatrix(dm);
    auto ksp = createKSP(mField.get_comm());

    CHKERR assemble_mass_mat(M);
    CHKERR set_mass_ksp(ksp, M);

    auto solve_rhs = [&]() {

      MoFEMFunctionBegin;
      if (domianRhsFEPtr->vecAssembleSwitch) {
        CHKERR VecGhostUpdateBegin(domianRhsFEPtr->ts_F, ADD_VALUES,
                                   SCATTER_REVERSE);
        CHKERR VecGhostUpdateEnd(domianRhsFEPtr->ts_F, ADD_VALUES,
                                 SCATTER_REVERSE);
        CHKERR VecAssemblyBegin(domianRhsFEPtr->ts_F);
        CHKERR VecAssemblyEnd(domianRhsFEPtr->ts_F);
        CHKERR KSPSolve(ksp, domianRhsFEPtr->ts_F, domianRhsFEPtr->ts_F);
        CHKERR VecScale(domianRhsFEPtr->ts_F, -1);
        *domianRhsFEPtr->vecAssembleSwitch = false;
      }
      MoFEMFunctionReturn(0);
    };

    domianRhsFEPtr->postProcessHook = solve_rhs;

    MoFEM::SmartPetscObj<TS> ts;
    ts = pipeline_mng->createTSEX();

    boost::shared_ptr<FEMethod> null_fe;
    auto monitor_ptr = boost::make_shared<Monitor>(dm, get_fe_post_proc());
    CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                               null_fe, monitor_ptr);

    // Add monitor to time solver
    double ftime = 1;
    // CHKERR TSSetMaxTime(ts, ftime);
    CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

    auto T = smartCreateDMVector(simple->getDM());
    CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                   SCATTER_FORWARD);

    CHKERR TSSetSolution(ts, T);
    CHKERR TSSetFromOptions(ts);
    CHKERR set_initial_step(ts);
    CHKERR TSSetUp(ts);

    CHKERR TSSolve(ts, NULL);
    CHKERR TSGetTime(ts, &ftime);
  }

  MoFEMFunctionReturn(0);
}
//! [Solve]

//! [Postprocess results]
MoFEMErrorCode Example::outputResults() {
  MoFEMFunctionBegin;
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
  core_log->add_sink(LogManager::createSink(LogManager::getStrmWorld(), "SW"));
  LogManager::setLog("SW");
  MOFEM_LOG_TAG("SW", "example");

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
