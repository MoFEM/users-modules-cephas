/**
 * \file approximation.cpp
 * \FreeSurface approximation.cpp
 *
 * Using PipelineManager interface calculate the divergence of base functions,
 * and integral of flux on the boundary. Since the h-div space is used, volume
 * integral and boundary integral should give the same result.
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

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

constexpr int BASE_DIM = 1;
constexpr int SPACE_DIM = 2;
constexpr int U_FIELD_DIM = SPACE_DIM;
constexpr int H_FIELD_DIM = 1;

template <int DIM> struct ElementsAndOps {};

template <> struct ElementsAndOps<2> {
  using DomainEle = PipelineManager::FaceEle;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::EdgeEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcFaceOnRefinedMesh;
};

template <> struct ElementsAndOps<3> {
  using DomainEle = VolumeElementForcesAndSourcesCore;
  using DomainEleOp = DomainEle::UserDataOperator;
  using BoundaryEle = PipelineManager::FaceEle;
  using BoundaryEleOp = BoundaryEle::UserDataOperator;
  using PostProcEle = PostProcVolumeOnRefinedMesh;
};

using DomainEle = ElementsAndOps<SPACE_DIM>::DomainEle;
using DomainEleOp = DomainEle::UserDataOperator;
using BoundaryEle = ElementsAndOps<SPACE_DIM>::BoundaryEle;
using BoundaryEleOp = BoundaryEle::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;

using PostProcEle = ElementsAndOps<SPACE_DIM>::PostProcEle;
using AssemblyDomainEleOp =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::OpBase;
using AssemblyBoundaryEleOp =
    FormsIntegrators<BoundaryEleOp>::Assembly<PETSC>::OpBase;

using OpDomainMassU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<BASE_DIM, U_FIELD_DIM>;
using OpDomainMassH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpMass<BASE_DIM, H_FIELD_DIM>;

using OpDomainSourceU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, U_FIELD_DIM>;
using OpDomainSourceH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpSource<BASE_DIM, H_FIELD_DIM>;

using OpDomianHDotPhi = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpBaseTimesScalarField<BASE_DIM, 1>;
using OpDomianGradHGradHRhs =
    FormsIntegrators<DomainEleOp>::Assembly<PETSC>::LinearForm<
        GAUSS>::OpGradTimesTensor<BASE_DIM, H_FIELD_DIM, SPACE_DIM, 1>;
using OpDomianGradHGradHLhs = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpGradGrad<BASE_DIM, H_FIELD_DIM, SPACE_DIM>;

using OpConvectivePhi = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::LinearForm<GAUSS>::OpConvectiveTermRhs<1, 1, 2>;
using OpConvectivePhi_dU = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpConvectiveTermLhsDu<1, 1, 2>;
using OpConvectivePhi_dGradH = FormsIntegrators<DomainEleOp>::Assembly<
    PETSC>::BiLinearForm<GAUSS>::OpConvectiveTermLhsDy<1, 1, 2>;

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;

constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();

constexpr double Re_p = 1;
constexpr double Re_m = 1;
constexpr double Ri_p = 1e3;
constexpr double Ri_m = 0;
constexpr double lambda = 0.01;
constexpr double eta = 2.5e-2;
constexpr double K = 1e6;

struct OpNormalConstrainbRhs : public AssemblyBoundaryEleOp {
  OpNormalConstrainbRhs(const std::string field_name,
                        boost::shared_ptr<MatrixDouble> u_ptr)
      : AssemblyBoundaryEleOp(field_name, field_name,
                              AssemblyBoundaryEleOp::OPROW),
        uPtr(u_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data) {
    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_normal = getFTensor1Normal();
    auto t_u = getFTensor1FromMat<SPACE_DIM>(*uPtr);
    auto t_row_base = row_data.getFTensor0N();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      int bb = 0;
      for (; bb != nbRows; ++bb) {
        locF[bb] += t_w * t_row_base * (t_normal(i) * t_u(i));
        ++t_row_base;
      }

      for (; bb < nbRowBaseFunctions; ++bb)
        ++t_row_base;

      ++t_w;
      ++t_u;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
};

struct OpNormalForcebRhs : public AssemblyBoundaryEleOp {
  OpNormalForcebRhs(const std::string field_name,
                    boost::shared_ptr<VectorDouble> lambda_ptr)
      : AssemblyBoundaryEleOp(field_name, field_name,
                              AssemblyDomainEleOp::OPROW),
        lambdaPtr(lambda_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data) {
    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_normal = getFTensor1Normal();
    auto t_lambda = getFTensor0FromVec(*lambdaPtr);
    auto t_row_base = row_data.getFTensor0N();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      auto t_nf = getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(locF);

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        t_nf(i) += t_w * t_row_base * t_normal(i) * t_lambda;
        ++t_row_base;
        ++t_nf;
      }

      for (; bb < nbRowBaseFunctions; ++bb)
        ++t_row_base;

      ++t_w;
      ++t_lambda;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<VectorDouble> lambdaPtr;
};

struct OpNormalConstrainLhs : public AssemblyBoundaryEleOp {

  OpNormalConstrainLhs(const std::string field_name_row,
                        const std::string field_name_col)
      : AssemblyBoundaryEleOp(field_name_row, field_name_col,
                              AssemblyBoundaryEleOp::OPROWCOL) {
    assembleTranspose = true;
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_normal = getFTensor1Normal();
    auto t_row_base = row_data.getFTensor0N();

    for (int gg = 0; gg != nbIntegrationPts; ++gg) {

      auto t_mat =
          getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(&locMat(0, 0));

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        const double a = t_w * t_row_base;

        for (int cc = 0; cc != nbCols / U_FIELD_DIM; ++cc) {
          t_mat(i) += (a * t_col_base) * t_normal(i);
          ++t_col_base;
          ++t_mat;
        }
        ++t_row_base;
      }

      for (; rr < nbRowBaseFunctions; ++rr)
        ++t_row_base;

      ++t_w;
    }

    MoFEMFunctionReturn(0);
  };
};

auto get_D = [](const double A, const double K) {
  FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_D;
  t_D(i, j, k, l) =
      A * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) + K * (t_kd(i, j) * t_kd(k, l));
  return t_D;
};

/**
 * @brief Rhs for U
 *
 */
struct OpRhsU : public AssemblyDomainEleOp {

  OpRhsU(const std::string field_name,
         boost::shared_ptr<MatrixDouble> dot_u_ptr,
         boost::shared_ptr<MatrixDouble> u_ptr,
         boost::shared_ptr<MatrixDouble> grad_u_ptr,
         boost::shared_ptr<VectorDouble> phi_ptr,
         boost::shared_ptr<MatrixDouble> grad_phi_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        dotUPtr(dot_u_ptr), uPtr(u_ptr), gradUPtr(grad_u_ptr), phiPtr(phi_ptr),
        gradPhiPtr(grad_phi_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_dot_u = getFTensor1FromMat<U_FIELD_DIM>(*dotUPtr);
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_phi = getFTensor0FromVec(*phiPtr);
    auto t_grad_phi = getFTensor1FromMat<SPACE_DIM>(*gradPhiPtr);

    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<double>();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha = t_w * vol;

      const double Re = 0.5 * ((1 + t_phi) * Re_p + (1 - t_phi) * Re_m);
      const double tmp_buoyancy =
          (0.5 * ((1 + t_phi) * Ri_p + (1 - t_phi) * Ri_m));
      const double buoyancy = t_phi * tmp_buoyancy;

      auto t_D = get_D(1. / Re, K);

      FTensor::Tensor2_symmetric<double, SPACE_DIM> t_stress;
      t_stress(i, j) = t_D(i, j, k, l) * t_grad_u(k, l);
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_tension;
      t_tension(i, j) = (lambda) * (t_grad_phi(i) * t_grad_phi(j));
      FTensor::Tensor1<double, SPACE_DIM> t_convection;
      t_convection(i) = t_u(j) * t_grad_u(i, j);
      FTensor::Tensor1<double, SPACE_DIM> t_buoyancy;
      t_buoyancy(i) = 0;
      t_buoyancy(SPACE_DIM - 1) = buoyancy;

      auto t_nf = getFTensor1FromArray<U_FIELD_DIM, U_FIELD_DIM>(locF);

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        t_nf(i) +=
            (t_base * alpha) * (t_dot_u(i) + t_convection(i) + t_buoyancy(i));
        t_nf(i) +=
            (t_diff_base(j) * alpha) * (t_stress(i, j) + t_tension(i, j));

        ++t_base;
        ++t_diff_base;
        ++t_nf;
      }

      for (; bb < nbRowBaseFunctions; ++bb) {
        ++t_diff_base;
        ++t_base;
      }

      ++t_dot_u;
      ++t_u;
      ++t_grad_u;
      ++t_phi;
      ++t_grad_phi;

      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> dotUPtr;
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> phiPtr;
  boost::shared_ptr<MatrixDouble> gradPhiPtr;
};

/**
 * @brief Lhs for U dU
 *
 */
struct OpLhsU_dU : public AssemblyDomainEleOp {

  OpLhsU_dU(const std::string field_name, boost::shared_ptr<MatrixDouble> u_ptr,
            boost::shared_ptr<MatrixDouble> grad_u_ptr,
            boost::shared_ptr<VectorDouble> phi_ptr)
      : AssemblyDomainEleOp(field_name, field_name,
                            AssemblyDomainEleOp::OPROWCOL),
        uPtr(u_ptr), gradUPtr(grad_u_ptr), phiPtr(phi_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_phi = getFTensor0FromVec(*phiPtr);

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    auto get_mat = [&](const int rr) {
      return getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(locMat, rr);
    };

    auto ts_a = getTSa();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha = t_w * vol;

      const double Re = 0.5 * ((1 + t_phi) * Re_p + (1 - t_phi) * Re_m);

      auto t_D = get_D(1. / Re, K);

      FTensor::Tensor2<double, U_FIELD_DIM, SPACE_DIM> t_base_lhs;
      t_base_lhs(i, j) = ts_a * t_kd(i, j) + t_grad_u(i, j);

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        auto t_mat = get_mat(bb * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        FTensor::Christof<double, SPACE_DIM, SPACE_DIM> t_rowD;
        // I mix up the indices here so that it behaves like a
        // Dg.  That way I don't have to have a separate wrapper
        // class Christof_Expr, which simplifies things.
        t_rowD(l, j, k) = t_D(i, j, k, l) * (alpha * t_row_diff_base(i));

        for (int cc = 0; cc != nbCols / U_FIELD_DIM; ++cc) {

          t_mat(i, j) += (t_row_base * t_col_base * alpha) * t_base_lhs(i, j);
          t_mat(i, j) += (alpha * t_row_base) * (t_col_diff_base(k) * t_u(k));

          // integrate block local stiffens matrix
          t_mat(i, j) += t_rowD(i, j, k) * t_col_diff_base(k);

          ++t_mat;
          ++t_col_base;
          ++t_col_diff_base;
        }

        ++t_row_base;
        ++t_row_diff_base;
      }

      for (; bb < nbRowBaseFunctions; ++bb) {
        ++t_row_diff_base;
        ++t_row_base;
      }

      ++t_u;
      ++t_grad_u;
      ++t_phi;

      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> phiPtr;
};

/**
 * @brief Lhs for U dU
 *
 */
struct OpLhsU_dH : public AssemblyDomainEleOp {

  OpLhsU_dH(const std::string field_name_u, const std::string field_name_h,
            boost::shared_ptr<MatrixDouble> dot_u_ptr,
            boost::shared_ptr<MatrixDouble> u_ptr,
            boost::shared_ptr<MatrixDouble> grad_u_ptr,
            boost::shared_ptr<VectorDouble> phi_ptr,
            boost::shared_ptr<MatrixDouble> grad_phi_ptr)
      : AssemblyDomainEleOp(field_name_u, field_name_h,
                            AssemblyDomainEleOp::OPROWCOL),
        dotUPtr(dot_u_ptr), uPtr(u_ptr), gradUPtr(grad_u_ptr), phiPtr(phi_ptr),
        gradPhiPtr(grad_phi_ptr) {
    sYmm = false;
  }

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_u = getFTensor1FromMat<U_FIELD_DIM>(*uPtr);
    auto t_grad_u = getFTensor2FromMat<U_FIELD_DIM, SPACE_DIM>(*gradUPtr);
    auto t_phi = getFTensor0FromVec(*phiPtr);
    auto t_grad_phi = getFTensor1FromMat<SPACE_DIM>(*gradPhiPtr);

    auto t_row_base = row_data.getFTensor0N();
    auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

    auto t_w = getFTensor0IntegrationWeight();

    auto get_mat = [&](const int rr) {
      std::array<double *, U_FIELD_DIM> ptrs;
      for (auto i = 0; i != U_FIELD_DIM; ++i)
        ptrs[i] = &locMat(rr + i, 0);
      return FTensor::Tensor1<FTensor::PackPtr<double *, 1>, U_FIELD_DIM>(ptrs);
    };

    auto ts_a = getTSa();
    constexpr auto t_kd = FTensor::Kronecker_Delta<double>();

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha = t_w * vol;

      const double Re = 0.5 * ((1 + t_phi) * Re_p + (1 - t_phi) * Re_m);
      const double d_Re = 0.5 * (Re_p - Re_m);

      const double tmp_b = (0.5 * ((1 + t_phi) * Ri_p + (1 - t_phi) * Ri_m));
      const double d_tmp_b = 0.5 * (Re_p - Re_m);
      const double d_buoyancy = t_phi * d_tmp_b + tmp_b;


      auto t_D_dH = get_D(-(d_Re / (Re * Re)), 0);

      FTensor::Tensor2_symmetric<double, SPACE_DIM> t_stress_dH;
      t_stress_dH(i, j) = t_D_dH(i, j, k, l) * t_grad_u(k, l);

      FTensor::Tensor1<double, SPACE_DIM> t_buoyancy_dH;
      t_buoyancy_dH(i) = 0;
      t_buoyancy_dH(SPACE_DIM - 1) = d_buoyancy;

      int bb = 0;
      for (; bb != nbRows / U_FIELD_DIM; ++bb) {

        auto t_mat = get_mat(bb * U_FIELD_DIM);
        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (int cc = 0; cc != nbCols; ++cc) {

          t_mat(i) += (t_row_base * t_col_base * alpha) * (t_buoyancy_dH(i));
          t_mat(i) +=
              (t_row_diff_base(j) * (alpha * t_col_base)) * t_stress_dH(i, j);
          t_mat(i) += (t_row_diff_base(j) * (alpha * lambda)) *
                      (t_grad_phi(i) * t_col_diff_base(j) +
                       t_col_diff_base(i) * t_grad_phi(j));

          ++t_mat;
          ++t_col_base;
          ++t_col_diff_base;
        }

        ++t_row_base;
        ++t_row_diff_base;
      }

      for (; bb < nbRowBaseFunctions; ++bb) {
        ++t_row_diff_base;
        ++t_row_base;
      }

      ++t_u;
      ++t_grad_u;
      ++t_phi;
      ++t_grad_phi;

      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> dotUPtr;
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradUPtr;
  boost::shared_ptr<VectorDouble> phiPtr;
  boost::shared_ptr<MatrixDouble> gradPhiPtr;
};

/**
 * @brief Explict term for IMEX method
 *
 */
struct OpRhsExplicitTermH : public AssemblyDomainEleOp {

  OpRhsExplicitTermH(const std::string field_name,
                     boost::shared_ptr<VectorDouble> phi_ptr,
                     boost::shared_ptr<double> ksi_ptr)
      : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
        phiPtr(phi_ptr), ksiPtr(ksi_ptr) {}

  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();
    auto t_phi = getFTensor0FromVec(*phiPtr);
    const double ksi = *ksiPtr;

    for (int gg = 0; gg != nbIntegrationPts; gg++) {

      const double alpha = t_w * vol;
      auto nf_ptr = &locF[0];

      const double f = (1 / (eta * eta)) * t_phi * (t_phi * t_phi - 1);

      int rr = 0;
      for (; rr != nbRows; ++rr) {

        (*nf_ptr) += (alpha * lambda) * (-f + ksi) * t_base;

        ++nf_ptr;
        ++t_base;
      }

      ++t_phi;
      ++t_w;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<MatrixDouble> uPtr;
  boost::shared_ptr<MatrixDouble> gradPhiPtr;
  boost::shared_ptr<VectorDouble> phiPtr;
  boost::shared_ptr<double> ksiPtr;
};

struct OpCalculateKsi : public DomainEleOp {

  OpCalculateKsi(const std::string field_name,
                 boost::shared_ptr<VectorDouble> phi_ptr,
                 SmartPetscObj<Vec> ksi_vec)
      : DomainEleOp(field_name, DomainEleOp::OPROW), phiPtr(phi_ptr),
        ksiVec(ksi_vec, true) {
    doEntities[MBVERTEX] = true;
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    const double vol = getMeasure();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = data.getFTensor0N();
    auto t_phi = getFTensor0FromVec(*phiPtr);
    auto nb_gauss_pts = getGaussPts().size2();

    double ksi = 0;
    double vol_ele = 0;
    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      const double alpha = t_w * vol;
      ksi += alpha * t_phi * (t_phi * t_phi - 1);
      vol_ele += alpha;
      ++t_phi;
      ++t_w;
    }

    CHKERR VecSetValue(ksiVec, 0, ksi, ADD_VALUES);
    CHKERR VecSetValue(ksiVec, 1, vol_ele, ADD_VALUES);

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<Vec> ksiVec;
  boost::shared_ptr<VectorDouble> phiPtr;
};

struct FreeSurface {

  FreeSurface(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode runProblem();

private:
  MoFEMErrorCode readMesh();
  MoFEMErrorCode setupProblem();
  MoFEMErrorCode boundaryCondition();
  MoFEMErrorCode assembleSystem();
  MoFEMErrorCode solveSystem();

  MoFEM::Interface &mField;

  boost::shared_ptr<FEMethod> domianLhsFEPtr;
};

//! [Run programme]
MoFEMErrorCode FreeSurface::runProblem() {
  MoFEMFunctionBegin;
  CHKERR readMesh();
  CHKERR setupProblem();
  CHKERR boundaryCondition();
  CHKERR assembleSystem();
  CHKERR solveSystem();
  MoFEMFunctionReturn(0);
}
//! [Run programme]

//! [Read mesh]
MoFEMErrorCode FreeSurface::readMesh() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  CHKERR simple->getOptions();
  CHKERR simple->loadFile();

  MoFEMFunctionReturn(0);
}
//! [Read mesh]

//! [Set up problem]
MoFEMErrorCode FreeSurface::setupProblem() {
  MoFEMFunctionBegin;

  auto simple = mField.getInterface<Simple>();
  CHKERR simple->addDomainField("U", H1, AINSWORTH_LEGENDRE_BASE, U_FIELD_DIM);
  CHKERR simple->addDomainField("H", H1, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple->addBoundaryField("U", H1, AINSWORTH_LEGENDRE_BASE,
                                  U_FIELD_DIM);
  CHKERR simple->addBoundaryField("L", H1, AINSWORTH_LEGENDRE_BASE, 1);

  constexpr int order = 4;
  CHKERR simple->setFieldOrder("U", order);
  CHKERR simple->setFieldOrder("H", order);
  CHKERR simple->setFieldOrder("L", order);
  CHKERR simple->setUp();

  MoFEMFunctionReturn(0);
}
//! [Set up problem]

//! [Boundary condition]
MoFEMErrorCode FreeSurface::boundaryCondition() {
  MoFEMFunctionBegin;

  auto init_h = [](double, double y, double) {
    return tanh((y - 0.5) * (1 / eta));
  };

  auto set_domain_general = [&](auto &pipeline) {
    pipeline.push_back(new OpSetHOWeightsOnFace());
  };

  auto set_domain_rhs = [&](auto &pipeline) {
    pipeline.push_back(new OpDomainSourceH("H", init_h));
  };

  auto set_domain_lhs = [&](auto &pipeline) {
    pipeline.push_back(
        new OpDomainMassU("U", "U", [](double, double, double) { return 1; }));
    pipeline.push_back(
        new OpDomainMassH("H", "H", [](double, double, double) { return 1; }));
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

    post_proc_fe->addFieldValuesPostProc("U");
    post_proc_fe->addFieldValuesPostProc("H");

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

  MoFEMFunctionReturn(0);
}
//! [Boundary condition]

//! [Push operators to pipeline]
MoFEMErrorCode FreeSurface::assembleSystem() {
  MoFEMFunctionBegin;

  auto dot_u_ptr = boost::make_shared<MatrixDouble>();
  auto u_ptr = boost::make_shared<MatrixDouble>();
  auto grad_u_ptr = boost::make_shared<MatrixDouble>();
  auto dot_phi_ptr = boost::make_shared<VectorDouble>();
  auto phi_ptr = boost::make_shared<VectorDouble>();
  auto grad_phi_ptr = boost::make_shared<MatrixDouble>();
  auto lambda_ptr = boost::make_shared<VectorDouble>();

  // Push element from reference configuration to current configuration in 3d
  // space
  auto set_domain_general = [&](auto &pipeline) {
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
    pipeline.push_back(new OpSetHOWeightsOnFace());
    pipeline.push_back(new OpCalculateHOJacForFace(jac_ptr));
    pipeline.push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    pipeline.push_back(new OpSetInvJacH1ForFace(inv_jac_ptr));

    pipeline.push_back(
        new OpCalculateVectorFieldValuesDot<U_FIELD_DIM>("U", dot_u_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    pipeline.push_back(
        new OpCalculateVectorFieldGradient<U_FIELD_DIM, SPACE_DIM>("U",
                                                                   grad_u_ptr));

    pipeline.push_back(new OpCalculateScalarFieldValuesDot("H", dot_phi_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("H", phi_ptr));
    pipeline.push_back(
        new OpCalculateScalarFieldGradient<SPACE_DIM>("H", grad_phi_ptr));
  };

  auto set_domain_rhs = [&](auto &pipeline) {
    pipeline.push_back(
        new OpRhsU("U", dot_u_ptr, u_ptr, grad_u_ptr, phi_ptr, grad_phi_ptr));
    pipeline.push_back(new OpDomianHDotPhi(
        "H", dot_phi_ptr, [](double, double, double) { return 1; }));
    pipeline.push_back(new OpDomianGradHGradHRhs(
        "H", grad_phi_ptr, [&](double, double, double) { return lambda; }));
    pipeline.push_back(new OpConvectivePhi("H", u_ptr, grad_phi_ptr));
  };

  auto set_domain_lhs = [&](auto &pipeline) {
    pipeline.push_back(new OpLhsU_dU("U", u_ptr, grad_u_ptr, phi_ptr));
    pipeline.push_back(new OpLhsU_dH("U", "H", dot_u_ptr, u_ptr, grad_u_ptr,
                                     phi_ptr, grad_phi_ptr));
    pipeline.push_back(new OpDomainMassH("H", "H", [&](double, double, double) {
      return domianLhsFEPtr->ts_a;
    }));
    pipeline.push_back(new OpDomianGradHGradHLhs(
        "H", "H", [&](double, double, double) { return lambda; }));
    pipeline.push_back(
        new OpConvectivePhi_dU("H", "U", grad_phi_ptr, []() { return 1; }));
    pipeline.push_back(
        new OpConvectivePhi_dGradH("H", "H", u_ptr, []() { return 1; }));
  };

  auto set_boundary_rhs = [&](auto &pipeline) {



    pipeline.push_back(
        new OpCalculateVectorFieldValues<U_FIELD_DIM>("U", u_ptr));
    pipeline.push_back(new OpCalculateScalarFieldValues("L", lambda_ptr));
    pipeline.push_back(new OpNormalConstrainbRhs("L", u_ptr));
    pipeline.push_back(new OpNormalForcebRhs("U", lambda_ptr));
  };

  auto set_boundary_lhs = [&](auto &pipeline) {
    pipeline.push_back(new OpNormalConstrainLhs("L", "U"));
  };

  auto *pipeline_mng = mField.getInterface<PipelineManager>();

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order + 1;
  };
  CHKERR pipeline_mng->setDomainRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setDomainLhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryRhsIntegrationRule(integration_rule);
  CHKERR pipeline_mng->setBoundaryLhsIntegrationRule(integration_rule);

  set_domain_general(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_general(pipeline_mng->getOpDomainLhsPipeline());
  set_domain_rhs(pipeline_mng->getOpDomainRhsPipeline());
  set_domain_lhs(pipeline_mng->getOpDomainLhsPipeline());
  set_boundary_rhs(pipeline_mng->getOpBoundaryRhsPipeline());
  set_boundary_lhs(pipeline_mng->getOpBoundaryLhsPipeline());

  domianLhsFEPtr = pipeline_mng->getDomainLhsFE();

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
    constexpr int save_every_nth_step = 10;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc,
                                      this->getCacheWeakPtr());
      CHKERR postProc->writeFile(
          "out_step_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      // MOFEM_LOG("FS", Sev::verbose)
      //     << "writing vector in binary to vector.dat ...";
      // PetscViewer viewer;
      // PetscViewerBinaryOpen(PETSC_COMM_WORLD, "vector.dat", FILE_MODE_WRITE,
      //                       &viewer);
      // VecView(ts_u, viewer);
      // PetscViewerDestroy(&viewer);
    }
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProc;
};

//! [Solve]
MoFEMErrorCode FreeSurface::solveSystem() {
  MoFEMFunctionBegin;

  auto *simple = mField.getInterface<Simple>();
  auto *pipeline_mng = mField.getInterface<PipelineManager>();
  auto dm = simple->getDM();

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

  auto assemble_mass_mat = [&](auto M) {
    MoFEMFunctionBegin;
    auto fe = boost::make_shared<DomainEle>(mField);
    CHKERR MatZeroEntries(M);
    fe->getOpPtrVector().push_back(new OpSetHOWeightsOnFace());
    fe->getOpPtrVector().push_back(
        new OpDomainMassU("U", "U", [&](double, double, double) { return 1; }));
    fe->getOpPtrVector().push_back(
        new OpDomainMassH("H", "H", [&](double, double, double) { return 1; }));
    fe->B = M;
    CHKERR DMoFEMLoopFiniteElements(dm, "dFE", fe);
    CHKERR MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    MoFEMFunctionReturn(0);
  };

  auto set_mass_ksp = [&](auto ksp, auto M) {
    MoFEMFunctionBegin;
    CHKERR KSPSetOperators(ksp, M, M);
    CHKERR KSPSetFromOptions(ksp);
    CHKERR KSPSetUp(ksp);
    MoFEMFunctionReturn(0);
  };

  auto get_fe_post_proc = [&]() {
    auto post_proc_fe = boost::make_shared<PostProcEle>(mField);
    post_proc_fe->generateReferenceElementMesh();
    auto det_ptr = boost::make_shared<VectorDouble>();
    auto jac_ptr = boost::make_shared<MatrixDouble>();
    auto inv_jac_ptr = boost::make_shared<MatrixDouble>();

    post_proc_fe->getOpPtrVector().push_back(
        new OpCalculateHOJacForFace(jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpInvertMatrix<SPACE_DIM>(jac_ptr, det_ptr, inv_jac_ptr));
    post_proc_fe->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(inv_jac_ptr));

    post_proc_fe->addFieldValuesPostProc("U");
    post_proc_fe->addFieldValuesPostProc("H");
    post_proc_fe->addFieldValuesGradientPostProc("U", 2);
    post_proc_fe->addFieldValuesGradientPostProc("H", 2);
    return post_proc_fe;
  };

  auto set_ts = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    CHKERR set_fieldsplit_preconditioner_ksp(ksp);
    MoFEMFunctionReturn(0);
  };

  auto create_ksi_vec = [&]() {
    constexpr int ghost[] = {0, 1};
    return createSmartGhostVector(mField.get_comm(),
                                  (!mField.get_comm_rank()) ? 2 : 0, 2,
                                  (!mField.get_comm_rank()) ? 0 : 2, ghost);
  };

  auto integration_rule = [](int, int, int approx_order) {
    return 2 * approx_order;
  };
  CHKERR pipeline_mng->setDomainExplicitRhsIntegrationRule(integration_rule);
  auto fe_explicit_rhs = pipeline_mng->getDomainExplicitRhsFE();

  auto M = smartCreateDMMatrix(dm);
  auto ksi_vec = create_ksi_vec();
  auto ksi_ptr = boost::make_shared<double>();
  auto ksp = createKSP(mField.get_comm());
  auto phi_ptr = boost::make_shared<VectorDouble>();

  auto fe_global_terms = boost::make_shared<DomainEle>(mField);
  fe_global_terms->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("H", phi_ptr));
  fe_global_terms->getOpPtrVector().push_back(
      new OpCalculateKsi("H", phi_ptr, ksi_vec));

  CHKERR assemble_mass_mat(M);
  CHKERR set_mass_ksp(ksp, M);

  auto caluclate_global_terms = [&]() {
    MoFEMFunctionBegin;
    MOFEM_LOG("FS", Sev::verbose) << "Assemble global terms -> Start";

    CHKERR VecZeroEntries(ksi_vec);
    CHKERR VecGhostUpdateBegin(ksi_vec, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(ksi_vec, INSERT_VALUES, SCATTER_FORWARD);

    if (!fe_explicit_rhs->getCacheWeakPtr().use_count())
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "Weak pointer to cache entities not valid");

    CHKERR DMoFEMLoopFiniteElements(dm, "dFE", fe_global_terms,
                                    fe_explicit_rhs->getCacheWeakPtr());

    CHKERR VecAssemblyBegin(ksi_vec);
    CHKERR VecAssemblyEnd(ksi_vec);
    CHKERR VecGhostUpdateBegin(ksi_vec, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(ksi_vec, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateBegin(ksi_vec, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(ksi_vec, INSERT_VALUES, SCATTER_FORWARD);

    const double *a;
    CHKERR VecGetArrayRead(ksi_vec, &a);
    *ksi_ptr = a[0] / a[1];
    MOFEM_LOG("FS", Sev::inform)
        << "Value of ksi " << *ksi_ptr << " Volume " << a[1];
    CHKERR VecRestoreArrayRead(ksi_vec, &a);

    MOFEM_LOG("FS", Sev::verbose) << "Assemble global terms <- End";
    MoFEMFunctionReturn(0);
  };

  auto solve_explicit_rhs = [&]() {
    MoFEMFunctionBegin;
    MOFEM_LOG("FS", Sev::verbose) << "Solve explicit term -> Start";
    if (fe_explicit_rhs->vecAssembleSwitch) {
      CHKERR VecGhostUpdateBegin(fe_explicit_rhs->ts_F, ADD_VALUES,
                                 SCATTER_REVERSE);
      CHKERR VecGhostUpdateEnd(fe_explicit_rhs->ts_F, ADD_VALUES,
                               SCATTER_REVERSE);
      CHKERR VecAssemblyBegin(fe_explicit_rhs->ts_F);
      CHKERR VecAssemblyEnd(fe_explicit_rhs->ts_F);
      CHKERR KSPSolve(ksp, fe_explicit_rhs->ts_F, fe_explicit_rhs->ts_F);
      *fe_explicit_rhs->vecAssembleSwitch = false;
    }
    MOFEM_LOG("FS", Sev::verbose) << "Solve explicit term <- Done";
    MoFEMFunctionReturn(0);
  };

  auto set_domain_rhs_explicit = [&](auto &pipeline) {
    pipeline.push_back(new OpSetHOWeightsOnFace());
    auto phi_ptr = boost::make_shared<VectorDouble>();
    pipeline.push_back(new OpCalculateScalarFieldValues("H", phi_ptr));
    pipeline.push_back(new OpRhsExplicitTermH("H", phi_ptr, ksi_ptr));
  };

  set_domain_rhs_explicit(pipeline_mng->getOpDomainExplicitRhsPipeline());

  fe_explicit_rhs->preProcessHook = caluclate_global_terms;
  fe_explicit_rhs->postProcessHook = solve_explicit_rhs;

  MoFEM::SmartPetscObj<TS> ts;
  ts = pipeline_mng->createTSIMEX();

  CHKERR TSSetType(ts, TSARKIMEX);
  CHKERR TSARKIMEXSetType(ts, TSARKIMEXA2);

  auto set_post_proc_monitor = [&](auto dm) {
    MoFEMFunctionBegin;
    boost::shared_ptr<FEMethod> null_fe;
    auto monitor_ptr = boost::make_shared<Monitor>(dm, get_fe_post_proc());
    CHKERR DMMoFEMTSSetMonitor(dm, ts, simple->getDomainFEName(), null_fe,
                               null_fe, monitor_ptr);
    MoFEMFunctionReturn(0);
  };
  CHKERR set_post_proc_monitor(dm);

  auto set_section_monitor = [&](auto solver) {
    MoFEMFunctionBegin;
    SNES snes;
    CHKERR TSGetSNES(solver, &snes);
    PetscViewerAndFormat *vf;
    CHKERR PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,
                                      PETSC_VIEWER_DEFAULT, &vf);
    CHKERR SNESMonitorSet(
        snes,
        (MoFEMErrorCode(*)(SNES, PetscInt, PetscReal, void *))SNESMonitorFields,
        vf, (MoFEMErrorCode(*)(void **))PetscViewerAndFormatDestroy);
    MoFEMFunctionReturn(0);
  };

  // Add monitor to time solver
  double ftime = 1;
  // CHKERR TSSetMaxTime(ts, ftime);
  CHKERR TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);

  auto T = smartCreateDMVector(simple->getDM());
  CHKERR DMoFEMMeshToLocalVector(simple->getDM(), T, INSERT_VALUES,
                                 SCATTER_FORWARD);
  CHKERR TSSetSolution(ts, T);
  CHKERR TSSetFromOptions(ts);
  CHKERR set_ts(ts);
  CHKERR set_section_monitor(ts);
  CHKERR TSSetUp(ts);
  CHKERR TSSolve(ts, NULL);
  CHKERR TSGetTime(ts, &ftime);

  MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {

  // Initialisation of MoFEM/PETSc and MOAB data structures
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);

  // Add logging channel for example
  auto core_log = logging::core::get();
  core_log->add_sink(LogManager::createSink(LogManager::getStrmWorld(), "FS"));
  LogManager::setLog("FS");
  MOFEM_LOG_TAG("FS", "free surface");

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

    //! [FreeSurface]
    FreeSurface ex(m_field);
    CHKERR ex.runProblem();
    //! [FreeSurface]
  }
  CATCH_ERRORS;

  CHKERR MoFEM::Core::Finalize();
}
