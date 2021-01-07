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

namespace HenkyOps {

constexpr double eps = 1e-8;

auto f = [](double v) { return 0.5 * log(v); };
auto d_f = [](double v) { return 0.5 / v; };
auto dd_f = [](double v) { return -0.5 / (v * v); };

inline auto is_eq(const double &a, const double &b) {
  return std::abs(a - b) < eps;
};

template <int DIM> inline auto get_uniq_nb(double *ptr) {
  return std::distance(ptr, unique(ptr, ptr + DIM, is_eq));
};

template <int DIM>
inline auto sort_eigen_vals(FTensor::Tensor1<double, DIM> &eig,
                            FTensor::Tensor2<double, DIM, DIM> &eigen_vec) {
  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  if (is_eq(eig(0), eig(1))) {
    FTensor::Tensor2<double, 3, 3> eigen_vec_c{
        eigen_vec(0, 0), eigen_vec(0, 1), eigen_vec(0, 2),
        eigen_vec(2, 0), eigen_vec(2, 1), eigen_vec(2, 2),
        eigen_vec(1, 0), eigen_vec(1, 1), eigen_vec(1, 2)};
    FTensor::Tensor1<double, 3> eig_c{eig(0), eig(2), eig(1)};
    eig(i) = eig_c(i);
    eigen_vec(i, j) = eigen_vec_c(i, j);
  } else {
    FTensor::Tensor2<double, 3, 3> eigen_vec_c{
        eigen_vec(1, 0), eigen_vec(1, 1), eigen_vec(1, 2),
        eigen_vec(0, 0), eigen_vec(0, 1), eigen_vec(0, 2),
        eigen_vec(2, 0), eigen_vec(2, 1), eigen_vec(2, 2)};
    FTensor::Tensor1<double, 3> eig_c{eig(1), eig(0), eig(2)};
    eig(i) = eig_c(i);
    eigen_vec(i, j) = eigen_vec_c(i, j);
  }
};

struct CommonData : public boost::enable_shared_from_this<CommonData> {
  boost::shared_ptr<MatrixDouble> matGradPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;

  MatrixDouble matEigVal;
  MatrixDouble matEigVec;
  MatrixDouble matLogC;
  MatrixDouble matLogCdC;
  MatrixDouble matPiolaStress;
  MatrixDouble matHenkyStress;

  inline auto getMatPiolaStress() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &matPiolaStress);
  }
};

template <int DIM>
struct OpCalculateEigenVals : public DomainEleOp {

  OpCalculateEigenVals(const std::string field_name,
                       boost::shared_ptr<CommonData> common_data)
      : DomainEleOp(field_name, DomainEleOp::OPROW),
        commonDataPtr(common_data) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    FTensor::Index<'i', DIM> i;
    FTensor::Index<'j', DIM> j;
    FTensor::Index<'k', DIM> k;

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    // const size_t nb_gauss_pts = matGradPtr->size2();
    const size_t nb_gauss_pts = getGaussPts().size2();

    auto t_grad = getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->matGradPtr));

    commonDataPtr->matEigVal.resize(DIM, nb_gauss_pts, false);
    commonDataPtr->matEigVec.resize(DIM * DIM, nb_gauss_pts, false);
    auto t_eig_val = getFTensor1FromMat<DIM>(commonDataPtr->matEigVal);
    auto t_eig_vec = getFTensor2FromMat<DIM, DIM>(commonDataPtr->matEigVec);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      FTensor::Tensor2<double, DIM, DIM> F;
      FTensor::Tensor2_symmetric<double, DIM> C;
      FTensor::Tensor1<double, DIM> eig;
      FTensor::Tensor2<double, DIM, DIM> eigen_vec;

      F(i, j) = t_grad(i, j) + t_kd(i, j);
      C(i, j) = F(k, i) ^ F(k, j);

      for (int ii = 0; ii != DIM; ii++)
        for (int jj = 0; jj != DIM; jj++)
          eigen_vec(ii, jj) = C(ii, jj);

      CHKERR computeEigenValuesSymmetric<DIM>(eigen_vec, eig);

      // rare case when two eigen values are equal
      auto nb_uniq = get_uniq_nb<DIM>(&eig(0));
      if (DIM == 3 && nb_uniq == 2)
        sort_eigen_vals<DIM>(eig, eigen_vec);

      t_eig_val(i) = eig(i);
      t_eig_vec(i, j) = eigen_vec(i, j);

      ++t_grad;
      ++t_eig_val;
      ++t_eig_vec;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM> struct OpCalculateLogC : public DomainEleOp {

  OpCalculateLogC(const std::string field_name,
                  boost::shared_ptr<CommonData> common_data)
      : DomainEleOp(field_name, DomainEleOp::OPROW),
        commonDataPtr(common_data) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    FTensor::Index<'i', DIM> i;
    FTensor::Index<'j', DIM> j;

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    // const size_t nb_gauss_pts = matGradPtr->size2();
    const size_t nb_gauss_pts = getGaussPts().size2();
    constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
    commonDataPtr->matLogC.resize(size_symm, nb_gauss_pts, false);

    auto t_eig_val = getFTensor1FromMat<DIM>(commonDataPtr->matEigVal);
    auto t_eig_vec = getFTensor2FromMat<DIM, DIM>(commonDataPtr->matEigVec);

    auto c = getFTensor2FromMat<DIM, DIM>(commonDataPtr->matEigVec);
    auto t_logC = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matLogC);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      // rare case when two eigen values are equal
      auto nb_uniq = get_uniq_nb<DIM>(&(t_eig_val(0)));

      FTensor::Tensor1<double, DIM> eig;
      FTensor::Tensor2<double, DIM, DIM> eigen_vec;
      eig(i) = t_eig_val(i);
      eigen_vec(i, j) = t_eig_vec(i, j);
      auto logC = EigenMatrix::getMat(eig, eigen_vec, f);
      t_logC(i, j) = logC(i, j);

      ++t_eig_val;
      ++t_eig_vec;
      ++t_logC;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM> struct OpCalculateLogC_dC : public DomainEleOp {

  OpCalculateLogC_dC(const std::string field_name,
                     boost::shared_ptr<CommonData> common_data)
      : DomainEleOp(field_name, DomainEleOp::OPROW), commonDataPtr(common_data) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    FTensor::Index<'i', DIM> i;
    FTensor::Index<'j', DIM> j;
    FTensor::Index<'k', DIM> k;
    FTensor::Index<'l', DIM> l;

    // const size_t nb_gauss_pts = matGradPtr->size2();
    const size_t nb_gauss_pts = getGaussPts().size2();
    constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
    commonDataPtr->matLogCdC.resize(size_symm * size_symm, nb_gauss_pts, false);
    auto t_logC_dC = getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->matLogCdC);
    auto t_eig_val = getFTensor1FromMat<DIM>(commonDataPtr->matEigVal);
    auto t_eig_vec = getFTensor2FromMat<DIM, DIM>(commonDataPtr->matEigVec);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      FTensor::Tensor1<double, DIM> eig;
      FTensor::Tensor2<double, DIM, DIM> eigen_vec;
      eig(i) = t_eig_val(i);
      eigen_vec(i, j) = t_eig_vec(i, j);

      // rare case when two eigen values are equal
      auto nb_uniq = get_uniq_nb<DIM>(&eig(0));
      auto dlogC_dC = EigenMatrix::getDiffMat(eig, eigen_vec, f, d_f, nb_uniq);
      dlogC_dC(i, j, k, l) *= 2;
      
      t_logC_dC(i, j, k, l) = dlogC_dC(i, j, k, l);

      ++t_logC_dC;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM> struct OpCalculatePiolaStress : public DomainEleOp {

  OpCalculatePiolaStress(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data)
      : DomainEleOp(field_name, DomainEleOp::OPROW),
        commonDataPtr(common_data) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    FTensor::Index<'i', DIM> i;
    FTensor::Index<'j', DIM> j;
    FTensor::Index<'k', DIM> k;
    FTensor::Index<'l', DIM> l;

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

    // const size_t nb_gauss_pts = matGradPtr->size2();
    const size_t nb_gauss_pts = getGaussPts().size2();
    auto t_D = getFTensor4DdgFromMat<DIM, DIM, 0>(*commonDataPtr->matDPtr);
    auto t_logC = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matLogC);
    auto t_logC_dC = getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->matLogCdC);
    constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
    commonDataPtr->matPiolaStress.resize(DIM * DIM, nb_gauss_pts, false);
    commonDataPtr->matHenkyStress.resize(size_symm, nb_gauss_pts, false);
    auto t_P = getFTensor2FromMat<DIM, DIM>(commonDataPtr->matPiolaStress);
    auto t_T = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matHenkyStress);
    auto t_grad = getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->matGradPtr));


    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      FTensor::Tensor2<double, DIM, DIM> t_F;
      FTensor::Tensor2_symmetric<double, DIM> t_S;

      t_F(i, j) = t_grad(i, j) + t_kd(i, j);

      t_T(i, j) = t_D(i, j, k, l) * t_logC(k, l);
      t_S(k, l) = t_T(i, j) * t_logC_dC(i, j, k, l);
      t_P(i, l) = t_F(i, k) * t_S(k, l);

      ++t_grad;
      ++t_logC;
      ++t_logC_dC;
      ++t_P;
      ++t_T;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM, bool IS_LHS>
struct OpHenkyStressAndTangent : public DomainEleOp {
  OpHenkyStressAndTangent(const std::string field_name,
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

      for (int ii = 0; ii != DIM; ii++)
        for (int jj = 0; jj != DIM; jj++)
          eigen_vec(ii, jj) = C(ii, jj);

      CHKERR computeEigenValuesSymmetric<DIM>(eigen_vec, eig);

      // rare case when two eigen values are equal
      auto nb_uniq = get_uniq_nb<DIM>(&eig(0));
      if (DIM == 3 && nb_uniq == 2)
        sort_eigen_vals<DIM>(eig, eigen_vec);
      auto logC = EigenMatrix::getMat(eig, eigen_vec, f);

      T(i, j) = t_D(i, j, k, l) * logC(k, l);

      auto dlogC_dC = EigenMatrix::getDiffMat(eig, eigen_vec, f, d_f, nb_uniq);
      dlogC_dC(i, j, k, l) *= 2;

      S(k, l) = T(i, j) * dlogC_dC(i, j, k, l);
      P(i, l) = F(i, k) * S(k, l);

      if (IS_LHS) {
        FTensor::Tensor4<double, DIM, DIM, DIM, DIM> dC_dF;
        dC_dF(i, j, k, l) = (t_kd(i, l) * F(k, j)) + (t_kd(j, l) * F(k, i));

        auto TL = EigenMatrix::getDiffDiffMat(eig, eigen_vec, f, d_f, dd_f, T,
                                              nb_uniq);

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

} // namespace HenkyOps