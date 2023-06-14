/**
 * \file HenkyOps.hpp
 * \example HenkyOps.hpp
 *
 * @copyright Copyright (c) 2023
 */

#ifndef __HENKY_OPS_HPP__
#define __HENKY_OPS_HPP__

namespace HenckyOps {

constexpr double eps = std::numeric_limits<float>::epsilon();

auto f = [](double v) { return 0.5 * std::log(v); };
auto d_f = [](double v) { return 0.5 / v; };
auto dd_f = [](double v) { return -0.5 / (v * v); };

struct isEq {
  static inline auto check(const double &a, const double &b) {
    return std::abs(a - b) / absMax < eps;
  }
  static double absMax;
};

double isEq::absMax = 1;

inline auto is_eq(const double &a, const double &b) {
  return isEq::check(a, b);
};

template <int DIM> inline auto get_uniq_nb(double *ptr) {
  std::array<double, DIM> tmp;
  std::copy(ptr, &ptr[DIM], tmp.begin());
  std::sort(tmp.begin(), tmp.end());
  isEq::absMax = std::max(std::abs(tmp[0]), std::abs(tmp[DIM - 1]));
  return std::distance(tmp.begin(), std::unique(tmp.begin(), tmp.end(), is_eq));
};

template <int DIM>
inline auto sort_eigen_vals(FTensor::Tensor1<double, DIM> &eig,
                            FTensor::Tensor2<double, DIM, DIM> &eigen_vec) {

  isEq::absMax =
      std::max(std::max(std::abs(eig(0)), std::abs(eig(1))), std::abs(eig(2)));

  int i = 0, j = 1, k = 2;

  if (is_eq(eig(0), eig(1))) {
    i = 0;
    j = 2;
    k = 1;
  } else if (is_eq(eig(0), eig(2))) {
    i = 0;
    j = 1;
    k = 2;
  } else if (is_eq(eig(1), eig(2))) {
    i = 1;
    j = 0;
    k = 2;
  }

  FTensor::Tensor2<double, 3, 3> eigen_vec_c{
      eigen_vec(i, 0), eigen_vec(i, 1), eigen_vec(i, 2),

      eigen_vec(j, 0), eigen_vec(j, 1), eigen_vec(j, 2),

      eigen_vec(k, 0), eigen_vec(k, 1), eigen_vec(k, 2)};

  FTensor::Tensor1<double, 3> eig_c{eig(i), eig(j), eig(k)};

  {
    FTensor::Index<'i', DIM> i;
    FTensor::Index<'j', DIM> j;
    eig(i) = eig_c(i);
    eigen_vec(i, j) = eigen_vec_c(i, j);
  }
};

struct CommonData : public boost::enable_shared_from_this<CommonData> {
  boost::shared_ptr<MatrixDouble> matGradPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;
  boost::shared_ptr<MatrixDouble> matLogCPlastic;

  MatrixDouble matEigVal;
  MatrixDouble matEigVec;
  MatrixDouble matLogC;
  MatrixDouble matLogCdC;
  MatrixDouble matFirstPiolaStress;
  MatrixDouble matSecondPiolaStress;
  MatrixDouble matHenckyStress;
  MatrixDouble matTangent;

  inline auto getMatFirstPiolaStress() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(),
                                           &matFirstPiolaStress);
  }

  inline auto getMatHenckyStress() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(),
                                           &matHenckyStress);
  }

  inline auto getMatLogC() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &matLogC);
  }

  inline auto getMatTangent() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &matTangent);
  }
};

template <int DIM> struct OpCalculateEigenVals : public DomainEleOp {

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
      for(auto ii = 0;ii!=DIM;++ii)
        eig(ii) = std::max(std::numeric_limits<double>::epsilon(), eig(ii));

      // rare case when two eigen values are equal
      auto nb_uniq = get_uniq_nb<DIM>(&eig(0));
      if constexpr (DIM == 3) {
        if (nb_uniq == 2) {
          sort_eigen_vals<DIM>(eig, eigen_vec);
        }
      }

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
      ++t_eig_val;
      ++t_eig_vec;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM> struct OpCalculateHenckyStress : public DomainEleOp {

  OpCalculateHenckyStress(const std::string field_name,
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
    constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
    commonDataPtr->matHenckyStress.resize(size_symm, nb_gauss_pts, false);
    auto t_T = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matHenckyStress);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      t_T(i, j) = t_D(i, j, k, l) * t_logC(k, l);
      ++t_logC;
      ++t_T;
      ++t_D;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM> struct OpCalculateHenckyPlasticStress : public DomainEleOp {

  OpCalculateHenckyPlasticStress(const std::string field_name,
                                 boost::shared_ptr<CommonData> common_data,
                                 boost::shared_ptr<MatrixDouble> mat_D_ptr,
                                 const double scale = 1)
      : DomainEleOp(field_name, DomainEleOp::OPROW), commonDataPtr(common_data),
        scaleStress(scale), matDPtr(mat_D_ptr) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);

    matLogCPlastic = commonDataPtr->matLogCPlastic;
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
    auto t_D = getFTensor4DdgFromMat<DIM, DIM, 0>(*matDPtr);
    auto t_logC = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matLogC);
    auto t_logCPlastic = getFTensor2SymmetricFromMat<DIM>(*matLogCPlastic);
    constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
    commonDataPtr->matHenckyStress.resize(size_symm, nb_gauss_pts, false);
    auto t_T = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matHenckyStress);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      t_T(i, j) = t_D(i, j, k, l) * (t_logC(k, l) - t_logCPlastic(k, l));
      t_T(i, j) /= scaleStress;
      ++t_logC;
      ++t_T;
      ++t_D;
      ++t_logCPlastic;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;
  boost::shared_ptr<MatrixDouble> matLogCPlastic;
  const double scaleStress;
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
    commonDataPtr->matFirstPiolaStress.resize(DIM * DIM, nb_gauss_pts, false);
    commonDataPtr->matSecondPiolaStress.resize(size_symm, nb_gauss_pts, false);
    auto t_P = getFTensor2FromMat<DIM, DIM>(commonDataPtr->matFirstPiolaStress);
    auto t_T = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matHenckyStress);
    auto t_S =
        getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matSecondPiolaStress);
    auto t_grad = getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->matGradPtr));

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      FTensor::Tensor2<double, DIM, DIM> t_F;
      t_F(i, j) = t_grad(i, j) + t_kd(i, j);
      t_S(k, l) = t_T(i, j) * t_logC_dC(i, j, k, l);
      t_P(i, l) = t_F(i, k) * t_S(k, l);

      ++t_grad;
      ++t_logC;
      ++t_logC_dC;
      ++t_P;
      ++t_T;
      ++t_S;
      ++t_D;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM> struct OpHenckyTangent : public DomainEleOp {
  OpHenckyTangent(const std::string field_name,
                  boost::shared_ptr<CommonData> common_data,
                  boost::shared_ptr<MatrixDouble> mat_D_ptr = nullptr)
      : DomainEleOp(field_name, DomainEleOp::OPROW),
        commonDataPtr(common_data) {
    std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
    if (mat_D_ptr)
      matDPtr = mat_D_ptr;
    else
      matDPtr = commonDataPtr->matDPtr;
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
    commonDataPtr->matTangent.resize(DIM * DIM * DIM * DIM, nb_gauss_pts);
    auto dP_dF =
        getFTensor4FromMat<DIM, DIM, DIM, DIM, 1>(commonDataPtr->matTangent);

    auto t_D = getFTensor4DdgFromMat<DIM, DIM, 0>(*matDPtr);
    auto t_eig_val = getFTensor1FromMat<DIM>(commonDataPtr->matEigVal);
    auto t_eig_vec = getFTensor2FromMat<DIM, DIM>(commonDataPtr->matEigVec);
    auto t_T = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matHenckyStress);
    auto t_S =
        getFTensor2SymmetricFromMat<DIM>(commonDataPtr->matSecondPiolaStress);
    auto t_grad = getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->matGradPtr));
    auto t_logC_dC = getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->matLogCdC);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      FTensor::Tensor2<double, DIM, DIM> t_F;
      t_F(i, j) = t_grad(i, j) + t_kd(i, j);

      FTensor::Tensor1<double, DIM> eig;
      FTensor::Tensor2<double, DIM, DIM> eigen_vec;
      FTensor::Tensor2_symmetric<double, DIM> T;
      eig(i) = t_eig_val(i);
      eigen_vec(i, j) = t_eig_vec(i, j);
      T(i, j) = t_T(i, j);

      // rare case when two eigen values are equal
      auto nb_uniq = get_uniq_nb<DIM>(&eig(0));

      FTensor::Tensor4<double, DIM, DIM, DIM, DIM> dC_dF;
      dC_dF(i, j, k, l) = (t_kd(i, l) * t_F(k, j)) + (t_kd(j, l) * t_F(k, i));

      auto TL =
          EigenMatrix::getDiffDiffMat(eig, eigen_vec, f, d_f, dd_f, T, nb_uniq);

      TL(i, j, k, l) *= 4;
      FTensor::Ddg<double, DIM, DIM> P_D_P_plus_TL;
      P_D_P_plus_TL(i, j, k, l) =
          TL(i, j, k, l) +
          (t_logC_dC(i, j, o, p) * t_D(o, p, m, n)) * t_logC_dC(m, n, k, l);
      P_D_P_plus_TL(i, j, k, l) *= 0.5;
      dP_dF(i, j, m, n) = t_kd(i, m) * (t_kd(k, n) * t_S(k, j));
      dP_dF(i, j, m, n) +=
          t_F(i, k) * (P_D_P_plus_TL(k, j, o, p) * dC_dF(o, p, m, n));

      ++dP_dF;

      ++t_grad;
      ++t_eig_val;
      ++t_eig_vec;
      ++t_logC_dC;
      ++t_S;
      ++t_T;
      ++t_D;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> matDPtr;
};

template <int DIM> struct OpPostProcHencky : public DomainEleOp {
  OpPostProcHencky(const std::string field_name,
                   moab::Interface &post_proc_mesh,
                   std::vector<EntityHandle> &map_gauss_pts,
                   boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM>
OpPostProcHencky<DIM>::OpPostProcHencky(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
template <int DIM>
MoFEMErrorCode OpPostProcHencky<DIM>::doWork(int side, EntityType type,
                                             EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;

  std::array<double, 9> def;
  std::fill(def.begin(), def.end(), 0);

  auto get_tag = [&](const std::string name, size_t size) {
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);

  auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != SPACE_DIM; ++r)
      for (size_t c = 0; c != SPACE_DIM; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_grad = get_tag("GRAD", 9);
  auto th_stress = get_tag("STRESS", 9);

  auto t_piola =
      getFTensor2FromMat<DIM, DIM>(commonDataPtr->matFirstPiolaStress);
  auto t_grad = getFTensor2FromMat<DIM, DIM>(*commonDataPtr->matGradPtr);

  FTensor::Tensor2<double, DIM, DIM> F;
  FTensor::Tensor2_symmetric<double, DIM> cauchy_stress;

  for (int gg = 0; gg != commonDataPtr->matGradPtr->size2(); ++gg) {

    F(i, j) = t_grad(i, j) + kronecker_delta(i, j);
    double inv_t_detF;

    if (DIM == 2)
      determinantTensor2by2(F, inv_t_detF);
    else
      determinantTensor3by3(F, inv_t_detF);

    inv_t_detF = 1. / inv_t_detF;

    cauchy_stress(i, j) = inv_t_detF * (t_piola(i, k) ^ F(j, k));

    CHKERR set_tag(th_grad, gg, set_matrix(t_grad));
    CHKERR set_tag(th_stress, gg, set_matrix(cauchy_stress));

    ++t_piola;
    ++t_grad;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
addMatBlockOps(MoFEM::Interface &m_field,
               boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip,
               std::string field_name, std::string block_name,
               boost::shared_ptr<MatrixDouble> mat_D_Ptr, Sev sev) {
  MoFEMFunctionBegin;

  struct OpMatBlocks : public DomainEleOp {
    OpMatBlocks(std::string field_name, boost::shared_ptr<MatrixDouble> m,
                double bulk_modulus_K, double shear_modulus_G,
                MoFEM::Interface &m_field, Sev sev,
                std::vector<const CubitMeshSets *> meshset_vec_ptr)
        : DomainEleOp(field_name, DomainEleOp::OPROW), matDPtr(m),
          bulkModulusKDefault(bulk_modulus_K),
          shearModulusGDefault(shear_modulus_G) {
      std::fill(&(doEntities[MBEDGE]), &(doEntities[MBMAXTYPE]), false);
      CHK_THROW_MESSAGE(extractBlockData(m_field, meshset_vec_ptr, sev),
                        "Can not get data from block");
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;

      for (auto &b : blockData) {

        if (b.blockEnts.find(getFEEntityHandle()) != b.blockEnts.end()) {
          CHKERR getMatDPtr(matDPtr, b.bulkModulusK, b.shearModulusG);
          MoFEMFunctionReturnHot(0);
        }
      }

      CHKERR getMatDPtr(matDPtr, bulkModulusKDefault, shearModulusGDefault);
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<MatrixDouble> matDPtr;

    struct BlockData {
      double bulkModulusK;
      double shearModulusG;
      Range blockEnts;
    };

    double bulkModulusKDefault;
    double shearModulusGDefault;
    std::vector<BlockData> blockData;

    MoFEMErrorCode
    extractBlockData(MoFEM::Interface &m_field,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr,
                     Sev sev) {
      MoFEMFunctionBegin;

      for (auto m : meshset_vec_ptr) {
        MOFEM_TAG_AND_LOG("WORLD", sev, "MatBlock") << *m;
        std::vector<double> block_data;
        CHKERR m->getAttributes(block_data);
        if (block_data.size() != 2) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Expected that block has two attribute");
        }
        auto get_block_ents = [&]() {
          Range ents;
          CHKERR
          m_field.get_moab().get_entities_by_handle(m->meshset, ents, true);
          return ents;
        };

        double young_modulus = block_data[0];
        double poisson_ratio = block_data[1];
        double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
        double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

        MOFEM_TAG_AND_LOG("WORLD", sev, "MatBlock")
            << "E = " << young_modulus << " nu = " << poisson_ratio;

        blockData.push_back(
            {bulk_modulus_K, shear_modulus_G, get_block_ents()});
      }
      MOFEM_LOG_CHANNEL("WORLD");
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode getMatDPtr(boost::shared_ptr<MatrixDouble> mat_D_ptr,
                              double bulk_modulus_K, double shear_modulus_G) {
      MoFEMFunctionBegin;
      //! [Calculate elasticity tensor]
      auto set_material_stiffness = [&]() {
        FTensor::Index<'i', SPACE_DIM> i;
        FTensor::Index<'j', SPACE_DIM> j;
        FTensor::Index<'k', SPACE_DIM> k;
        FTensor::Index<'l', SPACE_DIM> l;
        constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
        double A = (SPACE_DIM == 2)
                       ? 2 * shear_modulus_G /
                             (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                       : 1;
        auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mat_D_ptr);
        t_D(i, j, k, l) =
            2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
            A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) * t_kd(i, j) *
                t_kd(k, l);
      };
      //! [Calculate elasticity tensor]
      constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
      mat_D_ptr->resize(size_symm * size_symm, 1);
      set_material_stiffness();
      MoFEMFunctionReturn(0);
    }
  };

  double bulk_modulus_K = young_modulus / (3 * (1 - 2 * poisson_ratio));
  double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));
  pip.push_back(new OpMatBlocks(
      field_name, mat_D_Ptr, bulk_modulus_K, shear_modulus_G, m_field, sev,

      // Get blockset using regular expression
      m_field.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

          (boost::format("%s(.*)") % block_name).str()

              ))

          ));

  MoFEMFunctionReturn(0);
}

template <typename DomainEleOp, AssemblyType A, IntegrationType I>
MoFEMErrorCode opFactoryDomainRhs(
    MoFEM::Interface &m_field,
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip,
    std::string field_name, std::string block_name, Sev sev) {
  MoFEMFunctionBegin;

  using B = typename FormsIntegrators<DomainEleOp>::template Assembly<
      A>::template LinearForm<I>;
  using OpInternalForcePiola =
      typename B::template OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

  auto common_ptr = boost::make_shared<HenckyOps::CommonData>();
  common_ptr->matDPtr = boost::make_shared<MatrixDouble>();
  common_ptr->matGradPtr = boost::make_shared<MatrixDouble>();

  CHKERR addMatBlockOps(m_field, pip, field_name, block_name,
                        common_ptr->matDPtr, sev);

  pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
      "U", common_ptr->matGradPtr));
  pip.push_back(new OpCalculateEigenVals<SPACE_DIM>(field_name, common_ptr));
  pip.push_back(new OpCalculateLogC<SPACE_DIM>(field_name, common_ptr));
  pip.push_back(new OpCalculateLogC_dC<SPACE_DIM>(field_name, common_ptr));
  pip.push_back(new OpCalculateHenckyStress<SPACE_DIM>(field_name, common_ptr));
  pip.push_back(new OpCalculatePiolaStress<SPACE_DIM>(field_name, common_ptr));
  pip.push_back(
      new OpInternalForcePiola("U", common_ptr->getMatFirstPiolaStress()));

  MoFEMFunctionReturn(0);
}

template <typename DomainEleOp, AssemblyType A, IntegrationType I>
MoFEMErrorCode opFactoryDomainLhs(
    MoFEM::Interface &m_field,
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip,
    std::string field_name, std::string block_name, Sev sev) {
  MoFEMFunctionBegin;

  using B = typename FormsIntegrators<DomainEleOp>::template Assembly<
      A>::template BiLinearForm<I>;
  using OpKPiola =
      typename B::template OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;

  auto common_ptr = boost::make_shared<HenckyOps::CommonData>();
  common_ptr->matDPtr = boost::make_shared<MatrixDouble>();
  common_ptr->matGradPtr = boost::make_shared<MatrixDouble>();

  CHKERR addMatBlockOps(m_field, pip, field_name, block_name,
                        common_ptr->matDPtr, sev);

    pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
        field_name, common_ptr->matGradPtr));
    pip.push_back(
        new OpCalculateEigenVals<SPACE_DIM>(field_name, common_ptr));
    pip.push_back(new OpCalculateLogC<SPACE_DIM>(field_name, common_ptr));
    pip.push_back(
        new OpCalculateLogC_dC<SPACE_DIM>(field_name, common_ptr));
    pip.push_back(
        new OpCalculateHenckyStress<SPACE_DIM>(field_name, common_ptr));
    pip.push_back(
        new OpCalculatePiolaStress<SPACE_DIM>(field_name, common_ptr));
    pip.push_back(new OpHenckyTangent<SPACE_DIM>(field_name, common_ptr));
    pip.push_back(
        new OpKPiola(field_name, field_name, common_ptr->getMatTangent()));

  MoFEMFunctionReturn(0);
}

} // namespace HenckyOps

#endif // __HENKY_OPS_HPP__