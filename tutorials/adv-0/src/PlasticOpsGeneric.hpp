

/** \file PlasticOpsGeneric.hpp
 * \example PlasticOpsGeneric.hpp
 */

namespace PlasticOps {

//! [Lambda functions]
template <int DIM> inline auto diff_tensor(FTensor::Number<DIM>) {
  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  FTensor::Ddg<double, DIM, DIM> t_diff;
  constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
  t_diff(i, j, k, l) = (t_kd(i, k) ^ t_kd(j, l)) / 4.;
  return t_diff;
};

template <int DIM> inline auto symm_L_tensor(FTensor::Number<DIM>) {
  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
  FTensor::Index<'L', size_symm> L;
  FTensor::Dg<double, DIM, size_symm> t_L;
  t_L(i, j, L) = 0;
  if constexpr (DIM == 2) {
    t_L(0, 0, 0) = 1;
    t_L(1, 0, 1) = 1;
    t_L(1, 1, 2) = 1;
  } else {
    t_L(0, 0, 0) = 1;
    t_L(1, 0, 1) = 1;
    t_L(2, 0, 2) = 1;
    t_L(1, 1, 3) = 1;
    t_L(2, 1, 4) = 1;
    t_L(2, 2, 5) = 1;
  }
  return t_L;
}

template <int DIM> inline auto diff_symmetrize(FTensor::Number<DIM>) {

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;

  FTensor::Tensor4<double, DIM, DIM, DIM, DIM> t_diff;

  t_diff(i, j, k, l) = 0;
  t_diff(0, 0, 0, 0) = 1;
  t_diff(1, 1, 1, 1) = 1;

  t_diff(1, 0, 1, 0) = 0.5;
  t_diff(1, 0, 0, 1) = 0.5;

  t_diff(0, 1, 0, 1) = 0.5;
  t_diff(0, 1, 1, 0) = 0.5;

  if constexpr (DIM == 3) {
    t_diff(2, 2, 2, 2) = 1;

    t_diff(2, 0, 2, 0) = 0.5;
    t_diff(2, 0, 0, 2) = 0.5;
    t_diff(0, 2, 0, 2) = 0.5;
    t_diff(0, 2, 2, 0) = 0.5;

    t_diff(2, 1, 2, 1) = 0.5;
    t_diff(2, 1, 1, 2) = 0.5;
    t_diff(1, 2, 1, 2) = 0.5;
    t_diff(1, 2, 2, 1) = 0.5;
  }

  return t_diff;
};

template <typename T>
inline double trace(FTensor::Tensor2_symmetric<T, 2> &t_stress) {
  constexpr double third = boost::math::constants::third<double>();
  return (t_stress(0, 0) + t_stress(1, 1)) * third;
};

template <typename T>
inline double trace(FTensor::Tensor2_symmetric<T, 3> &t_stress) {
  constexpr double third = boost::math::constants::third<double>();
  return (t_stress(0, 0) + t_stress(1, 1) + t_stress(2, 2)) * third;
};

template <typename T, int DIM>
inline auto deviator(

    FTensor::Tensor2_symmetric<T, DIM> &t_stress, double trace,
    FTensor::Tensor2_symmetric<double, DIM> &t_alpha, FTensor::Number<DIM>

) {
  FTensor::Tensor2_symmetric<double, 3> t_dev;
  t_dev(I, J) = 0;
  for (int ii = 0; ii != DIM; ++ii)
    for (int jj = ii; jj != DIM; ++jj)
      t_dev(ii, jj) = t_stress(ii, jj);
  t_dev(0, 0) -= trace;
  t_dev(1, 1) -= trace;
  t_dev(2, 2) -= trace;
  for (int ii = 0; ii != DIM; ++ii)
    for (int jj = ii; jj != DIM; ++jj)
      t_dev(ii, jj) -= t_alpha(ii, jj);
  return t_dev;
};

template <typename T>
inline auto deviator(FTensor::Tensor2_symmetric<T, 2> &t_stress, double trace,
                     FTensor::Tensor2_symmetric<double, 2> &&t_alpha) {
  return deviator(t_stress, trace, t_alpha, FTensor::Number<2>());
};

template <typename T>
inline auto deviator(FTensor::Tensor2_symmetric<T, 3> &t_stress, double trace,
                     FTensor::Tensor2_symmetric<double, 3> &&t_alpha) {
  return deviator(t_stress, trace, t_alpha, FTensor::Number<3>());
};

template <int DIM>
inline auto diff_deviator(FTensor::Ddg<double, DIM, DIM> &&t_diff_stress,
                          FTensor::Number<DIM>) {
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  FTensor::Ddg<double, 3, DIM> t_diff_deviator;
  t_diff_deviator(I, J, k, l) = 0;
  for (int ii = 0; ii != DIM; ++ii)
    for (int jj = ii; jj != DIM; ++jj)
      for (int kk = 0; kk != DIM; ++kk)
        for (int ll = kk; ll != DIM; ++ll)
          t_diff_deviator(ii, jj, kk, ll) = t_diff_stress(ii, jj, kk, ll);

  constexpr double third = boost::math::constants::third<double>();

  t_diff_deviator(0, 0, 0, 0) -= third;
  t_diff_deviator(0, 0, 1, 1) -= third;

  t_diff_deviator(1, 1, 0, 0) -= third;
  t_diff_deviator(1, 1, 1, 1) -= third;

  t_diff_deviator(2, 2, 0, 0) -= third;
  t_diff_deviator(2, 2, 1, 1) -= third;

  if constexpr (DIM == 3) {
    t_diff_deviator(0, 0, 2, 2) -= third;
    t_diff_deviator(1, 1, 2, 2) -= third;
    t_diff_deviator(2, 2, 2, 2) -= third;
  }

  return t_diff_deviator;
};

inline auto diff_deviator(FTensor::Ddg<double, 2, 2> &&t_diff_stress) {
  return diff_deviator(std::move(t_diff_stress), FTensor::Number<2>());
}

inline auto diff_deviator(FTensor::Ddg<double, 3, 3> &&t_diff_stress) {
  return diff_deviator(std::move(t_diff_stress), FTensor::Number<3>());
}

/**
 *

\f[
\begin{split}
f&=\sqrt{s_{ij}s_{ij}}\\
A_{ij}&=\frac{\partial f}{\partial \sigma_{ij}}=
\frac{1}{f} s_{kl} \frac{\partial s_{kl}}{\partial \sigma_{ij}}\\
\frac{\partial A_{ij}}{\partial \sigma_{kl}}&= \frac{\partial^2 f}{\partial
\sigma_{ij}\partial\sigma_{mn}}= \frac{1}{f} \left( \frac{\partial
s_{kl}}{\partial \sigma_{mn}}\frac{\partial s_{kl}}{\partial \sigma_{ij}}
-A_{mn}A_{ij}
\right)\\
\frac{\partial f}{\partial \varepsilon_{ij}}&=A_{mn}D_{mnij}
\\
\frac{\partial A_{ij}}{\partial \varepsilon_{kl}}&=
\frac{\partial A_{ij}}{\partial \sigma_{mn}} \frac{\partial
\sigma_{mn}}{\partial \varepsilon_{kl}}= \frac{\partial A_{ij}}{\partial
\sigma_{mn}} D_{mnkl}
\end{split}
\f]

 */
inline double
platsic_surface(FTensor::Tensor2_symmetric<double, 3> &&t_stress_deviator) {
  return std::sqrt(1.5 * t_stress_deviator(I, J) * t_stress_deviator(I, J)) +
         std::numeric_limits<double>::epsilon();
};

template <int DIM>
inline auto plastic_flow(long double f,
                         FTensor::Tensor2_symmetric<double, 3> &&t_dev_stress,
                         FTensor::Ddg<double, 3, DIM> &&t_diff_deviator) {
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  FTensor::Tensor2_symmetric<double, DIM> t_diff_f;
  t_diff_f(k, l) =
      (1.5 * (t_dev_stress(I, J) * t_diff_deviator(I, J, k, l))) / f;
  return t_diff_f;
};

template <typename T, int DIM>
inline auto
diff_plastic_flow_dstress(long double f,
                          FTensor::Tensor2_symmetric<T, DIM> &t_flow,
                          FTensor::Ddg<double, 3, DIM> &&t_diff_deviator) {
  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  FTensor::Ddg<double, DIM, DIM> t_diff_flow;
  t_diff_flow(i, j, k, l) =
      (1.5 * (t_diff_deviator(M, N, i, j) * t_diff_deviator(M, N, k, l) -
              (2. / 3.) * t_flow(i, j) * t_flow(k, l))) /
      f;
  return t_diff_flow;
};

template <typename T, int DIM>
inline auto diff_plastic_flow_dstrain(
    FTensor::Ddg<T, DIM, DIM> &t_D,
    FTensor::Ddg<double, DIM, DIM> &&t_diff_plastic_flow_dstress) {
  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  FTensor::Index<'m', DIM> m;
  FTensor::Index<'n', DIM> n;
  FTensor::Ddg<double, DIM, DIM> t_diff_flow;
  t_diff_flow(i, j, k, l) =
      t_diff_plastic_flow_dstress(i, j, m, n) * t_D(m, n, k, l);
  return t_diff_flow;
};

// inline double constrain_diff_sign(double x) {
//   const auto y = x / zeta;
//   if (y > std::numeric_limits<float>::max_exponent10 ||
//       y < std::numeric_limits<float>::min_exponent10) {
//     return 0;
//   } else {
//     const auto e = std::exp(y);
//     const auto ep1 = e + 1;
//     return (2 / zeta) * (e / (ep1 * ep1));
//   }
// };

inline double constrian_sign(double x) {
  const auto y = x / zeta;
  if (y > std::numeric_limits<float>::max_exponent10 ||
      y < std::numeric_limits<float>::min_exponent10) {
    if (x > 0)
      return 1.;
    else
      return -1.;
  } else {
    const auto e = std::exp(y);
    return (e - 1) / (1 + e);
  }
};

inline double constrain_abs(double x) {
  const auto y = -x / zeta;
  if (y > std::numeric_limits<float>::max_exponent10 ||
      y < std::numeric_limits<float>::min_exponent10) {
    return std::abs(x);
  } else {
    const double e = std::exp(y);
    return x + 2 * zeta * std::log1p(e);
  }
};

inline double w(double eqiv, double dot_tau, double f, double sigma_y,
                double sigma_Y) {
  return (f - sigma_y) / sigma_Y + cn1 * (eqiv);
};

/**

\f[
\dot{\tau} - \frac{1}{2}\left\{\dot{\tau} + (f(\pmb\sigma) - \sigma_y) +
\| \dot{\tau} + (f(\pmb\sigma) - \sigma_y) \|\right\} = 0 \\
c_n \sigma_y \dot{\tau} - \frac{1}{2}\left\{c_n\sigma_y \dot{\tau} +
(f(\pmb\sigma) - \sigma_y) +
\| c_n \sigma_y \dot{\tau} + (f(\pmb\sigma) - \sigma_y) \|\right\} = 0
\f]

 */
inline double constraint(double eqiv, double dot_tau, double f, double sigma_y,
                         double abs_w, double vis_H, double sigma_Y) {
  return vis_H * dot_tau +
         (sigma_Y / 2) * ((cn0 * (dot_tau - eqiv) + cn1 * (eqiv) -
                           (f - sigma_y) / sigma_Y) -
                          abs_w);
};

inline double diff_constrain_ddot_tau(double sign, double eqiv, double dot_tau,
                                      double vis_H, double sigma_Y) {
  return vis_H + (sigma_Y / 2) * (cn0);
};

inline double diff_constrain_deqiv(double sign, double eqiv, double dot_tau,
                                   double sigma_Y) {
  return (sigma_Y / 2) * (-cn0 + cn1 * (1 - sign));
};

inline auto diff_constrain_df(double sign) { return (-1 - sign) / 2; };

inline auto diff_constrain_dsigma_y(double sign) { return (1 + sign) / 2; }

template <typename T, int DIM>
inline auto
diff_constrain_dstress(double diff_constrain_df,
                       FTensor::Tensor2_symmetric<T, DIM> &t_plastic_flow) {
  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Tensor2_symmetric<double, DIM> t_diff_constrain_dstress;
  t_diff_constrain_dstress(i, j) = diff_constrain_df * t_plastic_flow(i, j);
  return t_diff_constrain_dstress;
};

template <typename T1, typename T2, int DIM>
inline auto diff_constrain_dstrain(T1 &t_D, T2 &&t_diff_constrain_dstress) {
  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  FTensor::Tensor2_symmetric<double, DIM> t_diff_constrain_dstrain;
  t_diff_constrain_dstrain(k, l) =
      t_diff_constrain_dstress(i, j) * t_D(i, j, k, l);
  return t_diff_constrain_dstrain;
};

template <typename T, int DIM>
inline auto equivalent_strain_dot(
    FTensor::Tensor2_symmetric<T, DIM> &t_plastic_strain_dot) {
  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  constexpr double A = 2. / 3;
  return std::sqrt(A * t_plastic_strain_dot(i, j) *
                   t_plastic_strain_dot(i, j)) +
         std::numeric_limits<double>::epsilon();
};

template <typename T1, typename T2, typename T3, int DIM>
inline auto diff_equivalent_strain_dot(const T1 eqiv, T2 &t_plastic_strain_dot,
                                       T3 &t_diff_plastic_strain,
                                       FTensor::Number<DIM>) {
  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  constexpr double A = 2. / 3;
  FTensor::Tensor2_symmetric<double, DIM> t_diff_eqiv;
  t_diff_eqiv(i, j) = A * (t_plastic_strain_dot(k, l) / eqiv) *
                      t_diff_plastic_strain(k, l, i, j);
  return t_diff_eqiv;
};

//! [Lambda functions]

//! [Auxiliary functions functions
static inline auto get_mat_tensor_sym_dvector(size_t rr, MatrixDouble &mat,
                                              FTensor::Number<2>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 3, 2>{
      &mat(3 * rr + 0, 0), &mat(3 * rr + 0, 1), &mat(3 * rr + 1, 0),
      &mat(3 * rr + 1, 1), &mat(3 * rr + 2, 0), &mat(3 * rr + 2, 1)};
}

static inline auto get_mat_tensor_sym_dvector(size_t rr, MatrixDouble &mat,
                                              FTensor::Number<3>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 6, 3>{
      &mat(6 * rr + 0, 0), &mat(6 * rr + 0, 1), &mat(6 * rr + 0, 2),
      &mat(6 * rr + 1, 0), &mat(6 * rr + 1, 1), &mat(6 * rr + 1, 2),
      &mat(6 * rr + 2, 0), &mat(6 * rr + 2, 1), &mat(6 * rr + 2, 2),
      &mat(6 * rr + 3, 0), &mat(6 * rr + 3, 1), &mat(6 * rr + 3, 2),
      &mat(6 * rr + 4, 0), &mat(6 * rr + 4, 1), &mat(6 * rr + 4, 2),
      &mat(6 * rr + 5, 0), &mat(6 * rr + 5, 1), &mat(6 * rr + 5, 2)};
}

//! [Auxiliary functions functions

template <int DIM, typename DomainEleOp>
struct OpCalculatePlasticSurfaceImpl<DIM, GAUSS, DomainEleOp>
    : public DomainEleOp {
  OpCalculatePlasticSurfaceImpl(const std::string field_name,
                                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM, typename DomainEleOp>
OpCalculatePlasticSurfaceImpl<DIM, GAUSS, DomainEleOp>::
    OpCalculatePlasticSurfaceImpl(const std::string field_name,
                                  boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Operator is only executed for vertices
  std::fill(&DomainEleOp::doEntities[MBEDGE],
            &DomainEleOp::doEntities[MBMAXTYPE], false);
}

template <int DIM, typename DomainEleOp>
MoFEMErrorCode OpCalculatePlasticSurfaceImpl<DIM, GAUSS, DomainEleOp>::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;

  const size_t nb_gauss_pts = commonDataPtr->mStressPtr->size2();
  auto t_stress =
      getFTensor2SymmetricFromMat<DIM>(*(commonDataPtr->mStressPtr));
  auto t_plastic_strain =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->plasticStrain);

  commonDataPtr->plasticSurface.resize(nb_gauss_pts, false);
  commonDataPtr->plasticFlow.resize(size_symm, nb_gauss_pts, false);
  auto t_flow = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->plasticFlow);

  auto &params = commonDataPtr->blockParams;

  for (auto &f : commonDataPtr->plasticSurface) {

    f = platsic_surface(

        deviator(
            t_stress, trace(t_stress),
            kinematic_hardening(t_plastic_strain, params[CommonData::C1_k]))

    );

    auto t_flow_tmp =
        plastic_flow(f,

                     deviator(t_stress, trace(t_stress),
                              kinematic_hardening(t_plastic_strain,
                                                  params[CommonData::C1_k])),

                     diff_deviator(diff_tensor(FTensor::Number<DIM>())));
    t_flow(i, j) = t_flow_tmp(i, j);

    ++t_plastic_strain;
    ++t_flow;
    ++t_stress;
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, typename DomainEleOp>
struct OpCalculatePlasticityImpl<DIM, GAUSS, DomainEleOp> : public DomainEleOp {
  OpCalculatePlasticityImpl(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr,
                            boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

template <int DIM, typename DomainEleOp>
OpCalculatePlasticityImpl<DIM, GAUSS, DomainEleOp>::OpCalculatePlasticityImpl(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  // Opetor is only executed for vertices
  std::fill(&DomainEleOp::doEntities[MBEDGE],
            &DomainEleOp::doEntities[MBMAXTYPE], false);
}

template <int DIM, typename DomainEleOp>
MoFEMErrorCode OpCalculatePlasticityImpl<DIM, GAUSS, DomainEleOp>::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  FTensor::Index<'m', DIM> m;
  FTensor::Index<'n', DIM> n;
  
  auto &params = commonDataPtr->blockParams; ///< material parameters

  const size_t nb_gauss_pts = DomainEleOp::getGaussPts().size2();
  auto t_w = DomainEleOp::getFTensor0IntegrationWeight();
  auto t_tau = getFTensor0FromVec(commonDataPtr->plasticTau);
  auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);
  auto t_f = getFTensor0FromVec(commonDataPtr->plasticSurface);
  auto t_flow = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->plasticFlow);
  auto t_plastic_strain =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->plasticStrain);
  auto t_plastic_strain_dot =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->plasticStrainDot);
  auto t_stress =
      getFTensor2SymmetricFromMat<DIM>(*(commonDataPtr->mStressPtr));

  auto t_D = getFTensor4DdgFromMat<DIM, DIM, 0>(*commonDataPtr->mDPtr);
  auto t_D_Op = getFTensor4DdgFromMat<DIM, DIM, 0>(*mDPtr);

  auto t_diff_plastic_strain = diff_tensor(FTensor::Number<DIM>());
  auto t_diff_deviator = diff_deviator(diff_tensor(FTensor::Number<DIM>()));

  FTensor::Ddg<double, DIM, DIM> t_flow_dir_dstress;
  FTensor::Ddg<double, DIM, DIM> t_flow_dir_dstrain;
  t_flow_dir_dstress(i, j, k, l) =
      1.5 * (t_diff_deviator(M, N, i, j) * t_diff_deviator(M, N, k, l));
  t_flow_dir_dstrain(i, j, k, l) =
      t_flow_dir_dstress(i, j, m, n) * t_D_Op(m, n, k, l);


  auto t_alpha_dir =
      kinematic_hardening_dplastic_strain<DIM>(params[CommonData::C1_k]);

  commonDataPtr->resC.resize(nb_gauss_pts, false);
  commonDataPtr->resCdTau.resize(nb_gauss_pts, false);
  commonDataPtr->resCdStrain.resize(size_symm, nb_gauss_pts, false);
  commonDataPtr->resCdPlasticStrain.resize(size_symm, nb_gauss_pts, false);
  commonDataPtr->resFlow.resize(size_symm, nb_gauss_pts, false);
  commonDataPtr->resFlowDtau.resize(size_symm, nb_gauss_pts, false);
  commonDataPtr->resFlowDstrain.resize(size_symm * size_symm, nb_gauss_pts,
                                       false);
  commonDataPtr->resFlowDstrainDot.resize(size_symm * size_symm, nb_gauss_pts,
                                          false);

  commonDataPtr->resC.clear();
  commonDataPtr->resCdTau.clear();
  commonDataPtr->resCdStrain.clear();
  commonDataPtr->resCdPlasticStrain.clear();
  commonDataPtr->resFlow.clear();
  commonDataPtr->resFlowDtau.clear();
  commonDataPtr->resFlowDstrain.clear();
  commonDataPtr->resFlowDstrainDot.clear();

  auto t_res_c = getFTensor0FromVec(commonDataPtr->resC);
  auto t_res_c_dtau = getFTensor0FromVec(commonDataPtr->resCdTau);
  auto t_res_c_dstrain =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resCdStrain);
  auto t_res_c_plastic_strain =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resCdPlasticStrain);
  auto t_res_flow = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resFlow);
  auto t_res_flow_dtau =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resFlowDtau);
  auto t_res_flow_dstrain =
      getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->resFlowDstrain);
  auto t_res_flow_dplastic_strain =
      getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->resFlowDstrainDot);

  auto next = [&]() {
    ++t_tau;
    ++t_tau_dot;
    ++t_f;
    ++t_flow;
    ++t_plastic_strain;
    ++t_plastic_strain_dot;
    ++t_stress;
    ++t_res_c;
    ++t_res_c_dtau;
    ++t_res_c_dstrain;
    ++t_res_c_plastic_strain;
    ++t_res_flow;
    ++t_res_flow_dtau;
    ++t_res_flow_dstrain;
    ++t_res_flow_dplastic_strain;
    ++t_w;
  };

  auto get_avtive_pts = [&]() {
    int nb_points_avtive_on_elem = 0;
    int nb_points_on_elem = 0;

    auto t_tau = getFTensor0FromVec(commonDataPtr->plasticTau);
    auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);
    auto t_f = getFTensor0FromVec(commonDataPtr->plasticSurface);
    auto t_plastic_strain_dot =
        getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticStrainDot);

    for (auto &f : commonDataPtr->plasticSurface) {
      auto eqiv = equivalent_strain_dot(t_plastic_strain_dot);
      const auto ww = w(
          eqiv, t_tau_dot, t_f,
          iso_hardening(t_tau, params[CommonData::H], params[CommonData::QINF],
                        params[CommonData::BISO], params[CommonData::SIGMA_Y]),
          params[CommonData::SIGMA_Y]);
      const auto sign_ww = constrian_sign(ww);

      ++nb_points_on_elem;
      if (sign_ww > 0) {
        ++nb_points_avtive_on_elem;
      }

      ++t_tau;
      ++t_tau_dot;
      ++t_f;
      ++t_plastic_strain_dot;
    }

    int &active_points = PlasticOps::CommonData::activityData[0];
    int &avtive_full_elems = PlasticOps::CommonData::activityData[1];
    int &avtive_elems = PlasticOps::CommonData::activityData[2];
    int &nb_points = PlasticOps::CommonData::activityData[3];
    int &nb_elements = PlasticOps::CommonData::activityData[4];

    ++nb_elements;
    nb_points += nb_points_on_elem;
    if (nb_points_avtive_on_elem > 0) {
      ++avtive_elems;
      active_points += nb_points_avtive_on_elem;
      if (nb_points_avtive_on_elem == nb_points_on_elem) {
        ++avtive_full_elems;
      }
    }

    if (nb_points_avtive_on_elem != nb_points_on_elem)
      return 1;
    else
      return 0;
  };

  if (DomainEleOp::getTSCtx() == TSMethod::TSContext::CTX_TSSETIJACOBIAN) {
    get_avtive_pts();
  }

  for (auto &f : commonDataPtr->plasticSurface) {

    auto eqiv = equivalent_strain_dot(t_plastic_strain_dot);
    auto t_diff_eqiv = diff_equivalent_strain_dot(eqiv, t_plastic_strain_dot,
                                                  t_diff_plastic_strain,
                                                  FTensor::Number<DIM>());

    const auto sigma_y =
        iso_hardening(t_tau, params[CommonData::H], params[CommonData::QINF],
                  params[CommonData::BISO], params[CommonData::SIGMA_Y]);
    const auto d_sigma_y =
        iso_hardening_dtau(t_tau, params[CommonData::H],
                           params[CommonData::QINF], params[CommonData::BISO]);

    auto ww = w(eqiv, t_tau_dot, t_f, sigma_y, params[CommonData::SIGMA_Y]);
    auto abs_ww = constrain_abs(ww);
    auto sign_ww = constrian_sign(ww);

    auto c = constraint(eqiv, t_tau_dot, t_f, sigma_y, abs_ww,
                        params[CommonData::VIS_H], params[CommonData::SIGMA_Y]);
    auto c_dot_tau = diff_constrain_ddot_tau(sign_ww, eqiv, t_tau_dot,
                                             params[CommonData::VIS_H],
                                             params[CommonData::SIGMA_Y]);
    auto c_equiv = diff_constrain_deqiv(sign_ww, eqiv, t_tau_dot,
                                        params[CommonData::SIGMA_Y]);
    auto c_sigma_y = diff_constrain_dsigma_y(sign_ww);
    auto c_f = diff_constrain_df(sign_ww);

    auto t_dev_stress = deviator(

        t_stress, trace(t_stress),

        kinematic_hardening(t_plastic_strain, params[CommonData::C1_k])

    );

    FTensor::Tensor2_symmetric<double, DIM> t_flow_dir;
    t_flow_dir(k, l) = 1.5 * (t_dev_stress(I, J) * t_diff_deviator(I, J, k, l));
    FTensor::Tensor2_symmetric<double, DIM> t_flow_dstrain;
    t_flow_dstrain(i, j) = t_flow(k, l) * t_D_Op(k, l, i, j);

    auto get_res_c = [&]() { return c; };

    auto get_res_c_dstrain = [&](auto &t_diff_res) {
      t_diff_res(i, j) = c_f * t_flow_dstrain(i, j);
    };

    auto get_res_c_dplastic_strain = [&](auto &t_diff_res) {
      t_diff_res(i, j) = (DomainEleOp::getTSa() * c_equiv) * t_diff_eqiv(i, j);
      t_diff_res(k, l) -= c_f * t_flow(i, j) * t_alpha_dir(i, j, k, l);
    };

    auto get_res_c_dtau = [&]() {
      return DomainEleOp::getTSa() * c_dot_tau + c_sigma_y * d_sigma_y;
    };

    auto get_res_c_plastic_strain = [&](auto &t_diff_res) {
      t_diff_res(k, l) = -c_f * t_flow(i, j) * t_alpha_dir(i, j, k, l);
    };

    auto get_res_flow = [&](auto &t_res_flow) {
      const auto a = sigma_y;
      const auto b = t_tau_dot;
      t_res_flow(k, l) = a * t_plastic_strain_dot(k, l) - b * t_flow_dir(k, l);
    };

    auto get_res_flow_dtau = [&](auto &t_res_flow_dtau) {
      const auto da = d_sigma_y;
      const auto db = DomainEleOp::getTSa();
      t_res_flow_dtau(k, l) =
          da * t_plastic_strain_dot(k, l) - db * t_flow_dir(k, l);
    };

    auto get_res_flow_dstrain = [&](auto &t_res_flow_dstrain) {
      const auto b = t_tau_dot;
      t_res_flow_dstrain(m, n, k, l) = -t_flow_dir_dstrain(m, n, k, l) * b;
    };

    auto get_res_flow_dplastic_strain = [&](auto &t_res_flow_dplastic_strain) {
      const auto a = sigma_y;
      t_res_flow_dplastic_strain(m, n, k, l) =
          (a * DomainEleOp::getTSa()) * t_diff_plastic_strain(m, n, k, l);
      const auto b = t_tau_dot;
      t_res_flow_dplastic_strain(m, n, i, j) +=
          (t_flow_dir_dstrain(m, n, k, l) * t_alpha_dir(k, l, i, j)) * b;
    };

    t_res_c = get_res_c();
    get_res_flow(t_res_flow);

    if (DomainEleOp::getTSCtx() == TSMethod::TSContext::CTX_TSSETIJACOBIAN) {
      t_res_c_dtau = get_res_c_dtau();
      get_res_c_dstrain(t_res_c_dstrain);
      get_res_c_dplastic_strain(t_res_c_plastic_strain);
      get_res_flow_dtau(t_res_flow_dtau);
      get_res_flow_dstrain(t_res_flow_dstrain);
      get_res_flow_dplastic_strain(t_res_flow_dplastic_strain);
    }

    next();
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, typename DomainEleOp>
struct OpPlasticStressImpl<DIM, GAUSS, DomainEleOp> : public DomainEleOp {
  OpPlasticStressImpl(const std::string field_name,
                      boost::shared_ptr<CommonData> common_data_ptr,
                      boost::shared_ptr<MatrixDouble> mDPtr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM, typename DomainEleOp>
OpPlasticStressImpl<DIM, GAUSS, DomainEleOp>::OpPlasticStressImpl(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  // Operator is only executed for vertices
  std::fill(&DomainEleOp::doEntities[MBEDGE],
            &DomainEleOp::doEntities[MBMAXTYPE], false);
}

//! [Calculate stress]
template <int DIM, typename DomainEleOp>
MoFEMErrorCode
OpPlasticStressImpl<DIM, GAUSS, DomainEleOp>::doWork(int side, EntityType type,
                                                     EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;

  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  commonDataPtr->mStressPtr->resize((DIM * (DIM + 1)) / 2, nb_gauss_pts);
  auto t_D = getFTensor4DdgFromMat<DIM, DIM, 0>(*mDPtr);
  auto t_strain =
      getFTensor2SymmetricFromMat<DIM>(*(commonDataPtr->mStrainPtr));
  auto t_plastic_strain =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->plasticStrain);
  auto t_stress =
      getFTensor2SymmetricFromMat<DIM>(*(commonDataPtr->mStressPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    t_stress(i, j) =
        t_D(i, j, k, l) * (t_strain(k, l) - t_plastic_strain(k, l));
    ++t_strain;
    ++t_plastic_strain;
    ++t_stress;
  }

  MoFEMFunctionReturn(0);
}
//! [Calculate stress]

template <int DIM, typename AssemblyDomainEleOp>
struct OpCalculatePlasticFlowRhsImpl<DIM, GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowRhsImpl(const std::string field_name,
                                boost::shared_ptr<CommonData> common_data_ptr,
                                boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

template <int DIM, typename AssemblyDomainEleOp>
OpCalculatePlasticFlowRhsImpl<DIM, GAUSS, AssemblyDomainEleOp>::
    OpCalculatePlasticFlowRhsImpl(const std::string field_name,
                                  boost::shared_ptr<CommonData> common_data_ptr,
                                  boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {}

template <int DIM, typename AssemblyDomainEleOp>
MoFEMErrorCode
OpCalculatePlasticFlowRhsImpl<DIM, GAUSS, AssemblyDomainEleOp>::iNtegrate(
    EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
  FTensor::Index<'L', size_symm> L;

  const auto nb_integration_pts = AssemblyDomainEleOp::getGaussPts().size2();
  const auto nb_base_functions = data.getN().size2();

  auto t_res_flow = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resFlow);

  auto t_L = symm_L_tensor(FTensor::Number<DIM>());

  auto next = [&]() { ++t_res_flow; };

  auto t_w = AssemblyDomainEleOp::getFTensor0IntegrationWeight();
  auto t_base = data.getFTensor0N();
  auto &nf = AssemblyDomainEleOp::locF;
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = AssemblyDomainEleOp::getMeasure() * t_w;
    ++t_w;

    FTensor::Tensor1<double, size_symm> t_rhs;
    t_rhs(L) = alpha * (t_res_flow(i, j) * t_L(i, j, L));
    next();

    auto t_nf = getFTensor1FromArray<size_symm, size_symm>(nf);
    size_t bb = 0;
    for (; bb != AssemblyDomainEleOp::nbRows / size_symm; ++bb) {
      t_nf(L) += t_base * t_rhs(L);
      ++t_base;
      ++t_nf;
    }
    for (; bb < nb_base_functions; ++bb)
      ++t_base;
  }

  MoFEMFunctionReturn(0);
}

template <typename AssemblyDomainEleOp>
struct OpCalculateConstraintsRhsImpl<GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculateConstraintsRhsImpl(const std::string field_name,
                                boost::shared_ptr<CommonData> common_data_ptr,
                                boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

template <typename AssemblyDomainEleOp>
OpCalculateConstraintsRhsImpl<GAUSS, AssemblyDomainEleOp>::
    OpCalculateConstraintsRhsImpl(const std::string field_name,
                                  boost::shared_ptr<CommonData> common_data_ptr,
                                  boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(field_name, field_name, AssemblyDomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {}

template <typename AssemblyDomainEleOp>
MoFEMErrorCode
OpCalculateConstraintsRhsImpl<GAUSS, AssemblyDomainEleOp>::iNtegrate(
    EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_integration_pts = AssemblyDomainEleOp::getGaussPts().size2();
  const size_t nb_base_functions = data.getN().size2();

  auto t_res_c = getFTensor0FromVec(commonDataPtr->resC);

  auto next = [&]() { ++t_res_c; };

  auto t_w = AssemblyDomainEleOp::getFTensor0IntegrationWeight();
  auto &nf = AssemblyDomainEleOp::locF;
  auto t_base = data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = AssemblyDomainEleOp::getMeasure() * t_w;
    ++t_w;
    const auto res = alpha * t_res_c;
    next();

    size_t bb = 0;
    for (; bb != AssemblyDomainEleOp::nbRows; ++bb) {
      nf[bb] += t_base * res;
      ++t_base;
    }
    for (; bb < nb_base_functions; ++bb)
      ++t_base;
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, typename AssemblyDomainEleOp>
struct OpCalculatePlasticFlowLhs_dEPImpl<DIM, GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dEPImpl(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

template <int DIM, typename AssemblyDomainEleOp>
OpCalculatePlasticFlowLhs_dEPImpl<DIM, GAUSS, AssemblyDomainEleOp>::
    OpCalculatePlasticFlowLhs_dEPImpl(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr,
        boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          AssemblyDomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  AssemblyDomainEleOp::sYmm = false;
}

static inline auto get_mat_tensor_sym_dtensor_sym(size_t rr, MatrixDouble &mat,
                                                  FTensor::Number<2>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>{
      &mat(3 * rr + 0, 0), &mat(3 * rr + 0, 1), &mat(3 * rr + 0, 2),
      &mat(3 * rr + 1, 0), &mat(3 * rr + 1, 1), &mat(3 * rr + 1, 2),
      &mat(3 * rr + 2, 0), &mat(3 * rr + 2, 1), &mat(3 * rr + 2, 2)};
}

static inline auto get_mat_tensor_sym_dtensor_sym(size_t rr, MatrixDouble &mat,
                                                  FTensor::Number<3>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 6, 6>{
      &mat(6 * rr + 0, 0), &mat(6 * rr + 0, 1), &mat(6 * rr + 0, 2),
      &mat(6 * rr + 0, 3), &mat(6 * rr + 0, 4), &mat(6 * rr + 0, 5),
      &mat(6 * rr + 1, 0), &mat(6 * rr + 1, 1), &mat(6 * rr + 1, 2),
      &mat(6 * rr + 1, 3), &mat(6 * rr + 1, 4), &mat(6 * rr + 1, 5),
      &mat(6 * rr + 2, 0), &mat(6 * rr + 2, 1), &mat(6 * rr + 2, 2),
      &mat(6 * rr + 2, 3), &mat(6 * rr + 2, 4), &mat(6 * rr + 2, 5),
      &mat(6 * rr + 3, 0), &mat(6 * rr + 3, 1), &mat(6 * rr + 3, 2),
      &mat(6 * rr + 3, 3), &mat(6 * rr + 3, 4), &mat(6 * rr + 3, 5),
      &mat(6 * rr + 4, 0), &mat(6 * rr + 4, 1), &mat(6 * rr + 4, 2),
      &mat(6 * rr + 4, 3), &mat(6 * rr + 4, 4), &mat(6 * rr + 4, 5),
      &mat(6 * rr + 5, 0), &mat(6 * rr + 5, 1), &mat(6 * rr + 5, 2),
      &mat(6 * rr + 5, 3), &mat(6 * rr + 5, 4), &mat(6 * rr + 5, 5)};
}

template <int DIM, typename AssemblyDomainEleOp>
MoFEMErrorCode
OpCalculatePlasticFlowLhs_dEPImpl<DIM, GAUSS, AssemblyDomainEleOp>::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
  FTensor::Index<'L', size_symm> L;
  FTensor::Index<'O', size_symm> O;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const auto nb_integration_pts = AssemblyDomainEleOp::getGaussPts().size2();
  const auto nb_row_base_functions = row_data.getN().size2();

  auto t_res_flow_dstrain =
      getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->resFlowDstrain);
  auto t_res_flow_dplastic_strain =
      getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->resFlowDstrainDot);
  auto t_L = symm_L_tensor(FTensor::Number<DIM>());

  auto next = [&]() {
    ++t_res_flow_dstrain;
    ++t_res_flow_dplastic_strain;
  };

  auto t_w = AssemblyDomainEleOp::getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = AssemblyDomainEleOp::getMeasure() * t_w;
    ++t_w;

    FTensor::Tensor2<double, size_symm, size_symm> t_res_mat;
    t_res_mat(O, L) =
        alpha * (t_L(i, j, O) * ((t_res_flow_dplastic_strain(i, j, k, l) -
                                  t_res_flow_dstrain(i, j, k, l)) *
                                 t_L(k, l, L)));
    next();

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / size_symm; ++rr) {
      auto t_mat = get_mat_tensor_sym_dtensor_sym(rr, locMat,
                                                  FTensor::Number<SPACE_DIM>());
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / size_symm; ++cc) {
        t_mat(O, L) += ((t_row_base * t_col_base) * t_res_mat(O, L));
        ++t_mat;
        ++t_col_base;
      }

      ++t_row_base;
    }

    for (; rr < nb_row_base_functions; ++rr)
      ++t_row_base;
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, typename AssemblyDomainEleOp>
struct OpCalculateConstraintsLhs_dUImpl<DIM, GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_dUImpl(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

template <int DIM, typename AssemblyDomainEleOp>
struct OpCalculatePlasticFlowLhs_dTAUImpl<DIM, GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dTAUImpl(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

template <int DIM, typename AssemblyDomainEleOp>
OpCalculatePlasticFlowLhs_dTAUImpl<DIM, GAUSS, AssemblyDomainEleOp>::
    OpCalculatePlasticFlowLhs_dTAUImpl(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr,
        boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  AssemblyDomainEleOp::sYmm = false;
}

static inline auto get_mat_tensor_sym_dscalar(size_t rr, MatrixDouble &mat,
                                              FTensor::Number<2>) {
  return FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3>{
      &mat(3 * rr + 0, 0), &mat(3 * rr + 1, 0), &mat(3 * rr + 2, 0)};
}

static inline auto get_mat_tensor_sym_dscalar(size_t rr, MatrixDouble &mat,
                                              FTensor::Number<3>) {
  return FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 6>{
      &mat(6 * rr + 0, 0), &mat(6 * rr + 1, 0), &mat(6 * rr + 2, 0),
      &mat(6 * rr + 3, 0), &mat(6 * rr + 4, 0), &mat(6 * rr + 5, 0)};
}

template <int DIM, typename AssemblyDomainEleOp>
MoFEMErrorCode
OpCalculatePlasticFlowLhs_dTAUImpl<DIM, GAUSS, AssemblyDomainEleOp>::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
  FTensor::Index<'L', size_symm> L;

  const auto nb_integration_pts = AssemblyDomainEleOp::getGaussPts().size2();
  const size_t nb_row_base_functions = row_data.getN().size2();
  auto &locMat = AssemblyDomainEleOp::locMat;

  const auto type = AssemblyDomainEleOp::getFEType();
  const auto nb_nodes = moab::CN::VerticesPerEntity(type);

  auto t_res_flow_dtau =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resFlowDtau);

  auto t_L = symm_L_tensor(FTensor::Number<DIM>());

  auto next = [&]() { ++t_res_flow_dtau; };

  auto t_w = AssemblyDomainEleOp::getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = AssemblyDomainEleOp::getMeasure() * t_w;
    ++t_w;
    FTensor::Tensor1<double, size_symm> t_res_vec;
    t_res_vec(L) = alpha * (t_res_flow_dtau(i, j) * t_L(i, j, L));
    next();

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / size_symm; ++rr) {
      auto t_mat =
          get_mat_tensor_sym_dscalar(rr, locMat, FTensor::Number<DIM>());
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols; cc++) {
        t_mat(L) += t_row_base * t_col_base * t_res_vec(L);
        ++t_mat;
        ++t_col_base;
      }
      ++t_row_base;
    }
    for (; rr != nb_row_base_functions; ++rr)
      ++t_row_base;
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, typename AssemblyDomainEleOp>
struct OpCalculateConstraintsLhs_dEPImpl<DIM, GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_dEPImpl(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<MatrixDouble> mat_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

template <int DIM, typename AssemblyDomainEleOp>
OpCalculateConstraintsLhs_dEPImpl<DIM, GAUSS, AssemblyDomainEleOp>::
    OpCalculateConstraintsLhs_dEPImpl(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr,
        boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  AssemblyDomainEleOp::sYmm = false;
}

auto get_mat_scalar_dtensor_sym(MatrixDouble &mat, FTensor::Number<2>) {
  return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>{
      &mat(0, 0), &mat(0, 1), &mat(0, 2)};
}

auto get_mat_scalar_dtensor_sym(MatrixDouble &mat, FTensor::Number<3>) {
  return FTensor::Tensor1<FTensor::PackPtr<double *, 6>, 6>{
      &mat(0, 0), &mat(0, 1), &mat(0, 2), &mat(0, 3), &mat(0, 4), &mat(0, 5)};
}

template <int DIM, typename AssemblyDomainEleOp>
MoFEMErrorCode
OpCalculateConstraintsLhs_dEPImpl<DIM, GAUSS, AssemblyDomainEleOp>::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
  FTensor::Index<'L', size_symm> L;

  const auto nb_integration_pts = AssemblyDomainEleOp::getGaussPts().size2();
  const auto nb_row_base_functions = row_data.getN().size2();

  auto t_c_dstrain =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->resCdStrain);
  auto t_c_dplastic_strain =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->resCdPlasticStrain);

  auto next = [&]() {
    ++t_c_dstrain;
    ++t_c_dplastic_strain;
  };

  auto t_L = symm_L_tensor(FTensor::Number<SPACE_DIM>());

  auto t_w = AssemblyDomainEleOp::getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (auto gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = AssemblyDomainEleOp::getMeasure() * t_w;
    ++t_w;

    FTensor::Tensor1<double, size_symm> t_res_vec;
    t_res_vec(L) =
        t_L(i, j, L) * (t_c_dplastic_strain(i, j) - t_c_dstrain(i, j));
    next();

    auto t_mat = get_mat_scalar_dtensor_sym(AssemblyDomainEleOp::locMat,
                                            FTensor::Number<SPACE_DIM>());
    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows; ++rr) {
      const auto row_base = alpha * t_row_base;
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / size_symm; cc++) {
        t_mat(L) += (row_base * t_col_base) * t_res_vec(L);
        ++t_mat;
        ++t_col_base;
      }
      ++t_row_base;
    }
    for (; rr != nb_row_base_functions; ++rr)
      ++t_row_base;
  }

  MoFEMFunctionReturn(0);
}

template <typename AssemblyDomainEleOp>
struct OpCalculateConstraintsLhs_dTAUImpl<GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_dTAUImpl(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <typename AssemblyDomainEleOp>
OpCalculateConstraintsLhs_dTAUImpl<GAUSS, AssemblyDomainEleOp>::
    OpCalculateConstraintsLhs_dTAUImpl(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  AssemblyDomainEleOp::sYmm = false;
}

template <typename AssemblyDomainEleOp>
MoFEMErrorCode
OpCalculateConstraintsLhs_dTAUImpl<GAUSS, AssemblyDomainEleOp>::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  const auto nb_integration_pts = AssemblyDomainEleOp::getGaussPts().size2();
  const auto nb_row_base_functions = row_data.getN().size2();

  auto t_res_c_dtau = getFTensor0FromVec(commonDataPtr->resCdTau);
  auto next = [&]() { ++t_res_c_dtau; };

  auto t_w = AssemblyDomainEleOp::getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = AssemblyDomainEleOp::getMeasure() * t_w;
    ++t_w;

    const auto res = alpha * (t_res_c_dtau);
    next();

    auto mat_ptr = AssemblyDomainEleOp::locMat.data().begin();
    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows; ++rr) {
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols; ++cc) {
        *mat_ptr += t_row_base * t_col_base * res;
        ++t_col_base;
        ++mat_ptr;
      }
      ++t_row_base;
    }
    for (; rr < nb_row_base_functions; ++rr)
      ++t_row_base;
  }

  MoFEMFunctionReturn(0);
}
}; // namespace PlasticOps
