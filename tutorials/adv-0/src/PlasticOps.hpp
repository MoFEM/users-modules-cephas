

/** \file PlasticOps.hpp
 * \example PlasticOps.hpp

\f[
\left\{
\begin{array}{ll}
\frac{\partial \sigma_{ij}}{\partial x_j} - b_i = 0 & \forall x \in \Omega \\
\varepsilon_{ij} = \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} +
\frac{\partial u_j}{\partial x_i} \right)\\
\sigma_{ij} = D_{ijkl}\left(\varepsilon_{kl}-\varepsilon^p_{kl}\right) \\
\dot{\varepsilon}^p_{kl} - \dot{\tau} \left( \left. \frac{\partial f}{\partial
\sigma_{kl}} \right|_{(\sigma,\tau) } \right) = 0 \\
f(\sigma, \tau) \leq 0,\; \dot{\tau} \geq 0,\;\dot{\tau}f(\sigma, \tau)=0\\
u_i = \overline{u}_i & \forall x \in \partial\Omega_u \\
\sigma_{ij}n_j = \overline{t}_i & \forall x \in \partial\Omega_\sigma \\
\Omega_u \cup \Omega_\sigma = \Omega \\
\Omega_u \cap \Omega_\sigma = \emptyset
\end{array}
\right.
\f]

\f[
\left\{
\begin{array}{ll}
\left(\frac{\partial \delta u_i}{\partial x_j},\sigma_{ij}\right)_\Omega-(\delta
u_i,b_i)_\Omega -(\delta u_i,\overline{t}_i)_{\partial\Omega_\sigma}=0 & \forall
\delta u_i \in H^1(\Omega)\\ \left(\delta\varepsilon^p_{kl} ,D_{ijkl}\left(
\dot{\varepsilon}^p_{kl} - \dot{\tau} A_{kl} \right)\right) = 0
& \forall \delta\varepsilon^p_{ij} \in L^2(\Omega) \cap \mathcal{S} \\
\left(\delta\tau,c_n\dot{\tau} - \frac{1}{2}\left\{c_n \dot{\tau} +
(f(\pmb\sigma,\tau) - \sigma_y) +
\| c_n \dot{\tau} + (f(\pmb\sigma,\tau) - \sigma_y) \|\right\}\right) = 0 &
\forall \delta\tau \in L^2(\Omega) \end{array} \right.
\f]

*/

namespace PlasticOps {

//! [Common data]
struct CommonData : public boost::enable_shared_from_this<CommonData> {
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<MatrixDouble> mDPtr_Axiator;
  boost::shared_ptr<MatrixDouble> mDPtr_Deviator;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;

  VectorDouble plasticSurface;
  MatrixDouble plasticFlow;
  VectorDouble plasticTau;
  VectorDouble plasticTauDot;
  MatrixDouble plasticStrain;
  MatrixDouble plasticStrainDot;

  VectorDouble resC;
  VectorDouble resCdTau;
  MatrixDouble resCdStrain;
  MatrixDouble resCdStrainDot;
  MatrixDouble resFlow;
  MatrixDouble resFlowDtau;
  MatrixDouble resFlowDstrain;
  MatrixDouble resFlowDstrainDot;

  std::array<int, 5> activityData;

  inline auto getPlasticSurfacePtr() {
    return boost::shared_ptr<VectorDouble>(shared_from_this(), &plasticSurface);
  }
  inline auto getPlasticTauPtr() {
    return boost::shared_ptr<VectorDouble>(shared_from_this(), &plasticTau);
  }
  inline auto getPlasticTauDotPtr() {
    return boost::shared_ptr<VectorDouble>(shared_from_this(), &plasticTauDot);
  }
  inline auto getPlasticStrainPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &plasticStrain);
  }
  inline auto getPlasticStrainDotPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(),
                                           &plasticStrainDot);
  }
  inline auto getPlasticFlowPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &plasticFlow);
  }
};
//! [Common data]

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;
FTensor::Index<'m', SPACE_DIM> m;
FTensor::Index<'n', SPACE_DIM> n;
FTensor::Index<'o', SPACE_DIM> o;
FTensor::Index<'p', SPACE_DIM> p;

FTensor::Index<'I', 3> I;
FTensor::Index<'J', 3> J;
FTensor::Index<'M', 3> M;
FTensor::Index<'N', 3> N;

FTensor::Index<'L', size_symm> L;
FTensor::Index<'O', size_symm> O;

//! [Operators definitions]
struct OpCalculatePlasticSurface : public DomainEleOp {
  OpCalculatePlasticSurface(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticity : public DomainEleOp {
  OpCalculatePlasticity(const std::string field_name,
                        boost::shared_ptr<CommonData> common_data_ptr,
                        boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpPlasticStress : public DomainEleOp {
  OpPlasticStress(const std::string field_name,
                  boost::shared_ptr<CommonData> common_data_ptr,
                  boost::shared_ptr<MatrixDouble> mDPtr,
                  const double scale = 1);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<MatrixDouble> mDPtr;
  const double scaleStress;
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticFlowRhs : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowRhs(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr,
                            boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculateConstraintsRhs : public AssemblyDomainEleOp {
  OpCalculateConstraintsRhs(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr,
                            boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculatePlasticInternalForceLhs_dEP : public AssemblyDomainEleOp {
  OpCalculatePlasticInternalForceLhs_dEP(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculatePlasticInternalForceLhs_LogStrain_dEP
    : public AssemblyDomainEleOp {
  OpCalculatePlasticInternalForceLhs_LogStrain_dEP(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<HenckyOps::CommonData> common_henky_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
  MatrixDouble locMat;
  MatrixDouble resDiff;
};

struct OpCalculatePlasticFlowLhs_dU : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr,
                               boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculatePlasticFlowLhs_LogStrain_dU : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_LogStrain_dU(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<HenckyOps::CommonData> comman_henky_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
  MatrixDouble resDiff;
};

struct OpCalculatePlasticFlowLhs_dEP : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dEP(const std::string row_field_name,
                                const std::string col_field_name,
                                boost::shared_ptr<CommonData> common_data_ptr,
                                boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculatePlasticFlowLhs_dTAU : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dTAU(const std::string row_field_name,
                                 const std::string col_field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr,
                                 boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculateConstraintsLhs_dU : public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_dU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr,
                               boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculateConstraintsLhs_LogStrain_dU : public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_LogStrain_dU(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<HenckyOps::CommonData> comman_henky_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
  MatrixDouble resDiff;
};

struct OpCalculateConstraintsLhs_dEP : public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_dEP(const std::string row_field_name,
                                const std::string col_field_name,
                                boost::shared_ptr<CommonData> common_data_ptr,
                                boost::shared_ptr<MatrixDouble> mat_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculateConstraintsLhs_dTAU : public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_dTAU(const std::string row_field_name,
                                 const std::string col_field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

//! [Operators definitions]

//! [Lambda functions]
inline auto diff_tensor() {
  FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_diff;
  constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
  t_diff(i, j, k, l) = (t_kd(i, k) ^ t_kd(j, l)) / 4.;
  return t_diff;
};

inline auto symm_L_tensor() {
  FTensor::Dg<double, SPACE_DIM, size_symm> t_L;
  t_L(i, j, L) = 0;
  if constexpr (SPACE_DIM == 2) {
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

inline auto diff_symmetrize() {
  FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM> t_diff;

  t_diff(i, j, k, l) = 0;
  t_diff(0, 0, 0, 0) = 1;
  t_diff(1, 1, 1, 1) = 1;

  t_diff(1, 0, 1, 0) = 0.5;
  t_diff(1, 0, 0, 1) = 0.5;

  t_diff(0, 1, 0, 1) = 0.5;
  t_diff(0, 1, 1, 0) = 0.5;

  if constexpr (SPACE_DIM == 3) {
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
inline double trace(FTensor::Tensor2_symmetric<T, SPACE_DIM> &t_stress) {
  constexpr double third = boost::math::constants::third<double>();
  if constexpr (SPACE_DIM == 2)
    return (t_stress(0, 0) + t_stress(1, 1)) * third;
  else
    return (t_stress(0, 0) + t_stress(1, 1) + t_stress(2, 2)) * third;
};

template <typename T>
inline auto deviator(FTensor::Tensor2_symmetric<T, SPACE_DIM> &t_stress,
                     double trace) {
  FTensor::Tensor2_symmetric<double, 3> t_dev;
  t_dev(I, J) = 0;
  for (int ii = 0; ii != SPACE_DIM; ++ii)
    for (int jj = ii; jj != SPACE_DIM; ++jj)
      t_dev(ii, jj) = t_stress(ii, jj);
  t_dev(0, 0) -= trace;
  t_dev(1, 1) -= trace;
  t_dev(2, 2) -= trace;
  return t_dev;
};

inline auto
diff_deviator(FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> &&t_diff_stress) {
  FTensor::Ddg<double, 3, SPACE_DIM> t_diff_deviator;
  t_diff_deviator(I, J, k, l) = 0;
  for (int ii = 0; ii != SPACE_DIM; ++ii)
    for (int jj = ii; jj != SPACE_DIM; ++jj)
      for (int kk = 0; kk != SPACE_DIM; ++kk)
        for (int ll = kk; ll != SPACE_DIM; ++ll)
          t_diff_deviator(ii, jj, kk, ll) = t_diff_stress(ii, jj, kk, ll);

  constexpr double third = boost::math::constants::third<double>();

  t_diff_deviator(0, 0, 0, 0) -= third;
  t_diff_deviator(0, 0, 1, 1) -= third;

  t_diff_deviator(1, 1, 0, 0) -= third;
  t_diff_deviator(1, 1, 1, 1) -= third;

  t_diff_deviator(2, 2, 0, 0) -= third;
  t_diff_deviator(2, 2, 1, 1) -= third;

  if constexpr (SPACE_DIM == 3) {
    t_diff_deviator(0, 0, 2, 2) -= third;
    t_diff_deviator(1, 1, 2, 2) -= third;
    t_diff_deviator(2, 2, 2, 2) -= third;
  }

  return t_diff_deviator;
};

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

inline auto plastic_flow(long double f,
                         FTensor::Tensor2_symmetric<double, 3> &&t_dev_stress,
                         FTensor::Ddg<double, 3, SPACE_DIM> &&t_diff_deviator) {
  FTensor::Tensor2_symmetric<double, SPACE_DIM> t_diff_f;
  t_diff_f(k, l) =
      (1.5 * (t_dev_stress(I, J) * t_diff_deviator(I, J, k, l))) / f;
  return t_diff_f;
};

template <typename T>
inline auto diff_plastic_flow_dstress(
    long double f, FTensor::Tensor2_symmetric<T, SPACE_DIM> &t_flow,
    FTensor::Ddg<double, 3, SPACE_DIM> &&t_diff_deviator) {
  FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_diff_flow;
  t_diff_flow(i, j, k, l) =
      (1.5 * (t_diff_deviator(M, N, i, j) * t_diff_deviator(M, N, k, l) -
              (2. / 3.) * t_flow(i, j) * t_flow(k, l))) /
      f;
  return t_diff_flow;
};

template <typename T>
inline auto diff_plastic_flow_dstrain(
    FTensor::Ddg<T, SPACE_DIM, SPACE_DIM> &t_D,
    FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> &&t_diff_plastic_flow_dstress) {
  FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_diff_flow;
  t_diff_flow(i, j, k, l) =
      t_diff_plastic_flow_dstress(i, j, m, n) * t_D(m, n, k, l);
  return t_diff_flow;
};

inline double constrain_diff_sign(double x) {
  const auto y = x / zeta;
  if (y > std::numeric_limits<float>::max_exponent10 ||
      y < std::numeric_limits<float>::min_exponent10) {
    return 0;
  } else {
    const auto e = std::exp(y);
    const auto ep1 = e + 1;
    return (2 / zeta) * (e / (ep1 * ep1));
  }
};

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

inline double w(double eqiv, double dot_tau, double f, double sigma_y) {
  return (f - sigma_y) / sigmaY + cn1 * (dot_tau * std::sqrt(eqiv));
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
                         double abs_w) {
  return visH * dot_tau + (sigmaY / 2) * ((cn0 * (dot_tau - eqiv) +
                                           cn1 * (std::sqrt(eqiv) * dot_tau) -
                                           (f - sigma_y) / sigmaY) -
                                          abs_w);
};

inline double diff_constrain_ddot_tau(double sign, double eqiv) {
  return visH + (sigmaY / 2) * (cn0 + cn1 * std::sqrt(eqiv) * (1 - sign));
};

inline double diff_constrain_eqiv(double sign, double eqiv, double dot_tau) {
  return (sigmaY / 2) *
         (-cn0 + cn1 * dot_tau * (0.5 / std::sqrt(eqiv)) * (1 - sign));
};

inline auto diff_constrain_df(double sign) { return (-1 - sign) / 2; };

inline auto diff_constrain_dsigma_y(double sign) { return (1 + sign) / 2; }

template <typename T>
inline auto diff_constrain_dstress(
    double diff_constrain_df,
    FTensor::Tensor2_symmetric<T, SPACE_DIM> &t_plastic_flow) {
  FTensor::Tensor2_symmetric<double, SPACE_DIM> t_diff_constrain_dstress;
  t_diff_constrain_dstress(i, j) = diff_constrain_df * t_plastic_flow(i, j);
  return t_diff_constrain_dstress;
};

template <typename T1, typename T2>
inline auto diff_constrain_dstrain(T1 &t_D, T2 &&t_diff_constrain_dstress) {
  FTensor::Tensor2_symmetric<double, SPACE_DIM> t_diff_constrain_dstrain;
  t_diff_constrain_dstrain(k, l) =
      t_diff_constrain_dstress(i, j) * t_D(i, j, k, l);
  return t_diff_constrain_dstrain;
};

template <typename T>
inline auto equivalent_strain_dot(
    FTensor::Tensor2_symmetric<T, SPACE_DIM> &t_plastic_strain_dot) {
  constexpr double A = 2. / 3;
  return std::sqrt(A * t_plastic_strain_dot(i, j) *
                   t_plastic_strain_dot(i, j)) +
         std::numeric_limits<double>::epsilon();
};

template <typename T1, typename T2, typename T3>
inline auto diff_equivalent_strain_dot(const T1 eqiv, T2 &t_plastic_strain_dot,
                                       T3 &t_diff_plastic_strain) {
  constexpr double A = 2. / 3;
  FTensor::Tensor2_symmetric<double, SPACE_DIM> t_diff_eqiv;
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

}; // namespace PlasticOps

#include <PlasticOpsGeneric.hpp>
#include <PlasticOpsSmallStrains.hpp>
#include <PlasticOpsLargeStrains.hpp>
#include <PlasticOpsMonitor.hpp>

namespace PlasticOps {

using Pip = boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator>;
using CommonPlasticPtr = boost::shared_ptr<PlasticOps::CommonData>;
using CommonHenkyPtr = boost::shared_ptr<HenckyOps::CommonData>;

struct OpFactory {

  static CommonPlasticPtr createCommonPlasticOps();
  static CommonHenkyPtr
  createCommonHekyOps(CommonPlasticPtr common_plastic_ptr);

  static MoFEMErrorCode addDomainBaseOps(Pip &pip,
                                         CommonPlasticPtr common_plastoc_ptr,
                                         CommonHenkyPtr common_henky_ptr,
                                         std::string ep, std::string u,
                                         std::string tau);
};

CommonPlasticPtr OpFactory::createCommonPlasticOps() {

  auto common_ptr = boost::make_shared<PlasticOps::CommonData>();

  auto set_common = [&]() {
    MoFEMFunctionBegin;
    auto make_d_mat = []() {
      return boost::make_shared<MatrixDouble>(size_symm * size_symm, 1);
    };

    auto set_matrial_stiffness = [&]() {
      MoFEMFunctionBegin;
      FTensor::Index<'i', SPACE_DIM> i;
      FTensor::Index<'j', SPACE_DIM> j;
      FTensor::Index<'k', SPACE_DIM> k;
      FTensor::Index<'l', SPACE_DIM> l;
      constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
      const double bulk_modulus_K =
          young_modulus / (3 * (1 - 2 * poisson_ratio));
      const double shear_modulus_G = young_modulus / (2 * (1 + poisson_ratio));

      // Plane stress or when 1, plane strain or 3d
      const double A = (SPACE_DIM == 2)
                           ? 2 * shear_modulus_G /
                                 (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                           : 1;

      auto t_D =
          getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*common_ptr->mDPtr);
      auto t_D_axiator = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
          *common_ptr->mDPtr_Axiator);
      auto t_D_deviator = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(
          *common_ptr->mDPtr_Deviator);

      constexpr double third = boost::math::constants::third<double>();
      t_D_axiator(i, j, k, l) = A *
                                (bulk_modulus_K - (2. / 3.) * shear_modulus_G) *
                                t_kd(i, j) * t_kd(k, l);
      t_D_deviator(i, j, k, l) =
          2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.);
      t_D(i, j, k, l) = t_D_axiator(i, j, k, l) + t_D_deviator(i, j, k, l);

      MoFEMFunctionReturn(0);
    };

    common_ptr = boost::make_shared<PlasticOps::CommonData>();
    common_ptr->mDPtr = make_d_mat();
    common_ptr->mDPtr_Axiator = make_d_mat();
    common_ptr->mDPtr_Deviator = make_d_mat();
    common_ptr->mGradPtr = boost::make_shared<MatrixDouble>();
    common_ptr->mStrainPtr = boost::make_shared<MatrixDouble>();
    common_ptr->mStressPtr = boost::make_shared<MatrixDouble>();

    CHKERR set_matrial_stiffness();

    MoFEMFunctionReturn(0);
  };

  CHK_THROW_MESSAGE(set_common(), "set common");

  return common_ptr;
};

CommonHenkyPtr
OpFactory::createCommonHekyOps(CommonPlasticPtr common_plastic_ptr) {
  CommonHenkyPtr common_henky_ptr;

  if (is_large_strains) {
    common_henky_ptr = boost::make_shared<HenckyOps::CommonData>();
    common_henky_ptr->matGradPtr = common_plastic_ptr->mGradPtr;
    common_henky_ptr->matDPtr = common_plastic_ptr->mDPtr;
    common_henky_ptr->matLogCPlastic =
        common_plastic_ptr->getPlasticStrainPtr();
    common_plastic_ptr->mStrainPtr = common_henky_ptr->getMatLogC();
    common_plastic_ptr->mStressPtr = common_henky_ptr->getMatHenckyStress();
  }

  return common_henky_ptr;
}

MoFEMErrorCode OpFactory::addDomainBaseOps(Pip &pip,
                                           CommonPlasticPtr common_plastic_ptr,
                                           CommonHenkyPtr common_henky_ptr,
                                           std::string u, std::string ep,
                                           std::string tau) {
  MoFEMFunctionBegin;

  pip.push_back(new OpCalculateScalarFieldValuesDot(
      tau, common_plastic_ptr->getPlasticTauDotPtr()));
  pip.push_back(new OpCalculateScalarFieldValues(
      tau, common_plastic_ptr->getPlasticTauPtr()));
  pip.push_back(new OpCalculateTensor2SymmetricFieldValues<SPACE_DIM>(
      ep, common_plastic_ptr->getPlasticStrainPtr()));
  pip.push_back(new OpCalculateTensor2SymmetricFieldValuesDot<SPACE_DIM>(
      ep, common_plastic_ptr->getPlasticStrainDotPtr()));
  pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
      u, common_plastic_ptr->mGradPtr));

  if (common_henky_ptr) {

    if (common_plastic_ptr->mGradPtr != common_henky_ptr->matGradPtr)
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "Wrong pointer for grad");

    using namespace HenckyOps;

    pip.push_back(new OpCalculateEigenVals<SPACE_DIM>(u, common_henky_ptr));
    pip.push_back(new OpCalculateLogC<SPACE_DIM>(u, common_henky_ptr));
    pip.push_back(new OpCalculateLogC_dC<SPACE_DIM>(u, common_henky_ptr));
    pip.push_back(new OpCalculateHenckyPlasticStress<SPACE_DIM>(
        u, common_henky_ptr, common_plastic_ptr->mDPtr));
    pip.push_back(new OpCalculatePiolaStress<SPACE_DIM>(u, common_henky_ptr));

  } else {
    pip.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
        u, common_plastic_ptr->mGradPtr, common_plastic_ptr->mStrainPtr));
    pip.push_back(new OpPlasticStress(u, common_plastic_ptr,
                                      common_plastic_ptr->mDPtr, 1));
  }

  pip.push_back(new OpCalculatePlasticSurface(u, common_plastic_ptr));
  pip.push_back(new OpCalculatePlasticity(u, common_plastic_ptr,
                                          common_plastic_ptr->mDPtr));

  MoFEMFunctionReturn(0);
};

template <typename DomainEleOp, AssemblyType A, IntegrationType I>
MoFEMErrorCode opFactoryDomainRhs(Pip &pip, std::string u, std::string ep,
                                  std::string tau) {
  MoFEMFunctionBegin;

  using B = typename FormsIntegrators<DomainEleOp>::template Assembly<
      A>::template LinearForm<I>;
  using OpInternalForceCauchy =
      typename B::template OpGradTimesSymTensor<1, SPACE_DIM, SPACE_DIM>;
  using OpInternalForcePiola =
      typename B::template OpGradTimesTensor<1, SPACE_DIM, SPACE_DIM>;

  auto common_plastic_ptr = OpFactory::createCommonPlasticOps();
  auto common_henky_ptr = OpFactory::createCommonHekyOps(common_plastic_ptr);

  CHKERR OpFactory::addDomainBaseOps(pip, common_plastic_ptr, common_henky_ptr,
                                     u, ep, tau);

  auto m_D_ptr = common_plastic_ptr->mDPtr;

  // Calculate internal forces
  if (common_henky_ptr) {
    pip.push_back(new OpInternalForcePiola(
        u, common_henky_ptr->getMatFirstPiolaStress()));
  } else {
    pip.push_back(new OpInternalForceCauchy(u, common_plastic_ptr->mStressPtr));
  }

  pip.push_back(
      new OpCalculateConstraintsRhs(tau, common_plastic_ptr, m_D_ptr));
  pip.push_back(new OpCalculatePlasticFlowRhs(ep, common_plastic_ptr, m_D_ptr));

  MoFEMFunctionReturn(0);
}

template <typename DomainEleOp, AssemblyType A, IntegrationType I>
MoFEMErrorCode opFactoryDomainLhs(Pip &pip, std::string u, std::string ep,
                                  std::string tau) {
  MoFEMFunctionBegin;

  using namespace HenckyOps;

  using B = typename FormsIntegrators<DomainEleOp>::template Assembly<
      A>::template BiLinearForm<I>;
  using OpKPiola =
      typename B::template OpGradTensorGrad<1, SPACE_DIM, SPACE_DIM, 1>;
  using OpKCauchy =
      typename B::template OpGradSymTensorGrad<1, SPACE_DIM, SPACE_DIM, 0>;

  auto common_plastic_ptr = OpFactory::createCommonPlasticOps();
  auto common_henky_ptr = OpFactory::createCommonHekyOps(common_plastic_ptr);

  CHKERR OpFactory::addDomainBaseOps(pip, common_plastic_ptr, common_henky_ptr,
                                     u, ep, tau);

  auto m_D_ptr = common_plastic_ptr->mDPtr;

  if (common_henky_ptr) {
    pip.push_back(new OpHenckyTangent<SPACE_DIM>(u, common_henky_ptr,
                                                 common_plastic_ptr->mDPtr));
    pip.push_back(new OpKPiola(u, u, common_henky_ptr->getMatTangent()));
    pip.push_back(new OpCalculatePlasticInternalForceLhs_LogStrain_dEP(
        u, ep, common_plastic_ptr, common_henky_ptr,
        common_plastic_ptr->mDPtr));
  } else {
    pip.push_back(new OpKCauchy(u, u, m_D_ptr));
    pip.push_back(new OpCalculatePlasticInternalForceLhs_dEP(
        u, ep, common_plastic_ptr, m_D_ptr));
  }

  if (common_henky_ptr) {
    pip.push_back(new OpCalculateConstraintsLhs_LogStrain_dU(
        tau, u, common_plastic_ptr, common_henky_ptr, m_D_ptr));
    pip.push_back(new OpCalculatePlasticFlowLhs_LogStrain_dU(
        ep, u, common_plastic_ptr, common_henky_ptr, m_D_ptr));
  } else {
    pip.push_back(
        new OpCalculateConstraintsLhs_dU(tau, u, common_plastic_ptr, m_D_ptr));
    pip.push_back(
        new OpCalculatePlasticFlowLhs_dU(ep, u, common_plastic_ptr, m_D_ptr));
  }

  pip.push_back(
      new OpCalculatePlasticFlowLhs_dEP(ep, ep, common_plastic_ptr, m_D_ptr));
  pip.push_back(
      new OpCalculatePlasticFlowLhs_dTAU(ep, tau, common_plastic_ptr, m_D_ptr));
  pip.push_back(
      new OpCalculateConstraintsLhs_dEP(tau, ep, common_plastic_ptr, m_D_ptr));
  pip.push_back(
      new OpCalculateConstraintsLhs_dTAU(tau, tau, common_plastic_ptr));

  MoFEMFunctionReturn(0);
}

} // namespace PlasticOps
