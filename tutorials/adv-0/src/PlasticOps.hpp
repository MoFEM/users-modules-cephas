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
  VectorDouble tempVal;

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
  inline auto getTempValPtr() {
    return boost::shared_ptr<VectorDouble>(shared_from_this(), &tempVal);
  }
};
//! [Common data]

FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;
FTensor::Index<'m', SPACE_DIM> m;
FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'n', SPACE_DIM> n;

FTensor::Index<'I', 3> I;
FTensor::Index<'J', 3> J;
FTensor::Index<'M', 3> M;
FTensor::Index<'N', 3> N;

FTensor::Index<'L', (SPACE_DIM * (SPACE_DIM + 1)) / 2> L;

//! [Operators definitions]
struct OpCalculatePlasticSurface : public DomainEleOp {
  OpCalculatePlasticSurface(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
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
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateContrainsRhs : public AssemblyDomainEleOp {
  OpCalculateContrainsRhs(const std::string field_name,
                          boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticInternalForceLhs_dEP : public AssemblyDomainEleOp {
  OpCalculatePlasticInternalForceLhs_dEP(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

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
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
  MatrixDouble locMat;
};

struct OpCalculatePlasticFlowLhs_dU : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr,
                               boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

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
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculatePlasticFlowLhs_dEP : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dEP(const std::string row_field_name,
                                const std::string col_field_name,
                                boost::shared_ptr<CommonData> common_data_ptr,
                                boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculatePlasticFlowLhs_dEP_ALE : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dEP_ALE(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr,
      boost::shared_ptr<MatrixDouble> velocity_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<MatrixDouble> velocityPtr;
};

struct OpCalculatePlasticFlowLhs_dTAU : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dTAU(const std::string row_field_name,
                                 const std::string col_field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticFlowLhs_dTAU_ALE
    : public OpCalculatePlasticFlowLhs_dEP_ALE {
  using OpCalculatePlasticFlowLhs_dEP_ALE::OpCalculatePlasticFlowLhs_dEP_ALE;
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);
};

struct OpCalculateConstrainsLhs_dTAU_ALE
    : public OpCalculatePlasticFlowLhs_dEP_ALE {
  using OpCalculatePlasticFlowLhs_dEP_ALE::OpCalculatePlasticFlowLhs_dEP_ALE;
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);
};

struct OpCalculateContrainsLhs_dU : public AssemblyDomainEleOp {
  OpCalculateContrainsLhs_dU(const std::string row_field_name,
                             const std::string col_field_name,
                             boost::shared_ptr<CommonData> common_data_ptr,
                             boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculateContrainsLhs_LogStrain_dU : public AssemblyDomainEleOp {
  OpCalculateContrainsLhs_LogStrain_dU(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<HenckyOps::CommonData> comman_henky_data_ptr,
      boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> commonHenckyDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculateContrainsLhs_dEP : public AssemblyDomainEleOp {
  OpCalculateContrainsLhs_dEP(const std::string row_field_name,
                              const std::string col_field_name,
                              boost::shared_ptr<CommonData> common_data_ptr,
                              boost::shared_ptr<MatrixDouble> mat_D_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpCalculateContrainsLhs_dTAU : public AssemblyDomainEleOp {
  OpCalculateContrainsLhs_dTAU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);
private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPostProcPlastic : public DomainEleOp {
  OpPostProcPlastic(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
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
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  FTensor::Dg<double, SPACE_DIM, size_symm> t_L;
  t_L(i, j, L) = 0;
  if (SPACE_DIM == 2) {
    t_L(0, 0, 0) = 1;
    t_L(1, 0, 1) = 1;
    t_L(1, 1, 2) = 1;
  } else if (SPACE_DIM == 3) {
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

  if (SPACE_DIM == 3) {
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
  if (SPACE_DIM == 2)
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

  if (SPACE_DIM == 3) {
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

inline double constrain_abs(long double x) {
  return std::abs(x);
};

inline double constrian_sign(long double x) {
  if (x > 0)
    return 1;
  else if (x < 0)
    return -1;
  else
    return 0;
};

inline double k_penalty(const double tau) {
  return 0;
  constexpr auto eps = std::numeric_limits<float>::epsilon();
  constexpr auto inv_eps = 1. / eps;
  return tau > -eps ? 0 : inv_eps;
}

inline double cut_off(const double tau) {
  if (is_cut_off) 
    return (tau + std::abs(tau)) / 2.;
  return tau; // comment
}

inline unsigned int cut_off_dtau(const double tau) {
  constexpr auto eps = std::numeric_limits<float>::epsilon();
  constexpr auto inv_eps = 1. / eps;
  if (is_cut_off)
    return tau > -std::numeric_limits<double>::epsilon() ? 1 : 0;
  return 1;
};

inline double w(long double dot_tau, long double f, long double sigma_y) {
  return (f - sigma_y) / sigmaY + cn * dot_tau;
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
inline double constrain(long double dot_tau, long double f,
                        long double sigma_y) {
  return visH * dot_tau +
         (sigmaY / 2) * ((cn * dot_tau - (f - sigma_y) / sigmaY) -
                         constrain_abs(w(dot_tau, f, sigma_y)));
};

inline double diff_constrain_ddot_tau(long double dot_tau, long double f,
                                      long double sigma_y) {
  return visH +
         (sigmaY / 2) * (cn - cn * constrian_sign(w(dot_tau, f, sigma_y)));
};

inline auto diff_constrain_df(long double dot_tau, long double f,
                              long double sigma_y) {
  return (-1 - constrian_sign(w(dot_tau, f, sigma_y))) / 2;
};

inline auto diff_constrain_dsigma_y(long double dot_tau, long double f,
                                    long double sigma_y) {
  return (1 + constrian_sign(w(dot_tau, f, sigma_y))) / 2;
}

template <typename T>
inline auto diff_constrain_dstress(
    double &&diff_constrain_df,
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
//! [Lambda functions]

//! [Auxiliary functions functions
static inline FTensor::Tensor3<FTensor::PackPtr<double *, 2>, 2, 2, 2>
get_mat_tensor_sym_dvector(size_t rr, MatrixDouble &mat, FTensor::Number<2>) {
  return FTensor::Tensor3<FTensor::PackPtr<double *, 2>, 2, 2, 2>{
      &mat(3 * rr + 0, 0), &mat(3 * rr + 0, 1), &mat(3 * rr + 1, 0),
      &mat(3 * rr + 1, 1), &mat(3 * rr + 1, 0), &mat(3 * rr + 1, 1),
      &mat(3 * rr + 2, 0), &mat(3 * rr + 2, 1)};
}

static inline FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3>
get_mat_tensor_sym_dvector(size_t rr, MatrixDouble &mat, FTensor::Number<3>) {
  return FTensor::Tensor3<FTensor::PackPtr<double *, 3>, 3, 3, 3>{
      &mat(6 * rr + 0, 0), &mat(6 * rr + 0, 1), &mat(6 * rr + 0, 2),
      &mat(6 * rr + 1, 0), &mat(6 * rr + 1, 1), &mat(6 * rr + 1, 2),
      &mat(6 * rr + 2, 0), &mat(6 * rr + 2, 1), &mat(6 * rr + 2, 2),
      &mat(6 * rr + 1, 0), &mat(6 * rr + 1, 1), &mat(6 * rr + 1, 2),
      &mat(6 * rr + 3, 0), &mat(6 * rr + 3, 1), &mat(6 * rr + 3, 2),
      &mat(6 * rr + 4, 0), &mat(6 * rr + 4, 1), &mat(6 * rr + 4, 2),
      &mat(6 * rr + 2, 0), &mat(6 * rr + 2, 1), &mat(6 * rr + 2, 2),
      &mat(6 * rr + 4, 0), &mat(6 * rr + 4, 1), &mat(6 * rr + 4, 2),
      &mat(6 * rr + 5, 0), &mat(6 * rr + 5, 1), &mat(6 * rr + 5, 2)};
}

static inline auto get_mat_scalar_dvector(MatrixDouble &mat,
                                          FTensor::Number<2>) {
  return FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2>{&mat(0, 0),
                                                            &mat(0, 1)};
}

static inline auto get_mat_scalar_dvector(MatrixDouble &mat,
                                          FTensor::Number<3>) {
  return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>{
      &mat(0, 0), &mat(0, 1), &mat(0, 2)};
}
//! [Auxiliary functions functions


}; // namespace PlasticOps

#include <PlasticOpsGeneric.hpp>
#include<PlasticOpsSmallStrains.hpp> 
#include<PlasticOpsLargeStrains.hpp>
#include <PlasticOpsMonitor.hpp>
