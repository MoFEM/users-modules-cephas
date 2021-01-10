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
\forall \delta\tau \in L^2(\Omega) \end{array} \right. \f]

*/

namespace PlasticOps {

//! [Common data]
struct CommonData {
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;

  boost::shared_ptr<VectorDouble> plasticSurfacePtr;
  boost::shared_ptr<MatrixDouble> plasticFlowPtr;
  boost::shared_ptr<VectorDouble> plasticTauPtr;
  boost::shared_ptr<VectorDouble> plasticTauDotPtr;
  boost::shared_ptr<MatrixDouble> plasticStrainPtr;
  boost::shared_ptr<MatrixDouble> plasticStrainDotPtr;
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

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPlasticStress : public DomainEleOp {
  OpPlasticStress(const std::string field_name,
                  boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticFlowRhs : public DomainEleOp {
  OpCalculatePlasticFlowRhs(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateContrainsRhs : public DomainEleOp {
  OpCalculateContrainsRhs(const std::string field_name,
                          boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticInternalForceLhs_dEP : public DomainEleOp {
  OpCalculatePlasticInternalForceLhs_dEP(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculatePlasticInternalForceLhs_LogStrain_dEP : public DomainEleOp {
  OpCalculatePlasticInternalForceLhs_LogStrain_dEP(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<HenkyOps::CommonData> common_henky_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenkyOps::CommonData> commonHenkyDataPtr;
  MatrixDouble locMat;
};

struct OpCalculatePlasticFlowLhs_dU : public DomainEleOp {
  OpCalculatePlasticFlowLhs_dU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculatePlasticFlowLhs_LogStrain_dU : public DomainEleOp {
  OpCalculatePlasticFlowLhs_LogStrain_dU(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<HenkyOps::CommonData> comman_henky_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenkyOps::CommonData> commonHenkyDataPtr;
  MatrixDouble locMat;
};

struct OpCalculatePlasticFlowLhs_dEP : public DomainEleOp {
  OpCalculatePlasticFlowLhs_dEP(const std::string row_field_name,
                                const std::string col_field_name,
                                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculatePlasticFlowLhs_dTAU : public DomainEleOp {
  OpCalculatePlasticFlowLhs_dTAU(const std::string row_field_name,
                                 const std::string col_field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculateContrainsLhs_dU : public DomainEleOp {
  OpCalculateContrainsLhs_dU(const std::string row_field_name,
                             const std::string col_field_name,
                             boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculateContrainsLhs_LogStrain_dU : public DomainEleOp {
  OpCalculateContrainsLhs_LogStrain_dU(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<HenkyOps::CommonData> comman_henky_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenkyOps::CommonData> commonHenkyDataPtr;
  MatrixDouble locMat;
};

struct OpCalculateContrainsLhs_dEP : public DomainEleOp {
  OpCalculateContrainsLhs_dEP(const std::string row_field_name,
                              const std::string col_field_name,
                              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculateContrainsLhs_dTAU : public DomainEleOp {
  OpCalculateContrainsLhs_dTAU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
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
  // t_diff(i, j, k, l) = 0;
  // for (size_t ii = 0; ii != 2; ++ii)
  //   for (size_t jj = ii; jj != 2; ++jj)
  //     t_diff(ii, jj, ii, jj) = 1;
  constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
  t_diff(i, j, k, l) = (t_kd(i, k) ^ t_kd(j, l)) / 4.;
  return t_diff;
};

inline auto symm_L_tensor() {
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  FTensor::Dg<double, SPACE_DIM, size_symm> L;
  L(i, j, k) = 0;
  L(0, 0, 0) = 1;
  L(1, 0, 1) = 1;
  L(0, 1, 1) = 1;
  L(1, 1, 2) = 1;
  return L;
}

inline auto diff_symmetrize() {
  FTensor::Tensor4<double, 2, 2, 2, 2> t_diff;

  t_diff(i, j, k, l) = 0;
  t_diff(0, 0, 0, 0) = 1;
  t_diff(1, 1, 1, 1) = 1;
  
  t_diff(1, 0, 1, 0) = 0.5;
  t_diff(1, 0, 0, 1) = 0.5;

  t_diff(0, 1, 0, 1) = 0.5;
  t_diff(0, 1, 1, 0) = 0.5;
  
  return t_diff;
};

template <typename T>
inline double trace(FTensor::Tensor2_symmetric<T, 2> &t_stress) {
  constexpr double third = boost::math::constants::third<double>();
  return (t_stress(0, 0) + t_stress(1, 1)) * third;
};

template <typename T>
inline auto deviator(FTensor::Tensor2_symmetric<T, 2> &t_stress, double trace) {
  FTensor::Tensor2_symmetric<double, 3> t_dev;
  t_dev(I, J) = 0;
  for (int ii = 0; ii != 2; ++ii)
    for (int jj = ii; jj != 2; ++jj)
      t_dev(ii, jj) = t_stress(ii, jj);
  t_dev(0, 0) -= trace;
  t_dev(1, 1) -= trace;
  t_dev(2, 2) -= trace;
  return t_dev;
};

inline auto diff_deviator(FTensor::Ddg<double, 2, 2> &&t_diff_stress) {
  FTensor::Ddg<double, 3, 2> t_diff_deviator;
  t_diff_deviator(I, J, k, l) = 0;
  for (int ii = 0; ii != 2; ++ii)
    for (int jj = ii; jj != 2; ++jj)
      for (int kk = 0; kk != 2; ++kk)
        for (int ll = kk; ll != 2; ++ll)
          t_diff_deviator(ii, jj, kk, ll) = t_diff_stress(ii, jj, kk, ll);

  constexpr double third = boost::math::constants::third<double>();

  t_diff_deviator(0, 0, 0, 0) -= third;
  t_diff_deviator(0, 0, 1, 1) -= third;

  t_diff_deviator(1, 1, 0, 0) -= third;
  t_diff_deviator(1, 1, 1, 1) -= third;

  t_diff_deviator(2, 2, 0, 0) -= third;
  t_diff_deviator(2, 2, 1, 1) -= third;

  return t_diff_deviator;
};

inline auto hardening(double tau) { return H * tau + sigmaY; }

inline auto hardening_dtau() { return H; }

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
  return std::sqrt(1.5 * t_stress_deviator(I, J) * t_stress_deviator(I, J));
};

inline auto plastic_flow(double f,
                         FTensor::Tensor2_symmetric<double, 3> &&t_dev_stress,
                         FTensor::Ddg<double, 3, 2> &&t_diff_deviator) {
  FTensor::Tensor2_symmetric<double, 2> t_diff_f;
  if (std::abs(f) > std::numeric_limits<double>::epsilon())
    t_diff_f(k, l) =
        (1.5 / f) * (t_dev_stress(I, J) * t_diff_deviator(I, J, k, l));
  else
    t_diff_f(k, l) = 0;
  return t_diff_f;
};

template <typename T>
inline auto
diff_plastic_flow_dstress(double f, FTensor::Tensor2_symmetric<T, 2> &t_flow,
                          FTensor::Ddg<double, 3, 2> &&t_diff_deviator) {
  FTensor::Ddg<double, 2, 2> t_diff_flow;
  if (std::abs(f) > std::numeric_limits<double>::epsilon())
    t_diff_flow(i, j, k, l) =
        (1.5 / f) * (t_diff_deviator(M, N, i, j) * t_diff_deviator(M, N, k, l) -
                     (2. / 3.) * t_flow(i, j) * t_flow(k, l));
  else
    t_diff_flow(i, j, k, l) = 0;

  return t_diff_flow;
};

template <typename T>
inline auto diff_plastic_flow_dstrain(
    FTensor::Ddg<T, 2, 2> &t_D,
    FTensor::Ddg<double, 2, 2> &&t_diff_plastic_flow_dstress) {
  FTensor::Ddg<double, 2, 2> t_diff_flow;
  t_diff_flow(i, j, k, l) =
      t_diff_plastic_flow_dstress(i, j, m, n) * t_D(m, n, k, l);

  return t_diff_flow;
};

/**

\f[
\dot{\tau} - \frac{1}{2}\left\{\dot{\tau} + (f(\pmb\sigma) - \sigma_y) +
\| \dot{\tau} + (f(\pmb\sigma) - \sigma_y) \|\right\} = 0
\f]

 */
inline double contrains(double tau, double f){
  return (cn * tau - f) - std::abs(cn * tau + f);
};

inline double sign(double x) {
  if (x == 0)
    return 0;
  else if (x > 0)
    return 1;
  else
    return -1;
};

inline double diff_constrain_dtau(double tau, double f) {
  return (cn - cn * sign(f + cn * tau));
};

inline auto diff_constrain_df(double tau, double f) {
  return (-1 - sign(f + cn * tau));
};

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

OpCalculatePlasticSurface::OpCalculatePlasticSurface(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

MoFEMErrorCode OpCalculatePlasticSurface::doWork(int side, EntityType type,
                                                 EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mStressPtr->size2();
  auto t_stress = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStressPtr));

  commonDataPtr->plasticSurfacePtr->resize(nb_gauss_pts, false);
  commonDataPtr->plasticFlowPtr->resize(3, nb_gauss_pts, false);
  auto t_flow =
      getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));

  for (auto &f : *(commonDataPtr->plasticSurfacePtr)) {
    f = platsic_surface(deviator(t_stress, trace(t_stress)));
    auto t_flow_tmp = plastic_flow(f,

                                   deviator(t_stress, trace(t_stress)),

                                   diff_deviator(diff_tensor()));
    t_flow(i, j) = t_flow_tmp(i, j);
    ++t_flow;
    ++t_stress;
  }

  MoFEMFunctionReturn(0);
}

OpPlasticStress::OpPlasticStress(const std::string field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Calculate stress]
MoFEMErrorCode OpPlasticStress::doWork(int side, EntityType type,
                                       EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  commonDataPtr->mStressPtr->resize(3, nb_gauss_pts);
  auto t_D =
      getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
  auto t_strain = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStrainPtr));
  auto t_plastic_strain =
      getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticStrainPtr));
  auto t_stress = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStressPtr));

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

OpCalculatePlasticFlowRhs::OpCalculatePlasticFlowRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculatePlasticFlowRhs::doWork(int side, EntityType type,
                                                 EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();

    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));
    auto t_plastic_strain_dot =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticStrainDotPtr));
    auto t_tau_dot = getFTensor0FromVec(*(commonDataPtr->plasticTauDotPtr));

    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);

    const size_t nb_integration_pts = data.getN().size1();
    const size_t nb_base_functions = data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = dt * getMeasure() * t_w;

      FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 2, 2> t_nf{
          &nf[0], &nf[1], &nf[1], &nf[2]};

      FTensor::Tensor2_symmetric<double, 2> t_flow_stress_diff;
      t_flow_stress_diff(i, j) = t_D(i, j, k, l) * (t_plastic_strain_dot(k, l) -
                                                    t_tau_dot * t_flow(k, l));

      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i, j) += (alpha * t_base) * t_flow_stress_diff(i, j);
        ++t_base;
        ++t_nf;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_flow;
      ++t_plastic_strain_dot;
      ++t_tau_dot;
      ++t_w;
    }

    CHKERR VecSetValues(getTSf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsRhs::OpCalculateContrainsRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculateContrainsRhs::doWork(int side, EntityType type,
                                               EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();

    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_tau_dot = getFTensor0FromVec(*(commonDataPtr->plasticTauDotPtr));
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_w = getFTensor0IntegrationWeight();

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_base = data.getFTensor0N();
    const size_t nb_integration_pts = data.getN().size1();
    const size_t nb_base_functions = data.getN().size2();
    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha = dt * getMeasure() * t_w;
      const double beta = alpha * contrains(t_tau_dot, t_f - hardening(t_tau));

      size_t bb = 0;
      for (; bb != nb_dofs; ++bb) {
        nf[bb] += beta * t_base;
        ++t_base;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_tau;
      ++t_tau_dot;
      ++t_f;
      ++t_w;
    }

    CHKERR VecSetValues(getTSf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticInternalForceLhs_dEP::OpCalculatePlasticInternalForceLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculatePlasticInternalForceLhs_dEP::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_diff_base = row_data.getFTensor1DiffN<2>();
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != nb_row_dofs / 2; ++rr) {

        FTensor::Christof<FTensor::PackPtr<double *, 3>, 2, 2> t_mat{

            &locMat(2 * rr + 0, 0), &locMat(2 * rr + 0, 1),
            &locMat(2 * rr + 0, 2),

            &locMat(2 * rr + 1, 0), &locMat(2 * rr + 1, 1),
            &locMat(2 * rr + 1, 2)

        };

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 3; ++cc) {

          // I mix up the indices here so that it behaves like a
          // Dg.  That way I don't have to have a separate wrapper
          // class Christof_Expr, which simplifies things.
          // You cyclicly has to shift index, i, j, k -> l, i, j
          FTensor::Christof<double, 2, 2> t_tmp;
          t_tmp(l, i, k) =
              (t_D(i, j, k, l) * ((alpha * t_col_base) * t_row_diff_base(j)));

          for (int ii = 0; ii != 2; ++ii)
            for (int kk = 0; kk != 2; ++kk)
              for (int ll = 0; ll != 2; ++ll)
                t_mat(ii, kk, ll) -= t_tmp(ii, kk, ll);

          ++t_mat;
          ++t_col_base;
        }

        ++t_row_diff_base;
      }

      for (; rr < nb_row_base_functions; ++rr)
        ++t_row_diff_base;

      ++t_w;
    }

    MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                 ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticInternalForceLhs_LogStrain_dEP::
    OpCalculatePlasticInternalForceLhs_LogStrain_dEP(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr,
        boost::shared_ptr<HenkyOps::CommonData> common_henky_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr),
      commonHenkyDataPtr(common_henky_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculatePlasticInternalForceLhs_LogStrain_dEP::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_diff_base = row_data.getFTensor1DiffN<2>();
    // auto dP_dF =
    //     getFTensor4FromMat<SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM, 1>(
    //         commonHenkyDataPtr->matTangent);

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
    auto t_logC_dC = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
        commonHenkyDataPtr->matLogCdC);
    auto t_grad = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(
        *(commonHenkyDataPtr->matGradPtr));

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
      t_F(i, j) = t_grad(i, j) + t_kd(i, j);

      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          t_DLogC_dC;
      t_DLogC_dC(i, j, k, l) = 0;
      for (int ii = 0; ii != SPACE_DIM; ++ii)
        for (int jj = 0; jj != SPACE_DIM; ++jj)
          for (int kk = 0; kk != SPACE_DIM; ++kk)
            for (int ll = 0; ll != SPACE_DIM; ++ll)
              for (int mm = 0; mm != SPACE_DIM; ++mm)
                for (int nn = 0; nn != SPACE_DIM; ++nn)
                  t_DLogC_dC(ii, jj, kk, ll) +=
                      t_D(mm, nn, kk, ll) * t_logC_dC(mm, nn, ii, jj);

      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          t_FDLogC_dC;
      t_FDLogC_dC(i, j, k, l) = t_F(i, m) * t_DLogC_dC(m, j, k, l);

      size_t rr = 0;
      for (; rr != nb_row_dofs / 2; ++rr) {

        FTensor::Christof<FTensor::PackPtr<double *, 3>, 2, 2> t_mat{

            &locMat(2 * rr + 0, 0), &locMat(2 * rr + 0, 1),
            &locMat(2 * rr + 0, 2),

            &locMat(2 * rr + 1, 0), &locMat(2 * rr + 1, 1),
            &locMat(2 * rr + 1, 2)

        };

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 3; ++cc) {

          FTensor::Tensor3<double, SPACE_DIM, SPACE_DIM, SPACE_DIM> t_tmp;
          t_tmp(i, k, l) = t_FDLogC_dC(i, j, k, l) *
                           ((alpha * t_col_base) * t_row_diff_base(j));

          for (int ii = 0; ii != SPACE_DIM; ++ii)
            for (int kk = 0; kk != SPACE_DIM; ++kk)
              for (int ll = 0; ll != SPACE_DIM; ++ll)
                t_mat(ii, kk, ll) -= t_tmp(ii, kk, ll);

          ++t_mat;
          ++t_col_base;
        }

        ++t_row_diff_base;
      }

      for (; rr < nb_row_base_functions; ++rr)
        ++t_row_diff_base;

      ++t_w;
      // ++dP_dF;
      ++t_logC_dC;
      ++t_grad;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dU::OpCalculatePlasticFlowLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculatePlasticFlowLhs_dU::doWork(int row_side, int col_side,
                                                    EntityType row_type,
                                                    EntityType col_type,
                                                    EntData &row_data,
                                                    EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau_dot = getFTensor0FromVec(*(commonDataPtr->plasticTauDotPtr));
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
    auto t_diff_symmetrize = diff_symmetrize();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

      double alpha = dt * getMeasure() * t_w;
      auto t_diff_plastic_flow_dstrain = diff_plastic_flow_dstrain(
          t_D,
          diff_plastic_flow_dstress(t_f, t_flow, diff_deviator(diff_tensor())));
      FTensor::Ddg<double, 2, 2> t_flow_stress_dstrain;
      t_flow_stress_dstrain(i, j, k, l) =
          t_D(i, j, m, n) * t_diff_plastic_flow_dstrain(m, n, k, l);

      FTensor::Tensor4<double, 2, 2, 2, 2> t_diff_plastic_flow_stress_dgrad;
      t_diff_plastic_flow_stress_dgrad(i, j, k, l) =
          t_flow_stress_dstrain(i, j, m, n) * t_diff_symmetrize(m, n, k, l);

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        // Tensor3(T * d000, T * d001, T * d010, T * d011, T * d100, T * d101,
                // T * d110, T * d111)

        FTensor::Tensor3<FTensor::PackPtr<double *, 2>, 2, 2, 2> t_mat{
            &locMat(3 * rr + 0, 0), // 000
            &locMat(3 * rr + 0, 1), // 001

            &locMat(3 * rr + 1, 0), // 010
            &locMat(3 * rr + 1, 1), // 011

            &locMat(3 * rr + 1, 0), // 100
            &locMat(3 * rr + 1, 1), // 101

            &locMat(3 * rr + 2, 0), // 110
            &locMat(3 * rr + 2, 1)  // 111

        };

        const double c0 = alpha * t_row_base * t_tau_dot;

        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 2; ++cc) {

          FTensor::Tensor3<double, 2, 2, 2> t_tmp;
          t_mat(i, j, l) -= c0 * (t_diff_plastic_flow_stress_dgrad(i, j, l, k) *
                                  t_col_diff_base(k));

          ++t_mat;
          ++t_col_diff_base;
        }

        ++t_row_base;
      }
      for (; rr < nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_f;
      ++t_flow;
      ++t_tau_dot;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_LogStrain_dU::OpCalculatePlasticFlowLhs_LogStrain_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<HenkyOps::CommonData> comman_henky_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr),
      commonHenkyDataPtr(comman_henky_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculatePlasticFlowLhs_LogStrain_dU::doWork(int row_side, int col_side,
                                                    EntityType row_type,
                                                    EntityType col_type,
                                                    EntData &row_data,
                                                    EntData &col_data) {
  MoFEMFunctionBegin;

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau_dot = getFTensor0FromVec(*(commonDataPtr->plasticTauDotPtr));
    auto t_flow = getFTensor2SymmetricFromMat<SPACE_DIM>(
        *(commonDataPtr->plasticFlowPtr));
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);

    auto t_grad =
        getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*(commonDataPtr->mGradPtr));
    auto t_logC_dC = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
        commonHenkyDataPtr->matLogCdC);

    auto t_diff_symmetrize = diff_symmetrize();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

      double alpha = dt * getMeasure() * t_w;

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
      t_F(i, j) = t_grad(i, j) + t_kd(i, j);
      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          dC_dF;
      dC_dF(i, j, k, l) = (t_kd(i, l) * t_F(k, j)) + (t_kd(j, l) * t_F(k, i));

      auto t_diff_plastic_flow_dstrain = diff_plastic_flow_dstrain(
          t_D,
          diff_plastic_flow_dstress(t_f, t_flow, diff_deviator(diff_tensor())));
      FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_flow_stress_dlogC;
      t_flow_stress_dlogC(i, j, k, l) =
          t_D(i, j, m, n) * t_diff_plastic_flow_dstrain(m, n, k, l);

      // FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
      //     t_diff_plastic_flow_stress_diff_symm;
      // t_diff_plastic_flow_stress_diff_symm(i, j, k, l) =
      //     t_flow_stress_dlogC(i, j, m, n) * t_diff_symmetrize(m, n, k, l);

      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          t_diff_plastic_flow_stress_dC;
      t_diff_plastic_flow_stress_dC(i, j, m, n) =
          t_diff_plastic_flow_dstrain(i, j, m, n) *
          (t_logC_dC(m, n, k, l) / 2);

      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          t_diff_plastic_flow_stress_dgrad;
      t_diff_plastic_flow_stress_dgrad(i, j, k, l) = 0;
      for (int ii = 0; ii != SPACE_DIM; ++ii)
        for (int jj = 0; jj != SPACE_DIM; ++jj)
          for (int kk = 0; kk != SPACE_DIM; ++kk)
            for (int ll = 0; ll != SPACE_DIM; ++ll)
              for (int mm = 0; mm != SPACE_DIM; ++mm)
                for (int nn = 0; nn != SPACE_DIM; ++nn)
                  t_diff_plastic_flow_stress_dgrad(ii, jj, kk, ll) +=
                      t_diff_plastic_flow_stress_dC(ii, jj, mm, nn) *
                      dC_dF(mm, nn, kk, ll);

      // t_diff_plastic_flow_stress_dgrad(i, j, k, l) =
      //     t_diff_plastic_flow_stress_dC(i, j, m, n) * dC_dF(m, n, k, l);

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        
        // Tensor3(T * d000, T * d001, T * d010, T * d011, T * d100, T * d101,
                // T * d110, T * d111)
        FTensor::Tensor3<FTensor::PackPtr<double *, 2>, 2, 2, 2> t_mat{
            &locMat(3 * rr + 0, 0), // 000
            &locMat(3 * rr + 0, 1), // 001

            &locMat(3 * rr + 1, 0), // 010
            &locMat(3 * rr + 1, 1), // 011

            &locMat(3 * rr + 1, 0), // 100
            &locMat(3 * rr + 1, 1), // 101

            &locMat(3 * rr + 2, 0), // 110
            &locMat(3 * rr + 2, 1)  // 111

        };

        const double c0 = alpha * t_row_base * t_tau_dot;

        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / SPACE_DIM; ++cc) {

          FTensor::Tensor3<double, SPACE_DIM, SPACE_DIM, SPACE_DIM> t_tmp;
          t_mat(i, j, l) -= c0 * (t_diff_plastic_flow_stress_dgrad(i, j, l, k) *
                                  t_col_diff_base(k));

          ++t_mat;
          ++t_col_diff_base;
        }

        ++t_row_base;
      }
      for (; rr < nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_f;
      ++t_flow;
      ++t_tau_dot;

      ++t_logC_dC;
      ++t_grad;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dEP::OpCalculatePlasticFlowLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculatePlasticFlowLhs_dEP::doWork(int row_side, int col_side,
                                                     EntityType row_type,
                                                     EntityType col_type,
                                                     EntData &row_data,
                                                     EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau_dot = getFTensor0FromVec(*(commonDataPtr->plasticTauDotPtr));
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));

    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
    auto t_diff_plastic_strain = diff_tensor();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = dt * getMeasure() * t_w;
      double beta = alpha * getTSa();

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        const double c0 = alpha * t_row_base * t_tau_dot;
        const double c1 = beta * t_row_base;

        auto t_diff_plastic_flow_dstrain = diff_plastic_flow_dstrain(
            t_D, diff_plastic_flow_dstress(t_f, t_flow,
                                           diff_deviator(diff_tensor())));

        // Tensor4(T *d0000, T *d0001, T *d0010, T *d0011, T *d0100, T *d0101,
        //         T *d0110, T *d0111, T *d1000, T *d1001, T *d1010, T *d1011,
        //         T *d1100, T *d1101, T *d1110, T *d1111)
        FTensor::Tensor4<FTensor::PackPtr<double *, 3>, 2, 2, 2, 2> t_mat{

            &locMat(3 * rr + 0, 0), // 0000
            &locMat(3 * rr + 0, 1), // 0001
            &locMat(3 * rr + 0, 1), // 0010
            &locMat(3 * rr + 0, 2), // 0011

            &locMat(3 * rr + 1, 0), // 0100
            &locMat(3 * rr + 1, 1), // 0101
            &locMat(3 * rr + 1, 1), // 0110
            &locMat(3 * rr + 1, 2), // 0111

            &locMat(3 * rr + 1, 0), // 1000
            &locMat(3 * rr + 1, 1), // 1001
            &locMat(3 * rr + 1, 1), // 1010
            &locMat(3 * rr + 1, 2), // 1011

            &locMat(3 * rr + 2, 0), // 1100
            &locMat(3 * rr + 2, 1), // 1101
            &locMat(3 * rr + 2, 1), // 1110
            &locMat(3 * rr + 2, 2)  // 1111

        };

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 3; ++cc) {

          t_mat(i, j, k, l) +=
              t_col_base * (t_D(i, j, m, n) *
                            (c1 * t_diff_plastic_strain(m, n, k, l) +
                             c0 * t_diff_plastic_flow_dstrain(m, n, k, l)));

          ++t_mat;
          ++t_col_base;
        }

        ++t_row_base;
      }
      for (; rr < nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_f;
      ++t_flow;
      ++t_tau_dot;
    }

    CHKERR MatSetValues(getTSB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dTAU::OpCalculatePlasticFlowLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode
OpCalculatePlasticFlowLhs_dTAU::doWork(int row_side, int col_side,
                                       EntityType row_type, EntityType col_type,
                                       EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));

    auto t_row_base = row_data.getFTensor0N();
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = dt * getMeasure() * t_w * getTSa();

      FTensor::Tensor2_symmetric<double, 2> t_flow_stress;
      t_flow_stress(i, j) = t_D(i, j, m, n) * t_flow(m, n);

      for (size_t rr = 0; rr != nb_row_dofs / 3; ++rr) {

        FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 2> t_mat{
            &locMat(3 * rr + 0, 0), &locMat(3 * rr + 1, 0),
            &locMat(3 * rr + 1, 0), &locMat(3 * rr + 2, 0)};

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs; cc++) {
          t_mat(i, j) -= alpha * t_row_base * t_col_base * t_flow_stress(i, j);
          ++t_mat;
          ++t_col_base;
        }

        ++t_row_base;
      }

      ++t_w;
      ++t_flow;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dU::OpCalculateContrainsLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculateContrainsLhs_dU::doWork(int row_side, int col_side,
                                                  EntityType row_type,
                                                  EntityType col_type,
                                                  EntData &row_data,
                                                  EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();


    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_tau_dot = getFTensor0FromVec(*(commonDataPtr->plasticTauDotPtr));
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));
    auto t_stress =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStressPtr));
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
    auto t_diff_symmetrize = diff_symmetrize();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = dt * getMeasure() * t_w;

      auto t_diff_constrain_dstrain = diff_constrain_dstrain(
          t_D,
          diff_constrain_dstress(
              diff_constrain_df(t_tau_dot, t_f - hardening(t_tau)), t_flow));
      FTensor::Tensor2<double, 2, 2> t_diff_constrain_dgrad;
      t_diff_constrain_dgrad(k, l) =
          t_diff_constrain_dstrain(i, j) * t_diff_symmetrize(i, j, k, l);

      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_mat{&locMat(0, 0),
                                                               &locMat(0, 1)};

      size_t rr = 0;
      for (; rr != nb_row_dofs; ++rr) {

        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 2; cc++) {

          t_mat(i) += alpha * t_row_base * t_diff_constrain_dgrad(i, j) *
                      t_col_diff_base(j);

          ++t_mat;
          ++t_col_diff_base;
        }

        ++t_row_base;
      }
      for (; rr != nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_f;
      ++t_tau;
      ++t_tau_dot;
      ++t_flow;
      ++t_stress;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_LogStrain_dU::OpCalculateContrainsLhs_LogStrain_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<HenkyOps::CommonData> comman_henky_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr),
      commonHenkyDataPtr(comman_henky_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculateContrainsLhs_LogStrain_dU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_tau_dot = getFTensor0FromVec(*(commonDataPtr->plasticTauDotPtr));
    auto t_flow = getFTensor2SymmetricFromMat<SPACE_DIM>(
        *(commonDataPtr->plasticFlowPtr));
    auto t_stress =
        getFTensor2SymmetricFromMat<SPACE_DIM>(*(commonDataPtr->mStressPtr));
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);

    auto t_grad =
        getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*(commonDataPtr->mGradPtr));
    auto t_logC_dC = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
        commonHenkyDataPtr->matLogCdC);
    auto t_diff_symmetrize = diff_symmetrize();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = dt * getMeasure() * t_w;

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
      t_F(i, j) = t_grad(i, j) + t_kd(i, j);
      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          dC_dF;
      dC_dF(i, j, k, l) = (t_kd(i, l) * t_F(k, j)) + (t_kd(j, l) * t_F(k, i));

      auto t_diff_constrain_dstrain = diff_constrain_dstrain(
          t_D,
          diff_constrain_dstress(
              diff_constrain_df(t_tau_dot, t_f - hardening(t_tau)), t_flow));

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_diff_constrain_diff_symm;
      t_diff_constrain_diff_symm(k, l) =
          t_diff_constrain_dstrain(i, j) * t_diff_symmetrize(i, j, k, l);

      FTensor::Tensor2_symmetric<double, SPACE_DIM> t_diff_constrain_dlog_c;
      t_diff_constrain_dlog_c(k, l) =
          t_diff_constrain_diff_symm(i, j) * (t_logC_dC(i, j, k, l) / 2);

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_diff_constrain_dgrad;
      t_diff_constrain_dgrad(k, l) =
          t_diff_constrain_dlog_c(i, j) * dC_dF(i, j, k, l);

      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_mat{&locMat(0, 0),
                                                               &locMat(0, 1)};

      size_t rr = 0;
      for (; rr != nb_row_dofs; ++rr) {

        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 2; cc++) {

          t_mat(i) += alpha * t_row_base * t_diff_constrain_dgrad(i, j) *
                      t_col_diff_base(j);

          ++t_mat;
          ++t_col_diff_base;
        }

        ++t_row_base;
      }
      for (; rr != nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_f;
      ++t_tau;
      ++t_tau_dot;
      ++t_flow;
      ++t_stress;
      ++t_w;
      ++t_grad;
      ++t_logC_dC;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dEP::OpCalculateContrainsLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculateContrainsLhs_dEP::doWork(int row_side, int col_side,
                                                   EntityType row_type,
                                                   EntityType col_type,
                                                   EntData &row_data,
                                                   EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();


    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_tau_dot = getFTensor0FromVec(*(commonDataPtr->plasticTauDotPtr));
    auto t_flow =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = dt * getMeasure() * t_w;

      auto mat_ptr = locMat.data().begin();
      auto t_diff_constrain_dstrain = diff_constrain_dstrain(
          t_D,
          diff_constrain_dstress(
              diff_constrain_df(t_tau_dot, t_f - hardening(t_tau)), t_flow));

      constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
      auto t_L = symm_L_tensor();
      FTensor::Tensor1<FTensor::PackPtr<double *, size_symm>, size_symm> t_mat{
          &locMat(0, 0), &locMat(0, 1), &locMat(0, 2)};

      size_t rr = 0;
      for (; rr != nb_row_dofs; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / 3; cc++) {

          t_mat(L) -= (alpha * t_row_base * t_col_base) *
                      (t_diff_constrain_dstrain(i, j) * t_L(i, j, L));

          ++t_mat;
          ++t_col_base;
        }

        ++t_row_base;
      }
      for (; rr != nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_f;
      ++t_tau;
      ++t_tau_dot;
      ++t_flow;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dTAU::OpCalculateContrainsLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculateContrainsLhs_dTAU::doWork(int row_side, int col_side,
                                                    EntityType row_type,
                                                    EntityType col_type,
                                                    EntData &row_data,
                                                    EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    auto get_dt = [&]() {
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      return dt;
    };
    const auto dt = get_dt();


    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_tau_dot = getFTensor0FromVec(*(commonDataPtr->plasticTauDotPtr));

    auto t_row_base = row_data.getFTensor0N();
    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha = dt * getMeasure() * t_w;
      const double c0 = alpha * getTSa() *
                        diff_constrain_dtau(t_tau_dot, t_f - hardening(t_tau));
      const double c1 = alpha *
                        diff_constrain_df(t_tau_dot, t_f - hardening(t_tau)) *
                        hardening_dtau();

      auto mat_ptr = locMat.data().begin();

      size_t rr = 0;
      for (; rr != nb_row_dofs; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs; ++cc) {
          *mat_ptr += (c0 - c1) * t_row_base * t_col_base;
          ++mat_ptr;
          ++t_col_base;
        }
        ++t_row_base;
      }
      for (; rr < nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_f;
      ++t_tau;
      ++t_tau_dot;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpPostProcPlastic::OpPostProcPlastic(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode OpPostProcPlastic::doWork(int side, EntityType type,
                                         EntData &data) {
  MoFEMFunctionBegin;

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

  auto set_matrix_2d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 2; ++r)
      for (size_t c = 0; c != 2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_matrix_3d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 3; ++r)
      for (size_t c = 0; c != 3; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_matrix_2d_symm = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 2; ++r)
      for (size_t c = 0; c != 2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_scalar = [&](auto t) -> MatrixDouble3by3 & {
    mat.clear();
    mat(0, 0) = t;
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_plastic_surface = get_tag("PLASTIC_SURFACE", 1);
  auto th_tau = get_tag("PLASTIC_MULTIPLIER", 1);
  auto th_plastic_flow = get_tag("PLASTIC_FLOW", 9);
  auto th_plastic_strain = get_tag("PLASTIC_STRAIN", 9);

  auto t_flow =
      getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticFlowPtr));
  auto t_plastic_strain =
      getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticStrainPtr));
  size_t gg = 0;
  for (int gg = 0; gg != commonDataPtr->plasticSurfacePtr->size(); ++gg) {
    const double f = (*(commonDataPtr->plasticSurfacePtr))[gg];
    const double tau = (*(commonDataPtr->plasticTauPtr))[gg];
    CHKERR set_tag(th_plastic_surface, gg, set_scalar(f - hardening(tau)));
    CHKERR set_tag(th_tau, gg, set_scalar(tau));
    CHKERR set_tag(th_plastic_flow, gg, set_matrix_2d(t_flow));
    CHKERR set_tag(th_plastic_strain, gg, set_matrix_2d(t_plastic_strain));
    ++t_flow;
    ++t_plastic_strain;
  }

  MoFEMFunctionReturn(0);
}

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcFaceOnRefinedMesh> &post_proc_fe,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter

          )
      : dM(dm), postProcFe(post_proc_fe), uXScatter(ux_scatter),
        uYScatter(uy_scatter){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    auto make_vtk = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_plastic_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    auto print_max_min = [&](auto &tuple, const std::string msg) {
      MoFEMFunctionBegin;
      CHKERR VecScatterBegin(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                             INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecScatterEnd(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                           INSERT_VALUES, SCATTER_FORWARD);
      double max, min;
      CHKERR VecMax(std::get<0>(tuple), PETSC_NULL, &max);
      CHKERR VecMin(std::get<0>(tuple), PETSC_NULL, &min);
      PetscPrintf(PETSC_COMM_WORLD, "%s time %3.4e min %3.4e max %3.4e\n",
                  msg.c_str(), ts_t, min, max);
      MoFEMFunctionReturn(0);
    };

    CHKERR make_vtk();
    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
};

//! [Postprocessing]
}; // namespace OpPlasticTools
