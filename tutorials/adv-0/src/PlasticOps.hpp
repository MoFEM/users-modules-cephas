

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

  enum ParamsIndexes {
    YOUNG_MODULUS,
    POISSON_RATIO,
    SIGMA_Y,
    H,
    VIS_H,
    QINF,
    BISO,
    LAST_PARAM
  };

  using BlockParams = std::array<double, LAST_PARAM>;
  BlockParams blockParams;

  inline auto getParamsPtr() {
    return boost::shared_ptr<BlockParams>(shared_from_this(), &blockParams);
  };

  //! [Common data set externally]
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<MatrixDouble> mDPtr_Axiator;
  boost::shared_ptr<MatrixDouble> mDPtr_Deviator;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;
  //! [Common data set externally]

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

  static std::array<int, 5> activityData;
};

std::array<int, 5> CommonData::activityData = {0, 0, 0, 0, 0};

//! [Common data]


FTensor::Index<'I', 3> I;
FTensor::Index<'J', 3> J;
FTensor::Index<'M', 3> M;
FTensor::Index<'N', 3> N;

template <int DIM, IntegrationType I, typename DomainEleOp>
struct OpCalculatePlasticSurfaceImpl;

template <int DIM, IntegrationType I, typename DomainEleOp>
struct OpCalculatePlasticityImpl;

template <int DIM, IntegrationType I, typename DomainEleOp>
struct OpPlasticStressImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculatePlasticFlowRhsImpl;

template <IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculateConstraintsRhsImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculatePlasticInternalForceLhs_dEPImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculatePlasticInternalForceLhs_LogStrain_dEPImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculatePlasticFlowLhs_dUImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculatePlasticFlowLhs_LogStrain_dUImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculatePlasticFlowLhs_dEPImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculatePlasticFlowLhs_dTAUImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculateConstraintsLhs_dUImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculateConstraintsLhs_LogStrain_dUImpl;

template <int DIM, IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculateConstraintsLhs_dEPImpl;

template <IntegrationType I, typename AssemblyDomainEleOp>
struct OpCalculateConstraintsLhs_dTAUImpl;

template <typename DomainEleOp> struct PlasticityIntegrators {
  template <int DIM, IntegrationType I>
  using OpCalculatePlasticSurface =
      OpCalculatePlasticSurfaceImpl<DIM, I, DomainEleOp>;

  template <int DIM, IntegrationType I>
  using OpCalculatePlasticity = OpCalculatePlasticityImpl<DIM, I, DomainEleOp>;

  template <int DIM, IntegrationType I>
  using OpPlasticStress = OpPlasticStressImpl<DIM, I, DomainEleOp>;

  template <AssemblyType A> struct Assembly {

    using AssemblyDomainEleOp =
        typename FormsIntegrators<DomainEleOp>::template Assembly<A>::OpBase;

    template <int DIM, IntegrationType I>
    using OpCalculatePlasticFlowRhs =
        OpCalculatePlasticFlowRhsImpl<DIM, I, AssemblyDomainEleOp>;

    template <IntegrationType I>
    using OpCalculateConstraintsRhs =
        OpCalculateConstraintsRhsImpl<I, AssemblyDomainEleOp>;

    template <int DIM, IntegrationType I>
    using OpCalculatePlasticInternalForceLhs_dEP =
        OpCalculatePlasticInternalForceLhs_dEPImpl<DIM, I, AssemblyDomainEleOp>;

    template <int DIM, IntegrationType I>
    using OpCalculatePlasticInternalForceLhs_LogStrain_dEP =
        OpCalculatePlasticInternalForceLhs_LogStrain_dEPImpl<
            DIM, I, AssemblyDomainEleOp>;

    template <int DIM, IntegrationType I>
    using OpCalculatePlasticFlowLhs_dU =
        OpCalculatePlasticFlowLhs_dUImpl<DIM, I, AssemblyDomainEleOp>;

    template <int DIM, IntegrationType I>
    using OpCalculatePlasticFlowLhs_LogStrain_dU =
        OpCalculatePlasticFlowLhs_LogStrain_dUImpl<DIM, I, AssemblyDomainEleOp>;

    template <int DIM, IntegrationType I>
    using OpCalculatePlasticFlowLhs_dEP =
        OpCalculatePlasticFlowLhs_dEPImpl<DIM, I, AssemblyDomainEleOp>;

    template <int DIM, IntegrationType I>
    using OpCalculatePlasticFlowLhs_dTAU =
        OpCalculatePlasticFlowLhs_dTAUImpl<DIM, I, AssemblyDomainEleOp>;

    template <int DIM, IntegrationType I>
    using OpCalculateConstraintsLhs_dU =
        OpCalculateConstraintsLhs_dUImpl<DIM, I, AssemblyDomainEleOp>;

    template <int DIM, IntegrationType I>
    using OpCalculateConstraintsLhs_LogStrain_dU =
        OpCalculateConstraintsLhs_LogStrain_dUImpl<DIM, I, AssemblyDomainEleOp>;

    template <int DIM, IntegrationType I>
    using OpCalculateConstraintsLhs_dEP =
        OpCalculateConstraintsLhs_dEPImpl<DIM, I, AssemblyDomainEleOp>;

    template <IntegrationType I>
    using OpCalculateConstraintsLhs_dTAU =
        OpCalculateConstraintsLhs_dTAUImpl<I, AssemblyDomainEleOp>;

  };
};

//! [Operators definitions]
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
struct OpPlasticStressImpl<DIM, GAUSS, DomainEleOp> : public DomainEleOp {
  OpPlasticStressImpl(const std::string field_name,
                      boost::shared_ptr<CommonData> common_data_ptr,
                      boost::shared_ptr<MatrixDouble> mDPtr,
                      const double scale = 1);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<MatrixDouble> mDPtr;
  const double scaleStress;
  boost::shared_ptr<CommonData> commonDataPtr;
};

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

template <int DIM, typename AssemblyDomainEleOp>
struct OpCalculatePlasticInternalForceLhs_dEPImpl<DIM, GAUSS,
                                                  AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculatePlasticInternalForceLhs_dEPImpl(
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
struct OpCalculatePlasticInternalForceLhs_LogStrain_dEPImpl<DIM, GAUSS,
                                                            AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculatePlasticInternalForceLhs_LogStrain_dEPImpl(
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

template <int DIM, typename AssemblyDomainEleOp>
struct OpCalculatePlasticFlowLhs_dUImpl<DIM, GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dUImpl(
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
struct OpCalculatePlasticFlowLhs_LogStrain_dUImpl<DIM, GAUSS,
                                                  AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_LogStrain_dUImpl(
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

template <int DIM, typename AssembleDomainEleOp>
struct OpCalculatePlasticFlowLhs_dTAUImpl<DIM, GAUSS, AssembleDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculatePlasticFlowLhs_dTAUImpl(const std::string row_field_name,
                                 const std::string col_field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr,
                                 boost::shared_ptr<MatrixDouble> m_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

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
struct OpCalculateConstraintsLhs_LogStrain_dUImpl<DIM, GAUSS, AssemblyDomainEleOp> :
 public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_LogStrain_dUImpl(
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

template <int DIM, typename AssemblyDomainEleOp>
struct OpCalculateConstraintsLhs_dEPImpl<DIM, GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_dEPImpl(const std::string row_field_name,
                                const std::string col_field_name,
                                boost::shared_ptr<CommonData> common_data_ptr,
                                boost::shared_ptr<MatrixDouble> mat_D_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

template <typename AssemblyDomainEleOp>
struct OpCalculateConstraintsLhs_dTAUImpl<GAUSS, AssemblyDomainEleOp>
    : public AssemblyDomainEleOp {
  OpCalculateConstraintsLhs_dTAUImpl(const std::string row_field_name,
                                 const std::string col_field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

//! [Operators definitions]

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
inline auto deviator(FTensor::Tensor2_symmetric<T, SPACE_DIM> &t_stress,
                     double trace, FTensor::Number<DIM>) {
  FTensor::Tensor2_symmetric<double, 3> t_dev;
  t_dev(I, J) = 0;
  for (int ii = 0; ii != DIM; ++ii)
    for (int jj = ii; jj != DIM; ++jj)
      t_dev(ii, jj) = t_stress(ii, jj);
  t_dev(0, 0) -= trace;
  t_dev(1, 1) -= trace;
  t_dev(2, 2) -= trace;
  return t_dev;
};

template <typename T>
inline auto deviator(FTensor::Tensor2_symmetric<T, 2> &t_stress, double trace) {
  return deviator(t_stress, trace, FTensor::Number<2>());
};

template <typename T>
inline auto deviator(FTensor::Tensor2_symmetric<T, 3> &t_stress, double trace) {
  return deviator(t_stress, trace, FTensor::Number<3>());
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

inline double w(double eqiv, double dot_tau, double f, double sigma_y,
                double sigma_Y) {
  return (f - sigma_y) / sigma_Y + cn1 * (dot_tau * std::sqrt(eqiv));
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
  return vis_H * dot_tau + (sigma_Y / 2) * ((cn0 * (dot_tau - eqiv) +
                                             cn1 * (std::sqrt(eqiv) * dot_tau) -
                                             (f - sigma_y) / sigma_Y) -
                                            abs_w);
};

inline double diff_constrain_ddot_tau(double sign, double eqiv, double vis_H,
                                      double sigma_Y) {
  return vis_H + (sigma_Y / 2) * (cn0 + cn1 * std::sqrt(eqiv) * (1 - sign));
};

inline double diff_constrain_deqiv(double sign, double eqiv, double dot_tau,
                                   double sigma_Y) {
  return (sigma_Y / 2) *
         (-cn0 + cn1 * dot_tau * (0.5 / std::sqrt(eqiv)) * (1 - sign));
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

}; // namespace PlasticOps

#include <PlasticOpsGeneric.hpp>
#include <PlasticOpsSmallStrains.hpp>
#include <PlasticOpsLargeStrains.hpp>
#include <PlasticOpsMonitor.hpp>

namespace PlasticOps {

using Pip = boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator>;
using CommonPlasticPtr = boost::shared_ptr<PlasticOps::CommonData>;
using CommonHenkyPtr = boost::shared_ptr<HenckyOps::CommonData>;

template <int DIM>
MoFEMErrorCode
addMatBlockOps(MoFEM::Interface &m_field, std::string block_name, Pip &pip,
               boost::shared_ptr<MatrixDouble> mat_D_Ptr,
               boost::shared_ptr<CommonData::BlockParams> mat_params_ptr,
               double scale, Sev sev) {
  MoFEMFunctionBegin;

  struct OpMatBlocks : public DomainEleOp {
    OpMatBlocks(boost::shared_ptr<MatrixDouble> m_D_ptr,
                boost::shared_ptr<CommonData::BlockParams> mat_params_ptr,
                double scale, MoFEM::Interface &m_field, Sev sev,
                std::vector<const CubitMeshSets *> meshset_vec_ptr)
        : DomainEleOp(NOSPACE, DomainEleOp::OPSPACE), matDPtr(m_D_ptr),
          matParamsPtr(mat_params_ptr), scaleVal(scale) {
      CHK_THROW_MESSAGE(extractBlockData(m_field, meshset_vec_ptr, sev),
                        "Can not get data from block");
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          EntitiesFieldData::EntData &data) {
      MoFEMFunctionBegin;

      auto getK = [](auto &p) {
        auto young_modulus = p[CommonData::YOUNG_MODULUS];
        auto poisson_ratio = p[CommonData::POISSON_RATIO];
        return young_modulus / (3 * (1 - 2 * poisson_ratio));
      };

      auto getG = [](auto &p) {
        auto young_modulus = p[CommonData::YOUNG_MODULUS];
        auto poisson_ratio = p[CommonData::POISSON_RATIO];
        return young_modulus / (2 * (1 + poisson_ratio));
      };

      auto scale = [this](auto &p) {
        for (auto &v : p)
          v *= scaleVal;
      };

      for (auto &b : blockData) {
        if (b.blockEnts.find(getFEEntityHandle()) != b.blockEnts.end()) {
          *matParamsPtr = b.bParams;
          scale(*matParamsPtr);
          CHKERR getMatDPtr(matDPtr, getK(*matParamsPtr), getG(*matParamsPtr));
          MoFEMFunctionReturnHot(0);
        }
      }

      (*matParamsPtr) = {young_modulus, poisson_ratio, sigmaY, H,
                         visH,          Qinf,          b_iso};
      scale(*matParamsPtr);
      CHKERR getMatDPtr(matDPtr, getK(*matParamsPtr), getG(*matParamsPtr));

      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<MatrixDouble> matDPtr;
    boost::shared_ptr<CommonData::BlockParams> matParamsPtr;
    double scaleVal;

    struct BlockData {
      std::array<double, CommonData::LAST_PARAM> bParams;
      Range blockEnts;
    };
    std::vector<BlockData> blockData;

    /**
     * @brief Extract block data from meshsets
     *
     * @param m_field
     * @param meshset_vec_ptr
     * @param sev
     * @return MoFEMErrorCode
     */
    MoFEMErrorCode
    extractBlockData(MoFEM::Interface &m_field,
                     std::vector<const CubitMeshSets *> meshset_vec_ptr,
                     Sev sev) {
      MoFEMFunctionBegin;

      for (auto m : meshset_vec_ptr) {
        MOFEM_TAG_AND_LOG("WORLD", sev, "MatBlock") << *m;
        std::vector<double> block_data;
        CHKERR m->getAttributes(block_data);
        if (block_data.size() != 2 + CommonData::LAST_PARAM) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                  "Wron number of block attribute");
        }
        auto get_block_ents = [&]() {
          Range ents;
          CHKERR m_field.get_moab().get_entities_by_handle(m->meshset, ents,
                                                           true);
          return ents;
        };

        CommonData::BlockParams block_params;
        for (auto i = 0; i != CommonData::LAST_PARAM; ++i) {
          block_params[i] = block_data[i];
        }

        MOFEM_TAG_AND_LOG("WORLD", sev, "MatBlock")
            << "E = " << block_params[CommonData::YOUNG_MODULUS]
            << " nu = " << block_params[CommonData::POISSON_RATIO];
        MOFEM_TAG_AND_LOG("WORLD", sev, "MatBlock")
            << std::endl
            << "sigma_y = " << block_params[CommonData::SIGMA_Y] << std::endl
            << "h = " << block_params[CommonData::H] << std::endl
            << "vis_h = " << block_params[CommonData::VIS_H] << std::endl
            << "qinf = " << block_params[CommonData::QINF] << std::endl
            << "biso = " << block_params[CommonData::BISO] << std::endl;

        blockData.push_back({block_params, get_block_ents()});
      }
      MOFEM_LOG_CHANNEL("WORLD");
      MoFEMFunctionReturn(0);
    }

    /**
     * @brief Get elasticity tensor
     *
     * Calculate elasticity tensor for given material parameters
     *
     * @param mat_D_ptr
     * @param bulk_modulus_K
     * @param shear_modulus_G
     * @return MoFEMErrorCode
     *
     */
    MoFEMErrorCode getMatDPtr(boost::shared_ptr<MatrixDouble> mat_D_ptr,
                              double bulk_modulus_K, double shear_modulus_G) {
      MoFEMFunctionBegin;
      //! [Calculate elasticity tensor]
      auto set_material_stiffness = [&]() {
        FTensor::Index<'i', DIM> i;
        FTensor::Index<'j', DIM> j;
        FTensor::Index<'k', DIM> k;
        FTensor::Index<'l', DIM> l;
        constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
        double A = (DIM == 2)
                       ? 2 * shear_modulus_G /
                             (bulk_modulus_K + (4. / 3.) * shear_modulus_G)
                       : 1;
        auto t_D = getFTensor4DdgFromMat<DIM, DIM, 0>(*mat_D_ptr);
        t_D(i, j, k, l) =
            2 * shear_modulus_G * ((t_kd(i, k) ^ t_kd(j, l)) / 4.) +
            A * (bulk_modulus_K - (2. / 3.) * shear_modulus_G) * t_kd(i, j) *
                t_kd(k, l);
      };
      //! [Calculate elasticity tensor]
      constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
      mat_D_ptr->resize(size_symm * size_symm, 1);
      set_material_stiffness();
      MoFEMFunctionReturn(0);
    }
  };

  // push operator to calculate material stiffness matrix for each block
  pip.push_back(new OpMatBlocks(
      mat_D_Ptr, mat_params_ptr, scale, m_field, sev,

      // Get blockset using regular expression
      m_field.getInterface<MeshsetsManager>()->getCubitMeshsetPtr(std::regex(

          (boost::format("%s(.*)") % block_name).str()

              ))

          ));

  MoFEMFunctionReturn(0);
}

template <int DIM, IntegrationType I, typename DomainEleOp>
auto createCommonPlasticOps(
    MoFEM::Interface &m_field, std::string block_name,
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip,
    std::string u, std::string ep, std::string tau, double scale, Sev sev) {

  using P = PlasticityIntegrators<DomainEleOp>;

  auto common_plastic_ptr = boost::make_shared<PlasticOps::CommonData>();
  common_plastic_ptr = boost::make_shared<PlasticOps::CommonData>();

  constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
  auto make_d_mat = []() {
    return boost::make_shared<MatrixDouble>(size_symm * size_symm, 1);
  };

  common_plastic_ptr->mDPtr = make_d_mat();
  common_plastic_ptr->mDPtr_Axiator = make_d_mat();
  common_plastic_ptr->mDPtr_Deviator = make_d_mat();
  common_plastic_ptr->mGradPtr = boost::make_shared<MatrixDouble>();
  common_plastic_ptr->mStrainPtr = boost::make_shared<MatrixDouble>();
  common_plastic_ptr->mStressPtr = boost::make_shared<MatrixDouble>();

  auto m_D_ptr = common_plastic_ptr->mDPtr;

  CHK_THROW_MESSAGE(addMatBlockOps<DIM>(m_field, block_name, pip, m_D_ptr,
                                        common_plastic_ptr->getParamsPtr(),
                                        scale, sev),
                    "add mat block ops");

  pip.push_back(new OpCalculateScalarFieldValues(
      tau, common_plastic_ptr->getPlasticTauPtr()));
  pip.push_back(new OpCalculateTensor2SymmetricFieldValues<DIM>(
      ep, common_plastic_ptr->getPlasticStrainPtr()));
  pip.push_back(new OpCalculateVectorFieldGradient<DIM, DIM>(
      u, common_plastic_ptr->mGradPtr));

  CommonHenkyPtr common_henky_ptr;

  if (is_large_strains) {
    common_henky_ptr = boost::make_shared<HenckyOps::CommonData>();
    common_henky_ptr->matGradPtr = common_plastic_ptr->mGradPtr;
    common_henky_ptr->matDPtr = common_plastic_ptr->mDPtr;
    common_henky_ptr->matLogCPlastic =
        common_plastic_ptr->getPlasticStrainPtr();
    common_plastic_ptr->mStrainPtr = common_henky_ptr->getMatLogC();
    common_plastic_ptr->mStressPtr = common_henky_ptr->getMatHenckyStress();

    using H = HenckyOps::HenkyIntegrators<DomainEleOp>;

    pip.push_back(new typename H::template OpCalculateEigenVals<DIM, I>(
        u, common_henky_ptr));
    pip.push_back(
        new typename H::template OpCalculateLogC<DIM, I>(u, common_henky_ptr));
    pip.push_back(new typename H::template OpCalculateLogC_dC<DIM, I>(
        u, common_henky_ptr));
    pip.push_back(new
                  typename H::template OpCalculateHenckyPlasticStress<DIM, I>(
                      u, common_henky_ptr, m_D_ptr));
    pip.push_back(new typename H::template OpCalculatePiolaStress<DIM, I>(
        u, common_henky_ptr));
  } else {

    pip.push_back(new OpSymmetrizeTensor<SPACE_DIM>(
        u, common_plastic_ptr->mGradPtr, common_plastic_ptr->mStrainPtr));
    pip.push_back(new typename P::template OpPlasticStress<DIM, I>(
        u, common_plastic_ptr, m_D_ptr, 1));
  }

  pip.push_back(new typename P::template OpCalculatePlasticSurface<DIM, I>(
      u, common_plastic_ptr));

  return std::make_tuple(common_plastic_ptr, common_henky_ptr);
}

template <int DIM, AssemblyType A, IntegrationType I, typename DomainEleOp>
MoFEMErrorCode
opFactoryDomainRhs(MoFEM::Interface &m_field, std::string block_name, Pip &pip,
                   std::string u, std::string ep, std::string tau) {
  MoFEMFunctionBegin;

  using B = typename FormsIntegrators<DomainEleOp>::template Assembly<
      A>::template LinearForm<I>;
  using OpInternalForceCauchy =
      typename B::template OpGradTimesSymTensor<1, DIM, DIM>;
  using OpInternalForcePiola =
      typename B::template OpGradTimesTensor<1, DIM, DIM>;

  using P = PlasticityIntegrators<DomainEleOp>;

  auto [common_plastic_ptr, common_henky_ptr] =
      createCommonPlasticOps<DIM, I, DomainEleOp>(m_field, block_name, pip, u,
                                                  ep, tau, scale, Sev::inform);

  auto m_D_ptr = common_plastic_ptr->mDPtr;

  pip.push_back(new OpCalculateTensor2SymmetricFieldValuesDot<DIM>(
      ep, common_plastic_ptr->getPlasticStrainDotPtr()));
  pip.push_back(new OpCalculateScalarFieldValuesDot(
      tau, common_plastic_ptr->getPlasticTauDotPtr()));
  pip.push_back(new typename P::template OpCalculatePlasticity<DIM, I>(
      u, common_plastic_ptr, m_D_ptr));

  // Calculate internal forces
  if (common_henky_ptr) {
    pip.push_back(new OpInternalForcePiola(
        u, common_henky_ptr->getMatFirstPiolaStress()));
  } else {
    pip.push_back(new OpInternalForceCauchy(u, common_plastic_ptr->mStressPtr));
  }

  pip.push_back(
      new
      typename P::template Assembly<A>::template OpCalculateConstraintsRhs<I>(
          tau, common_plastic_ptr, m_D_ptr));
  pip.push_back(
      new typename P::template Assembly<A>::template OpCalculatePlasticFlowRhs<
          DIM, I>(ep, common_plastic_ptr, m_D_ptr));

  MoFEMFunctionReturn(0);
}

template <int DIM, AssemblyType A, IntegrationType I, typename DomainEleOp>
MoFEMErrorCode
opFactoryDomainLhs(MoFEM::Interface &m_field, std::string block_name, Pip &pip,
                   std::string u, std::string ep, std::string tau) {
  MoFEMFunctionBegin;

  using namespace HenckyOps;

  using B = typename FormsIntegrators<DomainEleOp>::template Assembly<
      A>::template BiLinearForm<I>;
  using OpKPiola = typename B::template OpGradTensorGrad<1, DIM, DIM, 1>;
  using OpKCauchy = typename B::template OpGradSymTensorGrad<1, DIM, DIM, 0>;

  using P = PlasticityIntegrators<DomainEleOp>;

  auto [common_plastic_ptr, common_henky_ptr] =
      createCommonPlasticOps<DIM, I, DomainEleOp>(m_field, block_name, pip, u,
                                                  ep, tau, scale, Sev::verbose);

  auto m_D_ptr = common_plastic_ptr->mDPtr;

  pip.push_back(new OpCalculateTensor2SymmetricFieldValuesDot<DIM>(
      ep, common_plastic_ptr->getPlasticStrainDotPtr()));
  pip.push_back(new OpCalculateScalarFieldValuesDot(
      tau, common_plastic_ptr->getPlasticTauDotPtr()));
  pip.push_back(new typename P::template OpCalculatePlasticity<DIM, I>(
      u, common_plastic_ptr, m_D_ptr));

  if (common_henky_ptr) {
    using H = HenckyOps::HenkyIntegrators<DomainEleOp>;
    pip.push_back(new typename H::template OpHenckyTangent<DIM, I>(
        u, common_henky_ptr, m_D_ptr));
    pip.push_back(new OpKPiola(u, u, common_henky_ptr->getMatTangent()));
    pip.push_back(
        new typename P::template Assembly<A>::
            template OpCalculatePlasticInternalForceLhs_LogStrain_dEP<DIM, I>(
                u, ep, common_plastic_ptr, common_henky_ptr, m_D_ptr));
  } else {
    pip.push_back(new OpKCauchy(u, u, m_D_ptr));
    pip.push_back(new typename P::template Assembly<A>::
                      template OpCalculatePlasticInternalForceLhs_dEP<DIM, I>(
                          u, ep, common_plastic_ptr, m_D_ptr));
  }

  if (common_henky_ptr) {
    pip.push_back(
        new typename P::template Assembly<A>::
            template OpCalculateConstraintsLhs_LogStrain_dU<DIM, I>(
                tau, u, common_plastic_ptr, common_henky_ptr, m_D_ptr));
    pip.push_back(
        new typename P::template Assembly<A>::
            template OpCalculatePlasticFlowLhs_LogStrain_dU<DIM, I>(
                ep, u, common_plastic_ptr, common_henky_ptr, m_D_ptr));
  } else {
    pip.push_back(
        new
        typename P::template Assembly<A>::template OpCalculateConstraintsLhs_dU<
            DIM, I>(tau, u, common_plastic_ptr, m_D_ptr));
    pip.push_back(
        new
        typename P::template Assembly<A>::template OpCalculatePlasticFlowLhs_dU<
            DIM, I>(ep, u, common_plastic_ptr, m_D_ptr));
  }

  pip.push_back(
      new
      typename P::template Assembly<A>::template OpCalculatePlasticFlowLhs_dEP<
          DIM, I>(ep, ep, common_plastic_ptr, m_D_ptr));
  pip.push_back(
      new
      typename P::template Assembly<A>::template OpCalculatePlasticFlowLhs_dTAU<
          DIM, I>(ep, tau, common_plastic_ptr, m_D_ptr));
  pip.push_back(
      new
      typename P::template Assembly<A>::template OpCalculateConstraintsLhs_dEP<
          DIM, I>(tau, ep, common_plastic_ptr, m_D_ptr));
  pip.push_back(
      new
      typename P::template Assembly<A>::template OpCalculateConstraintsLhs_dTAU<
          I>(tau, tau, common_plastic_ptr));

  MoFEMFunctionReturn(0);
}

template <int DIM, AssemblyType A, IntegrationType I, typename DomainEleOp>
MoFEMErrorCode opFactoryDomainReactions(MoFEM::Interface &m_field,
                                        std::string block_name, Pip &pip,
                                        std::string u, std::string ep,
                                        std::string tau) {
  MoFEMFunctionBegin;

  using B = typename FormsIntegrators<DomainEleOp>::template Assembly<
      A>::template LinearForm<I>;
  using OpInternalForceCauchy =
      typename B::template OpGradTimesSymTensor<1, DIM, DIM>;
  using OpInternalForcePiola =
      typename B::template OpGradTimesTensor<1, DIM, DIM>;

  auto [common_plastic_ptr, common_henky_ptr] =
      createCommonPlasticOps<DIM, I, DomainEleOp>(m_field, block_name, pip, u,
                                                  ep, tau, 1, Sev::inform);

  auto m_D_ptr = common_plastic_ptr->mDPtr;

  // Calculate internal forces
  if (common_henky_ptr) {
    pip.push_back(new OpInternalForcePiola(
        u, common_henky_ptr->getMatFirstPiolaStress()));
  } else {
    pip.push_back(new OpInternalForceCauchy(u, common_plastic_ptr->mStressPtr));
  }

  MoFEMFunctionReturn(0);
}

} // namespace PlasticOps
