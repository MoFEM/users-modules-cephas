

/** \file PlasticOpsGeneric.hpp
 * \example PlasticOpsGeneric.hpp
 */

namespace PlasticOps {

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
  auto t_stress =
      getFTensor2SymmetricFromMat<SPACE_DIM>(*(commonDataPtr->mStressPtr));

  commonDataPtr->plasticSurface.resize(nb_gauss_pts, false);
  commonDataPtr->plasticFlow.resize((SPACE_DIM * (SPACE_DIM + 1)) / 2,
                                    nb_gauss_pts, false);
  auto t_flow =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticFlow);

  for (auto &f : commonDataPtr->plasticSurface) {
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
                                 boost::shared_ptr<CommonData> common_data_ptr,
                                 boost::shared_ptr<MatrixDouble> m_D_ptr,
                                 const double scale)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), scaleStress(scale), mDPtr(m_D_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Calculate stress]
MoFEMErrorCode OpPlasticStress::doWork(int side, EntityType type,
                                       EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  commonDataPtr->mStressPtr->resize((SPACE_DIM * (SPACE_DIM + 1)) / 2,
                                    nb_gauss_pts);
  auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);
  auto t_strain =
      getFTensor2SymmetricFromMat<SPACE_DIM>(*(commonDataPtr->mStrainPtr));
  auto t_plastic_strain =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticStrain);
  auto t_stress =
      getFTensor2SymmetricFromMat<SPACE_DIM>(*(commonDataPtr->mStressPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    t_stress(i, j) =
        t_D(i, j, k, l) * (t_strain(k, l) - t_plastic_strain(k, l));
    t_stress(i, j) /= scaleStress;
    ++t_strain;
    ++t_plastic_strain;
    ++t_stress;
  }

  MoFEMFunctionReturn(0);
}
//! [Calculate stress]

OpCalculatePlasticFlowRhs::OpCalculatePlasticFlowRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyDomainEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

static inline FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 2, 2>
get_nf(VectorDouble &nf, FTensor::Number<2>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 2, 2>{&nf[0], &nf[1],
                                                               &nf[1], &nf[2]};
}

static inline FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 3>
get_nf(VectorDouble &nf, FTensor::Number<3>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 3>{
      &nf[0], &nf[1], &nf[2], &nf[1], &nf[3], &nf[4], &nf[2], &nf[4], &nf[5]};
}

MoFEMErrorCode
OpCalculatePlasticFlowRhs::iNtegrate(EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  auto t_flow =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticFlow);

  auto t_plastic_strain_dot =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticStrainDot);
  auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);

  auto t_D =
      getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  const size_t nb_integration_pts = data.getN().size1();
  const size_t nb_base_functions = data.getN().size2();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_base = data.getFTensor0N();

  auto &nf = AssemblyDomainEleOp::locF;

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;

    auto t_nf = get_nf(nf, FTensor::Number<SPACE_DIM>());

    const double tau_dot = t_tau_dot;

    FTensor::Tensor2_symmetric<double, SPACE_DIM> t_flow_stress_diff;
    t_flow_stress_diff(i, j) = t_D(i, j, k, l) * (t_plastic_strain_dot(k, l) -
                                                  t_tau_dot * t_flow(k, l));

    size_t bb = 0;
    for (; bb != AssemblyDomainEleOp::nbRows / size_symm; ++bb) {
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

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsRhs::OpCalculateContrainsRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyDomainEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpCalculateContrainsRhs::iNtegrate(EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_integration_pts = data.getN().size1();
  const size_t nb_base_functions = data.getN().size2();

  auto t_tau = getFTensor0FromVec(commonDataPtr->plasticTau);
  auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);
  auto t_f = getFTensor0FromVec(commonDataPtr->plasticSurface);
  auto t_w = getFTensor0IntegrationWeight();

  auto &nf = AssemblyDomainEleOp::locF;

  auto t_base = data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = getMeasure() * t_w;

    const double beta =
        alpha * constrain(t_tau_dot, t_f, hardening(t_tau));
        
    size_t bb = 0;
    for (; bb != AssemblyDomainEleOp::nbRows; ++bb) {
      nf[bb] += beta * t_base;
      // nf[bb] += alpha * t_base * k_penalty(t_tau);
      ++t_base;
    }
    for (; bb < nb_base_functions; ++bb)
      ++t_base;

    ++t_tau;
    ++t_tau_dot;
    ++t_f;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticInternalForceLhs_dEP::OpCalculatePlasticInternalForceLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  sYmm = false;
}

static inline FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 2, 3>
get_mat_vector_dtensor_sym(size_t rr, MatrixDouble &mat, FTensor::Number<2>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 2, 3>{
      &mat(2 * rr + 0, 0), &mat(2 * rr + 0, 1), &mat(2 * rr + 0, 2),
      &mat(2 * rr + 1, 0), &mat(2 * rr + 1, 1), &mat(2 * rr + 1, 2)};
}

static inline FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 6>
get_mat_vector_dtensor_sym(size_t rr, MatrixDouble &mat, FTensor::Number<3>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 6>{
      &mat(3 * rr + 0, 0), &mat(3 * rr + 0, 1), &mat(3 * rr + 0, 2),
      &mat(3 * rr + 0, 3), &mat(3 * rr + 0, 4), &mat(3 * rr + 0, 5),

      &mat(3 * rr + 1, 0), &mat(3 * rr + 1, 1), &mat(3 * rr + 1, 2),
      &mat(3 * rr + 1, 3), &mat(3 * rr + 1, 4), &mat(3 * rr + 1, 5),

      &mat(3 * rr + 2, 0), &mat(3 * rr + 2, 1), &mat(3 * rr + 2, 2),
      &mat(3 * rr + 2, 3), &mat(3 * rr + 2, 4), &mat(3 * rr + 2, 5)};
}

MoFEMErrorCode OpCalculatePlasticInternalForceLhs_dEP::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const size_t nb_integration_pts = row_data.getN().size1();
  const size_t nb_row_base_functions = row_data.getN().size2();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
  auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  auto t_L = symm_L_tensor();

  FTensor::Dg<double, SPACE_DIM, size_symm> t_DL;
  t_DL(i, j, L) = 0;
  for (int ii = 0; ii != SPACE_DIM; ++ii)
    for (int jj = ii; jj != SPACE_DIM; ++jj)
      for (int kk = 0; kk != SPACE_DIM; ++kk)
        for (int ll = 0; ll != SPACE_DIM; ++ll)
          for (int LL = 0; LL != size_symm; ++LL)
            t_DL(ii, jj, LL) += t_D(ii, jj, kk, ll) * t_L(kk, ll, LL);

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / SPACE_DIM; ++rr) {

      auto t_mat =
          get_mat_vector_dtensor_sym(rr, locMat, FTensor::Number<SPACE_DIM>());

      FTensor::Tensor2<double, SPACE_DIM, size_symm> t_tmp;
      t_tmp(i, L) = (t_DL(i, j, L) * (alpha * t_row_diff_base(j)));

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / size_symm; ++cc) {

        t_mat(i, L) -= (t_col_base * t_tmp(i, L));

        ++t_mat;
        ++t_col_base;
      }

      ++t_row_diff_base;
    }

    for (; rr < nb_row_base_functions; ++rr)
      ++t_row_diff_base;

    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticInternalForceLhs_LogStrain_dEP::
    OpCalculatePlasticInternalForceLhs_LogStrain_dEP(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr,
        boost::shared_ptr<HenckyOps::CommonData> common_henky_data_ptr,
        boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr),
      commonHenckyDataPtr(common_henky_data_ptr), mDPtr(m_D_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculatePlasticInternalForceLhs_LogStrain_dEP::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const size_t nb_integration_pts = row_data.getN().size1();
  const size_t nb_row_base_functions = row_data.getN().size2();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);
  auto t_logC_dC = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
      commonHenckyDataPtr->matLogCdC);
  auto t_grad = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(
      *(commonHenckyDataPtr->matGradPtr));

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

    // t_DLogC_dC(i, j, k, l) = t_D(k, l, m, n) * t_logC_dC(m, n, i, j);

    FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
        t_FDLogC_dC;
    t_FDLogC_dC(i, j, k, l) = t_F(i, m) * t_DLogC_dC(m, j, k, l);

    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
    auto t_L = symm_L_tensor();

    FTensor::Tensor3<double, SPACE_DIM, SPACE_DIM, size_symm> t_DL;
    t_DL(i, j, L) = 0;
    for (int ii = 0; ii != SPACE_DIM; ++ii)
      for (int jj = 0; jj != SPACE_DIM; ++jj)
        for (int kk = 0; kk != SPACE_DIM; ++kk)
          for (int ll = 0; ll != SPACE_DIM; ++ll)
            for (int LL = 0; LL != size_symm; ++LL)
              t_DL(ii, jj, LL) += t_FDLogC_dC(ii, jj, kk, ll) * t_L(kk, ll, LL);

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / SPACE_DIM; ++rr) {

      auto t_mat =
          get_mat_vector_dtensor_sym(rr, locMat, FTensor::Number<SPACE_DIM>());

      FTensor::Tensor2<double, SPACE_DIM, size_symm> t_tmp;
      t_tmp(i, L) = (t_DL(i, j, L) * (alpha * t_row_diff_base(j)));

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / size_symm; ++cc) {

        t_mat(i, L) -= (t_col_base * t_tmp(i, L));

        ++t_mat;
        ++t_col_base;
      }

      ++t_row_diff_base;
    }

    for (; rr < nb_row_base_functions; ++rr)
      ++t_row_diff_base;

    ++t_w;
    ++t_logC_dC;
    ++t_grad;
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dEP::OpCalculatePlasticFlowLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  sYmm = false;
}

static inline auto get_mat_tensor_sym_dtensor_sym(size_t rr, MatrixDouble &mat,
                                                  FTensor::Number<2>) {
  return FTensor::Tensor4<FTensor::PackPtr<double *, 3>, 2, 2, 2, 2>{
      &mat(3 * rr + 0, 0), &mat(3 * rr + 0, 1), &mat(3 * rr + 0, 1),
      &mat(3 * rr + 0, 2), &mat(3 * rr + 1, 0), &mat(3 * rr + 1, 1),
      &mat(3 * rr + 1, 1), &mat(3 * rr + 1, 2), &mat(3 * rr + 1, 0),
      &mat(3 * rr + 1, 1), &mat(3 * rr + 1, 1), &mat(3 * rr + 1, 2),
      &mat(3 * rr + 2, 0), &mat(3 * rr + 2, 1), &mat(3 * rr + 2, 1),
      &mat(3 * rr + 2, 2)};
}

static inline auto get_mat_tensor_sym_dtensor_sym(size_t rr, MatrixDouble &mat,
                                                  FTensor::Number<3>) {
  return FTensor::Tensor4<FTensor::PackPtr<double *, 6>, 3, 3, 3, 3>{

      &mat(6 * rr + 0, 0), &mat(6 * rr + 0, 1), &mat(6 * rr + 0, 2),
      &mat(6 * rr + 0, 1), &mat(6 * rr + 0, 3), &mat(6 * rr + 0, 4),
      &mat(6 * rr + 0, 2), &mat(6 * rr + 0, 4), &mat(6 * rr + 0, 5),

      &mat(6 * rr + 1, 0), &mat(6 * rr + 1, 1), &mat(6 * rr + 1, 2),
      &mat(6 * rr + 1, 1), &mat(6 * rr + 1, 3), &mat(6 * rr + 1, 4),
      &mat(6 * rr + 1, 2), &mat(6 * rr + 1, 4), &mat(6 * rr + 1, 5),

      &mat(6 * rr + 2, 0), &mat(6 * rr + 2, 1), &mat(6 * rr + 2, 2),
      &mat(6 * rr + 2, 1), &mat(6 * rr + 2, 3), &mat(6 * rr + 2, 4),
      &mat(6 * rr + 2, 2), &mat(6 * rr + 2, 4), &mat(6 * rr + 2, 5),

      &mat(6 * rr + 1, 0), &mat(6 * rr + 1, 1), &mat(6 * rr + 1, 2),
      &mat(6 * rr + 1, 1), &mat(6 * rr + 1, 3), &mat(6 * rr + 1, 4),
      &mat(6 * rr + 1, 2), &mat(6 * rr + 1, 4), &mat(6 * rr + 1, 5),

      &mat(6 * rr + 3, 0), &mat(6 * rr + 3, 1), &mat(6 * rr + 3, 2),
      &mat(6 * rr + 3, 1), &mat(6 * rr + 3, 3), &mat(6 * rr + 3, 4),
      &mat(6 * rr + 3, 2), &mat(6 * rr + 3, 4), &mat(6 * rr + 3, 5),

      &mat(6 * rr + 4, 0), &mat(6 * rr + 4, 1), &mat(6 * rr + 4, 2),
      &mat(6 * rr + 4, 1), &mat(6 * rr + 4, 3), &mat(6 * rr + 4, 4),
      &mat(6 * rr + 4, 2), &mat(6 * rr + 4, 4), &mat(6 * rr + 4, 5),

      &mat(6 * rr + 2, 0), &mat(6 * rr + 2, 1), &mat(6 * rr + 2, 2),
      &mat(6 * rr + 2, 1), &mat(6 * rr + 2, 3), &mat(6 * rr + 2, 4),
      &mat(6 * rr + 2, 2), &mat(6 * rr + 2, 4), &mat(6 * rr + 2, 5),

      &mat(6 * rr + 4, 0), &mat(6 * rr + 4, 1), &mat(6 * rr + 4, 2),
      &mat(6 * rr + 4, 1), &mat(6 * rr + 4, 3), &mat(6 * rr + 4, 4),
      &mat(6 * rr + 4, 2), &mat(6 * rr + 4, 4), &mat(6 * rr + 4, 5),

      &mat(6 * rr + 5, 0), &mat(6 * rr + 5, 1), &mat(6 * rr + 5, 2),
      &mat(6 * rr + 5, 1), &mat(6 * rr + 5, 3), &mat(6 * rr + 5, 4),
      &mat(6 * rr + 5, 2), &mat(6 * rr + 5, 4), &mat(6 * rr + 5, 5)};
}

MoFEMErrorCode OpCalculatePlasticFlowLhs_dEP::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const size_t nb_integration_pts = row_data.getN().size1();
  const size_t nb_row_base_functions = row_data.getN().size2();
  auto t_w = getFTensor0IntegrationWeight();

  auto t_row_base = row_data.getFTensor0N();

  auto t_f = getFTensor0FromVec(commonDataPtr->plasticSurface);
  auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);
  auto t_flow =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticFlow);

  auto t_D =
      getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
  // t_D_Op would be user operator used to calculate stress, i.e. D_deviator
  auto t_D_Op = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);
  auto t_diff_plastic_strain = diff_tensor();
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;
    double beta = alpha * getTSa();

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / size_symm; ++rr) {

      const double c0 = alpha * t_row_base * t_tau_dot;
      const double c1 = beta * t_row_base;

      auto t_diff_plastic_flow_dstrain = diff_plastic_flow_dstrain(
          t_D_Op,
          diff_plastic_flow_dstress(t_f, t_flow, diff_deviator(diff_tensor())));

      auto t_mat = get_mat_tensor_sym_dtensor_sym(rr, locMat,
                                                  FTensor::Number<SPACE_DIM>());

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / size_symm; ++cc) {

        t_mat(i, j, k, l) +=
            t_col_base *
            (t_D(i, j, m, n) * (c1 * t_diff_plastic_strain(m, n, k, l) +
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

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dTAU::OpCalculatePlasticFlowLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

static inline auto get_mat_tensor_sym_dscalar(size_t rr, MatrixDouble &mat,
                                              FTensor::Number<2>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 2, 2>{
      &mat(3 * rr + 0, 0), &mat(3 * rr + 1, 0), &mat(3 * rr + 1, 0),
      &mat(3 * rr + 2, 0)};
}

static inline auto get_mat_tensor_sym_dscalar(size_t rr, MatrixDouble &mat,
                                              FTensor::Number<3>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 1>, 3, 3>{
      &mat(6 * rr + 0, 0), &mat(6 * rr + 1, 0), &mat(6 * rr + 2, 0),
      &mat(6 * rr + 1, 0), &mat(6 * rr + 3, 0), &mat(6 * rr + 4, 0),
      &mat(6 * rr + 2, 0), &mat(6 * rr + 4, 0), &mat(6 * rr + 5, 0)};
}

MoFEMErrorCode OpCalculatePlasticFlowLhs_dTAU::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const size_t nb_integration_pts = row_data.getN().size1();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_flow =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticFlow);

  auto t_row_base = row_data.getFTensor0N();

  auto t_D =
      getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w * getTSa();

    FTensor::Tensor2_symmetric<double, SPACE_DIM> t_flow_stress;
    t_flow_stress(i, j) = t_D(i, j, m, n) * t_flow(m, n);

    for (size_t rr = 0; rr != AssemblyDomainEleOp::nbRows / size_symm; ++rr) {

      auto t_mat =
          get_mat_tensor_sym_dscalar(rr, locMat, FTensor::Number<SPACE_DIM>());

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols; cc++) {
        t_mat(i, j) -= alpha * t_row_base * t_col_base * t_flow_stress(i, j);
        ++t_mat;
        ++t_col_base;
      }

      ++t_row_base;
    }

    ++t_w;
    ++t_flow;
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dEP::OpCalculateContrainsLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  sYmm = false;
}

auto get_mat_scalar_dtensor_sym(MatrixDouble &mat, FTensor::Number<2>) {
  return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>{
      &mat(0, 0), &mat(0, 1), &mat(0, 2)};
}

auto get_mat_scalar_dtensor_sym(MatrixDouble &mat, FTensor::Number<3>) {
  return FTensor::Tensor1<FTensor::PackPtr<double *, 6>, 6>{
      &mat(0, 0), &mat(0, 1), &mat(0, 2), &mat(0, 3), &mat(0, 4), &mat(0, 5)};
}

MoFEMErrorCode OpCalculateContrainsLhs_dEP::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  locMat.resize(AssemblyDomainEleOp::nbRows, AssemblyDomainEleOp::nbCols,
                false);
  locMat.clear();

  const size_t nb_integration_pts = row_data.getN().size1();
  const size_t nb_row_base_functions = row_data.getN().size2();
  auto t_w = getFTensor0IntegrationWeight();

  auto t_row_base = row_data.getFTensor0N();

  auto t_f = getFTensor0FromVec(commonDataPtr->plasticSurface);
  auto t_tau = getFTensor0FromVec(commonDataPtr->plasticTau);
  auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);
  auto t_flow =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticFlow);
  auto t_D =
      getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
  // Operator set in constructor, i.e. D_deviator
  auto t_D_Op = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;

    auto mat_ptr = locMat.data().begin();
    auto t_diff_constrain_dstrain = diff_constrain_dstrain(
        t_D_Op, diff_constrain_dstress(
                    diff_constrain_df(t_tau_dot, t_f, hardening(t_tau)),
                    t_flow));

    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
    auto t_L = symm_L_tensor();
    auto t_mat =
        get_mat_scalar_dtensor_sym(locMat, FTensor::Number<SPACE_DIM>());

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows; ++rr) {

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / size_symm; cc++) {

        t_mat(L) -= (alpha * t_row_base * t_col_base) *
                    ((

                         t_diff_constrain_dstrain(i, j)

                             ) *
                     t_L(i, j, L));

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

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dTAU::OpCalculateContrainsLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculateContrainsLhs_dTAU::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  locMat.resize(AssemblyDomainEleOp::nbRows, AssemblyDomainEleOp::nbCols,
                false);
  locMat.clear();

  const size_t nb_integration_pts = row_data.getN().size1();
  const size_t nb_row_base_functions = row_data.getN().size2();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_f = getFTensor0FromVec(commonDataPtr->plasticSurface);
  auto t_tau = getFTensor0FromVec(commonDataPtr->plasticTau);
  auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);

  const double t_a = getTSa();

  auto t_row_base = row_data.getFTensor0N();

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = getMeasure() * t_w;

    const double c0 =
        alpha * t_a *
        diff_constrain_ddot_tau(t_tau_dot, t_f, hardening(t_tau));

    const double c1 =
        alpha *
        diff_constrain_dsigma_y(t_tau_dot, t_f, hardening(t_tau)) *
        hardening_dtau(t_tau);

    auto mat_ptr = locMat.data().begin();

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows; ++rr) {

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols; ++cc) {
        *mat_ptr += (c0 + c1) * t_row_base * t_col_base;
        // *mat_ptr += alpha * k_penalty(t_tau) * t_row_base * t_col_base;
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

  MoFEMFunctionReturn(0);
}

}; // namespace PlasticOps
