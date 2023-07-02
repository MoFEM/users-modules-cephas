

/** \file PlasticOpsGeneric.hpp
 * \example PlasticOpsGeneric.hpp
 */

namespace PlasticOps {

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

  const size_t nb_gauss_pts = commonDataPtr->mStressPtr->size2();
  auto t_stress =
      getFTensor2SymmetricFromMat<DIM>(*(commonDataPtr->mStressPtr));

  commonDataPtr->plasticSurface.resize(nb_gauss_pts, false);
  commonDataPtr->plasticFlow.resize(size_symm, nb_gauss_pts, false);
  auto t_flow = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->plasticFlow);

  for (auto &f : commonDataPtr->plasticSurface) {
    f = platsic_surface(deviator(t_stress, trace(t_stress)));
    auto t_flow_tmp =
        plastic_flow(f,

                     deviator(t_stress, trace(t_stress)),

                     diff_deviator(diff_tensor(FTensor::Number<DIM>())));
    t_flow(i, j) = t_flow_tmp(i, j);
    ++t_flow;
    ++t_stress;
  }

  MoFEMFunctionReturn(0);
}

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

  const size_t nb_gauss_pts = DomainEleOp::getGaussPts().size2();
  auto t_w = DomainEleOp::getFTensor0IntegrationWeight();
  auto t_tau = getFTensor0FromVec(commonDataPtr->plasticTau);
  auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);
  auto t_f = getFTensor0FromVec(commonDataPtr->plasticSurface);
  auto t_flow = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->plasticFlow);
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

  commonDataPtr->resC.resize(nb_gauss_pts, false);
  commonDataPtr->resCdTau.resize(nb_gauss_pts, false);
  commonDataPtr->resCdStrain.resize(size_symm, nb_gauss_pts, false);
  commonDataPtr->resCdStrainDot.resize(size_symm, nb_gauss_pts, false);
  commonDataPtr->resFlow.resize(size_symm, nb_gauss_pts, false);
  commonDataPtr->resFlowDtau.resize(size_symm, nb_gauss_pts, false);
  commonDataPtr->resFlowDstrain.resize(size_symm * size_symm, nb_gauss_pts,
                                       false);
  commonDataPtr->resFlowDstrainDot.resize(size_symm * size_symm, nb_gauss_pts,
                                          false);

  commonDataPtr->resC.clear();
  commonDataPtr->resCdTau.clear();
  commonDataPtr->resCdStrain.clear();
  commonDataPtr->resCdStrainDot.clear();
  commonDataPtr->resFlow.clear();
  commonDataPtr->resFlowDtau.clear();
  commonDataPtr->resFlowDstrain.clear();
  commonDataPtr->resFlowDstrainDot.clear();

  auto &params = commonDataPtr->blockParams;

  auto t_res_c = getFTensor0FromVec(commonDataPtr->resC);
  auto t_res_c_dtau = getFTensor0FromVec(commonDataPtr->resCdTau);
  auto t_res_c_dstrain =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resCdStrain);
  auto t_res_c_dstrain_dot =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resCdStrainDot);
  auto t_res_flow = getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resFlow);
  auto t_res_flow_dtau =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resFlowDtau);
  auto t_res_flow_dstrain =
      getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->resFlowDstrain);
  auto t_res_flow_dstrain_dot =
      getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->resFlowDstrainDot);

  auto next = [&]() {
    ++t_tau;
    ++t_tau_dot;
    ++t_f;
    ++t_flow;
    ++t_plastic_strain_dot;
    ++t_stress;
    ++t_res_c;
    ++t_res_c_dtau;
    ++t_res_c_dstrain;
    ++t_res_c_dstrain_dot;
    ++t_res_flow;
    ++t_res_flow_dtau;
    ++t_res_flow_dstrain;
    ++t_res_flow_dstrain_dot;
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
      const auto ww =
          w(eqiv, t_tau_dot, t_f,
            hardening(t_tau, params[CommonData::H], params[CommonData::QINF],
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
        hardening(t_tau, params[CommonData::H], params[CommonData::QINF],
                  params[CommonData::BISO], params[CommonData::SIGMA_Y]);
    const auto d_sigma_y =
        hardening_dtau(t_tau, params[CommonData::H], params[CommonData::QINF],
                       params[CommonData::BISO]);

    auto ww = w(eqiv, t_tau_dot, t_f, sigma_y, params[CommonData::SIGMA_Y]);
    auto abs_ww = constrain_abs(ww);
    auto sign_ww = constrian_sign(ww);

    auto c = constraint(eqiv, t_tau_dot, t_f, sigma_y, abs_ww,
                        params[CommonData::VIS_H], params[CommonData::SIGMA_Y]);
    auto c_dot_tau = diff_constrain_ddot_tau(
        sign_ww, eqiv, params[CommonData::VIS_H], params[CommonData::SIGMA_Y]);
    auto c_equiv = diff_constrain_deqiv(sign_ww, eqiv, t_tau_dot,
                                        params[CommonData::SIGMA_Y]);
    auto c_sigma_y = diff_constrain_dsigma_y(sign_ww);
    auto c_f = diff_constrain_df(sign_ww);

    auto t_dev_stress = deviator(t_stress, trace(t_stress));
    FTensor::Tensor2_symmetric<double, DIM> t_flow_dir;
    t_flow_dir(k, l) = 1.5 * (t_dev_stress(I, J) * t_diff_deviator(I, J, k, l));
    FTensor::Tensor2_symmetric<double, DIM> t_flow_dstrain;
    t_flow_dstrain(i, j) = t_flow(k, l) * t_D_Op(k, l, i, j);

    auto get_res_c = [&]() { return c; };

    auto get_res_c_dstrain = [&](auto &t_diff_res) {
      t_diff_res(i, j) = c_f * t_flow_dstrain(i, j);
    };

    auto get_res_c_dstrain_dot = [&](auto &t_diff_res) {
      t_diff_res(i, j) = (DomainEleOp::getTSa() * c_equiv) * t_diff_eqiv(i, j);
    };

    auto get_res_c_dtau = [&]() {
      return DomainEleOp::getTSa() * c_dot_tau + c_sigma_y * d_sigma_y;
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

    auto get_res_flow_dstrain_dot = [&](auto &t_res_flow_dstrain_dot) {
      const auto a = sigma_y;
      t_res_flow_dstrain_dot(m, n, k, l) =
          (a * DomainEleOp::getTSa()) * t_diff_plastic_strain(m, n, k, l);
    };

    t_res_c = get_res_c();
    get_res_flow(t_res_flow);

    if (DomainEleOp::getTSCtx() == TSMethod::TSContext::CTX_TSSETIJACOBIAN) {
      t_res_c_dtau = get_res_c_dtau();
      get_res_c_dstrain(t_res_c_dstrain);
      get_res_c_dstrain_dot(t_res_c_dstrain_dot);
      get_res_flow_dtau(t_res_flow_dtau);
      get_res_flow_dstrain(t_res_flow_dstrain);
      get_res_flow_dstrain_dot(t_res_flow_dstrain_dot);
    }

    next();
  }

  MoFEMFunctionReturn(0);
}

OpPlasticStress::OpPlasticStress(const std::string field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr,
                                 boost::shared_ptr<MatrixDouble> m_D_ptr,
                                 const double scale)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), scaleStress(scale), mDPtr(m_D_ptr) {
  // Operator is only executed for vertices
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
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {}

MoFEMErrorCode
OpCalculatePlasticFlowRhs::iNtegrate(EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  const auto nb_integration_pts = getGaussPts().size2();
  const auto nb_base_functions = data.getN().size2();

  auto t_res_flow =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->resFlow);

  auto t_L = symm_L_tensor(FTensor::Number<SPACE_DIM>());

  auto next = [&]() { ++t_res_flow; };

  auto t_w = getFTensor0IntegrationWeight();
  auto t_base = data.getFTensor0N();
  auto &nf = AssemblyDomainEleOp::locF;
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;
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

OpCalculateConstraintsRhs::OpCalculateConstraintsRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {}

MoFEMErrorCode
OpCalculateConstraintsRhs::iNtegrate(EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_integration_pts = getGaussPts().size2();
  const size_t nb_base_functions = data.getN().size2();

  auto t_res_c = getFTensor0FromVec(commonDataPtr->resC);

  auto next = [&]() { ++t_res_c; };

  auto t_w = getFTensor0IntegrationWeight();
  auto &nf = AssemblyDomainEleOp::locF;
  auto t_base = data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = getMeasure() * t_w;
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

MoFEMErrorCode
OpCalculatePlasticFlowLhs_dEP::iNtegrate(EntitiesFieldData::EntData &row_data,
                                         EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const auto nb_integration_pts = getGaussPts().size2();
  const auto nb_row_base_functions = row_data.getN().size2();

  auto t_res_flow_dstrain = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
      commonDataPtr->resFlowDstrain);
  auto t_res_flow_dstrain_dot = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
      commonDataPtr->resFlowDstrainDot);
  auto t_L = symm_L_tensor(FTensor::Number<SPACE_DIM>());

  auto next = [&]() {
    ++t_res_flow_dstrain;
    ++t_res_flow_dstrain_dot;
  };

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;
    ++t_w;

    FTensor::Tensor2<double, size_symm, size_symm> t_res_mat;
    t_res_mat(O, L) =
        alpha * (t_L(i, j, O) * ((t_res_flow_dstrain_dot(i, j, k, l) -
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

OpCalculatePlasticFlowLhs_dTAU::OpCalculatePlasticFlowLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  sYmm = false;
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

MoFEMErrorCode OpCalculatePlasticFlowLhs_dTAU::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  const auto nb_integration_pts = getGaussPts().size2();
  const size_t nb_row_base_functions = row_data.getN().size2();
  auto &locMat = AssemblyDomainEleOp::locMat;

  const auto type = getFEType();
  const auto nb_nodes = moab::CN::VerticesPerEntity(type);

  auto t_res_flow_dtau =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->resFlowDtau);

  auto t_L = symm_L_tensor(FTensor::Number<SPACE_DIM>());

  auto next = [&]() { ++t_res_flow_dtau; };

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;
    ++t_w;
    FTensor::Tensor1<double, size_symm> t_res_vec;
    t_res_vec(L) = alpha * (t_res_flow_dtau(i, j) * t_L(i, j, L));
    next();

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / size_symm; ++rr) {
      auto t_mat =
          get_mat_tensor_sym_dscalar(rr, locMat, FTensor::Number<SPACE_DIM>());
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

OpCalculateConstraintsLhs_dEP::OpCalculateConstraintsLhs_dEP(
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

MoFEMErrorCode
OpCalculateConstraintsLhs_dEP::iNtegrate(EntitiesFieldData::EntData &row_data,
                                         EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  const auto nb_integration_pts = getGaussPts().size2();
  const auto nb_row_base_functions = row_data.getN().size2();

  auto t_c_dstrain =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->resCdStrain);
  auto t_c_dstrain_dot =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->resCdStrainDot);

  auto next = [&]() {
    ++t_c_dstrain;
    ++t_c_dstrain_dot;
  };

  auto t_L = symm_L_tensor(FTensor::Number<SPACE_DIM>());

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (auto gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = getMeasure() * t_w;
    ++t_w;

    FTensor::Tensor1<double, size_symm> t_res_vec;
    t_res_vec(L) = t_L(i, j, L) * (t_c_dstrain_dot(i, j) - t_c_dstrain(i, j));
    next();

    auto t_mat =
        get_mat_scalar_dtensor_sym(locMat, FTensor::Number<SPACE_DIM>());
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

OpCalculateConstraintsLhs_dTAU::OpCalculateConstraintsLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculateConstraintsLhs_dTAU::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  const auto nb_integration_pts = getGaussPts().size2();
  const auto nb_row_base_functions = row_data.getN().size2();

  auto t_res_c_dtau = getFTensor0FromVec(commonDataPtr->resCdTau);
  auto next = [&]() { ++t_res_c_dtau; };

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = getMeasure() * t_w;
    ++t_w;

    const auto res = alpha * (t_res_c_dtau);
    next();

    auto mat_ptr = locMat.data().begin();
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
