/** \file PlasticOpsLargeStrains.hpp
 * \example PlasticOpsLargeStrains.hpp
 */

namespace PlasticOps {

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

  if (AssemblyDomainEleOp::rowType == MBVERTEX &&
      AssemblyDomainEleOp::rowSide == 0) {
    resDiff.resize(SPACE_DIM * SPACE_DIM * size_symm, nb_integration_pts,
                   false);
    auto t_res_diff =
        getFTensor3FromMat<SPACE_DIM, SPACE_DIM, size_symm>(resDiff);
    auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);
    auto t_logC_dC = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
        commonHenckyDataPtr->matLogCdC);
    auto t_grad = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(
        *(commonHenckyDataPtr->matGradPtr));
    auto t_L = symm_L_tensor();
    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
      t_F(i, j) = t_grad(i, j) + t_kd(i, j);
      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          t_DLogC_dC;
      t_DLogC_dC(i, j, k, l) = t_D(m, n, k, l) * t_logC_dC(m, n, i, j);
      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          t_FDLogC_dC;
      t_FDLogC_dC(i, j, k, l) = t_F(i, m) * t_DLogC_dC(m, j, k, l);
      FTensor::Tensor3<double, SPACE_DIM, SPACE_DIM, size_symm> t_DL;
      t_res_diff(i, j, L) = t_FDLogC_dC(i, j, k, l) * t_L(k, l, L);
      ++t_logC_dC;
      ++t_grad;
      ++t_res_diff;
    }
  }

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
  auto t_res_diff =
      getFTensor3FromMat<SPACE_DIM, SPACE_DIM, size_symm>(resDiff);

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / SPACE_DIM; ++rr) {

      auto t_mat =
          get_mat_vector_dtensor_sym(rr, locMat, FTensor::Number<SPACE_DIM>());

      FTensor::Tensor2<double, SPACE_DIM, size_symm> t_tmp;
      t_tmp(i, L) = (t_res_diff(i, j, L) * (alpha * t_row_diff_base(j)));

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
    ++t_res_diff;
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_LogStrain_dU::OpCalculatePlasticFlowLhs_LogStrain_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<HenckyOps::CommonData> comman_henky_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr),
      commonHenckyDataPtr(comman_henky_data_ptr), mDPtr(m_D_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculatePlasticFlowLhs_LogStrain_dU::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const auto nb_integration_pts = getGaussPts().size2();
  const auto nb_row_base_functions = row_data.getN().size2();

  if (AssemblyDomainEleOp::colType == MBVERTEX &&
      AssemblyDomainEleOp::colSide == 0) {

    resDiff.resize(size_symm * SPACE_DIM * SPACE_DIM, nb_integration_pts,
                   false);
    auto t_res_diff =
        getFTensor3FromMat<size_symm, SPACE_DIM, SPACE_DIM>(resDiff);

    auto t_res_flow_dstrain = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
        commonDataPtr->resFlowDstrain);
    auto t_grad =
        getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*(commonDataPtr->mGradPtr));
    auto t_logC_dC = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
        commonHenckyDataPtr->matLogCdC);

    auto next = [&]() {
      ++t_res_flow_dstrain;
      ++t_grad;
      ++t_logC_dC;
      ++t_res_diff;
    };

    auto t_L = symm_L_tensor();
    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
        t_diff_grad;
    t_diff_grad(i, j, k, l) = t_kd(i, k) * t_kd(j, l);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_diff_ls_dlogC_dC;
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          t_dC_dF;

      t_diff_ls_dlogC_dC(i, j, k, l) =
          (t_res_flow_dstrain(i, j, m, n)) * (t_logC_dC(m, n, k, l) / 2);

      t_F(i, j) = t_grad(i, j) + t_kd(i, j);
      t_dC_dF(i, j, k, l) = (t_kd(i, l) * t_F(k, j)) + (t_kd(j, l) * t_F(k, i));

      FTensor::Tensor3<double, size_symm, SPACE_DIM, SPACE_DIM> t_res_tens;
      t_res_diff(L, i, j) =
          (t_L(m, n, L) * t_diff_ls_dlogC_dC(m, n, k, l)) * t_dC_dF(k, l, i, j);
      next();
    }
  }

  auto t_res_diff =
      getFTensor3FromMat<size_symm, SPACE_DIM, SPACE_DIM>(resDiff);

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;
    ++t_w;

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / size_symm; ++rr) {
      const auto row_base = alpha * t_row_base;
      auto t_mat =
          get_mat_tensor_sym_dvector(rr, locMat, FTensor::Number<SPACE_DIM>());
      auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);
      for (auto cc = 0; cc != AssemblyDomainEleOp::nbCols / SPACE_DIM; ++cc) {
        t_mat(L, l) += row_base * (t_res_diff(L, l, k) * t_col_diff_base(k));
        ++t_mat;
        ++t_col_diff_base;
      }
      ++t_row_base;
    }

    for (; rr < nb_row_base_functions; ++rr)
      ++t_row_base;

    ++t_res_diff;
  }

  MoFEMFunctionReturn(0);
}

OpCalculateConstraintsLhs_LogStrain_dU::OpCalculateConstraintsLhs_LogStrain_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<HenckyOps::CommonData> comman_henky_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr),
      commonHenckyDataPtr(comman_henky_data_ptr), mDPtr(m_D_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculateConstraintsLhs_LogStrain_dU::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  locMat.resize(AssemblyDomainEleOp::nbRows, AssemblyDomainEleOp::nbCols,
                false);
  locMat.clear();

  const auto nb_integration_pts = getGaussPts().size2();
  const auto nb_row_base_functions = row_data.getN().size2();

  if (AssemblyDomainEleOp::colType == MBVERTEX &&
      AssemblyDomainEleOp::colSide == 0) {

    resDiff.resize(SPACE_DIM * SPACE_DIM, nb_integration_pts, false);
    auto t_res_diff = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(resDiff);
    auto t_grad =
        getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*(commonDataPtr->mGradPtr));
    auto t_logC_dC = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
        commonHenckyDataPtr->matLogCdC);
    auto t_c_dstrain =
        getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->resCdStrain);

    auto next = [&]() {
      ++t_grad;
      ++t_logC_dC;
      ++t_c_dstrain;
      ++t_res_diff;
    };

    auto t_diff_grad_symmetrise = diff_symmetrize();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      FTensor::Tensor2_symmetric<double, SPACE_DIM> t_diff_ls_dlog_c;
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          t_dC_dF;

      t_diff_ls_dlog_c(k, l) =
          (t_c_dstrain(i, j)) * (t_logC_dC(i, j, k, l) / 2);
      t_F(i, j) = t_grad(i, j) + t_kd(i, j);
      t_dC_dF(i, j, k, l) = (t_kd(i, l) * t_F(k, j)) + (t_kd(j, l) * t_F(k, i));

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_res_mat;
      t_res_diff(i, j) = (t_diff_ls_dlog_c(k, l) * t_dC_dF(k, l, i, j));
      next();
    }
  }

  auto t_res_diff = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(resDiff);

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (auto gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;
    ++t_w;

    auto t_mat = getFTensor1FromPtr<SPACE_DIM>(locMat.data().data());
    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows; ++rr) {
      const auto row_base = alpha * t_row_base;
      auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / SPACE_DIM; cc++) {
        t_mat(i) += row_base * (t_res_diff(i, j) * t_col_diff_base(j));
        ++t_mat;
        ++t_col_diff_base;
      }
      ++t_row_base;
    }
    for (; rr != nb_row_base_functions; ++rr)
      ++t_row_base;

    ++t_res_diff;
  }

  MoFEMFunctionReturn(0);
}

}; // namespace PlasticOps