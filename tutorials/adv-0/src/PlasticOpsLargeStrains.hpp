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

/** \file PlasticOpsLargeStrains.hpp
 * \example PlasticOpsLargeStrains.hpp
 */

namespace PlasticOps {

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

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

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
  // t_D_Op is D_deviator set by user in constructor
  auto t_D_Op = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);

  auto t_grad =
      getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*(commonDataPtr->mGradPtr));
  auto t_logC_dC = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
      commonHenckyDataPtr->matLogCdC);

  auto t_diff_symmetrize = diff_symmetrize();
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

    double alpha = getMeasure() * t_w;

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
    t_F(i, j) = t_grad(i, j) + t_kd(i, j);
    FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM> dC_dF;
    dC_dF(i, j, k, l) = (t_kd(i, l) * t_F(k, j)) + (t_kd(j, l) * t_F(k, i));

    auto t_diff_plastic_flow_dstrain = diff_plastic_flow_dstrain(
        t_D_Op,
        diff_plastic_flow_dstress(t_f, t_flow, diff_deviator(diff_tensor())));
    FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_flow_stress_dlogC;
    t_flow_stress_dlogC(i, j, k, l) =
        t_D(i, j, m, n) * t_diff_plastic_flow_dstrain(m, n, k, l);

    FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
        t_diff_plastic_flow_stress_diff_symm;
    t_diff_plastic_flow_stress_diff_symm(i, j, k, l) =
        t_flow_stress_dlogC(i, j, m, n) * t_diff_symmetrize(m, n, k, l);

    FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
        t_diff_plastic_flow_stress_dC;
    t_diff_plastic_flow_stress_dC(i, j, k, l) =
        t_flow_stress_dlogC(i, j, m, n) * (t_logC_dC(m, n, k, l) / 2);

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
    for (; rr != AssemblyDomainEleOp::nbRows / size_symm; ++rr) {

      auto t_mat =
          get_mat_tensor_sym_dvector(rr, locMat, FTensor::Number<SPACE_DIM>());

      const double c0 = alpha * t_row_base * t_tau_dot;

      auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / SPACE_DIM; ++cc) {

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

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_LogStrain_dU::OpCalculateContrainsLhs_LogStrain_dU(
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

MoFEMErrorCode OpCalculateContrainsLhs_LogStrain_dU::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  locMat.resize(AssemblyDomainEleOp::nbRows, AssemblyDomainEleOp::nbCols,
                false);
  locMat.clear();

  const size_t nb_integration_pts = row_data.getN().size1();
  const size_t nb_row_base_functions = row_data.getN().size2();
  auto t_w = getFTensor0IntegrationWeight();

  auto t_row_base = row_data.getFTensor0N();

  auto t_f = getFTensor0FromVec(commonDataPtr->plasticSurface);
  auto t_tau = getFTensor0FromVec(commonDataPtr->plasticTau);
  if (commonDataPtr->tempVal.size() != nb_integration_pts) {
    commonDataPtr->tempVal.resize(nb_integration_pts, 0);
    commonDataPtr->tempVal.clear();
  }
  auto t_temp = getFTensor0FromVec(commonDataPtr->tempVal);
  auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);
  auto t_flow =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticFlow);
  auto t_stress =
      getFTensor2SymmetricFromMat<SPACE_DIM>(*(commonDataPtr->mStressPtr));
  auto t_D =
      getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
  // Operator set in constructor, i.e. D_deviator
  auto t_D_Op = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);

  auto t_grad =
      getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*(commonDataPtr->mGradPtr));
  auto t_logC_dC = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM>(
      commonHenckyDataPtr->matLogCdC);

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = getMeasure() * t_w;

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
    t_F(i, j) = t_grad(i, j) + t_kd(i, j);
    FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM> dC_dF;
    dC_dF(i, j, k, l) = (t_kd(i, l) * t_F(k, j)) + (t_kd(j, l) * t_F(k, i));

    auto t_diff_constrain_dstrain = diff_constrain_dstrain(
        t_D_Op, diff_constrain_dstress(
                    diff_constrain_df(t_tau_dot, t_f, hardening(t_tau, t_temp)),
                    t_flow));

    FTensor::Tensor2_symmetric<double, SPACE_DIM> t_diff_constrain_dlog_c;
    t_diff_constrain_dlog_c(k, l) =
        t_diff_constrain_dstrain(i, j) * (t_logC_dC(i, j, k, l) / 2);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_diff_constrain_dgrad;
    t_diff_constrain_dgrad(k, l) =
        t_diff_constrain_dlog_c(i, j) * dC_dF(i, j, k, l);

    auto t_mat = getFTensor1FromPtr<SPACE_DIM>(locMat.data().data());

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows; ++rr) {

      auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / SPACE_DIM; cc++) {

        t_mat(i) += alpha * t_row_base *
                    (

                        t_diff_constrain_dgrad(i, j)

                            ) *
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
    ++t_temp;
    ++t_tau_dot;
    ++t_flow;
    ++t_stress;
    ++t_w;
    ++t_grad;
    ++t_logC_dC;
  }

  MoFEMFunctionReturn(0);
}

}; // namespace PlasticOps