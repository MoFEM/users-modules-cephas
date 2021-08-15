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

/** \file PlasticOpsSmallStrains.hpp
 * \example PlasticOpsSmallStrains.hpp
 */

namespace PlasticOps {

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

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
    locMat.clear();

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

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

    FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
        t_diff_grad;
    t_diff_grad(i, j, k, l) = t_kd(i, k) * t_kd(j, l);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

      double alpha = getMeasure() * t_w;
      auto t_diff_plastic_flow_dstrain = diff_plastic_flow_dstrain(
          t_D,
          diff_plastic_flow_dstress(t_f, t_flow, diff_deviator(diff_tensor())));
      FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> t_flow_stress_dstrain;
      t_flow_stress_dstrain(i, j, k, l) =
          t_D(i, j, m, n) * t_diff_plastic_flow_dstrain(m, n, k, l);

      FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
          t_diff_plastic_flow_stress_dgrad;
      t_diff_plastic_flow_stress_dgrad(i, j, k, l) =
          t_flow_stress_dstrain(i, j, m, n) * t_diff_grad(m, n, k, l);

      size_t rr = 0;
      for (; rr != nb_row_dofs / size_symm; ++rr) {

        auto t_mat = get_mat_tensor_sym_dvector(rr, locMat,
                                                FTensor::Number<SPACE_DIM>());

        const double c0 = alpha * t_row_base * t_tau_dot;

        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / SPACE_DIM; ++cc) {

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

    CHKERR MatSetValues<EssentialBcStorage>(
        getSNESB(), row_data, col_data, &*locMat.data().begin(), ADD_VALUES);
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

    locMat.resize(nb_row_dofs, nb_col_dofs, false);
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

    // constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    // FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
    //     t_diff_grad;
    // t_diff_grad(i, j, k, l) = t_kd(i, k) * t_kd(j, l);

    auto t_diff_grad_symmetrise = diff_symmetrize();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      auto t_diff_constrain_dstrain = diff_constrain_dstrain(
          t_D,
          diff_constrain_dstress(
              diff_constrain_df(t_tau_dot, t_f, hardening(t_tau, 0)), t_flow));
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_diff_constrain_dgrad;
      t_diff_constrain_dgrad(k, l) =
          t_diff_constrain_dstrain(i, j) * t_diff_grad_symmetrise(i, j, k, l);

      auto t_mat = get_mat_scalar_dvector(locMat, FTensor::Number<SPACE_DIM>());
      size_t rr = 0;
      for (; rr != nb_row_dofs; ++rr) {

        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / SPACE_DIM; cc++) {

          t_mat(i) += alpha * t_row_base * (t_diff_constrain_dgrad(i, j)) *
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
      ++t_w;
    }

    CHKERR MatSetValues<EssentialBcStorage>(
        getSNESB(), row_data, col_data, &*locMat.data().begin(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}


};