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

/** \file PlasticThermalOps.hpp
 * \example PlasticThermalOps.hpp
 */

namespace PlasticThermalOps {

//! [Common data]
struct CommonData : public PlasticOps::CommonData {

  MatrixDouble tempFluxVal;
  VectorDouble templDivFlux;
  VectorDouble tempValDot;

  inline auto getTempFluxValPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &tempFluxVal);
  }
  inline auto getTempValDotPtr() {
    return boost::shared_ptr<VectorDouble>(shared_from_this(), &tempValDot);
  }
  inline auto getTempDivFluxPtr() {
    return boost::shared_ptr<VectorDouble>(shared_from_this(), &templDivFlux);
  }
};
//! [Common data]

struct OpPlasticHeatProduction : public DomainEleOp {
  OpPlasticHeatProduction(const std::string field_name,
                          boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPlasticHeatProduction_dEP : public DomainEleOp {
  OpPlasticHeatProduction_dEP(const std::string row_field_name,
                             const std::string col_field_name,
                             boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpCalculateContrainsLhs_dT : public DomainEleOp {
  OpCalculateContrainsLhs_dT(const std::string row_field_name,
                             const std::string col_field_name,
                             boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

OpPlasticHeatProduction::OpPlasticHeatProduction(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpPlasticHeatProduction::doWork(int side, EntityType type,
                                               EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const size_t nb_integration_pts = data.getN().size1();
    const size_t nb_base_functions = data.getN().size2();

    auto t_plastic_strain_dot =
        getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticStrainDot);
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);

    auto t_w = getFTensor0IntegrationWeight();

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_base = data.getFTensor0N();
    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha = getMeasure() * t_w;

      // dot(Eps^p): D : dot(Eps^p) * fraction_of_dissipation
      const double beta = (alpha * fraction_of_dissipation) *
                          (t_D(i, j, k, l) * t_plastic_strain_dot(k, l)) *
                          t_plastic_strain_dot(i, j);

      size_t bb = 0;
      for (; bb != nb_dofs; ++bb) {
        nf[bb] += beta * t_base;
        ++t_base;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_w;
      ++t_plastic_strain_dot;
    }

    CHKERR VecSetValues<EssentialBcStorage>(getTSf(), data, nf.data(),
                                            ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpPlasticHeatProduction_dEP::OpPlasticHeatProduction_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpPlasticHeatProduction_dEP::doWork(int row_side, int col_side,
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

    auto t_plastic_strain_dot =
        getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticStrainDot);
    auto t_D =
        getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);

    const double ts_a = getTSa();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w * fraction_of_dissipation * ts_a * 2;

      auto mat_ptr = locMat.data().begin();

      constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
      auto t_L = symm_L_tensor();
      auto t_mat =
          get_mat_scalar_dtensor_sym(locMat, FTensor::Number<SPACE_DIM>());

      FTensor::Tensor1<double, size_symm> t_D_ep_dot;
      t_D_ep_dot(L) =
          ((t_D(i, j, k, l) * t_plastic_strain_dot(k, l)) * t_L(i, j, L)) *
          (alpha);

      size_t rr = 0;
      for (; rr != nb_row_dofs; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs / size_symm; cc++) {

          t_mat(L) += (t_row_base * t_col_base) * (t_D_ep_dot(L));

          ++t_mat;
          ++t_col_base;
        }

        ++t_row_base;
      }
      for (; rr != nb_row_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
      ++t_plastic_strain_dot;
    }

    CHKERR MatSetValues<EssentialBcStorage>(
        getSNESB(), row_data, col_data, &*locMat.data().begin(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dT::OpCalculateContrainsLhs_dT(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpCalculateContrainsLhs_dT::doWork(int row_side, int col_side,
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
    auto t_f = getFTensor0FromVec(commonDataPtr->plasticSurface);
    auto t_tau = getFTensor0FromVec(commonDataPtr->plasticTau);
    if (commonDataPtr->tempVal.size() != nb_integration_pts) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Temperature not set");
    }
    auto t_temp = getFTensor0FromVec(commonDataPtr->tempVal);
    auto t_tau_dot = getFTensor0FromVec(commonDataPtr->plasticTauDot);

    const double t_a = getTSa();

    auto t_row_base = row_data.getFTensor0N();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha = getMeasure() * t_w;
      const double c1 =
          alpha

          * diff_constrain_dsigma_y(t_tau_dot, t_f, hardening(t_tau, t_temp)) *
          hardening_dtemp(t_tau, t_temp);

      auto mat_ptr = locMat.data().begin();

      size_t rr = 0;
      for (; rr != nb_row_dofs; ++rr) {

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != nb_col_dofs; ++cc) {
          *mat_ptr += c1 * t_row_base * t_col_base;
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
      ++t_temp;
      ++t_tau_dot;
    }

    CHKERR MatSetValues<EssentialBcStorage>(
        getSNESB(), row_data, col_data, &*locMat.data().begin(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

} // namespace PlasticThermalOps