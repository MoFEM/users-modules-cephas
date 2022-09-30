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

  // boost::shared_ptr<MatrixDouble> matGradPtr;
  // boost::shared_ptr<MatrixDouble> matDPtr;
  // boost::shared_ptr<MatrixDouble> matLogCPlastic;

  // MatrixDouble tempFluxVal;
  VectorDouble templDivFlux;
  VectorDouble tempValDot;

  // inline auto getTempFluxValPtr() {
  //   return boost::shared_ptr<MatrixDouble>(shared_from_this(), &tempFluxVal);
  // }
  inline auto getTempValDotPtr() {
    return boost::shared_ptr<VectorDouble>(shared_from_this(), &tempValDot);
  }
  inline auto getTempDivFluxPtr() {
    return boost::shared_ptr<VectorDouble>(shared_from_this(), &templDivFlux);
  }
};
//! [Common data]

struct OpPlasticHeatProduction : public AssemblyDomainEleOp {
  OpPlasticHeatProduction(const std::string field_name,
                          boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPlasticHeatProduction_dEP : public AssemblyDomainEleOp {
  OpPlasticHeatProduction_dEP(const std::string row_field_name,
                              const std::string col_field_name,
                              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpHdivFluxLargeStrains : public AssemblyDomainEleOp {
  OpHdivFluxLargeStrains(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data_ptr,
                         ScalarFun beta_coeff);

  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data);

protected:
  ScalarFun betaCoeff;
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpHdivHdivLargeStrains : public AssemblyDomainEleOp {
  OpHdivHdivLargeStrains(const std::string row_field_name,
                         const std::string col_field_name,
                         boost::shared_ptr<CommonData> common_data_ptr,
                         ScalarFun beta_coeff);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  ScalarFun betaCoeff;
};

struct OpKCauchyThermoElasticity : public AssemblyDomainEleOp {
  OpKCauchyThermoElasticity(const std::string row_field_name,
                            const std::string col_field_name,
                            boost::shared_ptr<CommonData> common_data_ptr,
                            boost::shared_ptr<MatrixDouble> mDptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<MatrixDouble> mDPtr;
};

struct OpKPiolaThermal : public AssemblyDomainEleOp {
  OpKPiolaThermal(
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
};

struct OpPlasticStressThermal : public DomainEleOp {
  OpPlasticStressThermal(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data_ptr,
                         boost::shared_ptr<MatrixDouble> mDPtr,
                         const double scale = 1);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<MatrixDouble> mDPtr;
  const double scaleStress;
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateContrainsLhs_dT : public AssemblyDomainEleOp {
  OpCalculateContrainsLhs_dT(const std::string row_field_name,
                             const std::string col_field_name,
                             boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

OpPlasticHeatProduction::OpPlasticHeatProduction(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyDomainEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpPlasticHeatProduction::iNtegrate(EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_integration_pts = data.getN().size1();
  const size_t nb_base_functions = data.getN().size2();

  auto t_plastic_strain_dot =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticStrainDot);
  auto t_D =
      getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);

  auto t_w = getFTensor0IntegrationWeight();
  auto &nf = AssemblyDomainEleOp::locF;

  auto t_base = data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha = getMeasure() * t_w;

    // dot(Eps^p): D : dot(Eps^p) * fraction_of_dissipation
    const double beta = (alpha * fraction_of_dissipation) *
                        (t_D(i, j, k, l) * t_plastic_strain_dot(k, l)) *
                        t_plastic_strain_dot(i, j);

    size_t bb = 0;
    for (; bb != AssemblyDomainEleOp::nbRows; ++bb) {
      nf[bb] += beta * t_base;
      ++t_base;
    }
    for (; bb < nb_base_functions; ++bb)
      ++t_base;

    ++t_w;
    ++t_plastic_strain_dot;
  }

  MoFEMFunctionReturn(0);
}

OpHdivFluxLargeStrains::OpHdivFluxLargeStrains(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr,
    ScalarFun beta_coeff)
    : AssemblyDomainEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), betaCoeff(beta_coeff) {}

MoFEMErrorCode
OpHdivFluxLargeStrains::iNtegrate(EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_integration_pts = data.getN().size1();

  const size_t nb_base_functions = data.getN().size2() / 3;
  // get volume or area
  const double vol = getMeasure();
  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_base = data.getFTensor1N<3>();
  // get coordinates
  auto t_coords = getFTensor1CoordsAtGaussPts();
  // create delta kronecker
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  //  get the values of flux
  auto t_vec = getFTensor1FromMat<3, 1>(commonDataPtr->tempFluxVal);
  // calculate gradient
  auto t_grad =
      getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*(commonDataPtr->mGradPtr));

  auto &nf = AssemblyDomainEleOp::locF;
  // loop the integration over all the integration points
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    // define the deformation grandient tensor
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
    // define the determinant of the deformation gradient tensor
    double t_detF;
    double inv_t_detF;
    // calculate deformation gradient tensor
    t_F(i, j) = t_grad(i, j) + t_kd(i, j);
    // calculate deforemation gradient determinant
    if (SPACE_DIM == 2) {
      determinantTensor2by2(t_F, t_detF);
    } else {
      determinantTensor3by3(t_F, t_detF);
    }

    inv_t_detF = 1. / t_detF;

    const double alpha = vol * t_w *
                         betaCoeff(t_coords(0), t_coords(1), t_coords(2)) *
                         inv_t_detF;

    size_t bb = 0;
    for (; bb != AssemblyDomainEleOp::nbRows; ++bb) {
      nf[bb] += alpha * t_base(i) * ((t_F(k, i) * t_F(k, j)) * t_vec(j));
      ++t_base;
    }
    for (; bb < nb_base_functions; ++bb)
      ++t_base;

    ++t_w;
    ++t_vec;
    ++t_coords;
    ++t_grad;
  }

  MoFEMFunctionReturn(0);
}

OpHdivHdivLargeStrains::OpHdivHdivLargeStrains(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr, ScalarFun beta_coeff)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), betaCoeff(beta_coeff) {
  sYmm = false;
}

MoFEMErrorCode
OpHdivHdivLargeStrains::iNtegrate(EntitiesFieldData::EntData &row_data,
                                  EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_integration_pts = row_data.getN().size1();

  const size_t nb_row_base_functions = row_data.getN().size2() / 3;
  // get volume or area
  const double vol = getMeasure();
  // get integration weights
  auto t_w = getFTensor0IntegrationWeight();
  // get base function gradient on rows
  auto t_row_base = row_data.getFTensor1N<3>();
  // get coordinates
  auto t_coords = getFTensor1CoordsAtGaussPts();
  // create delta kronecker
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  // calculate gradient
  auto t_grad =
      getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*(commonDataPtr->mGradPtr));

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    // define the deformation grandient tensor
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_F;
    // define the determinant of the deformation gradient tensor
    double t_detF;
    double inv_t_detF;
    // calculate deformation gradient tensor
    t_F(i, j) = t_grad(i, j) + t_kd(i, j);
    // calculate deforemation gradient determinant
    if (SPACE_DIM == 2) {
      determinantTensor2by2(t_F, t_detF);
    } else {
      determinantTensor3by3(t_F, t_detF);
    }

    inv_t_detF = 1. / t_detF;

    const double beta = vol * betaCoeff(t_coords(0), t_coords(1), t_coords(2));
    const double alpha = beta * t_w * inv_t_detF;

    auto a_mat_ptr = &*AssemblyDomainEleOp::locMat.data().begin();
    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / SPACE_DIM; ++rr) {
      auto t_col_base = col_data.getFTensor1N<3>(gg, 0);

      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols; cc++) {
        // calculate element of local matrix
        (*a_mat_ptr) += alpha * (t_row_base(i) * t_col_base(i));
        ++t_col_base;
        ++a_mat_ptr;
      }

      ++t_row_base;
    }
    for (; rr != nb_row_base_functions; ++rr)
      ++t_row_base;

    ++t_w;
    ++t_coords;
    ++t_grad;
  }

  MoFEMFunctionReturn(0);
}

OpPlasticHeatProduction_dEP::OpPlasticHeatProduction_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode
OpPlasticHeatProduction_dEP::iNtegrate(EntitiesFieldData::EntData &row_data,
                                       EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

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
    for (; rr != AssemblyDomainEleOp::nbRows; ++rr) {

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / size_symm; cc++) {

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

  MoFEMFunctionReturn(0);
}

OpKCauchyThermoElasticity::OpKCauchyThermoElasticity(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> mDptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(mDptr) {
  sYmm = false;
}

MoFEMErrorCode
OpKCauchyThermoElasticity::iNtegrate(EntitiesFieldData::EntData &row_data,
                                     EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const size_t nb_integration_pts = row_data.getN().size1();
  const size_t nb_row_base_functions = row_data.getN().size2();
  auto t_w = getFTensor0IntegrationWeight();

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
  auto t_D =
      getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
  FTensor::Tensor2_symmetric<double, SPACE_DIM> t_eigen_strain;
  t_eigen_strain(i, j) = (t_D(i, j, k, l) * t_kd(k, l)) * coeff_expansion;

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

    double alpha = getMeasure() * t_w;
    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / SPACE_DIM; ++rr) {
      auto t_mat = getFTensor1FromMat<SPACE_DIM, 1>(locMat, rr * SPACE_DIM);
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols; cc++) {

        t_mat(i) -=
            (t_row_diff_base(j) * t_eigen_strain(i, j)) * (t_col_base * alpha);

        ++t_mat;
        ++t_col_base;
      }

      ++t_row_diff_base;
    }
    for (; rr != nb_row_base_functions; ++rr)
      ++t_row_diff_base;

    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

//  New operator for thermo elasticity
OpKPiolaThermal::OpKPiolaThermal(
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

MoFEMErrorCode
OpKPiolaThermal::iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const size_t nb_integration_pts = row_data.getN().size1();
  const size_t nb_row_base_functions = row_data.getN().size2();

  auto t_w = getFTensor0IntegrationWeight();

  constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
  auto t_row_diff_base = row_data.getFTensor1DiffN<SPACE_DIM>();
  auto t_D =
      getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*commonDataPtr->mDPtr);
  FTensor::Tensor2_symmetric<double, SPACE_DIM> t_eigen_strain;
  t_eigen_strain(i, j) = (t_D(i, j, k, l) * t_kd(k, l)) * coeff_expansion;

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

    double alpha = getMeasure() * t_w;

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / SPACE_DIM; ++rr) {
      auto t_mat = getFTensor1FromMat<SPACE_DIM, 1>(locMat, rr * SPACE_DIM);
      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols; cc++) {

        t_mat(i) -=
            (t_row_diff_base(j) * t_eigen_strain(i, j)) * (t_col_base * alpha);

        ++t_mat;
        ++t_col_base;
      }

      ++t_row_diff_base;
    }
    for (; rr != nb_row_base_functions; ++rr)
      ++t_row_diff_base;

    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

OpPlasticStressThermal::OpPlasticStressThermal(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<MatrixDouble> m_D_ptr, const double scale)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr), scaleStress(scale), mDPtr(m_D_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Calculate stress]
MoFEMErrorCode OpPlasticStressThermal::doWork(int side, EntityType type,
                                              EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  commonDataPtr->mStressPtr->resize((SPACE_DIM * (SPACE_DIM + 1)) / 2,
                                    nb_gauss_pts);
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  auto t_D = getFTensor4DdgFromMat<SPACE_DIM, SPACE_DIM, 0>(*mDPtr);
  auto t_strain =
      getFTensor2SymmetricFromMat<SPACE_DIM>(*(commonDataPtr->mStrainPtr));
  auto t_plastic_strain =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticStrain);
  auto t_stress =
      getFTensor2SymmetricFromMat<SPACE_DIM>(*(commonDataPtr->mStressPtr));
  auto t_temp = getFTensor0FromVec(commonDataPtr->tempVal);
  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    t_stress(i, j) =
        t_D(i, j, k, l) * (t_strain(k, l) - t_plastic_strain(k, l)) -
        t_D(i, j, k, l) * t_kd(k, l) * (t_temp - ref_temp) * coeff_expansion;
    t_stress(i, j) /= scaleStress;
    ++t_strain;
    ++t_plastic_strain;
    ++t_stress;
    ++t_temp;
  }

  MoFEMFunctionReturn(0);
}
//! [Calculate stress]

//

OpCalculateContrainsLhs_dT::OpCalculateContrainsLhs_dT(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode
OpCalculateContrainsLhs_dT::iNtegrate(EntitiesFieldData::EntData &row_data,
                                      EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  auto &locMat = AssemblyDomainEleOp::locMat;

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
        alpha *
        diff_constrain_dsigma_y(t_tau_dot, t_f, hardening(t_tau, t_temp)) *
        hardening_dtemp(t_tau, t_temp);

    auto mat_ptr = locMat.data().begin();

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows; ++rr) {

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols; ++cc) {
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

  MoFEMFunctionReturn(0);
}

} // namespace PlasticThermalOps