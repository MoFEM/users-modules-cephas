

/** \file PlasticOpsSmallStrains.hpp
 * \example PlasticOpsSmallStrains.hpp
 */

namespace PlasticOps {

template <int DIM, typename AssemblyDomainEleOp>
OpCalculatePlasticInternalForceLhs_dEPImpl<DIM, GAUSS, AssemblyDomainEleOp>::
    OpCalculatePlasticInternalForceLhs_dEPImpl(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr,
        boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          AssemblyDomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  AssemblyDomainEleOp::sYmm = false;
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

template <int DIM, typename AssemblyDomainEleOp>
MoFEMErrorCode
OpCalculatePlasticInternalForceLhs_dEPImpl<DIM, GAUSS, AssemblyDomainEleOp>::
    iNtegrate(EntitiesFieldData::EntData &row_data,
              EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
  FTensor::Index<'L', size_symm> L;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const size_t nb_integration_pts = row_data.getN().size1();
  const size_t nb_row_base_functions = row_data.getN().size2();

  auto t_D = getFTensor4DdgFromMat<DIM, DIM, 0>(*mDPtr);
  auto t_L = symm_L_tensor(FTensor::Number<DIM>());

  FTensor::Dg<double, DIM, size_symm> t_DL;
  t_DL(i, j, L) = t_D(i, j, k, l) * t_L(k, l, L);

  auto t_w = AssemblyDomainEleOp::getFTensor0IntegrationWeight();
  auto t_row_diff_base = row_data.getFTensor1DiffN<DIM>();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = AssemblyDomainEleOp::getMeasure() * t_w;
    ++t_w;

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / DIM; ++rr) {

      auto t_mat =
          get_mat_vector_dtensor_sym(rr, locMat, FTensor::Number<DIM>());

      FTensor::Tensor2<double, DIM, size_symm> t_tmp;
      t_tmp(i, L) = (t_DL(i, j, L)) * (alpha * t_row_diff_base(j));

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
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, typename AssemblyDomainEleOp>
OpCalculatePlasticFlowLhs_dUImpl<DIM, GAUSS, AssemblyDomainEleOp>::
    OpCalculatePlasticFlowLhs_dUImpl(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr,
        boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          AssemblyDomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  AssemblyDomainEleOp::sYmm = false;
}

template <int DIM, typename AssemblyDomainEleOp>
MoFEMErrorCode
OpCalculatePlasticFlowLhs_dUImpl<DIM, GAUSS, AssemblyDomainEleOp>::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;
  constexpr auto size_symm = (DIM * (DIM + 1)) / 2;
  FTensor::Index<'L', size_symm> L;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const size_t nb_integration_pts = AssemblyDomainEleOp::getGaussPts().size2();
  const size_t nb_row_base_functions = row_data.getN().size2();

  auto t_res_flow_dstrain =
      getFTensor4DdgFromMat<DIM, DIM>(commonDataPtr->resFlowDstrain);

  auto next = [&]() { ++t_res_flow_dstrain; };

  auto t_L = symm_L_tensor(FTensor::Number<DIM>());
  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
  FTensor::Tensor4<double, DIM, DIM, DIM, DIM> t_diff_grad;
  t_diff_grad(i, j, k, l) = t_kd(i, k) * t_kd(j, l);

  auto t_w = AssemblyDomainEleOp::getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

    double alpha = AssemblyDomainEleOp::getMeasure() * t_w;
    ++t_w;
    FTensor::Tensor3<double, size_symm, DIM, DIM> t_res_tens;
    t_res_tens(L, i, j) =
        alpha * ((t_L(m, n, L) * (t_res_flow_dstrain(m, n, k, l))) *
                 t_diff_grad(k, l, i, j));
    next();

    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows / size_symm; ++rr) {
      auto t_mat =
          get_mat_tensor_sym_dvector(rr, locMat, FTensor::Number<DIM>());
      auto t_col_diff_base = col_data.getFTensor1DiffN<DIM>(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / DIM; ++cc) {
        t_mat(L, l) += t_row_base * (t_res_tens(L, l, k) * t_col_diff_base(k));
        ++t_mat;
        ++t_col_diff_base;
      }
      ++t_row_base;
    }

    for (; rr < nb_row_base_functions; ++rr)
      ++t_row_base;
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, typename AssemblyDomainEleOp>
OpCalculateConstraintsLhs_dUImpl<DIM, GAUSS, AssemblyDomainEleOp>::
    OpCalculateConstraintsLhs_dUImpl(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr,
        boost::shared_ptr<MatrixDouble> m_D_ptr)
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), mDPtr(m_D_ptr) {
  AssemblyDomainEleOp::sYmm = false;
}

template <int DIM, typename AssemblyDomainEleOp>
MoFEMErrorCode
OpCalculateConstraintsLhs_dUImpl<DIM, GAUSS, AssemblyDomainEleOp>::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;

  auto &locMat = AssemblyDomainEleOp::locMat;

  const auto nb_integration_pts = AssemblyDomainEleOp::getGaussPts().size2();
  const auto nb_row_base_functions = row_data.getN().size2();

  auto t_c_dstrain =
      getFTensor2SymmetricFromMat<DIM>(commonDataPtr->resCdStrain);
  auto t_diff_grad_symmetrise = diff_symmetrize(FTensor::Number<DIM>());

  auto next = [&]() { ++t_c_dstrain; };

  auto get_mat_scalar_dvector = [&]() {
    if constexpr (DIM == 2)
      return FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2>{&locMat(0, 0),
                                                                &locMat(0, 1)};
    else
      return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>{
          &locMat(0, 0), &locMat(0, 1), &locMat(0, 2)};
  };

  auto t_w = AssemblyDomainEleOp::getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor0N();
  for (auto gg = 0; gg != nb_integration_pts; ++gg) {
    double alpha = AssemblyDomainEleOp::getMeasure() * t_w;
    ++t_w;

    FTensor::Tensor2<double, DIM, DIM> t_res_mat;
    t_res_mat(i, j) =
        ((t_c_dstrain(k, l)) * t_diff_grad_symmetrise(k, l, i, j));
    next();

    auto t_mat = get_mat_scalar_dvector();
    size_t rr = 0;
    for (; rr != AssemblyDomainEleOp::nbRows; ++rr) {
      const double row_base = alpha * t_row_base;
      auto t_col_diff_base = col_data.getFTensor1DiffN<DIM>(gg, 0);
      for (size_t cc = 0; cc != AssemblyDomainEleOp::nbCols / DIM; cc++) {
        t_mat(i) += row_base * (t_res_mat(i, j) * t_col_diff_base(j));
        ++t_mat;
        ++t_col_diff_base;
      }
      ++t_row_base;
    }
    for (; rr != nb_row_base_functions; ++rr)
      ++t_row_base;
  }

  MoFEMFunctionReturn(0);
}

}; // namespace PlasticOps