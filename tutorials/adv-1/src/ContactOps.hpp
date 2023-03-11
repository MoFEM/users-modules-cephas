

/**
 * \file ContactOps.hpp
 * \example ContactOps.hpp
 */

namespace ContactOps {


//! [Common data]
struct CommonData {
  boost::shared_ptr<MatrixDouble> mDPtr;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;

  boost::shared_ptr<MatrixDouble> contactStressPtr;
  boost::shared_ptr<MatrixDouble> contactStressDivergencePtr;
  boost::shared_ptr<MatrixDouble> contactTractionPtr;
  boost::shared_ptr<MatrixDouble> contactDispPtr;

  boost::shared_ptr<MatrixDouble> curlContactStressPtr;
};
//! [Common data]

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;

struct OpConstrainBoundaryRhs : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryRhs(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainBoundaryLhs_dU : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryLhs_dU(const std::string row_field_name,
                            const std::string col_field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainBoundaryLhs_dTraction : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryLhs_dTraction(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <typename T>
inline auto surface_distance_function(FTensor::Tensor1<T, 3> &t_coords) {
  return t_coords(1) + 0.5;
};

constexpr double cx_eps = 1e-12;

template <typename T>
inline FTensor::Tensor1<double, SPACE_DIM>
grad_surface_distance_function(FTensor::Tensor1<T, 3> &t_coords) {
  FTensor::Tensor1<double, SPACE_DIM> t_grad;
  FTensor::Tensor1<std::complex<double>, 3> t_cx_coords;
  FTensor::Index<'I', 3> I;
  for (auto d = 0; d != SPACE_DIM; ++d) {
    t_cx_coords(I) = t_coords(I);
    t_cx_coords(d) += 1i * cx_eps;
    t_grad(d) = std::imag(surface_distance_function(t_cx_coords)) / cx_eps;
  };
  return t_grad;
};

template <typename T>
inline double normal_traction(FTensor::Tensor1<T, SPACE_DIM> &t_traction,
                              FTensor::Tensor1<double, SPACE_DIM> &t_normal) {
  return -t_traction(i) * t_normal(i);
}

inline double sign(double x) {
  if (x == 0)
    return 0;
  else if (x > 0)
    return 1;
  else
    return -1;
};

inline double w(const double sdf, const double t) { return sdf - cn * t; }

inline double constrain(double sdf, double t) {
  return (w(sdf, t) + std::abs(w(sdf, t))) / 2;
};

inline double diff_constrains_dtraction(double sdf, double t) {
  return -cn * (1 + sign(w(sdf, t))) / 2;
}

inline double diff_constrains_dsdf(double sdf, double t) {
  return (1 + sign(w(sdf, t))) / 2;
}

OpConstrainBoundaryRhs::OpConstrainBoundaryRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyBoundaryEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpConstrainBoundaryRhs::iNtegrate(EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = AssemblyBoundaryEleOp::getGaussPts().size2();

  auto &nf = AssemblyBoundaryEleOp::locF;

  auto t_normal = getFTensor1Normal();
  t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

  auto t_w = getFTensor0IntegrationWeight();
  auto t_disp = getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));
  auto t_coords = getFTensor1CoordsAtGaussPts();

  size_t nb_base_functions = data.getN().size2() / 3;
  auto t_base = data.getFTensor1N<3>();
  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    auto t_nf = getFTensor1FromPtr<SPACE_DIM>(&nf[0]);
    const double alpha = t_w * getMeasure();

    FTensor::Tensor1<double, 3> t_spatial_coords;
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto sdf0 = surface_distance_function(t_coords);
    auto sdf = surface_distance_function(t_spatial_coords);
    auto t_std_grad = grad_surface_distance_function(t_coords);

    auto t = normal_traction(t_traction, t_std_grad);
    auto c = constrain(sdf, t);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
    t_P(i, j) = t_std_grad(i) * t_std_grad(j);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
    t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

    FTensor::Tensor1<double, SPACE_DIM> t_rhs_constrains;
    t_rhs_constrains(i) = t_std_grad(i) * (c - sdf0);

    FTensor::Tensor1<double, SPACE_DIM> t_rhs_tangent_disp,
        t_rhs_tangent_traction;
    t_rhs_tangent_disp(i) = t_Q(i, j) * t_disp(j);
    t_rhs_tangent_traction(i) = cn * t_Q(i, j) * t_traction(j);

    size_t bb = 0;
    for (; bb != AssemblyBoundaryEleOp::nbRows / SPACE_DIM; ++bb) {
      const double beta = alpha * (t_base(i) * t_normal(i));

      t_nf(i) -= beta * t_rhs_constrains(i);
      t_nf(i) -= beta * t_rhs_tangent_disp(i);
      t_nf(i) += beta * t_rhs_tangent_traction(i);

      ++t_nf;
      ++t_base;
    }
    for (; bb < nb_base_functions; ++bb)
      ++t_base;

    ++t_disp;
    ++t_traction;
    ++t_coords;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

OpConstrainBoundaryLhs_dU::OpConstrainBoundaryLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyBoundaryEleOp(row_field_name, col_field_name,
                            DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpConstrainBoundaryLhs_dU::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  auto &locMat = AssemblyBoundaryEleOp::locMat;

  auto t_normal = getFTensor1Normal();
  t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

  auto t_disp = getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));
  auto t_coords = getFTensor1CoordsAtGaussPts();

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor1N<3>();
  size_t nb_face_functions = row_data.getN().size2() / 3;

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    const double alpha = t_w * getMeasure();

    FTensor::Tensor1<double, 3> t_spatial_coords{0., 0., 0.};
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto sdf = surface_distance_function(t_spatial_coords);
    auto t_grad_sdf = grad_surface_distance_function(t_coords);
    auto t = normal_traction(t_traction, t_grad_sdf);
    auto diff_constrain = diff_constrains_dsdf(sdf, t);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
    t_P(i, j) = t_grad_sdf(i) * t_grad_sdf(j);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
    t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

    size_t rr = 0;
    for (; rr != AssemblyBoundaryEleOp::nbRows / SPACE_DIM; ++rr) {

      auto t_mat = getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(
          locMat, SPACE_DIM * rr);

      const double row_base = t_row_base(i) * t_normal(i);

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyBoundaryEleOp::nbCols / SPACE_DIM;
           ++cc) {
        const double beta = alpha * row_base * t_col_base;

        t_mat(i, j) -= (beta * diff_constrain) * t_P(i, j);
        t_mat(i, j) -= beta * t_Q(i, j);

        ++t_col_base;
        ++t_mat;
      }

      ++t_row_base;
    }
    for (; rr < nb_face_functions; ++rr)
      ++t_row_base;

    ++t_disp;
    ++t_traction;
    ++t_coords;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

OpConstrainBoundaryLhs_dTraction::OpConstrainBoundaryLhs_dTraction(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyBoundaryEleOp(row_field_name, col_field_name,
                            DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpConstrainBoundaryLhs_dTraction::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  auto &locMat = AssemblyBoundaryEleOp::locMat;

  auto t_normal = getFTensor1Normal();
  t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

  auto t_disp = getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));
  auto t_coords = getFTensor1CoordsAtGaussPts();

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor1N<3>();
  size_t nb_face_functions = row_data.getN().size2() / 3;

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    const double alpha = t_w * getMeasure();

    FTensor::Tensor1<double, 3> t_spatial_coords{0., 0., 0.};
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto sdf = surface_distance_function(t_spatial_coords);
    auto t_grad_sdf = grad_surface_distance_function(t_coords);
    auto t = normal_traction(t_traction, t_grad_sdf);
    const double dc_dt = diff_constrains_dtraction(sdf, t);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
    t_P(i, j) = t_grad_sdf(i) * t_grad_sdf(j);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
    t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

    size_t rr = 0;
    for (; rr != AssemblyBoundaryEleOp::nbRows / SPACE_DIM; ++rr) {

      auto t_mat = getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(
          locMat, SPACE_DIM * rr);
      const double row_base = t_row_base(i) * t_normal(i);

      auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
      for (size_t cc = 0; cc != AssemblyBoundaryEleOp::nbCols / SPACE_DIM;
           ++cc) {
        const double col_base = t_col_base(i) * t_normal(i);
        const double beta = alpha * row_base * col_base;

        t_mat(i, j) += (beta * dc_dt) * t_P(i, j);
        t_mat(i, j) += beta * cn * t_Q(i, j);

        ++t_col_base;
        ++t_mat;
      }

      ++t_row_base;
    }
    for (; rr < nb_face_functions; ++rr)
      ++t_row_base;

    ++t_disp;
    ++t_traction;
    ++t_coords;
    ++t_w;
  }

  MoFEMFunctionReturn(0);
}

}; // namespace ContactOps
