

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

//! [Surface distance function from python]
#ifdef PYTHON_SFD
struct SDFPython {
  SDFPython() = default;
  virtual ~SDFPython() = default;

  MoFEMErrorCode sdfInit(const std::string py_file) {
    MoFEMFunctionBegin;
    try {

      // create main module
      auto main_module = bp::import("__main__");
      mainNamespace = main_module.attr("__dict__");
      bp::exec_file(py_file.c_str(), mainNamespace, mainNamespace);
      // create a reference to python function
      sdfFun = mainNamespace["sdf"];
      sdfGradFun = mainNamespace["grad_sdf"];
      sdfHessFun = mainNamespace["hess_sdf"];

    } catch (bp::error_already_set const &) {
      // print all other errors to stderr
      PyErr_Print();
      CHK_THROW_MESSAGE(MOFEM_OPERATION_UNSUCCESSFUL, "Python error");
    }
    MoFEMFunctionReturn(0);
  };

  template <typename T>
  inline std::vector<T>
  py_list_to_std_vector(const boost::python::object &iterable) {
    return std::vector<T>(boost::python::stl_input_iterator<T>(iterable),
                          boost::python::stl_input_iterator<T>());
  }

  MoFEMErrorCode evalSdf(

      double t, double x, double y, double z, double &sdf

  ) {
    MoFEMFunctionBegin;
    try {

      // call python function
      sdf = bp::extract<double>(sdfFun(t, x, y, z));

    } catch (bp::error_already_set const &) {
      // print all other errors to stderr
      PyErr_Print();
      CHK_THROW_MESSAGE(MOFEM_OPERATION_UNSUCCESSFUL, "Python error");
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode evalGradSdf(

      double t, double x, double y, double z, std::vector<double> &grad_sdf

  ) {
    MoFEMFunctionBegin;
    try {

      // call python function
      grad_sdf = py_list_to_std_vector<double>(sdfGradFun(t, x, y, z));

    } catch (bp::error_already_set const &) {
      // print all other errors to stderr
      PyErr_Print();
      CHK_THROW_MESSAGE(MOFEM_OPERATION_UNSUCCESSFUL, "Python error");
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode evalHessSdf(

      double t, double x, double y, double z, std::vector<double> &hess_sdf

  ) {
    MoFEMFunctionBegin;
    try {

      // call python function
      hess_sdf = py_list_to_std_vector<double>(sdfHessFun(t, x, y, z));

    } catch (bp::error_already_set const &) {
      // print all other errors to stderr
      PyErr_Print();
      CHK_THROW_MESSAGE(MOFEM_OPERATION_UNSUCCESSFUL, "Python error");
    }
    MoFEMFunctionReturn(0);
  }

private:
  bp::object mainNamespace;
  bp::object sdfFun;
  bp::object sdfGradFun;
  bp::object sdfHessFun;
};

static boost::weak_ptr<SDFPython> sdfPythonWeakPtr;

#endif
//! [Surface distance function from python]

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
inline double surface_distance_function(double t,
                                        FTensor::Tensor1<T, 3> &t_coords) {
#ifdef PYTHON_SFD
  if (auto sdf_ptr = sdfPythonWeakPtr.lock()) {
    double sdf;
    CHK_MOAB_THROW(
        sdf_ptr->evalSdf(t, t_coords(0), t_coords(1), t_coords(2), sdf),
        "Failed python call");
    return sdf;
  }
#endif
  return t_coords(1) + 0.5;
};

constexpr double cx_eps = 1e-12;

template <typename T>
inline FTensor::Tensor1<double, 3>
grad_surface_distance_function(double t, FTensor::Tensor1<T, 3> &t_coords) {
#ifdef PYTHON_SFD
  if (auto sdf_ptr = sdfPythonWeakPtr.lock()) {
    std::vector<double> grad_sdf;
    CHK_MOAB_THROW(sdf_ptr->evalGradSdf(t, t_coords(0), t_coords(1),
                                        t_coords(2), grad_sdf),
                   "Failed python call");
    if (grad_sdf.size() != 3) {
      CHK_THROW_MESSAGE(MOFEM_DATA_INCONSISTENCY, "Expected size 6");
    }
    return FTensor::Tensor1<double, 3>{grad_sdf[0], grad_sdf[1], grad_sdf[2]};
  }
#endif
  return FTensor::Tensor1<double, 3>{0., 1., 0.};
};

template <typename T>
inline FTensor::Tensor2_symmetric<double, 3>
hess_surface_distance_function(double t, FTensor::Tensor1<T, 3> &t_coords) {
#ifdef PYTHON_SFD
  if (auto sdf_ptr = sdfPythonWeakPtr.lock()) {
    std::vector<double> hess_sdf;
    CHK_MOAB_THROW(sdf_ptr->evalHessSdf(t, t_coords(0), t_coords(1),
                                        t_coords(2), hess_sdf),
                   "Failed python call");
    if (hess_sdf.size() != 6) {
      CHK_THROW_MESSAGE(MOFEM_DATA_INCONSISTENCY, "Expected size 6");
    }
    return FTensor::Tensor2_symmetric<double, 3>{hess_sdf[0], hess_sdf[1],
                                                 hess_sdf[2], hess_sdf[3],
                                                 hess_sdf[4], hess_sdf[5]};
  }
#endif
  return FTensor::Tensor2_symmetric<double, 3>{0., 0., 0., 0., 0., 0.};
};

inline double sign(double x) {
  if (x == 0)
    return 0;
  else if (x > 0)
    return 1;
  else
    return -1;
};

inline double w(const double sdf, const double t) { return sdf - cn * t; }

inline double constrain(double sdf, double un, double tn) {
  const auto s = sign(w(sdf, tn));
  return (s + 1) * (un - cn * tn) / 2;
}

inline double diff_constrains_traction(double sdf, double tn) {
  const auto s = sign(w(sdf, tn));
  return -(s + 1) * cn / 2;
}

inline double diff_constrains_dun(double sdf, double tn) {
  const auto s = sign(w(sdf, tn));
  return (s + 1) / 2;
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

    auto sdf = surface_distance_function(getTStime(), t_spatial_coords);
    auto t_grad_sdf =
        grad_surface_distance_function(getTStime(), t_spatial_coords);

    auto un = t_disp(i) * t_grad_sdf(i);
    auto tn = -t_traction(i) * t_grad_sdf(i);
    auto c = constrain(sdf, un, tn);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
    t_P(i, j) = t_grad_sdf(i) * t_grad_sdf(j);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
    t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

    FTensor::Tensor1<double, SPACE_DIM> t_rhs_constrains;
    t_rhs_constrains(i) = c * t_grad_sdf(i);

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

MoFEMErrorCode
OpConstrainBoundaryLhs_dU::iNtegrate(EntitiesFieldData::EntData &row_data,
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

    auto sdf = surface_distance_function(getTStime(), t_spatial_coords);
    auto t_grad_sdf =
        grad_surface_distance_function(getTStime(), t_spatial_coords);
    auto t_hess_sdf =
        hess_surface_distance_function(getTStime(), t_spatial_coords);

    auto un = t_disp(i) * t_grad_sdf(i);
    auto tn = -t_traction(i) * t_grad_sdf(i);
    auto c = constrain(sdf, un, tn);
    auto diff_c = diff_constrains_dun(sdf, tn);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
    t_P(i, j) = t_grad_sdf(i) * t_grad_sdf(j);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
    t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_rhs_constrains_dU;
    t_rhs_constrains_dU(i, j) = diff_c * t_P(i, j);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_rhs_constrains_hessian_dU;
    t_rhs_constrains_hessian_dU(i, j) = t_hess_sdf(i, j) * c;
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM>
        t_rhs_tangent_disp_hessian_dU, t_rhs_tangent_traction_hessian_dU;
    t_rhs_tangent_disp_hessian_dU(i, j) = -(

        t_hess_sdf(i, j) * (t_grad_sdf(k) * t_disp(k)) +

        t_grad_sdf(i) * (t_hess_sdf(k, j) * t_disp(k))

    );
    t_rhs_tangent_traction_hessian_dU(i, j) =
        -cn * (

                  t_hess_sdf(i, j) * (t_grad_sdf(k) * t_traction(k)) +

                  t_grad_sdf(i) * (t_hess_sdf(k, j) * t_traction(k))

              );

    size_t rr = 0;
    for (; rr != AssemblyBoundaryEleOp::nbRows / SPACE_DIM; ++rr) {

      auto t_mat = getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(
          locMat, SPACE_DIM * rr);

      const double row_base = t_row_base(i) * t_normal(i);

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyBoundaryEleOp::nbCols / SPACE_DIM;
           ++cc) {
        const double beta = alpha * row_base * t_col_base;

        t_mat(i, j) -= beta * t_rhs_constrains_dU(i, j);
        t_mat(i, j) -= beta * t_Q(i, j);
        t_mat(i, j) -= beta * t_rhs_constrains_hessian_dU(i, j);
        t_mat(i, j) -= beta * t_rhs_tangent_disp_hessian_dU(i, j);
        t_mat(i, j) += beta * t_rhs_tangent_traction_hessian_dU(i, j);

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

    auto sdf = surface_distance_function(getTStime(), t_spatial_coords);
    auto t_grad_sdf =
        grad_surface_distance_function(getTStime(), t_spatial_coords);

    auto tn = -t_traction(i) * t_grad_sdf(i);
    const double dc_dt = -diff_constrains_traction(sdf, tn);

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

        t_mat(i, j) -= (beta * dc_dt) * t_P(i, j);
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
