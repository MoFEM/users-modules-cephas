

/**
 * \file ContactOps.hpp
 * \example ContactOps.hpp
 */

#ifndef __CONTACTOPS_HPP__
#define __CONTACTOPS_HPP__

namespace ContactOps {

//! [Common data]
struct CommonData : public boost::enable_shared_from_this<CommonData> {
  // MatrixDouble contactStress;
  MatrixDouble contactTraction;
  MatrixDouble contactDisp;

  static SmartPetscObj<Vec>
      totalTraction; // User have to release and create vector when appropiate.

  static auto createTotalTraction(MoFEM::Interface &m_field) {
    constexpr int ghosts[] = {0, 1, 2};
    totalTraction =
        createGhostVector(m_field.get_comm(),

                          (m_field.get_comm_rank() == 0) ? 3 : 0, 3,

                          (m_field.get_comm_rank() == 0) ? 0 : 3, ghosts);
    return totalTraction;
  }

  static auto getFTensor1TotalTraction() {
    if (CommonData::totalTraction) {
      const double *t_ptr;
      CHK_THROW_MESSAGE(VecGetArrayRead(CommonData::totalTraction, &t_ptr),
                        "get array");
      FTensor::Tensor1<double, 3> t{t_ptr[0], t_ptr[1], t_ptr[2]};
      CHK_THROW_MESSAGE(VecRestoreArrayRead(CommonData::totalTraction, &t_ptr),
                        "restore array");
      return t;
    } else {
      return FTensor::Tensor1<double, 3>{0., 0., 0.};
    }
  }

  inline auto contactTractionPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(),
                                           &contactTraction);
  }

  inline auto contactDispPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &contactDisp);
  }
};

SmartPetscObj<Vec> CommonData::totalTraction;

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

      double t, double x, double y, double z, double tx, double ty, double tz,
      double &sdf

  ) {
    MoFEMFunctionBegin;
    try {

      // call python function
      sdf = bp::extract<double>(sdfFun(t, x, y, z, tx, ty, tz));

    } catch (bp::error_already_set const &) {
      // print all other errors to stderr
      PyErr_Print();
      CHK_THROW_MESSAGE(MOFEM_OPERATION_UNSUCCESSFUL, "Python error");
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode evalGradSdf(

      double t, double x, double y, double z, double tx, double ty, double tz,
      std::vector<double> &grad_sdf

  ) {
    MoFEMFunctionBegin;
    try {

      // call python function
      grad_sdf =
          py_list_to_std_vector<double>(sdfGradFun(t, x, y, z, tx, ty, tz));

    } catch (bp::error_already_set const &) {
      // print all other errors to stderr
      PyErr_Print();
      CHK_THROW_MESSAGE(MOFEM_OPERATION_UNSUCCESSFUL, "Python error");
    }

    if (grad_sdf.size() != 3) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Expected size 3");
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode evalHessSdf(

      double t, double x, double y, double z, double tx, double ty, double tz,
      std::vector<double> &hess_sdf

  ) {
    MoFEMFunctionBegin;
    try {

      // call python function
      hess_sdf =
          py_list_to_std_vector<double>(sdfHessFun(t, x, y, z, tx, ty, tz));

    } catch (bp::error_already_set const &) {
      // print all other errors to stderr
      PyErr_Print();
      CHK_THROW_MESSAGE(MOFEM_OPERATION_UNSUCCESSFUL, "Python error");
    }

    if (hess_sdf.size() != 6) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Expected size 6");
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

using SurfaceDistanceFunction = boost::function<double(
    double t, double x, double y, double z, double tx, double ty, double tz)>;

using GradSurfaceDistanceFunction = boost::function<FTensor::Tensor1<double, 3>(
    double t, double x, double y, double z, double tx, double ty, double tz)>;

using HessSurfaceDistanceFunction =
    boost::function<FTensor::Tensor2_symmetric<double, 3>(
        double t, double x, double y, double z, double tx, double ty,
        double tz)>;

inline double surface_distance_function(double t, double x, double y, double z,
                                        double tx, double ty, double tz) {

#ifdef PYTHON_SFD
  if (auto sdf_ptr = sdfPythonWeakPtr.lock()) {
    double sdf;
    CHK_MOAB_THROW(sdf_ptr->evalSdf(t, x, y, z, tx, ty, tz, sdf),
                   "Failed python call");
    return sdf;
  }
#endif
  return y + 0.5;
}

inline FTensor::Tensor1<double, 3>
grad_surface_distance_function(double t, double x, double y, double z,
                               double tx, double ty, double tz) {
#ifdef PYTHON_SFD
  if (auto sdf_ptr = sdfPythonWeakPtr.lock()) {
    std::vector<double> grad_sdf;
    CHK_MOAB_THROW(sdf_ptr->evalGradSdf(t, x, y, z, tx, ty, tz, grad_sdf),
                   "Failed python call");
    return FTensor::Tensor1<double, 3>{grad_sdf[0], grad_sdf[1], grad_sdf[2]};
  }
#endif
  return FTensor::Tensor1<double, 3>{0., 1., 0.};
}

inline FTensor::Tensor2_symmetric<double, 3>
hess_surface_distance_function(double t, double x, double y, double z,
                               double tx, double ty, double tz) {
#ifdef PYTHON_SFD
  if (auto sdf_ptr = sdfPythonWeakPtr.lock()) {
    std::vector<double> hess_sdf;
    CHK_MOAB_THROW(sdf_ptr->evalHessSdf(t, x, y, z, tx, ty, tz, hess_sdf),
                   "Failed python call");
    return FTensor::Tensor2_symmetric<double, 3>{hess_sdf[0], hess_sdf[1],
                                                 hess_sdf[2], hess_sdf[3],
                                                 hess_sdf[4], hess_sdf[5]};
  }
#endif
  return FTensor::Tensor2_symmetric<double, 3>{0., 0., 0., 0., 0., 0.};
}

template <int DIM, IntegrationType I, typename BoundaryEleOp>
struct OpAssembleTotalContactTractionImpl;

template <int DIM, IntegrationType I, typename AssemblyBoundaryEleOp>
struct OpConstrainBoundaryRhsImpl;

template <int DIM, IntegrationType I, typename AssemblyBoundaryEleOp>
struct OpConstrainBoundaryLhs_dUImpl;

template <int DIM, IntegrationType I, typename AssemblyBoundaryEleOp>
struct OpConstrainBoundaryLhs_dTractionImpl;

template <int DIM, typename BoundaryEleOp>
struct OpAssembleTotalContactTractionImpl<DIM, GAUSS, BoundaryEleOp>
    : public BoundaryEleOp {
  OpAssembleTotalContactTractionImpl(
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM, typename AssemblyBoundaryEleOp>
struct OpConstrainBoundaryRhsImpl<DIM, GAUSS, AssemblyBoundaryEleOp>
    : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryRhsImpl(const std::string field_name,
                             boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data);

  SurfaceDistanceFunction surfaceDistanceFunction = surface_distance_function;
  GradSurfaceDistanceFunction gradSurfaceDistanceFunction =
      grad_surface_distance_function;

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM, typename AssemblyBoundaryEleOp>
struct OpConstrainBoundaryLhs_dUImpl<DIM, GAUSS, AssemblyBoundaryEleOp>
    : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryLhs_dUImpl(const std::string row_field_name,
                                const std::string col_field_name,
                                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

  SurfaceDistanceFunction surfaceDistanceFunction = surface_distance_function;
  GradSurfaceDistanceFunction gradSurfaceDistanceFunction =
      grad_surface_distance_function;
  HessSurfaceDistanceFunction hessSurfaceDistanceFunction =
      hess_surface_distance_function;

  boost::shared_ptr<CommonData> commonDataPtr;
};

template <int DIM, typename AssemblyBoundaryEleOp>
struct OpConstrainBoundaryLhs_dTractionImpl<DIM, GAUSS, AssemblyBoundaryEleOp>
    : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryLhs_dTractionImpl(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data);

  SurfaceDistanceFunction surfaceDistanceFunction = surface_distance_function;
  GradSurfaceDistanceFunction gradSurfaceDistanceFunction =
      grad_surface_distance_function;

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <typename BoundaryEleOp> struct ContactIntegrators {

  template <int DIM, IntegrationType I>
  using OpAssembleTotalContactTraction =
      OpAssembleTotalContactTractionImpl<DIM, I, BoundaryEleOp>;

  template <AssemblyType A> struct Assembly {

    using AssemblyBoundaryEleOp =
        typename FormsIntegrators<BoundaryEleOp>::template Assembly<A>::OpBase;

    template <int DIM, IntegrationType I>
    using OpConstrainBoundaryRhs =
        OpConstrainBoundaryRhsImpl<DIM, I, AssemblyBoundaryEleOp>;

    template <int DIM, IntegrationType I>
    using OpConstrainBoundaryLhs_dU =
        OpConstrainBoundaryLhs_dUImpl<DIM, I, AssemblyBoundaryEleOp>;

    template <int DIM, IntegrationType I>
    using OpConstrainBoundaryLhs_dTraction =
        OpConstrainBoundaryLhs_dTractionImpl<DIM, I, AssemblyBoundaryEleOp>;
  };
};

inline double sign(double x) {
  if (x == 0)
    return 0;
  else if (x > 0)
    return 1;
  else
    return -1;
};

inline double w(const double sdf, const double tn) {
  return sdf - cn_contact * tn;
}

/**
 * @brief constrain function
 *
 * return 1 if negative sdn or positive tn
 *
 * @param sdf signed distance
 * @param tn traction
 * @return double
 */
inline double constrain(double sdf, double tn) {
  const auto s = sign(w(sdf, tn));
  return (1 - s) / 2;
}

template <int DIM, typename BoundaryEleOp>
OpAssembleTotalContactTractionImpl<DIM, GAUSS, BoundaryEleOp>::
    OpAssembleTotalContactTractionImpl(
        boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE),
      commonDataPtr(common_data_ptr) {}

template <int DIM, typename BoundaryEleOp>
MoFEMErrorCode
OpAssembleTotalContactTractionImpl<DIM, GAUSS, BoundaryEleOp>::doWork(
    int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Tensor1<double, 3> t_sum_t{0., 0., 0.};

  auto t_w = BoundaryEleOp::getFTensor0IntegrationWeight();
  auto t_traction = getFTensor1FromMat<DIM>(commonDataPtr->contactTraction);

  const auto nb_gauss_pts = BoundaryEleOp::getGaussPts().size2();
  for (auto gg = 0; gg != nb_gauss_pts; ++gg) {
    const double alpha = t_w * BoundaryEleOp::getMeasure();
    t_sum_t(i) += alpha * t_traction(i);
    ++t_w;
    ++t_traction;
  }

  constexpr int ind[] = {0, 1, 2};
  CHKERR VecSetValues(commonDataPtr->totalTraction, 3, ind, &t_sum_t(0),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
}

template <int DIM, typename AssemblyBoundaryEleOp>
OpConstrainBoundaryRhsImpl<DIM, GAUSS, AssemblyBoundaryEleOp>::
    OpConstrainBoundaryRhsImpl(const std::string field_name,
                               boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyBoundaryEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

template <int DIM, typename AssemblyBoundaryEleOp>
MoFEMErrorCode
OpConstrainBoundaryRhsImpl<DIM, GAUSS, AssemblyBoundaryEleOp>::iNtegrate(
    EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;
  FTensor::Index<'l', DIM> l;

  const size_t nb_gauss_pts = AssemblyBoundaryEleOp::getGaussPts().size2();

  auto &nf = AssemblyBoundaryEleOp::locF;

  auto t_normal_at_pts = AssemblyBoundaryEleOp::getFTensor1NormalsAtGaussPts();
  auto t_total_traction = CommonData::getFTensor1TotalTraction();

  auto t_w = AssemblyBoundaryEleOp::getFTensor0IntegrationWeight();
  auto t_disp = getFTensor1FromMat<DIM>(commonDataPtr->contactDisp);
  auto t_traction = getFTensor1FromMat<DIM>(commonDataPtr->contactTraction);
  auto t_coords = AssemblyBoundaryEleOp::getFTensor1CoordsAtGaussPts();

  size_t nb_base_functions = data.getN().size2() / 3;
  auto t_base = data.getFTensor1N<3>();
  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor1<double, DIM> t_normal;
    t_normal(i) =
        t_normal_at_pts(i) / std::sqrt(t_normal_at_pts(i) * t_normal_at_pts(i));

    auto t_nf = getFTensor1FromPtr<DIM>(&nf[0]);
    const double alpha = t_w * AssemblyBoundaryEleOp::getMeasure();

    FTensor::Tensor1<double, 3> t_spatial_coords{0., 0., 0.};
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto ts_a = AssemblyBoundaryEleOp::getTStime();

    auto sdf = surfaceDistanceFunction(
        ts_a, t_spatial_coords(0), t_spatial_coords(1), t_spatial_coords(2),
        t_total_traction(0), t_total_traction(1), t_total_traction(2));

    auto t_grad_sdf = gradSurfaceDistanceFunction(
        ts_a, t_spatial_coords(0), t_spatial_coords(1), t_spatial_coords(2),
        t_total_traction(0), t_total_traction(1), t_total_traction(2));

    auto tn = -t_traction(i) * t_grad_sdf(i);
    auto c = constrain(sdf, tn);

    FTensor::Tensor2<double, DIM, DIM> t_cP;
    t_cP(i, j) = (c * t_grad_sdf(i)) * t_grad_sdf(j);
    FTensor::Tensor2<double, DIM, DIM> t_cQ;
    t_cQ(i, j) = kronecker_delta(i, j) - t_cP(i, j);

    FTensor::Tensor1<double, DIM> t_rhs;
    t_rhs(i) =

        t_cQ(i, j) * (t_disp(j) - cn_contact * t_traction(j))

        +

        t_cP(i, j) * t_disp(j) +
        c * (sdf * t_grad_sdf(i)); // add gap0 displacements

    size_t bb = 0;
    for (; bb != AssemblyBoundaryEleOp::nbRows / DIM; ++bb) {
      const double beta = alpha * (t_base(i) * t_normal(i));
      t_nf(i) -= beta * t_rhs(i);

      ++t_nf;
      ++t_base;
    }
    for (; bb < nb_base_functions; ++bb)
      ++t_base;

    ++t_disp;
    ++t_traction;
    ++t_coords;
    ++t_w;
    ++t_normal_at_pts;
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, typename AssemblyBoundaryEleOp>
OpConstrainBoundaryLhs_dUImpl<DIM, GAUSS, AssemblyBoundaryEleOp>::
    OpConstrainBoundaryLhs_dUImpl(const std::string row_field_name,
                                  const std::string col_field_name,
                                  boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyBoundaryEleOp(row_field_name, col_field_name,
                            DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  AssemblyBoundaryEleOp::sYmm = false;
}

template <int DIM, typename AssemblyBoundaryEleOp>
MoFEMErrorCode
OpConstrainBoundaryLhs_dUImpl<DIM, GAUSS, AssemblyBoundaryEleOp>::iNtegrate(
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;

  const size_t nb_gauss_pts = AssemblyBoundaryEleOp::getGaussPts().size2();
  auto &locMat = AssemblyBoundaryEleOp::locMat;

  auto t_normal_at_pts = AssemblyBoundaryEleOp::getFTensor1NormalsAtGaussPts();
  auto t_total_traction = CommonData::getFTensor1TotalTraction();

  auto t_disp = getFTensor1FromMat<DIM>(commonDataPtr->contactDisp);
  auto t_traction = getFTensor1FromMat<DIM>(commonDataPtr->contactTraction);
  auto t_coords = AssemblyBoundaryEleOp::getFTensor1CoordsAtGaussPts();

  auto t_w = AssemblyBoundaryEleOp::getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor1N<3>();
  size_t nb_face_functions = row_data.getN().size2() / 3;

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor1<double, DIM> t_normal;
    t_normal(i) =
        t_normal_at_pts(i) / std::sqrt(t_normal_at_pts(i) * t_normal_at_pts(i));

    const double alpha = t_w * AssemblyBoundaryEleOp::getMeasure();

    FTensor::Tensor1<double, 3> t_spatial_coords{0., 0., 0.};
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto ts_time = AssemblyBoundaryEleOp::getTStime();

    auto sdf = surfaceDistanceFunction(
        ts_time, t_spatial_coords(0), t_spatial_coords(1), t_spatial_coords(2),
        t_total_traction(0), t_total_traction(1), t_total_traction(2));
    auto t_grad_sdf = gradSurfaceDistanceFunction(
        ts_time, t_spatial_coords(0), t_spatial_coords(1), t_spatial_coords(2),
        t_total_traction(0), t_total_traction(1), t_total_traction(2));

    auto tn = -t_traction(i) * t_grad_sdf(i);
    auto c = constrain(sdf, tn);

    FTensor::Tensor2<double, DIM, DIM> t_cP;
    t_cP(i, j) = (c * t_grad_sdf(i)) * t_grad_sdf(j);
    FTensor::Tensor2<double, DIM, DIM> t_cQ;
    t_cQ(i, j) = kronecker_delta(i, j) - t_cP(i, j);

    FTensor::Tensor2<double, 3, 3> t_res_dU;
    t_res_dU(i, j) = kronecker_delta(i, j) + t_cP(i, j);

    if (c > 0) {
      auto t_hess_sdf = hessSurfaceDistanceFunction(
          ts_time, t_spatial_coords(0), t_spatial_coords(1),
          t_spatial_coords(2), t_total_traction(0), t_total_traction(1),
          t_total_traction(2));
      t_res_dU(i, j) +=
          (c * cn_contact) *
              (t_hess_sdf(i, j) * (t_grad_sdf(k) * t_traction(k)) +
               t_grad_sdf(i) * t_hess_sdf(k, j) * t_traction(k)) +
          c * sdf * t_hess_sdf(i, j);
    }

    size_t rr = 0;
    for (; rr != AssemblyBoundaryEleOp::nbRows / DIM; ++rr) {

      auto t_mat = getFTensor2FromArray<DIM, DIM, DIM>(locMat, DIM * rr);

      const double row_base = t_row_base(i) * t_normal(i);

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyBoundaryEleOp::nbCols / DIM; ++cc) {
        const double beta = alpha * row_base * t_col_base;

        t_mat(i, j) -= beta * t_res_dU(i, j);

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
    ++t_normal_at_pts;
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, typename AssemblyBoundaryEleOp>
OpConstrainBoundaryLhs_dTractionImpl<DIM, GAUSS, AssemblyBoundaryEleOp>::
    OpConstrainBoundaryLhs_dTractionImpl(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyBoundaryEleOp(row_field_name, col_field_name,
                            DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  AssemblyBoundaryEleOp::sYmm = false;
}

template <int DIM, typename AssemblyBoundaryEleOp>
MoFEMErrorCode
OpConstrainBoundaryLhs_dTractionImpl<DIM, GAUSS, AssemblyBoundaryEleOp>::
    iNtegrate(EntitiesFieldData::EntData &row_data,
              EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', DIM> i;
  FTensor::Index<'j', DIM> j;
  FTensor::Index<'k', DIM> k;

  const size_t nb_gauss_pts = AssemblyBoundaryEleOp::getGaussPts().size2();
  auto &locMat = AssemblyBoundaryEleOp::locMat;

  auto t_normal_at_pts = AssemblyBoundaryEleOp::getFTensor1NormalsAtGaussPts();
  auto t_total_traction = CommonData::getFTensor1TotalTraction();

  auto t_disp = getFTensor1FromMat<DIM>(commonDataPtr->contactDisp);
  auto t_traction = getFTensor1FromMat<DIM>(commonDataPtr->contactTraction);
  auto t_coords = AssemblyBoundaryEleOp::getFTensor1CoordsAtGaussPts();

  auto t_w = AssemblyBoundaryEleOp::getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor1N<3>();
  size_t nb_face_functions = row_data.getN().size2() / 3;

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor1<double, DIM> t_normal;
    t_normal(i) =
        t_normal_at_pts(i) / std::sqrt(t_normal_at_pts(i) * t_normal_at_pts(i));

    const double alpha = t_w * AssemblyBoundaryEleOp::getMeasure();

    FTensor::Tensor1<double, 3> t_spatial_coords{0., 0., 0.};
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto ts_time = AssemblyBoundaryEleOp::getTStime();

    auto sdf = surfaceDistanceFunction(
        ts_time, t_spatial_coords(0), t_spatial_coords(1), t_spatial_coords(2),
        t_total_traction(0), t_total_traction(1), t_total_traction(2));
    auto t_grad_sdf = gradSurfaceDistanceFunction(
        ts_time, t_spatial_coords(0), t_spatial_coords(1), t_spatial_coords(2),
        t_total_traction(0), t_total_traction(1), t_total_traction(2));

    auto tn = -t_traction(i) * t_grad_sdf(i);
    auto c = constrain(sdf, tn);

    FTensor::Tensor2<double, DIM, DIM> t_cP;
    t_cP(i, j) = (c * t_grad_sdf(i)) * t_grad_sdf(j);
    FTensor::Tensor2<double, DIM, DIM> t_cQ;
    t_cQ(i, j) = kronecker_delta(i, j) - t_cP(i, j);

    FTensor::Tensor2<double, DIM, DIM> t_res_dt;
    t_res_dt(i, j) = -cn_contact * t_cQ(i, j);

    size_t rr = 0;
    for (; rr != AssemblyBoundaryEleOp::nbRows / DIM; ++rr) {

      auto t_mat = getFTensor2FromArray<DIM, DIM, DIM>(locMat, DIM * rr);
      const double row_base = t_row_base(i) * t_normal(i);

      auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
      for (size_t cc = 0; cc != AssemblyBoundaryEleOp::nbCols / DIM; ++cc) {
        const double col_base = t_col_base(i) * t_normal(i);
        const double beta = alpha * row_base * col_base;

        t_mat(i, j) -= beta * t_res_dt(i, j);

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
    ++t_normal_at_pts;
  }

  MoFEMFunctionReturn(0);
}

template <int DIM, AssemblyType A, IntegrationType I, typename DomainEleOp>
MoFEMErrorCode opFactoryDomainRhs(
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip,
    std::string sigma, std::string u) {
  MoFEMFunctionBegin;
  using B = typename FormsIntegrators<DomainEleOp>::template Assembly<
      A>::template LinearForm<I>;
  using OpMixDivURhs = typename B::template OpMixDivTimesU<3, DIM, DIM>;
  using OpMixLambdaGradURhs = typename B::template OpMixTensorTimesGradU<DIM>;
  using OpMixUTimesDivLambdaRhs =
      typename B::template OpMixVecTimesDivLambda<SPACE_DIM>;
  using OpMixUTimesLambdaRhs =
      typename B::template OpGradTimesTensor<1, DIM, DIM>;

  auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();
  auto mat_grad_ptr = boost::make_shared<MatrixDouble>();
  auto div_stress_ptr = boost::make_shared<MatrixDouble>();
  auto contact_stress_ptr = boost::make_shared<MatrixDouble>();

  pip.push_back(new OpCalculateVectorFieldValues<DIM>(
      u, common_data_ptr->contactDispPtr()));
  pip.push_back(
      new OpCalculateHVecTensorField<DIM, DIM>(sigma, contact_stress_ptr));
  pip.push_back(
      new OpCalculateHVecTensorDivergence<DIM, DIM>(sigma, div_stress_ptr));

  pip.push_back(new OpCalculateVectorFieldGradient<DIM, DIM>(u, mat_grad_ptr));

  pip.push_back(
      new OpMixDivURhs(sigma, common_data_ptr->contactDispPtr(),
                       [](double, double, double) constexpr { return 1; }));
  pip.push_back(new OpMixLambdaGradURhs(sigma, mat_grad_ptr));
  pip.push_back(new OpMixUTimesDivLambdaRhs(u, div_stress_ptr));
  pip.push_back(new OpMixUTimesLambdaRhs(u, contact_stress_ptr));

  MoFEMFunctionReturn(0);
}

template <typename OpMixLhs> struct OpMixLhsSide : public OpMixLhs {
  using OpMixLhs::OpMixLhs;
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
    auto side_fe_entity = OpMixLhs::getSidePtrFE()->getFEEntityHandle();
    auto side_fe_data = OpMixLhs::getSideEntity(row_side, row_type);
    // Only assemble side which correspond to edge entity on boundary
    if (side_fe_entity == side_fe_data) {
      CHKERR OpMixLhs::doWork(row_side, col_side, row_type, col_type, row_data,
                              col_data);
    }
    MoFEMFunctionReturn(0);
  }
};

template <int DIM, AssemblyType A, IntegrationType I, typename DomainEle>
MoFEMErrorCode opFactoryBoundaryToDomainLhs(
    MoFEM::Interface &m_field,
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip,
    std::string fe_domain_name, std::string sigma, std::string u,
    std::string geom, ForcesAndSourcesCore::RuleHookFun rule) {
  MoFEMFunctionBegin;
  auto op_loop_side = new OpLoopSide<DomainEle>(
      m_field, fe_domain_name, DIM, Sev::noisy,
      boost::make_shared<ForcesAndSourcesCore::UserDataOperator::AdjCache>());
  pip.push_back(op_loop_side);

  CHKERR AddHOOps<DIM, DIM, DIM>::add(op_loop_side->getOpPtrVector(),
                                      {H1, HDIV}, geom);

  using B = typename FormsIntegrators<DomainEleOp>::template Assembly<
      A>::template BiLinearForm<I>;

  using OpMixDivULhs = typename B::template OpMixDivTimesVec<DIM>;
  using OpLambdaGraULhs = typename B::template OpMixTensorTimesGrad<DIM>;
  using OpMixDivULhsSide = OpMixLhsSide<OpMixDivULhs>;
  using OpLambdaGraULhsSide = OpMixLhsSide<OpLambdaGraULhs>;

  auto unity = []() { return 1; };
  op_loop_side->getOpPtrVector().push_back(
      new OpMixDivULhsSide(sigma, u, unity, true));
  op_loop_side->getOpPtrVector().push_back(
      new OpLambdaGraULhsSide(sigma, u, unity, true));

  op_loop_side->getSideFEPtr()->getRuleHook = rule;
  MoFEMFunctionReturn(0);
}

template <int DIM, AssemblyType A, IntegrationType I, typename BoundaryEleOp>
MoFEMErrorCode opFactoryBoundaryLhs(
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip,
    std::string sigma, std::string u) {
  MoFEMFunctionBegin;

  using C = ContactIntegrators<BoundaryEleOp>;

  auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();

  pip.push_back(new OpCalculateVectorFieldValues<DIM>(
      u, common_data_ptr->contactDispPtr()));
  pip.push_back(new OpCalculateHVecTensorTrace<DIM, BoundaryEleOp>(
      sigma, common_data_ptr->contactTractionPtr()));
  pip.push_back(
      new typename C::template Assembly<A>::template OpConstrainBoundaryLhs_dU<
          DIM, GAUSS>(sigma, u, common_data_ptr));
  pip.push_back(new typename C::template Assembly<A>::
                    template OpConstrainBoundaryLhs_dTraction<DIM, GAUSS>(
                        sigma, sigma, common_data_ptr));

  MoFEMFunctionReturn(0);
}

template <int DIM, AssemblyType A, IntegrationType I, typename BoundaryEleOp>
MoFEMErrorCode opFactoryBoundaryRhs(
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip,
    std::string sigma, std::string u) {
  MoFEMFunctionBegin;

  using C = ContactIntegrators<BoundaryEleOp>;

  auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();

  pip.push_back(new OpCalculateVectorFieldValues<DIM>(
      u, common_data_ptr->contactDispPtr()));
  pip.push_back(new OpCalculateHVecTensorTrace<DIM, BoundaryEleOp>(
      sigma, common_data_ptr->contactTractionPtr()));
  pip.push_back(
      new typename C::template Assembly<A>::template OpConstrainBoundaryRhs<
          DIM, GAUSS>(sigma, common_data_ptr));

  MoFEMFunctionReturn(0);
}

template <int DIM, IntegrationType I, typename BoundaryEleOp>
MoFEMErrorCode opFactoryCalculateTraction(
    boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> &pip,
    std::string sigma) {
  MoFEMFunctionBegin;

  using C = ContactIntegrators<BoundaryEleOp>;

  auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();
  pip.push_back(new OpCalculateHVecTensorTrace<DIM, BoundaryEleOp>(
      sigma, common_data_ptr->contactTractionPtr()));
  pip.push_back(new typename C::template OpAssembleTotalContactTraction<DIM, I>(
      common_data_ptr));

  MoFEMFunctionReturn(0);
}

};     // namespace ContactOps

#endif // __CONTACTOPS_HPP__
