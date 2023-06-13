

/**
 * \file ContactOps.hpp
 * \example ContactOps.hpp
 */

#ifndef __CONTACTOPS_HPP__
#define __CONTACTOPS_HPP__

namespace ContactOps {

//! [Common data]
struct CommonData : public boost::enable_shared_from_this<CommonData> {
  MatrixDouble mD;
  MatrixDouble mGrad;

  MatrixDouble contactStress;
  MatrixDouble contactStressDivergence;
  MatrixDouble contactTraction;
  MatrixDouble contactDisp;
  MatrixDouble stressTraction;

  static SmartPetscObj<Vec> totalTraction;
  ~CommonData() { totalTraction.reset(); }

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

  inline auto mDPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &mD);
  }

  inline auto mGradPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &mGrad);
  }

  inline auto contactStressPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &contactStress);
  }

  inline auto contactStressDivergencePtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(),
                                           &contactStressDivergence);
  }

  inline auto contactTractionPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(),
                                           &contactTraction);
  }

  inline auto contactDispPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &contactDisp);
  }

  inline auto stressTractionPtr() {
    return boost::shared_ptr<MatrixDouble>(shared_from_this(), &stressTraction);
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

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;

#ifdef __HENKY_OPS_HPP__

struct OpCalculateStressTraction : public BoundaryEleOp {
  OpCalculateStressTraction(
      boost::shared_ptr<CommonData> common_data_ptr,
      boost::shared_ptr<HenckyOps::CommonData> hencky_common_data);
  MoFEMErrorCode doWork(int side, EntityType type,
                        EntitiesFieldData::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<HenckyOps::CommonData> henckyCommonDataPtr;
};

#endif

struct OpAssembleTotalContactTraction : public BoundaryEleOp {
  OpAssembleTotalContactTraction(boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainBoundaryRhs : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryRhs(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(EntitiesFieldData::EntData &data);

  SurfaceDistanceFunction surfaceDistanceFunction = surface_distance_function;
  GradSurfaceDistanceFunction gradSurfaceDistanceFunction =
      grad_surface_distance_function;

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainBoundaryLhs_dU : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryLhs_dU(const std::string row_field_name,
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

struct OpConstrainBoundaryLhs_dTraction : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryLhs_dTraction(
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

#ifdef __HENKY_OPS_HPP__

OpCalculateStressTraction::OpCalculateStressTraction(
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<HenckyOps::CommonData> hencky_common_data_ptr)
    : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE),
      commonDataPtr(common_data_ptr),
      henckyCommonDataPtr(hencky_common_data_ptr) {}

MoFEMErrorCode
OpCalculateStressTraction::doWork(int side, EntityType type,
                                  EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  auto t_normal_at_pts = getFTensor1NormalsAtGaussPts();
  const auto nb_integration_pts = getGaussPts().size2();
  auto traction_ptr = commonDataPtr->stressTractionPtr();
  traction_ptr->resize(SPACE_DIM, nb_integration_pts, false);
  auto t_traction = getFTensor1FromMat<SPACE_DIM>(*traction_ptr);
  auto t_P = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(
      *(henckyCommonDataPtr->getMatFirstPiolaStress()));
  for (auto gg = 0; gg != nb_integration_pts; ++gg) {
    FTensor::Tensor1<double, SPACE_DIM> t_normal;
    t_normal(i) =
        t_normal_at_pts(i) / std::sqrt(t_normal_at_pts(i) * t_normal_at_pts(i));
    t_traction(i) = t_P(i, j) * t_normal(j);
    ++t_P;
    ++t_traction;
    ++t_normal_at_pts;
  }
  MoFEMFunctionReturn(0);
}

#endif //__HENKY_OPS_HPP__

OpAssembleTotalContactTraction::OpAssembleTotalContactTraction(
    boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpAssembleTotalContactTraction::doWork(int side, EntityType type,
                                                      EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Tensor1<double, 3> t_sum_t{0., 0., 0.};

  auto t_w = getFTensor0IntegrationWeight();
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactTraction);

  const auto nb_gauss_pts = getGaussPts().size2();
  for (auto gg = 0; gg != nb_gauss_pts; ++gg) {
    const double alpha = t_w * getMeasure();
    t_sum_t(i) += alpha * t_traction(i);
    ++t_w;
    ++t_traction;
  }

  constexpr int ind[] = {0, 1, 2};
  CHKERR VecSetValues(commonDataPtr->totalTraction, 3, ind, &t_sum_t(0),
                      ADD_VALUES);

  MoFEMFunctionReturn(0);
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

  auto t_normal_at_pts = getFTensor1NormalsAtGaussPts();
  auto t_total_traction = CommonData::getFTensor1TotalTraction();

  auto t_w = getFTensor0IntegrationWeight();
  auto t_disp = getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactDisp);
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactTraction);
  auto t_coords = getFTensor1CoordsAtGaussPts();

  size_t nb_base_functions = data.getN().size2() / 3;
  auto t_base = data.getFTensor1N<3>();
  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor1<double, SPACE_DIM> t_normal;
    t_normal(i) =
        t_normal_at_pts(i) / std::sqrt(t_normal_at_pts(i) * t_normal_at_pts(i));

    auto t_nf = getFTensor1FromPtr<SPACE_DIM>(&nf[0]);
    const double alpha = t_w * getMeasure();

    FTensor::Tensor1<double, 3> t_spatial_coords{0., 0., 0.};
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto sdf = surfaceDistanceFunction(getTStime(), t_spatial_coords(0),
                                       t_spatial_coords(1), t_spatial_coords(2),
                                       t_total_traction(0), t_total_traction(1),
                                       t_total_traction(2));

    auto t_grad_sdf = gradSurfaceDistanceFunction(
        getTStime(), t_spatial_coords(0), t_spatial_coords(1),
        t_spatial_coords(2), t_total_traction(0), t_total_traction(1),
        t_total_traction(2));

    auto tn = -t_traction(i) * t_grad_sdf(i);
    auto c = constrain(sdf, tn);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_cP;
    t_cP(i, j) = (c * t_grad_sdf(i)) * t_grad_sdf(j);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_cQ;
    t_cQ(i, j) = kronecker_delta(i, j) - t_cP(i, j);

    FTensor::Tensor1<double, SPACE_DIM> t_rhs;
    t_rhs(i) =

        t_cQ(i, j) * (t_disp(j) - cn_contact * t_traction(j))

        +

        t_cP(i, j) * t_disp(j) +
        c * (sdf * t_grad_sdf(i)); // add gap0 displacements

    size_t bb = 0;
    for (; bb != AssemblyBoundaryEleOp::nbRows / SPACE_DIM; ++bb) {
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

  auto t_normal_at_pts = getFTensor1NormalsAtGaussPts();
  auto t_total_traction = CommonData::getFTensor1TotalTraction();

  auto t_disp = getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactDisp);
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactTraction);
  auto t_coords = getFTensor1CoordsAtGaussPts();

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor1N<3>();
  size_t nb_face_functions = row_data.getN().size2() / 3;

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor1<double, SPACE_DIM> t_normal;
    t_normal(i) =
        t_normal_at_pts(i) / std::sqrt(t_normal_at_pts(i) * t_normal_at_pts(i));

    const double alpha = t_w * getMeasure();

    FTensor::Tensor1<double, 3> t_spatial_coords{0., 0., 0.};
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto sdf = surfaceDistanceFunction(getTStime(), t_spatial_coords(0),
                                       t_spatial_coords(1), t_spatial_coords(2),
                                       t_total_traction(0), t_total_traction(1),
                                       t_total_traction(2));
    auto t_grad_sdf = gradSurfaceDistanceFunction(
        getTStime(), t_spatial_coords(0), t_spatial_coords(1),
        t_spatial_coords(2), t_total_traction(0), t_total_traction(1),
        t_total_traction(2));

    auto tn = -t_traction(i) * t_grad_sdf(i);
    auto c = constrain(sdf, tn);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_cP;
    t_cP(i, j) = (c * t_grad_sdf(i)) * t_grad_sdf(j);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_cQ;
    t_cQ(i, j) = kronecker_delta(i, j) - t_cP(i, j);

    FTensor::Tensor2<double, 3, 3> t_res_dU;
    t_res_dU(i, j) = kronecker_delta(i, j) + t_cP(i, j);

    if (c > 0) {
      auto t_hess_sdf = hessSurfaceDistanceFunction(
          getTStime(), t_spatial_coords(0), t_spatial_coords(1),
          t_spatial_coords(2), t_total_traction(0), t_total_traction(1),
          t_total_traction(2));
      t_res_dU(i, j) +=
          (c * cn_contact) *
              (t_hess_sdf(i, j) * (t_grad_sdf(k) * t_traction(k)) +
               t_grad_sdf(i) * t_hess_sdf(k, j) * t_traction(k)) +
          c * sdf * t_hess_sdf(i, j);
    }

    size_t rr = 0;
    for (; rr != AssemblyBoundaryEleOp::nbRows / SPACE_DIM; ++rr) {

      auto t_mat = getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(
          locMat, SPACE_DIM * rr);

      const double row_base = t_row_base(i) * t_normal(i);

      auto t_col_base = col_data.getFTensor0N(gg, 0);
      for (size_t cc = 0; cc != AssemblyBoundaryEleOp::nbCols / SPACE_DIM;
           ++cc) {
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

  auto t_normal_at_pts = getFTensor1NormalsAtGaussPts();
  auto t_total_traction = CommonData::getFTensor1TotalTraction();

  auto t_disp = getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactDisp);
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactTraction);
  auto t_coords = getFTensor1CoordsAtGaussPts();

  auto t_w = getFTensor0IntegrationWeight();
  auto t_row_base = row_data.getFTensor1N<3>();
  size_t nb_face_functions = row_data.getN().size2() / 3;

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    FTensor::Tensor1<double, SPACE_DIM> t_normal;
    t_normal(i) =
        t_normal_at_pts(i) / std::sqrt(t_normal_at_pts(i) * t_normal_at_pts(i));

    const double alpha = t_w * getMeasure();

    FTensor::Tensor1<double, 3> t_spatial_coords{0., 0., 0.};
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto sdf = surfaceDistanceFunction(getTStime(), t_spatial_coords(0),
                                       t_spatial_coords(1), t_spatial_coords(2),
                                       t_total_traction(0), t_total_traction(1),
                                       t_total_traction(2));
    auto t_grad_sdf = gradSurfaceDistanceFunction(
        getTStime(), t_spatial_coords(0), t_spatial_coords(1),
        t_spatial_coords(2), t_total_traction(0), t_total_traction(1),
        t_total_traction(2));

    auto tn = -t_traction(i) * t_grad_sdf(i);
    auto c = constrain(sdf, tn);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_cP;
    t_cP(i, j) = (c * t_grad_sdf(i)) * t_grad_sdf(j);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_cQ;
    t_cQ(i, j) = kronecker_delta(i, j) - t_cP(i, j);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_res_dt;
    t_res_dt(i, j) = -cn_contact * t_cQ(i, j);

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

};     // namespace ContactOps

#endif // __CONTACTOPS_HPP__
