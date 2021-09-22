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

/**
 * \file ContactOps.hpp
 * \example ContactOps.hpp
 */

namespace ContactOps {

constexpr auto VecSetValues = MoFEM::VecSetValues<EssentialBcStorage>;
constexpr auto MatSetValues = MoFEM::MatSetValues<EssentialBcStorage>;

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

struct NestedMatrixContainer {
  AO aO[4]; // row - row - col - col
  ublas::matrix<SmartPetscObj<Mat>> nested_matrices;

  NestedMatrixContainer() { nested_matrices.resize(2, 2); }
  ~NestedMatrixContainer() {
    cout << "DESTRUCTOR CALLED " << endl;
    for (int i = 0; i != 4; i++)
      CHKERR AODestroy(&aO[i]);
  }
  static boost::weak_ptr<NestedMatrixContainer> nestContainerPtr;
};

boost::weak_ptr<NestedMatrixContainer> NestedMatrixContainer::nestContainerPtr;

MoFEMErrorCode CsMatZeroRowsColumns(Mat mat, PetscInt numRows,
                                    const PetscInt rows[], PetscScalar diag,
                                    Vec x, Vec b) {
  MoFEMFunctionBegin;

  if (auto nested_cont = NestedMatrixContainer::nestContainerPtr.lock()) {
    auto &ao = nested_cont->aO;
    auto &nested_matrices = nested_cont->nested_matrices;
    if (numRows > 0) {
      std::vector<int> row_vec0(rows, rows + numRows);
      auto row_vec1 = row_vec0;

      CHKERR AOApplicationToPetsc(ao[0], numRows, &*row_vec0.begin());
      CHKERR AOApplicationToPetsc(ao[1], numRows, &*row_vec1.begin());

      CHKERR MatZeroRowsColumns(nested_matrices(0, 0), numRows,
                                &*row_vec0.begin(), diag, x, b);
      CHKERR MatZeroRowsColumns(nested_matrices(0, 1), numRows,
                                &*row_vec0.begin(), diag, x, b);
      CHKERR MatZeroRowsColumns(nested_matrices(1, 1), numRows,
                                &*row_vec1.begin(), diag, x, b);
    }

  } else
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "nested container weak pointer problem");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CsMatMissingDiagonal(Mat mat, PetscBool *missing, PetscInt *dd) {
  MoFEMFunctionBegin;

  if (auto nested_cont = NestedMatrixContainer::nestContainerPtr.lock()) {
    auto &ao = nested_cont->aO;
    auto &nested_matrices = nested_cont->nested_matrices;
    PetscBool missing0;
    CHKERR MatMissingDiagonal(nested_matrices(0, 0), &missing0, dd);
    CHKERR MatMissingDiagonal(nested_matrices(1, 1), missing, dd);
    *missing = PetscBool(*missing || missing0);
  } else
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "nested container weak pointer problem");

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode CsMatSetValues(Mat mat, PetscInt m, const PetscInt idxm[],
                              PetscInt n, const PetscInt idxn[],
                              const PetscScalar v[], InsertMode addv) {
  MoFEMFunctionBegin;

  if (auto nested_cont = NestedMatrixContainer::nestContainerPtr.lock()) {
    auto &ao = nested_cont->aO;
    auto &nested_matrices = nested_cont->nested_matrices;

    std::vector<int> my_row_indices0(idxm, idxm + m);
    std::vector<int> my_row_indices1(idxm, idxm + m);

    std::vector<int> my_col_indices0(idxn, idxn + n);
    std::vector<int> my_col_indices1(idxn, idxn + n);

    CHKERR AOApplicationToPetsc(ao[0], my_row_indices0.size(),
                                &*my_row_indices0.begin());
    CHKERR AOApplicationToPetsc(ao[1], my_row_indices1.size(),
                                &*my_row_indices1.begin());
    CHKERR AOApplicationToPetsc(ao[2], my_col_indices0.size(),
                                &*my_col_indices0.begin());
    CHKERR AOApplicationToPetsc(ao[3], my_col_indices1.size(),
                                &*my_col_indices1.begin());

    ::MatSetValues(nested_matrices(0, 0), my_row_indices0.size(),
                   &*my_row_indices0.begin(), my_col_indices0.size(),
                   &*my_col_indices0.begin(), v, addv);
    ::MatSetValues(nested_matrices(0, 1), my_row_indices0.size(),
                   &*my_row_indices0.begin(), my_col_indices1.size(),
                   &*my_col_indices1.begin(), v, addv);

    return ::MatSetValues(nested_matrices(1, 1), my_row_indices1.size(),
                          &*my_row_indices1.begin(), my_col_indices1.size(),
                          &*my_col_indices1.begin(), v, addv);
  } else
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "nested container weak pointer problem");

  MoFEMFunctionReturn(0);
}

struct OpConstrainBoundaryRhs : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryRhs(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainBoundaryLhs_dU : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryLhs_dU(const std::string row_field_name,
                            const std::string col_field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainBoundaryLhs_dTraction : public AssemblyBoundaryEleOp {
  OpConstrainBoundaryLhs_dTraction(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                           DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

template <typename T1, typename T2>
inline FTensor::Tensor1<double, SPACE_DIM>
normal(FTensor::Tensor1<T1, 3> &t_coords,
       FTensor::Tensor1<T2, SPACE_DIM> &t_disp) {
  FTensor::Tensor1<double, SPACE_DIM> t_normal;
  t_normal(i) = 0;
  t_normal(1) = 1.;
  return t_normal;
}

template <typename T>
inline double gap0(FTensor::Tensor1<T, 3> &t_coords,
                   FTensor::Tensor1<double, SPACE_DIM> &t_normal) {
  return (-0.5 - t_coords(1)) * t_normal(1);
}

template <typename T>
inline double gap(FTensor::Tensor1<T, SPACE_DIM> &t_disp,
                  FTensor::Tensor1<double, SPACE_DIM> &t_normal) {
  return t_disp(i) * t_normal(i);
}

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

inline double w(const double g, const double t) { return g - cn * t; }

inline double constrian(double &&g0, double &&g, double &&t) {
  return (w(g - g0, t) + std::abs(w(g - g0, t))) / 2 + g0;
};

inline double diff_constrains_dtraction(double &&g0, double &&g, double &&t) {
  return -cn * (1 + sign(w(g - g0, t))) / 2;
}

inline double diff_constrains_dgap(double &&g0, double &&g, double &&t) {
  return (1 + sign(w(g - g0, t))) / 2;
}

OpConstrainBoundaryRhs::OpConstrainBoundaryRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblyBoundaryEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpConstrainBoundaryRhs::iNtegrate(DataForcesAndSourcesCore::EntData &data) {
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

    auto t_contact_normal = normal(t_coords, t_disp);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
    t_P(i, j) = t_contact_normal(i) * t_contact_normal(j);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
    t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

    FTensor::Tensor1<double, SPACE_DIM> t_rhs_constrains;

    t_rhs_constrains(i) =
        t_contact_normal(i) *
        constrian(gap0(t_coords, t_contact_normal),
                  gap(t_disp, t_contact_normal),
                  normal_traction(t_traction, t_contact_normal));

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
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
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

    auto t_contact_normal = normal(t_coords, t_disp);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
    t_P(i, j) = t_contact_normal(i) * t_contact_normal(j);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
    t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

    auto diff_constrain = diff_constrains_dgap(
        gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
        normal_traction(t_traction, t_contact_normal));

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
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
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

    auto t_contact_normal = normal(t_coords, t_disp);
    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_P;
    t_P(i, j) = t_contact_normal(i) * t_contact_normal(j);

    FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_Q;
    t_Q(i, j) = kronecker_delta(i, j) - t_P(i, j);

    const double diff_traction = diff_constrains_dtraction(
        gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
        normal_traction(t_traction, t_contact_normal));

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

        t_mat(i, j) += (beta * diff_traction) * t_P(i, j);
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

struct OpRotate : BoundaryEleOp {

  using ScalarFun =
      boost::function<double(const double, const double, const double)>;

  OpRotate(std::string disp_name, boost::shared_ptr<VectorDouble> omega,
           boost::shared_ptr<VectorDouble> c_coords, ScalarFun scalar_fun,
           boost::shared_ptr<Range> ents_ptr)
      : BoundaryEleOp(disp_name, BoundaryEleOp::OPROW), entsPtr(ents_ptr),
        sourceVec(omega), centerCoords(c_coords), betaCoeff(scalar_fun) {}

  MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_row_base_functions = row_data.getN().size2();

    if (nb_row_dofs) {

      EntityHandle ent = getFEEntityHandle();
      if (entsPtr->find(ent) == entsPtr->end()) {
        MoFEMFunctionReturnHot(0);
      }

      auto t_coords = getFTensor1CoordsAtGaussPts();

      locRes.resize(nb_row_dofs, false);
      locRes.clear();

      const int nb_integration_pts = getGaussPts().size2();
      auto t_row_val = row_data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      FTensor::Tensor1<double, SPACE_DIM> rot_disp;
      auto t_omega = getFTensor1FromArray<3, 3>(*sourceVec);
      auto t_cen_coords = getFTensor1FromArray<3, 3>(*centerCoords);
      FTensor::Tensor1<double, 3> scaled_omega;
      FTensor::Tensor1<double, 3> c_dist;
      FTensor::Index<'I', 3> I;

      auto get_rotation = [&](auto vec) {
        FTensor::Tensor2<double, 3, 3> t_R;

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;

        constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
        t_R(i, j) = t_kd(i, j);

        const double angle = sqrt(vec(i) * vec(i));
        if (std::abs(angle) < 1e-18)
          return t_R;

        FTensor::Tensor2<double, 3, 3> t_ang;
        t_ang(i, j) = FTensor::levi_civita<double>(i, j, k) * vec(k);
        const double a = sin(angle) / angle;
        const double ss_2 = sin(angle / 2.);
        const double b = 2. * ss_2 * ss_2 / (angle * angle);
        t_R(i, j) += a * t_ang(i, j);
        t_R(i, j) += b * t_ang(i, k) * t_ang(k, j);

        return t_R;
      };

      // scale with time (t_coords is just a placeholder)
      scaled_omega(I) =
          t_omega(I) * -betaCoeff(t_coords(0), t_coords(1), t_coords(2));
      auto rot = get_rotation(scaled_omega);

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        c_dist(i) = t_cen_coords(i) - t_coords(i);
        rot_disp(i) = c_dist(i) - rot(j, i) * c_dist(j);

        auto t_nf = getFTensor1FromArray<SPACE_DIM, SPACE_DIM>(locRes);
        for (int rr = 0; rr != nb_row_dofs / SPACE_DIM; ++rr) {
          t_nf(i) -= a * t_row_val * rot_disp(i);
          ++t_row_val;
          ++t_nf;
        }

        ++t_coords;
        ++t_w;
      }

      CHKERR VecSetValues(getSNESf(), row_data, &*locRes.begin(), ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

private:
  VectorDouble locRes;
  ScalarFun betaCoeff;
  boost::shared_ptr<Range> entsPtr;
  boost::shared_ptr<VectorDouble> sourceVec;
  boost::shared_ptr<VectorDouble> centerCoords;
};

}; // namespace ContactOps
