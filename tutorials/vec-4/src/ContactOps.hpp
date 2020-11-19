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

namespace OpContactTools {

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

struct OpConstrainBoundaryRhs : public BoundaryEleOp {
  OpConstrainBoundaryRhs(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainBoundaryLhs_dU : public BoundaryEleOp {
  OpConstrainBoundaryLhs_dU(const std::string row_field_name,
                            const std::string col_field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpConstrainBoundaryLhs_dTraction : public BoundaryEleOp {
  OpConstrainBoundaryLhs_dTraction(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
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
    : BoundaryEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpConstrainBoundaryRhs::doWork(int side, EntityType type,
                                              EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_normal = getFTensor1Normal();
    t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

    auto t_w = getFTensor0IntegrationWeight();
    auto t_disp =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
    auto t_traction =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));
    auto t_coords = getFTensor1CoordsAtGaussPts();

    size_t nb_base_functions = data.getN().size2() / 3;
    auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_nf = getFTensor1FromPtr<SPACE_DIM>(nf.data());

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
      for (; bb != nb_dofs / SPACE_DIM; ++bb) {
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

    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpConstrainBoundaryLhs_dU::OpConstrainBoundaryLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}
MoFEMErrorCode OpConstrainBoundaryLhs_dU::doWork(int row_side, int col_side,
                                                 EntityType row_type,
                                                 EntityType col_type,
                                                 EntData &row_data,
                                                 EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_normal = getFTensor1Normal();
    t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

    auto t_disp =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
    auto t_traction =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));
    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

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
      for (; rr != row_nb_dofs / SPACE_DIM; ++rr) {

        auto t_mat = getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(
            locMat, SPACE_DIM * rr);

        const double row_base = t_row_base(i) * t_normal(i);

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / SPACE_DIM; ++cc) {
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

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpConstrainBoundaryLhs_dTraction::OpConstrainBoundaryLhs_dTraction(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}
MoFEMErrorCode OpConstrainBoundaryLhs_dTraction::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_normal = getFTensor1Normal();
    t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

    auto t_disp =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
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
      for (; rr != row_nb_dofs / SPACE_DIM; ++rr) {

        auto t_mat = getFTensor2FromArray<SPACE_DIM, SPACE_DIM, SPACE_DIM>(
            locMat, SPACE_DIM * rr);
        const double row_base = t_row_base(i) * t_normal(i);

        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / SPACE_DIM; ++cc) {
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

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}






}; // namespace OpContactTools
