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
};
//! [Common data]

FTensor::Index<'i', SPACE_DIM> i;
FTensor::Index<'j', SPACE_DIM> j;
FTensor::Index<'k', SPACE_DIM> k;
FTensor::Index<'l', SPACE_DIM> l;

struct OpInternalBoundaryContactRhs : public BoundaryEleOp {
  OpInternalBoundaryContactRhs(const std::string field_name,
                               boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpInternalDomainContactRhs : public DomainEleOp {
  OpInternalDomainContactRhs(const std::string field_name,
                             boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

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

struct OpSpringRhs : public BoundaryEleOp {
  OpSpringRhs(const std::string field_name,
              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpSpringLhs : public BoundaryEleOp {
  OpSpringLhs(const std::string row_field_name,
              const std::string col_field_name);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  MatrixDouble locMat;
};

struct OpConstrainBoundaryTraction : public BoundaryEleOp {
  OpConstrainBoundaryTraction(const std::string field_name,
                              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainDomainRhs : public DomainEleOp {
  OpConstrainDomainRhs(const std::string field_name,
                       boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainDomainLhs_dU : public DomainEleOp {
  OpConstrainDomainLhs_dU(const std::string row_field_name,
                          const std::string col_field_name,
                          boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
  MatrixDouble transLocMat;
};

struct OpConstrainDomainLhs_dLdL : public DomainEleOp {
  OpConstrainDomainLhs_dLdL(const std::string row_field_name,
                          const std::string col_field_name,
                          boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
  MatrixDouble transLocMat;
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
  return (-1 - t_coords(1)) * t_normal(1);
}

template <typename T>
inline double gap(FTensor::Tensor1<T, SPACE_DIM> &t_disp,
                  FTensor::Tensor1<double, SPACE_DIM> &t_normal) {
  return t_disp(i) * t_normal(i);
}

template <typename T>
inline double normal_traction(FTensor::Tensor1<T, SPACE_DIM> &t_traction,
                              FTensor::Tensor1<double, SPACE_DIM> &t_normal) {
  return t_traction(i) * t_normal(i);
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

OpInternalBoundaryContactRhs::OpInternalBoundaryContactRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpInternalBoundaryContactRhs::doWork(int side, EntityType type,
                                                    EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_traction =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));

    size_t nb_base_functions = data.getN().size2();
    auto t_base = data.getFTensor0N();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_nf = getFTensor1FromPtr<SPACE_DIM>(nf.data());

      const double alpha = t_w * getMeasure();

      size_t bb = 0;
      for (; bb != nb_dofs / SPACE_DIM; ++bb) {
        t_nf(i) -= alpha * t_base * t_traction(i);
        for (int zz = 0; zz != SPACE_DIM; ++zz)
          integral_1_rhs -= alpha * t_base * t_traction(zz);

        ++t_nf;
        ++t_base;
      }
      for (; bb != nb_base_functions; ++bb)
        ++t_base;

      ++t_traction;
      ++t_w;
    }

    // CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpInternalDomainContactRhs::OpInternalDomainContactRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpInternalDomainContactRhs::doWork(int side, EntityType type,
                                                  EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    const size_t nb_base_functions = data.getN().size2();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();
    auto t_diff_base = data.getFTensor1DiffN<SPACE_DIM>();
    auto t_stress = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(
        *(commonDataPtr->contactStressPtr));
    auto t_div_stress = getFTensor1FromMat<SPACE_DIM>(
        *(commonDataPtr->contactStressDivergencePtr));

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = getMeasure() * t_w;
      auto t_nf = getFTensor1FromPtr<SPACE_DIM>(nf.data());

      size_t bb = 0;
      for (; bb != nb_dofs / SPACE_DIM; ++bb) {

        t_nf(i) += alpha * t_base * t_div_stress(i);
        t_nf(i) += alpha * t_diff_base(j) * t_stress(i, j);

        for (int zz = 0; zz != SPACE_DIM; ++zz)
          integral_1_lhs += alpha * t_base * t_div_stress(zz);
        FTensor::Tensor1<double, SPACE_DIM> inte;
        inte(i) = alpha * t_diff_base(j) * t_stress(i, j);
        for (int zz = 0; zz != SPACE_DIM; ++zz)
          integral_1_lhs += inte(zz);
        // if(SPACE_DIM == 3)
        //   t_nf(i) *= -1;
        ++t_nf;
        ++t_base;
        ++t_diff_base;
      }
      for (; bb < nb_base_functions; ++bb) {
        ++t_base;
        ++t_diff_base;
      }

      ++t_div_stress;
      ++t_stress;
      ++t_w;
    }

    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
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
    if (SPACE_DIM == 2) //FIXME: make normal outward
      t_normal(i) *= -1;

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
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) =
          kronecker_delta(i, j) - t_contact_normal_tensor(i, j);

      FTensor::Tensor1<double, SPACE_DIM> t_rhs_constrains;

      t_rhs_constrains(i) =
          t_contact_normal(i) *
          constrian(gap0(t_coords, t_contact_normal),
                    gap(t_disp, t_contact_normal),
                    normal_traction(t_traction, t_contact_normal));

      FTensor::Tensor1<double, SPACE_DIM> t_rhs_tangent_disp,
          t_rhs_tangent_traction;
      t_rhs_tangent_disp(i) = t_contact_tangent_tensor(i, j) * t_disp(j);
      t_rhs_tangent_traction(i) =
          cn * t_contact_tangent_tensor(i, j) * t_traction(j);

      size_t bb = 0;
      for (; bb != nb_dofs / SPACE_DIM; ++bb) {
        const double beta = alpha * (t_base(i) * t_normal(i));

        t_nf(i) += beta * t_rhs_constrains(i);
        t_nf(i) += beta * t_rhs_tangent_disp(i);
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

OpConstrainBoundaryTraction::OpConstrainBoundaryTraction(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  doEntities[boundary_ent] = true;
}

MoFEMErrorCode OpConstrainBoundaryTraction::doWork(int side, EntityType type,
                                                   EntData &data) {

  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();

  if (side == 0 && type == boundary_ent) {
    commonDataPtr->contactTractionPtr->resize(SPACE_DIM, nb_gauss_pts);
    commonDataPtr->contactTractionPtr->clear();
  }

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto t_normal = getFTensor1Normal();
    t_normal(i) /= sqrt(t_normal(j) * t_normal(j));
    if (SPACE_DIM == 2) //FIXME: make normal outward
      t_normal(i) *= -1;

    auto t_traction =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));

    size_t nb_base_functions = data.getN().size2() / 3;
    auto t_base = data.getFTensor1N<3>();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_field_data = data.getFTensor1FieldData<SPACE_DIM>();

      size_t bb = 0;
      for (; bb != nb_dofs / SPACE_DIM; ++bb) {
        t_traction(j) += (t_base(i) * t_normal(i)) * t_field_data(j);
        ++t_field_data;
        ++t_base;
      }

      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_traction;
    }
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
    if (SPACE_DIM == 2) //FIXME: make normal outward
      t_normal(i) *= -1;

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
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) =
          kronecker_delta(i, j) - t_contact_normal_tensor(i, j);

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

          t_mat(i, j) +=
              (beta * diff_constrain) * t_contact_normal_tensor(i, j);
          t_mat(i, j) += beta * t_contact_tangent_tensor(i, j);

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
    if (SPACE_DIM == 2) //FIXME: make normal outward
      t_normal(i) *= -1;

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
      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_contact_normal_tensor;
      t_contact_normal_tensor(i, j) = t_contact_normal(i) * t_contact_normal(j);

      FTensor::Tensor2<double, SPACE_DIM, SPACE_DIM> t_contact_tangent_tensor;
      t_contact_tangent_tensor(i, j) =
          kronecker_delta(i, j) - t_contact_normal_tensor(i, j);

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

          t_mat(i, j) += (beta * diff_traction) * t_contact_normal_tensor(i, j);
          t_mat(i, j) += beta * cn * t_contact_tangent_tensor(i, j);

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

OpConstrainDomainRhs::OpConstrainDomainRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpConstrainDomainRhs::doWork(int side, EntityType type,
                                            EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    const size_t nb_base_functions = data.getN().size2() / 3;
    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor1N<3>();
    auto t_diff_base = data.getFTensor2DiffN<3, SPACE_DIM>();
    auto t_stress = getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(
        *(commonDataPtr->contactStressPtr));
    auto t_disp =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
    auto t_grad =
        getFTensor2FromMat<SPACE_DIM, SPACE_DIM>(*(commonDataPtr->mGradPtr));

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = getMeasure() * t_w;
      auto t_nf = getFTensor1FromPtr<SPACE_DIM>(nf.data());

      size_t bb = 0;
      for (; bb != nb_dofs / SPACE_DIM; ++bb) {
        const double t_div_base = t_diff_base(i, i);

        t_nf(i) += alpha * t_base(j) * t_grad(i, j);
        t_nf(i) += alpha * t_div_base * t_disp(i);

        for (int zz = 0; zz != SPACE_DIM; ++zz)
          integral_2_lhs += alpha * t_div_base * t_disp(zz);

        ++t_nf;
        ++t_base;
        ++t_diff_base;
      }
      for (; bb < nb_base_functions; ++bb) {
        ++t_base;
        ++t_diff_base;
      }

      ++t_stress;
      ++t_disp;
      ++t_grad;
      ++t_w;
    }

    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpConstrainDomainLhs_dU::OpConstrainDomainLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpConstrainDomainLhs_dU::doWork(int row_side, int col_side,
                                               EntityType row_type,
                                               EntityType col_type,
                                               EntData &row_data,
                                               EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    auto t_w = getFTensor0IntegrationWeight();

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();
    transLocMat.resize(col_nb_dofs, row_nb_dofs, false);

    size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_row_base = row_data.getFTensor1N<3>();
    auto t_row_diff_base = row_data.getFTensor2DiffN<3, SPACE_DIM>();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != row_nb_dofs / SPACE_DIM; ++rr) {

        auto t_mat_diag = getFTensor1FromArrayDiag<SPACE_DIM, SPACE_DIM>(
            locMat, SPACE_DIM * rr);
        const double t_row_div_base = t_row_diff_base(i, i);

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (size_t cc = 0; cc != col_nb_dofs / SPACE_DIM; ++cc) {

          t_mat_diag(i) += alpha * t_row_base(j) * t_col_diff_base(j);
          t_mat_diag(i) += alpha * t_row_div_base * t_col_base;

          ++t_col_base;
          ++t_col_diff_base;
          ++t_mat_diag;
        }

        ++t_row_diff_base;
        ++t_row_base;
      }
      for (; rr < nb_base_functions; ++rr) {
        ++t_row_diff_base;
        ++t_row_base;
      }

      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
    noalias(transLocMat) = trans(locMat);
    CHKERR MatSetValues(getSNESB(), col_data, row_data,
                        &*transLocMat.data().begin(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpConstrainDomainLhs_dU::OpConstrainDomainLhs_dLdL(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode OpConstrainDomainLhs_dLdL::doWork(int row_side, int col_side,
                                               EntityType row_type,
                                               EntityType col_type,
                                               EntData &row_data,
                                               EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    auto t_w = getFTensor0IntegrationWeight();

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();
    transLocMat.resize(col_nb_dofs, row_nb_dofs, false);

    size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_row_base = row_data.getFTensor1N<3>();
    auto t_row_diff_base = row_data.getFTensor2DiffN<3, SPACE_DIM>();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != row_nb_dofs / SPACE_DIM; ++rr) {

        auto t_mat_diag = getFTensor1FromArrayDiag<SPACE_DIM, SPACE_DIM>(
            locMat, SPACE_DIM * rr);
        const double t_row_div_base = t_row_diff_base(i, i);

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<SPACE_DIM>(gg, 0);

        for (size_t cc = 0; cc != col_nb_dofs / SPACE_DIM; ++cc) {

          t_mat_diag(i) += alpha * t_row_base(j) * t_col_diff_base(j);
          t_mat_diag(i) += alpha * t_row_div_base * t_col_base;

          ++t_col_base;
          ++t_col_diff_base;
          ++t_mat_diag;
        }

        ++t_row_diff_base;
        ++t_row_base;
      }
      for (; rr < nb_base_functions; ++rr) {
        ++t_row_diff_base;
        ++t_row_base;
      }

      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
    noalias(transLocMat) = trans(locMat);
    CHKERR MatSetValues(getSNESB(), col_data, row_data,
                        &*transLocMat.data().begin(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpSpringRhs::OpSpringRhs(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpSpringRhs::doWork(int side, EntityType type, EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_disp =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));

    size_t nb_base_functions = data.getN().size2();
    auto t_base = data.getFTensor0N();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_nf = getFTensor1FromPtr<SPACE_DIM>(nf.data());
      const double alpha = t_w * getMeasure();

      size_t bb = 0;
      for (; bb != nb_dofs / SPACE_DIM; ++bb) {

        const double beta = alpha * t_base;
        t_nf(i) += beta * spring_stiffness * t_disp(i);

        ++t_nf;
        ++t_base;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_disp;
      ++t_w;
    }

    CHKERR VecSetValues(getSNESf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpSpringLhs::OpSpringLhs(const std::string row_field_name,
                         const std::string col_field_name)
    : BoundaryEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL) {
  sYmm = false;
}

MoFEMErrorCode OpSpringLhs::doWork(int row_side, int col_side,
                                   EntityType row_type, EntityType col_type,
                                   EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    size_t nb_face_functions = row_data.getN().size2();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = t_w * getMeasure();

      size_t rr = 0;
      for (; rr != row_nb_dofs / SPACE_DIM; ++rr) {

        auto t_mat_diag = getFTensor1FromArrayDiag<SPACE_DIM, SPACE_DIM>(
            locMat, SPACE_DIM * rr);

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / SPACE_DIM; ++cc) {
          t_mat_diag(i) += alpha * spring_stiffness * t_row_base * t_col_base;
          ++t_col_base;
          ++t_mat_diag;
        }

        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

struct OpPostProcDebug : public BoundaryEleOp {
  OpPostProcDebug(MoFEM::Interface &m_field, const std::string field_name,
                  boost::shared_ptr<CommonData> common_data_ptr,
                  moab::Interface *moab_debug = nullptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  MoFEM::Interface &mField;
  moab::Interface *moabDebug;
  boost::shared_ptr<CommonData> commonDataPtr;
};

OpPostProcDebug::OpPostProcDebug(MoFEM::Interface &m_field,
                                 const std::string field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr,
                                 moab::Interface *moab_debug)
    : mField(m_field), BoundaryEleOp(field_name, BoundaryEleOp::OPROW),
      commonDataPtr(common_data_ptr), moabDebug(moab_debug) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  doEntities[boundary_ent] = true;
}

MoFEMErrorCode OpPostProcDebug::doWork(int side, EntityType type,
                                       EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();
  std::array<double, 9> def;
  std::fill(def.begin(), def.end(), 0);

  auto get_tag = [&](const std::string name, size_t size) {
    Tag th;
    CHKERR moabDebug->tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                     MB_TAG_CREAT | MB_TAG_SPARSE, def.data());
    return th;
  };

  // if (nb_dofs) {

  auto t_coords = getFTensor1CoordsAtGaussPts();
  auto t_disp = getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));

  auto th_gap = get_tag("GAP", 1);
  auto th_cons = get_tag("CONSTRAINT", 1);
  auto th_traction = get_tag("TRACTION", 3);
  auto th_normal = get_tag("NORMAL", 3);
  auto th_cont_normal = get_tag("CONTACT_NORMAL", 3);
  auto th_disp = get_tag("DISPLACEMENT", 3);

  EntityHandle ent = getFEEntityHandle();

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    double coords[] = {0, 0, 0};
    EntityHandle new_vertex;
    for (int dd = 0; dd != 3; dd++) {
      coords[dd] = getCoordsAtGaussPts()(gg, dd);
    }
    CHKERR moabDebug->create_vertex(&coords[0], new_vertex);
    FTensor::Tensor1<double, 3> trac(t_traction(0), t_traction(1), 0.);
    FTensor::Tensor1<double, 3> disp(t_disp(0), t_disp(1), 0.);

    auto t_contact_normal = normal(t_coords, t_disp);
    double _gap0 = gap0(t_coords, t_contact_normal);
    double _gap = gap(t_disp, t_contact_normal);
    double gap_tot = _gap - _gap0;
    double constra = constrian(gap0(t_coords, t_contact_normal),
                               gap(t_disp, t_contact_normal),
                               normal_traction(t_traction, t_contact_normal));
    CHKERR moabDebug->tag_set_data(th_gap, &new_vertex, 1, &gap_tot);
    CHKERR moabDebug->tag_set_data(th_cons, &new_vertex, 1, &constra);

    FTensor::Tensor1<double, 3> norm(t_contact_normal(0), t_contact_normal(1),
                                     0.);

    if (SPACE_DIM == 3) {
      trac(2) = t_traction(2);
      norm(2) = t_contact_normal(2);
      disp(2) = t_disp(2);
    }

    CHKERR moabDebug->tag_set_data(th_traction, &new_vertex, 1, &trac(0));
    CHKERR moabDebug->tag_set_data(th_cont_normal, &new_vertex, 1, &norm(0));
    CHKERR moabDebug->tag_set_data(th_disp, &new_vertex, 1, &disp(0));
    auto t_normal = getFTensor1Normal();
    if (SPACE_DIM == 2) //FIXME: make normal outward
      t_normal(i) *= -1;
    CHKERR moabDebug->tag_set_data(th_normal, &new_vertex, 1, &t_normal(0));
    ++t_traction;
    ++t_coords;
    ++t_disp;
  }
  // }

  MoFEMFunctionReturn(0);
}

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm, boost::shared_ptr<PostProcEle> &post_proc_fe,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), postProcFe(post_proc_fe), uXScatter(ux_scatter),
        uYScatter(uy_scatter), uZScatter(uz_scatter){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    auto make_vtk = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_contact_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    { // for debugging

      std::ostringstream ostrm;
      ostrm << "out_debug_" << ts_step << ".vtk";
      CHKERR DMoFEMLoopFiniteElements(dM, "bFE", debug_post_proc);
      CHKERR moab_debug.write_file(ostrm.str().c_str(), "VTK", "");
      mb_post_debug.delete_mesh();
    }

    auto print_max_min = [&](auto &tuple, const std::string msg) {
      MoFEMFunctionBegin;
      CHKERR VecScatterBegin(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                             INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecScatterEnd(std::get<1>(tuple), ts_u, std::get<0>(tuple),
                           INSERT_VALUES, SCATTER_FORWARD);
      double max, min;
      CHKERR VecMax(std::get<0>(tuple), PETSC_NULL, &max);
      CHKERR VecMin(std::get<0>(tuple), PETSC_NULL, &min);
      PetscPrintf(PETSC_COMM_WORLD, "%s time %3.4e min %3.4e max %3.4e\n",
                  msg.c_str(), ts_t, min, max);
      MoFEMFunctionReturn(0);
    };

    CHKERR make_vtk();
    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");

    cout << " integral_1_lhs: " << integral_1_lhs
         << " integral_1_rhs: " << integral_1_rhs << endl;
    cout << " integral_2_lhs: " << integral_2_lhs
         << " integral_2_rhs: " << integral_2_rhs << endl;
    cout << " integral_3_lhs: " << integral_3_lhs
         << " integral_3_rhs: " << integral_3_rhs << endl;

    integral_1_lhs = integral_1_rhs = integral_2_lhs = integral_2_rhs =
        integral_3_lhs = integral_3_rhs = 0;

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProcFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;
};

template <int DIM> struct OpPostProcContact : public DomainEleOp {
  OpPostProcContact(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
};
//! [Operators definitions]
template <int DIM>
OpPostProcContact<DIM>::OpPostProcContact(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[boundary_ent], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
template <int DIM>
MoFEMErrorCode OpPostProcContact<DIM>::doWork(int side, EntityType type,
                                              EntData &data) {
  MoFEMFunctionBegin;

  auto get_tag_mat = [&](const std::string name) {
    std::array<double, 9> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), 9, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  auto get_tag_vec = [&](const std::string name) {
    std::array<double, 3> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), 3, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);

  auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != DIM; ++r)
      for (size_t c = 0; c != DIM; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_vector = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != DIM; ++r)
      mat(0, r) = t(r);
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_stress = get_tag_mat("SIGMA");
  auto th_div = get_tag_vec("DIV_SIGMA");

  size_t nb_gauss_pts = getGaussPts().size2();
  auto t_stress =
      getFTensor2FromMat<DIM, DIM>(*(commonDataPtr->contactStressPtr));
  auto t_div =
      getFTensor1FromMat<DIM>(*(commonDataPtr->contactStressDivergencePtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    CHKERR set_tag(th_stress, gg, set_matrix(t_stress));
    CHKERR set_tag(th_div, gg, set_vector(t_div));
    ++t_stress;
    ++t_div;
  }

  MoFEMFunctionReturn(0);
}
//! [Postprocessing]

}; // namespace OpContactTools
