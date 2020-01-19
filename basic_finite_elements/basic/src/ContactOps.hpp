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
struct CommonData : public OpElasticTools::CommonData {
  boost::shared_ptr<MatrixDouble> contactStressPtr;
  boost::shared_ptr<MatrixDouble> contactStressDivergencePtr;
  boost::shared_ptr<MatrixDouble> contactTractionPtr;
  boost::shared_ptr<MatrixDouble> contactDispPtr;
  FTensor::Ddg<double, 2, 2> tC;
};
//! [Common data]

FTensor::Index<'i', 2> i;
FTensor::Index<'j', 2> j;
FTensor::Index<'k', 2> k;
FTensor::Index<'l', 2> l;

struct OpInternalBoundaryContactRhs : public BoundaryEleOp {
  OpInternalBoundaryContactRhs(const std::string field_name,
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

struct OpConstrainBoundaryTraction : public BoundaryEleOp {
  OpConstrainBoundaryTraction(const std::string field_name,
                              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpInternalBoundaryContactLhs : public BoundaryEleOp {
  OpInternalBoundaryContactLhs(const std::string row_field_name,
                               const std::string col_field_name);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  MatrixDouble locMat;
};

struct OpConstrainDomainRhs : public DomainEleOp {
  OpConstrainDomainRhs(const std::string field_name,
                       boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainDomainLhs_dSigma : public DomainEleOp {
  OpConstrainDomainLhs_dSigma(const std::string row_field_name,
                              const std::string col_field_name,
                              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpConstrainDomainLhs_dU : public DomainEleOp {
  OpConstrainDomainLhs_dU(const std::string row_field_name,
                          const std::string col_field_name);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data);

private:
  MatrixDouble locMat;
};

template <typename T>
inline double gap(FTensor::Tensor1<T, 2> &t_disp,
                  FTensor::Tensor1<double, 2> &t_normal) {
  return t_disp(i) * t_normal(i);
}

template <typename T>
inline double normal_traction(FTensor::Tensor1<T, 2> &t_traction,
                              FTensor::Tensor1<double, 2> &t_normal) {
  return t_traction(i) * t_normal(i);
}

inline auto diff_gap(FTensor::Tensor1<double, 2> &t_normal) {
  return FTensor::Tensor1<double, 2>{t_normal(0), t_normal(1)};
}

inline auto diff_traction(FTensor::Tensor1<double, 2> &t_normal) {
  return FTensor::Tensor1<double, 2>{t_normal(0), t_normal(1)};
}

inline double constrian(double &&gap, double &&normal_traction) {
  if ((cn * gap + normal_traction) < 0)
    return -cn * gap;
  else
    return normal_traction;
};

inline double sign(double x) {
  if (x == 0)
    return 0;
  else if (x > 0)
    return 1;
  else
    return -1;
};

inline double diff_constrains_dgap(double &&gap, double &&normal_traction) {
  return (cn * (-1 + sign(cn * gap + normal_traction))) / 2.;
}

inline double diff_constrains_dtraction(double &&gap,
                                        double &&normal_traction) {
  return (1 + sign(cn * gap + normal_traction)) / 2.;
}

auto diff_constrains_du(double &&diff_constrains_dgap,
                        FTensor::Tensor1<double, 2> &&t_diff_gap) {
  FTensor::Tensor1<double, 2> t_diff;
  t_diff(i) = diff_constrains_dgap * t_diff_gap(i);
  return t_diff;
}

auto diff_constrains_dstress(
    double &&diff_constrains_dtraction,
    FTensor::Tensor1<double, 2> &&t_diff_normal_traction) {
  FTensor::Tensor1<double, 2> t_diff;
  t_diff(i) = diff_constrains_dtraction * t_diff_normal_traction(i);
  return t_diff;
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

    FTensor::Tensor1<double, 2> t_direction{getDirection()[0],
                                            getDirection()[1]};
    FTensor::Tensor1<double, 2> t_normal{-t_direction(1), t_direction(0)};
    const double l = sqrt(t_normal(i) * t_normal(i));
    t_normal(i) /= l;
    t_direction(i) /= l;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_traction =
        getFTensor1FromMat<2>(*(commonDataPtr->contactTractionPtr));

    size_t nb_base_functions = data.getN().size2();
    auto t_base = data.getFTensor0N();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf{&nf[0], &nf[1]};

      const double alpha = t_w * l;

      size_t bb = 0;
      for (; bb != nb_dofs / 2; ++bb) {
        t_nf(i) += alpha * t_base * t_traction(i);
        ++t_nf;
        ++t_base;
      }
      for (; bb != nb_base_functions; ++bb)
        ++t_base;

      ++t_traction;
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

    FTensor::Tensor1<double, 2> t_direction{getDirection()[0],
                                            getDirection()[1]};
    FTensor::Tensor1<double, 2> t_normal{-t_direction(1), t_direction(0)};
    const double l = sqrt(t_normal(i) * t_normal(i));
    t_normal(i) /= l;
    t_direction(i) /= l;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_disp = getFTensor1FromMat<2>(*(commonDataPtr->contactDispPtr));
    auto t_traction =
        getFTensor1FromMat<2>(*(commonDataPtr->contactTractionPtr));

    size_t nb_base_functions = data.getN().size2() / 3;
    auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf{&nf[0], &nf[1]};

      const double alpha = t_w * l;

      FTensor::Tensor2<double, 2, 2> t_tangent_tensor;
      t_tangent_tensor(i, j) = t_direction(i) * t_direction(j);

      FTensor::Tensor1<double, 2> t_rhs;
      t_rhs(i) = t_tangent_tensor(i, j) * t_disp(j);

      size_t bb = 0;
      for (; bb != nb_dofs / 2; ++bb) {
        const double beta = alpha * (t_base(i) * t_normal(i));

        t_nf(i) += beta * t_rhs(i);

        ++t_nf;
        ++t_base;
      }
      for (; bb != nb_base_functions; ++bb)
        ++t_base;

      ++t_disp;
      ++t_traction;
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
  doEntities[MBEDGE] = true;
}

MoFEMErrorCode OpConstrainBoundaryTraction::doWork(int side, EntityType type,
                                                   EntData &data) {

  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();

  if (side == 0 && type == MBEDGE) {
    commonDataPtr->contactTractionPtr->resize(2, nb_gauss_pts);
    commonDataPtr->contactTractionPtr->clear();
  }

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto t_direction = getFTensor1Direction();
    FTensor::Tensor1<double, 2> t_normal{-t_direction(1), t_direction(0)};
    const double l = sqrt(t_normal(i) * t_normal(i));
    t_normal(i) /= l;

    auto t_traction =
        getFTensor1FromMat<2>(*(commonDataPtr->contactTractionPtr));

    size_t nb_base_functions = data.getN().size2() / 3;
    auto t_base = data.getFTensor1N<3>();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_field_data = data.getFTensor1FieldData<2>();

      size_t bb = 0;
      for (; bb != nb_dofs / 2; ++bb) {
        t_traction(j) += (t_base(i) * t_normal(i)) * t_field_data(j);
        ++t_field_data;
        ++t_base;
      }

      // cerr << "t_traction " << t_traction << endl;

      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_traction;
    }
  }

  MoFEMFunctionReturn(0);
}

OpInternalBoundaryContactLhs::OpInternalBoundaryContactLhs(
    const std::string row_field_name, const std::string col_field_name)
    : BoundaryEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL) {
  sYmm = false;
}

MoFEMErrorCode OpInternalBoundaryContactLhs::doWork(int row_side, int col_side,
                                                    EntityType row_type,
                                                    EntityType col_type,
                                                    EntData &row_data,
                                                    EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    auto t_w = getFTensor0IntegrationWeight();

    locMat.resize(row_nb_dofs, col_nb_dofs, false);
    locMat.clear();

    FTensor::Tensor1<double, 2> t_direction{getDirection()[0],
                                            getDirection()[1]};
    FTensor::Tensor1<double, 2> t_normal{-t_direction(1), t_direction(0)};
    const double l = sqrt(t_normal(i) * t_normal(i));
    t_normal(i) /= l;

    const size_t nb_gauss_pts = getGaussPts().size2();
    const size_t nb_base_functions = row_data.getN().size2();
    auto t_row_base = row_data.getFTensor0N();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = l * t_w;

      size_t rr = 0;
      for (; rr != row_nb_dofs / 2; ++rr) {

        FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_mat{
            &locMat(2 * rr + 0, 0), &locMat(2 * rr + 1, 1)};

        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);

        for (size_t cc = 0; cc != col_nb_dofs / 2; ++cc) {
          const double col_base = t_col_base(i) * t_normal(i);
          t_mat(i) += alpha * t_row_base * col_base;
          ++t_col_base;
          ++t_mat;
        }

        ++t_row_base;
      }
      for (; rr < nb_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
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

  // const size_t nb_gauss_pts = getGaussPts().size2();
  // const size_t row_nb_dofs = row_data.getIndices().size();
  // const size_t col_nb_dofs = col_data.getIndices().size();

  // if (row_nb_dofs && col_nb_dofs) {

  //   locMat.resize(row_nb_dofs, col_nb_dofs, false);
  //   locMat.clear();

  //   FTensor::Tensor1<double, 2> t_direction{getDirection()[0],
  //                                           getDirection()[1]};
  //   FTensor::Tensor1<double, 2> t_normal{-t_direction(1), t_direction(0)};
  //   const double l = sqrt(t_normal(i) * t_normal(i));
  //   t_normal(i) /= l;
  //   t_direction(i) /= l;

  //   auto t_disp = getFTensor1FromMat<2>(*(commonDataPtr->contactDispPtr));
  //   auto t_traction =
  //       getFTensor1FromMat<2>(*(commonDataPtr->contactTractionPtr));

  //   auto t_w = getFTensor0IntegrationWeight();
  //   auto t_row_base = row_data.getFTensor1N<3>();
  //   size_t nb_face_functions = row_data.getN().size2() / 3;

  //   locMat.resize(row_nb_dofs, col_nb_dofs, false);
  //   locMat.clear();

  //   for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

  //     const double alpha = t_w * l;

  //     FTensor::Tensor2<double, 2, 2> t_tangent_tensor;
  //     t_tangent_tensor(i, j) = t_direction(i) * t_direction(j);

  //     // auto t_diff_du = diff_constrains_du(
  //     //     diff_constrains_dgap(gap(t_disp, t_normal),
  //     //                          normal_traction(t_traction, t_normal)),
  //     //     diff_gap(t_normal));
  //     // FTensor::Tensor2<double, 2, 2> t_diff_rhs;
  //     // t_diff_rhs(i, j) = t_normal(i) * t_diff_du(j);

  //     size_t rr = 0;
  //     for (; rr != row_nb_dofs / 2; ++rr) {

  //       FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 2, 2> t_mat(
  //           &locMat(2 * rr + 0, 0), &locMat(2 * rr + 0, 1),
  //           &locMat(2 * rr + 1, 0), &locMat(2 * rr + 1, 1));
  //       const double row_base = t_row_base(i) * t_normal(i);

  //       auto t_col_base = col_data.getFTensor0N(gg, 0);
  //       for (size_t cc = 0; cc != col_nb_dofs / 2; ++cc) {
  //         const double beta = alpha * row_base * t_col_base;

  //         t_mat(i, j) += beta * t_diff_rhs(i, j);

  //         ++t_col_base;
  //         ++t_mat;
  //       }

  //       ++t_row_base;
  //     }
  //     for (; rr < nb_face_functions; ++rr)
  //       ++t_row_base;

  //     ++t_disp;
  //     ++t_traction;
  //     ++t_w;
  //   }

  //   CHKERR MatSetValues(getSNESB(), row_data, col_data,
  //   &*locMat.data().begin(),
  //                       ADD_VALUES);
  // }

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

    FTensor::Tensor1<double, 2> t_direction{getDirection()[0],
                                            getDirection()[1]};
    FTensor::Tensor1<double, 2> t_normal{-t_direction(1), t_direction(0)};
    const double l = sqrt(t_normal(i) * t_normal(i));
    t_normal(i) /= l;
    t_direction(i) /= l;

    auto t_disp = getFTensor1FromMat<2>(*(commonDataPtr->contactDispPtr));
    auto t_traction =
        getFTensor1FromMat<2>(*(commonDataPtr->contactTractionPtr));

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor1N<3>();
    size_t nb_face_functions = row_data.getN().size2() / 3;

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = t_w * l;

      auto t_diff_dstress = diff_constrains_dstress(
          diff_constrains_dtraction(gap(t_disp, t_normal),
                                    normal_traction(t_traction, t_normal)),
          diff_traction(t_normal));

      FTensor::Tensor2<double, 2, 2> t_diff_rhs;
      t_diff_rhs(i, j) =
          t_normal(i) * t_diff_dstress(j) + t_direction(i) * t_direction(j);

      size_t rr = 0;
      for (; rr != row_nb_dofs / 2; ++rr) {
        FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 2, 2> t_mat(
            &locMat(2 * rr + 0, 0), &locMat(2 * rr + 0, 1),
            &locMat(2 * rr + 1, 0), &locMat(2 * rr + 1, 1));

        const double row_base = t_row_base(i) * t_normal(i);

        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);
        for (size_t cc = 0; cc != col_nb_dofs / 2; ++cc) {
          const double col_base = t_col_base(i) * t_normal(i);
          const double beta = alpha * row_base * col_base;

          t_mat(i, j) += beta * t_diff_rhs(i, j);

          ++t_col_base;
          ++t_mat;
        }

        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_disp;
      ++t_traction;
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
    auto t_diff_base = data.getFTensor2DiffN<3, 2>();
    auto t_stress =
        getFTensor2FromMat<2, 2>(*(commonDataPtr->contactStressPtr));
    auto t_disp = getFTensor1FromMat<2>(*(commonDataPtr->contactDispPtr));
    auto t_grad = getFTensor2FromMat<2, 2>((*commonDataPtr->mGradPtr));
    auto &t_C = commonDataPtr->tC;

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = getMeasure() * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf{&nf[0], &nf[1]};

      FTensor::Tensor2_symmetric<double, 2> t_epsilon;
      t_epsilon(i, j) = t_C(i, j, k, l) * t_stress(k, l);

      FTensor::Tensor2<double, 2, 2> t_omega;
      t_omega(i, j) = (t_grad(i, j) - t_grad(j, i)) / 2;

      size_t bb = 0;
      for (; bb != nb_dofs / 2; ++bb) {
        const double t_div_base = t_diff_base(0, 0) + t_diff_base(1, 1);

        t_nf(i) +=
            alpha * (t_base(j) * t_epsilon(i, j) + t_div_base * t_disp(i));
        t_nf(i) += alpha * (t_base(j) * t_omega(i, j));

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

OpConstrainDomainLhs_dSigma::OpConstrainDomainLhs_dSigma(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}
MoFEMErrorCode OpConstrainDomainLhs_dSigma::doWork(int row_side, int col_side,
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

    auto &t_C = commonDataPtr->tC;

    size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_row_base = row_data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != row_nb_dofs / 2; ++rr) {

        FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 2, 2> t_mat{
            &locMat(2 * rr + 0, 0), &locMat(2 * rr + 0, 1),
            &locMat(2 * rr + 1, 0), &locMat(2 * rr + 1, 1)};

        auto t_col_base = col_data.getFTensor1N<3>(gg, 0);

        for (size_t cc = 0; cc != col_nb_dofs / 2; ++cc) {

          t_mat(i, j) +=
              alpha * t_C(i, k, j, l) * (t_row_base(k) * t_col_base(l));

          ++t_col_base;
          ++t_mat;
        }

        ++t_row_base;
      }
      for (; rr < nb_base_functions; ++rr)
        ++t_row_base;

      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*locMat.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpConstrainDomainLhs_dU::OpConstrainDomainLhs_dU(
    const std::string row_field_name, const std::string col_field_name)
    : DomainEleOp(row_field_name, col_field_name, DomainEleOp::OPROWCOL) {
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

    size_t nb_base_functions = row_data.getN().size2() / 3;
    auto t_row_base = row_data.getFTensor1N<3>();
    auto t_row_diff_base = row_data.getFTensor2DiffN<3, 2>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      const double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != row_nb_dofs / 2; ++rr) {

        FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_mat{
            &locMat(2 * rr + 0, 0), &locMat(2 * rr + 1, 1)};

        FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 2, 2> t_mat_asymmetric{
            &locMat(2 * rr + 0, 0), &locMat(2 * rr + 0, 1),
            &locMat(2 * rr + 1, 0), &locMat(2 * rr + 1, 1)};

        const double t_row_div_base =
            t_row_diff_base(0, 0) + t_row_diff_base(1, 1);

        auto t_col_base = col_data.getFTensor0N(gg, 0);
        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

        for (size_t cc = 0; cc != col_nb_dofs / 2; ++cc) {

          t_mat(i) += alpha * t_row_div_base * t_col_base;

          const double omega_u = t_col_diff_base(1) / 2;
          const double omega_v = -t_col_diff_base(0) / 2;

          t_mat_asymmetric(0, 0) += alpha * t_row_base(1) * omega_u;
          t_mat_asymmetric(0, 1) += alpha * t_row_base(1) * omega_v;
          t_mat_asymmetric(1, 0) -= alpha * t_row_base(0) * omega_u;
          t_mat_asymmetric(1, 1) -= alpha * t_row_base(0) * omega_v;

          ++t_col_base;
          ++t_col_diff_base;
          ++t_mat;
          ++t_mat_asymmetric;
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
  }

  MoFEMFunctionReturn(0);
}

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcFaceOnRefinedMeshFor2D> &post_proc_fe,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter)
      : dM(dm), postProcFe(post_proc_fe), uXScatter(ux_scatter),
        uYScatter(uy_scatter){};

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

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcFaceOnRefinedMeshFor2D> postProcFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
};

struct OpPostProcContact : public DomainEleOp {
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

OpPostProcContact::OpPostProcContact(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode OpPostProcContact::doWork(int side, EntityType type,
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
    for (size_t r = 0; r != 2; ++r)
      for (size_t c = 0; c != 2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_vector = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 2; ++r)
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
  auto t_stress = getFTensor2FromMat<2, 2>(*(commonDataPtr->contactStressPtr));
  auto t_div =
      getFTensor1FromMat<2>(*(commonDataPtr->contactStressDivergencePtr));

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
