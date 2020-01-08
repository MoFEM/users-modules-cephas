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
  boost::shared_ptr<VectorDouble> contactStressDivergencePtr;
  boost::shared_ptr<MatrixDouble> contactTractionPtr;
  boost::shared_ptr<MatrixDouble> contactDispPtr;
};
//! [Common data]

FTensor::Index<'i', 2> i;
FTensor::Index<'j', 2> j;

struct OpInternalContactRhs : public DomianEleOp {
  OpInternalContactRhs(const std::string field_name,
                       boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainRhs : public BoundaryEleOp {
  OpConstrainRhs(const std::string field_name,
                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainLhs_dU : public BoundaryEleOp {
  OpConstrainLhs_dU(const std::string row_field_name,
                       const std::string col_field_name,
                       boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);
private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpConstrainLhs_dSigma : public BoundaryEleOp {
  OpConstrainLhs_dSigma(const std::string row_field_name,
                       const std::string col_field_name,
                       boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);
private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locMat;
};

struct OpConstrainTraction : public BoundaryEleOp {
  OpConstrainTraction(const std::string field_name,
                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpConstrainDisp : public BoundaryEleOp {
  OpConstrainDisp(const std::string field_name,
                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpInternalContactLhs : public DomianEleOp {
  OpInternalContactLhs(const std::string row_field_name,
                       const std::string col_field_name,
                       boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
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

inline double constrian(double &&gap, double &&normal_traction) {
  if ((cn * gap + normal_traction) < 0)
    return -cn * gap;
  else
    return normal_traction;
};

inline auto diff_gap(FTensor::Tensor1<double, 2> &t_normal) {
  return FTensor::Tensor1<double, 2>{t_normal(0), t_normal(1)};
}

inline auto diff_traction(FTensor::Tensor1<double, 2> &t_normal) {
  return FTensor::Tensor1<double, 2>{t_normal(0), t_normal(1)};
}

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

inline double diff_constrains_dtracion_normal(double &&gap,
                                              double &&normal_traction) {
  return (1 + sign(cn * gap + normal_traction)) / 2.;
}

auto diff_constrains_dgap(double &&diff_constrains_dgap,
                          FTensor::Tensor1<double, 2> &&t_diff_gap) {
  FTensor::Tensor1<double, 2> t_diff;
  t_diff(i) = diff_constrains_dgap * t_diff_gap(i);
  return t_diff;
}

auto diff_constrains_dstress(
    double &&diff_constrains_dtracion_normal,
    FTensor::Tensor1<double, 2> &&t_diff_normal_traction) {
  FTensor::Tensor1<double, 2> t_diff;
  t_diff(i) = diff_constrains_dtracion_normal * t_diff_normal_traction(i);
  return t_diff;
}

OpInternalContactRhs::OpInternalContactRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}
MoFEMErrorCode
OpInternalContactRhs::doWork(int side, EntityType type,
                             DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();
    auto t_div =
        getFTensor0FromVec(*(commonDataPtr->contactStressDivergencePtr));

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      double alpha = getMeasure() * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf{&nf[0], &nf[1]};

      for (size_t bb = 0; bb != nb_dofs / 2; ++bb) {
        t_nf(i) += alpha * t_base * t_div;
        ++t_nf;
        ++t_base;
      };
      ++t_div;
      ++t_w;
    }
  }

  MoFEMFunctionReturn(0);
}

OpConstrainRhs::OpConstrainRhs(const std::string field_name,
                               boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpConstrainRhs::doWork(int side, EntityType type,
                                      DataForcesAndSourcesCore::EntData &data) {

  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_direction = getFTensor1Direction();
    FTensor::Tensor1<double, 2> t_normal{t_direction(0), t_direction(1)};
    const double l = sqrt(t_normal(i) * t_normal(i));
    t_normal(i) /= l;

    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor1N<3>();
    auto t_disp = getFTensor1FromMat<2>(*(commonDataPtr->contactDispPtr));
    auto t_traction =
        getFTensor1FromMat<2>(*(commonDataPtr->contactTractionPtr));

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf{&nf[0], &nf[1]};

      const double alpha = t_w * l;

      FTensor::Tensor1<double, 2> t_constrain;
      t_constrain(i) =
          t_normal(i) * constrian(gap(t_disp, t_normal),
                                  normal_traction(t_traction, t_normal));

      for (size_t bb = 0; bb != nb_dofs / 2; ++bb) {

        t_nf(j) += (alpha * (t_base(i) * t_normal(i))) * t_constrain(j);

        ++t_nf;
        ++t_base;
      };
      ++t_disp;
      ++t_traction;
      ++t_w;
    }
  }

  MoFEMFunctionReturn(0);
}

OpConstrainTraction::OpConstrainTraction(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  doEntities[MBEDGE] = true;
}

MoFEMErrorCode OpConstrainTraction::doWork(int side, EntityType type,
                                      DataForcesAndSourcesCore::EntData &data) {

  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    commonDataPtr->contactTractionPtr->resize(2, nb_gauss_pts);
    commonDataPtr->contactTractionPtr->clear();

    auto t_direction = getFTensor1Direction();
    FTensor::Tensor1<double, 2> t_normal{t_direction(0), t_direction(1)};
    const double l = sqrt(t_normal(i) * t_normal(i));
    t_normal(i) /= l;

    auto t_traction =
        getFTensor1FromMat<2>(*(commonDataPtr->contactTractionPtr));

    auto t_base = data.getFTensor1N<3>();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_field_data = data.getFTensor1FieldData<2>();
      for (size_t bb = 0; bb != nb_dofs / 2; ++bb) {
        t_traction(j) += (t_base(i) * t_normal(i)) * t_field_data(j);
        ++t_field_data;
        ++t_base;
      };
      ++t_traction;
    }
  }

  MoFEMFunctionReturn(0);
}

OpConstrainDisp::OpConstrainDisp(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : BoundaryEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBTRI], &doEntities[MBMAXTYPE], false);
}

MoFEMErrorCode OpConstrainDisp::doWork(int side, EntityType type,
                                      DataForcesAndSourcesCore::EntData &data) {

  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    commonDataPtr->contactDispPtr->resize(2, nb_gauss_pts);
    commonDataPtr->contactDispPtr->clear();

    auto t_disp = getFTensor1FromMat<2>(*(commonDataPtr->contactDispPtr));

    auto t_base = data.getFTensor0N();
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      auto t_field_data = data.getFTensor1FieldData<2>();
      for (size_t bb = 0; bb != nb_dofs / 2; ++bb) {
        t_disp(j) += t_base * t_field_data(j);
        ++t_field_data;
        ++t_base;
      };
      ++t_disp;
    }
  }

  MoFEMFunctionReturn(0);
}

OpInternalContactLhs::OpInternalContactLhs(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode
OpInternalContactLhs::doWork(int row_side, int col_side, EntityType row_type,
                             EntityType col_type,
                             DataForcesAndSourcesCore::EntData &row_data,
                             DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if(row_nb_dofs && col_nb_dofs) {

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    size_t nb_face_functions = row_data.getN().size2();

    locMat.resize(row_nb_dofs, col_nb_dofs, false);

    for(size_t gg = 0; gg!=nb_gauss_pts; ++gg) {

      const double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != row_nb_dofs / 2; ++rr) {

        FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 2> t_mat{
            &locMat(2 * rr + 0, 0), &locMat(2 * rr + 1, 0)};

        auto t_col_diff_base = col_data.getFTensor2DiffN<3, 2>();
        for (size_t cc = 0; cc != col_nb_dofs / 2; ++cc) {
          const double div = t_col_diff_base(j, j);
          t_mat(i) += alpha * t_row_base * div;
          ++t_mat;
          ++t_col_diff_base;
        }

        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base; 
      
      ++t_w;
    }

  }

  MoFEMFunctionReturn(0);
}

OpConstrainLhs_dU::OpConstrainLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}
MoFEMErrorCode
OpConstrainLhs_dU::doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  const size_t row_nb_dofs = row_data.getIndices().size();
  const size_t col_nb_dofs = col_data.getIndices().size();

  if (row_nb_dofs && col_nb_dofs) {

    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();
    size_t nb_face_functions = row_data.getN().size2();

    locMat.resize(row_nb_dofs, col_nb_dofs, false);

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      size_t rr = 0;
      for (; rr != row_nb_dofs / 2; ++rr) {

        for (size_t cc = 0; cc != col_nb_dofs / 2; ++cc) {
        }

        ++t_row_base;
      }
      for (; rr < nb_face_functions; ++rr)
        ++t_row_base;

      ++t_w;
    }
  }

  MoFEMFunctionReturn(0);
}

OpConstrainLhs_dSigma::OpConstrainLhs_dSigma(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}
MoFEMErrorCode
OpConstrainLhs_dSigma::doWork(int row_side, int col_side, EntityType row_type,
                              EntityType col_type,
                              DataForcesAndSourcesCore::EntData &row_data,
                              DataForcesAndSourcesCore::EntData &col_data) {}

}; // namespace OpContactTools