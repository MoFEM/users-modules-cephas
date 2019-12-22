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

namespace OpElasticTools {

struct CommonData {
  FTensor::Ddg<double, 2, 2> tD;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;
};

typedef boost::function<FTensor::Tensor1<double, 2>(const double, const double)>
    VectorFun;

struct OpStrain : public DomianEleOp {
  OpStrain(const std::string field_name,
           boost::shared_ptr<CommonData> &common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpStress : public DomianEleOp {
  OpStress(const std::string field_name,
           boost::shared_ptr<CommonData> &common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpInternalForce : public DomianEleOp {
  OpInternalForce(const std::string field_name,
                  boost::shared_ptr<CommonData> &common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpBodyForce : public DomianEleOp {
  OpBodyForce(const std::string field_name,
              boost::shared_ptr<CommonData> &common_data_ptr,
              VectorFun body_force);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  VectorFun bodyForce;
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpStiffnessMatrix : public DomianEleOp {
  OpStiffnessMatrix(const std::string row_field_name,
                    const std::string col_field_name,
                    boost::shared_ptr<CommonData> &common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  MatrixDouble locK;
  boost::shared_ptr<CommonData> commonDataPtr;
};

OpStrain::OpStrain(const std::string field_name,
                   boost::shared_ptr<CommonData> &common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

MoFEMErrorCode OpStrain::doWork(int side, EntityType type,
                                DataForcesAndSourcesCore::EntData &data) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = commonDataPtr->mGradPtr->size2();
  commonDataPtr->mStrainPtr->resize(3, nb_gauss_pts);
  auto t_grad = getFTensor2FromMat<2, 2>(*(commonDataPtr->mGradPtr));
  auto t_strain = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStrainPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    t_strain(i, j) = 0.5 * (t_grad(i, j) || t_grad(j, i));
    ++t_grad;
    ++t_strain;
  }

  MoFEMFunctionReturn(0);
}

OpStress::OpStress(const std::string field_name,
                   boost::shared_ptr<CommonData> &common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

MoFEMErrorCode OpStress::doWork(int side, EntityType type,
                                DataForcesAndSourcesCore::EntData &data) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  MoFEMFunctionBegin;
  const size_t nb_gauss_pts = commonDataPtr->mStrainPtr->size2();
  commonDataPtr->mStressPtr->resize(3, nb_gauss_pts);
  auto &t_D = commonDataPtr->tD;
  auto t_strain = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStrainPtr));
  auto t_stress = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStressPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    t_stress(i, j) = t_D(i, j, k, l) * t_strain(k, l);
    ++t_strain;
    ++t_stress;
  }

  MoFEMFunctionReturn(0);
}

OpInternalForce::OpInternalForce(const std::string field_name,
                                 boost::shared_ptr<CommonData> &common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpInternalForce::doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;

  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const size_t nb_base_functions = data.getN().size2();
    if (3 * nb_base_functions < nb_dofs)
      SETERRQ(
          PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
          "Number of DOFs is larger than number of base functions on entity");

    const size_t nb_gauss_pts = data.getN().size1();
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_stress =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStressPtr));
    auto t_diff_base = data.getFTensor1DiffN<2>();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      double alpha = getMeasure() * t_w;
      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf{&nf[0], &nf[1]};

      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i) += alpha * t_diff_base(j) * t_stress(i, j);
        ++t_diff_base;
        ++t_nf;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_diff_base;

      ++t_stress;
      ++t_w;
    }

    CHKERR VecSetValues(getFEMethod()->ksp_f, data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpBodyForce::OpBodyForce(const std::string field_name,
                         boost::shared_ptr<CommonData> &common_data_ptr,
                         VectorFun body_force)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr), bodyForce(body_force) {}

MoFEMErrorCode
OpBodyForce::doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;

  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const size_t nb_base_functions = data.getN().size2();
    if (3 * nb_base_functions < nb_dofs)
      SETERRQ(
          PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
          "Number of DOFs is larger than number of base functions on entity");

    const size_t nb_gauss_pts = data.getN().size1();
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();
    auto t_coords = getFTensor1Coords();

    auto get_gravity = [](auto &t_coords) {
      FTensor::Tensor1<double, 2> t_gravity;
      t_gravity(0) = 0;
      t_gravity(1) = -1;
    };

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      double alpha = getMeasure() * t_w;
      auto t_gravity = bodyForce(t_coords(0), t_coords(1));

      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf{&nf[0], &nf[1]};
      size_t bb = 0;
      for (; bb != nb_dofs / 3; ++bb) {
        t_nf(i) += alpha * t_base * t_gravity(i);
        ++t_base;
        ++t_nf;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_w;
      ++t_coords;
    }

    CHKERR VecSetValues(getFEMethod()->ksp_f, data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpStiffnessMatrix::OpStiffnessMatrix(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> &common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

MoFEMErrorCode
OpStiffnessMatrix::doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();

  if (nb_row_dofs && nb_col_dofs) {

    locK.resize(nb_row_dofs, nb_col_dofs, false);

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_funcs = row_data.getN().size2();
    auto t_row_diff_base = row_data.getFTensor1DiffN<2>();
    auto t_w = getFTensor0IntegrationWeight();
    auto &t_D = commonDataPtr->tD;

        for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != nb_row_dofs / 3; ++rr) {

        FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 2, 2> t_a{
            &locK(2 * rr, 0), &locK(2 * rr, 1), &locK(2 * rr, 0),
            &locK(2 * rr, 1)};
        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

        for (size_t cc = 0; cc != nb_col_dofs / 3; ++cc) {
          t_a(i, k) += alpha * (t_D(i, j, k, l) *
                                (t_row_diff_base(j) * t_col_diff_base(l)));
          ++t_col_diff_base;
          ++t_a;
        }

        ++t_row_diff_base;
      }
      for (; rr != nb_row_base_funcs; ++rr)
        ++t_row_diff_base;

      ++t_w;
    }

    CHKERR MatSetValues(getFEMethod()->ksp_B, row_data, col_data, &locK(0, 0),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
}; // namespace OpElasticTools