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

//! [Common data]
struct CommonData {
  FTensor::Ddg<double, 2, 2> tD;
  boost::shared_ptr<MatrixDouble> mGradPtr;
  boost::shared_ptr<MatrixDouble> mStrainPtr;
  boost::shared_ptr<MatrixDouble> mStressPtr;
};
//! [Common data]

//! [Operators definitions]
typedef boost::function<FTensor::Tensor1<double, 2>(const double, const double)>
    VectorFun;

struct OpStrain : public DomianEleOp {
  OpStrain(const std::string field_name,
           boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpStress : public DomianEleOp {
  OpStress(const std::string field_name,
           boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpInternalForceRhs : public DomianEleOp {
  OpInternalForceRhs(const std::string field_name,
                  boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpForceRhs : public DomianEleOp {
  OpForceRhs(const std::string field_name,
              boost::shared_ptr<CommonData> common_data_ptr,
              VectorFun body_force);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  VectorFun funForce;
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpStiffnessMatrixLhs : public DomianEleOp {
  OpStiffnessMatrixLhs(const std::string row_field_name,
                    const std::string col_field_name,
                    boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  MatrixDouble locK;
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPostProcElastic : public DomianEleOp {
  OpPostProcElastic(const std::string field_name,
                    moab::Interface &post_proc_mesh,
                    std::vector<EntityHandle> &map_gauss_pts,
                    boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);
private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
};
//! [Operators definitions]


OpStrain::OpStrain(const std::string field_name,
                   boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Calculate strain]
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
    t_strain(i, j) = (t_grad(i, j) || t_grad(j, i)) / 2;
    ++t_grad;
    ++t_strain;
  }

  MoFEMFunctionReturn(0);
}
//! [Calculate strain]


OpStress::OpStress(const std::string field_name,
                   boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Calculate stress]
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
//! [Calculate stress]

OpInternalForceRhs::OpInternalForceRhs(const std::string field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

//! [Internal force]
MoFEMErrorCode
OpInternalForceRhs::doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;

  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const size_t nb_base_functions = data.getN().size2();
    if (2 * nb_base_functions < nb_dofs)
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
      for (; bb != nb_dofs / 2; ++bb) {
        t_nf(i) += alpha * t_diff_base(j) * t_stress(i, j);
        ++t_diff_base;
        ++t_nf;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_diff_base;

      ++t_stress;
      ++t_w;
    }

    CHKERR VecSetValues(getKSPf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
//! [Internal force]

OpForceRhs::OpForceRhs(const std::string field_name,
                         boost::shared_ptr<CommonData> common_data_ptr,
                         VectorFun body_force)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr), funForce(body_force) {}

//! [Body force]
MoFEMErrorCode OpForceRhs::doWork(int side, EntityType type,
                                   DataForcesAndSourcesCore::EntData &data) {
  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;

  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    const size_t nb_base_functions = data.getN().size2();
    if (2 * nb_base_functions < nb_dofs)
      SETERRQ(
          PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
          "Number of DOFs is larger than number of base functions on entity");

    const size_t nb_gauss_pts = data.getN().size1();
    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();
    auto t_coords = getFTensor1Coords();

    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

      double alpha = getMeasure() * t_w;
      auto t_force = funForce(t_coords(0), t_coords(1));

      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf{&nf[0], &nf[1]};
      size_t bb = 0;
      for (; bb != nb_dofs / 2; ++bb) {
        t_nf(i) += alpha * t_base * t_force(i);
        ++t_base;
        ++t_nf;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_base;

      ++t_w;
      ++t_coords;
    }

    if ((getDataCtx() & PetscData::CtxSetTime).any())
      for (int dd = 0; dd != nb_dofs; ++dd)
        nf[dd] *= getTStime();

    CHKERR VecSetValues(getKSPf(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
//! [Body force]

OpStiffnessMatrixLhs::OpStiffnessMatrixLhs(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

//! [Stiffness]
MoFEMErrorCode
OpStiffnessMatrixLhs::doWork(int row_side, int col_side, EntityType row_type,
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

    locK.clear();
    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      size_t rr = 0;
      for (; rr != nb_row_dofs / 2; ++rr) {

        FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 2, 2> t_a{

            &locK(2 * rr + 0, 0), &locK(2 * rr + 0, 1),

            &locK(2 * rr + 1, 0), &locK(2 * rr + 1, 1)};
        auto t_col_diff_base = col_data.getFTensor1DiffN<2>(gg, 0);

        for (size_t cc = 0; cc != nb_col_dofs / 2; ++cc) {
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

    CHKERR MatSetValues(getKSPB(), row_data, col_data, &locK(0, 0),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
//! [Stiffness]

OpPostProcElastic::OpPostProcElastic(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode
OpPostProcElastic::doWork(int side, EntityType type,
                         DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  auto get_tag = [&](const std::string name) {
    std::array<double, 9> def;
    std::fill(def.begin(), def.end(), 0);
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), 9, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3,3);
  
  auto set_matrix = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 2; ++r)
      for (size_t c = 0; c != 2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_matrix_symm = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 2; ++r)
      for (size_t c = 0; c != 2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_plain_stress_strain = [&](auto &mat, auto &t) -> MatrixDouble3by3 & {
    mat(2, 2) = -poisson_ratio * (t(0, 0) + t(1, 1));
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_grad = get_tag("GRAD");
  auto th_strain = get_tag("STRAIN");
  auto th_stress = get_tag("STRESS");

  size_t nb_gauss_pts = data.getN().size1();
  auto t_grad = getFTensor2FromMat<2, 2>(*(commonDataPtr->mGradPtr));
  auto t_strain = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStrainPtr));
  auto t_stress = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStressPtr));

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
    CHKERR set_tag(th_grad, gg, set_matrix(t_grad));
    CHKERR set_tag(
        th_strain, gg,
        set_plain_stress_strain(set_matrix_symm(t_strain), t_stress));
    CHKERR set_tag(th_stress, gg, set_matrix_symm(t_stress));
    ++t_grad;
    ++t_strain;
    ++t_stress;
  }

  MoFEMFunctionReturn(0);
}
//! [Postprocessing]

}; // namespace OpElasticTools