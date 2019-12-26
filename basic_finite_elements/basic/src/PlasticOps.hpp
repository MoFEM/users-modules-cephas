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

namespace OpPlasticTools {

//! [Common data]
struct CommonData : public OpElasticTools::CommonData {
  boost::shared_ptr<VectorDouble> plasticSurfacePtr;
  boost::shared_ptr<MatrixDouble> plasticFlowPtr;
  boost::shared_ptr<VectorDouble> plasticTauPtr;
  boost::shared_ptr<MatrixDouble> plasticStrainPtr;
  boost::shared_ptr<MatrixDouble> plasticStrainDotPtr;

  double sigmaY;
};
//! [Common data]

//! [Operators definitions]
struct OpCalculatePlasticSurface : public DomianEleOp {
  OpCalculatePlasticSurface(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticFlowRhs : public DomianEleOp {
  OpCalculatePlasticFlowRhs(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateContrainsRhs : public DomianEleOp {
  OpCalculateContrainsRhs(const std::string field_name,
                          boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticInternalForceRhs : public DomianEleOp {
  OpCalculatePlasticInternalForceRhs(
      const std::string field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticInternalForceLhs_dEP : public DomianEleOp {
  OpCalculatePlasticInternalForceLhs_dEP(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  MatrixDouble locM;
};

struct OpCalculatePlasticFlowLhs_dU : public DomianEleOp {
  OpCalculatePlasticFlowLhs_dU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticFlowLhs_dEP : public DomianEleOp {
  OpCalculatePlasticFlowLhs_dEP(const std::string row_field_name,
                                const std::string col_field_name,
                                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticFlowLhs_dTAU : public DomianEleOp {
  OpCalculatePlasticFlowLhs_dTAU(const std::string row_field_name,
                                 const std::string col_field_name,
                                 boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateContrainsLhs_dU : public DomianEleOp {
  OpCalculateContrainsLhs_dU(const std::string row_field_name,
                             const std::string col_field_name,
                             boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateContrainsLhs_dEP : public DomianEleOp {
  OpCalculateContrainsLhs_dEP(const std::string row_field_name,
                              const std::string col_field_name,
                              boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateContrainsLhs_dTAU : public DomianEleOp {
  OpCalculateContrainsLhs_dTAU(const std::string row_field_name,
                               const std::string col_field_name,
                               boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPostProcPlastic : public DomianEleOp {
  OpPostProcPlastic(const std::string field_name,
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

OpCalculatePlasticSurface::OpCalculatePlasticSurface(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

MoFEMErrorCode
OpCalculatePlasticSurface::doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  FTensor::Index<'I', 3> I;
  FTensor::Index<'J', 3> J;
  FTensor::Index<'K', 3> K;
  FTensor::Index<'L', 3> L;

  constexpr double third = boost::math::constants::third<double>();

  const size_t nb_gauss_pts = commonDataPtr->mStressPtr->size2();
  auto t_stress = getFTensor2SymmetricFromMat<2>(*(commonDataPtr->mStressPtr));

  auto trace = [&](auto &t_stress) {
    return (t_stress(0, 0) + t_stress(1, 1)) * third;
  };

  auto deviator = [&](auto &t_stress, auto &&trace) {
    FTensor::Tensor2_symmetric<double, 3> t_dev;
    t_dev(i, j) = t_stress(i, j);
    t_dev(0, 0) -= trace;
    t_dev(1, 1) -= trace;
    t_dev(2, 2) -= trace;
    return t_dev;
  };

  auto platsic_surface = [&](auto &&t_dev) {
    return sqrt(t_dev(I, J) * t_dev(I, J));
  };

  auto diff_trace = [&]() {
    FTensor::Tensor2<double, 3, 3> t_diff_trace;
    t_diff_trace(I, J) = 0;
    t_diff_trace(0, 0) = third;
    t_diff_trace(1, 1) = third;
    t_diff_trace(2, 2) = 0;
    return t_diff_trace;
  };

  auto diff_stress = [&]() {
    FTensor::Tensor4<double, 3, 3, 3, 3> t_diff_stress;
    FTensor::Tensor2<double, 3, 3> t_one;
    t_one(I, J) = 0;
    t_one(0, 0) = 1;
    t_one(1, 1) = 1;
    t_one(2, 2) = 0;
    t_diff_stress(I, J, K, L) = t_one(I, K) * t_one(J, L);
    return t_diff_stress;
  };

  auto t_diff_trace = diff_trace();
  auto t_diff_stress = diff_stress();

  auto diff_dev_stress = [&] {
    FTensor::Tensor4<double, 3, 3, 3, 3> t_diff_dev_stress;

    t_diff_dev_stress(I, J, K, L) = t_diff_stress(I, J, K, L);

    t_diff_dev_stress(0, 0, 0, 0) -= third;
    t_diff_dev_stress(0, 0, 1, 1) -= third;

    t_diff_dev_stress(1, 1, 0, 0) -= third;
    t_diff_dev_stress(1, 1, 1, 1) -= third;

    t_diff_dev_stress(2, 2, 0, 0) -= third;
    t_diff_dev_stress(2, 2, 1, 1) -= third;

    return t_diff_dev_stress;
  };

  auto t_diff_dev_stress = diff_dev_stress();

  auto plastic_flow = [&](auto &f, auto &&t_dev_stress) {
    FTensor::Tensor2<double, 3, 3> t_diff_f;
    t_diff_f(K, L) =
        (1. / f) * (t_dev_stress(I, J) * t_diff_dev_stress(I, J, K, L));
    return t_diff_f;
  };

  // auto diff_plastic_flow = [&](auto &f) {
  //   FTensor::Tensor4<double, 3, 3, 3, 3> t_diff_flow;
  //   t_diff_flow(I, J, K, L) = -(1. / (f * f)) * t_diff_dev_stress(I, J, K,
  //   L);
  // };

  commonDataPtr->plasticSurfacePtr->resize(nb_gauss_pts, false);
  commonDataPtr->plasticFlowPtr->resize(9, nb_gauss_pts, false);
  auto t_flow = getFTensor2FromMat<3, 3>(*(commonDataPtr->plasticFlowPtr));

  for (auto &f : *(commonDataPtr->plasticSurfacePtr)) {
    f = platsic_surface(deviator(t_stress, trace(t_stress)));
    auto t_flow_tmp = plastic_flow(f, deviator(t_stress, trace(t_stress)));
    t_flow(I, J) = t_flow_tmp(I, J);
    ++t_flow;
    ++t_stress;
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowRhs::OpCalculatePlasticFlowRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpCalculatePlasticFlowRhs::doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 1>, 2> t_flow(
        &(*commonDataPtr->plasticFlowPtr)(0, 0),
        &(*commonDataPtr->plasticFlowPtr)(1, 0),
        &(*commonDataPtr->plasticFlowPtr)(4, 0));
    auto t_plastic_strain_dot =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticStrainDotPtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto &t_D = commonDataPtr->tD;

    const size_t nb_integration_pts = data.getN().size1();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 3>, 2> t_nf{
          &nf[0], &nf[1], &nf[2]};
      for (size_t bb = 0; bb != nb_dofs / 3; ++bb) {
        t_nf(i, j) += alpha * (t_D(i, j, k, l) * t_plastic_strain_dot(k, l) -
                               t_tau * t_flow(i, j));

        ++t_base;
        ++t_nf;
      }

      ++t_flow;
      ++t_plastic_strain_dot;
      ++t_tau;
    }
  }

  MoFEMFunctionReturn(0);
}

OpCalculateContrainsRhs::OpCalculateContrainsRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpCalculateContrainsRhs::doWork(int side, EntityType type,
                                DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto t_f = getFTensor0FromVec(*(commonDataPtr->plasticSurfacePtr));
    auto t_w = getFTensor0IntegrationWeight();

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    const double sigma_y = commonDataPtr->sigmaY;

    auto t_base = data.getFTensor0N();
    const size_t nb_integration_pts = data.getN().size1();
    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      for (size_t bb = 0; bb != nb_dofs; ++bb) {
        const double w = t_tau + (t_f - sigma_y);
        nf[bb] += alpha * t_base * (w + std::abs(w)) / 2.;
        ++t_base;
      }

      ++t_tau;
      ++t_f;
      ++t_w;
    }
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticInternalForceRhs::OpCalculatePlasticInternalForceRhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(field_name, DomianEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculatePlasticInternalForceRhs::doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {

    auto t_plastic_strain =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticStrainPtr));
    auto &t_D = commonDataPtr->tD;

    const size_t nb_integration_pts = data.getN().size1();
    const size_t nb_base_functions = data.getN().size2();
    auto t_diff_base = data.getFTensor1DiffN<2>();
    auto t_w = getFTensor0IntegrationWeight();

    std::array<double, MAX_DOFS_ON_ENTITY> nf;
    std::fill(&nf[0], &nf[nb_dofs], 0);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_nf{&nf[0], &nf[1]};

      size_t bb = 0;
      for (; bb != nb_dofs / 2; ++bb) {

        t_nf(i) -=
            alpha * t_diff_base(j) * (t_D(i, j, k, l) * t_plastic_strain(k, l));

        ++t_diff_base;
        ++t_nf;
      }
      for (; bb < nb_base_functions; ++bb)
        ++t_diff_base;

      ++t_w;
      ++t_plastic_strain;
    }

    CHKERR VecSetValues(getSnesF(), data, nf.data(), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticInternalForceLhs_dEP::OpCalculatePlasticInternalForceLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculatePlasticInternalForceLhs_dEP::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = col_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    locM.resize(nb_row_dofs, nb_row_dofs, false);
    locM.clear();

    const size_t nb_integration_pts = row_data.getN().size1();
    const size_t nb_row_base_functions = row_data.getN().size2();

    MatSetValues(getSnesB(), row_data, col_data, &*locM.data().begin(),
                 ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dU::OpCalculatePlasticFlowLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculatePlasticFlowLhs_dU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', 2> i;
  FTensor::Index<'j', 2> j;
  FTensor::Index<'k', 2> k;
  FTensor::Index<'l', 2> l;

  const size_t nb_row_dofs = row_data.getIndices().size();
  const size_t nb_col_dofs = row_data.getIndices().size();
  if (nb_row_dofs && nb_col_dofs) {

    FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 1>, 2> t_flow(
        &(*commonDataPtr->plasticFlowPtr)(0, 0),
        &(*commonDataPtr->plasticFlowPtr)(1, 0),
        &(*commonDataPtr->plasticFlowPtr)(4, 0));
    auto t_plastic_strain_dot =
        getFTensor2SymmetricFromMat<2>(*(commonDataPtr->plasticStrainDotPtr));
    auto t_tau = getFTensor0FromVec(*(commonDataPtr->plasticTauPtr));
    auto &t_D = commonDataPtr->tD;

    const size_t nb_integration_pts = row_data.getN().size1();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_row_base = row_data.getFTensor0N();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      // FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 3>, 2> t_nf{
      //     &nf[0], &nf[1], &nf[2]};
      // for (size_t rr = 0; rr != nb_row_dofs / 3; ++rr) {
      //   auto t_col_diff_base = col_data.getFTensor1<2>(gg, 0);

      //   for (size_t cc = 0; cc != nb_col_dofs / 2; cc++) {

      //     ++t_col_diff_basel
      //   }

      //   ++t_row_base;
      // }
    }
  }

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dEP::OpCalculatePlasticFlowLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculatePlasticFlowLhs_dEP::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowLhs_dTAU::OpCalculatePlasticFlowLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculatePlasticFlowLhs_dTAU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dU::OpCalculateContrainsLhs_dU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculateContrainsLhs_dU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dEP::OpCalculateContrainsLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculateContrainsLhs_dEP::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

OpCalculateContrainsLhs_dTAU::OpCalculateContrainsLhs_dTAU(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomianEleOp(row_field_name, col_field_name, DomianEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculateContrainsLhs_dTAU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  MoFEMFunctionReturn(0);
}

OpPostProcPlastic::OpPostProcPlastic(
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
OpPostProcPlastic::doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  std::array<double, 9> def;
  std::fill(def.begin(), def.end(), 0);

  auto get_tag = [&](const std::string name, size_t size) {
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);

  auto set_matrix_2d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 2; ++r)
      for (size_t c = 0; c != 2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_matrix_3d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 3; ++r)
      for (size_t c = 0; c != 3; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_matrix_2d_symm = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != 2; ++r)
      for (size_t c = 0; c != 2; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_scalar = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    mat(0, 0) = t;
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_plastic_surface = get_tag("PLASTIC_SURFACE", 1);
  auto th_plastic_flow = get_tag("PLASTIC_FLOW", 9);

  auto t_flow = getFTensor2FromMat<3, 3>(*(commonDataPtr->plasticFlowPtr));
  size_t gg = 0;
  for (auto &f : *(commonDataPtr->plasticSurfacePtr)) {
    CHKERR set_tag(th_plastic_surface, gg, set_scalar(f));
    CHKERR set_tag(th_plastic_flow, gg, set_matrix_3d(t_flow));
    ++gg;
    ++t_flow;
  }

  MoFEMFunctionReturn(0);
}
//! [Postprocessing]

}; // namespace OpPlasticTools