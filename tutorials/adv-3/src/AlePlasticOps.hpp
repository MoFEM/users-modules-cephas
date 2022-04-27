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

/** \file AlePlasticOps.hpp
 * \example AlePlasticOps.hpp
 */

namespace AlePlasticOps {

enum ElementSide { LEFT_SIDE = 0, RIGHT_SIDE };

//! [Common data]
struct CommonData : public PlasticOps::CommonData {

  // MatrixDouble tempFluxVal;

  // inline auto getTempFluxValPtr() {
  //   return boost::shared_ptr<MatrixDouble>(shared_from_this(), &tempFluxVal);
  // }

  // for rotation
  boost::shared_ptr<VectorDouble> plasticTauJumpPtr;
  boost::shared_ptr<MatrixDouble> plasticStrainJumpPtr;

  boost::shared_ptr<MatrixDouble> plasticTauDiffAvgPtr;
  boost::shared_ptr<MatrixDouble> plasticStrainDiffAvgPtr;

  boost::shared_ptr<MatrixDouble> guidingVelocityPtr;
  boost::shared_ptr<VectorDouble> velocityDotNormalPtr;

  boost::shared_ptr<MatrixDouble> plasticGradTauPtr;
  boost::shared_ptr<MatrixDouble> plasticGradStrainPtr;

  // data for skeleton computation
  map<int, VectorInt> indicesRowTauSideMap;
  map<int, VectorInt> indicesRowStrainSideMap;

  map<int, VectorInt> indicesColTauSideMap;
  map<int, VectorInt> indicesColStrainSideMap;

  map<int, MatrixDouble> rowBaseSideMap;
  map<int, MatrixDouble> colBaseSideMap;

  map<int, MatrixDouble> rowDiffBaseSideMap;
  map<int, MatrixDouble> colDiffBaseSideMap;

  map<int, MatrixDouble> strainSideMap;
  map<int, VectorDouble> tauSideMap;

  map<int, MatrixDouble> strainDiffSideMap;
  map<int, MatrixDouble> tauDiffSideMap;

  std::array<double, 2> areaMap;
  std::array<int, 2> senseMap;

  FTensor::Ddg<double, SPACE_DIM, SPACE_DIM> Is;
};
//! [Common data]

template <typename T>
inline auto get_rotation_R(FTensor::Tensor1<T, 3> t1_omega, double tt) {

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_rotation = [&](FTensor::Tensor1<double, 3> &t_omega) {
    FTensor::Tensor2<double, 3, 3> t_R;

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
    t_R(i, j) = t_kd(i, j);

    const double angle = sqrt(t_omega(i) * t_omega(i));
    if (std::abs(angle) < 1e-18)
      return t_R;

    FTensor::Tensor2<double, 3, 3> t_Omega;
    t_Omega(i, j) = FTensor::levi_civita<double>(i, j, k) * t_omega(k);
    const double a = sin(angle) / angle;
    const double ss_2 = sin(angle / 2.);
    const double b = 2. * ss_2 * ss_2 / (angle * angle);
    t_R(i, j) += a * t_Omega(i, j);
    t_R(i, j) += b * t_Omega(i, k) * t_Omega(k, j);

    return t_R;
  };

  FTensor::Tensor1<double, 3> my_c_omega;
  my_c_omega(i) = t1_omega(i) * tt;
  auto t_R = get_rotation(my_c_omega);

  return get_rotation(my_c_omega);
};

template <typename T> inline auto get_ntensor(T &base_mat) {
  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
      &*base_mat.data().begin());
};

template <typename T> inline auto get_ntensor(T &base_mat, int gg, int bb) {
  double *ptr = &base_mat(gg, bb);
  return FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(ptr);
};

template <typename T> inline auto get_diff_ntensor(T &base_mat) {
  double *ptr = &*base_mat.data().begin();
  return FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2>(ptr, &ptr[1]);
};

template <typename T>
inline auto get_diff_ntensor(T &base_mat, int gg, int bb) {
  double *ptr = &base_mat(gg, 2 * bb);
  return FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2>(ptr, &ptr[1]);
};

struct OpCalculatePlasticConvRotatingFrame : public DomainEleOp {
  OpCalculatePlasticConvRotatingFrame(
      const std::string field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateVelocityOnSkeleton : public SkeletonEleOp {
  OpCalculateVelocityOnSkeleton(const std::string field_name,
                                boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateJumpOnSkeleton : public SkeletonEleOp {

  OpCalculateJumpOnSkeleton(const std::string field_name,
                            boost::shared_ptr<CommonData> common_data_ptr,
                            boost::shared_ptr<DomainSideEle> side_op_fe);

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<DomainSideEle> sideOpFe;
};

struct OpDomainSideGetColData : public DomainSideEleOp {

  OpDomainSideGetColData(const std::string row_name, const std::string col_name,
                         boost::shared_ptr<CommonData> common_data_ptr);

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpVolumeSideCalculateEP : public DomainSideEleOp {
  OpVolumeSideCalculateEP(const std::string field_name,
                          boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpVolumeSideCalculateTAU : public DomainSideEleOp {
  OpVolumeSideCalculateTAU(const std::string field_name,
                           boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculatePlasticFlowPenalty_Rhs : public AssemblySkeletonEleOp {
  OpCalculatePlasticFlowPenalty_Rhs(
      const std::string field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpCalculateConstraintPenalty_Rhs : public AssemblySkeletonEleOp {
  OpCalculateConstraintPenalty_Rhs(
      const std::string field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
};

struct OpPostProcAleSkeleton : public SkeletonEleOp {
  OpPostProcAleSkeleton(const std::string field_name,
                        moab::Interface &post_proc_mesh,
                        std::vector<EntityHandle> &map_gauss_pts,
                        boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  boost::shared_ptr<CommonData> commonDataPtr;
};

OpCalculatePlasticConvRotatingFrame::OpCalculatePlasticConvRotatingFrame(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, field_name, DomainEleOp::OPROW),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode OpCalculatePlasticConvRotatingFrame::doWork(int side,
                                                           EntityType type,
                                                           EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  FTensor::Index<'k', SPACE_DIM> k;
  FTensor::Index<'I', 3> I;
  FTensor::Index<'J', 3> J;
  FTensor::Index<'K', 3> K;

  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    const size_t nb_integration_pts = data.getN().size1();

    auto t_ep_dot = getFTensor2SymmetricFromMat<SPACE_DIM>(
        *(commonDataPtr->getPlasticStrainDotPtr()));
    auto t_tau_dot =
        getFTensor0FromVec(*(commonDataPtr->getPlasticTauDotPtr()));
    auto t_coords = getFTensor1CoordsAtGaussPts();
    auto t_grad_tau =
        getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->plasticGradTauPtr));

    // OpCalculateTensor2SymmetricFieldGradient
    auto t_grad_plastic_strain = getFTensor3DgFromMat<SPACE_DIM, SPACE_DIM>(
        *(commonDataPtr->plasticGradStrainPtr));
    commonDataPtr->guidingVelocityPtr->resize(SPACE_DIM, nb_integration_pts,
                                              false);
    commonDataPtr->guidingVelocityPtr->clear();
    auto t_omega =
        getFTensor1FromMat<SPACE_DIM>(*commonDataPtr->guidingVelocityPtr);

    FTensor::Tensor1<double, 3> t1_omega(
        angular_velocity[0], angular_velocity[1], angular_velocity[2]);
    FTensor::Tensor2<double, 3, 3> Omega;
    Omega(I, K) = FTensor::levi_civita<double>(I, J, K) * t1_omega(J);
    auto t_tau = getFTensor0FromVec(commonDataPtr->plasticTau);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      // FIXME: maybe here we should take cooordinates that are rotated, X
      // but rotation will not change the velocity, right?
      t_omega(i) = Omega(i, j) * t_coords(j);

      t_tau_dot += t_grad_tau(i) * t_omega(i);
      t_ep_dot(i, j) += t_grad_plastic_strain(i, j, k) * t_omega(k);

      ++t_coords;
      ++t_omega;
      ++t_grad_plastic_strain;
      ++t_ep_dot;
      ++t_tau_dot;
      ++t_tau;
      ++t_grad_tau;
    }
  }

  MoFEMFunctionReturn(0);
}
OpCalculateVelocityOnSkeleton::OpCalculateVelocityOnSkeleton(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : SkeletonEleOp(field_name, field_name, SkeletonEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  // if constexpr (SPACE_DIM == 3)
  //   doEntities[MBTRI] = doEntities[MBQUAD] = true;
  // else
  //   doEntities[MBEDGE] = true;
  doEntities[MBVERTEX] = true;
}

MoFEMErrorCode OpCalculateVelocityOnSkeleton::doWork(int side, EntityType type,
                                                     EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  FTensor::Index<'k', SPACE_DIM> k;
  FTensor::Index<'I', 3> I;
  FTensor::Index<'J', 3> J;
  FTensor::Index<'K', 3> K;

  // const size_t nb_dofs = data.getIndices().size();
  if (true) {
    const size_t nb_integration_pts = getGaussPts().size2();

    auto t_coords = getFTensor1CoordsAtGaussPts();

    auto t_normal = getFTensor1Normal();
    t_normal.normalize();

    // OpCalculateTensor2SymmetricFieldGradient
    commonDataPtr->guidingVelocityPtr->resize(SPACE_DIM, nb_integration_pts,
                                              false);
    commonDataPtr->guidingVelocityPtr->clear();

    commonDataPtr->velocityDotNormalPtr->resize(nb_integration_pts, false);
    commonDataPtr->velocityDotNormalPtr->clear();
    auto t_omega =
        getFTensor1FromMat<SPACE_DIM>(*commonDataPtr->guidingVelocityPtr);
    auto t_X_dot_n = getFTensor0FromVec(*commonDataPtr->velocityDotNormalPtr);
    FTensor::Tensor1<double, 3> t1_omega(
        angular_velocity[0], angular_velocity[1], angular_velocity[2]);
    FTensor::Tensor2<double, 3, 3> Omega;
    Omega(I, K) = FTensor::levi_civita<double>(I, J, K) * t1_omega(J);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

      t_omega(i) = Omega(i, j) * t_coords(j);
      t_X_dot_n = t_omega(j) * t_normal(j);
      // FIXME: which normal should we take?
      //  t_X_dot_n = 1;

      ++t_X_dot_n;
      ++t_coords;
      ++t_omega;
    }
  }

  MoFEMFunctionReturn(0);
}

OpVolumeSideCalculateEP::OpVolumeSideCalculateEP(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainSideEleOp(field_name, field_name, DomainSideEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  if (test_H1)
    doEntities[MBVERTEX] = true;
  else if constexpr (SPACE_DIM == 3)
    doEntities[MBHEX] = doEntities[MBTET] = true;
  else
    doEntities[MBTRI] = doEntities[MBQUAD] = true;
}

MoFEMErrorCode OpVolumeSideCalculateEP::doWork(int side, EntityType type,
                                               EntData &data) {
  MoFEMFunctionBegin;
  const int nb_gauss_pts = getGaussPts().size2();
  const int nb_base_functions = data.getN().size2();

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  FTensor::Index<'k', SPACE_DIM> k;

  // = getFTensor2SymmetricFromMat<3>(*(commonDataPtr->mStressPtr));
  const size_t nb_dofs = data.getIndices().size();

  if (nb_dofs) {

    auto base_function = data.getFTensor0N();
    auto diff_base_function = data.getFTensor1DiffN<SPACE_DIM>();
    int nb_in_loop = getFEMethod()->nInTheLoop;

    auto &cmdata = *commonDataPtr;
    auto &mat_ep = cmdata.strainSideMap[nb_in_loop];
    auto &diff_mat_ep = cmdata.strainDiffSideMap[nb_in_loop];

    cmdata.indicesRowStrainSideMap[nb_in_loop] = data.getIndices();
    cmdata.rowBaseSideMap[nb_in_loop] = data.getN();
    cmdata.rowDiffBaseSideMap[nb_in_loop] = data.getDiffN();

    cmdata.areaMap[nb_in_loop] = getMeasure();
    cmdata.senseMap[nb_in_loop] = getEdgeSense();

    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
    diff_mat_ep.resize(size_symm * SPACE_DIM, nb_gauss_pts, false);
    diff_mat_ep.clear();
    mat_ep.resize(size_symm, nb_gauss_pts, false);
    mat_ep.clear();

    auto gradients_at_gauss_pts =
        getFTensor3DgFromMat<SPACE_DIM, SPACE_DIM>(diff_mat_ep);
    auto values_at_gauss_pts = getFTensor2SymmetricFromMat<SPACE_DIM>(mat_ep);
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      auto field_data = data.getFTensor2SymmetricFieldData<SPACE_DIM>();
      int bb = 0;
      for (; bb != nb_dofs / size_symm; ++bb) {
        values_at_gauss_pts(i, j) += field_data(i, j) * base_function;
        gradients_at_gauss_pts(i, j, k) +=
            field_data(i, j) * diff_base_function(k);

        ++field_data;
        ++base_function;
        ++diff_base_function;
      }
      for (; bb != nb_base_functions; ++bb) {
        ++base_function;
        ++diff_base_function;
      }
      ++values_at_gauss_pts;
      ++gradients_at_gauss_pts;
    }
  }
  MoFEMFunctionReturn(0);
}

OpVolumeSideCalculateTAU::OpVolumeSideCalculateTAU(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainSideEleOp(field_name, field_name, DomainSideEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  if (test_H1)
    doEntities[MBVERTEX] = true;
  else if constexpr (SPACE_DIM == 3)
    doEntities[MBHEX] = doEntities[MBTET] = true;
  else
    doEntities[MBTRI] = doEntities[MBQUAD] = true;
  // doEntities[MBEDGE] = true;
}

MoFEMErrorCode OpVolumeSideCalculateTAU::doWork(int side, EntityType type,
                                                EntData &data) {
  MoFEMFunctionBegin;

  // FIXME:  look into zero_type, zeroType variable setting
  const size_t nb_dofs = data.getIndices().size();
  if (nb_dofs) {
    int nb_gauss_pts = getGaussPts().size2();
    int nb_in_loop = getFEMethod()->nInTheLoop;
    const int nb_base_functions = data.getN().size2();

    commonDataPtr->indicesRowTauSideMap[nb_in_loop] = data.getIndices();
    commonDataPtr->rowBaseSideMap[nb_in_loop] = data.getN();
    commonDataPtr->rowDiffBaseSideMap[nb_in_loop] = data.getDiffN();
    auto &mat_diff_tau = commonDataPtr->tauDiffSideMap[nb_in_loop];
    mat_diff_tau.resize(SPACE_DIM, nb_gauss_pts, false);
    mat_diff_tau.clear();

    auto &vec_tau = commonDataPtr->tauSideMap[nb_in_loop];
    vec_tau.resize(nb_gauss_pts, false);
    vec_tau.clear();

    auto t_tau = getFTensor0FromVec(vec_tau);
    auto gradients_at_gauss_pts = getFTensor1FromMat<SPACE_DIM>(mat_diff_tau);

    auto base_function = data.getFTensor0N();
    auto diff_base_function = data.getFTensor1DiffN<SPACE_DIM>();

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      auto field_data = data.getFTensor0FieldData();
      int bb = 0;
      for (; bb != nb_dofs; ++bb) {
        t_tau += field_data * base_function;
        gradients_at_gauss_pts(i) += field_data * diff_base_function(i);
        ++field_data;
        ++base_function;
        ++diff_base_function;
      }
      for (; bb < nb_base_functions; ++bb) {
        ++base_function;
        ++diff_base_function;
      }
      ++t_tau;
      ++gradients_at_gauss_pts;

      // vec_tau(gg) = inner_prod(trans(data.getN(gg)), data.getFieldData());
    }
  }
  MoFEMFunctionReturn(0);
}

OpCalculateJumpOnSkeleton::OpCalculateJumpOnSkeleton(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<DomainSideEle> side_op_fe)
    : SkeletonEleOp(field_name, field_name, SkeletonEleOp::OPROW),
      commonDataPtr(common_data_ptr), sideOpFe(side_op_fe) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  doEntities[MBVERTEX] = true;
}

MoFEMErrorCode OpCalculateJumpOnSkeleton::doWork(int side, EntityType type,
                                                 EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  FTensor::Index<'k', SPACE_DIM> k;

  EntityHandle ent = getFEEntityHandle();
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  const size_t nb_dofs = data.getIndices().size();
  // if (side == 0) {
  if (nb_dofs) {

    const size_t nb_integration_pts = getGaussPts().size2();

    commonDataPtr->strainSideMap.clear();
    commonDataPtr->tauSideMap.clear();
    commonDataPtr->strainDiffSideMap.clear();
    commonDataPtr->tauDiffSideMap.clear();

    CHKERR loopSide("dFE",
                    dynamic_cast<MoFEM::ForcesAndSourcesCore *>(sideOpFe.get()),
                    SPACE_DIM);
    auto &ep_data_map = commonDataPtr->strainSideMap;
    auto &tau_data_map = commonDataPtr->tauSideMap;
    auto &diff_ep_data_map = commonDataPtr->strainDiffSideMap;
    auto &diff_tau_data_map = commonDataPtr->tauDiffSideMap;

#ifndef NDEBUG
    // check all dimensions
    const int check_size = ep_data_map.size();
    if (ep_data_map.size() != 2)
      if (static_cast<int>(ep_data_map.begin()->second.size2()) !=
          nb_integration_pts)
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "wrong size of the strain map");
    if (nb_integration_pts != data.getN().size1())
      SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
              "wrong number of integration points");
#endif

    auto t_ep_l =
        getFTensor2SymmetricFromMat<SPACE_DIM>(ep_data_map.at(LEFT_SIDE));
    auto t_ep_r =
        getFTensor2SymmetricFromMat<SPACE_DIM>(ep_data_map.at(RIGHT_SIDE));

    auto t_diff_ep_l = getFTensor3DgFromMat<SPACE_DIM, SPACE_DIM>(
        diff_ep_data_map.at(LEFT_SIDE));
    auto t_diff_ep_r = getFTensor3DgFromMat<SPACE_DIM, SPACE_DIM>(
        diff_ep_data_map.at(RIGHT_SIDE));

    auto t_tau_l = getFTensor0FromVec(tau_data_map.at(LEFT_SIDE));
    auto t_tau_r = getFTensor0FromVec(tau_data_map.at(RIGHT_SIDE));

    auto t_diff_tau_l =
        getFTensor1FromMat<SPACE_DIM>(diff_tau_data_map.at(LEFT_SIDE));
    auto t_diff_tau_r =
        getFTensor1FromMat<SPACE_DIM>(diff_tau_data_map.at(RIGHT_SIDE));

    // fl_L * jump
    // -fl_R * jump

    // field values
    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

    commonDataPtr->plasticTauJumpPtr->resize(nb_integration_pts, false);
    commonDataPtr->plasticTauDiffAvgPtr->resize(SPACE_DIM, nb_integration_pts);
    commonDataPtr->plasticStrainJumpPtr->resize(size_symm, nb_integration_pts,
                                                false);
    commonDataPtr->plasticStrainDiffAvgPtr->resize(size_symm * SPACE_DIM,
                                                   nb_integration_pts);

    auto t_tau_jump = getFTensor0FromVec(*commonDataPtr->plasticTauJumpPtr);
    auto t_ep_jump = getFTensor2SymmetricFromMat<SPACE_DIM>(
        *commonDataPtr->plasticStrainJumpPtr);

    auto t_diff_tau_avg =
        getFTensor1FromMat<SPACE_DIM>(*commonDataPtr->plasticTauDiffAvgPtr);
    auto t_diff_ep_avg = getFTensor3DgFromMat<SPACE_DIM, SPACE_DIM>(
        *(commonDataPtr->plasticStrainDiffAvgPtr));

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

      t_ep_jump(i, j) = t_ep_r(i, j) - t_ep_l(i, j);
      t_tau_jump = t_tau_r - t_tau_l;

      t_diff_tau_avg(i) = 0.5 * (t_diff_tau_r(i) + t_diff_tau_l(i));
      t_diff_ep_avg(i, j, k) =
          0.5 * (t_diff_ep_r(i, j, k) + t_diff_ep_l(i, j, k));

      ++t_tau_jump;
      ++t_tau_l;
      ++t_tau_r;

      ++t_ep_jump;
      ++t_ep_l;
      ++t_ep_r;

      ++t_diff_tau_avg;
      ++t_diff_tau_l;
      ++t_diff_tau_r;

      ++t_diff_ep_avg;
      ++t_diff_ep_l;
      ++t_diff_ep_r;
    }
  }

  MoFEMFunctionReturn(0);
}

OpDomainSideGetColData::OpDomainSideGetColData(
    const std::string row_name, const std::string col_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainSideEleOp(row_name, col_name, DomainSideEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {}

MoFEMErrorCode
OpDomainSideGetColData::doWork(int row_side, int col_side, EntityType row_type,
                               EntityType col_type,
                               DataForcesAndSourcesCore::EntData &row_data,
                               DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = row_data.getIndices().size();
  const int nb_in_loop = getFEMethod()->nInTheLoop;

  auto &cmdata = *commonDataPtr;
  if ((CN::Dimension(row_type) == SPACE_DIM) &&
      (CN::Dimension(col_type) == SPACE_DIM)) {
    if (colFieldName == "TAU") {
      cmdata.indicesRowTauSideMap[nb_in_loop] = row_data.getIndices();
      cmdata.indicesColTauSideMap[nb_in_loop] = col_data.getIndices();
    } else if (colFieldName == "EP") {
      cmdata.indicesRowStrainSideMap[nb_in_loop] = row_data.getIndices();
      cmdata.indicesColStrainSideMap[nb_in_loop] = col_data.getIndices();
    } else
      MOFEM_LOG("WORLD", Sev::error) << "wrong side field";

    cmdata.rowBaseSideMap[nb_in_loop] = row_data.getN();
    cmdata.rowDiffBaseSideMap[nb_in_loop] = row_data.getDiffN();
    cmdata.colBaseSideMap[nb_in_loop] = col_data.getN();
    cmdata.colDiffBaseSideMap[nb_in_loop] = col_data.getDiffN();
  }
  cmdata.areaMap[nb_in_loop] = getMeasure();
  cmdata.senseMap[nb_in_loop] = getEdgeSense();

  MoFEMFunctionReturn(0);
}

OpCalculatePlasticFlowPenalty_Rhs::OpCalculatePlasticFlowPenalty_Rhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblySkeletonEleOp(field_name, field_name,
                            AssemblySkeletonEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  doEntities[MBVERTEX] = true;
}

MoFEMErrorCode OpCalculatePlasticFlowPenalty_Rhs::doWork(int side,
                                                         EntityType type,
                                                         EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  FTensor::Index<'k', SPACE_DIM> k;

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  const auto in_the_loop = 1; // skeleton

  EntityHandle ent = getFEEntityHandle();
  auto &cmdata = *commonDataPtr;

  // calculate  penalty
  const double s = getMeasure() / (cmdata.areaMap[0] + cmdata.areaMap[1]);
  const double p = penalty * s;

  auto t_normal = getFTensor1Normal();
  t_normal.normalize();

  auto &idx_rmap = cmdata.indicesRowStrainSideMap;
  const double beta = static_cast<double>(nitsche) / (in_the_loop + 1);

  for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

    const size_t nb_dofs = idx_rmap.at(s0).size();
    const size_t nb_integration_pts = getGaussPts().size2();

    auto t_w = getFTensor0IntegrationWeight();
    auto t_X_dot_n = getFTensor0FromVec(*cmdata.velocityDotNormalPtr);
    auto t_omega = getFTensor1FromMat<SPACE_DIM>(*cmdata.guidingVelocityPtr);

    auto t_ep_jump =
        getFTensor2SymmetricFromMat<SPACE_DIM>(*cmdata.plasticStrainJumpPtr);
    auto t_grad_ep_avg =
        getFTensor3DgFromMat<SPACE_DIM, SPACE_DIM>(*cmdata.plasticStrainDiffAvgPtr);

    FTensor::Tensor1<double, SPACE_DIM> t_vn_plus;
    FTensor::Dg<double, SPACE_DIM, size_symm> t_n;
    FTensor::Tensor1<double, SPACE_DIM> t_vn;

    if (nb_dofs) {

      const auto sense_row = cmdata.senseMap[s0];
      const auto nb_row_base_functions = cmdata.rowBaseSideMap[s0].size2();

      VectorDouble nf(nb_dofs, false);
      nf.clear();

      auto t_row_base = get_ntensor(cmdata.rowBaseSideMap[s0]);
      auto t_diff_row_base = get_diff_ntensor(cmdata.rowDiffBaseSideMap[s0]);

      for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
        const double alpha = getMeasure() * t_w;

        auto t_nf = get_nf(nf, FTensor::Number<SPACE_DIM>());

        size_t rr = 0;
        for (; rr != nb_dofs / size_symm; ++rr) {

          // FTensor::Tensor1<double, SPACE_DIM> t_vn_plus;
          // t_vn_plus(i) = beta * (phi * t_diff_row_base(i) / p);
          // FTensor::Tensor1<double, SPACE_DIM> t_vn;
          // t_vn(i) = t_row_base * t_normal(i) * sense_row - t_vn_plus(i);

          // t_nf(i, j) -= alpha * (t_vn(k) * t_normal(k)) * t_ep_jump(i, j);

          t_n(i, j, k) = p * t_ep_jump(i, j) * t_normal(k) * sense_row -
                   (phi / p) * t_grad_ep_avg(i, j, k) ;
          auto c = phi;

          t_vn_plus(i) = beta * (phi * t_diff_row_base(i) / p);
          t_vn(i) = t_row_base * t_normal(i) * sense_row - t_vn_plus(i);

          t_nf(i, j) += alpha * (t_vn(k) * t_omega(k)) * (t_n(i, j, k) * t_omega(k));
          t_nf(i, j) += alpha * p * t_vn_plus(i) * t_omega(i) *
                        (t_grad_ep_avg(i, j, k) * t_omega(k));

          ++t_nf;
          ++t_row_base;
          ++t_diff_row_base;
        }
        for (; rr < nb_row_base_functions; ++rr) {
          ++t_row_base;
          ++t_diff_row_base;
        }
        
        ++t_w;
        ++t_grad_ep_avg;
        ++t_X_dot_n;
        ++t_omega;
        ++t_ep_jump;
      }
      CHKERR ::VecSetValues(getKSPf(), idx_rmap.at(s0).size(),
                            &*idx_rmap.at(s0).begin(), &*nf.data().begin(),
                            ADD_VALUES);
    }
  }

  MoFEMFunctionReturn(0);
}

OpCalculateConstraintPenalty_Rhs::OpCalculateConstraintPenalty_Rhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblySkeletonEleOp(field_name, field_name,
                            AssemblySkeletonEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  doEntities[MBVERTEX] = true;
}

MoFEMErrorCode OpCalculateConstraintPenalty_Rhs::doWork(int side,
                                                        EntityType type,
                                                        EntData &data) {
  MoFEMFunctionBegin;

  const auto in_the_loop = 1; // skeleton

  EntityHandle ent = getFEEntityHandle();
  auto &cmdata = *commonDataPtr;

  // calculate  penalty
  const double s = getMeasure() / (cmdata.areaMap[0] + cmdata.areaMap[1]);
  const double p = penalty * s;

  auto t_normal = getFTensor1Normal();
  t_normal.normalize();

  auto &idx_rmap = cmdata.indicesRowTauSideMap;
  const double beta = static_cast<double>(nitsche) / (in_the_loop + 1);

  for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

    const size_t nb_dofs = idx_rmap.at(s0).size();
    const size_t nb_integration_pts = getGaussPts().size2();

    auto t_w = getFTensor0IntegrationWeight();
    auto t_X_dot_n = getFTensor0FromVec(*cmdata.velocityDotNormalPtr);
    auto t_omega = getFTensor1FromMat<SPACE_DIM>(*cmdata.guidingVelocityPtr);

    auto t_tau_jump = getFTensor0FromVec(*cmdata.plasticTauJumpPtr);
    auto t_grad_tau_avg =
        getFTensor1FromMat<SPACE_DIM>(*cmdata.plasticTauDiffAvgPtr);

    FTensor::Tensor1<double, SPACE_DIM> t_vn_plus;
    FTensor::Tensor1<double, SPACE_DIM> t_n;
    FTensor::Tensor1<double, SPACE_DIM> t_vn;

    if (nb_dofs) {

      const auto sense_row = cmdata.senseMap[s0];
      const auto nb_row_base_functions = cmdata.rowBaseSideMap[s0].size2();

      VectorDouble nf(nb_dofs, false);
      nf.clear();

      auto t_row_base = get_ntensor(cmdata.rowBaseSideMap[s0]);
      auto t_diff_row_base = get_diff_ntensor(cmdata.rowDiffBaseSideMap[s0]);

      for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
        const double alpha = getMeasure() * t_w;
        auto t_nf = nf.data().begin();

        size_t rr = 0;
        for (; rr != nb_dofs; ++rr) {

          t_n(i) = p * t_tau_jump * t_normal(i) * sense_row -
                   (phi / p) * t_grad_tau_avg(i);

          t_vn_plus(i) = beta * (phi * t_diff_row_base(i) / p);
          t_vn(i) = t_row_base * t_normal(i) * sense_row - t_vn_plus(i);

          *t_nf += alpha * t_vn(k) * t_omega(k) * (t_n(i) * t_omega(i));
          *t_nf += alpha * p * t_vn_plus(i) * t_omega(i) *
                  (t_grad_tau_avg(k) * t_omega(k));

          ++t_nf;
          ++t_row_base;
          ++t_diff_row_base;
        }
        for (; rr < nb_row_base_functions; ++rr) {
          ++t_row_base;
          ++t_diff_row_base;
        }
        
        ++t_w;
        ++t_grad_tau_avg;
        ++t_X_dot_n;
        ++t_omega;
        ++t_tau_jump;
      }
      CHKERR ::VecSetValues(getKSPf(), idx_rmap.at(s0).size(),
                            &*idx_rmap.at(s0).begin(), &*nf.data().begin(),
                            ADD_VALUES);
    }
  }

  MoFEMFunctionReturn(0);
}

struct OpCalculatePlasticFlowPenaltyLhs_dEP : public AssemblySkeletonEleOp {
  OpCalculatePlasticFlowPenaltyLhs_dEP(
      const std::string row_field_name, const std::string col_field_name,
      boost::shared_ptr<CommonData> common_data_ptr);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

protected:
  boost::shared_ptr<CommonData> commonDataPtr;
};

OpCalculatePlasticFlowPenaltyLhs_dEP::OpCalculatePlasticFlowPenaltyLhs_dEP(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblySkeletonEleOp(row_field_name, col_field_name,
                            AssemblySkeletonEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
  sYmm = false;
}

struct OpCalculateConstraintPenaltyLhs_dTAU
    : public OpCalculatePlasticFlowPenaltyLhs_dEP {
  using OpCalculatePlasticFlowPenaltyLhs_dEP::
      OpCalculatePlasticFlowPenaltyLhs_dEP;
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);
};

MoFEMErrorCode OpCalculatePlasticFlowPenaltyLhs_dEP::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  FTensor::Index<'k', SPACE_DIM> k;
  FTensor::Index<'l', SPACE_DIM> l;
  FTensor::Index<'m', SPACE_DIM> m;
  FTensor::Index<'n', SPACE_DIM> n;

  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  if (row_type != MBVERTEX || col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  const auto in_the_loop = 1; // skeleton

  EntityHandle ent = getFEEntityHandle();
  auto &cmdata = *commonDataPtr;

  // calculate  penalty
  const double s = getMeasure() / (cmdata.areaMap[0] + cmdata.areaMap[1]);
  const double p = penalty * s;

  auto t_diff_plastic_strain = diff_tensor();
  // FTensor::Tensor4<double, SPACE_DIM, SPACE_DIM, SPACE_DIM, SPACE_DIM>
  //     t_diff_tensor_symm;
  // t_diff_tensor_symm(i, j, k, l) =
  //     cmdata.Is(i, j, m, n) * t_diff_plastic_strain(m, n, k, l);

  auto t_normal = getFTensor1Normal();
  t_normal.normalize();

  const size_t nb_integration_pts = getGaussPts().size2();
  const double beta = static_cast<double>(nitsche) / (in_the_loop + 1);

  for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

    const auto nb_rows = cmdata.indicesRowStrainSideMap[s0].size();

    if (nb_rows) {

      const auto sense_row = cmdata.senseMap[s0];

      const auto nb_row_base_functions = cmdata.rowBaseSideMap[s0].size2();
      for (auto s1 : {LEFT_SIDE, RIGHT_SIDE}) {

        // get orientation of the edge
        const auto sense_col = cmdata.senseMap[s1];
        const auto nb_cols = cmdata.indicesColStrainSideMap[s1].size();

        // resize local element matrix
        MatrixDouble locMat(nb_rows, nb_cols, false);
        locMat.clear();

        auto t_row_base = get_ntensor(cmdata.rowBaseSideMap[s0]);
        auto t_diff_row_base = get_diff_ntensor(cmdata.rowDiffBaseSideMap[s0]);
        auto t_w = getFTensor0IntegrationWeight();
        auto t_X_dot_n = getFTensor0FromVec(*cmdata.velocityDotNormalPtr);
        auto t_omega =
            getFTensor1FromMat<SPACE_DIM>(*cmdata.guidingVelocityPtr);

        for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

          const double alpha = getMeasure() * t_w * t_X_dot_n;

          // iterate rows
          size_t rr = 0;
          for (; rr != nb_rows / size_symm; ++rr) {

            auto t_mat = get_mat_tensor_sym_dtensor_sym(
                rr, locMat, FTensor::Number<SPACE_DIM>());

            // calculate tetting function
            FTensor::Tensor1<double, SPACE_DIM> t_vn_plus;
            t_vn_plus(i) = beta * (phi * t_diff_row_base(i) / p);
            FTensor::Tensor1<double, SPACE_DIM> t_vn;
            t_vn(i) = t_row_base * t_normal(i) * sense_row - t_vn_plus(i);

            // get base functions on columns
            auto t_col_base = get_ntensor(cmdata.colBaseSideMap[s1], gg, 0);
            auto t_diff_col_base =
                get_diff_ntensor(cmdata.colDiffBaseSideMap[s1], gg, 0);

            // iterate columns
            for (size_t cc = 0; cc != nb_cols / size_symm; ++cc) {

              // calculate variance of tested function
              FTensor::Tensor1<double, SPACE_DIM> t_un;
              t_un(i) = -p * (t_col_base * t_normal(i) * sense_col -
                              beta * t_diff_col_base(i) / p);

              // assemble matrix
              t_mat(i, j, k, l) -= (alpha * (t_vn(m) * t_un(m))) *
                                   cmdata.Is(i, j, m, n) *
                                   t_diff_plastic_strain(m, n, k, l);

              t_mat(i, j, k, l) -=
                  (alpha * (t_vn_plus(m) * (beta * t_diff_col_base(m)))) *
                  cmdata.Is(i, j, m, n) * t_diff_plastic_strain(m, n, k, l);

              // move to next column base and element of matrix
              ++t_col_base;
              ++t_diff_col_base;
              ++t_mat;
            }

            // move to next row base
            ++t_row_base;
            ++t_diff_row_base;
          }
          for (; rr < nb_row_base_functions; ++rr) {
            ++t_row_base;
            ++t_diff_row_base;
          }

          ++t_w;
          ++t_X_dot_n;
          ++t_omega;
        }

        // assemble system
        CHKERR ::MatSetValues(getKSPB(),
                              cmdata.indicesRowStrainSideMap[s0].size(),
                              &*cmdata.indicesRowStrainSideMap[s0].begin(),
                              cmdata.indicesColStrainSideMap[s1].size(),
                              &*cmdata.indicesColStrainSideMap[s1].begin(),
                              &*locMat.data().begin(), ADD_VALUES);
      }
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode OpCalculateConstraintPenaltyLhs_dTAU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_type != MBVERTEX || col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  constexpr auto size_symm = 1;

  const auto in_the_loop = 1; // skeleton

  EntityHandle ent = getFEEntityHandle();
  auto &cmdata = *commonDataPtr;

  // calculate  penalty
  const double s = getMeasure() / (cmdata.areaMap[0] + cmdata.areaMap[1]);
  const double p = penalty * s;

  auto t_normal = getFTensor1Normal();
  t_normal.normalize();

  const size_t nb_integration_pts = getGaussPts().size2();
  const double beta = static_cast<double>(nitsche) / (in_the_loop + 1);

  for (auto s0 : {LEFT_SIDE, RIGHT_SIDE}) {

    const auto nb_rows = cmdata.indicesRowTauSideMap[s0].size();

    if (nb_rows) {

      const auto sense_row = cmdata.senseMap[s0];

      const auto nb_row_base_functions = cmdata.rowBaseSideMap[s0].size2();
      for (auto s1 : {LEFT_SIDE, RIGHT_SIDE}) {

        // get orientation of the edge
        const auto sense_col = cmdata.senseMap[s1];
        const auto nb_cols = cmdata.indicesColTauSideMap[s1].size();

        // resize local element matrix
        MatrixDouble locMat(nb_rows, nb_cols, false);
        locMat.clear();

        auto t_row_base = get_ntensor(cmdata.rowBaseSideMap[s0]);
        auto t_diff_row_base = get_diff_ntensor(cmdata.rowDiffBaseSideMap[s0]);
        auto t_w = getFTensor0IntegrationWeight();
        auto t_X_dot_n = getFTensor0FromVec(*cmdata.velocityDotNormalPtr);
        auto t_omega =
            getFTensor1FromMat<SPACE_DIM>(*cmdata.guidingVelocityPtr);

        for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

          const double alpha = getMeasure() * t_w * t_X_dot_n;

          // iterate rows
          auto t_mat = locMat.data().begin();
          size_t rr = 0;
          for (; rr != nb_rows / size_symm; ++rr) {

            // calculate tetting function
            FTensor::Tensor1<double, SPACE_DIM> t_vn_plus;
            t_vn_plus(i) = beta * (phi * t_diff_row_base(i) / p);
            FTensor::Tensor1<double, SPACE_DIM> t_vn;
            t_vn(i) = t_row_base * t_normal(i) * sense_row - t_vn_plus(i);

            // get base functions on columns
            auto t_col_base = get_ntensor(cmdata.colBaseSideMap[s1], gg, 0);
            auto t_diff_col_base =
                get_diff_ntensor(cmdata.colDiffBaseSideMap[s1], gg, 0);

            // iterate columns
            for (size_t cc = 0; cc != nb_cols / size_symm; ++cc) {

              // calculate variance of tested function
              FTensor::Tensor1<double, SPACE_DIM> t_un;
              t_un(i) = -p * (t_col_base * t_normal(i) * sense_col -
                              beta * t_diff_col_base(i) / p);

              // assemble matrix
              t_mat -= (alpha * (t_vn(m) * t_un(m)));
              t_mat -= (alpha * (t_vn_plus(m) * (beta * t_diff_col_base(m))));

              // move to next column base and element of matrix
              ++t_col_base;
              ++t_diff_col_base;
              ++t_mat;
            }

            // move to next row base
            ++t_row_base;
            ++t_diff_row_base;
          }
          for (; rr < nb_row_base_functions; ++rr) {
            ++t_row_base;
            ++t_diff_row_base;
          }

          ++t_w;
          ++t_omega;
          ++t_X_dot_n;
        }

        // assemble system
        CHKERR ::MatSetValues(getKSPB(), cmdata.indicesRowTauSideMap[s0].size(),
                              &*cmdata.indicesRowTauSideMap[s0].begin(),
                              cmdata.indicesColTauSideMap[s1].size(),
                              &*cmdata.indicesColTauSideMap[s1].begin(),
                              &*locMat.data().begin(), ADD_VALUES);
      }
    }
  }

  MoFEMFunctionReturn(0);
}

OpPostProcAleSkeleton::OpPostProcAleSkeleton(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : SkeletonEleOp(field_name, SkeletonEleOp::OPROW),
      postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
  doEntities[MBVERTEX] = true;
}

//! [Postprocessing]
MoFEMErrorCode OpPostProcAleSkeleton::doWork(int side, EntityType type,
                                             EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;

  if (side != 0)
    MoFEMFunctionReturnHot(0);

  std::array<double, 9> def;
  std::fill(def.begin(), def.end(), 0);
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  auto get_tag = [&](const std::string name, size_t size) {
    Tag th;
    CHKERR postProcMesh.tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       def.data());
    return th;
  };

  MatrixDouble3by3 mat(3, 3);
  auto set_float_precision = [](const double x) {
    if (std::abs(x) < std::numeric_limits<float>::epsilon())
      return 0.;
    else
      return x;
  };

  auto set_matrix_3d = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != SPACE_DIM; ++r)
      for (size_t c = 0; c != SPACE_DIM; ++c)
        mat(r, c) = t(r, c);
    return mat;
  };

  auto set_scalar = [&](auto t) -> MatrixDouble3by3 & {
    mat.clear();
    mat(0, 0) = t;
    return mat;
  };

  auto set_vector = [&](auto &t) -> MatrixDouble3by3 & {
    mat.clear();
    for (size_t r = 0; r != SPACE_DIM; ++r)
      mat(0, r) = t(r);
    return mat;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    for (auto &v : mat.data())
      v = set_float_precision(v);
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_tau_jump = get_tag("TAU_JUMP", 1);
  auto th_velocity = get_tag("VELOCITY", 3);
  auto th_rotation = get_tag("ROTATION", 3);
  auto th_strain_jump = get_tag("EP_JUMP", 9);
#ifndef NDEBUG
  auto th_normal = get_tag("NORMAL", 3);
  auto t_normal = getFTensor1Normal();
  t_normal(i) /= sqrt(t_normal(j) * t_normal(j));
#endif // NDEBUG

  auto t_ep_jump = getFTensor2SymmetricFromMat<SPACE_DIM>(
      *commonDataPtr->plasticStrainJumpPtr);
  auto t_omega =
      getFTensor1FromMat<SPACE_DIM>(*commonDataPtr->guidingVelocityPtr);

  FTensor::Tensor1<double, 3> omega(angular_velocity[0], angular_velocity[1],
                                    angular_velocity[2]);

  int nb_gauss_pts = mapGaussPts.size();
  auto t_X_dot_n = getFTensor0FromVec(*commonDataPtr->velocityDotNormalPtr);
  auto t_coords = getFTensor1CoordsAtGaussPts();

  auto rot = get_rotation_R(omega, petsc_time);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    FTensor::Tensor1<double, SPACE_DIM> rot_disp;
    rot_disp(i) = rot(j, i) * t_coords(j) - t_coords(i);

    const double tau_jump = (*commonDataPtr->plasticTauJumpPtr)[gg] * t_X_dot_n;

    CHKERR set_tag(th_tau_jump, gg, set_scalar(tau_jump));
    CHKERR set_tag(th_velocity, gg, set_vector(t_omega));
    CHKERR set_tag(th_rotation, gg, set_vector(rot_disp));

#ifndef NDEBUG
    CHKERR set_tag(th_normal, gg, set_vector(t_normal));
#endif // NDEBUG

    t_ep_jump(i, j) *= t_X_dot_n;
    CHKERR set_tag(th_strain_jump, gg, set_matrix_3d(t_ep_jump));

    ++t_coords;
    ++t_X_dot_n;
    ++t_omega;
    ++t_ep_jump;
  }

  MoFEMFunctionReturn(0);
}

struct MonitorPostProcSkeleton : public FEMethod {

  MonitorPostProcSkeleton(SmartPetscObj<DM> &dm,
                          boost::shared_ptr<PostProcEleSkeleton> &post_proc_fe)
      : dM(dm), postProcFeSkeletonFe(post_proc_fe){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    CHKERR TSGetTime(ts, &petsc_time);
    // FIXME: testing
    // double max_velocity = 0.00;
    // angular_velocity[2] = 0;
    // if (petsc_time >= max_load_time * 2)
    //   angular_velocity[2] =
    //       std::min(abs(max_velocity), 0.01 * (petsc_time - max_load_time
    //       * 2.));
    // cout << "angular velocity is " << angular_velocity[2] << endl;
    auto make_vtk = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "sFE", postProcFeSkeletonFe);
      CHKERR postProcFeSkeletonFe->writeFile(
          "out_skeleton_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    CHKERR make_vtk();

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEleSkeleton> postProcFeSkeletonFe;
};

} // namespace AlePlasticOps