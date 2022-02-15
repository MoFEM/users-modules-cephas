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
  boost::shared_ptr<MatrixDouble> plastic_N_TauJumpPtr;
  boost::shared_ptr<MatrixDouble> plastic_N_StrainJumpPtr;
  boost::shared_ptr<MatrixDouble> guidingVelocityPtr;
  boost::shared_ptr<VectorDouble> velocityDotNormalPtr;

  boost::shared_ptr<MatrixDouble> plasticGradTauPtr;
  boost::shared_ptr<MatrixDouble> plasticGradStrainPtr;

  // data for skeleton computation
  map<int, EntData *> plasticDataStrainSideMap;
  map<int, EntData *> plasticDataTauSideMap;

  map<int, MatrixDouble> plasticStrainSideMap;
  map<int, VectorDouble> plasticTauSideMap;
  map<int, MatrixDouble> velocityVecSideMap;
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
  OpCalculateJumpOnSkeleton(const std::string row_field,
                            const std::string col_field,
                            boost::shared_ptr<CommonData> common_data_ptr,
                            boost::shared_ptr<DomainSideEle> side_op_fe);

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<DomainSideEle> sideOpFe;
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

    auto t_plastic_strain_dot = getFTensor2SymmetricFromMat<SPACE_DIM>(
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

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      // FIXME: maybe here we should take cooordinates that are rotated, X
      // but rotation will not change the velocity, right?
      t_omega(i) = Omega(i, j) * t_coords(j);
      t_tau_dot += t_grad_tau(i) * t_omega(i);
      t_plastic_strain_dot(i, j) += t_grad_plastic_strain(i, j, k) * t_omega(k);

      ++t_coords;
      ++t_omega;
      ++t_grad_plastic_strain;
      ++t_plastic_strain_dot;
      ++t_tau_dot;
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
    t_normal(i) /= sqrt(t_normal(j) * t_normal(j));

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
      // FIXME: maybe here we should take cooordinates that are rotated, X

      t_omega(i) = Omega(i, j) * t_coords(j);
      t_X_dot_n = t_omega(j) * t_normal(j);

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
  if constexpr (SPACE_DIM == 3)
    doEntities[MBHEX] = doEntities[MBTET] = true;
  else
    doEntities[MBTRI] = doEntities[MBQUAD] = true;
  // doEntities[MBEDGE] = true;
}

MoFEMErrorCode OpVolumeSideCalculateEP::doWork(int side, EntityType type,
                                               EntData &data) {
  MoFEMFunctionBegin;
  const int nb_gauss_pts = getGaussPts().size2();
  const int nb_base_functions = data.getN().size2();

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  FTensor::Index<'k', SPACE_DIM> k;

  // commonDataPtr->mStressPtr->resize(6, nb_gauss_pts);
  // auto &t_D = commonDataPtr->tD;
  // auto t_strain =
  // getFTensor2SymmetricFromMat<3>(*(commonDataPtr->mStrainPtr)); auto t_stress
  // = getFTensor2SymmetricFromMat<3>(*(commonDataPtr->mStressPtr));
  // FIXME: this is wrong, resize only on zero type
  const size_t nb_dofs = data.getIndices().size();
  // data.getFieldData().size() == 0
  // if (side == getFaceSideNumber()) {
  if (nb_dofs) {

    auto base_function = data.getFTensor0N();
    int nb_in_loop = getFEMethod()->nInTheLoop;

    int sense = 0;
    // if constexpr (std::is_same<
    //                   T,
    //                   FaceElementForcesAndSourcesCoreOnSideSwitch<0>>::value)
    //   sense = getEdgeSense();
    // else
    //   sense = getFaceSense();

    auto &mat_ep = commonDataPtr->plasticStrainSideMap[nb_in_loop];
    commonDataPtr->plasticDataStrainSideMap[nb_in_loop] = &data;

    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
    mat_ep.resize(size_symm, nb_gauss_pts, false);
    mat_ep.clear();

    auto values_at_gauss_pts = getFTensor2SymmetricFromMat<SPACE_DIM>(mat_ep);
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      auto field_data = data.getFTensor2SymmetricFieldData<SPACE_DIM>();
      size_t bb = 0;
      for (; bb != nb_dofs / size_symm; ++bb) {
        values_at_gauss_pts(i, j) += field_data(i, j) * base_function;
        ++base_function;
      }
      for (; bb != nb_base_functions; ++bb)
        ++base_function;
      ++values_at_gauss_pts;
    }
  }
  MoFEMFunctionReturn(0);
}

OpVolumeSideCalculateTAU::OpVolumeSideCalculateTAU(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : DomainSideEleOp(field_name, field_name, DomainSideEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  if constexpr (SPACE_DIM == 3)
    doEntities[MBHEX] = doEntities[MBTET] = true;
  else
    doEntities[MBTRI] = doEntities[MBQUAD] = true;
  // doEntities[MBEDGE] = true;
}

MoFEMErrorCode OpVolumeSideCalculateTAU::doWork(int side, EntityType type,
                                                EntData &data) {
  MoFEMFunctionBegin;

  // FIXME: this is wrong, resize only on zero type
  const size_t nb_dofs = data.getIndices().size();
  // data.getFieldData().size()
  if (true) {
    int nb_gauss_pts = getGaussPts().size2();
    int nb_in_loop = getFEMethod()->nInTheLoop;
    int sense = 0;
    // if constexpr (std::is_same<
    //                   T,
    //                   FaceElementForcesAndSourcesCoreOnSideSwitch<0>>::value)
    //   sense = getEdgeSense();
    // else
    //   sense = getFaceSense();

    auto &vec_tau = commonDataPtr->plasticTauSideMap[nb_in_loop];
    commonDataPtr->plasticDataTauSideMap[nb_in_loop] = &data;

    vec_tau.resize(nb_gauss_pts, false);
    vec_tau.clear();

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      vec_tau(gg) = inner_prod(trans(data.getN(gg)), data.getFieldData());
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
  // if constexpr (SPACE_DIM == 3)
  //   doEntities[MBTRI] = doEntities[MBQUAD] = true;
  // else
  doEntities[MBVERTEX] = true;
}

MoFEMErrorCode OpCalculateJumpOnSkeleton::doWork(int side, EntityType type,
                                                 EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;

  EntityHandle ent = getFEEntityHandle();
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  // if (side == 0) {
  if (true) {

    const size_t nb_integration_pts = getGaussPts().size2();
    // const size_t nb_integration_pts = data.getN().size1();
    const size_t nb_base_functions = data.getN().size2();
    commonDataPtr->plasticStrainSideMap.clear();
    CHKERR loopSide("dFE",
                    dynamic_cast<MoFEM::ForcesAndSourcesCore *>(sideOpFe.get()),
                    SPACE_DIM);

#ifndef NDEBUG
    // check all dimensions
    const int check_size = commonDataPtr->plasticStrainSideMap.size();
    if (commonDataPtr->plasticStrainSideMap.size() == 1)
      if (static_cast<int>(
              commonDataPtr->plasticStrainSideMap.begin()->second.size2()) !=
          nb_integration_pts)
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "wrong number of integration points");
    if (nb_integration_pts != data.getN().size1())
      SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
              "wrong number of integration points");
#endif

    auto &plastic_mat_l = commonDataPtr->plasticStrainSideMap.at(LEFT_SIDE);
    auto &plastic_mat_r = commonDataPtr->plasticStrainSideMap.at(RIGHT_SIDE);
    auto t_plastic_strain_l =
        getFTensor2SymmetricFromMat<SPACE_DIM>(plastic_mat_l);
    auto t_plastic_strain_r =
        getFTensor2SymmetricFromMat<SPACE_DIM>(plastic_mat_r);
    auto &tau_vec_l = commonDataPtr->plasticTauSideMap.at(LEFT_SIDE);
    auto &tau_vec_r = commonDataPtr->plasticTauSideMap.at(RIGHT_SIDE);
    auto t_tau_l = getFTensor0FromVec(tau_vec_l);
    auto t_tau_r = getFTensor0FromVec(tau_vec_r);

    // commonDataPtr->domainGaussPts = getGaussPts();
    auto t_w = getFTensor0IntegrationWeight();
    auto t_base = data.getFTensor0N();
    auto t_coords = getFTensor1CoordsAtGaussPts();

    // fl_L * jump
    // -fl_R * jump

    // field values
    commonDataPtr->plasticTauJumpPtr->resize(nb_integration_pts, false);
    commonDataPtr->plasticStrainJumpPtr->resize(size_symm, nb_integration_pts,
                                                false);

    // shape funcs
    auto t_plastic_strain_jump = getFTensor2SymmetricFromMat<SPACE_DIM>(
        *commonDataPtr->plasticStrainJumpPtr);
    auto t_tau_jump = getFTensor0FromVec(*commonDataPtr->plasticTauJumpPtr);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      double alpha = getMeasure() * t_w;

      t_plastic_strain_jump(i, j) =
          t_plastic_strain_l(i, j) - t_plastic_strain_r(i, j);
      t_tau_jump = t_tau_l - t_tau_r;

      ++t_plastic_strain_jump;
      ++t_tau_jump;
      ++t_plastic_strain_l;
      ++t_plastic_strain_r;
      ++t_tau_l;
      ++t_tau_r;
      ++t_w;
    }
  }

  MoFEMFunctionReturn(0);
}

OpCalculateJumpOnSkeleton::OpCalculateJumpOnSkeleton(
    const std::string row_field_name, const std::string col_field_name,
    boost::shared_ptr<CommonData> common_data_ptr,
    boost::shared_ptr<DomainSideEle> side_op_fe)
    : SkeletonEleOp(row_field_name, col_field_name, SkeletonEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr), sideOpFe(side_op_fe) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  doEntities[MBVERTEX] = true;
}

MoFEMErrorCode
OpCalculateJumpOnSkeleton::doWork(int row_side, int col_side,
                                  EntityType row_type, EntityType col_type,
                                  DataForcesAndSourcesCore::EntData &row_data,
                                  DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "not implemented");

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

static inline FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 2, 2>
get_nf(VectorDouble &nf, FTensor::Number<2>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 2, 2>{&nf[0], &nf[1],
                                                               &nf[1], &nf[2]};
}

static inline FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 3>
get_nf(VectorDouble &nf, FTensor::Number<3>) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 3>{
      &nf[0], &nf[1], &nf[2], &nf[1], &nf[3], &nf[4], &nf[2], &nf[4], &nf[5]};
}

MoFEMErrorCode OpCalculatePlasticFlowPenalty_Rhs::doWork(int side,
                                                         EntityType type,
                                                         EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  // if (side == 0) {
  if (true) {
    EntityHandle ent = getFEEntityHandle();

    auto t_w = getFTensor0IntegrationWeight();
    auto tw_test = getCoordsAtGaussPts();
    auto t_base = data.getFTensor0N();

    auto &data_map = commonDataPtr->plasticDataStrainSideMap;
    // shape funcs
    auto &ep_N_l = data_map.at(LEFT_SIDE)->getN();
    auto &ep_N_r = data_map.at(RIGHT_SIDE)->getN();
    auto &vec_ep_Idx_l = data_map.at(LEFT_SIDE)->getIndices();
    auto &vec_ep_Idx_r = data_map.at(RIGHT_SIDE)->getIndices();

    auto t_ep_base_l = FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
        &*ep_N_l.data().begin());
    auto t_ep_base_r = FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
        &*ep_N_r.data().begin());

    // const size_t nb_dofs = data.getIndices().size();
    const size_t nb_dofs = vec_ep_Idx_l.size();
    VectorDouble nf_ep_l(vec_ep_Idx_l.size(), false);
    nf_ep_l.clear();
    VectorDouble nf_ep_r(vec_ep_Idx_r.size(), false);
    nf_ep_r.clear();

    // jump data
    const double penalty_coefficient = penalty;
    auto t_X_dot_n = getFTensor0FromVec(*commonDataPtr->velocityDotNormalPtr);
    auto t_plastic_strain_jump = getFTensor2SymmetricFromMat<SPACE_DIM>(
        *commonDataPtr->plasticStrainJumpPtr);

    // const size_t nb_integration_pts = data.getN().size1();
    // const size_t nb_base_functions = data.getN().size2();
    const size_t nb_integration_pts = getGaussPts().size2();
    const size_t nb_base_functions = ep_N_l.size2();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha =
          getMeasure() * t_w * t_X_dot_n * t_X_dot_n * penalty_coefficient;

      auto t_nf_ep_l = get_nf(nf_ep_l, FTensor::Number<SPACE_DIM>());
      auto t_nf_ep_r = get_nf(nf_ep_r, FTensor::Number<SPACE_DIM>());

      size_t bb = 0;
      for (; bb != nb_dofs / size_symm; ++bb) {
        t_nf_ep_r(i, j) += t_ep_base_l * t_plastic_strain_jump(i, j) * alpha;
        t_nf_ep_l(i, j) -= t_ep_base_r * t_plastic_strain_jump(i, j) * alpha;

        ++t_nf_ep_l;
        ++t_nf_ep_r;
        ++t_ep_base_l;
        ++t_ep_base_r;
        ++t_base;
      }

      for (; bb < nb_base_functions; ++bb) {
        ++t_ep_base_l;
        ++t_ep_base_r;
        ++t_base;
      }

      ++t_w;
      ++t_X_dot_n;
      ++t_plastic_strain_jump;
    }

    CHKERR ::VecSetValues(getKSPf(), vec_ep_Idx_r.size(),
                          &*vec_ep_Idx_r.begin(), &*nf_ep_r.data().begin(),
                          ADD_VALUES);
    CHKERR ::VecSetValues(getKSPf(), vec_ep_Idx_l.size(),
                          &*vec_ep_Idx_l.begin(), &*nf_ep_l.data().begin(),
                          ADD_VALUES);
  }
  MoFEMFunctionReturn(0);
}

OpCalculateConstraintPenalty_Rhs::OpCalculateConstraintPenalty_Rhs(
    const std::string field_name, boost::shared_ptr<CommonData> common_data_ptr)
    : AssemblySkeletonEleOp(field_name, field_name,
                            AssemblySkeletonEleOp::OPROW),
      commonDataPtr(common_data_ptr) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  // if constexpr (SPACE_DIM == 3)
  //   doEntities[MBTRI] = doEntities[MBQUAD] = true;
  // else
  //   doEntities[MBEDGE] = true;
  doEntities[MBVERTEX] = true;
}

MoFEMErrorCode OpCalculateConstraintPenalty_Rhs::doWork(int side,
                                                        EntityType type,
                                                        EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  if (true) {
    EntityHandle ent = getFEEntityHandle();
    // auto t_w = getFTensor0IntegrationWeight();
    // auto &gg_mat = commonDataPtr->domainGaussPts;
    // auto t_w = FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
    //     &(gg_mat(gg_mat.size1() - 1, 0)));
    auto t_w = getFTensor0IntegrationWeight();
    auto &data_map = commonDataPtr->plasticDataTauSideMap;

    // shape funcs
    auto &tau_N_l = data_map.at(LEFT_SIDE)->getN();
    auto &tau_N_r = data_map.at(RIGHT_SIDE)->getN();
    auto &vec_tau_Idx_l = data_map.at(LEFT_SIDE)->getIndices();
    auto &vec_tau_Idx_r = data_map.at(RIGHT_SIDE)->getIndices();

    auto t_tau_base_l = FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
        &*tau_N_l.data().begin());
    auto t_tau_base_r = FTensor::Tensor0<FTensor::PackPtr<double *, 1>>(
        &*tau_N_r.data().begin());

    // const size_t nb_dofs = data.getIndices().size();
    const size_t nb_dofs = vec_tau_Idx_l.size();
    VectorDouble nf_tau_l(vec_tau_Idx_l.size(), false);
    nf_tau_l.clear();
    VectorDouble nf_tau_r(vec_tau_Idx_r.size(), false);
    nf_tau_r.clear();

    // jump data
    const double penalty_coefficient = penalty;
    auto t_X_dot_n = getFTensor0FromVec(*commonDataPtr->velocityDotNormalPtr);
    auto t_tau_jump = getFTensor0FromVec(*commonDataPtr->plasticTauJumpPtr);

    // const size_t nb_integration_pts2 = data.getN().size1();
    const size_t nb_integration_pts = getGaussPts().size2();
    // const size_t nb_base_functions = data.getN().size2();
    // const size_t nb_integration_pts = tau_N_l.size1();
    const size_t nb_base_functions = tau_N_l.size2();
    auto t_base = data.getFTensor0N();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha =
          getMeasure() * t_w * t_X_dot_n * t_X_dot_n * penalty_coefficient;

      // assemble tau
      size_t bb = 0;
      for (; bb != nb_dofs; ++bb) {
        nf_tau_r[bb] += t_tau_base_l * t_tau_jump * alpha;
        nf_tau_l[bb] -= t_tau_base_r * t_tau_jump * alpha;
        ++t_tau_base_l;
        ++t_tau_base_r;
      }
      for (; bb < nb_base_functions; ++bb) {
        ++t_tau_base_l;
        ++t_tau_base_r;
      }

      ++t_w;
      ++t_X_dot_n;
      ++t_tau_jump;
    }

    CHKERR ::VecSetValues(getKSPf(), vec_tau_Idx_l.size(), &vec_tau_Idx_l[0],
                          &nf_tau_l[0], ADD_VALUES);
    CHKERR ::VecSetValues(getKSPf(), vec_tau_Idx_r.size(), &vec_tau_Idx_r[0],
                          &nf_tau_r[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

struct OpCalculatePlasticFlowPenaltyLhs_dEP : public AssemblyDomainEleOp {
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
    : AssemblyDomainEleOp(row_field_name, col_field_name,
                          DomainEleOp::OPROWCOL),
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
  SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "not implemented");
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode OpCalculateConstraintPenaltyLhs_dTAU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;
  SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "not implemented");
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

  auto t_plastic_strain_jump = getFTensor2SymmetricFromMat<SPACE_DIM>(
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

    t_plastic_strain_jump(i, j) *= t_X_dot_n;
    CHKERR set_tag(th_strain_jump, gg, set_matrix_3d(t_plastic_strain_jump));

    ++t_coords;
    ++t_X_dot_n;
    ++t_omega;
    ++t_plastic_strain_jump;
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