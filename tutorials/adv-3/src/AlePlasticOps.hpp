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

  map<int, MatrixDouble> strainSideMap;
  map<int, VectorDouble> tauSideMap;

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

      t_omega(i) = Omega(i, j) * t_coords(j);
      t_X_dot_n =
          t_omega(j) * t_normal(j);
      // t_X_dot_n = 1;
      
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
    int nb_in_loop = getFEMethod()->nInTheLoop;

    int sense = 0;

    auto &mat_ep = commonDataPtr->strainSideMap[nb_in_loop];
    commonDataPtr->indicesRowStrainSideMap[nb_in_loop] = data.getIndices();
    commonDataPtr->rowBaseSideMap[nb_in_loop] = data.getN();

    constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
    mat_ep.resize(size_symm, nb_gauss_pts, false);
    mat_ep.clear();

    auto values_at_gauss_pts = getFTensor2SymmetricFromMat<SPACE_DIM>(mat_ep);
    for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {
      auto field_data = data.getFTensor2SymmetricFieldData<SPACE_DIM>();
      // cout << "field data: " << field_data << endl;
      size_t bb = 0;
      for (; bb != nb_dofs / size_symm; ++bb) {
        values_at_gauss_pts(i, j) += field_data(i, j) * base_function;
        ++field_data;
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

  const size_t nb_dofs = data.getIndices().size();
  // data.getFieldData().size()
  if (nb_dofs) {
    int nb_gauss_pts = getGaussPts().size2();
    int nb_in_loop = getFEMethod()->nInTheLoop;

    auto &vec_tau = commonDataPtr->tauSideMap[nb_in_loop];
    commonDataPtr->indicesRowTauSideMap[nb_in_loop] = data.getIndices();
    commonDataPtr->rowBaseSideMap[nb_in_loop] = data.getN();

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
  doEntities[MBVERTEX] = true;
}

MoFEMErrorCode OpCalculateJumpOnSkeleton::doWork(int side, EntityType type,
                                                   EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Index<'i', SPACE_DIM> i;
  FTensor::Index<'j', SPACE_DIM> j;

  EntityHandle ent = getFEEntityHandle();
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;
  const size_t nb_dofs = data.getIndices().size();
  // if (side == 0) {
  if (nb_dofs) {

    const size_t nb_integration_pts = getGaussPts().size2();

    commonDataPtr->strainSideMap.clear();
    commonDataPtr->tauSideMap.clear();

    CHKERR loopSide("dFE",
                    dynamic_cast<MoFEM::ForcesAndSourcesCore *>(sideOpFe.get()),
                    SPACE_DIM);
    auto &ep_data_map = commonDataPtr->strainSideMap;
    auto &tau_data_map = commonDataPtr->tauSideMap;

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

    auto t_plastic_strain_l =
        getFTensor2SymmetricFromMat<SPACE_DIM>(ep_data_map.at(LEFT_SIDE));
    auto t_plastic_strain_r =
        getFTensor2SymmetricFromMat<SPACE_DIM>(ep_data_map.at(RIGHT_SIDE));

    auto t_tau_l = getFTensor0FromVec(tau_data_map.at(LEFT_SIDE));
    auto t_tau_r = getFTensor0FromVec(tau_data_map.at(RIGHT_SIDE));

    // fl_L * jump
    // -fl_R * jump

    // field values
    commonDataPtr->plasticTauJumpPtr->resize(nb_integration_pts, false);
    auto t_tau_jump = getFTensor0FromVec(*commonDataPtr->plasticTauJumpPtr);
    commonDataPtr->plasticStrainJumpPtr->resize(size_symm, nb_integration_pts,
                                                false);
    auto t_plastic_strain_jump = getFTensor2SymmetricFromMat<SPACE_DIM>(
        *commonDataPtr->plasticStrainJumpPtr);

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {

      t_plastic_strain_jump(i, j) =
          t_plastic_strain_l(i, j) - t_plastic_strain_r(i, j);

      t_tau_jump = t_tau_l - t_tau_r;

      ++t_tau_jump;
      ++t_tau_l;
      ++t_tau_r;

      ++t_plastic_strain_jump;
      ++t_plastic_strain_l;
      ++t_plastic_strain_r;
    }
  }

  MoFEMFunctionReturn(0);
}

OpDomainSideGetColData::OpDomainSideGetColData(
    const std::string row_name, const std::string col_name,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainSideEleOp(row_name, col_name, DomainSideEleOp::OPROWCOL),
      commonDataPtr(common_data_ptr) {
}

MoFEMErrorCode
OpDomainSideGetColData::doWork(int row_side, int col_side, EntityType row_type,
                               EntityType col_type,
                               DataForcesAndSourcesCore::EntData &row_data,
                               DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  const size_t nb_dofs = row_data.getIndices().size();
  const int nb_in_loop = getFEMethod()->nInTheLoop;

  EntityType ent_types[] = {MBTRI, MBQUAD};
  if constexpr (SPACE_DIM == 3) {
    ent_types[0] = MBHEX;
    ent_types[1] = MBTET;
  }

  if (row_type == ent_types[0] || row_type == ent_types[1])
    if (col_type == ent_types[0] || col_type == ent_types[1]) {
      if (colFieldName == "TAU") {
        commonDataPtr->indicesRowTauSideMap[nb_in_loop] = row_data.getIndices();
        commonDataPtr->indicesColTauSideMap[nb_in_loop] = col_data.getIndices();
        commonDataPtr->rowBaseSideMap[nb_in_loop] = row_data.getN();
        commonDataPtr->colBaseSideMap[nb_in_loop] = col_data.getN();
      } else if (colFieldName == "EP") {
        commonDataPtr->indicesRowStrainSideMap[nb_in_loop] =
            row_data.getIndices();
        commonDataPtr->indicesColStrainSideMap[nb_in_loop] =
            col_data.getIndices();
        commonDataPtr->rowBaseSideMap[nb_in_loop] = row_data.getN();
        commonDataPtr->colBaseSideMap[nb_in_loop] = col_data.getN();
      } else {
        MOFEM_LOG("WORLD", Sev::error) << "wrong side field";
      }
    }
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
  constexpr auto size_symm = (SPACE_DIM * (SPACE_DIM + 1)) / 2;

  // if (side == 0) {
  if (true) {
    EntityHandle ent = getFEEntityHandle();

    auto t_w = getFTensor0IntegrationWeight();

    // shape funcs
    auto &idx_rmap = commonDataPtr->indicesRowStrainSideMap;
    auto t_base_l = get_ntensor(commonDataPtr->rowBaseSideMap.at(LEFT_SIDE));
    auto t_base_r = get_ntensor(commonDataPtr->rowBaseSideMap.at(RIGHT_SIDE));
    const size_t nb_dofs = idx_rmap.at(LEFT_SIDE).size();

    VectorDouble nf_l(nb_dofs, false);
    nf_l.clear();
    VectorDouble nf_r(nb_dofs, false);
    nf_r.clear();

    // jump data
    const double penalty_coefficient = penalty;
    auto t_X_dot_n = getFTensor0FromVec(*commonDataPtr->velocityDotNormalPtr);
    auto t_plastic_strain_jump = getFTensor2SymmetricFromMat<SPACE_DIM>(
        *commonDataPtr->plasticStrainJumpPtr);

    auto &ep_mat = *commonDataPtr->plasticStrainJumpPtr;
    auto Is = commonDataPtr->Is;

    const size_t nb_integration_pts = getGaussPts().size2();
    const size_t nb_base_functions =
        commonDataPtr->rowBaseSideMap[LEFT_SIDE].size2();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha =
          getMeasure() * t_w * t_X_dot_n * t_X_dot_n * penalty_coefficient;

      auto t_nf_l = get_nf(nf_l, FTensor::Number<SPACE_DIM>());
      auto t_nf_r = get_nf(nf_r, FTensor::Number<SPACE_DIM>());

      size_t bb = 0;
      for (; bb != nb_dofs / size_symm; ++bb) {
        // for (int dd = 0; dd != size_symm; ++dd) {
        //   nf_r[size_symm * bb + dd] += t_base_l * ep_mat(dd, gg) * alpha;
        //   nf_l[size_symm * bb + dd] -= t_base_r * ep_mat(dd, gg) * alpha;
        // }
        // t_nf_r(i, j) +=
        //     (t_base_l * alpha) * Is(i, j, k, l) * t_plastic_strain_jump(k, l);
        // t_nf_l(i, j) -=
        //     (t_base_r * alpha) * Is(i, j, k, l) * t_plastic_strain_jump(k, l);
        // FIXME: we are penalising off diagonals more than the rest
        t_nf_r(i, j) += (t_base_l * alpha) * t_plastic_strain_jump(i, j);
        t_nf_l(i, j) -= (t_base_r * alpha) * t_plastic_strain_jump(i, j);

        ++t_nf_l;
        ++t_nf_r;
        ++t_base_l;
        ++t_base_r;
      }

      for (; bb < nb_base_functions; ++bb) {
        ++t_base_l;
        ++t_base_r;
      }

      ++t_w;
      ++t_X_dot_n;
      ++t_plastic_strain_jump;
    }

    CHKERR VecSetValues(getKSPf(), idx_rmap.at(LEFT_SIDE).size(),
                        &*idx_rmap.at(LEFT_SIDE).begin(), &*nf_l.data().begin(),
                        ADD_VALUES);
    CHKERR VecSetValues(getKSPf(), idx_rmap.at(RIGHT_SIDE).size(),
                        &*idx_rmap.at(RIGHT_SIDE).begin(),
                        &*nf_r.data().begin(), ADD_VALUES);
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

    // shape funcs
    auto t_base_l = get_ntensor(commonDataPtr->rowBaseSideMap.at(LEFT_SIDE));
    auto t_base_r = get_ntensor(commonDataPtr->rowBaseSideMap.at(RIGHT_SIDE));
    auto &idx_rmap = commonDataPtr->indicesRowTauSideMap;

    const size_t nb_dofs = idx_rmap.at(LEFT_SIDE).size();

    VectorDouble nf_l(nb_dofs, false);
    nf_l.clear();
    VectorDouble nf_r(nb_dofs, false);
    nf_r.clear();

    // jump data
    const double penalty_coefficient = penalty;
    auto t_X_dot_n = getFTensor0FromVec(*commonDataPtr->velocityDotNormalPtr);
    auto t_tau_jump = getFTensor0FromVec(*commonDataPtr->plasticTauJumpPtr);

    const size_t nb_integration_pts = getGaussPts().size2();
    const size_t nb_base_functions =
        commonDataPtr->rowBaseSideMap[LEFT_SIDE].size2();

    for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
      const double alpha =
          getMeasure() * t_w * t_X_dot_n * t_X_dot_n * penalty_coefficient;

      // assemble tau
      size_t bb = 0;
      for (; bb != nb_dofs; ++bb) {
        nf_r[bb] += t_base_l * t_tau_jump * alpha;
        nf_l[bb] -= t_base_r * t_tau_jump * alpha;
        ++t_base_l;
        ++t_base_r;
      }
      for (; bb < nb_base_functions; ++bb) {
        ++t_base_l;
        ++t_base_r;
      }

      ++t_w;
      ++t_X_dot_n;
      ++t_tau_jump;
    }

    CHKERR VecSetValues(getKSPf(), idx_rmap.at(LEFT_SIDE).size(),
                        &*idx_rmap.at(LEFT_SIDE).begin(), &*nf_l.data().begin(),
                        ADD_VALUES);
    CHKERR VecSetValues(getKSPf(), idx_rmap.at(RIGHT_SIDE).size(),
                        &*idx_rmap.at(RIGHT_SIDE).begin(),
                        &*nf_r.data().begin(), ADD_VALUES);
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

  EntityHandle ent = getFEEntityHandle();
  auto t_w = getFTensor0IntegrationWeight();

  array<map<int, VectorInt>, 2> idx_map;
  idx_map[ROW] = commonDataPtr->indicesRowStrainSideMap;
  idx_map[COL] = commonDataPtr->indicesColStrainSideMap;

  // shape funcs
  auto t_row_base_l = get_ntensor(commonDataPtr->rowBaseSideMap.at(LEFT_SIDE));
  auto t_row_base_r = get_ntensor(commonDataPtr->rowBaseSideMap.at(RIGHT_SIDE));

  const size_t nb_rows = idx_map[ROW].at(LEFT_SIDE).size();
  const size_t nb_cols = idx_map[COL].at(LEFT_SIDE).size();

  if (!nb_cols)
    MoFEMFunctionReturnHot(0);

  std::array<std::array<MatrixDouble, 2>, 2> locMat;
  for (auto side0 : {LEFT_SIDE, RIGHT_SIDE})
    for (auto side1 : {LEFT_SIDE, RIGHT_SIDE}) {
      locMat[side0][side1].resize(nb_rows, nb_cols, false);
      locMat[side0][side1].clear();
    }

  const size_t nb_integration_pts = getGaussPts().size2();
  const size_t nb_row_base_functions =
      commonDataPtr->rowBaseSideMap.at(LEFT_SIDE).size2();

  constexpr auto t_kd = FTensor::Kronecker_Delta<int>();

  // jump data
  const double penalty_coefficient = penalty;
  auto t_diff_plastic_strain = diff_tensor();
  auto t_X_dot_n = getFTensor0FromVec(*commonDataPtr->velocityDotNormalPtr);
  auto t_plastic_strain_jump = getFTensor2SymmetricFromMat<SPACE_DIM>(
      *commonDataPtr->plasticStrainJumpPtr);
  auto Is = commonDataPtr->Is;

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha =
        getMeasure() * t_w * t_X_dot_n * t_X_dot_n * penalty_coefficient;

    size_t rr = 0;
    for (; rr != nb_rows / size_symm; ++rr) {

      auto t_mat_rr = get_mat_tensor_sym_dtensor_sym(
          rr, locMat[RIGHT_SIDE][RIGHT_SIDE], FTensor::Number<SPACE_DIM>());
      auto t_mat_rl = get_mat_tensor_sym_dtensor_sym(
          rr, locMat[RIGHT_SIDE][LEFT_SIDE], FTensor::Number<SPACE_DIM>());
      auto t_mat_lr = get_mat_tensor_sym_dtensor_sym(
          rr, locMat[LEFT_SIDE][RIGHT_SIDE], FTensor::Number<SPACE_DIM>());
      auto t_mat_ll = get_mat_tensor_sym_dtensor_sym(
          rr, locMat[LEFT_SIDE][LEFT_SIDE], FTensor::Number<SPACE_DIM>());

      auto t_col_base_l =
          get_ntensor(commonDataPtr->colBaseSideMap.at(LEFT_SIDE), gg, 0);
      auto t_col_base_r =
          get_ntensor(commonDataPtr->colBaseSideMap.at(RIGHT_SIDE), gg, 0);

      for (size_t cc = 0; cc != nb_cols / size_symm; ++cc) {

        t_mat_rr(i, j, k, l) -= (alpha * t_row_base_l * t_col_base_r) *
                                Is(i, j, m, n) *
                                t_diff_plastic_strain(m, n, k, l);
        t_mat_rl(i, j, k, l) += (alpha * t_row_base_l * t_col_base_l) *
                                Is(i, j, m, n) *
                                t_diff_plastic_strain(m, n, k, l);
        t_mat_lr(i, j, k, l) += (alpha * t_row_base_r * t_col_base_r) *
                                Is(i, j, m, n) *
                                t_diff_plastic_strain(m, n, k, l);
        t_mat_ll(i, j, k, l) -= (alpha * t_row_base_r * t_col_base_l) *
                                Is(i, j, m, n) *
                                t_diff_plastic_strain(m, n, k, l);

        ++t_mat_rr;
        ++t_mat_rl;
        ++t_mat_lr;
        ++t_mat_ll;

        ++t_col_base_r;
        ++t_col_base_l;
      }

        ++t_row_base_r;
        ++t_row_base_l;
      }
    for (; rr < nb_row_base_functions; ++rr) {
      ++t_row_base_r;
      ++t_row_base_l;
    }

    ++t_w;
    ++t_plastic_strain_jump;
    ++t_X_dot_n;
  }

  for (auto s0 : {LEFT_SIDE, RIGHT_SIDE})
    for (auto s1 : {LEFT_SIDE, RIGHT_SIDE})
      CHKERR ::MatSetValues(
          getKSPB(), idx_map[ROW].at(s0).size(), &*idx_map[ROW].at(s0).begin(),
          idx_map[COL].at(s1).size(), &*idx_map[COL].at(s1).begin(),
          &*locMat[s0][s1].data().begin(), ADD_VALUES);
 
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode OpCalculateConstraintPenaltyLhs_dTAU::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    DataForcesAndSourcesCore::EntData &row_data,
    DataForcesAndSourcesCore::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_type != MBVERTEX || col_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);
  
  EntityHandle ent = getFEEntityHandle();
  auto t_w = getFTensor0IntegrationWeight();

  array<map<int, VectorInt>, 2> idx_map;
  idx_map[ROW] = commonDataPtr->indicesRowTauSideMap;
  idx_map[COL] = commonDataPtr->indicesColTauSideMap;

  // shape funcs
  auto t_row_base_l = get_ntensor(commonDataPtr->rowBaseSideMap.at(LEFT_SIDE));
  auto t_row_base_r = get_ntensor(commonDataPtr->rowBaseSideMap.at(RIGHT_SIDE));

  const size_t nb_rows = idx_map[ROW].at(LEFT_SIDE).size();
  const size_t nb_cols = idx_map[COL].at(LEFT_SIDE).size();

  if (!nb_cols)
    MoFEMFunctionReturnHot(0);

  std::array<std::array<MatrixDouble, 2>, 2> locMat;
  for (auto side0 : {LEFT_SIDE, RIGHT_SIDE})
    for (auto side1 : {LEFT_SIDE, RIGHT_SIDE}) {
      locMat[side0][side1].resize(nb_rows, nb_cols, false);
      locMat[side0][side1].clear();
    }

  const size_t nb_integration_pts = getGaussPts().size2();
  const size_t nb_row_base_functions =
      commonDataPtr->rowBaseSideMap.at(LEFT_SIDE).size2();

  // jump data
  const double penalty_coefficient = penalty;

  auto t_X_dot_n = getFTensor0FromVec(*commonDataPtr->velocityDotNormalPtr);
  auto t_tau_jump = getFTensor0FromVec(*commonDataPtr->plasticTauJumpPtr);

  for (size_t gg = 0; gg != nb_integration_pts; ++gg) {
    const double alpha =
        getMeasure() * t_w * t_X_dot_n * t_X_dot_n * penalty_coefficient;

    auto t_mat_rr = locMat[RIGHT_SIDE][RIGHT_SIDE].data().begin();
    auto t_mat_rl = locMat[RIGHT_SIDE][LEFT_SIDE].data().begin();
    auto t_mat_lr = locMat[LEFT_SIDE][RIGHT_SIDE].data().begin();
    auto t_mat_ll = locMat[LEFT_SIDE][LEFT_SIDE].data().begin();
    
    size_t rr = 0;
    for (; rr != nb_rows; ++rr) {

      auto t_col_base_l =
          get_ntensor(commonDataPtr->colBaseSideMap.at(LEFT_SIDE), gg, 0);
      auto t_col_base_r =
          get_ntensor(commonDataPtr->colBaseSideMap.at(RIGHT_SIDE), gg, 0);

      for (size_t cc = 0; cc != nb_cols; ++cc) {

        *t_mat_rr -= alpha * t_row_base_l * t_col_base_r;
        *t_mat_rl += alpha * t_row_base_l * t_col_base_l;
        *t_mat_lr += alpha * t_row_base_r * t_col_base_r;
        *t_mat_ll -= alpha * t_row_base_r * t_col_base_l;

        ++t_mat_rr;
        ++t_mat_rl;
        ++t_mat_lr;
        ++t_mat_ll;
        ++t_col_base_r;
        ++t_col_base_l;
      }

      ++t_row_base_r;
      ++t_row_base_l;
    }
    for (; rr < nb_row_base_functions; ++rr) {
      ++t_row_base_r;
      ++t_row_base_l;
    }

    ++t_w;
    ++t_tau_jump;
    ++t_X_dot_n;
  }

  for (auto s0 : {LEFT_SIDE, RIGHT_SIDE})
    for (auto s1 : {LEFT_SIDE, RIGHT_SIDE}) 
      CHKERR ::MatSetValues(
          getKSPB(), idx_map[ROW].at(s0).size(), &*idx_map[ROW].at(s0).begin(),
          idx_map[COL].at(s1).size(), &*idx_map[COL].at(s1).begin(),
          &*locMat[s0][s1].data().begin(), ADD_VALUES);

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

#ifndef NDEBUG
    CHKERR set_tag(th_normal, gg, set_vector(t_normal));
#endif // NDEBUG

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
    // FIXME: testing
    // double max_velocity = 0.04;
    // if (petsc_time >= max_load_time * 2)
    //   angular_velocity[2] =
    //       std::min(abs(max_velocity), (petsc_time - max_load_time * 2.));

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