/**
 * \file PostProcContact.hpp
 *
 *
 * @copyright Copyright (c) 2023
 */

namespace ContactOps {

template <int DIM> struct PostProcEleByDim;

template <> struct PostProcEleByDim<2> {
  using PostProcEleDomain = PostProcBrokenMeshInMoabBaseCont<DomainEle>;
  using PostProcEleBdy = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<2>::FaceSideEle;
};

template <> struct PostProcEleByDim<3> {
  using PostProcEleDomain = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;
  using PostProcEleBdy = PostProcBrokenMeshInMoabBaseCont<BoundaryEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<3>::FaceSideEle;
};

using PostProcEleDomain = PostProcEleByDim<SPACE_DIM>::PostProcEleDomain;
using PostProcEleBdy = PostProcEleByDim<SPACE_DIM>::PostProcEleBdy;

struct OpAssembleTraction : public BoundaryEleOp {
  OpAssembleTraction(boost::shared_ptr<CommonData> common_data_ptr,
                     SmartPetscObj<Vec> total_traction);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  SmartPetscObj<Vec> totalTraction;
};

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), uXScatter(ux_scatter), uYScatter(uy_scatter),
        uZScatter(uz_scatter), moabVertex(mbVertexPostproc), sTEP(0) {

    MoFEM::Interface *m_field_ptr;
    CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);
    totalTraction =
        createSmartVectorMPI(m_field_ptr->get_comm(),
                             (m_field_ptr->get_comm_rank() == 0) ? 3 : 0, 3);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto get_post_proc_domain_fe = [&]() {
      auto post_proc_fe =
          boost::make_shared<PostProcEleDomain>(*m_field_ptr, postProcMesh);
      auto &pip = post_proc_fe->getOpPtrVector();

      auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();
      auto henky_common_data_ptr = boost::make_shared<HenckyOps::CommonData>();
      henky_common_data_ptr->matGradPtr = common_data_ptr->mGradPtr();
      henky_common_data_ptr->matDPtr = common_data_ptr->mDPtr();

      auto push_domain_ops = [&](auto &pip) {
        CHK_THROW_MESSAGE(
            (AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1, HDIV})),
            "Apply base transform");
        CHK_THROW_MESSAGE(
            ContactOps::addMatBlockOps(*m_field_ptr, pip, "U", "MAT_ELASTIC",
                                       common_data_ptr->mDPtr(), Sev::inform),
            "Set block data");
        pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
            "U", common_data_ptr->mGradPtr()));
        pip.push_back(
            new OpCalculateEigenVals<SPACE_DIM>("U", henky_common_data_ptr));
        pip.push_back(
            new OpCalculateLogC<SPACE_DIM>("U", henky_common_data_ptr));
        pip.push_back(
            new OpCalculateLogC_dC<SPACE_DIM>("U", henky_common_data_ptr));
        pip.push_back(
            new OpCalculateHenckyStress<SPACE_DIM>("U", henky_common_data_ptr));
        pip.push_back(
            new OpCalculatePiolaStress<SPACE_DIM>("U", henky_common_data_ptr));
        pip.push_back(new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
            "SIGMA", common_data_ptr->contactStressPtr()));
      };

      // Evaluate domain on side element
      if constexpr (SPACE_DIM == 3) {
        // create OP which run element on side
        auto op_loop_side =
            new OpLoopSide<FaceElementForcesAndSourcesCoreOnSide>(
                *m_field_ptr, "dFE", SPACE_DIM);
        // push ops to side element, through op_loop_side operator
        push_domain_ops(op_loop_side->getOpPtrVector());
        pip.push_back(op_loop_side);
        // evaluate traction
        pip.push_back(new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
            "SIGMA", common_data_ptr->contactTractionPtr()));
      } else {
        push_domain_ops(pip);
      }

      auto u_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

      post_proc_fe->getOpPtrVector().push_back(

          new OpPPMap(

              post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

              {},

              {{"U", u_ptr},

               // Note: post-process tractions in 3d, i.e. when mesh is
               // post-process on skin
               {"t", (SPACE_DIM == 3) ? common_data_ptr->contactTractionPtr()
                                      : nullptr}},

              {

                  {"SIGMA", common_data_ptr->contactStressPtr()},

                  {"G", common_data_ptr->mGradPtr()},

                  {"P2", henky_common_data_ptr->getMatFirstPiolaStress()}

              },

              {}

              )

      );

      return post_proc_fe;
    };

    auto get_post_proc_bdy_fe = [&]() {
      auto post_proc_fe =
          boost::make_shared<PostProcEleBdy>(*m_field_ptr, postProcMesh);
      auto &pip = post_proc_fe->getOpPtrVector();

      auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();

      CHK_THROW_MESSAGE(
          (AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(pip, {HDIV})),
          "Apply transform");
      pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
          "U", common_data_ptr->contactDispPtr()));
      pip.push_back(new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
          "SIGMA", common_data_ptr->contactTractionPtr()));

      pip.push_back(

          new OpPPMap(

              post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

              {},

              {{"U", common_data_ptr->contactDispPtr()},
               {"t", common_data_ptr->contactTractionPtr()}},

              {},

              {}

              )

      );

      return post_proc_fe;
    };

    auto get_integrate_traction = [&]() {
      auto integrate_traction = boost::make_shared<BoundaryEle>(*m_field_ptr);
      auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();
      CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
          integrate_traction->getOpPtrVector(), {HDIV});
      integrate_traction->getOpPtrVector().push_back(
          new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
              "SIGMA", common_data_ptr->contactTractionPtr()));
      integrate_traction->getOpPtrVector().push_back(
          new OpAssembleTraction(common_data_ptr, totalTraction));
      integrate_traction->getRuleHook = [](int, int, int approx_order) {
        return 2 * approx_order;
      };
      return integrate_traction;
    };

    postProcDomainFe = get_post_proc_domain_fe();
    if constexpr (SPACE_DIM == 2)
      postProcBdyFe = get_post_proc_bdy_fe();
    integrateTraction = get_integrate_traction();
  }

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    MoFEM::Interface *m_field_ptr;
    CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);

    auto post_proc = [&]() {
      MoFEMFunctionBegin;

      auto post_proc_begin =
          boost::make_shared<PostProcBrokenMeshInMoabBaseBegin>(*m_field_ptr,
                                                                postProcMesh);
      auto post_proc_end = boost::make_shared<PostProcBrokenMeshInMoabBaseEnd>(
          *m_field_ptr, postProcMesh);

      CHKERR DMoFEMPreProcessFiniteElements(dM, post_proc_begin->getFEMethod());
      if (!postProcBdyFe) {
        CHKERR DMoFEMLoopFiniteElements(dM, "bFE", postProcDomainFe);
      } else {
        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcDomainFe);
        CHKERR DMoFEMLoopFiniteElements(dM, "bFE", postProcBdyFe);
      }
      CHKERR DMoFEMPostProcessFiniteElements(dM, post_proc_end->getFEMethod());

      CHKERR post_proc_end->writeFile(
          "out_contact_" + boost::lexical_cast<std::string>(sTEP) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    auto calculate_traction = [&] {
      MoFEMFunctionBegin;
      CHKERR VecZeroEntries(totalTraction);
      CHKERR DMoFEMLoopFiniteElements(dM, "bFE", integrateTraction);
      CHKERR VecAssemblyBegin(totalTraction);
      CHKERR VecAssemblyEnd(totalTraction);
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
      MOFEM_LOG_C("CONTACT", Sev::inform, "%s time %3.4e min %3.4e max %3.4e",
                  msg.c_str(), ts_t, min, max);
      MoFEMFunctionReturn(0);
    };

    auto print_traction = [&](const std::string msg) {
      MoFEMFunctionBegin;
      MoFEM::Interface *m_field_ptr;
      CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);
      if (!m_field_ptr->get_comm_rank()) {
        const double *t_ptr;
        CHKERR VecGetArrayRead(totalTraction, &t_ptr);
        MOFEM_LOG_C("CONTACT", Sev::inform, "%s time %3.4e %3.4e %3.4e %3.4e",
                    msg.c_str(), ts_t, t_ptr[0], t_ptr[1], t_ptr[2]);
        CHKERR VecRestoreArrayRead(totalTraction, &t_ptr);
      }
      MoFEMFunctionReturn(0);
    };

    MOFEM_LOG("CONTACT", Sev::inform)
        << "Write file at time " << ts_t << " write step " << sTEP;

    ++sTEP;
    CHKERR post_proc();
    CHKERR calculate_traction();

    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");
    CHKERR print_traction("Force");

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  SmartPetscObj<Vec> totalTraction;

  boost::shared_ptr<moab::Core> postProcMesh = boost::make_shared<moab::Core>();

  boost::shared_ptr<PostProcEleDomain> postProcDomainFe;
  boost::shared_ptr<PostProcEleBdy> postProcBdyFe;
  boost::shared_ptr<PostProcEleByDim<SPACE_DIM>::SideEle> postProcSideFe;

  boost::shared_ptr<BoundaryEle> integrateTraction;

  moab::Core mbVertexPostproc;
  moab::Interface &moabVertex;

  double lastTime;
  double deltaTime;
  int sTEP;
};

OpAssembleTraction::OpAssembleTraction(
    boost::shared_ptr<CommonData> common_data_ptr,
    SmartPetscObj<Vec> total_traction)
    : BoundaryEleOp(NOSPACE, BoundaryEleOp::OPSPACE),
      commonDataPtr(common_data_ptr), totalTraction(total_traction) {}

MoFEMErrorCode OpAssembleTraction::doWork(int side, EntityType type,
                                          EntData &data) {
  MoFEMFunctionBegin;

  FTensor::Tensor1<double, 3> t_sum_t{0., 0., 0.};

  auto t_w = getFTensor0IntegrationWeight();
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactTraction);

  const auto nb_gauss_pts = getGaussPts().size2();
  for (auto gg = 0; gg != nb_gauss_pts; ++gg) {
    const double alpha = t_w * getMeasure();
    t_sum_t(i) += alpha * t_traction(i);
    ++t_w;
    ++t_traction;
  }

  constexpr int ind[] = {0, 1, 2};
  CHKERR VecSetValues(totalTraction, 3, ind, &t_sum_t(0), ADD_VALUES);

  MoFEMFunctionReturn(0);
}

} // namespace ContactOps