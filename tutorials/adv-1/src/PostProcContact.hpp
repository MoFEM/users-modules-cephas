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
using SideEle = PostProcEleByDim<SPACE_DIM>::SideEle;
using PostProcEleBdy = PostProcEleByDim<SPACE_DIM>::PostProcEleBdy;

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), uXScatter(ux_scatter), uYScatter(uy_scatter),
        uZScatter(uz_scatter), moabVertex(mbVertexPostproc), sTEP(0) {

    MoFEM::Interface *m_field_ptr;
    CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    auto push_domain_ops = [&](auto &pip) {
      CHK_THROW_MESSAGE((AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
                            pip, {H1, HDIV}, "GEOMETRY")),
                        "Apply base transform");
      auto henky_common_data_ptr =
          commonDataFactory<SPACE_DIM, GAUSS, DomainEleOp>(
              *m_field_ptr, pip, "U", "MAT_ELASTIC", Sev::inform);
      auto contact_stress_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
          "SIGMA", contact_stress_ptr));
      return std::make_tuple(henky_common_data_ptr, contact_stress_ptr);
    };

    auto push_bdy_ops_traction = [&](auto &pip) {
        // evaluate traction
        auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();
        pip.push_back(new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
            "SIGMA", common_data_ptr->contactTractionPtr()));
        return common_data_ptr;
    };

    auto push_bdy_ops_sdf = [&](auto &pip) {
      // evaluate traction
      auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();
      pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>(
          "U", common_data_ptr->contactDispPtr()));
      using C = ContactIntegrators<BoundaryEleOp>;
      pip.push_back(new typename C::template OpEvaluateSDF<SPACE_DIM, GAUSS>(
          common_data_ptr));
      return common_data_ptr;
    };

    auto get_domain_pip = [&](auto &pip)
        -> boost::ptr_deque<ForcesAndSourcesCore::UserDataOperator> & {
      if constexpr (SPACE_DIM == 3) {
        auto op_loop_side = new OpLoopSide<SideEle>(
            *m_field_ptr, "dFE", SPACE_DIM, Sev::noisy,
            boost::make_shared<
                ForcesAndSourcesCore::UserDataOperator::AdjCache>());
        pip.push_back(op_loop_side);
        return op_loop_side->getOpPtrVector();
      } else {
        return pip;
      }
    };

    auto get_post_proc_domain_fe = [&]() {
      auto post_proc_fe =
          boost::make_shared<PostProcEleDomain>(*m_field_ptr, postProcMesh);
      auto &pip = post_proc_fe->getOpPtrVector();

      auto [henky_common_data_ptr, contact_stress_ptr] =
          push_domain_ops(get_domain_pip(pip));

      auto u_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));
      auto X_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(
          new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", X_ptr));

      pip.push_back(

          new OpPPMap(

              post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

              {},
              {

                  {"U", u_ptr}, {"GEOMETRY", X_ptr}

              },
              {

                  {"SIGMA", contact_stress_ptr},

                  {"G", henky_common_data_ptr->matGradPtr},

                  {"P2", henky_common_data_ptr->getMatFirstPiolaStress()}

              },
              {}

              )

      );

      if (SPACE_DIM == 3) {

        CHK_THROW_MESSAGE((AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
                              pip, {HDIV}, "GEOMETRY")),
                          "Apply transform");
        auto common_data_traction_ptr = push_bdy_ops_traction(pip);
        auto common_data_sdf_ptr = push_bdy_ops_sdf(pip);

        pip.push_back(

            new OpPPMap(

                post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

                {{"SDF", common_data_sdf_ptr->sdfPtr()}},
                
                {

                    {"TRACTION_CONTACT",
                     common_data_traction_ptr->contactTractionPtr()},
                    {"GRAD_SDF", common_data_sdf_ptr->gradSdfPtr()}

                },

                {},

                {{"HESS_SDF", common_data_sdf_ptr->hessSdfPtr()}}

                )

        );
      }

      return post_proc_fe;
    };

    auto get_post_proc_bdy_fe = [&]() {
      auto post_proc_fe =
          boost::make_shared<PostProcEleBdy>(*m_field_ptr, postProcMesh);
      auto &pip = post_proc_fe->getOpPtrVector();

      CHK_THROW_MESSAGE((AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
                            pip, {HDIV}, "GEOMETRY")),
                        "Apply transform");
      auto common_data_traction_ptr = push_bdy_ops_traction(pip);
      auto common_data_sdf_ptr = push_bdy_ops_sdf(pip);

      // create OP which run element on side
      auto op_loop_side = new OpLoopSide<SideEle>(
          *m_field_ptr, "dFE", SPACE_DIM, Sev::noisy,
          boost::make_shared<
              ForcesAndSourcesCore::UserDataOperator::AdjCache>());
      pip.push_back(op_loop_side);

      auto [henky_common_data_ptr, contact_stress_ptr] =
          push_domain_ops(op_loop_side->getOpPtrVector());
      
      auto X_ptr = boost::make_shared<MatrixDouble>();
      pip.push_back(
          new OpCalculateVectorFieldValues<SPACE_DIM>("GEOMETRY", X_ptr));

      pip.push_back(

          new OpPPMap(

              post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

              {{"SDF", common_data_sdf_ptr->sdfPtr()}},

              {{"U", common_data_sdf_ptr->contactDispPtr()},
               {"GEOMETRY", X_ptr},
               {"TRACTION_CONTACT",
                common_data_traction_ptr->contactTractionPtr()},
               {"GRAD_SDF", common_data_sdf_ptr->gradSdfPtr()}

              },

              {},

              {{"HESS_SDF", common_data_sdf_ptr->hessSdfPtr()}}

              )

      );

      return post_proc_fe;
    };

    auto get_integrate_traction = [&]() {
      auto integrate_traction = boost::make_shared<BoundaryEle>(*m_field_ptr);
      auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();
      CHK_THROW_MESSAGE(
          (AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
              integrate_traction->getOpPtrVector(), {HDIV}, "GEOMETRY")),
          "Apply transform");
      // We have to integrate on curved face geometry, thus integration weight
      // have to adjusted.
      integrate_traction->getOpPtrVector().push_back(
          new OpSetHOWeightsOnSubDim<SPACE_DIM>());
      integrate_traction->getRuleHook = [](int, int, int approx_order) {
        return 2 * approx_order + geom_order - 1;
      };

      CHK_THROW_MESSAGE(
          (opFactoryCalculateTraction<SPACE_DIM, GAUSS, BoundaryEleOp>(
              integrate_traction->getOpPtrVector(), "SIGMA")),
          "push operators to calculate traction");

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
        postProcDomainFe->ts_t = this->ts_t; // this here is a Monitor
        CHKERR DMoFEMLoopFiniteElements(dM, "bFE", postProcDomainFe);
      } else {
        postProcDomainFe->ts_t = this->ts_t; // this here is a Monitor
        postProcBdyFe->ts_t = this->ts_t;
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
      CHKERR VecZeroEntries(CommonData::totalTraction);
      CHKERR DMoFEMLoopFiniteElements(dM, "bFE", integrateTraction);
      CHKERR VecAssemblyBegin(CommonData::totalTraction);
      CHKERR VecAssemblyEnd(CommonData::totalTraction);
      MoFEMFunctionReturn(0);
    };

    auto calculate_reactions = [&]() {
      MoFEMFunctionBegin;
      
      auto res = createDMVector(dM);

      auto assemble_domain = [&]() {
        MoFEMFunctionBegin;
        auto fe_rhs = boost::make_shared<DomainEle>(*m_field_ptr);
        auto &pip = fe_rhs->getOpPtrVector();
        fe_rhs->f = res;

        auto integration_rule = [](int, int, int approx_order) {
          return 2 * approx_order + geom_order - 1;
        };
        fe_rhs->getRuleHook = integration_rule;
        CHKERR HenckyOps::opFactoryDomainRhs<SPACE_DIM, PETSC, IT, DomainEleOp>(
            *m_field_ptr, pip, "U", "MAT_ELASTIC", Sev::inform);
        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", fe_rhs);
        MoFEMFunctionReturn(0);
      };

      auto assemble_boundary = [&]() {
        MoFEMFunctionBegin;
        auto fe_rhs = boost::make_shared<BoundaryEle>(*m_field_ptr);
        auto &pip = fe_rhs->getOpPtrVector();
        fe_rhs->f = res;

        auto integration_rule = [](int, int, int approx_order) {
          return 2 * approx_order + geom_order - 1;
        };
        fe_rhs->getRuleHook = integration_rule;

        CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(pip, {HDIV},
                                                                  "GEOMETRY");
        // We have to integrate on curved face geometry, thus integration weight
        // have to adjusted.
        pip.push_back(new OpSetHOWeightsOnSubDim<SPACE_DIM>());

        auto u_disp = boost::make_shared<MatrixDouble>();
        pip.push_back(new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_disp));
        pip.push_back(
            new OpSpringRhs("U", u_disp, [this](double, double, double) {
              return spring_stiffness;
            }));

        CHKERR DMoFEMLoopFiniteElements(dM, "bFE", fe_rhs);

        MoFEMFunctionReturn(0);
      };

      CHKERR assemble_domain();
      CHKERR assemble_boundary();

      auto fe_post_proc_ptr = boost::make_shared<FEMethod>();
      auto get_post_proc_hook_rhs = [this, fe_post_proc_ptr, res,
                                     m_field_ptr]() {
        MoFEMFunctionBegin;
        CHKERR EssentialPreProcReaction<DisplacementCubitBcData>(
            *m_field_ptr, fe_post_proc_ptr, res)();
        MoFEMFunctionReturn(0);
      };
      fe_post_proc_ptr->postProcessHook = get_post_proc_hook_rhs;
      CHKERR DMoFEMPostProcessFiniteElements(dM, fe_post_proc_ptr.get());

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
        CHKERR VecGetArrayRead(CommonData::totalTraction, &t_ptr);
        MOFEM_LOG_C("CONTACT", Sev::inform, "%s time %3.4e %3.4e %3.4e %3.4e",
                    msg.c_str(), ts_t, t_ptr[0], t_ptr[1], t_ptr[2]);
        CHKERR VecRestoreArrayRead(CommonData::totalTraction, &t_ptr);
      }
      MoFEMFunctionReturn(0);
    };

    MOFEM_LOG("CONTACT", Sev::inform)
        << "Write file at time " << ts_t << " write step " << sTEP;

    CHKERR post_proc();
    CHKERR calculate_traction();
    CHKERR calculate_reactions();

    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");
    CHKERR print_traction("Contact force");

    ++sTEP;

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<moab::Core> postProcMesh = boost::make_shared<moab::Core>();

  boost::shared_ptr<PostProcEleDomain> postProcDomainFe;
  boost::shared_ptr<PostProcEleBdy> postProcBdyFe;

  boost::shared_ptr<BoundaryEle> integrateTraction;

  moab::Core mbVertexPostproc;
  moab::Interface &moabVertex;

  double lastTime;
  double deltaTime;
  int sTEP;
};

} // namespace ContactOps