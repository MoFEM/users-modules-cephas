/**
 * \file PostProcContact.hpp
 *
 *
 * @copyright Copyright (c) 2023
 */

namespace ContactOps {

template<int DIM>
struct PostProcEleByDim;

template <> struct PostProcEleByDim<2> {
  using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<3>;
};

template <> struct PostProcEleByDim<3> {
  using PostProcEle = PostProcBrokenMeshInMoab<BoundaryEle>;
  using SideEle = PipelineManager::ElementsAndOpsByDim<3>;
};

using PostProcEle = PostProcBrokenMeshInMoab<DomainEle>;

struct OpPostProcVertex : public BoundaryEleOp {
  OpPostProcVertex(MoFEM::Interface &m_field, const std::string field_name,
                   boost::shared_ptr<CommonData> common_data_ptr,
                   moab::Interface *moab_vertex);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  MoFEM::Interface &mField;
  moab::Interface *moabVertex;
  boost::shared_ptr<CommonData> commonDataPtr;
  boost::shared_ptr<WrapMPIComm> moabCommWrap;
  ParallelComm *pComm;
};

OpPostProcVertex::OpPostProcVertex(
    MoFEM::Interface &m_field, const std::string field_name,
    boost::shared_ptr<CommonData> common_data_ptr, moab::Interface *moab_vertex)
    : mField(m_field), BoundaryEleOp(field_name, BoundaryEleOp::OPROW),
      commonDataPtr(common_data_ptr), moabVertex(moab_vertex) {
  std::fill(&doEntities[MBVERTEX], &doEntities[MBMAXTYPE], false);
  for (EntityType t = CN::TypeDimensionMap[SPACE_DIM - 1].first;
       t <= CN::TypeDimensionMap[SPACE_DIM - 1].second; ++t) {
    doEntities[t] = true;
  }
  pComm = ParallelComm::get_pcomm(moabVertex, MYPCOMM_INDEX);
  if (pComm == NULL) {
    moabCommWrap = boost::make_shared<WrapMPIComm>(mField.get_comm(), false);
    pComm = new ParallelComm(moabVertex, moabCommWrap->get_comm());
  }
}

MoFEMErrorCode OpPostProcVertex::doWork(int side, EntityType type,
                                        EntData &data) {
  MoFEMFunctionBegin;

  const size_t nb_gauss_pts = getGaussPts().size2();
  const size_t nb_dofs = data.getIndices().size();
  std::array<double, 9> def;
  std::fill(def.begin(), def.end(), 0);

  auto get_tag = [&](const std::string name, size_t size) {
    Tag th;
    CHKERR moabVertex->tag_get_handle(name.c_str(), size, MB_TYPE_DOUBLE, th,
                                      MB_TAG_CREAT | MB_TAG_SPARSE, def.data());
    return th;
  };

  auto t_coords = getFTensor1CoordsAtGaussPts();
  auto t_disp = getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactDisp);
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(commonDataPtr->contactTraction);

  auto th_cons = get_tag("CONSTRAINT", 1);
  auto th_traction = get_tag("TRACTION", 3);
  auto th_normal = get_tag("NORMAL", 3);
  auto th_cont_normal = get_tag("CONTACT_NORMAL", 3);
  auto th_disp = get_tag("DISPLACEMENT", 3);
  auto th_gap = get_tag("GAP", 3);

  EntityHandle ent = getFEEntityHandle();

  for (size_t gg = 0; gg != nb_gauss_pts; ++gg) {

    double coords[] = {0, 0, 0};
    EntityHandle new_vertex;
    for (int dd = 0; dd != 3; dd++) {
      coords[dd] = getCoordsAtGaussPts()(gg, dd);
    }
    CHKERR moabVertex->create_vertex(&coords[0], new_vertex);
    FTensor::Tensor1<double, 3> trac(t_traction(0), t_traction(1), 0.);
    FTensor::Tensor1<double, 3> disp(t_disp(0), t_disp(1), 0.);

    FTensor::Tensor1<double, 3> t_spatial_coords{0., 0., 0.};
    t_spatial_coords(i) = t_coords(i) + t_disp(i);

    auto sdf = surface_distance_function(getTStime(), t_spatial_coords);
    auto t_grad_sdf =
        grad_surface_distance_function(getTStime(), t_spatial_coords);
    auto un = t_disp(i) * t_grad_sdf(i);
    auto tn = -t_traction(i) * t_grad_sdf(i);
    auto c = constrain(sdf, tn);

    FTensor::Tensor1<double, 3> t_gap{0., 0., 0.};
    t_gap(i) = t_disp(i) - sdf * t_grad_sdf(i);

    CHKERR moabVertex->tag_set_data(th_cons, &new_vertex, 1, &c);

    FTensor::Tensor1<double, 3> norm(t_grad_sdf(0), t_grad_sdf(1), 0.);

    if (SPACE_DIM == 3) {
      trac(2) = t_traction(2);
      norm(2) = t_grad_sdf(2);
      disp(2) = t_disp(2);
    }

    CHKERR moabVertex->tag_set_data(th_traction, &new_vertex, 1, &trac(0));
    CHKERR moabVertex->tag_set_data(th_cont_normal, &new_vertex, 1, &norm(0));
    CHKERR moabVertex->tag_set_data(th_disp, &new_vertex, 1, &disp(0));
    auto t_normal = getFTensor1Normal();
    CHKERR moabVertex->tag_set_data(th_normal, &new_vertex, 1, &t_normal(0));
    CHKERR moabVertex->tag_set_data(th_gap, &new_vertex, 1, &t_gap(0));

    auto set_part = [&](const auto vert) {
      MoFEMFunctionBegin;
      const int rank = mField.get_comm_rank();
      CHKERR moabVertex->tag_set_data(pComm->part_tag(), &vert, 1, &rank);
      MoFEMFunctionReturn(0);
    };

    CHKERR set_part(new_vertex);

    ++t_traction;
    ++t_coords;
    ++t_disp;
  }

  MoFEMFunctionReturn(0);
}

struct OpAssembleTraction : public BoundaryEleOp {
  OpAssembleTraction(boost::shared_ptr<CommonData> common_data_ptr,
                     SmartPetscObj<Vec> total_traction);
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

private:
  boost::shared_ptr<CommonData> commonDataPtr;
  SmartPetscObj<Vec> totalTraction;
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

    auto common_data_ptr = boost::make_shared<ContactOps::CommonData>();
    auto henky_common_data_ptr = boost::make_shared<HenckyOps::CommonData>();
    henky_common_data_ptr->matGradPtr = common_data_ptr->mGradPtr();
    henky_common_data_ptr->matDPtr = common_data_ptr->mDPtr();

    auto get_vertex_post_proc = [&]() {
      auto vertex_post_proc = boost::make_shared<BoundaryEle>(*m_field_ptr);
      CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
          vertex_post_proc->getOpPtrVector(), {HDIV});
      vertex_post_proc->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<SPACE_DIM>(
              "U", common_data_ptr->contactDispPtr()));
      vertex_post_proc->getOpPtrVector().push_back(
          new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
              "SIGMA", common_data_ptr->contactTractionPtr()));
      vertex_post_proc->getOpPtrVector().push_back(new OpPostProcVertex(
          *m_field_ptr, "U", common_data_ptr, &moabVertex));
      return vertex_post_proc;
    };

    auto get_post_proc_push_ops = [&](auto &pip) {
      MoFEMFunctionBegin;

      CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(pip, {H1, HDIV});
      CHKERR ContactOps::addMatBlockOps(*m_field_ptr, pip, "U", "MAT_ELASTIC",
                                        common_data_ptr->mDPtr(), Sev::inform);
      pip.push_back(new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
          "U", common_data_ptr->mGradPtr()));
      pip.push_back(
          new OpCalculateEigenVals<SPACE_DIM>("U", henky_common_data_ptr));
      pip.push_back(new OpCalculateLogC<SPACE_DIM>("U", henky_common_data_ptr));
      pip.push_back(
          new OpCalculateLogC_dC<SPACE_DIM>("U", henky_common_data_ptr));
      pip.push_back(
          new OpCalculateHenckyStress<SPACE_DIM>("U", henky_common_data_ptr));
      pip.push_back(
          new OpCalculatePiolaStress<SPACE_DIM>("U", henky_common_data_ptr));

      pip.push_back(new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
          "SIGMA", common_data_ptr->contactStressPtr()));

      MoFEMFunctionReturn(0);
    };
    

    auto get_post_proc_fe = [&]() {
      auto post_proc_fe = boost::make_shared<PostProcEle>(*m_field_ptr);

      auto u_ptr = boost::make_shared<MatrixDouble>();
      post_proc_fe->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

      CHKERR get_post_proc_push_ops(post_proc_fe->getOpPtrVector());

      using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

      post_proc_fe->getOpPtrVector().push_back(

          new OpPPMap(

              post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

              {},

              {{"U", u_ptr}},

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

    auto get_integrate_traction = [&]() {
      auto integrate_traction = boost::make_shared<BoundaryEle>(*m_field_ptr);
      CHKERR
      AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
          integrate_traction->getOpPtrVector(), {HDIV});
      integrate_traction->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<SPACE_DIM>(
              "U", common_data_ptr->contactDispPtr()));
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

    vertexPostProc = get_vertex_post_proc();
    postProcFe = get_post_proc_fe();
    integrateTraction = get_integrate_traction();
  }

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    auto post_proc_volume = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_contact_" + boost::lexical_cast<std::string>(sTEP) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    auto post_proc_boundary = [&] {
      MoFEMFunctionBegin;
      std::ostringstream ostrm;
      ostrm << "out_boundary_contact_" << sTEP << ".h5m";
      CHKERR DMoFEMLoopFiniteElements(dM, "bFE", vertexPostProc);
      CHKERR moabVertex.write_file(ostrm.str().c_str(), "MOAB",
                                   "PARALLEL=WRITE_PART");
      moabVertex.delete_mesh();

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
    CHKERR post_proc_volume();
    CHKERR post_proc_boundary();

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

  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<PostProcEleByDim<SPACE_DIM>::SideEle> postProcSideFe;
  boost::shared_ptr<BoundaryEle> vertexPostProc;
  boost::shared_ptr<BoundaryEle> integrateTraction;

  moab::Core mbVertexPostproc;
  moab::Interface &moabVertex;

  double lastTime;
  double deltaTime;
  int sTEP;
};

} // namespace ContactOps