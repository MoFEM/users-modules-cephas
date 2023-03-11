/**
 * \file PostProcContact.hpp
 *
 *
 * @copyright Copyright (c) 2023
 */

namespace ContactOps {

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
  auto t_disp = getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactDispPtr));
  auto t_traction =
      getFTensor1FromMat<SPACE_DIM>(*(commonDataPtr->contactTractionPtr));

  auto th_gap = get_tag("GAP", 1);
  auto th_cons = get_tag("CONSTRAINT", 1);
  auto th_traction = get_tag("TRACTION", 3);
  auto th_normal = get_tag("NORMAL", 3);
  auto th_cont_normal = get_tag("CONTACT_NORMAL", 3);
  auto th_disp = get_tag("DISPLACEMENT", 3);

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

    auto t_contact_normal = normal(t_coords, t_disp);
    const double g0 = gap0(t_coords, t_contact_normal);
    const double g = gap(t_disp, t_contact_normal);
    const double gap_tot = g - g0;
    const double constra = constrian(
        gap0(t_coords, t_contact_normal), gap(t_disp, t_contact_normal),
        normal_traction(t_traction, t_contact_normal));
    CHKERR moabVertex->tag_set_data(th_gap, &new_vertex, 1, &gap_tot);
    CHKERR moabVertex->tag_set_data(th_cons, &new_vertex, 1, &constra);

    FTensor::Tensor1<double, 3> norm(t_contact_normal(0), t_contact_normal(1),
                                     0.);

    if (SPACE_DIM == 3) {
      trac(2) = t_traction(2);
      norm(2) = t_contact_normal(2);
      disp(2) = t_disp(2);
    }

    CHKERR moabVertex->tag_set_data(th_traction, &new_vertex, 1, &trac(0));
    CHKERR moabVertex->tag_set_data(th_cont_normal, &new_vertex, 1, &norm(0));
    CHKERR moabVertex->tag_set_data(th_disp, &new_vertex, 1, &disp(0));
    auto t_normal = getFTensor1Normal();
    CHKERR moabVertex->tag_set_data(th_normal, &new_vertex, 1, &t_normal(0));

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

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm, boost::shared_ptr<CommonData> common_data_ptr,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), commonDataPtr(common_data_ptr), uXScatter(ux_scatter),
        uYScatter(uy_scatter), uZScatter(uz_scatter),
        moabVertex(mbVertexPostproc), sTEP(0) {

    auto henky_common_data_ptr = boost::make_shared<HenckyOps::CommonData>();
    henky_common_data_ptr->matGradPtr = commonDataPtr->mGradPtr;
    henky_common_data_ptr->matDPtr = commonDataPtr->mDPtr;

    MoFEM::Interface *m_field_ptr;
    CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);
    vertexPostProc = boost::make_shared<BoundaryEle>(*m_field_ptr);

    CHKERR AddHOOps<SPACE_DIM - 1, SPACE_DIM, SPACE_DIM>::add(
        vertexPostProc->getOpPtrVector(), {HDIV});
    vertexPostProc->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>(
            "U", commonDataPtr->contactDispPtr));
    vertexPostProc->getOpPtrVector().push_back(
        new OpCalculateHVecTensorTrace<SPACE_DIM, BoundaryEleOp>(
            "SIGMA", commonDataPtr->contactTractionPtr));
    vertexPostProc->getOpPtrVector().push_back(
        new OpPostProcVertex(*m_field_ptr, "U", commonDataPtr, &moabVertex));

    postProcFe = boost::make_shared<PostProcEle>(*m_field_ptr);
    CHKERR AddHOOps<SPACE_DIM, SPACE_DIM, SPACE_DIM>::add(
        postProcFe->getOpPtrVector(), {H1, HDIV});
    CHKERR ContactOps::addMatBlockOps(
        *m_field_ptr, postProcFe->getOpPtrVector(), "U", "MAT_ELASTIC",
        commonDataPtr->mDPtr, Sev::inform);
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<SPACE_DIM, SPACE_DIM>(
            "U", commonDataPtr->mGradPtr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateEigenVals<SPACE_DIM>("U", henky_common_data_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateLogC<SPACE_DIM>("U", henky_common_data_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateLogC_dC<SPACE_DIM>("U", henky_common_data_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHenckyStress<SPACE_DIM>("U", henky_common_data_ptr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculatePiolaStress<SPACE_DIM>("U", henky_common_data_ptr));

    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHVecTensorDivergence<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressDivergencePtr));
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateHVecTensorField<SPACE_DIM, SPACE_DIM>(
            "SIGMA", commonDataPtr->contactStressPtr));

    auto u_ptr = boost::make_shared<MatrixDouble>();
    postProcFe->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_ptr));

    using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;

    postProcFe->getOpPtrVector().push_back(

        new OpPPMap(

            postProcFe->getPostProcMesh(), postProcFe->getMapGaussPts(),

            {},

            {{"U", u_ptr}},

            {

                {"SIGMA", commonDataPtr->contactStressPtr},

                {"G", henky_common_data_ptr->matGradPtr},

                {"P2", henky_common_data_ptr->getMatFirstPiolaStress()}

            },

            {}

            )

    );
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
      MOFEM_LOG_C("EXAMPLE", Sev::inform, "%s time %3.4e min %3.4e max %3.4e",
                  msg.c_str(), ts_t, min, max);
      MoFEMFunctionReturn(0);
    };

    MOFEM_LOG("EXAMPLE", Sev::inform)
        << "Write file at time " << ts_t << " write step " << sTEP;

    ++sTEP;
    CHKERR post_proc_volume();
    CHKERR post_proc_boundary();

    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<CommonData> commonDataPtr;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;

  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<BoundaryEle> vertexPostProc;
  moab::Core mbVertexPostproc;
  moab::Interface &moabVertex;

  double lastTime;
  double deltaTime;
  int sTEP;
};

} // namespace ContactOps