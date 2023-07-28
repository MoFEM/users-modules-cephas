

/** \file PlasticOpsMonitor.hpp
 * \example PlasticOpsMonitor.hpp
 */

namespace PlasticOps {

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> dm,
          std::pair<boost::shared_ptr<PostProcEle>,
                    boost::shared_ptr<SkinPostProcEle>>
              pair_post_proc_fe,
          boost::shared_ptr<DomainEle> reaction_fe,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), reactionFe(reaction_fe), uXScatter(ux_scatter),
        uYScatter(uy_scatter), uZScatter(uz_scatter) {
    postProcFe = pair_post_proc_fe.first;
    skinPostProcFe = pair_post_proc_fe.second;
  };

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    MoFEM::Interface *m_field_ptr;
    CHKERR DMoFEMGetInterfacePtr(dM, &m_field_ptr);

    auto make_vtk = [&]() {
      MoFEMFunctionBegin;
      if (postProcFe) {
        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe,
                                        getCacheWeakPtr());
        CHKERR postProcFe->writeFile("out_plastic_" +
                                     boost::lexical_cast<std::string>(ts_step) +
                                     ".h5m");
      }
      if (skinPostProcFe) {
        CHKERR DMoFEMLoopFiniteElements(dM, "bFE", skinPostProcFe,
                                        getCacheWeakPtr());
        CHKERR skinPostProcFe->writeFile(
            "out_skin_plastic_" + boost::lexical_cast<std::string>(ts_step) +
            ".h5m");
      }
      MoFEMFunctionReturn(0);
    };

    auto calculate_reaction = [&]() {
      MoFEMFunctionBegin;
      auto r = createDMVector(dM);
      reactionFe->f = r;
      CHKERR VecZeroEntries(r);
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", reactionFe, getCacheWeakPtr());

#ifndef NDEBUG
      auto post_proc_residual = [&](auto dm, auto f_res, auto out_name) {
        MoFEMFunctionBegin;
        auto post_proc_fe =
            boost::make_shared<PostProcBrokenMeshInMoab<DomainEle>>(
                *m_field_ptr);
        using OpPPMap = OpPostProcMapInMoab<SPACE_DIM, SPACE_DIM>;
        auto u_vec = boost::make_shared<MatrixDouble>();
        post_proc_fe->getOpPtrVector().push_back(
            new OpCalculateVectorFieldValues<SPACE_DIM>("U", u_vec, f_res));
        post_proc_fe->getOpPtrVector().push_back(

            new OpPPMap(

                post_proc_fe->getPostProcMesh(), post_proc_fe->getMapGaussPts(),

                {},

                {{"RES", u_vec}},

                {}, {})

        );

        CHKERR DMoFEMLoopFiniteElements(dM, "dFE", post_proc_fe);
        post_proc_fe->writeFile("res.h5m");
        MoFEMFunctionReturn(0);
      };

      CHKERR post_proc_residual(dM, r, "reaction");
#endif // NDEBUG

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
      MOFEM_LOG_C("PLASTICITY", Sev::inform, "%s time %3.4e min %3.4e max %3.4e",
                  msg.c_str(), ts_t, min, max);
      MoFEMFunctionReturn(0);
    };

    CHKERR make_vtk();
    if (reactionFe)
      CHKERR calculate_reaction();
    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if constexpr (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<SkinPostProcEle> skinPostProcFe;
  boost::shared_ptr<DomainEle> reactionFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;
};

}; // namespace PlasticOps