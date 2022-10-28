

/** \file PlasticOpsMonitor.hpp
 * \example PlasticOpsMonitor.hpp
 */

namespace PlasticOps {

struct Monitor : public FEMethod {

  Monitor(SmartPetscObj<DM> &dm, boost::shared_ptr<PostProcEle> &post_proc_fe,
          boost::shared_ptr<DomainEle> &reaction_fe,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> ux_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uy_scatter,
          std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uz_scatter)
      : dM(dm), postProcFe(post_proc_fe), reactionFe(reaction_fe),
        uXScatter(ux_scatter), uYScatter(uy_scatter), uZScatter(uz_scatter){};

  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;

    auto make_vtk = [&]() {
      MoFEMFunctionBegin;
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProcFe);
      CHKERR postProcFe->writeFile(
          "out_plastic_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
      MoFEMFunctionReturn(0);
    };

    auto calculate_reaction = [&]() {
      MoFEMFunctionBegin;
      auto r = smartCreateDMVector(dM);
      reactionFe->f = r;
      CHKERR VecZeroEntries(r);
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", reactionFe);
      CHKERR VecGhostUpdateBegin(r, ADD_VALUES, SCATTER_REVERSE);
      CHKERR VecGhostUpdateEnd(r, ADD_VALUES, SCATTER_REVERSE);
      CHKERR VecAssemblyBegin(r);
      CHKERR VecAssemblyEnd(r);

      double sum;
      CHKERR VecSum(r, &sum);
      MOFEM_LOG_C("EXAMPLE", Sev::inform, "reaction time %3.4e %3.4e", ts_t,
                  sum);

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

    CHKERR make_vtk();
    if (reactionFe)
      CHKERR calculate_reaction();
    CHKERR print_max_min(uXScatter, "Ux");
    CHKERR print_max_min(uYScatter, "Uy");
    if (SPACE_DIM == 3)
      CHKERR print_max_min(uZScatter, "Uz");

    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcEle> postProcFe;
  boost::shared_ptr<DomainEle> reactionFe;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uXScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uYScatter;
  std::tuple<SmartPetscObj<Vec>, SmartPetscObj<VecScatter>> uZScatter;
};

}; // namespace PlasticOps