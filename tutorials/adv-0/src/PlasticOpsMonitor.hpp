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

/** \file PlasticOpsMonitor.hpp
 * \example PlasticOpsMonitor.hpp
 */

namespace PlasticOps {

OpPostProcPlastic::OpPostProcPlastic(
    const std::string field_name, moab::Interface &post_proc_mesh,
    std::vector<EntityHandle> &map_gauss_pts,
    boost::shared_ptr<CommonData> common_data_ptr)
    : DomainEleOp(field_name, DomainEleOp::OPROW), postProcMesh(post_proc_mesh),
      mapGaussPts(map_gauss_pts), commonDataPtr(common_data_ptr) {
  // Opetor is only executed for vertices
  std::fill(&doEntities[MBEDGE], &doEntities[MBMAXTYPE], false);
}

//! [Postprocessing]
MoFEMErrorCode OpPostProcPlastic::doWork(int side, EntityType type,
                                         EntData &data) {
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

  auto set_float_precision = [](const double x) {
    if (std::abs(x) < std::numeric_limits<float>::epsilon())
      return 0.;
    else
      return x;
  };

  auto set_tag = [&](auto th, auto gg, MatrixDouble3by3 &mat) {
    for (auto &v : mat.data())
      v = set_float_precision(v);
    return postProcMesh.tag_set_data(th, &mapGaussPts[gg], 1,
                                     &*mat.data().begin());
  };

  auto th_plastic_surface = get_tag("PLASTIC_SURFACE", 1);
  auto th_tau = get_tag("PLASTIC_MULTIPLIER", 1);
  auto th_temperature = get_tag("TEMPERATURE", 1);
  auto th_plastic_flow = get_tag("PLASTIC_FLOW", 9);
  auto th_plastic_strain = get_tag("PLASTIC_STRAIN", 9);

  auto t_flow =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticFlow);
  auto t_plastic_strain =
      getFTensor2SymmetricFromMat<SPACE_DIM>(commonDataPtr->plasticStrain);
  if (commonDataPtr->tempVal.size() != commonDataPtr->plasticSurface.size()) {
    commonDataPtr->tempVal.resize(commonDataPtr->plasticSurface.size(), 0);
    commonDataPtr->tempVal.clear();
  }

  size_t gg = 0;
  for (int gg = 0; gg != commonDataPtr->plasticSurface.size(); ++gg) {
    const double temp = (commonDataPtr->tempVal)[gg];
    const double tau = (commonDataPtr->plasticTau)[gg];
    const double f = (commonDataPtr->plasticSurface)[gg] - hardening(tau, temp);
    CHKERR set_tag(th_plastic_surface, gg, set_scalar(f));
    CHKERR set_tag(th_tau, gg, set_scalar(tau));
    CHKERR set_tag(th_temperature, gg, set_scalar(temp));
    CHKERR set_tag(th_plastic_flow, gg, set_matrix_3d(t_flow));
    CHKERR set_tag(th_plastic_strain, gg, set_matrix_3d(t_plastic_strain));
    ++t_flow;
    ++t_plastic_strain;
  }

  MoFEMFunctionReturn(0);
}

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