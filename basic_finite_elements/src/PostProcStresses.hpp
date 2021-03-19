/** \file PostProcStresses.hpp
 * \brief Post-processing stresses for non-linear analysis
 * \ingroup nonlinear_elastic_elem
 *
 * Implementation of method for post-processing stresses.
 */

/*
 * This file is part of MoFEM.
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

#ifndef __POSTPROCSTRESSES_HPP__
#define __POSTPROCSTRESSES_HPP__

#ifndef WITH_ADOL_C
#error "MoFEM need to be compiled with ADOL-C"
#endif

struct PostProcStress
    : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;

  NonlinearElasticElement::BlockData &dAta;
  PostProcCommonOnRefMesh::CommonDataForVolume &commonData;
  const bool fieldDisp;
  const bool replaceNonANumberByMaxValue;
  const double maxVal;
  const bool printCauchy;

  PostProcStress(moab::Interface &post_proc_mesh,
                 std::vector<EntityHandle> &map_gauss_pts,
                 const std::string field_name,
                 NonlinearElasticElement::BlockData &data,
                 PostProcCommonOnRefMesh::CommonDataForVolume &common_data,
                 const bool field_disp = false,
                 const bool replace_nonanumber_by_max_value = false,
                 const double max_val = 1e16,
                 const bool print_cauchy_stress = false)
      : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
            field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
        postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts), dAta(data),
        commonData(common_data), fieldDisp(field_disp),
        replaceNonANumberByMaxValue(replace_nonanumber_by_max_value),
        maxVal(max_val), printCauchy(print_cauchy_stress) {}

  NonlinearElasticElement::CommonData nonLinearElementCommonData;

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    if (type != MBVERTEX)
      MoFEMFunctionReturnHot(0);
    if (data.getIndices().size() == 0)
      MoFEMFunctionReturnHot(0);
    if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tEts.end()) {
      MoFEMFunctionReturnHot(0);
    }

    const auto &dof_ptr = data.getFieldDofs()[0];

    int id = dAta.iD;

    Tag th_id;
    int def_block_id = -1;
    CHKERR postProcMesh.tag_get_handle("BLOCK_ID", 1, MB_TYPE_INTEGER, th_id,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       &def_block_id);
    Range::iterator tit = commonData.tEts.begin();
    for (; tit != commonData.tEts.end(); tit++) {
      CHKERR postProcMesh.tag_set_data(th_id, &*tit, 1, &id);
    }

    string tag_name_piola1 = dof_ptr->getName() + "_PIOLA1_STRESS";
    string tag_name_energy = dof_ptr->getName() + "_ENERGY_DENSITY";

    int tag_length = 9;
    double def_VAL[tag_length];
    bzero(def_VAL, tag_length * sizeof(double));
    Tag th_piola1, th_energy, th_cauchy;
    CHKERR postProcMesh.tag_get_handle(tag_name_piola1.c_str(), tag_length,
                                       MB_TYPE_DOUBLE, th_piola1,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    CHKERR postProcMesh.tag_get_handle(tag_name_energy.c_str(), 1,
                                       MB_TYPE_DOUBLE, th_energy,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

    if (printCauchy) {
      string tag_name_cauchy = "MED_" + dof_ptr->getName() + "_CAUCHY_STRESS";
      CHKERR postProcMesh.tag_get_handle(tag_name_cauchy.c_str(), tag_length,
                                         MB_TYPE_DOUBLE, th_cauchy,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    }

    int nb_gauss_pts = data.getN().size1();
    if (mapGaussPts.size() != (unsigned int)nb_gauss_pts) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "Nb. of integration points is not equal to number points on "
              "post-processing mesh");
    }
    if (commonData.gradMap[rowFieldName].size() != (unsigned int)nb_gauss_pts) {
      SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
               "Gradient of field not found, filed <%s> not found",
               rowFieldName.c_str());
    }

    MatrixDouble3by3 H, invH;
    double detH;

    dAta.materialDoublePtr->commonDataPtr = &nonLinearElementCommonData;
    dAta.materialDoublePtr->opPtr = this;
    CHKERR dAta.materialDoublePtr->getDataOnPostProcessor(commonData.fieldMap,
                                                          commonData.gradMap);

    nonLinearElementCommonData.dataAtGaussPts = commonData.fieldMap;
    nonLinearElementCommonData.gradAtGaussPts = commonData.gradMap;

    MatrixDouble3by3 maxP(3, 3);
    maxP.clear();

    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      dAta.materialDoublePtr->gG = gg;
      dAta.materialDoublePtr->F.resize(3, 3);
      noalias(dAta.materialDoublePtr->F) =
          (commonData.gradMap[rowFieldName])[gg];
      if (fieldDisp) {
        for (int dd = 0; dd != 3; dd++) {
          dAta.materialDoublePtr->F(dd, dd) += 1;
        }
      }
      if (commonData.gradMap["MESH_NODE_POSITIONS"].size() ==
          (unsigned int)nb_gauss_pts) {
        H.resize(3, 3);
        invH.resize(3, 3);
        noalias(H) = (commonData.gradMap["MESH_NODE_POSITIONS"])[gg];
        CHKERR dAta.materialDoublePtr->dEterminant(H, detH);
        CHKERR dAta.materialDoublePtr->iNvert(detH, H, invH);
        noalias(dAta.materialDoublePtr->F) =
            prod(dAta.materialDoublePtr->F, invH);
      }

      int nb_active_variables = 9;
      CHKERR dAta.materialDoublePtr->setUserActiveVariables(
          nb_active_variables);
      CHKERR dAta.materialDoublePtr->calculateP_PiolaKirchhoffI(
          dAta, getNumeredEntFiniteElementPtr());
      CHKERR dAta.materialDoublePtr->calculateElasticEnergy(
          dAta, getNumeredEntFiniteElementPtr());
      CHKERR postProcMesh.tag_set_data(th_piola1, &mapGaussPts[gg], 1,
                                       &dAta.materialDoublePtr->P(0, 0));
      CHKERR postProcMesh.tag_set_data(th_energy, &mapGaussPts[gg], 1,
                                       &dAta.materialDoublePtr->eNergy);
      if (printCauchy) {
        dAta.materialDoublePtr->sigmaCauchy.resize(3, 3);
        CHKERR dAta.materialDoublePtr->calculateCauchyStress(
            dAta, getNumeredEntFiniteElementPtr());
        CHKERR postProcMesh.tag_set_data(
            th_cauchy, &mapGaussPts[gg], 1,
            &dAta.materialDoublePtr->sigmaCauchy(0, 0));
      }
    }

    if (replaceNonANumberByMaxValue) {
      MatrixDouble3by3 P(3, 3);
      for (int gg = 0; gg != nb_gauss_pts; ++gg) {
        double val_energy;
        CHKERR postProcMesh.tag_get_data(th_energy, &mapGaussPts[gg], 1,
                                         &val_energy);
        if (!std::isnormal(val_energy)) {
          CHKERR postProcMesh.tag_set_data(th_energy, &mapGaussPts[gg], 1,
                                           &maxVal);
          CHKERR postProcMesh.tag_get_data(th_piola1, &mapGaussPts[gg], 1,
                                           &P(0, 0));
          for (unsigned int r = 0; r != P.size1(); ++r) {
            for (unsigned int c = 0; c != P.size2(); ++c) {
              if (!std::isnormal(P(r, c)))
                P(r, c) = copysign(maxVal, P(r, c));
            }
          }
          CHKERR postProcMesh.tag_set_data(th_piola1, &mapGaussPts[gg], 1,
                                           &P(0, 0));
        }
      }
    }

    MoFEMFunctionReturn(0);
  }
};


struct PostCellProcStress
    : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

  moab::Interface &outMesh;

  NonlinearElasticElement::BlockData &dAta;
  NonlinearElasticElement::CommonData &commonData;
  const bool fieldDisp;
  const bool replaceNonANumberByMaxValue;
  const double maxVal;
  const bool printCauchy;

  PostCellProcStress(moab::Interface &post_proc_mesh,
                 const std::string field_name,
                 NonlinearElasticElement::BlockData &data,
                 NonlinearElasticElement::CommonData &common_data,
                 const bool field_disp = false,
                 const bool replace_nonanumber_by_max_value = false,
                 const double max_val = 1e16,
                 const bool print_cauchy_stress = false)
      : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
            field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
        outMesh(post_proc_mesh), dAta(data),
        commonData(common_data), fieldDisp(field_disp),
        replaceNonANumberByMaxValue(replace_nonanumber_by_max_value),
        maxVal(max_val), printCauchy(print_cauchy_stress) {}

  NonlinearElasticElement::CommonData nonLinearElementCommonData;

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    if (type != MBVERTEX)
      MoFEMFunctionReturnHot(0);
    if (data.getIndices().size() == 0)
      MoFEMFunctionReturnHot(0);
    if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tEts.end()) {
      MoFEMFunctionReturnHot(0);
    }

    const auto &dof_ptr = data.getFieldDofs()[0];

    const EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();

    int id = dAta.iD;

    // Tag th_id;
    // int def_block_id = -1;
    // CHKERR outMesh.tag_get_handle("BLOCK_ID", 1, MB_TYPE_INTEGER, th_id,
    //                                    MB_TAG_CREAT | MB_TAG_SPARSE,
    //                                    &def_block_id);
    // Range::iterator tit = commonData.tEts.begin();
    // for (; tit != commonData.tEts.end(); tit++) {
    //   CHKERR outMesh.tag_set_data(th_id, &*tit, 1, &id);
    // }

    string tag_name_piola1 = dof_ptr->getName() + "_PIOLA1_STRESS";
    string tag_name_energy = dof_ptr->getName() + "_ENERGY_DENSITY";

    int tag_length = 9;
    double def_VAL[tag_length];
    bzero(def_VAL, tag_length * sizeof(double));
    Tag th_piola1, th_energy, th_cauchy;
    CHKERR outMesh.tag_get_handle(tag_name_piola1.c_str(), tag_length,
                                       MB_TYPE_DOUBLE, th_piola1,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    CHKERR outMesh.tag_get_handle(tag_name_energy.c_str(), 1,
                                       MB_TYPE_DOUBLE, th_energy,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

    if (printCauchy) {
      string tag_name_cauchy = "MED_" + dof_ptr->getName() + "_CAUCHY_STRESS";
      CHKERR outMesh.tag_get_handle(tag_name_cauchy.c_str(), tag_length,
                                         MB_TYPE_DOUBLE, th_cauchy,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    }

    int nb_gauss_pts = data.getN().size1();

    if (commonData.gradAtGaussPts[rowFieldName].size() != (unsigned int)nb_gauss_pts) {
      SETERRQ1(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
               "Gradient of field not found, filed <%s> not found",
               rowFieldName.c_str());
    }

    MatrixDouble3by3 H, invH;
    double detH;

    dAta.materialDoublePtr->commonDataPtr = &nonLinearElementCommonData;
    dAta.materialDoublePtr->opPtr = this;
    CHKERR dAta.materialDoublePtr->getDataOnPostProcessor(commonData.dataAtGaussPts,
                                                          commonData.gradAtGaussPts);

    nonLinearElementCommonData.dataAtGaussPts = commonData.dataAtGaussPts;
    nonLinearElementCommonData.gradAtGaussPts = commonData.gradAtGaussPts;

    MatrixDouble3by3 maxP(3, 3);
    maxP.clear();

    VectorDouble vec_cauchy_stress_mean;
    vec_cauchy_stress_mean.resize(9, false);
    vec_cauchy_stress_mean.clear();

    ublas::matrix<double, ublas::row_major, ublas::bounded_array<double, 9>> c_stress_data;
    c_stress_data.resize(3, 3);
    c_stress_data.clear();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      dAta.materialDoublePtr->gG = gg;
      dAta.materialDoublePtr->F.resize(3, 3);
      noalias(dAta.materialDoublePtr->F) =
          (commonData.gradAtGaussPts[rowFieldName])[gg];
      if (fieldDisp) {
        for (int dd = 0; dd != 3; dd++) {
          dAta.materialDoublePtr->F(dd, dd) += 1;
        }
      }
      if (commonData.gradAtGaussPts["MESH_NODE_POSITIONS"].size() ==
          (unsigned int)nb_gauss_pts) {
        H.resize(3, 3);
        invH.resize(3, 3);
        noalias(H) = (commonData.gradAtGaussPts["MESH_NODE_POSITIONS"])[gg];
        CHKERR dAta.materialDoublePtr->dEterminant(H, detH);
        CHKERR dAta.materialDoublePtr->iNvert(detH, H, invH);
        noalias(dAta.materialDoublePtr->F) =
            prod(dAta.materialDoublePtr->F, invH);
      }

      int nb_active_variables = 9;
      CHKERR dAta.materialDoublePtr->setUserActiveVariables(
          nb_active_variables);
      CHKERR dAta.materialDoublePtr->calculateP_PiolaKirchhoffI(
          dAta, getNumeredEntFiniteElementPtr());
      CHKERR dAta.materialDoublePtr->calculateElasticEnergy(
          dAta, getNumeredEntFiniteElementPtr());
      // CHKERR outMesh.tag_set_data(th_piola1, &mapGaussPts[gg], 1,
      //                                  &dAta.materialDoublePtr->P(0, 0));
      // CHKERR outMesh.tag_set_data(th_energy, &mapGaussPts[gg], 1,
      //                                  &dAta.materialDoublePtr->eNergy);
      if (printCauchy) {
        dAta.materialDoublePtr->sigmaCauchy.resize(3, 3);
        CHKERR dAta.materialDoublePtr->calculateCauchyStress(
            dAta, getNumeredEntFiniteElementPtr());
            c_stress_data += dAta.materialDoublePtr->sigmaCauchy/nb_gauss_pts;
        // CHKERR outMesh.tag_set_data(
        //     th_cauchy, &mapGaussPts[gg], 1,
        //     &dAta.materialDoublePtr->sigmaCauchy(0, 0));
      }
    }

    CHKERR outMesh.tag_set_data(
        th_cauchy, &ent, 1,
        &c_stress_data(0, 0));

    // cerr << "Stresses   " << c_stress_data(0, 0) << " " << c_stress_data(0, 1)
    //      << " " << c_stress_data(0, 2) << " " << c_stress_data(1, 0) << " "
    //      << c_stress_data(1, 1) << " " << c_stress_data(1, 2) << " "
    //      << c_stress_data(2, 0) << " " << c_stress_data(2, 1) << " "
    //      << c_stress_data(2, 2) << "\n";

    // if (replaceNonANumberByMaxValue) {
    //   MatrixDouble3by3 P(3, 3);
    //   for (int gg = 0; gg != nb_gauss_pts; ++gg) {
    //     double val_energy;
    //     CHKERR outMesh.tag_get_data(th_energy, &mapGaussPts[gg], 1,
    //                                      &val_energy);
    //     if (!std::isnormal(val_energy)) {
    //       CHKERR outMesh.tag_set_data(th_energy, &mapGaussPts[gg], 1,
    //                                        &maxVal);
    //       CHKERR outMesh.tag_get_data(th_piola1, &mapGaussPts[gg], 1,
    //                                        &P(0, 0));
    //       for (unsigned int r = 0; r != P.size1(); ++r) {
    //         for (unsigned int c = 0; c != P.size2(); ++c) {
    //           if (!std::isnormal(P(r, c)))
    //             P(r, c) = copysign(maxVal, P(r, c));
    //         }
    //       }
    //       CHKERR outMesh.tag_set_data(th_piola1, &mapGaussPts[gg], 1,
    //                                        &P(0, 0));
    //     }
    //   }
    // }

    MoFEMFunctionReturn(0);
  }
};

/// \deprecated Use PostProcStress
DEPRECATED typedef PostProcStress PostPorcStress;

#endif //__POSTPROCSTRESSES_HPP__
