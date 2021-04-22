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
  MatrixDouble &mwlsStresses;

  PostCellProcStress(moab::Interface &post_proc_mesh,
                 const std::string field_name,
                 NonlinearElasticElement::BlockData &data,
                 NonlinearElasticElement::CommonData &common_data,
                MatrixDouble &mwls_stresses,
                 const bool field_disp = false,
                 const bool replace_nonanumber_by_max_value = false,
                 const double max_val = 1e16,
                 const bool print_cauchy_stress = false)
      : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
            field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
        outMesh(post_proc_mesh), dAta(data),
        commonData(common_data), fieldDisp(field_disp),
        replaceNonANumberByMaxValue(replace_nonanumber_by_max_value),
        maxVal(max_val), printCauchy(print_cauchy_stress), mwlsStresses(mwls_stresses) {}

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

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    // Tag th_id;
    // int def_block_id = -1;
    // CHKERR outMesh.tag_get_handle("BLOCK_ID", 1, MB_TYPE_INTEGER, th_id,
    //                                    MB_TAG_CREAT | MB_TAG_SPARSE,
    //                                    &def_block_id);
    // Range::iterator tit = commonData.tEts.begin();
    // for (; tit != commonData.tEts.end(); tit++) {
    //   CHKERR outMesh.tag_set_data(th_id, &*tit, 1, &id);
    // }

    // Tag th_help;
    // CHKERR outMesh.tag_get_handle("MED_SIEFELGA",
    //                                            th_help);

    // VectorDouble approx_stresses(9);
    // // CHKERR outMesh.tag_get_data(th_help, ent, &*approx_stresses.begin());
    // CHKERR outMesh.tag_get_data(th_help, &ent, 9,
    //                                      &*approx_stresses.begin());

    // cerr << " approx_stresses  " << approx_stresses(0)<< "   " << approx_stresses(1)<< "  "<< approx_stresses(2) <<"\n";
    // cerr << " mwlsStresses  " << mwlsStresses(0,0)<< "   " << mwlsStresses(1,1)<< "  "<< mwlsStresses(2, 2) <<"\n";

    string tag_name_piola1 = "MED_" + dof_ptr->getName() + "_PIOLA1_STRESS";
    string tag_name_energy = dof_ptr->getName() + "_ENERGY_DENSITY";

    int tag_length = 9;
    double def_VAL[tag_length];
    bzero(def_VAL, tag_length * sizeof(double));
    Tag th_piola1, th_energy, th_cauchy, th_actual, th_approx;
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

      string tag_name_actual = "MED_" + dof_ptr->getName() + "_ACTUAL_STRESS";
      CHKERR outMesh.tag_get_handle(tag_name_actual.c_str(), tag_length,
                                    MB_TYPE_DOUBLE, th_actual,
                                    MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

      string tag_name_approx = "MED_" + dof_ptr->getName() + "_APPROX_STRESS";
      CHKERR outMesh.tag_get_handle(tag_name_approx.c_str(), tag_length,
                                    MB_TYPE_DOUBLE, th_approx,
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

        ublas::matrix<double, ublas::row_major, ublas::bounded_array<double, 9>> piola_stress_data;
    piola_stress_data.resize(3, 3);
    piola_stress_data.clear();

  //     MatrixDouble &stress = mwlsStresses;
  // FTensor::Tensor2_symmetric<FTensor::PackPtr<double *, 1>, 3> t_stress(
  //     &stress(0, 0), &stress(3, 0), &stress(4, 0), &stress(1, 0), &stress(5, 0),
  //     &stress(2, 0));

  // FTensor::Tensor2<double *, 3, 3> t_piola(
  //     &piola_stress_data(0, 0), &piola_stress_data(0, 1),
  //     &piola_stress_data(0, 2), &piola_stress_data(1, 0),
  //     &piola_stress_data(1, 1), &piola_stress_data(1, 2),
  //     &piola_stress_data(2, 0), &piola_stress_data(2, 1),
  //     &piola_stress_data(2, 2));
  auto t_w = getFTensor0IntegrationWeight();


  VectorDouble vec_approx;
  vec_approx.resize(9, false);
  vec_approx.clear();

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    // const double weight = getGaussPts()(3, gg);

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
    CHKERR dAta.materialDoublePtr->setUserActiveVariables(nb_active_variables);
    CHKERR dAta.materialDoublePtr->calculateP_PiolaKirchhoffI(
        dAta, getNumeredEntFiniteElementPtr());
    CHKERR dAta.materialDoublePtr->calculateElasticEnergy(
        dAta, getNumeredEntFiniteElementPtr());
    piola_stress_data += dAta.materialDoublePtr->P * t_w;
    //t_piola(i, j) += t_w * t_stress(i, j);
// cerr << " \n\n ########\n\n";
    // piola_stress_data(0, 0) += t_w * stress(0, gg);
    // piola_stress_data(1, 1) += t_w * stress(1, gg);
    // piola_stress_data(2, 2) += t_w * stress(2, gg);
    // piola_stress_data(0, 1) += t_w * stress(3, gg);
    // piola_stress_data(0, 2) += t_w * stress(4, gg);
    // piola_stress_data(1, 2) += t_w * stress(5, gg);


    // piola_stress_data() += dAta.materialDoublePtr->P * t_w;
    // CHKERR outMesh.tag_set_data(th_piola1, &mapGaussPts[gg], 1,
    //                                  &dAta.materialDoublePtr->P(0, 0));
    // CHKERR outMesh.tag_set_data(th_energy, &mapGaussPts[gg], 1,
    //                                  &dAta.materialDoublePtr->eNergy);
    if (printCauchy) {
      dAta.materialDoublePtr->sigmaCauchy.resize(3, 3);
      CHKERR dAta.materialDoublePtr->calculateCauchyStress(
          dAta, getNumeredEntFiniteElementPtr());
      c_stress_data += dAta.materialDoublePtr->sigmaCauchy * t_w;
      // CHKERR outMesh.tag_set_data(
      //     th_cauchy, &mapGaussPts[gg], 1,
      //     &dAta.materialDoublePtr->sigmaCauchy(0, 0));
    }

    vec_approx[0] += t_w * mwlsStresses(0, gg);
    vec_approx[1] += t_w * mwlsStresses(1, gg);
    vec_approx[2] += t_w * mwlsStresses(2, gg);
    vec_approx[3] += t_w * mwlsStresses(3, gg);
    vec_approx[4] += t_w * mwlsStresses(4, gg);
    vec_approx[5] += t_w * mwlsStresses(5, gg);

    ++t_w;
    }

      VectorDouble vec_cauchy_stress_integrated;
      vec_cauchy_stress_integrated.resize(9, false);
      vec_cauchy_stress_integrated.clear();

        VectorDouble vec_piola1_stress_integrated;
      vec_piola1_stress_integrated.resize(9, false);
      vec_piola1_stress_integrated.clear();

        vec_cauchy_stress_integrated[0] = c_stress_data(0, 0);
        vec_cauchy_stress_integrated[1] = c_stress_data(1, 1);
        vec_cauchy_stress_integrated[2] = c_stress_data(2, 2);
        vec_cauchy_stress_integrated[3] = c_stress_data(0, 1);
        vec_cauchy_stress_integrated[4] = c_stress_data(0, 2);
        vec_cauchy_stress_integrated[5] = c_stress_data(1, 2);


        vec_piola1_stress_integrated[0] = piola_stress_data(0, 0);
        vec_piola1_stress_integrated[1] = piola_stress_data(1, 1);
        vec_piola1_stress_integrated[2] = piola_stress_data(2, 2);
        vec_piola1_stress_integrated[3] = piola_stress_data(0, 1);
        vec_piola1_stress_integrated[4] = piola_stress_data(0, 2);
        vec_piola1_stress_integrated[5] = piola_stress_data(1, 2);


// cerr << "vec_piola1_stress_integrated " << vec_piola1_stress_integrated << endl;
    CHKERR outMesh.tag_set_data(
        th_piola1, &ent, 1,
        &*vec_piola1_stress_integrated.begin());

    CHKERR outMesh.tag_set_data(
        th_cauchy, &ent, 1,
        &*vec_cauchy_stress_integrated.begin());

    VectorDouble vec_actual;
    vec_actual.resize(9, false);
    vec_actual.clear();

    vec_actual[0] = vec_piola1_stress_integrated[0] + vec_approx[0];
    vec_actual[1] = vec_piola1_stress_integrated[1] + vec_approx[1];
    vec_actual[2] = vec_piola1_stress_integrated[2] + vec_approx[2];
    vec_actual[3] = vec_piola1_stress_integrated[3] + vec_approx[3];
    vec_actual[4] = vec_piola1_stress_integrated[4] + vec_approx[4];
    vec_actual[5] = vec_piola1_stress_integrated[5] + vec_approx[5];

    CHKERR outMesh.tag_set_data(th_actual, &ent, 1,
                                &*vec_actual.begin());

    CHKERR outMesh.tag_set_data(th_approx, &ent, 1,
                                &*vec_approx.begin());

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
