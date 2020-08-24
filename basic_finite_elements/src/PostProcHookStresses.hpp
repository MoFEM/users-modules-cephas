/**
 * \file PostProcHookStress.hpp
 * \brief Post-proc stresses for linear Hooke isotropic material
 *
 * \ingroup nonlinear_elastic_elem
 */

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

/**
 * \brief Operator post-procesing stresses for Hook isotropic material

 * Example how to use it

 \code
 PostProcVolumeOnRefinedMesh post_proc(m_field);
 {
   CHKERR post_proc.generateReferenceElementMesh();
   CHKERR post_proc.addFieldValuesPostProc("DISPLACEMENT");
   CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
   CHKERR post_proc.addFieldValuesGradientPostProc("DISPLACEMENT");
   //add postprocessing for stresses
   post_proc.getOpPtrVector().push_back(
     new PostProcHookStress(
       m_field,
       post_proc.postProcMesh,
       post_proc.mapGaussPts,
       "DISPLACEMENT",
       post_proc.commonData,
       &elastic.setOfBlocks
     )
   );
   CHKERR DMoFEMLoopFiniteElements(dm,"ELASTIC",&post_proc);
   CHKERR post_proc.writeFile("out.h5m");
 }

 \endcode

 */
struct PostProcHookStress
    : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

  MoFEM::Interface &mField;
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  bool isFieldDisp;

#ifdef __NONLINEAR_ELASTIC_HPP
  /// Material block data, ket is block id
  const std::map<int, NonlinearElasticElement::BlockData>
      *setOfBlocksMaterialDataPtr;
#endif //__NONLINEAR_ELASTIC_HPP

  PostProcVolumeOnRefinedMesh::CommonData &commonData;

  /**
   * Constructor
   */
  PostProcHookStress(MoFEM::Interface &m_field, moab::Interface &post_proc_mesh,
                     std::vector<EntityHandle> &map_gauss_pts,
                     const std::string field_name,
                     PostProcVolumeOnRefinedMesh::CommonData &common_data,
#ifdef __NONLINEAR_ELASTIC_HPP
                     const std::map<int, NonlinearElasticElement::BlockData>
                         *set_of_block_data_ptr = NULL,
#endif
                     const bool is_field_disp = true)
      : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
            field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
        mField(m_field), postProcMesh(post_proc_mesh),
        mapGaussPts(map_gauss_pts),
#ifdef __NONLINEAR_ELASTIC_HPP
        setOfBlocksMaterialDataPtr(set_of_block_data_ptr),
#endif //__NONLINEAR_ELASTIC_HPP
        commonData(common_data), isFieldDisp(is_field_disp) {
  }

  /**
   * \brief get material parameter

   * Material parameters are read form BlockSet, however if block data are
   present,
   * use data how are set for elastic element operators.

   * @param  _lambda   elastic material constant
   * @param  _mu       elastic material constant
   * @param  _block_id  block id
   * @return           error code

   */
  MoFEMErrorCode getMatParameters(double *_lambda, double *_mu,
                                  int *_block_id) {
    MoFEMFunctionBegin;

    *_lambda = 1;
    *_mu = 1;

    EntityHandle ent = getNumeredEntFiniteElementPtr()->getEnt();
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, BLOCKSET | MAT_ELASTICSET, it)) {
      Mat_Elastic mydata;
      CHKERR it->getAttributeDataStructure(mydata);

      Range meshsets;
      CHKERR mField.get_moab().get_entities_by_type(it->meshset, MBENTITYSET,
                                                    meshsets, false);
      meshsets.insert(it->meshset);
      for (Range::iterator mit = meshsets.begin(); mit != meshsets.end();
           mit++) {
        if (mField.get_moab().contains_entities(*mit, &ent, 1)) {
          *_lambda = LAMBDA(mydata.data.Young, mydata.data.Poisson);
          *_mu = MU(mydata.data.Young, mydata.data.Poisson);
          *_block_id = it->getMeshsetId();
#ifdef __NONLINEAR_ELASTIC_HPP
          if (setOfBlocksMaterialDataPtr) {
            *_lambda =
                LAMBDA(setOfBlocksMaterialDataPtr->at(*_block_id).E,
                       setOfBlocksMaterialDataPtr->at(*_block_id).PoissonRatio);
            *_mu = MU(setOfBlocksMaterialDataPtr->at(*_block_id).E,
                      setOfBlocksMaterialDataPtr->at(*_block_id).PoissonRatio);
          }
#endif //__NONLINEAR_ELASTIC_HPP
          MoFEMFunctionReturnHot(0);
        }
      }
    }

    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Element is not in elastic block, however you run linear elastic "
            "analysis with that element\n"
            "top tip: check if you update block sets after mesh refinements or "
            "interface insertion");

    MoFEMFunctionReturn(0);
  }

  /**
   * \brief Here real work is done
   */
  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;

    if (type != MBVERTEX)
      MoFEMFunctionReturnHot(0);
    if (data.getFieldData().size() == 0)
      MoFEMFunctionReturnHot(0);

    // const MoFEM::FEDofEntity *dof_ptr = data.getFieldDofs()[0];

    int id;
    double lambda, mu;
    CHKERR getMatParameters(&lambda, &mu, &id);

    MatrixDouble D_lambda, D_mu, D;
    D_lambda.resize(6, 6);
    D_lambda.clear();
    for (int rr = 0; rr < 3; rr++) {
      for (int cc = 0; cc < 3; cc++) {
        D_lambda(rr, cc) = 1;
      }
    }
    D_mu.resize(6, 6);
    D_mu.clear();
    for (int rr = 0; rr < 6; rr++) {
      D_mu(rr, rr) = rr < 3 ? 2 : 1;
    }
    D = lambda * D_lambda + mu * D_mu;

    int tag_length = 9;
    double def_VAL[tag_length];
    bzero(def_VAL, tag_length * sizeof(double));
    Tag th_stress;
    CHKERR postProcMesh.tag_get_handle("STRESS", 9, MB_TYPE_DOUBLE, th_stress,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

    Tag th_prin_stress_vect1, th_prin_stress_vect2, th_prin_stress_vect3;
    CHKERR postProcMesh.tag_get_handle("S1", 3, MB_TYPE_DOUBLE,
                                       th_prin_stress_vect1,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    CHKERR postProcMesh.tag_get_handle("S2", 3, MB_TYPE_DOUBLE,
                                       th_prin_stress_vect2,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    CHKERR postProcMesh.tag_get_handle("S3", 3, MB_TYPE_DOUBLE,
                                       th_prin_stress_vect3,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

    Tag th_prin_stress_vals;
    CHKERR postProcMesh.tag_get_handle("PRINCIPAL_STRESS", 3, MB_TYPE_DOUBLE,
                                       th_prin_stress_vals,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

    Tag th_id;
    int def_block_id = -1;
    CHKERR postProcMesh.tag_get_handle("BLOCK_ID", 1, MB_TYPE_INTEGER, th_id,
                                       MB_TAG_CREAT | MB_TAG_SPARSE,
                                       &def_block_id);
    Range::iterator tit = commonData.tEts.begin();
    for (; tit != commonData.tEts.end(); tit++) {
      CHKERR postProcMesh.tag_set_data(th_id, &*tit, 1, &id);
    }

    VectorDouble strain;
    VectorDouble stress;
    MatrixDouble Stress;

    // Combine eigenvalues and vectors to create principal stress vector
    MatrixDouble prin_stress_vect(3, 3);
    VectorDouble prin_vals_vect(3);

    int nb_gauss_pts = data.getN().size1();
    if (mapGaussPts.size() != (unsigned int)nb_gauss_pts) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "data inconsistency");
    }
    for (int gg = 0; gg < nb_gauss_pts; gg++) {

      strain.resize(6);
      strain[0] = (commonData.gradMap[rowFieldName][gg])(0, 0);
      strain[1] = (commonData.gradMap[rowFieldName][gg])(1, 1);
      strain[2] = (commonData.gradMap[rowFieldName][gg])(2, 2);
      strain[3] = (commonData.gradMap[rowFieldName][gg])(0, 1) +
                  (commonData.gradMap[rowFieldName][gg])(1, 0);
      strain[4] = (commonData.gradMap[rowFieldName][gg])(1, 2) +
                  (commonData.gradMap[rowFieldName][gg])(2, 1);
      strain[5] = (commonData.gradMap[rowFieldName][gg])(0, 2) +
                  (commonData.gradMap[rowFieldName][gg])(2, 0);

      if (!isFieldDisp) {
        strain[0] -= 1.;
        strain[1] -= 1.;
        strain[2] -= 1.;
      }

      stress.resize(6);
      noalias(stress) = prod(D, strain);

      Stress.resize(3, 3);
      Stress(0, 0) = stress[0];
      Stress(1, 1) = stress[1];
      Stress(2, 2) = stress[2];
      Stress(0, 1) = Stress(1, 0) = stress[3];
      Stress(1, 2) = Stress(2, 1) = stress[4];
      Stress(2, 0) = Stress(0, 2) = stress[5];

      CHKERR postProcMesh.tag_set_data(th_stress, &mapGaussPts[gg], 1,
                                       &Stress(0, 0));

      MatrixDouble eigen_vectors = Stress;
      VectorDouble eigen_values(3);

      // LAPACK - eigenvalues and vectors. Applied twice for initial creates
      // memory space
      int n = 3, lda = 3, info, lwork = -1;
      double wkopt;
      info = lapack_dsyev('V', 'U', n, &(eigen_vectors.data()[0]), lda,
                          &(eigen_values.data()[0]), &wkopt, lwork);
      if (info != 0)
        SETERRQ1(PETSC_COMM_SELF, 1,
                 "is something wrong with lapack_dsyev info = %d", info);
      lwork = (int)wkopt;
      double work[lwork];
      info = lapack_dsyev('V', 'U', n, &(eigen_vectors.data()[0]), lda,
                          &(eigen_values.data()[0]), work, lwork);
      if (info != 0)
        SETERRQ1(PETSC_COMM_SELF, 1,
                 "is something wrong with lapack_dsyev info = %d", info);

      map<double, int> eigen_sort;
      eigen_sort[eigen_values[0]] = 0;
      eigen_sort[eigen_values[1]] = 1;
      eigen_sort[eigen_values[2]] = 2;

      prin_stress_vect.clear();
      prin_vals_vect.clear();

      int ii = 0;
      for (map<double, int>::reverse_iterator mit = eigen_sort.rbegin();
           mit != eigen_sort.rend(); mit++) {
        prin_vals_vect[ii] = eigen_values[mit->second];
        for (int dd = 0; dd != 3; dd++) {
          prin_stress_vect(ii, dd) = eigen_vectors.data()[3 * mit->second + dd];
        }
        ii++;
      }

      // Tag principle stress vectors 1, 2, 3
      CHKERR postProcMesh.tag_set_data(th_prin_stress_vect1, &mapGaussPts[gg],
                                       1, &prin_stress_vect(0, 0));
      CHKERR postProcMesh.tag_set_data(th_prin_stress_vect2, &mapGaussPts[gg],
                                       1, &prin_stress_vect(1, 0));
      CHKERR postProcMesh.tag_set_data(th_prin_stress_vect3, &mapGaussPts[gg],
                                       1, &prin_stress_vect(2, 0));
      CHKERR postProcMesh.tag_set_data(th_prin_stress_vals, &mapGaussPts[gg], 1,
                                       &prin_vals_vect[0]);
    }

    MoFEMFunctionReturn(0);
  }
};

/// \deprecated Class name with spelling mistake
DEPRECATED typedef PostProcHookStress PostPorcHookStress;
