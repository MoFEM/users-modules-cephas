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


struct PostProcHookStressAnisotropic : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

  MoFEM::Interface &mField;
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;

  PostProcVolumeOnRefinedMesh::CommonData &commonData;

  /**
   * Constructor
   */
  PostProcHookStressAnisotropic(MoFEM::Interface &m_field, moab::Interface &post_proc_mesh,
                                std::vector<EntityHandle> &map_gauss_pts,
                                const std::string field_name,
                                PostProcVolumeOnRefinedMesh::CommonData &common_data,
                                FTensor::Tensor4<double, 3, 3, 3, 3> t_elastic,
                                FTensor::Tensor2<double, 3, 3> t_sigma0) :
  MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
  mField(m_field), postProcMesh(post_proc_mesh),
  mapGaussPts(map_gauss_pts),
  commonData(common_data),
  tElastic(t_elastic),
  tSigma0(t_sigma0) {}


protected:
  FTensor::Tensor4<double, 3, 3, 3, 3> tElastic;
  FTensor::Tensor2<double, 3, 3> tSigma0;



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

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Index<'l', 3> l;


    int tag_length = 1;
    double def_VAL[tag_length];
    bzero(def_VAL, tag_length * sizeof(double));
    Tag th_stress;
    CHKERR postProcMesh.tag_get_handle("STRESS", 9, MB_TYPE_DOUBLE, th_stress,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    
    Tag th_cylindricalStress;
    CHKERR postProcMesh.tag_get_handle("C_STRESS", 9, MB_TYPE_DOUBLE, th_cylindricalStress,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    
    FTensor::Tensor2<double, 3, 3> tStress;
    FTensor::Tensor2<double, 3, 3> tStrain;
    MatrixDouble Stress;

    int nb_gauss_pts = data.getN().size1();
    MatrixDouble coordsGaussPts = getCoordsAtGaussPts();
    if ((mapGaussPts.size() != nb_gauss_pts) || (coordsGaussPts.size1() != nb_gauss_pts)) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "data inconsistency");
    }
    for (int gg = 0; gg < nb_gauss_pts; ++gg) {

      for (int ii = 0; ii < 3; ++ ii) {
        for (int jj = 0; jj < 3; ++ jj) {

          tStrain(ii,jj) = (commonData.gradMap[rowFieldName][gg])(ii,jj);

        }
      }

      // tStrain(i,j) = (commonData.gradMap[rowFieldName][gg])(i,j);

      tStress(i,j) = tElastic(i,j,k,l) * tStrain(k,l) + tSigma0(i,j);

      Stress.resize(3, 3);

      for (int ii = 0; ii < 3; ++ ii) {
        for (int jj = 0; jj < 3; ++ jj) {

          Stress(ii,jj) = tStress(ii,jj);

        }
      }

      CHKERR postProcMesh.tag_set_data(th_stress, &mapGaussPts[gg], 1, &Stress(0,0));


      // cylindrical coordinate
      double xCoord = coordsGaussPts(gg, 0);
      double yCoord = coordsGaussPts(gg, 1);
      double zCoord = coordsGaussPts(gg, 2);

      double radius = sqrt(xCoord * xCoord + yCoord * yCoord);
      double valueSin = yCoord / radius;
      double valueCos = xCoord / radius;

      Eigen::MatrixXd beta(3,3), betaTranspose(3,3), tmpStress(3,3);

      for (int ii = 0; ii < 3; ++ ii) {
        for (int jj = 0; jj < 3; ++ jj) {

          tmpStress(ii,jj) = Stress(ii,jj);

        }
      }

      beta(0, 0) = beta(1, 1) = valueCos;
      beta(0, 1) = valueSin;
      beta(1, 0) = -valueSin;
      beta(2, 2) = 1.0;

      betaTranspose = beta.transpose();

      tmpStress = beta * tmpStress * betaTranspose;

      MatrixDouble cylindricalStress;
      cylindricalStress.resize(3,3);

      for (int ii = 0; ii < 3; ++ ii) {
        for (int jj = 0; jj < 3; ++ jj) {

          cylindricalStress(ii,jj) = tmpStress(ii,jj);

        }
      }

      CHKERR postProcMesh.tag_set_data(th_cylindricalStress, &mapGaussPts[gg], 1, &cylindricalStress(0,0));



      

      





    }










    MoFEMFunctionReturn(0);
  }
};