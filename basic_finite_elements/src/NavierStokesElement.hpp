/**
 * \file NavierStokesOperators.hpp
 * \example NavierStokesOperators.hpp
 *
 * \brief Implementation of operators for fluid flow
 *
 * Implementation of operators for computations of the fluid flow, governed by
 * for Stokes and Navier-Stokes equations, and computation of the viscous drag
 * force
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

#ifndef __NAVIERSTOKESELEMENT_HPP__
#define __NAVIERSTOKESELEMENT_HPP__

#ifndef __BASICFINITEELEMENTS_HPP__
#include <BasicFiniteElements.hpp>
#endif // __BASICFINITEELEMENTS_HPP__

using namespace boost::numeric;

struct NavierStokesElement {

  using UserDataOperator =
      MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator;

  using FaceUserDataOperator =
      MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;

  using EntData = DataForcesAndSourcesCore::EntData;

  struct BlockData {
    int iD;
    // int oRder;
    double fluidViscosity;
    double fluidDensity;
    Range tEts;
    BlockData() : fluidViscosity(-1), fluidDensity(-1) {}
  };

  struct CommonData {

    //MatrixDouble invJac;

    boost::shared_ptr<MatrixDouble> gradDispPtr;
    boost::shared_ptr<MatrixDouble> dispPtr;
    boost::shared_ptr<VectorDouble> pPtr;

    VectorDouble3 pressureDrag;
    VectorDouble3 viscousDrag;

    std::map<int, BlockData> setOfBlocksData;

    CommonData() { 

      gradDispPtr = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      dispPtr = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      pPtr = boost::shared_ptr<VectorDouble>(new VectorDouble());

      pressureDrag = VectorDouble3(3);
      viscousDrag = VectorDouble3(3);

      pressureDrag.clear();
      viscousDrag.clear();

    }

    MoFEMErrorCode getParameters() {
      MoFEMFunctionBegin;
      CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Problem", "none");

      ierr = PetscOptionsEnd();
      CHKERRQ(ierr);
      MoFEMFunctionReturn(0);
    }
  };

  struct LoadScale : public MethodForForceScaling {

    static double lambda;

    MoFEMErrorCode scaleNf(const FEMethod *fe, VectorDouble &nf) {
      MoFEMFunctionBegin;
      nf *= lambda;
      MoFEMFunctionReturn(0);
    }
  };

  static MoFEMErrorCode setNavierStokesOperators(
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> feRhs,
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> feLhs,
      const std::string velocity_field, const std::string pressure_field,
      boost::shared_ptr<CommonData> common_data);

  static MoFEMErrorCode
  setStokesOperators(boost::shared_ptr<VolumeElementForcesAndSourcesCore> feRhs,
                     boost::shared_ptr<VolumeElementForcesAndSourcesCore> feLhs,
                     const std::string velocity_field,
                     const std::string pressure_field,
                     boost::shared_ptr<CommonData> common_data);

  /**
   * \brief Set integration rule to volume elements
   *
   * This rule is used to integrate \f$\nabla v \cdot \nabla u\f$, thus
   * if the approximation field and the testing field are polynomials of order
   * "p", then the rule for the exact integration is 2*(p-1).
   *
   * Integration rule is order of polynomial which is calculated exactly. Finite
   * element selects integration method based on return of this function.
   *
   */
  struct VolRule {
    int operator()(int order_row, int order_col, int order_data) const {
      // return order_row < order_col ? 2 * order_col - 1
      //                   : 2 * order_row - 1;
      return 2 * order_data;
      // return order_row < order_col ? 3 * order_col - 1
      //                   : 3 * order_row - 1;
    }
  };

  struct FaceRule {
    int operator()(int order_row, int order_col, int order_data) const {
      return order_data + 1;
    }
  };

  struct OpAssembleLhs : public UserDataOperator {

    boost::shared_ptr<CommonData> commonData;
    MatrixDouble locMat;
    BlockData &blockData;

    bool diagonalBlock;

    int row_nb_dofs;
    int col_nb_dofs;
    int row_nb_gauss_pts;

    bool isOnDiagonal;

    OpAssembleLhs(const string field_name_row, const string field_name_col,
                  boost::shared_ptr<CommonData> common_data,
                  BlockData &block_data)
        : UserDataOperator(field_name_row, field_name_col,
                           UserDataOperator::OPROWCOL),
          commonData(common_data), blockData(block_data) {
      sYmm = false;
      diagonalBlock = false;
    };

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

    virtual MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    };

    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);
  };

  /** * @brief Assemble G *
   * \f[
   * {\bf{G}} =  - \int\limits_\Omega  {{{\bf{B}}^T}{\bf m}{{\bf{N}}_p}d\Omega }
   * \f]
   *
   */
  struct OpAssembleLhsOffDiag : public OpAssembleLhs {

    OpAssembleLhsOffDiag(const string field_name_row,
                         const string field_name_col,
                         boost::shared_ptr<CommonData> common_data,
                         BlockData &block_data)
        : OpAssembleLhs(field_name_row, field_name_col, common_data,
                        block_data) {
      sYmm = false;
      diagonalBlock = false;
    };

    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  /** * @brief Assemble K *
   * \f[
   * {\bf{K}} = \int\limits_\Omega  {{{\bf{B}}^T}{{\bf{D}}_d}{\bf{B}}d\Omega }
   * \f]
   *
   */
  struct OpAssembleLhsDiagLin : public OpAssembleLhs {

    FTensor::Tensor2<double, 3, 3> diffDiff;

    OpAssembleLhsDiagLin(const string field_name_row,
                         const string field_name_col,
                         boost::shared_ptr<CommonData> common_data,
                         BlockData &block_data)
        : OpAssembleLhs(field_name_row, field_name_col, common_data,
                        block_data) {
      sYmm = true;
      diagonalBlock = true;
    };

    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  /** * @brief Assemble K *
   * \f[
   * {\bf{K}} = \int\limits_\Omega  {{{\bf{B}}^T}{{\bf{D}}_d}{\bf{B}}d\Omega }
   * \f]
   *
   */
  struct OpAssembleLhsDiagNonLin : public OpAssembleLhs {

    FTensor::Tensor2<double, 3, 3> diffDiff;

    OpAssembleLhsDiagNonLin(const string field_name_row,
                            const string field_name_col,
                            boost::shared_ptr<CommonData> common_data,
                            BlockData &block_data)
        : OpAssembleLhs(field_name_row, field_name_col, common_data,
                        block_data) {
      sYmm = false;
      diagonalBlock = true;
    };

    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);
  };

  struct OpAssembleRhs : public UserDataOperator {

    boost::shared_ptr<CommonData> commonData;
    BlockData &blockData;
    int nbRows; ///< number of dofs on row
    int nbIntegrationPts;

    VectorDouble locVec;

    OpAssembleRhs(const string field_name,
                  boost::shared_ptr<CommonData> common_data,
                  BlockData &block_data)
        : UserDataOperator(field_name, UserDataOperator::OPROW),
          commonData(common_data), blockData(block_data){};

    MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data);

    virtual MoFEMErrorCode iNtegrate(EntData &data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    };

    MoFEMErrorCode aSsemble(EntData &data);
  };

  struct OpAssembleRhsVelocityLin : public OpAssembleRhs {

    OpAssembleRhsVelocityLin(const string field_name,
                             boost::shared_ptr<CommonData> common_data,
                             BlockData &block_data)
        : OpAssembleRhs(field_name, common_data, block_data){};

    /**
     * \brief Integrate local entity vector
     * @param  data entity data on element row
     * @return      error code
     */
    MoFEMErrorCode iNtegrate(EntData &data);
  };

  struct OpAssembleRhsVelocityNonLin : public OpAssembleRhs {

    OpAssembleRhsVelocityNonLin(const string field_name,
                                boost::shared_ptr<CommonData> common_data,
                                BlockData &block_data)
        : OpAssembleRhs(field_name, common_data, block_data){};

    /**
     * \brief Integrate local entity vector
     * @param  data entity data on element row
     * @return      error code
     */
    MoFEMErrorCode iNtegrate(EntData &data);
  };

  struct OpAssembleRhsPressure : public OpAssembleRhs {

    OpAssembleRhsPressure(const string field_name,
                          boost::shared_ptr<CommonData> common_data,
                          BlockData &block_data)
        : OpAssembleRhs(field_name, common_data, block_data){};

    /**
     * \brief Integrate local constrains vector
     */
    MoFEMErrorCode iNtegrate(EntData &data);
  };

  struct OpCalcPressureDrag : public FaceUserDataOperator {

    boost::shared_ptr<CommonData> &commonData;
    BlockData &blockData;

    OpCalcPressureDrag(boost::shared_ptr<CommonData> &common_data,
                       BlockData &block_data)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              "P", UserDataOperator::OPROW),
          commonData(common_data), blockData(block_data) {
      doVertices = true;
      doEdges = false;
      doQuads = false;
      doTris = false;
      doTets = false;
      doPrisms = false;
    };

    PetscErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;
      if (type != MBVERTEX)
        PetscFunctionReturn(0);
      // double def_VAL[9];
      // bzero(def_VAL, 9 * sizeof(double));

      // if (blockData.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      //     blockData.tEts.end()) {
      //   MoFEMFunctionReturnHot(0);
      // }

      auto t_p = getFTensor0FromVec(*commonData->pPtr);
      const int nb_gauss_pts = commonData->pPtr->size();
      auto t_normal = getFTensor1NormalsAtGaussPts();
      // auto t_normal = getFTensor1Normal();

      FTensor::Index<'i', 3> i;

      for (int gg = 0; gg != nb_gauss_pts; gg++) {

        double nrm2 = sqrt(t_normal(i) * t_normal(i));
        t_normal(i) = t_normal(i) / nrm2;

        double w = getArea() * getGaussPts()(2, gg);

        // if (getHoGaussPtsDetJac().size() > 0) {
        //   w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
        // }

        for (int dd = 0; dd != 3; dd++) {
          commonData->pressureDrag[dd] += w * t_p * t_normal(dd);
        }

        ++t_p;
        ++t_normal;
      }

      MoFEMFunctionReturn(0);
    }
  };

  struct OpCalcViscousDrag : public FaceUserDataOperator {

    boost::shared_ptr<CommonData> &commonData;
    BlockData &blockData;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> &sideFe;

    OpCalcViscousDrag(
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> &side_fe,
        boost::shared_ptr<CommonData> &common_data, BlockData &block_data)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              "U", UserDataOperator::OPROW),
          sideFe(side_fe), commonData(common_data), blockData(block_data) {
      doVertices = true;
      doEdges = false;
      doQuads = false;
      doTris = false;
      doTets = false;
      doPrisms = false;
    };

    PetscErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (type != MBVERTEX)
        PetscFunctionReturn(0);

      // double def_VAL[9];
      // bzero(def_VAL, 9 * sizeof(double));

      // if (blockData.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      //     blockData.tEts.end()) {
      //   MoFEMFunctionReturnHot(0);
      // }

      CHKERR loopSideVolumes("NAVIER_STOKES", *sideFe);

      // commonData->getBlockData(blockData);

      auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradDispPtr);
      const int nb_gauss_pts = commonData->gradDispPtr->size2();
      auto t_normal = getFTensor1NormalsAtGaussPts();
      // auto t_normal = getFTensor1Normal();

      FTensor::Index<'i', 3> i;

      for (int gg = 0; gg != nb_gauss_pts; gg++) {

        double nrm2 = sqrt(t_normal(i) * t_normal(i));
        t_normal(i) = t_normal(i) / nrm2;

        double w = getArea() * getGaussPts()(2, gg) * blockData.fluidViscosity;
        // if (getHoGaussPtsDetJac().size() > 0) {
        //   w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
        // }

        for (int ii = 0; ii != 3; ii++) {
          for (int jj = 0; jj != 3; jj++) {
            commonData->viscousDrag[ii] +=
                -w * (t_u_grad(ii, jj) + t_u_grad(jj, ii)) * t_normal(jj);
          }
        }

        ++t_u_grad;
        ++t_normal;
      }
      MoFEMFunctionReturn(0);
    }
  };

  struct OpPostProcVorticity : public UserDataOperator {

    boost::shared_ptr<CommonData> &commonData;
    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;
    BlockData &blockData;

    OpPostProcVorticity(moab::Interface &post_proc_mesh,
                        std::vector<EntityHandle> &map_gauss_pts,
                        boost::shared_ptr<CommonData> &common_data,
                        BlockData &block_data)
        : UserDataOperator("U", UserDataOperator::OPROW),
          commonData(common_data), postProcMesh(post_proc_mesh),
          mapGaussPts(map_gauss_pts), blockData(block_data) {
      doVertices = true;
      doEdges = false;
      doQuads = false;
      doTris = false;
      doTets = false;
      doPrisms = false;
    };

    PetscErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;
      if (type != MBVERTEX)
        PetscFunctionReturn(0);
      double def_VAL[9];
      bzero(def_VAL, 9 * sizeof(double));

      if (blockData.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
          blockData.tEts.end()) {
        MoFEMFunctionReturnHot(0);
      }
      // commonData->getBlockData(blockData);

      Tag th_vorticity;
      Tag th_q;
      Tag th_l2;
      CHKERR postProcMesh.tag_get_handle("VORTICITY", 3, MB_TYPE_DOUBLE,
                                         th_vorticity,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
      CHKERR postProcMesh.tag_get_handle("Q", 1, MB_TYPE_DOUBLE, th_q,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
      CHKERR postProcMesh.tag_get_handle("L2", 1, MB_TYPE_DOUBLE, th_l2,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

      auto t_u_grad = getFTensor2FromMat<3, 3>(*commonData->gradDispPtr);
      // auto p = getFTensor0FromVec(*commonData->pPtr);

      const int nb_gauss_pts = commonData->gradDispPtr->size2();
      // const int nb_gauss_pts2 = commonData->pPtr->size();

      // const double lambda = commonData->lAmbda;
      // const double mu = commonData->mU;

      // FTensor::Index<'i', 3> i;
      // FTensor::Index<'j', 3> j;
      // FTensor::Index<'j', 3> k;
      FTensor::Tensor1<double, 3> vorticity;
      // FTensor::Tensor2<double,3,3> t_s;
      double q;
      double l2;
      // FTensor::Tensor2<double, 3, 3> stress;
      MatrixDouble S;
      MatrixDouble Omega;
      MatrixDouble M;

      S.resize(3, 3);
      Omega.resize(3, 3);
      M.resize(3, 3);

      for (int gg = 0; gg != nb_gauss_pts; gg++) {

        vorticity(0) = t_u_grad(2, 1) - t_u_grad(1, 2);
        vorticity(1) = t_u_grad(0, 2) - t_u_grad(2, 0);
        vorticity(2) = t_u_grad(1, 0) - t_u_grad(0, 1);

        q = 0;
        for (int i = 0; i != 3; i++) {
          for (int j = 0; j != 3; j++) {
            q += -0.5 * t_u_grad(i, j) * t_u_grad(j, i);
          }
        }
        for (int i = 0; i != 3; i++) {
          for (int j = 0; j != 3; j++) {
            S(i, j) = 0.5 * (t_u_grad(i, j) + t_u_grad(j, i));
            Omega(i, j) = 0.5 * (t_u_grad(i, j) - t_u_grad(j, i));
            M(i, j) = 0.0;
          }
        }

        for (int i = 0; i != 3; i++) {
          for (int j = 0; j != 3; j++) {
            for (int k = 0; k != 3; k++) {
              M(i, j) += S(i, k) * S(k, j) + Omega(i, k) * Omega(k, j);
            }
          }
        }

        MatrixDouble eigen_vectors = M;
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

        // prin_stress_vect.clear();
        VectorDouble prin_vals_vect(3);
        prin_vals_vect.clear();

        int ii = 0;
        for (map<double, int>::reverse_iterator mit = eigen_sort.rbegin();
             mit != eigen_sort.rend(); mit++) {
          prin_vals_vect[ii] = eigen_values[mit->second];
          // for (int dd = 0; dd != 3; dd++) {
          //   prin_stress_vect(ii, dd) = eigen_vectors.data()[3 * mit->second +
          //   dd];
          // }
          ii++;
        }

        l2 = prin_vals_vect[1];
        // cout << prin_vals_vect << endl;
        // cout << "-0.5 sum: " << -0.5 * (prin_vals_vect[0] + prin_vals_vect[1]
        // + prin_vals_vect[2]) << endl; cout << "q: " << q << endl;

        // t_s(i,j) = 0.5*(t)

        // vorticity(0) = t_u_grad(1, 2) - t_u_grad(2, 1);
        // vorticity(1) = t_u_grad(2, 0) - t_u_grad(0, 2);
        // vorticity(2) = t_u_grad(0, 1) - t_u_grad(1, 0);

        CHKERR postProcMesh.tag_set_data(th_vorticity, &mapGaussPts[gg], 1,
                                         &vorticity(0));
        CHKERR postProcMesh.tag_set_data(th_q, &mapGaussPts[gg], 1, &q);
        CHKERR postProcMesh.tag_set_data(th_l2, &mapGaussPts[gg], 1, &l2);
        ++t_u_grad;
      }

      MoFEMFunctionReturn(0);
    }
  };
};

struct DirichletVelocityBc : public DirichletDisplacementBc {

  DirichletVelocityBc(MoFEM::Interface &m_field, const std::string &field_name,
                      Mat aij, Vec x, Vec f)
      : DirichletDisplacementBc(m_field, field_name, aij, x, f){};

  DirichletVelocityBc(MoFEM::Interface &m_field, const std::string &field_name)
      : DirichletDisplacementBc(m_field, field_name){};

  MoFEMErrorCode iNitalize();

  // MoFEMErrorCode postProcess();
};

struct DirichletPressureBc : public DirichletDisplacementBc {

  DirichletPressureBc(MoFEM::Interface &m_field, const std::string &field_name,
                      Mat aij, Vec x, Vec f)
      : DirichletDisplacementBc(m_field, field_name, aij, x, f){};

  DirichletPressureBc(MoFEM::Interface &m_field, const std::string &field_name)
      : DirichletDisplacementBc(m_field, field_name){};

  MoFEMErrorCode iNitalize();

  // MoFEMErrorCode postProcess();
};

#endif //__NAVIERSTOKESELEMENT_HPP__

// MoFEMErrorCode DirichletVelocityBc::postProcess() {
//   MoFEMFunctionBegin;

//   switch (ts_ctx) {
//   case CTX_TSSETIFUNCTION: {
//     snes_ctx = CTX_SNESSETFUNCTION;
//     snes_x = ts_u;
//     snes_f = ts_F;
//     break;
//   }
//   case CTX_TSSETIJACOBIAN: {
//     snes_ctx = CTX_SNESSETJACOBIAN;
//     snes_B = ts_B;
//     break;
//   }
//   default:
//     break;
//   }

//   if (snes_B) {
//     CHKERR MatAssemblyBegin(snes_B, MAT_FINAL_ASSEMBLY);
//     CHKERR MatAssemblyEnd(snes_B, MAT_FINAL_ASSEMBLY);
//     CHKERR MatZeroRowsColumns(snes_B, dofsIndices.size(),
//                               dofsIndices.empty() ? PETSC_NULL
//                                                   : &dofsIndices[0],
//                               dIag, PETSC_NULL, PETSC_NULL);
//   }
//   if (snes_f) {
//     CHKERR VecAssemblyBegin(snes_f);
//     CHKERR VecAssemblyEnd(snes_f);
//     for (std::vector<int>::iterator vit = dofsIndices.begin();
//          vit != dofsIndices.end(); vit++) {
//       CHKERR VecSetValue(snes_f, *vit, 0., INSERT_VALUES);
//     }
//     CHKERR VecAssemblyBegin(snes_f);
//     CHKERR VecAssemblyEnd(snes_f);
//   }

//   MoFEMFunctionReturn(0);
// }

// MoFEMErrorCode DirichletPressureBc::postProcess() {
//   MoFEMFunctionBegin;

//   switch (ts_ctx) {
//   case CTX_TSSETIFUNCTION: {
//     snes_ctx = CTX_SNESSETFUNCTION;
//     snes_x = ts_u;
//     snes_f = ts_F;
//     break;
//   }
//   case CTX_TSSETIJACOBIAN: {
//     snes_ctx = CTX_SNESSETJACOBIAN;
//     snes_B = ts_B;
//     break;
//   }
//   default:
//     break;
//   }

//   if (snes_B) {
//     CHKERR MatAssemblyBegin(snes_B, MAT_FINAL_ASSEMBLY);
//     CHKERR MatAssemblyEnd(snes_B, MAT_FINAL_ASSEMBLY);
//     CHKERR MatZeroRowsColumns(snes_B, dofsIndices.size(),
//                               dofsIndices.empty() ? PETSC_NULL
//                                                   : &dofsIndices[0],
//                               dIag, PETSC_NULL, PETSC_NULL);
//   }
//   if (snes_f) {
//     CHKERR VecAssemblyBegin(snes_f);
//     CHKERR VecAssemblyEnd(snes_f);
//     for (std::vector<int>::iterator vit = dofsIndices.begin();
//          vit != dofsIndices.end(); vit++) {
//       CHKERR VecSetValue(snes_f, *vit, 0., INSERT_VALUES);
//     }
//     CHKERR VecAssemblyBegin(snes_f);
//     CHKERR VecAssemblyEnd(snes_f);
//   }

//   MoFEMFunctionReturn(0);
// }