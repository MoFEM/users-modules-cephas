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

  struct DimScales {
    double length;
    double velocity;
    double pressure;
    double reNumber;
    DimScales() : length(-1), velocity(-1), pressure(-1), reNumber(-1) {}
  };

  struct BlockData {
    int iD;
    // int oRder;
    double fluidViscosity;
    double fluidDensity;
    double inertiaCoef;
    double viscousCoef;
    DimScales dimScales;
    Range eNts;
    BlockData()
        : iD(-1), fluidViscosity(-1), fluidDensity(-1), inertiaCoef(-1),
          viscousCoef(-1) {}
  };

  struct CommonData {

    MatrixDouble invJac;

    boost::shared_ptr<MatrixDouble> gradVelPtr;
    boost::shared_ptr<MatrixDouble> velPtr;
    boost::shared_ptr<VectorDouble> pressPtr;

    boost::shared_ptr<MatrixDouble> pressureDragTract;
    boost::shared_ptr<MatrixDouble> viscousDragTract;
    boost::shared_ptr<MatrixDouble> totalDragTract;

    VectorDouble3 pressureDragForce;
    VectorDouble3 viscousDragForce;
    VectorDouble3 totalDragForce;
    VectorDouble3 volumeFlux;

    std::map<int, BlockData> setOfBlocksData;
    std::map<int, BlockData> setOfFacesData;

    CommonData() {

      gradVelPtr = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      velPtr = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      pressPtr = boost::shared_ptr<VectorDouble>(new VectorDouble());

      pressureDragTract = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      viscousDragTract = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      totalDragTract = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      pressureDragForce = VectorDouble3(3);
      viscousDragForce = VectorDouble3(3);
      totalDragForce = VectorDouble3(3);
      volumeFlux = VectorDouble3(3);

      pressureDragForce.clear();
      viscousDragForce.clear();
      totalDragForce.clear();
      volumeFlux.clear();
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
      boost::shared_ptr<CommonData> common_data, const EntityType type = MBTET);

  static MoFEMErrorCode setStokesOperators(
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> feRhs,
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> feLhs,
      const std::string velocity_field, const std::string pressure_field,
      boost::shared_ptr<CommonData> common_data, const EntityType type = MBTET);

  static MoFEMErrorCode setCalcDragOperators(
      boost::shared_ptr<FaceElementForcesAndSourcesCore> dragFe,
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideDragFe,
      std::string side_fe_name, const std::string velocity_field,
      const std::string pressure_field,
      boost::shared_ptr<CommonData> common_data);

  static MoFEMErrorCode setPostProcDragOperators(
      boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcDragPtr,
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideDragFe,
      std::string side_fe_name, const std::string velocity_field,
      const std::string pressure_field,
      boost::shared_ptr<CommonData> common_data);

  static MoFEMErrorCode setCalcVolumeFluxOperators(
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_flux_ptr,
      const std::string velocity_field,
      boost::shared_ptr<CommonData> common_data, const EntityType type = MBTET);

  /**
   * \brief Set integration rule to volume elements
   *
   * This rule is used to integrate \f$\nabla v \cdot \nabla u\f$, thus
   * if the approximation field and the testing field are polynomials of
   * order "p", then the rule for the exact integration is 2*(p-1).
   *
   * Integration rule is order of polynomial which is calculated exactly.
   * Finite element selects integration method based on return of this
   * function.
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

  struct OpCalcVolumeFlux : public UserDataOperator {

    boost::shared_ptr<CommonData> commonData;
    BlockData &blockData;
    int nbRows; ///< number of dofs on row
    int nbIntegrationPts;

    OpCalcVolumeFlux(const string field_name,
                  boost::shared_ptr<CommonData> common_data,
                  BlockData &block_data)
        : UserDataOperator(field_name, UserDataOperator::OPROW),
          commonData(common_data), blockData(block_data){};

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpCalcDragForce: public FaceUserDataOperator {

    boost::shared_ptr<CommonData> commonData;
    BlockData &blockData;

    OpCalcDragForce(boost::shared_ptr<CommonData> &common_data,
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

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpCalcDragTraction : public FaceUserDataOperator {

    boost::shared_ptr<CommonData> commonData;
    BlockData &blockData;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;
    std::string sideFeName;

    OpCalcDragTraction(
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> &side_fe,
        std::string side_fe_name, boost::shared_ptr<CommonData> &common_data,
        BlockData &block_data)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              "U", UserDataOperator::OPROW),
          sideFe(side_fe), sideFeName(side_fe_name), commonData(common_data),
          blockData(block_data) {
      doVertices = true;
      doEdges = false;
      doQuads = false;
      doTris = false;
      doTets = false;
      doPrisms = false;
    };

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpPostProcDrag : public FaceUserDataOperator {

    boost::shared_ptr<CommonData> commonData;
    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;
    BlockData &blockData;

    OpPostProcDrag(
        moab::Interface &post_proc_mesh,
        std::vector<EntityHandle> &map_gauss_pts, 
        boost::shared_ptr<CommonData> &common_data,
        BlockData &block_data)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              "U", UserDataOperator::OPROW),
          commonData(common_data), postProcMesh(post_proc_mesh),
          mapGaussPts(map_gauss_pts), blockData(block_data) {
      doVertices = true;
      doEdges = false;
      doQuads = false;
      doTris = false;
      doTets = false;
      doPrisms = false;
    };

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpPostProcVorticity : public UserDataOperator {

    boost::shared_ptr<CommonData> commonData;
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

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
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