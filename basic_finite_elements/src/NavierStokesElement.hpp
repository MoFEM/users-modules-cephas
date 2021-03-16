/**
 * \file NavierStokesElement.hpp
 * \example NavierStokesElement.hpp
 *
 * \brief Implementation of operators for fluid flow
 *
 * Implementation of operators for computations of the fluid flow, governed by
 * for Stokes and Navier-Stokes equations, and computation of the drag force
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
    double fluidViscosity;
    double fluidDensity;
    double inertiaCoef;
    double viscousCoef;
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
    boost::shared_ptr<MatrixDouble> shearDragTract;
    boost::shared_ptr<MatrixDouble> totalDragTract;

    SmartPetscObj<Vec> pressureDragForceVec;
    SmartPetscObj<Vec> shearDragForceVec;
    SmartPetscObj<Vec> totalDragForceVec;

    SmartPetscObj<Vec> volumeFluxVec;

    std::map<int, BlockData> setOfBlocksData;
    std::map<int, BlockData> setOfFacesData;

    CommonData(MoFEM::Interface &m_field) {

      gradVelPtr = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      velPtr = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      pressPtr = boost::shared_ptr<VectorDouble>(new VectorDouble());

      pressureDragTract = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      shearDragTract = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      totalDragTract = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      int vec_size;
      if (m_field.get_comm_rank() == 0)
        vec_size = 3;
      else
        vec_size = 0;

      pressureDragForceVec = createSmartVectorMPI(m_field.get_comm(), vec_size, 3);
      shearDragForceVec = createSmartVectorMPI(m_field.get_comm(), vec_size, 3);
      totalDragForceVec = createSmartVectorMPI(m_field.get_comm(), vec_size, 3);

      volumeFluxVec = createSmartVectorMPI(m_field.get_comm(), vec_size, 3);
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

  /**
   * @brief Function for setting up element for solving Navier-Stokes equations
   *
   * @param  m_field                    MoFEM interface
   * @param  element_name               Name of the element
   * @param  velocity_field_name        Name of the velocity field
   * @param  pressure_field_name        Name of the pressure field
   * @param  mesh_field_name            Name for mesh node positions field
   * @param  ents                       Range of entities to be added to element
   * @return                            Error code
   *
   */
  static MoFEMErrorCode addElement(MoFEM::Interface &m_field,
                                               const string element_name,
                                               const string velocity_field_name,
                                               const string pressure_field_name,
                                               const string mesh_field_name,
                                               const int dim = 3,
                                               Range *ents = nullptr) {
    MoFEMFunctionBegin;

    CHKERR m_field.add_finite_element(element_name);

    CHKERR m_field.modify_finite_element_add_field_row(element_name,
                                                       velocity_field_name);
    CHKERR m_field.modify_finite_element_add_field_col(element_name,
                                                       velocity_field_name);
    CHKERR m_field.modify_finite_element_add_field_data(element_name,
                                                        velocity_field_name);

    CHKERR m_field.modify_finite_element_add_field_row(element_name,
                                                       pressure_field_name);
    CHKERR m_field.modify_finite_element_add_field_col(element_name,
                                                       pressure_field_name);
    CHKERR m_field.modify_finite_element_add_field_data(element_name,
                                                        pressure_field_name);

    CHKERR m_field.modify_finite_element_add_field_data(element_name,
                                                        mesh_field_name);

    if (ents != nullptr) {
      CHKERR m_field.add_ents_to_finite_element_by_dim(*ents, dim,
                                                       element_name);
    } else {
      CHKERR m_field.add_ents_to_finite_element_by_dim(0, dim, element_name);
    }

    MoFEMFunctionReturn(0);
  }

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
      return 2 * order_data;
    }
  };

  struct FaceRule {
    int operator()(int order_row, int order_col, int order_data) const {
      return order_data + 2;
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

  /** * @brief Assemble off-diagonal block of the LHS *
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

  struct OpCalcDragForce : public FaceUserDataOperator {

    boost::shared_ptr<CommonData> commonData;
    BlockData &blockData;

    OpCalcDragForce(const string field_name,
                    boost::shared_ptr<CommonData> &common_data,
                    BlockData &block_data)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
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
        const string field_name,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> &side_fe,
        std::string side_fe_name, boost::shared_ptr<CommonData> &common_data,
        BlockData &block_data)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
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

    OpPostProcDrag(const string field_name, moab::Interface &post_proc_mesh,
                   std::vector<EntityHandle> &map_gauss_pts,
                   boost::shared_ptr<CommonData> &common_data,
                   BlockData &block_data)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
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
};

#endif //__NAVIERSTOKESELEMENT_HPP__