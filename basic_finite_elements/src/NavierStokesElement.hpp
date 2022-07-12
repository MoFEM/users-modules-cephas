/**
 * \file NavierStokesElement.hpp
 * \example NavierStokesElement.hpp
 *
 * \brief Implementation of operators for fluid flow
 *
 * Implementation of operators for computations of the fluid flow, governed by
 * for Stokes and Navier-Stokes equations, and computation of the drag force
 */

/* MIT License
 *
 * Copyright (c) 2022
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __NAVIERSTOKESELEMENT_HPP__
#define __NAVIERSTOKESELEMENT_HPP__

#ifndef __BASICFINITEELEMENTS_HPP__
#include <BasicFiniteElements.hpp>
#endif // __BASICFINITEELEMENTS_HPP__

using namespace boost::numeric;

/**
   * @brief Element for simulating viscous fluid flow
*/
struct NavierStokesElement {

  using UserDataOperator =
      MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator;
  using FaceUserDataOperator =
      MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
  using EntData = EntitiesFieldData::EntData;

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
   * @brief Setting up elements
   *
   * This functions adds element to the database, adds provided fields to rows
   * and columns of the element, provides access of the element to the fields
   * data and adds entities of particular dimension (or a given range of
   * entities to the element)
   *
   * @param  m_field                    MoFEM interface
   * @param  element_name               Name of the element
   * @param  velocity_field_name        Name of the velocity field
   * @param  pressure_field_name        Name of the pressure field
   * @param  mesh_field_name            Name for mesh node positions field
   * @param  ents                       Range of entities to be added to element
   * @return                            Error code
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

  /**
   * @brief Setting up operators for solving Navier-Stokes equations
   * 
   * Pushes operators for solving Navier-Stokes equations to pipelines of RHS
   * and LHS element instances
   *
   * @param  feRhs                      pointer to RHS element instance
   * @param  feLhs                      pointer to LHS element instance
   * @param  velocity_field             name of the velocity field
   * @param  pressure_field             name of the pressure field
   * @param  common_data                pointer to common data object
   * @param  type                       type of entities in the domain
   * @return                            Error code
   */
  static MoFEMErrorCode setNavierStokesOperators(
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> feRhs,
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> feLhs,
      const std::string velocity_field, const std::string pressure_field,
      boost::shared_ptr<CommonData> common_data, const EntityType type = MBTET);

  /**
   * @brief Setting up operators for solving Stokes equations
   *
   * Pushes operators for solving Stokes equations to pipelines of RHS
   * and LHS element instances
   *
   * @param  feRhs                      pointer to RHS element instance
   * @param  feLhs                      pointer to LHS element instance
   * @param  velocity_field             name of the velocity field
   * @param  pressure_field             name of the pressure field
   * @param  common_data                pointer to common data object
   * @param  type                       type of entities in the domain
   * @return                            Error code
   */
  static MoFEMErrorCode setStokesOperators(
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> feRhs,
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> feLhs,
      const std::string velocity_field, const std::string pressure_field,
      boost::shared_ptr<CommonData> common_data, const EntityType type = MBTET);

  /**
   * @brief Setting up operators for calculating drag force on the solid surface
   *
   * Pushes operators for caluclating drag force components on the fluid-solid
   * interface
   *
   * @param  dragFe                   pointer to face element instance
   * @param  sideDragFe               pointer to volume on side element instance
   * @param  velocity_field           name of the velocity field
   * @param  pressure_field           name of the pressure field
   * @param  common_data              pointer to common data object
   * @return                          Error code
   */
  static MoFEMErrorCode setCalcDragOperators(
      boost::shared_ptr<FaceElementForcesAndSourcesCore> dragFe,
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideDragFe,
      std::string side_fe_name, const std::string velocity_field,
      const std::string pressure_field,
      boost::shared_ptr<CommonData> common_data);

  /**
   * @brief Setting up operators for post processing output of drag traction
   *
   * Pushes operators for post processing ouput of drag traction components on
   * the fluid-solid interface
   *
   * @param  dragFe                   pointer to face element instance
   * @param  sideDragFe               pointer to volume on side element instance
   * @param  velocity_field           name of the velocity field
   * @param  pressure_field           name of the pressure field
   * @param  common_data              pointer to common data object
   * @return                          Error code
   */
  static MoFEMErrorCode setPostProcDragOperators(
      boost::shared_ptr<PostProcFaceOnRefinedMesh> postProcDragPtr,
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideDragFe,
      std::string side_fe_name, const std::string velocity_field,
      const std::string pressure_field,
      boost::shared_ptr<CommonData> common_data);

  /**
   * @brief Setting up operators for calculation of volume flux
   */
  static MoFEMErrorCode setCalcVolumeFluxOperators(
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_flux_ptr,
      const std::string velocity_field,
      boost::shared_ptr<CommonData> common_data, const EntityType type = MBTET);

  /**
   * \brief Set integration rule to volume elements
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

  /**
   * \brief Base class for operators assembling LHS
   */
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

  /**
   * @brief Assemble off-diagonal block of the LHS 
   * Operator for assembling off-diagonal block of the LHS
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

  /**
   * @brief Assemble linear (symmetric) part of the diagonal block of the LHS
   * Operator for assembling linear (symmetric) part of the diagonal block of
   * the LHS
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

  /**
   * @brief Assemble non-linear (non-symmetric) part of the diagonal block of
   * the LHS Operator for assembling non-linear (non-symmetric) part of the
   * diagonal block of the LHS
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

  /**
   * \brief Base class for operators assembling RHS
   */
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

  /**
   * @brief Assemble linear part of the velocity component of the RHS vector
   *
   * Operator for assembling linear part of the velocity component of the RHS
   * vector: 
   * \f[ \mathbf{R}^{\textrm{S}}_{\mathbf{u}} =
   * C_\text{V}\int\limits_{\Omega}\nabla\mathbf{u}\mathbin{:}\nabla\mathbf{v}\,
   * d\Omega
   * \f]
   * where \f$C_\text{V}\f$ is the viscosity
   * coefficient: \f$C_\text{V}=\mu\f$ in the dimensional case and
   * \f$C_\text{V}=1\f$ in the non-dimensional case.
   */
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

  /**
   * @brief Assemble non-linear part of the velocity component of the RHS vector
   *
   * Operator for assembling non-linear part of the velocity component of the
   * RHS vector: 
   * \f[
   * \mathbf{R}^{\textrm{NS}}_{\mathbf{u}} =
   * C_\text{I}\int\limits_\Omega \left(\mathbf{u}\cdot\nabla\right)\mathbf{u}
   * \cdot\mathbf{v} \,d\Omega, 
   * \f] 
   * where \f$C_\text{I}\f$ is the inertia
   * coefficient: \f$C_\text{V}=\rho\f$ in the dimensional case and
   * \f$C_\text{V}=\mathcal{R}\f$ in the non-dimensional case.
   */
  struct OpAssembleRhsVelocityNonLin : public OpAssembleRhs {

    OpAssembleRhsVelocityNonLin(const string field_name,
                                boost::shared_ptr<CommonData> common_data,
                                BlockData &block_data)
        : OpAssembleRhs(field_name, common_data, block_data){};

    MoFEMErrorCode iNtegrate(EntData &data);
  };

  /**
   * @brief Assemble the pressure component of the RHS vector
   * 
   * Operator for assembling pressure component of the RHS vector:
   * \f[
   * \mathbf{R}^{\textrm{S}}_{p} = -\int\limits_{\Omega}p\,
   * \nabla\cdot\mathbf{v} \, d\Omega
   * \f]
   */
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

  /**
   * @brief Calculate drag force on the fluid-solid interface
   *
   * Operator fo calculating drag force components on the fluid-solid interface.
   * Integrates components of the drag traction:
   * \f[
   * \mathbf{F}_{\textrm{D}} =
   * -\int\limits_{\Gamma_{\textrm{S}}}\left(-p\mathbf{I} +
   * \mu\left(\nabla\mathbf{u}+\mathbf{u}^{\intercal}\right)\right) \, d\Gamma
   * \f]
   */
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

  /**
   * @brief Calculate drag traction on the fluid-solid interface
   *
   * Operator fo calculating drag traction on the fluid-solid interface
   * \f[
   * \mathbf{t} = -p\mathbf{I} +
   * \mu\left(\nabla\mathbf{u}+\mathbf{u}^{\intercal}\right) 
   * \f]
   */
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

  /**
   * @brief Post processing output of drag traction on the fluid-solid interface
   *
   */
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

  /**
   * @brief Post processing output of the vorticity criterion levels
   *
   */
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

  /**
   * @brief calculating volumetric flux
   *
   */
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