/** \file SpringElements.hpp
  \brief Header file for spring element implementation
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

#ifndef __SPRINGELEMENT_HPP__
#define __SPRINGELEMENT_HPP__

using VolOnSideUserDataOperator =
      MoFEM::VolumeElementForcesAndSourcesCoreOnSide::UserDataOperator;
using EntData = DataForcesAndSourcesCore::EntData;
using FaceDataOperator =
      MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator;
/** \brief Set of functions declaring elements and setting operators
 * to apply spring boundary condition
 */
struct MetaSpringBC {
  
  struct BlockOptionDataSprings {
    int iD;

    double springStiffnessNormal;
    double springStiffnessTangent;

    Range tRis;

    BlockOptionDataSprings()
        : springStiffnessNormal(-1), springStiffnessTangent(-1) {}
  };

  struct DataAtIntegrationPtsSprings
      : public boost::enable_shared_from_this<DataAtIntegrationPtsSprings> {

    boost::shared_ptr<MatrixDouble> gradDispPtr =
        boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    boost::shared_ptr<MatrixDouble> xAtPts =
        boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    boost::shared_ptr<MatrixDouble> xInitAtPts =
        boost::shared_ptr<MatrixDouble>(new MatrixDouble());

    boost::shared_ptr<MatrixDouble> hMat =
        boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    boost::shared_ptr<MatrixDouble> FMat =
        boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    boost::shared_ptr<MatrixDouble> HMat =
        boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    boost::shared_ptr<MatrixDouble> invHMat =
        boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    boost::shared_ptr<VectorDouble> detHVec =
        boost::shared_ptr<VectorDouble>(new VectorDouble());

    MatrixDouble tangent1;
    MatrixDouble tangent2;
    MatrixDouble normalVector;

    double springStiffnessNormal;
    double springStiffnessTangent;

    Range forcesOnlyOnEntitiesRow;
    Range forcesOnlyOnEntitiesCol;

    DataForcesAndSourcesCore::EntData *faceRowData;

    std::map<int, BlockOptionDataSprings> mapSpring;
    //   ~DataAtIntegrationPtsSprings() {}
    DataAtIntegrationPtsSprings(MoFEM::Interface &m_field)
        : mField(m_field), faceRowData(nullptr) {

      ierr = setBlocks();
      CHKERRABORT(PETSC_COMM_WORLD, ierr);
    }

    MoFEMErrorCode getParameters() {
      MoFEMFunctionBegin; // They will be overwritten by BlockData
      CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Problem", "none");

      ierr = PetscOptionsEnd();
      CHKERRQ(ierr);
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode getBlockData(BlockOptionDataSprings &data) {
      MoFEMFunctionBegin;

      springStiffnessNormal = data.springStiffnessNormal;
      springStiffnessTangent = data.springStiffnessTangent;

      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode setBlocks() {
      MoFEMFunctionBegin;

      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
        if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {

          const int id = bit->getMeshsetId();
          mapSpring[id].tRis.clear();
          CHKERR mField.get_moab().get_entities_by_type(
              bit->getMeshset(), MBTRI, mapSpring[id].tRis, true);

          std::vector<double> attributes;
          bit->getAttributes(attributes);
          if (attributes.size() < 2) {
            SETERRQ1(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                     "Springs should have 2 attributes but there is %d",
                     attributes.size());
          }
          mapSpring[id].iD = id;
          mapSpring[id].springStiffnessNormal = attributes[0];
          mapSpring[id].springStiffnessTangent = attributes[1];

          // Print spring blocks after being read
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "\nSpring block %d\n", id);
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "\tNormal stiffness %3.4g\n",
                             attributes[0]);
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "\tTangent stiffness %3.4g\n",
                             attributes[1]);
        }
      }

      MoFEMFunctionReturn(0);
    }

  private:
    MoFEM::Interface &mField;
  };

/**
   * @brief Base class for LHS-operators (material) on side volumes
   *
   */
  struct SpringALEMaterialVolOnSideLhs
      : public VolOnSideUserDataOperator {

    MatrixDouble NN;

    boost::shared_ptr<DataAtIntegrationPtsSprings> dataAtSpringPts;
    
    VectorInt rowIndices;
    VectorInt colIndices;

    int row_nb_dofs;
    int col_nb_dofs;
    int nb_gauss_pts;

    int nb_base_fun_row;
    int nb_base_fun_col;

    bool diagonal_block;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
    virtual MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }
    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    SpringALEMaterialVolOnSideLhs(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_spring_pts)
        : VolOnSideUserDataOperator(field_name_1, field_name_2,
                                    UserDataOperator::OPROWCOL),
          dataAtSpringPts(data_at_spring_pts) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };

/**
   * @brief LHS-operator (material configuration) on the side volume
   *
   * Computes the linearisation of the material component
   * with respect to a variation of spatial coordinates on the side volume.
   */
  struct SpringALEMaterialVolOnSideLhs_dX_dx
      : public SpringALEMaterialVolOnSideLhs {

    /**
     * @brief Integrates over a face contribution from a side volume
     *
     * Computes linearisation of the material component
     * with respect to a variation of spatial coordinates:
     *
     * \f[
     * \textrm{D} \delta W^\text{(side volume)}_p(\mathbf{x}, \mathbf{X},
     * \delta\mathbf{x})
     * [\Delta\mathbf{x}] = -\int\limits_{\mathcal{T}_{\xi}} p
     * \left\{\left[
     * \frac{\partial\Delta\mathbf{x}}{\partial\boldsymbol{\chi}}\,\mathbf{H}^{-1}
     * \right]^{\intercal}\cdot\left(\frac{\partial\mathbf{X}}{\partial\xi}
     * \times\frac{\partial\mathbf{X}}{\partial\eta}\right)\right\}
     * \cdot \delta\mathbf{X}\, \textrm{d}\xi\textrm{d}\eta
     * \f]
     *
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

    SpringALEMaterialVolOnSideLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_spring_pts)
        : SpringALEMaterialVolOnSideLhs(
              field_name_1, field_name_2, data_at_spring_pts) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  /**
   * @brief Base class for LHS-operators for pressure element (material
   * configuration)
   *
   * Linearisation of the material component with respect to
   * spatial and material coordinates consists of three parts, computed
   * by operators working on the face and on the side volume:
   *
   * \f[
   * \textrm{D} \delta W^\text{material}_p(\mathbf{x}, \mathbf{X},
   * \delta\mathbf{x})
   * [\Delta\mathbf{x}, \Delta\mathbf{X}] = \textrm{D} \delta
   * W^\text{(face)}_p(\mathbf{x}, \mathbf{X}, \delta\mathbf{x})
   * [\Delta\mathbf{X}] + \textrm{D} \delta
   * W^\text{(side volume)}_p(\mathbf{x}, \mathbf{X}, \delta\mathbf{x})
   * [\Delta\mathbf{x}] + \textrm{D} \delta W^\text{(side volume)}_p
   * (\mathbf{x}, \mathbf{X}, \delta\mathbf{x}) [\Delta\mathbf{X}]
   * \f]
   *
   */
  struct OpSpringALEMaterialLhs : public FaceDataOperator {

    boost::shared_ptr<DataAtIntegrationPtsSprings> dataAtSpringPts;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;
    std::string sideFeName;

    MatrixDouble NN;
    VectorInt rowIndices;
    VectorInt colIndices;

    int row_nb_dofs;
    int col_nb_dofs;
    int nb_gauss_pts;

    int nb_base_fun_row;
    int nb_base_fun_col;

    bool diagonal_block;

    virtual MoFEMErrorCode doWork(int row_side, int col_side,
                                  EntityType row_type, EntityType col_type,
                                  DataForcesAndSourcesCore::EntData &row_data,
                                  DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    virtual MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    OpSpringALEMaterialLhs(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_spring_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name)
        : FaceDataOperator(field_name_1, field_name_2,
                           FaceDataOperator::OPROWCOL),
          dataAtSpringPts(data_at_spring_pts), sideFe(side_fe), sideFeName(side_fe_name) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };

    /**
   * @brief LHS-operator for the pressure element (material configuration)
   *
   * Triggers loop over operators from the side volume
   *
   */
  struct OpSpringALEMaterialLhs_dX_dx
      : public OpSpringALEMaterialLhs {

    /*
     * Triggers loop over operators from the side volume
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    OpSpringALEMaterialLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name)
        : OpSpringALEMaterialLhs(field_name_1, field_name_2, data_at_pts,
                                       side_fe, side_fe_name) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };


/**
   * @brief LHS-operator for the pressure element (material configuration)
   *
   * Computes linearisation of the material component with respect to
   * material coordinates (also triggers a loop over operators
   * from the side volume).
   *
   */
  struct OpSpringALEMaterialLhs_dX_dX
      : public OpSpringALEMaterialLhs {

    /**
     * Integrates a contribution to the left-hand side and triggers a loop
     * over side volume operators.
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    /**
     * @brief Compute part of the left-hand side
     *
     * Computes the linearisation of the material component
     * with respect to a variation of material coordinates
     * \f$(\Delta\mathbf{X})\f$:
     *
     * \f[
     * \textrm{D} \delta W^\text{(face)}_p(\mathbf{x}, \mathbf{X},
     * \delta\mathbf{x})
     * [\Delta\mathbf{X}] = -\int\limits_{\mathcal{T}_{\xi}} p \,
     * \mathbf{F}^{\intercal}\cdot \left[ \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     *  \mathbf{X}}{\partial\eta}\times\delta\mathbf{x}\right)
     * -\frac{\partial\mathbf{X}}
     *  {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{x}\right)\right]
     * \textrm{d}\xi\textrm{d}\eta
     * \f]
     *
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

    OpSpringALEMaterialLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_spring_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name)
        : OpSpringALEMaterialLhs(field_name_1, field_name_2, data_at_spring_pts,
                                       side_fe, side_fe_name) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  /**
   * @brief LHS-operator (material configuration) on the side volume
   *
   * Computes the linearisation of the material component
   * with respect to a variation of material coordinates on the side volume.
   *
   */
  struct SpringALEMaterialVolOnSideLhs_dX_dX
      : public SpringALEMaterialVolOnSideLhs {

    /**
     * @brief Integrates over a face contribution from a side volume
     *
     * Computes linearisation of the material component
     * with respect to a variation of material coordinates:
     *
     * \f[
     * \textrm{D} \delta W^\text{(side volume)}_p(\mathbf{x}, \mathbf{X},
     * \delta\mathbf{x})
     * [\Delta\mathbf{X}] = \int\limits_{\mathcal{T}_{\xi}} p
     * \left\{\left[
     * \mathbf{h}\,\mathbf{H}^{-1}\,\frac{\partial\Delta\mathbf{X}}
     * {\partial\boldsymbol{\chi}}\,\mathbf{H}^{-1}
     * \right]^{\intercal}\cdot\left(\frac{\partial\mathbf{X}}{\partial\xi}
     * \times\frac{\partial\mathbf{X}}{\partial\eta}\right)\right\}
     * \cdot \delta\mathbf{X}\, \textrm{d}\xi\textrm{d}\eta
     * \f]
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

    SpringALEMaterialVolOnSideLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_spring_pts)
        : SpringALEMaterialVolOnSideLhs(
              field_name_1, field_name_2, data_at_spring_pts) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };
  /**
   * \brief Declare spring element
   *
   * Search cubit sidesets and blocksets with spring bc and declare surface
   * element

   * Blockset has to have name “SPRING_BC”. The first three attributes of the
   * blockset are spring stiffness value.

   *
   * @param  m_field               Interface insurance
   * @param  field_name            Field name (e.g. SPATIAL_POSITION)
   * @param  mesh_nodals_positions Name of field on which ho-geometry is defined
   * @return                       Error code
   */
  static MoFEMErrorCode addSpringElements(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /**
   * \brief Declare spring element
   *
   * Search cubit sidesets and blocksets with spring bc and declare surface
   * element

   * Blockset has to have name “SPRING_BC”. The first three attributes of the
   * blockset are spring stiffness value.

   *
   * @param  m_field               Interface insurance
   * @param  field_name            Field name (e.g. SPATIAL_POSITION)
   * @param  mesh_nodals_positions Name of field on which ho-geometry is defined
   * @return                       Error code
   */
  static MoFEMErrorCode addSpringElementsALE(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions, Range &spring_triangles);

  /**
   * \brief Implementation of spring element. Set operators to calculate LHS and
   * RHS
   *
   * @param m_field               Interface insurance
   * @param fe_spring_lhs_ptr     Pointer to the FE instance for LHS
   * @param fe_spring_rhs_ptr     Pointer to the FE instance for RHS
   * @param field_name            Field name (e.g. SPATIAL_POSITION)
   * @param mesh_nodals_positions Name of field on which ho-geometry is defined
   * @return                      Error code
   */
  static MoFEMErrorCode setSpringOperators(
      MoFEM::Interface &m_field,
      boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr,
      boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr,
      const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");

  /**
   * \brief Implementation of spring element. Set operators to calculate LHS and
   * RHS
   *
   * @param m_field               Interface insurance
   * @param fe_spring_lhs_ptr     Pointer to the FE instance for LHS
   * @param fe_spring_rhs_ptr     Pointer to the FE instance for RHS
   * @param field_name            Field name (e.g. SPATIAL_POSITION)
   * @param mesh_nodals_positions Name of field on which ho-geometry is defined
   * @return                      Error code
   */
  static MoFEMErrorCode setSpringOperatorsMaterial(
      MoFEM::Interface &m_field,
      boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr_dx,
      boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr_dX,
      boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr,
      boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_integration_pts,
      const std::string field_name,
      const std::string mesh_nodals_positions, std::string side_fe_name);

  /**
   * \brief Implementation of spring element. Set operators to calculate LHS and
   * RHS
   *
   * @param t_tangent1      First local tangent vector
   * @param t_tangent2      Second local tangent vector
   * @param t_normal        Local normal vector
   * @param t_spring_local    Spring stiffness in local coords
   * @return t_spring_global  Spring stiffness in global coords
  //  */
  static FTensor::Tensor2<double, 3, 3>
  transformLocalToGlobal(FTensor::Tensor1<double, 3> t_normal,
                         FTensor::Tensor1<double, 3> t_tangent1,
                         FTensor::Tensor1<double, 3> t_tangent2,
                         FTensor::Tensor2<double, 3, 3> t_spring_local);
};

#endif //__SPRINGELEMENT_HPP__