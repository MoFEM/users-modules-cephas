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
        boost::make_shared<MatrixDouble>();
    boost::shared_ptr<MatrixDouble> xAtPts = boost::make_shared<MatrixDouble>();
    boost::shared_ptr<MatrixDouble> xInitAtPts =
        boost::make_shared<MatrixDouble>();

    boost::shared_ptr<MatrixDouble> hMat = boost::make_shared<MatrixDouble>();
    boost::shared_ptr<MatrixDouble> FMat = boost::make_shared<MatrixDouble>();
    boost::shared_ptr<MatrixDouble> HMat = boost::make_shared<MatrixDouble>();
    boost::shared_ptr<MatrixDouble> invHMat =
        boost::make_shared<MatrixDouble>();
    boost::shared_ptr<VectorDouble> detHVec =
        boost::make_shared<VectorDouble>();

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
   * @brief RHS-operator for the spring boundary condition element
   *
   * Integrates  virtual
   * work of springs on displacement or spatial positions and assembles
   * components to RHS global vector.
   *
   */
  struct OpSpringFs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    // vector used to store force vector for each degree of freedom
    VectorDouble nF;

    boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings> commonDataPtr;
    MetaSpringBC::BlockOptionDataSprings &dAta;
    bool is_spatial_position = true;

    /** * @brief Integrates and assembles to global RHS vector virtual work of
     * springs 
     * 
     * Computes virtual work of springs on spatial positions or displacements for configurational changes:
     * 
     * \f[ 
     * f_s({\mathbf x}, {\mathbf X},
     * \delta \mathbf{x}) =  \int\limits_{\partial \Omega }^{} {{\delta
     * \mathbf{x}^T} \cdot \left[ k_{\rm n} \left( {\mathbf N} \otimes  {\mathbf
     * N} \right) + k_{\rm t} \left( {\mathbf I} - {\mathbf N} \otimes  {\mathbf
     * N}  \right) \right] \left( {\mathbf x} - {\mathbf X} \right) \partial
     * \Omega } 
     * \f]
     *
     * where \f$ \delta \mathbf{x} \f$ is the vector of base functions for either displacements or spatial positions,
     * \f$ k_{\rm n} \f$ is the stiffness of the springs normal to the surface,
     * \f$ k_{\rm t} \f$ is the stiffness of the springs tangential to the surface,
     * \f$ {\mathbf N} \f$ is the normal to the surface vector based on material positions,
     * \f$ {\mathbf x} \f$ is the vector of spatial positions or displacements of the surface with springs and
     * \f$ {\mathbf X} \f$ is the vector of material positions that is zero when displacements are considered
     * 
     */

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

    /**
     *
     * @param  common_data_ptr     Pointer to the common data for
     * spring element
     * @param  data                Variable containing data for normal and
     * tangential stiffnesses of springs attached to the current element
     * @param  field_name          String of field name for
     * spatial positions or displacements for rows
     */
    OpSpringFs(boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
                   &common_data_ptr,
               MetaSpringBC::BlockOptionDataSprings &data,
               const std::string field_name)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name.c_str(), OPROW),
          commonDataPtr(common_data_ptr), dAta(data) {
      if (field_name.compare(0, 16, "SPATIAL_POSITION") != 0)
        is_spatial_position = false;
    }
  };


  /**
   * @brief LHS-operator for the springs element
   *
   * Integrates Springs virtual work on spatial positions or displacements
   * \f$ f_s \f$ derivative with respect to spatial postions or dirplacements
   * on surface with springs side and assembles components of the LHS vector.
   *
   */
  struct OpSpringKs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings> commonDataPtr;
    MetaSpringBC::BlockOptionDataSprings &dAta;

    MatrixDouble locKs;
    MatrixDouble transLocKs;

    /** * @brief Integrates and assembles to global RHS vector virtual work of
     * springs
     *
     * Computes virtual work of springs on spatial positions or displacements
     * for configurational changes:
     *
     * \f[
     * \textrm{D} f_s({\mathbf x}, {\mathbf X},
     * \delta \mathbf{x})[\Delta \mathbf x] =  \int\limits_{\partial \Omega }^{}
     * {{\delta \mathbf{x}^T} \cdot \left[ k_{\rm n} \left( {\mathbf N} \otimes
     * {\mathbf N} \right) + k_{\rm t} \left( {\mathbf I} - {\mathbf N} \otimes
     * {\mathbf N}  \right) \right] \Delta {\mathbf x} \partial \Omega } \f]
     *
     * where \f$ \delta \mathbf{x} \f$ is the vector of base functions for
     * either displacements or spatial positions, \f$ k_{\rm n} \f$ is the
     * stiffness of the springs normal to the surface, \f$ k_{\rm t} \f$ is the
     * stiffness of the springs tangential to the surface, \f$ {\mathbf N} \f$
     * is the normal to the surface vector based on material positions, \f$
     * {\mathbf x} \f$ is the vector of spatial positions or displacements of
     * the surface with springs and \f$ {\mathbf X} \f$ is the vector of
     * material positions that is zero when displacements are considered
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);
    /**
     *
     * @param  common_data_ptr     Pointer to the common data for
     * spring element
     * @param  data                Variable containing data for normal and
     * tangential stiffnesses of springs attached to the current element
     * @param  field_name          String of field name for
     * spatial positions or displacements for rows
     */
    OpSpringKs(boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
                   &common_data_ptr,
               MetaSpringBC::BlockOptionDataSprings &data,
               const std::string field_name)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name.c_str(), field_name.c_str(), OPROWCOL),
          commonDataPtr(common_data_ptr), dAta(data) {
      // sYmm = false;
    }
  };

  struct OpSpringKs_dX
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings> commonDataPtr;
    MetaSpringBC::BlockOptionDataSprings &dAta;

    MatrixDouble locKs;
    MatrixDouble transLocKs;
    /**
     * @brief Compute part of the left-hand side
     *
     * Computes the linearisation of the material component
     * with respect to a variation of material coordinates
     * \f$(\Delta\mathbf{X})\f$:
     *
     * \f[
     * \textrm{D} f^{\textrm{(face)}}_s(\mathbf{x}, \mathbf{X},
     * \delta\mathbf{x})
     * [\Delta\mathbf{X}] = -\int\limits_{\mathcal{T}_{\xi}} \,
     * 0.5 (k_{\rm n} - k_{\rm s}) \cdot  \left\{ \left[
     * \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\eta}\times\delta\mathbf{x}\right)
     * -\frac{\partial\mathbf{X}}
     * {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{x}\right)\right] \otimes \dfrac{\mathbf{N}}{\|\mathbf{N}\|} +
     * \dfrac{\mathbf{N}}{\|\mathbf{N}\|} \otimes \left[
     * \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\eta}\times\delta\mathbf{x}\right)
     * -\frac{\partial\mathbf{X}}
     * {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{x}\right)\right]
     * - \dfrac{\mathbf{N} \otimes \mathbf{N}}{{\| \mathbf{N} \|}^3} \left[
     * \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\eta}\times\delta\mathbf{x}\right)
     * -\frac{\partial\mathbf{X}}
     * {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{x}\right)\right] \mathbf{N}
     * \right \}
     * \textrm{d}\xi\textrm{d}\eta 
     * \\
     * +\int\limits_{\mathcal{T}_{\xi}}
     *  0.5 k_{\rm s} \left[
     * \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\eta}\times\delta\mathbf{x}\right)
     * -\frac{\partial\mathbf{X}}
     * {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{x}\right)\right]
     *  {\mathbf N}^{\intercal} \cdot {\mathbf I} ( {\mathbf x} - {\mathbf X} )
     * \textrm{d}\xi\textrm{d}\eta 
     * -\int\limits_{\mathcal{T}_{\xi}} 
     * {{\delta
     * \mathbf{x}^T} \cdot \left[ k_{\rm n} \left( {\mathbf N} \otimes  {\mathbf
     * N} \right) + k_{\rm t} \left( {\mathbf I} - {\mathbf N} \otimes  {\mathbf
     * N}  \right) \right] \Delta {\mathbf X}   } 
     * \textrm{d}\xi\textrm{d}\eta 
     * 
     * \f]
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);
    OpSpringKs_dX(boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
                      &common_data_ptr,
                  MetaSpringBC::BlockOptionDataSprings &data,
                  const std::string field_name_1,
                  const std::string field_name_2)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              field_name_1.c_str(), field_name_2.c_str(), OPROWCOL),
          commonDataPtr(common_data_ptr), dAta(data) {
      sYmm = false;
    }
  };

  /**
   * @brief Base class for LHS-operators (material) on side volumes
   *
   */
  struct SpringALEMaterialVolOnSideLhs : public VolOnSideUserDataOperator {

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
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);
    virtual MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }
    MoFEMErrorCode aSsemble(EntData &row_data, EntData &col_data);

    /**
     * @param  field_name_1            String of material positions field name
     * @param  field_name_2            String of either master or material
     * positions field name, depending of the implementation of the child class
     * @param  data_at_spring_pts      Pointer to the common data for
     * spring element
     *
     */
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
   * @brief LHS-operator (material configuration) on the side volume for spring element
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
     * \textrm{D} f_s(\mathbf{x}, \mathbf{X}, \delta\mathbf{X})
     * [\Delta\mathbf{x}] = -\int\limits_{\mathcal{T}_{\xi}}
     * \left\{\left[
     * \frac{\partial\Delta\mathbf{x}}{\partial\boldsymbol{\chi}}\,\mathbf{H}^{-1}
     * \right]^{\intercal}\cdot \left[ k_{\rm n} \left( {\mathbf N} \otimes  {\mathbf
     * N} \right) + k_{\rm t} \left( {\mathbf I} - {\mathbf N} \otimes  {\mathbf
     * N}  \right) \right] \left( {\mathbf x} - {\mathbf X} \right)\right\}
     * \cdot \delta\mathbf{X}\, \textrm{d}\xi\textrm{d}\eta
     * \f]
     * 
     * 
     * where \f$ \delta \mathbf{X} \f$ is the vector of base functions for material positions,
     * \f$ k_{\rm n} \f$ is the stiffness of the springs normal to the surface,
     * \f$ k_{\rm t} \f$ is the stiffness of the springs tangential to the surface,
     * \f$ {\mathbf N} \f$ is the normal to the surface vector based on material positions,
     * \f$ {\mathbf x} \f$ is the vector of spatial positions or displacements of the surface with springs,
     * \f$ {\mathbf X} \f$ is the vector of material positions,
     * \f$\boldsymbol{\chi}\f$ are reference coordinated,
     * \f$\mathbf{H}\f$ is the gradient of the material map.
     *
     * 
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

    SpringALEMaterialVolOnSideLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_spring_pts)
        : SpringALEMaterialVolOnSideLhs(field_name_1, field_name_2,
                                        data_at_spring_pts) {
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

  /**
   * @brief LHS-operator (material configuration) on for spring element on faces
   *
   * This is base struct for integrating and assembling
   * derivatives of virtual work of springs on material positions contributing
   * to configurational changes
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
                                  EntData &row_data, EntData &col_data) {
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
          dataAtSpringPts(data_at_spring_pts), sideFe(side_fe),
          sideFeName(side_fe_name) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };

  /**
   * @brief LHS-operator for the pressure element (material configuration)
   *
   * Triggers loop over operators from the side volume
   *
   */
  struct OpSpringALEMaterialLhs_dX_dx : public OpSpringALEMaterialLhs {

    /*
     * Triggers loop over operators from the side volume
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

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
  struct OpSpringALEMaterialLhs_dX_dX : public OpSpringALEMaterialLhs {

    /**
     * Integrates a contribution to the left-hand side and triggers a loop
     * over side volume operators.
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

    /**
     * @brief Compute part of the left-hand side
     *
     * Computes the linearisation of the material component
     * with respect to a variation of material coordinates
     * \f$(\Delta\mathbf{X})\f$:
     *
     * \f[
     * \textrm{D} f^{\textrm{(face)}}_s(\mathbf{x}, \mathbf{X},
     * \delta\mathbf{X})
     * [\Delta\mathbf{X}] = -\int\limits_{\mathcal{T}_{\xi}} \,
     * 0.5 (k_{\rm n} - k_{\rm s}) \mathbf{F}^{\intercal}\cdot  \left\{ \left[
     * \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\eta}\times\delta\mathbf{X}\right)
     * -\frac{\partial\mathbf{X}}
     * {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{X}\right)\right] \otimes \dfrac{\mathbf{N}}{\|\mathbf{N}\|} +
     * \dfrac{\mathbf{N}}{\|\mathbf{N}\|} \otimes \left[
     * \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\eta}\times\delta\mathbf{X}\right)
     * -\frac{\partial\mathbf{X}}
     * {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{X}\right)\right]
     * - \dfrac{\mathbf{N} \otimes \mathbf{N}}{{\| \mathbf{N} \|}^3} \left[
     * \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\eta}\times\delta\mathbf{X}\right)
     * -\frac{\partial\mathbf{X}}
     * {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{X}\right)\right] \mathbf{N}
     * \right \}
     * \textrm{d}\xi\textrm{d}\eta 
     * \\
     * +\int\limits_{\mathcal{T}_{\xi}}
     *  0.5 k_{\rm s} {\mathbf{F}^{\intercal}} \left[
     * \frac{\partial\mathbf{X}}
     * {\partial\xi} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\eta}\times\delta\mathbf{X}\right)
     * -\frac{\partial\mathbf{X}}
     * {\partial\eta} \cdot \left(\frac{\partial\Delta
     * \mathbf{X}}{\partial\xi}\times \delta\mathbf{X}\right)\right]
     *  {\mathbf N}^{\intercal} \cdot {\mathbf I} ( {\mathbf x} - {\mathbf X} )
     * \textrm{d}\xi\textrm{d}\eta 
     * -\int\limits_{\mathcal{T}_{\xi}} 
     * {{\delta
     * \mathbf{X}^T} \cdot {\mathbf{F}^{\intercal}} \left[ k_{\rm n} \left( {\mathbf N} \otimes  {\mathbf
     * N} \right) + k_{\rm t} \left( {\mathbf I} - {\mathbf N} \otimes  {\mathbf
     * N}  \right) \right] \Delta {\mathbf X}   } 
     * \textrm{d}\xi\textrm{d}\eta 
     * 
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
     * \textrm{D} f^\text{(side volume)}_s(\mathbf{x}, \mathbf{X},
     * \delta\mathbf{X})
     * [\Delta\mathbf{X}] = \int\limits_{\mathcal{T}_{\xi}}
     * \left\{\left[
     * \mathbf{h}\,\mathbf{H}^{-1}\,\frac{\partial\Delta\mathbf{X}}
     * {\partial\boldsymbol{\chi}}\,\mathbf{H}^{-1}
     * \right]^{\intercal}\cdot \left[ k_{\rm n} \left( {\mathbf N} \otimes  {\mathbf
     * N} \right) + k_{\rm t} \left( {\mathbf I} - {\mathbf N} \otimes  {\mathbf
     * N}  \right) \right] \left( {\mathbf x} - {\mathbf X} \right) \right\}
     * \cdot \delta\mathbf{X}\, \textrm{d}\xi\textrm{d}\eta
     * \f]
     * 
     * where \f$ \delta \mathbf{X} \f$ is the vector of base functions for material positions,
     * \f$ k_{\rm n} \f$ is the stiffness of the springs normal to the surface,
     * \f$ k_{\rm t} \f$ is the stiffness of the springs tangential to the surface,
     * \f$ {\mathbf N} \f$ is the normal to the surface vector based on material positions,
     * \f$ {\mathbf x} \f$ is the vector of spatial positions or displacements of the surface with springs,
     * \f$ {\mathbf X} \f$ is the vector of material positions,
     * \f$\mathbf{h}\f$ and \f$\mathbf{H}\f$ are the gradients of the
     * spatial and material maps, respectively, and \f$\boldsymbol{\chi}\f$ are
     * the reference coordinates.
     * 
     * 
     */
    MoFEMErrorCode iNtegrate(EntData &row_data, EntData &col_data);

    SpringALEMaterialVolOnSideLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_spring_pts)
        : SpringALEMaterialVolOnSideLhs(field_name_1, field_name_2,
                                        data_at_spring_pts) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  /**
   * @brief Operator for computing deformation gradients in side volumes
   *
   */
  struct OpCalculateDeformation
      : public VolumeElementForcesAndSourcesCoreOnContactPrismSide::
            UserDataOperator {

    boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings> commonDataPtr;

    bool hoGeometry;

    MoFEMErrorCode doWork(int side, EntityType type, EntData &row_data);

    OpCalculateDeformation(
        const string field_name,
        boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
            common_data_ptr,
        bool ho_geometry = false)
        : VolumeElementForcesAndSourcesCoreOnContactPrismSide::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          commonDataPtr(common_data_ptr), hoGeometry(ho_geometry) {
      doEdges = false;
      doQuads = false;
      doTris = false;
      doTets = false;
      doPrisms = false;
      sYmm = false;
    };
  };

  /**
   * @brief RHS-operator for the spring boundary condition element for ALE
   * formulation
   *
   * Integrates  virtual
   * work of springs on material positions involved in configurational changes and assembles
   * components to RHS global vector.
   *
   */
  struct OpSpringFsMaterial
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings> dataAtPts;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;
    std::string sideFeName;
    Vec F;
    MetaSpringBC::BlockOptionDataSprings &dAta;
    bool hoGeometry;

    VectorDouble nF;

    VectorInt rowIndices;

    int nbRows;           ///< number of dofs on rows
    int nbIntegrationPts; ///< number of integration points

    MoFEMErrorCode doWork(int side, EntityType type, EntData &row_data);

    /** 
     * @brief Integrates and assembles to global RHS vector virtual work of
     * springs on material positions for configurational changes for ALE formulation
     * 
     * Computes virtual work of springs on material positions for configurational changes:
     * 
     * \f[ 
     * f_s ({\mathbf x}, {\mathbf X},
     * \delta \mathbf{X}) =  \int\limits_{\partial \Omega }^{} {{\delta
     * \mathbf{X}^T} \cdot {\mathbf{F}^{\intercal}} \left[ k_{\rm n} \left( {\mathbf N} \otimes  {\mathbf
     * N} \right) + k_{\rm t} \left( {\mathbf I} - {\mathbf N} \otimes  {\mathbf
     * N}  \right) \right] \left( {\mathbf x} - {\mathbf X} \right) \partial
     * \Omega } 
     * \f]
     *
     * where \f$ \delta \mathbf{X} \f$ is the vector of base functions for material positions,
     * \f$ k_{\rm n} \f$ is the stiffness of the springs normal to the surface,
     * \f$ k_{\rm t} \f$ is the stiffness of the springs tangential to the surface,
     * \f$ {\mathbf N} \f$ is the normal to the surface vector based on material positions,
     * \f$ {\mathbf x} \f$ is the vector of spatial positions or displacements of the surface with springs,
     * \f$ {\mathbf X} \f$ is the vector of material positions and finally
     * \f$\mathbf{F}\f$ is the deformation gradient
     * 
     * 
     * \f[
     * \mathbf{F} = \mathbf{h}(\mathbf{x})\,\mathbf{H}(\mathbf{X})^{-1} =
     * \frac{\partial\mathbf{x}}{\partial\boldsymbol{\chi}}
     * \frac{\partial\boldsymbol{\chi}}{\partial\mathbf{X}}
     * \f]
     *
     * where \f$\mathbf{h}\f$ and \f$\mathbf{H}\f$ are the gradients of the
     * spatial and material maps, respectively, and \f$\boldsymbol{\chi}\f$ are
     * the reference coordinates.
     * 
     */
    MoFEMErrorCode iNtegrate(EntData &row_data);
    MoFEMErrorCode aSsemble(EntData &row_data);

    /**
     *
     * @param  material_field          String of field name for
     * material positions for rows
     * @param  data_at_pts             Pointer to the common data for
     * spring element
     * @param  side_fe                 Pointer to the volume finite elements
     * adjacent to the spring face element
     * @param  side_fe_name            String of 3D element adjacent to
     * the present springs elements
     * @param  data                    Variable containing data for normal and
     * tangential stiffnesses of springs attached to the current element
     *
     */
    OpSpringFsMaterial(
        const string material_field,
        boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
            data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name, MetaSpringBC::BlockOptionDataSprings &data)
        : UserDataOperator(material_field, UserDataOperator::OPROW),
          dataAtPts(data_at_pts), sideFe(side_fe), sideFeName(side_fe_name),
          dAta(data){};
  };

  /// \brief Computes, for material configuration, tangent vectors to face that lie
  /// on a surface with springs
  struct OpGetTangentSpEle
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
        dataAtIntegrationPts;
    int ngp;
    /**
     *
     * @param  field_name                 String of field name for
     * material positions for rows
     * @param  data_at_integration_pts    Pointer to the common data for
     * spring element
     *
     */
    OpGetTangentSpEle(
        const string field_name,
        boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
            dataAtIntegrationPts)
        : UserDataOperator(field_name, UserDataOperator::OPCOL),
          dataAtIntegrationPts(dataAtIntegrationPts) {}

    /**
     * @brief Evaluates the two tangent vector to the triangle on surface with
     * springs based on material base coordinates
     *
     * Computes the two tangent vectors,\f$ {\mathbf T}^{(1)} \f$ and \f$ {\mathbf
     * T}^{(2)}\f$, based on material base coordinates based on mesh (moab
     * vertices) coordinates:
     *
     * \f[
     * {\mathbf T}^{(1)}({\mathbf X}(\xi, \eta)) =
     * \frac{\partial\mathbf{X}(\xi,
     * \eta)}{\partial\xi} \;\;\;\; {\mathbf T}^{(2)}({\mathbf X}(\xi, \eta)) =
     * \frac{\partial \mathbf{X}(\xi, \eta)}
     * {\partial\eta}
     * \f]
     *
     * where \f${\mathbf X}(\xi, \eta)\f$ is the vector of material
     * coordinates at the gauss point of surface with springs having parent
     * coordinates \f$\xi\f$ and \f$\eta\f$ evaluated according to
     *
     * \f[
     * {\mathbf X}(\xi, \eta) =
     * \sum\limits^{3}_{i = 1}
     * N_i(\xi, \eta){\overline{\mathbf X}}_i
     * \f]
     *
     * where \f$ N_i \f$ is the shape function corresponding to the \f$
     * i-{\rm{th}}\f$ degree of freedom in the material configuration
     * \f${\overline{\mathbf X}}_i\f$ corresponding to the 3 nodes of the
     * triangular face.
     *
     */

    MoFEMErrorCode doWork(int side, EntityType type, EntData &row_data);
  };

  /// \brief Computes, for material configuration, normal to face that lies
  /// on a surface with springs
  struct OpGetNormalSpEle
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
        dataAtIntegrationPts;
    int ngp;

    /**
     *
     * @param  field_name                 String of field name for
     * material positions for rows
     * @param  data_at_integration_pts    Pointer to the common data for
     * spring element
     *
     */
    OpGetNormalSpEle(
        const string field_name,
        boost::shared_ptr<MetaSpringBC::DataAtIntegrationPtsSprings>
            data_at_integration_pts)
        : UserDataOperator(field_name, UserDataOperator::OPCOL),
          dataAtIntegrationPts(data_at_integration_pts) {}

    /**
     * @brief Evaluates normal vector to the triangle on surface with springs based on
     * material base coordinates
     *
     * Computes normal vector based on material base coordinates based on mesh
     * (moab vertices) coordinates:
     *
     * \f[
     * {\mathbf N}({\mathbf X}(\xi, \eta)) =
     * \frac{\partial\mathbf{X}(\xi,
     * \eta)}{\partial\xi}\times\frac{\partial \mathbf{X}(\xi, \eta)}
     * {\partial\eta}
     * \f]
     *
     * where \f${\mathbf X}(\xi, \eta)\f$ is the vector of material
     * coordinates at the gauss point of surface with springs having parent coordinates
     * \f$\xi\f$ and \f$\eta\f$ evaluated according to
     *
     * \f[
     * {\mathbf X}(\xi, \eta) =
     * \sum\limits^{3}_{i = 1}
     * N_i(\xi, \eta){\overline{\mathbf X}}_i
     * \f]
     *
     * where \f$ N_i \f$ is the shape function corresponding to the \f$
     * i-{\rm{th}}\f$ degree of freedom in the material configuration
     * \f${\overline{\mathbf X}}_i\f$ corresponding to the 3 nodes of the
     * triangular face.
     *
     */

    MoFEMErrorCode doWork(int side, EntityType type, EntData &row_data);
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
  static MoFEMErrorCode
  addSpringElementsALE(MoFEM::Interface &m_field, const std::string field_name,
                       const std::string mesh_nodals_positions,
                       Range &spring_triangles);

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
      const std::string field_name, const std::string mesh_nodals_positions,
      std::string side_fe_name);
};

#endif //__SPRINGELEMENT_HPP__