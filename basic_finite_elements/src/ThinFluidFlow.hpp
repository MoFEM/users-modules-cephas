/** \file ThinFluidFlow.hpp
  \brief Header file for thin fluid flow element implementation
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

#ifndef __THIN_FLUID_FLOW__
#define __THIN_FLUID_FLOW__

/** \brief Set of functions declaring elements and setting operators
 * to apply contact conditions between surfaces with matching
 * meshes \ingroup simple_contact_problem
 */

struct ThinFluidFlowProblem {

  using ContactElement = ContactPrismElementForcesAndSourcesCore;
  using ContactOp =
      ContactPrismElementForcesAndSourcesCore::UserDataOperator;
  using EntData = DataForcesAndSourcesCore::EntData;
  using FaceUserDataOperator =
      FaceElementForcesAndSourcesCore::UserDataOperator;

  struct ThinFluidFlowPrismsData {
    Range pRisms;
  };

  map<int, ThinFluidFlowPrismsData>
      setOfThinFluidFlowPrisms; ///< maps side set id with appropriate data

  struct ThinFluidFlowElement : public ContactElement {

    MoFEM::Interface &mField;

    ThinFluidFlowElement(MoFEM::Interface &m_field)
        : ContactElement(m_field), mField(m_field) {}

    int getRule(int order) { return 2 * (order - 1) + 3; }
  };

  /**
   * @brief Function that adds field data for spatial positions and Lagrange
   * multipliers to rows and columns, provides access to field data and adds
   * prism entities to element.
   *
   * @param  element_name               String for the element name
   * @param  field_name                 String of field name for spatial
   * position
   * @param  lagrang_field_name         String of field name for Lagrange
   * multipliers
   * @param  range_slave_master_prisms  Range for prism entities used to create
   * contact elements
   * @param  lagrange_field             Boolean used to determine existence of
   * Lagrange multipliers field (default is true)
   * @return                            Error code
   *
   */
  MoFEMErrorCode addThinFluidFlowElement(const string element_name,
                                         const string spatial_field_name,
                                         const string pressure_field_name,
                                         Range &range_slave_master_prisms) {
    MoFEMFunctionBegin;

    CHKERR mField.add_finite_element(element_name, MF_ZERO);

    if (range_slave_master_prisms.size() > 0) {

      CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                        pressure_field_name);
      CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                        pressure_field_name);

      CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                        spatial_field_name);
      CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                        spatial_field_name);

      CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                         pressure_field_name);
      CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                         spatial_field_name);

      setOfThinFluidFlowPrisms[1].pRisms = range_slave_master_prisms;

      // Adding range_slave_master_prisms to Element element_name
      CHKERR mField.add_ents_to_finite_element_by_type(
          range_slave_master_prisms, MBPRISM, element_name);
    }

    MoFEMFunctionReturn(0);
  }

  struct CommonData
      : public boost::enable_shared_from_this<CommonData> {

    boost::shared_ptr<MatrixDouble> positionAtGaussPtsMasterPtr;
    boost::shared_ptr<MatrixDouble> positionAtGaussPtsSlavePtr;

    boost::shared_ptr<VectorDouble> pressureAtGaussPtsPtr;
    boost::shared_ptr<VectorDouble> gapPtr;

    boost::shared_ptr<VectorDouble> normalVectorSlavePtr;

    double areaSlave;
    double areaMaster;

    CommonData(MoFEM::Interface &m_field) : mField(m_field) {
      positionAtGaussPtsMasterPtr = boost::make_shared<MatrixDouble>();
      positionAtGaussPtsSlavePtr = boost::make_shared<MatrixDouble>();
      pressureAtGaussPtsPtr = boost::make_shared<VectorDouble>();
      gapPtr = boost::make_shared<VectorDouble>();
      normalVectorSlavePtr = boost::make_shared<VectorDouble>();
    }

  private:
    MoFEM::Interface &mField;
  };

  MoFEM::Interface &mField;

  ThinFluidFlowProblem(MoFEM::Interface &m_field) : mField(m_field) {}

  /**
   * @brief Function for the simple contact element that sets the user data
   * RHS-operators
   *
   * @param  fe_rhs_simple_contact      Pointer to the FE instance for RHS
   * @param  common_data_simple_contact Pointer to the common data for simple
   * contact element
   * @param  field_name                 String of field name for spatial
   * positions
   * @param  lagrang_field_name         String of field name for Lagrange
   * multipliers
   * @param  f_                         Right hand side vector
   * @return                            Error code
   *
   */
  MoFEMErrorCode setThinFluidFlowOperatorsRhs(
      boost::shared_ptr<ThinFluidFlowElement> fe_rhs,
      boost::shared_ptr<CommonData> common_data,
      string spatial_field_name, string pressure_field_name);

  struct OpGetNormalSlave : public ContactOp {

    boost::shared_ptr<CommonData> commonData;
    OpGetNormalSlave(
        const string field_name,
        boost::shared_ptr<CommonData> &common_data)
        : ContactOp(field_name, UserDataOperator::OPCOL, ContactOp::FACESLAVE),
          commonData(common_data) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };    

  struct OpGetPositionAtGaussPtsSlave : public ContactOp {

    boost::shared_ptr<CommonData> commonData;
    OpGetPositionAtGaussPtsSlave(
        const string field_name,
        boost::shared_ptr<CommonData> &common_data)
        : ContactOp(field_name, UserDataOperator::OPCOL, ContactOp::FACESLAVE),
          commonData(common_data) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpGetPositionAtGaussPtsMaster : public ContactOp {

    boost::shared_ptr<CommonData> commonData;
    OpGetPositionAtGaussPtsMaster(
        const string field_name,
        boost::shared_ptr<CommonData> &common_data)
        : ContactOp(field_name, UserDataOperator::OPCOL, ContactOp::FACEMASTER),
          commonData(common_data) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpGetGapSlave : public ContactOp {

    boost::shared_ptr<CommonData> commonData;

    OpGetGapSlave(
        const string field_name, // ign: does it matter??
        boost::shared_ptr<CommonData> &common_data)
        : ContactOp(field_name, UserDataOperator::OPROW, ContactOp::FACESLAVE),
          commonData(common_data) {}

    /**
     * @brief Evaluates gap function at slave face gauss points
     *
     * Computes gap function at slave face gauss points:
     *
     * \f[
     * g_{\textrm{n}} = - \mathbf{n}(\mathbf{x}^{(1)}) \cdot \left(
     * \mathbf{x}^{(1)} - \mathbf{x}^{(2)}  \right)
     * \f]
     * where \f$\mathbf{n}(\mathbf{x}^{(1)})\f$ is the outward normal vector at
     * the slave triangle gauss points, \f$\mathbf{x}^{(1)}\f$ and
     * \f$\mathbf{x}^{(2)}\f$ are the spatial coordinates of the overlapping
     * gauss points located at the slave and master triangles, respectively.
     *
     *
     */
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpGetPressureAtGaussPtsSlave : public ContactOp {

    boost::shared_ptr<CommonData> commonData;
    OpGetPressureAtGaussPtsSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonData> &common_data)
        : ContactOp(lagrang_field_name, UserDataOperator::OPROW,
                    ContactOp::FACESLAVE),
          commonData(common_data) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

};

#endif //__THIN_FLUID_FLOW__