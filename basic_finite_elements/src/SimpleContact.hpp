/** \file SimpleContact.hpp
  \brief Header file for simple contact element implementation
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

#ifndef __SIMPLE_CONTACT__
#define __SIMPLE_CONTACT__

/** \brief Set of functions declaring elements and setting operators
 * to apply contact conditions between surfaces with matching
 * meshes \ingroup simple_contact_problem
 */

struct SimpleContactProblem {

  using ContactEle = ContactPrismElementForcesAndSourcesCore;
  using ContactOp = ContactPrismElementForcesAndSourcesCore::UserDataOperator;
  using EntData = DataForcesAndSourcesCore::EntData;
  using FaceUserDataOperator =
      FaceElementForcesAndSourcesCore::UserDataOperator;

  static inline double Sign(double x);

  static inline double ConstrainFunction(const double cn, const double g,
                                         const double l);

  static inline double ConstrainFunction_dg(const double cn, const double g,
                                            const double l);

  static inline double ConstrainFunction_dl(const double cn, const double g,
                                            const double l);

  static constexpr double TOL = 1e-8;
  struct SimpleContactPrismsData {
    Range pRisms; // All boundary surfaces
  };

  map<int, SimpleContactPrismsData>
      setOfSimpleContactPrism; ///< maps side set id with appropriate FluxData

  struct ConvectSlaveIntegrationPts;

  struct SimpleContactElement : public ContactEle {

    MoFEM::Interface &mField;
    bool newtonCotes;
    SimpleContactElement(MoFEM::Interface &m_field, bool newton_cotes = false)
        : ContactEle(m_field), mField(m_field), newtonCotes(newton_cotes) {}

    int getRule(int order) {
      if (newtonCotes)
        return -1;
      else
        return 2 * order;
    }

    virtual MoFEMErrorCode setGaussPts(int order);

    friend ConvectSlaveIntegrationPts;
  };

  /**
   * @brief Class used to convect integration points on slave and master, and to
   * calculate  directional direvatives of change integration position point as
   * variation os spatial positions on contact surfaces.
   *
   */
  struct ConvectSlaveIntegrationPts
      : public boost::enable_shared_from_this<ConvectSlaveIntegrationPts> {

    ConvectSlaveIntegrationPts(SimpleContactElement *const fe_ptr,
                               std::string spat_pos, std::string mat_pos)
        : fePtr(fe_ptr), sparialPositionsField(spat_pos),
          materialPositionsField(mat_pos) {}

    template <bool CONVECT_MASTER> MoFEMErrorCode convectSlaveIntegrationPts();

    inline boost::shared_ptr<MatrixDouble> getDiffKsiSpatialMaster() {
      return boost::shared_ptr<MatrixDouble>(shared_from_this(),
                                             &diffKsiMaster);
    }

    inline boost::shared_ptr<MatrixDouble> getDiffKsiSpatialSlave() {
      return boost::shared_ptr<MatrixDouble>(shared_from_this(), &diffKsiSlave);
    }

  private:
    SimpleContactElement *const fePtr;

    const std::string sparialPositionsField;
    const std::string materialPositionsField;

    VectorDouble spatialCoords;
    VectorDouble materialCoords;
    MatrixDouble slaveSpatialCoords;
    MatrixDouble slaveMaterialCoords;
    MatrixDouble masterSpatialCoords;
    MatrixDouble masterMaterialCoords;
    MatrixDouble slaveN;
    MatrixDouble masterN;

    MatrixDouble diffKsiMaster;
    MatrixDouble diffKsiSlave;
  };

  /**
   * @brief Element used to integrate on slave surfaces. It convects integration
   * points on slaves, so that quantities like gap from master are evaluated at
   * correct points.
   *
   */
  struct ConvectMasterContactElement : public SimpleContactElement {

    ConvectMasterContactElement(MoFEM::Interface &m_field, std::string spat_pos,
                                std::string mat_pos, bool newton_cotes = false)
        : SimpleContactElement(m_field, newton_cotes),
          convectPtr(new ConvectSlaveIntegrationPts(this, spat_pos, mat_pos)) {}

    inline boost::shared_ptr<ConvectSlaveIntegrationPts> getConvectPtr() {
      return convectPtr;
    }

    int getRule(int order) { return -1; }

    MoFEMErrorCode setGaussPts(int order);

  protected:
    boost::shared_ptr<ConvectSlaveIntegrationPts> convectPtr;
  };

  /**
   * @brief Element used to integrate on master surfaces. It convects
   * integration points on slaves, so that quantities like Lagrange multiplier
   * from master are evaluated at correct points.
   *
   */
  struct ConvectSlaveContactElement : public ConvectMasterContactElement {
    using ConvectMasterContactElement::ConvectMasterContactElement;

    int getRule(int order) { return -1; }

    MoFEMErrorCode setGaussPts(int order);
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
   * @param  mesh_node_field_name       String of field name for material positions
   * @param  range_slave_master_prisms  Range for prism entities used to create
   * contact elements
   * @param  lagrange_field             Boolean used to determine existence of
   * Lagrange multipliers field (default is true)
   * @return                            Error code
   *
   */
  MoFEMErrorCode addContactElement(const string element_name,
                                   const string field_name,
                                   const string lagrang_field_name,
                                   const string mesh_node_field_name,
                                   Range &range_slave_master_prisms,
                                   bool lagrange_field = true) {
    MoFEMFunctionBegin;

    CHKERR mField.add_finite_element(element_name, MF_ZERO);

    if (range_slave_master_prisms.size() > 0) {

      // C row as Lagrange_mul and col as SPATIAL_POSITION
      if (lagrange_field)
        CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                          lagrang_field_name);

      CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                        field_name);

      // CT col as Lagrange_mul and row as SPATIAL_POSITION
      if (lagrange_field)
        CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                          lagrang_field_name);

      CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                        field_name);

      // data
      if (lagrange_field)
        CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                           lagrang_field_name);

      CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                         field_name);

      CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                         mesh_node_field_name);

      setOfSimpleContactPrism[1].pRisms = range_slave_master_prisms;

      // Adding range_slave_master_prisms to Element element_name
      CHKERR mField.add_ents_to_finite_element_by_type(
          range_slave_master_prisms, MBPRISM, element_name);
    }

    MoFEMFunctionReturn(0);
  }

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
   * @param  mesh_node_field_name       String of field name for material
   * positions
   * @param  range_slave_master_prisms  Range for prism entities used to create
   * contact elements
   * @param  lagrange_field             Boolean used to determine existence of
   * Lagrange multipliers field (default is true)
   * @return                            Error code
   *
   */
  MoFEMErrorCode addContactElementALE(const string element_name,
                                      const string field_name,
                                      const string mesh_node_field_name,
                                      const string lagrang_field_name,
                                      Range &range_slave_master_prisms,
                                      bool lagrange_field = true) {
    MoFEMFunctionBegin;

    CHKERR mField.add_finite_element(element_name, MF_ZERO);

    if (range_slave_master_prisms.size() > 0) {

      if (lagrange_field)
        CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                          lagrang_field_name);

      CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                        field_name);
      CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                        mesh_node_field_name);

      if (lagrange_field)
        CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                          lagrang_field_name);
      CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                        field_name);
      CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                        mesh_node_field_name);

      if (lagrange_field)
        CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                           lagrang_field_name);

      CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                         field_name);

      CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                         mesh_node_field_name);

      setOfSimpleContactPrism[1].pRisms = range_slave_master_prisms;

      // Adding range_slave_master_prisms to Element element_name
      CHKERR mField.add_ents_to_finite_element_by_type(
          range_slave_master_prisms, MBPRISM, element_name);
    }

    MoFEMFunctionReturn(0);
  }

  struct CommonDataSimpleContact
      : public boost::enable_shared_from_this<CommonDataSimpleContact> {

    boost::shared_ptr<MatrixDouble> positionAtGaussPtsMasterPtr;
    boost::shared_ptr<MatrixDouble> positionAtGaussPtsSlavePtr;
    boost::shared_ptr<MatrixDouble> gradKsiPositionAtGaussPtsPtr;
    boost::shared_ptr<MatrixDouble> gradKsiLambdaAtGaussPtsPtr;

    boost::shared_ptr<VectorDouble> lagMultAtGaussPtsPtr;
    boost::shared_ptr<VectorDouble> gapPtr;
    boost::shared_ptr<VectorDouble> lagGapProdPtr;

    boost::shared_ptr<VectorDouble> tangentOneVectorSlavePtr;
    boost::shared_ptr<VectorDouble> tangentTwoVectorSlavePtr;
    boost::shared_ptr<VectorDouble> tangentOneVectorMasterPtr;
    boost::shared_ptr<VectorDouble> tangentTwoVectorMasterPtr;

    boost::shared_ptr<VectorDouble> normalVectorSlavePtr;
    boost::shared_ptr<VectorDouble> normalVectorMasterPtr;

    boost::shared_ptr<MatrixDouble> hMat;
    boost::shared_ptr<MatrixDouble> FMat;
    boost::shared_ptr<MatrixDouble> HMat;
    boost::shared_ptr<MatrixDouble> invHMat;
    boost::shared_ptr<VectorDouble> detHVec;

    double areaSlave;
    double areaMaster;

    Range forcesOnlyOnEntitiesRow;
    Range forcesOnlyOnEntitiesCol;

    CommonDataSimpleContact(MoFEM::Interface &m_field) : mField(m_field) {
      positionAtGaussPtsMasterPtr = boost::make_shared<MatrixDouble>();
      positionAtGaussPtsSlavePtr = boost::make_shared<MatrixDouble>();
      gradKsiPositionAtGaussPtsPtr = boost::make_shared<MatrixDouble>();
      gradKsiLambdaAtGaussPtsPtr = boost::make_shared<MatrixDouble>();
      lagMultAtGaussPtsPtr = boost::make_shared<VectorDouble>();

      gapPtr = boost::make_shared<VectorDouble>();
      lagGapProdPtr = boost::make_shared<VectorDouble>();
      normalVectorSlavePtr = boost::make_shared<VectorDouble>();
      normalVectorMasterPtr = boost::make_shared<VectorDouble>();

      tangentOneVectorSlavePtr= boost::make_shared<VectorDouble>();
      tangentTwoVectorSlavePtr= boost::make_shared<VectorDouble>();
      tangentOneVectorMasterPtr= boost::make_shared<VectorDouble>();
      tangentTwoVectorMasterPtr= boost::make_shared<VectorDouble>();

      hMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      FMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      HMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      invHMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      detHVec = boost::shared_ptr<VectorDouble>(new VectorDouble());
    }

  private:
    MoFEM::Interface &mField;
  };

  double cnValue;
  bool newtonCotes;
  MoFEM::Interface &mField;

  SimpleContactProblem(MoFEM::Interface &m_field, double &cn_value,
                       bool newton_cotes = false)
      : mField(m_field), cnValue(cn_value), newtonCotes(newton_cotes) {}

  /// \brief Computes normal to slave face that is common to all gauss points
  struct OpGetNormalSlave : public ContactOp {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetNormalSlave(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, UserDataOperator::OPCOL, ContactOp::FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpGetNormalMaster : public ContactOp {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetNormalMaster(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, UserDataOperator::OPCOL, ContactOp::FACEMASTER),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Calculates the spacial coordinates of the gauss points of master triangle.
   *
   */
  struct OpGetPositionAtGaussPtsMaster : public ContactOp {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetPositionAtGaussPtsMaster(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, UserDataOperator::OPCOL, ContactOp::FACEMASTER),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Calculates the spacial coordinates of the gauss points of slave triangle.
   *
   */
  struct OpGetPositionAtGaussPtsSlave : public ContactOp {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetPositionAtGaussPtsSlave(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, UserDataOperator::OPCOL, ContactOp::FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Calculates gap function at the gauss points on the slave triangle.
   *
   */
  struct OpGetGapSlave : public ContactOp {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    OpGetGapSlave(
        const string field_name, // ign: does it matter??
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, UserDataOperator::OPROW, ContactOp::FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

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

  /**
   * @brief Operator for the simple contact element
   *
   * Calculates Lagrange multipliers at the gauss points on the slave triangle.
   *
   */
  struct OpGetLagMulAtGaussPtsSlave : public ContactOp {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetLagMulAtGaussPtsSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(lagrang_field_name, UserDataOperator::OPROW,
                    ContactOp::FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Prints Lagrange multipliers and gaps evaluated at the gauss points on the
   * slave triangle.
   *
   */
  struct OpPrintLagMulAtGaussPtsSlave : public ContactOp {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpPrintLagMulAtGaussPtsSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(lagrang_field_name, UserDataOperator::OPROW,
                    ContactOp::FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Calculates the product of Lagrange multipliers with gaps evaluated at the
   * gauss points on the slave triangle.
   *
   */
  struct OpLagGapProdGaussPtsSlave : public ContactOp {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpLagGapProdGaussPtsSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(lagrang_field_name, UserDataOperator::OPROW,
                    ContactOp::FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  /**
   * @brief RHS-operator for the simple contact element
   *
   * Integrates Lagrange multipliers virtual work on
   * master surfaceand assemble components to RHS global vector.
   *
   */
  struct OpCalContactTractionOnMaster : public ContactOp {

    OpCalContactTractionOnMaster(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, UserDataOperator::OPROW, ContactOp::FACEMASTER),
          commonDataSimpleContact(common_data_contact) {}

    /**
     * @brief Integrates Lagrange multipliers virtual work on
     * master surface and assembles to global RHS vector
     *
     * Integrates Lagrange multipliers virtual work \f$ \delta
     * W_{\text c}\f$ on master surface and assembles components to global RHS
     * vector
     *
     * \f[
     * {\delta
     * W^{(2)}_{\text c}(\lambda,
     * \delta \mathbf{x}^{(2)}}) \,\,  =
     * \int_{{\gamma}^{(2)}_{\text c}} \lambda
     * \delta{\mathbf{x}^{(2)}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where \f${\gamma}^{(2)}_{\text c}\f$ is the surface integration domain
     * of the master surface, \f$ \lambda\f$ is the Lagrange multiplier,
     * \f$\mathbf{x}^{(2)}\f$ are the coordinates of the overlapping gauss
     * points at master triangles.
     */
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    VectorDouble vecF;
  };

  /**
   * @brief RHS-operator for the simple contact element
   *
   * Integrates Lagrange multipliers virtual work on
   * slave surface and assembles components to the RHS global vector.
   *
   */
  struct OpCalContactTractionOnSlave : public ContactOp {

    OpCalContactTractionOnSlave(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, UserDataOperator::OPROW, ContactOp::FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    /**
     * @brief Integrates Lagrange multipliers virtual work on
     * slave surface and assembles components to the RHS global vector.
     *
     * Integrates Lagrange multipliers virtual work \f$ \delta
     * W_{\text c}\f$ on slave surface and assembles components to the RHS
     * global vector
     *
     * \f[
     * {\delta
     * W^{(1)}_{\text c}(\lambda,
     * \delta \mathbf{x}^{(1)}}) \,\,  = -
     * \int_{{\gamma}^{(1)}_{\text c}} \lambda
     * \delta{\mathbf{x}^{(1)}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where \f${\gamma}^{(1)}_{\text c}\f$ is the surface integration domain
     * of the slave surface, \f$ \lambda\f$ is the Lagrange multiplier,
     * \f$\mathbf{x}^{(1)}\f$ are the coordinates of the overlapping gauss
     * points at slave triangles.
     */
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    VectorDouble vecF;
  };

  /**
   * @brief RHS-operator for the simple contact element
   *
   * Integrates complementarity function that fulfills KKT
   * conditions over slave contact area and assembles components of the RHS
   * vector.
   *
   */
  struct OpCalIntCompFunSlave : public ContactOp {

    OpCalIntCompFunSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        const double cn)
        : ContactOp(lagrang_field_name, UserDataOperator::OPCOL,
                    ContactOp::FACESLAVE),
          commonDataSimpleContact(common_data_contact), cN(cn) {}

    /**
     * @brief Integrates the complementarity function at slave
     * face gauss points and assembles components to the RHS global vector.
     *
     * Integrates the complementarity function to fulfil KKT
     * conditions in the integral sense and assembles components to the RHS
     * global vector
     *
     * \f[
     * {\overline C(\lambda, \mathbf{x}^{(i)},
     * \delta \lambda)} = \int_{{\gamma}^{(1)}_{\text
     * c}} \left( \lambda + c_{\text n} g_{\textrm{n}}  - \dfrac{1}{r}{\left|
     * \lambda - c_{\text n} g_{\textrm{n}}\right|}^{r}\right) \delta{{\lambda}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where \f${\gamma}^{(1)}_{\text c}\f$ is the surface integration domain
     * of the slave surface, \f$ \lambda\f$ is the Lagrange multiplier,
     * \f$\mathbf{x}^{(i)}\f$ are the coordinates of the overlapping gauss
     * points at slave and master triangles for  \f$i = 1\f$ and \f$i = 2\f$,
     * respectively. Furthermore, \f$ c_{\text n}\f$ works as an augmentation
     * parameter and affects convergence, \f$r\f$ is regularisation parameter
     * that can be chosen in \f$[1, 1.1]\f$ (\f$r = 1\f$) is the default
     * value) and \f$ g_{\textrm{n}}\f$ is the gap function evaluated at the
     * slave triangle gauss points as: \f[ g_{\textrm{n}} = -
     * \mathbf{n}(\mathbf{x}^{(1)}) \cdot \left( \mathbf{x}^{(1)} -
     * \mathbf{x}^{(2)}  \right) \f]
     */
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    const double cN;
    VectorDouble vecR;
  };

  /**
   * @brief LHS-operator for the simple contact element
   *
   * Integrates Lagrange multipliers virtual
   * work, \f$ \delta W_{\text c}\f$ derivative with respect to Lagrange
   * multipliers on master side and assembles components of the RHS vector.
   *
   */
  struct OpCalContactTractionOverLambdaMasterSlave : public ContactOp {

    OpCalContactTractionOverLambdaMasterSlave(
        const string field_name, const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, lagrang_field_name, UserDataOperator::OPROWCOL,
                    ContactOp::FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }

    /**
     * @brief  Integrates Lagrange multipliers virtual
     * work, \f$ \delta W_{\text c}\f$ derivative with respect to Lagrange
     * multipliers with respect to Lagrange multipliers on master side and
     * assembles components to LHS global matrix.
     *
     * Computes linearisation of integrated on slave side complementarity
     * function and assembles Lagrange multipliers virtual work \f$ \delta
     * W_{\text c}\f$ with respect to Lagrange multipliers side and assembles
     * components to LHS global matrix
     *
     * \f[
     * {\text D} {\delta
     * W^{(2)}_{\text c}(\lambda,
     * \delta \mathbf{x}^{(2)}})[\Delta \lambda]
     *  \,\,  =
     * \int_{{\gamma}^{(2)}_{\text c}} \Delta \lambda
     * \delta{\mathbf{x}^{(2)}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where \f${\gamma}^{(2)}_{\text c}\f$ is the surface integration domain
     * of the master surface, \f$ \lambda\f$ is the Lagrange multiplier,
     * \f$\mathbf{x}^{(2)}\f$ are the coordinates of the overlapping gauss
     * points at master triangles.
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    MatrixDouble NN;
  };

  /**
   * @brief LHS-operator for the simple contact element
   *
   * Integrates Lagrange multipliers virtual
   * work, \f$ \delta W_{\text c}\f$ derivative with respect to Lagrange
   * multipliers with respect to Lagrange multipliers on slave side side and
   * assembles components to LHS global matrix.
   *
   */
  struct OpCalContactTractionOverLambdaSlaveSlave : public ContactOp {

    OpCalContactTractionOverLambdaSlaveSlave(
        const string field_name, const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactOp(field_name, lagrang_field_name, UserDataOperator::OPROWCOL,
                    ContactOp::FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }

    /**
     * @brief Integrates and assembles Lagrange multipliers virtual
     * work, \f$ \delta W_{\text c}\f$ derivative with respect to Lagrange
     * multipliers with respect to Lagrange multipliers on slave side and
     * assembles components to LHS global matrix.
     *
     * Computes linearisation of integrated on slave side complementarity
     * function and assembles Lagrange multipliers virtual work, \f$ \delta
     * W_{\text c}\f$ with respect to Lagrange multipliers
     *
     * \f[
     * {\text D} {\delta
     * W^{(1)}_{\text c}(\lambda,
     * \delta \mathbf{x}^{(1)}})[\Delta \lambda]
     * \,\, =
     * \int_{{\gamma}^{(1)}_{\text c}} -\Delta \lambda
     * \delta{\mathbf{x}^{(1)}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where \f${\gamma}^{(1)}_{\text c}\f$ is the surface integration domain
     * of the slave surface, \f$ \lambda\f$ is the Lagrange multiplier,
     * \f$\mathbf{x}^{(1)}\f$ are the coordinates of the overlapping gauss
     * points at slave triangles.
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    MatrixDouble NN;
  };

  /**
   * @brief LHS-operator for the simple contact element
   *
   * Integrates variation of the complementarity function
   * with respect to Lagrange multipliers to fulfils KKT conditions
   * in the integral sense on slave side and assembles
   * components to LHS global matrix.
   *
   */
  struct OpCalDerIntCompFunOverLambdaSlaveSlave : public ContactOp {

    OpCalDerIntCompFunOverLambdaSlaveSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        const double cn)
        : ContactOp(lagrang_field_name, UserDataOperator::OPROWCOL,
                    ContactOp::FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact), cN(cn) {
      sYmm = false; // This will make sure to loop over all entities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }

    /**
     * @brief Integrates the complementarity function at slave
     * face gauss points and assembles
     * components to LHS global matrix.
     *
     * Integrates variation of the complementarity function
     * with respect to Lagrange multipliers to fulfils KKT conditions
     * in the integral sense nd assembles
     * components to LHS global matrix.
     *
     * \f[
     * {\text D}{\overline C(\lambda, \mathbf{x}^{(i)},
     * \delta \lambda)}[\Delta \lambda] = \int_{{\gamma}^{(1)}_{\text
     * c}} \Delta \lambda \left( 1 - {\text {sign}}\left( \lambda - c_{\text n}
     * g_{\textrm{n}} \right) {\left| \lambda - c_{\text n}
     * g_{\textrm{n}}\right|}^{r-1}\right) \delta{{\lambda}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where \f${\gamma}^{(1)}_{\text c}\f$ is the surface integration domain
     * of the slave surface, \f$ \lambda\f$ is the Lagrange multiplier,
     * \f$\mathbf{x}^{(i)}\f$ are the coordinates of the overlapping gauss
     * points at slave and master triangles for  \f$i = 1\f$ and \f$i = 2\f$,
     * respectively. Furthermore, \f$ c_{\text n}\f$ works as an augmentation
     * parameter and affects convergence, \f$r\f$ is regularisation parameter
     * that can be chosen in \f$[1, 1.1]\f$ (\f$r = 1\f$) is the default
     * value and \f$ g_{\textrm{n}}\f$ is the gap function evaluated at the
     * slave triangle gauss points as: \f[ g_{\textrm{n}} = -
     * \mathbf{n}(\mathbf{x}^{(1)}) \cdot \left( \mathbf{x}^{(1)} -
     * \mathbf{x}^{(2)}  \right) \f]
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    const double cN;
    MatrixDouble NN;
  };

  /**
   * @brief LHS-operator for the simple contact element
   *
   * Integrates the variation with respect to master spatial
   * positions of the complementarity function to fulfill KKT conditions in
   * the integral senseand assembles
   * components to LHS global matrix.
   *
   */
  struct OpCalDerIntCompFunOverSpatPosSlaveMaster : public ContactOp {

    OpCalDerIntCompFunOverSpatPosSlaveMaster(
        const string field_name, const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        const double cn)
        : ContactOp(lagrang_field_name, field_name, UserDataOperator::OPROWCOL,
                    ContactOp::FACESLAVEMASTER),
          commonDataSimpleContact(common_data_contact), cN(cn) {
      sYmm = false; // This will make sure to loop over all entities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }

    /**
     * @brief Integrates linearisation of the complementarity function at slave
     * face gauss points and assembles
     * components to LHS global matrix.
     *
     * Integrates the variation with respect to master spatial
     * positions of the complementarity function to fulfill KKT conditions in
     * the integral sense and assembles
     * components to LHS global matrix.
     *
     * \f[
     * {\text D}{\overline C(\lambda, \mathbf{x}^{(i)},
     * \delta \lambda)}[\Delta \mathbf{x}^{(2)}] = \int_{{\gamma}^{(2)}_{\text
     * c}} \Delta \mathbf{x}^{(2)} \cdot
     * \mathbf{n}(\mathbf{x}^{(1)}) c_{\text n}  \left( 1 + {\text {sign}}\left(
     * \lambda - c_{\text n} g_{\textrm{n}} \right)
     * {\left| \lambda - c_{\text n}
     * g_{\textrm{n}}\right|}^{r-1}\right) \delta{{\lambda}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where \f${\gamma}^{(1)}_{\text c}\f$ is the surface integration domain
     * of the slave surface, \f$ \lambda\f$ is the Lagrange multiplier,
     * \f$\mathbf{x}^{(i)}\f$ are the coordinates of the overlapping gauss
     * points at slave and master triangles for  \f$i = 1\f$ and \f$i = 2\f$,
     * respectively. Furthermore, \f$ c_{\text n}\f$ works as an augmentation
     * parameter and affects convergence, \f$r\f$ is regularisation parameter
     * that can be chosen in \f$[1, 1.1]\f$ (\f$r = 1\f$ is the default
     * value) and \f$ g_{\textrm{n}}\f$ is the gap function evaluated at the
     * slave triangle gauss points as: \f[ g_{\textrm{n}} = -
     * \mathbf{n}(\mathbf{x}^{(1)}) \cdot \left( \mathbf{x}^{(1)} -
     * \mathbf{x}^{(2)}  \right) \f]
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    double cN;
    MatrixDouble NN;
  };

  /**
   * @brief LHS-operator for the simple contact element
   *
   * Integrates the variation with respect to slave spatial
   * positions of the complementarity function to fulfill KKT conditions in
   * the integral sense and assembles
   * components to LHS global matrix.
   *
   */
  struct OpCalDerIntCompFunOverSpatPosSlaveSlave : public ContactOp {

    OpCalDerIntCompFunOverSpatPosSlaveSlave(
        const string field_name, const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        const double cn)
        : ContactOp(lagrang_field_name, field_name, UserDataOperator::OPROWCOL,
                    ContactOp::FACESLAVESLAVE),
          cN(cn), commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all entities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }

    /**
     * @brief Integrates linearisation of the complementarity
     * function at slave face gauss points and assembles
     * components to LHS global matrix.
     *
     * Integrates and assembles the variation with respect to slave spatial
     * positions of the complementarity function to fulfill KKT conditions in
     * the integral sense and assembles
     * components to LHS global matrix.
     *
     * \f[
     * {\text D}{\overline C(\lambda, \mathbf{x}^{(i)},
     * \delta \lambda)}[\Delta \mathbf{x}^{(1)}] = \int_{{\gamma}^{(1)}_{\text
     * c}} -\Delta \mathbf{x}^{(1)} \cdot
     * \mathbf{n}(\mathbf{x}^{(1)}) c_{\text n}  \left( 1 + {\text
     * {sign}}\left( \lambda - c_{\text n} g_{\textrm{n}} \right)
     * {\left| \lambda - c_{\text n}
     * g_{\textrm{n}}\right|}^{r-1}\right) \delta{{\lambda}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where \f${\gamma}^{(1)}_{\text c}\f$ is the surface integration domain
     * of the slave surface, \f$ \lambda\f$ is the Lagrange multiplier,
     * \f$\mathbf{x}^{(i)}\f$ are the coordinates of the overlapping gauss
     * points at slave and master triangles for  \f$i = 1\f$ and \f$i = 2\f$,
     * respectively. Furthermore, \f$ c_{\text n}\f$ works as an augmentation
     * parameter and affects convergence, \f$r\f$ is regularisation parameter
     * that can be chosen in \f$[1, 1.1]\f$ (\f$r = 1\f$ is the default
     * value) and \f$ g_{\textrm{n}}\f$ is the gap function evaluated at the
     * slave triangle gauss points as: \f[ g_{\textrm{n}} = -
     * \mathbf{n}(\mathbf{x}^{(1)}) \cdot \left( \mathbf{x}^{(1)} -
     * \mathbf{x}^{(2)}  \right) \f]
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    const double cN;
    MatrixDouble NN;
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Prints to .vtk file pre-calculated gaps, Lagrange multipliers and their
   * product the gauss points on the slave triangle.
   *
   */
  struct OpMakeVtkSlave : public ContactOp {

    MoFEM::Interface &mField;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    moab::Interface &moabOut;
    bool lagFieldSet;

    OpMakeVtkSlave(MoFEM::Interface &m_field, string field_name,
                   boost::shared_ptr<CommonDataSimpleContact> &common_data,
                   moab::Interface &moab_out, bool lagrange_field = true)
        : ContactOp(field_name, UserDataOperator::OPROW, ContactOp::FACESLAVE),
          mField(m_field), commonDataSimpleContact(common_data),
          moabOut(moab_out), lagFieldSet(lagrange_field) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpMakeTestTextFile : public ContactOp {

    MoFEM::Interface &mField;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    bool lagFieldSet;
    std::ofstream &mySplit;
    // stream<tee_device<std::ostream, std::ofstream>> &mySplit;

    OpMakeTestTextFile(MoFEM::Interface &m_field, string field_name,
                       boost::shared_ptr<CommonDataSimpleContact> &common_data,
                       std::ofstream &_my_split, bool lagrange_field = true)
        : ContactOp(field_name, UserDataOperator::OPROW, ContactOp::FACESLAVE),
          mField(m_field), commonDataSimpleContact(common_data),
          lagFieldSet(lagrange_field), mySplit(_my_split) {
      mySplit << fixed << setprecision(8);
      mySplit << "[0] Lagrange multiplier [1] Gap" << endl;
    }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpCalLagrangeMultPostProc : public FaceUserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    OpCalLagrangeMultPostProc(
        const string lag_mult_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_simple_contact)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              lag_mult_name, UserDataOperator::OPROW),
          commonDataSimpleContact(common_data_simple_contact){};

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);
  };

  struct OpPostProcContactContinuous : public FaceUserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;
    string lagMultName;
    string fieldName;

    OpPostProcContactContinuous(
        const string lag_mult_name, const string field_name,
        moab::Interface &post_proc_mesh,
        std::vector<EntityHandle> &map_gauss_pts,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_simple_contact)
        : FaceElementForcesAndSourcesCore::UserDataOperator(
              lag_mult_name, UserDataOperator::OPROW),
          lagMultName(lag_mult_name), fieldName(field_name),
          commonDataSimpleContact(common_data_simple_contact),
          postProcMesh(post_proc_mesh), mapGaussPts(map_gauss_pts) {
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
  MoFEMErrorCode setContactOperatorsRhs(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name);

  /**
   * @brief Function for the simple contact element that sets the user data
   * LHS-operators
   *
   * @param  fe_lhs_simple_contact      Pointer to the FE instance for LHS
   * @param  common_data_simple_contact Pointer to the common data for simple
   * contact element
   * @param  field_name                 String of field name for spatial
   * positions
   * @param  lagrang_field_name         String of field name for Lagrange
   * multipliers
   * @return                            Error code
   *
   */
  MoFEMErrorCode setContactOperatorsLhs(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name);

  // add description
  MoFEMErrorCode setPostProcContactOperators(
      boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_contact_ptr,
      const std::string field_name, const std::string lagrang_field_name,
      boost::shared_ptr<CommonDataSimpleContact> common_data);

  /**
   * @copydoc SimpleContactProblem::setContactOperatorsLhs
   *
   * \note This overloaded variant add additional operators for convected
   * integration points.
   *
   */
  MoFEMErrorCode setContactOperatorsLhs(
      boost::shared_ptr<ConvectMasterContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name);

  MoFEMErrorCode setMasterForceOperatorsRhs(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name);

  MoFEMErrorCode setMasterForceOperatorsLhs(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name);

  MoFEMErrorCode setMasterForceOperatorsLhs(
      boost::shared_ptr<ConvectSlaveContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name);

  /**
   * @brief Function for the simple contact element that sets the user data
   * post processing operators
   *
   * @param  fe_post_proc_simple_contact Pointer to the FE instance for post
   * processing
   * @param  common_data_simple_contact  Pointer to the common data for simple
   * contact element
   * @param  field_name                  String of field name for spatial
   * positions
   * @param  lagrang_field_name          String of field name for Lagrange
   * multipliers
   * @param  moab_out                    MOAB interface used to output
   * values at integration points
   * @param  lagrange_field              Booleand to determine existence of
   * lagrange field
   * @return                             Error code
   *
   */
  MoFEMErrorCode setContactOperatorsForPostProc(
      boost::shared_ptr<SimpleContactElement> fe_post_proc_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      MoFEM::Interface &m_field, string field_name, string lagrang_field_name,
      moab::Interface &moab_out, bool lagrange_field = true);

  /**
   * @brief Calculate tangent operator for contact force for change of
   * integration point positions, as result of change os spatial positions.
   *
   */
  struct OpLhsConvectIntegrationPtsContactTraction : public ContactOp {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    OpLhsConvectIntegrationPtsContactTraction(
        const string row_field_name, const string col_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        ContactOp::FaceType face_type,
        boost::shared_ptr<MatrixDouble> diff_convect)
        : ContactOp(row_field_name, col_field_name, UserDataOperator::OPROWCOL,
                    face_type),
          commonDataSimpleContact(common_data_contact),
          diffConvect(diff_convect) {
      sYmm = false;
    }

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

  private:
    MatrixDouble matLhs;
    boost::shared_ptr<MatrixDouble> diffConvect;
  };

  /**
   * @brief Evaluate gradient position on reference master surface.
   *
   */
  struct OpCalculateGradPositionXi : public ContactOp {

    OpCalculateGradPositionXi(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, UserDataOperator::OPROW, ContactOp::FACEMASTER),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  };

  /**
   * @brief Evaluate gradient of Lagrange multipliers on reference slave surface
   *
   */
  struct OpCalculateGradLambdaXi : public ContactOp {

    OpCalculateGradLambdaXi(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactOp(field_name, UserDataOperator::OPROW, ContactOp::FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  };

  /**
   * @brief Tangent opeerator for contrains equation for change of spatial
   * positions on master and slave.
   *
   */
  struct OpLhsConvectIntegrationPtsConstrainMasterGap : public ContactOp {

    OpLhsConvectIntegrationPtsConstrainMasterGap(
        const string lagrange_field_name, const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        const double cn, ContactOp::FaceType face_type,
        boost::shared_ptr<MatrixDouble> diff_convect)
        : ContactOp(lagrange_field_name, field_name, UserDataOperator::OPROWCOL,
                    face_type),
          commonDataSimpleContact(common_data_contact), cN(cn),
          diffConvect(diff_convect) {
      sYmm = false;
    }

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type, EntData &row_data,
                          EntData &col_data);

  private:
    MatrixDouble matLhs;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    const double cN;
    boost::shared_ptr<MatrixDouble> diffConvect;
  };

  /**
   * @brief Trigers operators for side volume
   * element on master side
   *
   * Operator that trigers the pipeline to loop over the side volume element
   * operators that is adjacent to the master side.
   *
   */
  struct OpLoopMasterForSide
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;

    OpLoopMasterForSide(
        const string field_name,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          sideFe(side_fe), sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief Operator for computing deformation gradients in side volumes
   *
   */
  struct OpCalculateDeformation
      : public VolumeElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    bool hoGeometry;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

    OpCalculateDeformation(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        bool ho_geometry = false)
        : VolumeElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          commonDataSimpleContact(common_data_contact),
          hoGeometry(ho_geometry) {
      doEdges = false;
      doQuads = false;
      doTris = false;
      doTets = false;
      doPrisms = false;
      sYmm = false;
    };
  };

  /**
   * @brief Trigers operators for side volume
   * element on slave side
   *
   * Operator that trigers the pipeline to loop over the side volume element
   * operators that is adjacent to the slave side.
   *
   */
  struct OpLoopSlaveForSide
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;

    OpLoopSlaveForSide(
        const string field_name,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          sideFe(side_fe), sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };
  
  /**
  *@brief RHS - operator for the pressure element(material configuration) *
                    *Integrates pressure in the material configuration. **/
    struct OpCalMatForcesALEMaster :
      public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    VectorInt rowIndices;

    int nbRows;           ///< number of dofs on rows
    int nbIntegrationPts; ///< number of integration points

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    VectorDouble vecF;
   
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data);
    MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data);

    OpCalMatForcesALEMaster(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          commonDataSimpleContact(common_data_contact) {}
  };

  struct OpCalMatForcesALESlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    VectorInt rowIndices;

    int nbRows;           ///< number of dofs on rows
    int nbIntegrationPts; ///< number of integration points

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    VectorDouble vecF;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data);
    MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data);

    OpCalMatForcesALESlave(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}
  };

  struct OpContactConstraintMatrixMasterSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixMasterSlaveALE(
        const string field_name, const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, lagrang_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble matLhs;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpGetNormalSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetNormalSlaveALE(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpGetNormalMasterALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetNormalMasterALE(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          commonDataSimpleContact(common_data_contact) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpContactConstraintMatrixSlaveSlave_dX
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixSlaveSlave_dX(
        const string field_name, const string mesh_nodes_field,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, mesh_nodes_field, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble matLhs;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactConstraintMatrixSlaveSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixSlaveSlaveALE(
        const string field_name, const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, lagrang_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble matLhs;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactConstraintMatrixMasterSlave_dX
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixMasterSlave_dX(
        const string field_name, const string mesh_nodes_field,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, mesh_nodes_field, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble matLhs;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE_dX
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE_dX(
        const string field_name, const string lagrang_field_name,
        double &cn_value,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          cN(cn_value), commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all entities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble matLhs;
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  /**
   * @brief Function for the simple contact element that sets the user data
   * post processing operators
   *
   * @param  fe_post_proc_simple_contact Pointer to the FE instance for post
   * processing
   * @param  common_data_simple_contact  Pointer to the common data for simple
   * contact element
   * @param  field_name                  String of field name for spatial
   * positions
   * @param  mesh_node_field_name        String of field name for material
   * positions
   * @param  lagrang_field_name          String of field name for Lagrange
   * multipliers
   * @param  side_fe_name                String of 3D element adjacent to the
   * present contact element
   * @return                             Error code
   *
   */
  MoFEMErrorCode setContactOperatorsRhsALEMaterial(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact_ale,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      const string field_name, const string mesh_node_field_name,
      const string lagrang_field_name, const string side_fe_name);

  /**
   * @brief Function for the simple contact element that sets the user data
   * LHS-operators
   *
   * @param  fe_lhs_simple_contact_ale  Pointer to the FE instance for LHS
   * @param  common_data_simple_contact Pointer to the common data for simple
   * contact element
   * @param  field_name                 String of field name for spatial
   * positions
   * @param  mesh_node_field_name       String of field name for material
   * positions
   * @param  lagrang_field_name         String of field name for Lagrange
   * multipliers
   * @param  side_fe_name               String of 3D element adjacent to the
   * present contact element
   * @return                            Error code
   *
   */
  MoFEMErrorCode setContactOperatorsLhsALE(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact_ale,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      const string field_name, const string mesh_node_field_name,
      const string lagrang_field_name, const string side_fe_name);
};

double SimpleContactProblem::Sign(double x) {
  if (x == 0)
    return 0;
  else if (x > 0)
    return 1;
  else
    return -1;
};

double SimpleContactProblem::ConstrainFunction(const double cn, const double g,
                                               const double l) {
  if ((cn * g) <= l)
    return cn * g;
  else
    return l;
}

double SimpleContactProblem::ConstrainFunction_dg(const double cn,
                                                  const double g,
                                                  const double l) {
  return cn * (1 + Sign(l - cn * g)) / static_cast<double>(2);
}

double SimpleContactProblem::ConstrainFunction_dl(const double cn,
                                                  const double g,
                                                  const double l) {
  return (1 + Sign(cn * g - l)) / static_cast<double>(2);
}

#endif