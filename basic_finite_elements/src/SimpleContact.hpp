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

#include <boost/enable_shared_from_this.hpp>

/** \brief Set of functions declaring elements and setting operators
 * to apply contact conditions between surfaces with matching
 * meshes \ingroup simple_contact_problem
 */

struct SimpleContactProblem {

  static constexpr double TOL = 1e-8;

  struct SimpleContactPrismsData {
    Range pRisms; // All boundary surfaces
  };

  map<int, SimpleContactPrismsData>
      setOfSimpleContactPrism; ///< maps side set id with appropriate FluxData

  struct SimpleContactElement
      : public MoFEM::ContactPrismElementForcesAndSourcesCore {

    MoFEM::Interface &mField;
    bool newtonCotes;
    SimpleContactElement(MoFEM::Interface &m_field, bool newton_cotes = false)
        : MoFEM::ContactPrismElementForcesAndSourcesCore(m_field),
          mField(m_field), newtonCotes(newton_cotes) {}

    int getRule(int order) {
      if (newtonCotes)
        return -1;
      else
        return 2 * order;
    }

    MoFEMErrorCode setGaussPts(int order);

    virtual ~SimpleContactElement() = default;
  };

  struct ContactElement : public SimpleContactElement {

    ContactElement(MoFEM::Interface &m_field, std::string spat_pos,
                   std::string mat_pos, bool newton_cotes = false)
        : SimpleContactElement(m_field, newton_cotes),
          sparialPositionsField(spat_pos), materialPositionsField(mat_pos) {}

    MoFEMErrorCode setGaussPts(int order);

  private:
    std::string sparialPositionsField;
    std::string materialPositionsField;

    VectorDouble spatialCoords;
    VectorDouble materialCoords;
    MatrixDouble slaveSpatialCoords;
    MatrixDouble masterMaterialCoords;
    MatrixDouble masterSpatialCoords;
    MatrixDouble A;
    VectorDouble F;
    MatrixDouble slaveN;
    MatrixDouble masterN;


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
  MoFEMErrorCode addContactElement(const string element_name,
                                   const string field_name,
                                   const string lagrang_field_name,
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

    boost::shared_ptr<VectorDouble> lagMultAtGaussPtsPtr;
    boost::shared_ptr<VectorDouble> gapPtr;
    boost::shared_ptr<VectorDouble> lagGapProdPtr;
    boost::shared_ptr<VectorDouble> tildeCFunPtr;
    boost::shared_ptr<VectorDouble> lambdaGapDiffProductPtr;

    boost::shared_ptr<VectorDouble> normalVectorSlavePtr;
    boost::shared_ptr<VectorDouble> normalVectorMasterPtr;

    double areaSlave;
    double areaMaster;

    CommonDataSimpleContact(MoFEM::Interface &m_field) : mField(m_field) {
      positionAtGaussPtsMasterPtr = boost::make_shared<MatrixDouble>();
      positionAtGaussPtsSlavePtr = boost::make_shared<MatrixDouble>();
      lagMultAtGaussPtsPtr = boost::make_shared<VectorDouble>();

      gapPtr = boost::make_shared<VectorDouble>();
      lagGapProdPtr = boost::make_shared<VectorDouble>();
      tildeCFunPtr = boost::make_shared<VectorDouble>();
      lambdaGapDiffProductPtr = boost::make_shared<VectorDouble>();
      normalVectorSlavePtr = boost::make_shared<VectorDouble>();
      normalVectorMasterPtr = boost::make_shared<VectorDouble>();
    }

  private:
    MoFEM::Interface &mField;
  };

  double rValue;
  double cnValue;
  bool newtonCotes;
  MoFEM::Interface &mField;

  SimpleContactProblem(MoFEM::Interface &m_field, double &r_value_regular,
                       double &cn_value, bool newton_cotes = false)
      : mField(m_field), rValue(r_value_regular), cnValue(cn_value),
        newtonCotes(newton_cotes) {}

  /// \brief Computes normal to slave face that is common to all gauss points
  struct OpGetNormalSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetNormalSlave(
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

  struct OpGetNormalMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetNormalMaster(
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

  /**
   * @brief Operator for the simple contact element
   *
   * Calculates the spacial coordinates of the gauss points of master triangle.
   *
   */
  struct OpGetPositionAtGaussPtsMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetPositionAtGaussPtsMaster(
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

  /**
   * @brief Operator for the simple contact element
   *
   * Calculates the spacial coordinates of the gauss points of slave triangle.
   *
   */
  struct OpGetPositionAtGaussPtsSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetPositionAtGaussPtsSlave(
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

  /**
   * @brief Operator for the simple contact element
   *
   * Calculates gap function at the gauss points on the slave triangle.
   *
   */
  struct OpGetGapSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    OpGetGapSlave(
        const string field_name, // ign: does it matter??
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
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
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Calculates Lagrange multipliers at the gauss points on the slave triangle.
   *
   */
  struct OpGetLagMulAtGaussPtsSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetLagMulAtGaussPtsSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Prints Lagrange multipliers and gaps evaluated at the gauss points on the
   * slave triangle.
   *
   */
  struct OpPrintLagMulAtGaussPtsSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpPrintLagMulAtGaussPtsSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Calgulates the product of Lagrange multipliers with gaps evaluated at the
   * gauss points on the slave triangle.
   *
   */
  struct OpLagGapProdGaussPtsSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpLagGapProdGaussPtsSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief RHS-operator for the simple contact element
   *
   * Integrates Lagrange multipliers virtual work on
   * master surfaceand assemble components to RHS global vector.
   *
   */
  struct OpCalContactTractionOnMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalContactTractionOnMaster(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    VectorDouble vec_f;

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
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief RHS-operator for the simple contact element
   *
   * Integrates Lagrange multipliers virtual work on
   * slave surface and assembles components to the RHS global vector.
   *
   */
  struct OpCalContactTractionOnSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalContactTractionOnSlave(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    VectorDouble vec_f;

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
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Computes function that fulfils KKT conditions
   * when equal to zero
   *
   */
  struct OpGetCompFunSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    double r;  //@todo: ign: to become input parameter
    double cN; //@todo: ign: to become input parameter

    /**
     * @brief Evaluates the complementarity function at slave face
     * gauss points
     *
     * Computes the complementarity function that fulfils KKT conditions when
     * equal to zero
     *
     * \f[
     * {C(\lambda, \mathbf{x}^{(i)})} :=  \lambda + c_{\text
     * n} g_{\textrm{n}} - \dfrac{1}{r}{\left| \lambda - c_{\text n}
     * g_{\textrm{n}}\right|}^{r} \f]
     *
     * where \f$ \lambda\f$ is the Lagrange multiplier, \f$\mathbf{x}^{(i)}\f$
     * are the coordinates of the overlapping gauss points at slave and master
     * triangles for  \f$i = 1\f$ and \f$i = 2\f$, respectively. Furthermore,
     * \f$ c_{\text n}\f$ works as an augmentation parameter and affects
     * convergence, \f$r\f$ is regularisation parameter that can be chosen
     * in \f$[1, 1.1]\f$ and \f$ g_{\textrm{n}}\f$ is the gap function
     * evaluated at the slave triangle gauss points as:
     * \f[
     * g_{\textrm{n}} = - \mathbf{n}(\mathbf{x}^{(1)}) \cdot \left(
     * \mathbf{x}^{(1)} - \mathbf{x}^{(2)}  \right)
     * \f]
     *
     *
     */
    OpGetCompFunSlave(
        const string lagrang_field_name, // ign: does it matter?
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        double &r_value, double &cn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), r(r_value),
          cN(cn_value) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief RHS-operator for the simple contact element
   *
   * Integrates complementarity function that fulfills KKT
   * conditions over slave contact area and assembles components of the RHS
   * vector.
   *
   */
  struct OpCalIntCompFunSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalIntCompFunSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    VectorDouble vecR;

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
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief LHS-operator for the simple contact element
   *
   * Integrates Lagrange multipliers virtual
   * work, \f$ \delta W_{\text c}\f$ derivative with respect to Lagrange
   * multipliers on master side and assembles components of the RHS vector.
   *
   */
  struct OpCalContactTractionOverLambdaMasterSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalContactTractionOverLambdaMasterSlave(
        const string field_name, const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, lagrang_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact), Aij(aij) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

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
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
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
  struct OpCalContactTractionOverLambdaSlaveSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalContactTractionOverLambdaSlaveSlave(
        const string field_name, const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, lagrang_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact), Aij(aij) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

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
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
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
  struct OpCalDerIntCompFunOverLambdaSlaveSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalDerIntCompFunOverLambdaSlaveSlave(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          Aij(aij), commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all entities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

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
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
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
  struct OpCalDerIntCompFunOverSpatPosSlaveMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalDerIntCompFunOverSpatPosSlaveMaster(
        const string field_name, const string lagrang_field_name,
        double &cn_value,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVEMASTER),
          cN(cn_value), Aij(aij), commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all entities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

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
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
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
  struct OpCalDerIntCompFunOverSpatPosSlaveSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalDerIntCompFunOverSpatPosSlaveSlave(
        const string field_name, const string lagrang_field_name,
        double &cn_value,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          cN(cn_value), Aij(aij), commonDataSimpleContact(common_data_contact) {
      sYmm = false; // This will make sure to loop over all entities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

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
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  /**
   * @brief Operator for the simple contact element
   *
   * Prints to .vtk file pre-calculated gaps, Lagrange multipliers and their
   * productat the gauss points on the slave triangle.
   *
   */
  struct OpMakeVtkSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    MoFEM::Interface &mField;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    moab::Interface &moabOut;
    bool lagFieldSet;

    OpMakeVtkSlave(MoFEM::Interface &m_field, string field_name,
                   boost::shared_ptr<CommonDataSimpleContact> &common_data,
                   moab::Interface &moab_out, bool lagrange_field = true)
        : MoFEM::ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          mField(m_field), commonDataSimpleContact(common_data),
          moabOut(moab_out), lagFieldSet(lagrange_field) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpMakeTestTextFile
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    MoFEM::Interface &mField;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    bool lagFieldSet;
    std::ofstream &mySplit;
    // stream<tee_device<std::ostream, std::ofstream>> &mySplit;

    OpMakeTestTextFile(MoFEM::Interface &m_field, string field_name,
                       boost::shared_ptr<CommonDataSimpleContact> &common_data,
                       std::ofstream &_my_split, bool lagrange_field = true)
        : MoFEM::ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          mField(m_field), commonDataSimpleContact(common_data),
          lagFieldSet(lagrange_field), mySplit(_my_split) {
      mySplit << fixed << setprecision(8);
      mySplit << "[0] Lagrange multiplier [1] Gap" << endl;
    }

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

    ~OpMakeTestTextFile() {}
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
      string field_name, string lagrang_field_name, Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalContactTractionOnMaster(field_name,
                                           common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalContactTractionOnSlave(field_name,
                                          common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetCompFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalIntCompFunSlave(lagrang_field_name,
                                   common_data_simple_contact, f_));
    }
    MoFEMFunctionReturn(0);
  }

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
   * @param  aij                        Left hand side matrix
   * @return                            Error code
   *
   */
  MoFEMErrorCode setContactOperatorsLhs(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name, Mat aij) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpCalContactTractionOverLambdaMasterSlave(
              field_name, lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpCalContactTractionOverLambdaSlaveSlave(
              field_name, lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetCompFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpCalDerIntCompFunOverLambdaSlaveSlave(
              lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpCalDerIntCompFunOverSpatPosSlaveMaster(
              field_name, lagrang_field_name, cnValue,
              common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpCalDerIntCompFunOverSpatPosSlaveSlave(
              field_name, lagrang_field_name, cnValue,
              common_data_simple_contact, aij));
    }
    MoFEMFunctionReturn(0);
  }

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
      moab::Interface &moab_out, bool lagrange_field = true) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMaster(field_name, common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpLagGapProdGaussPtsSlave(lagrang_field_name,
                                        common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpMakeVtkSlave(m_field, field_name, common_data_simple_contact,
                             moab_out, lagrange_field));
    }
    MoFEMFunctionReturn(0);
  }
};

#endif