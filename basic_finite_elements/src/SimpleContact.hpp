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

#ifdef __cplusplus
extern "C" {
#endif
#include <cblas.h>
// #include <lapack_wrap.h>
// #include <gm_rule.h>
//#include <quad.h>
#include <triangle_ncc_rule.h>

#ifdef __cplusplus
}
#include <boost/enable_shared_from_this.hpp>
#endif

#define TOL 1.e-8
namespace bio = boost::iostreams;
using bio::stream;
using bio::tee_device;

/** \brief Set of functions declaring elements and setting operators
 * to apply contact conditions
 * \ingroup simple_contact_problem
 */

struct SimpleContactProblem {

  struct SimpleContactPrismsData {
    Range pRisms; // All boundary surfaces
  };

  map<int, SimpleContactPrismsData>
      setOfSimpleContactPrism; ///< maps side set id with appropriate FluxData

  struct SimpleContactElement
      : public MoFEM::ContactPrismElementForcesAndSourcesCore {

    MoFEM::Interface &mField;
    // map<int, SimpleContactPrismsData> &setOfSimpleContactPrisms;
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

    ~SimpleContactElement() {}
  };

  /**
   * @brief LHS-operator for pressure element (spatial configuration)
   *
   * Computes linearisation of the spatial component with respect to
   * material coordinates.
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

      //============================================================================================================
      // C row as Lagrange_mul and col as DISPLACEMENT
      if (lagrange_field)
        CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                          lagrang_field_name);

      CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                        field_name);

      // CT col as Lagrange_mul and row as DISPLACEMENT
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

  /// \brief tangents t1 and t2 to face f4 at all gauss points
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
   * @brief operator for the simple contact element
   *
   * Calgulates the spacial coordinated of the gauss points of master triangle.
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
   * @brief operator for the simple contact element
   *
   * Calgulates the spacial coordinated of the gauss points of slave triangle.
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
   * @brief operator for the simple contact element
   *
   * Calgulates gap function at the gauss points on the slave triangle.
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
     * Computes linearisation of the material component
     * with respect to a variation of material coordinates:
     * \f[
     * g_{\textrm{n}} = - \mathbf{n}(\mathbf{x}^{(1)}) \cdot \left(
     * \mathbf{x}^{(1)} - \mathbf{x}^{(2)}  \right)
     * \f]
     * where \f$\mathbf{n}(\mathbf{x}^{(1))}\f$ is the outward normal vector at
     * the slave triangle gauss points, \f$\mathbf{x}^{(1)}\f$ and
     * \f$\mathbf{x}^{(2)}\f$ are the spatial coordinated of the overlapping
     * gauss points located at the slave and master triangles, respectively.
     *
     *
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief operator for the simple contact element
   *
   * Calgulates lagrange multipliers at the gauss points on the slave triangle.
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
   * @brief operator for the simple contact element
   *
   * Prints lagrange multipliers and gaps evaluated at the gauss points on the
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
   * @brief operator for the simple contact element
   *
   * Calgulates the product of lagrange multipliers with gaps evaluated at the
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
   * Integrates and assembles lagrange multipliers virtual work on
   * master surface.
   *
   */
  struct OpCalFReConMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalFReConMaster(
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
     * @brief Integrates and assembles lagrange multipliers virtual work on
     * master surface
     *
     * Integrates and assembles lagrange multipliers virtual work, \f$ \delta
     * W_{\text c}\f$ on master surface
     *
     * \f[
     * {\delta
     * W_{\text c}(\lambda, \mathbf{x}^{(2)},
     * \delta \mathbf{x}^{(2)}}) \,\,  =
     * \int_{{\gamma}^{(2)}_{\text c}} \lambda 
     * \delta{\mathbf{x}^{(2)}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where, \f${\gamma}^{(2)}_{\text c}\f$ is the surface integration domain
     * of the master surface, \f$ \lambda\f$ is the lagrange multiplier,
     * \f$\mathbf{x}^{(2)}\f$ are the coordinated of the overlapping gauss
     * points at and master triangles.
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief RHS-operator for the simple contact element
   *
   * Integrates and assembles lagrange multipliers virtual work on
   * slave surface.
   *
   */
  struct OpCalFReConSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalFReConSlave(
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
     * @brief Integrates and assembles lagrange multipliers virtual work on
     * slave surface
     *
     * Integrates and assembles lagrange multipliers virtual work, \f$ \delta
     * W_{\text c}\f$ on slave surface
     *
     * \f[
     * {\delta
     * W_{\text c}(\lambda, \mathbf{x}^{(1)},
     * \delta \mathbf{x}^{(1)}}) \,\,  = -
     * \int_{{\gamma}^{(1)}_{\text c}} \lambda
     * \delta{\mathbf{x}^{(1)}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where, \f${\gamma}^{(1)}_{\text c}\f$ is the surface integration domain
     * of the slave surface, \f$ \lambda\f$ is the lagrange multiplier,
     * \f$\mathbf{x}^{(1)}\f$ are the coordinated of the overlapping gauss
     * points at and slave triangles.
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalTildeCFunSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    double r;  //@todo: ign: to become input parameter
    double cN; //@todo: ign: to become input parameter

    /**
     * @brief Evaluates the complementarity function function at slave face
     * gauss points
     *
     * Compbutes the complementarity function that fulfills KKT conditions when
     * equal to zero:
     *
     * \f[
     * {\overline C(\lambda, \mathbf{x}^{(i)})} :=  \lambda - c_{\text
     * n} g_{\textrm{n}}  - \dfrac{1}{r}{\left| \lambda + c_{\text n}
     * g_{\textrm{n}}\right|}^{r} \f]
     *
     * where, \f$ \lambda\f$ is the lagrange multiplier, \f$\mathbf{x}^{(i)}\f$
     * are the coordinated of the overlapping gauss points at slave and master
     * triangles for  \f$i = 1\f$ and \f$i = 2\f$, respectively. Furthermore,
     * \f$ c_{\text n}\f$ works as an augmentation parameter and affects
     * convergence, \f$r\f$ is regularisation parameter that can be chosen
     * between in \f$[1, 1.1]\f$ and \f$ g_{\textrm{n}}\f$ is the gap function
     * evaluated at the slave triangle gauss points as:
     * * \f[
     * g_{\textrm{n}} = - \mathbf{n}(\mathbf{x}^{(1)}) \cdot \left(
     * \mathbf{x}^{(1)} - \mathbf{x}^{(2)}  \right)
     * \f]
     *
     *
     */
    OpCalTildeCFunSlave(
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
   * Integrates and assembles complementarity function that funfills KKT
   * conditions over slave contact area.
   *
   */
  struct OpCalIntTildeCFunSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalIntTildeCFunSlave(
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
     * @brief Integrates and assembles the complementarity function at slave
     * face gauss points
     *
     * Integrates and assembles the complementarity function to fulfills KKT
     * conditions in the integral sence:
     *
     * \f[
     * {\overline C(\lambda, \mathbf{x}^{(i)},
     * \delta \lambda)} = \int_{{\gamma}^{(1)}_{\text
     * c}} \left( \lambda - c_{\text n} g_{\textrm{n}}  - \dfrac{1}{r}{\left|
     * \lambda + c_{\text n} g_{\textrm{n}}\right|}^{r}\right) \delta{{\lambda}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where, \f${\gamma}^{(1)}_{\text c}\f$ is the surface integration domain
     * of the slave surface, \f$ \lambda\f$ is the lagrange multiplier,
     * \f$\mathbf{x}^{(i)}\f$ are the coordinated of the overlapping gauss
     * points at slave and master triangles for  \f$i = 1\f$ and \f$i = 2\f$,
     * respectively. Furthermore, \f$ c_{\text n}\f$ works as an augmentation
     * parameter and affects convergence, \f$r\f$ is regularisation parameter
     * that can be chosen between in \f$[1, 1.1]\f$ (\f$r = 1\f$) is the default value) 
     * and \f$ g_{\textrm{n}}\f$ is
     * the gap function evaluated at the slave triangle gauss points as:
     * \f[
     * g_{\textrm{n}} = - \mathbf{n}(\mathbf{x}^{(1)}) \cdot \left(
     * \mathbf{x}^{(1)} - \mathbf{x}^{(2)}  \right)
     * \f]
     */
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  /**
   * @brief LHS-operator for the simple contact element
   *
   * Integrates and assembles for over master side.
   *
   */
  struct OpContactConstraintMatrixMasterSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixMasterSlave(
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
     * @brief Integrates and assembles for over master side
     *
     * Computes linearisation of and assembles lagrange multipliers virtual work, \f$ \delta
     * W_{\text c}\f$ with respect to lagrange multipliers
     *
     * \f[
     * {\text D} \int_{{\gamma}^{(2)}_{\text c}} {\delta
     * W_{\text c}(\lambda, \mathbf{x}^{(2)},
     * \delta \mathbf{x}^{(2)}})[\delta lambda]  =
     * \int_{{\gamma}^{(2)}_{\text c}} \delta \lambda
     * \delta{\mathbf{x}^{(2)}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where, \f${\gamma}^{(2)}_{\text c}\f$ is the surface integration domain
     * of the master surface, \f$ \lambda\f$ is the lagrange multiplier,
     * \f$\mathbf{x}^{(2)}\f$ are the coordinated of the overlapping gauss
     * points at and master triangles.
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  /**
   * @brief LHS-operator for the simple contact element
   *
   * Integrates and assembles for over slave side.
   *
   */
  struct OpContactConstraintMatrixSlaveSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixSlaveSlave(
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
     * @brief Integrates and assembles for over slave side
     *
     * Computes linearisation of and assembles lagrange multipliers virtual
     * work, \f$ \delta W_{\text c}\f$ with respect to lagrange multipliers
     *
     * \f[
     * {\text D} \int_{{\gamma}^{(1)}_{\text c}} {\delta
     * W_{\text c}(\lambda, \mathbf{x}^{(1)},
     * \delta \mathbf{x}^{(1)}})[\delta lambda]  =
     * \int_{{\gamma}^{(1)}_{\text c}} \delta \lambda
     * \delta{\mathbf{x}^{(1)}}
     * \,\,{ {\text d} {\gamma}}
     * \f]
     * where, \f${\gamma}^{(1)}_{\text c}\f$ is the surface integration domain
     * of the slave surface, \f$ \lambda\f$ is the lagrange multiplier,
     * \f$\mathbf{x}^{(1)}\f$ are the coordinated of the overlapping gauss
     * points at and slave triangles.
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpDerivativeBarTildeCFunOLambdaSlaveSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunOLambdaSlaveSlave(
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
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpDerivativeBarTildeCFunODisplacementsSlaveMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunODisplacementsSlaveMaster(
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
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpDerivativeBarTildeCFunODisplacementsSlaveSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunODisplacementsSlaveSlave(
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
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  /**
   * @brief operator for the simple contact element
   *
   * Prints to .vtk file pre-calculated gaps, lagrange multipliers and their
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

    stream<tee_device<std::ostream, std::ofstream>> &mySplit;

    OpMakeTestTextFile(
        MoFEM::Interface &m_field, string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data,
        stream<tee_device<std::ostream, std::ofstream>> &_my_split,
        bool lagrange_field = true)
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
   * @brief function for the simple contact element that sets the user data
   * RHS-operators
   *
   * @param  fe_rhs_simple_contact      Pointer to the FE instance for RHS
   * @param  common_data_simple_contact Pointer to the common data for simple
   * contact element
   * @param  field_name                 String of field name for spatial
   * positions
   * @param  lagrang_field_name         String of field name for lagrange
   * multipliers
   * @param  f_                         Right hand side vector
   * @return                            Error code
   *
   */
  MoFEMErrorCode setContactOperatorsRhsOperators(
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
          new OpCalFReConMaster(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConSlave(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalIntTildeCFunSlave(lagrang_field_name,
                                     common_data_simple_contact, f_));
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * @brief function for the simple contact element that sets the user data
   * LHS-operators
   *
   * @param  fe_lhs_simple_contact      Pointer to the FE instance for LHS
   * @param  common_data_simple_contact Pointer to the common data for simple
   * contact element
   * @param  field_name                 String of field name for spatial
   * positions
   * @param  lagrang_field_name         String of field name for lagrange
   * multipliers
   * @param  aij                        Left hand side matrix
   * @return                            Error code
   *
   */
  MoFEMErrorCode setContactOperatorsLhsOperators(
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
          new OpContactConstraintMatrixMasterSlave(
              field_name, lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlave(
              field_name, lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunOLambdaSlaveSlave(
              lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveMaster(
              field_name, lagrang_field_name, cnValue,
              common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlave(
              field_name, lagrang_field_name, cnValue,
              common_data_simple_contact, aij));
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * @brief function for the simple contact element that sets the user data
   * post processing operators
   *
   * @param  fe_post_proc_simple_contact Pointer to the FE instance for post
   * processing
   * @param  common_data_simple_contact  Pointer to the common data for simple
   * contact element
   * @param  field_name                  String of field name for spatial
   * positions
   * @param  lagrang_field_name          String of field name for lagrange
   * multipliers
   * @param  moab_out                    Left hand side matrix
   * @param  lagrange_field              Booleand to determine existance of
   * lagrange field
   * @return                             Error code
   *
   */
  MoFEMErrorCode setContactOperatorsForPostProc(
      boost::shared_ptr<SimpleContactElement> fe_post_proc_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      MoFEM::Interface &m_field, string field_name, string lagrang_field_name,
      moab::Interface &moab_out,
      bool lagrange_field = true) {
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