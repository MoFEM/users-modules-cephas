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
#endif

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
      //}

      CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                         field_name);

      CHKERR
      mField.modify_finite_element_add_field_data(element_name,
                                                  "MESH_NODE_POSITIONS");

      setOfSimpleContactPrism[1].pRisms = range_slave_master_prisms;

      // Adding range_slave_master_prisms to Element element_name
      CHKERR mField.add_ents_to_finite_element_by_type(
          range_slave_master_prisms, MBPRISM, element_name);
    }

    MoFEMFunctionReturn(0);
  }

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

  struct CommonDataSimpleContact {
    vector<vector<VectorDouble>> tangentSlaveALE;
    vector<vector<VectorDouble>> tangentMasterALE;

    vector<VectorDouble> normalSlaveALE;
    vector<VectorDouble> normalMasterALE;

    boost::shared_ptr<VectorDouble> normalSlaveLengthALEPtr;
    boost::shared_ptr<VectorDouble> normalMasterLengthALEPtr;

    boost::shared_ptr<MatrixDouble> positionAtGaussPtsMasterPtr;
    boost::shared_ptr<MatrixDouble> positionAtGaussPtsSlavePtr;
    boost::shared_ptr<VectorDouble> lagMultAtGaussPtsPtr;
    boost::shared_ptr<VectorDouble> gapPtr;
    boost::shared_ptr<VectorDouble> lagGapProdPtr;
    boost::shared_ptr<VectorDouble> tildeCFunPtr;
    boost::shared_ptr<VectorDouble> lambdaGapDiffProductPtr;
    boost::shared_ptr<VectorDouble> normalVectorSlavePtr;
    boost::shared_ptr<VectorDouble> normalVectorMasterPtr;

    boost::shared_ptr<MatrixDouble> hMatMaster;
    boost::shared_ptr<MatrixDouble> FMatMaster;
    boost::shared_ptr<MatrixDouble> HMatMaster;
    boost::shared_ptr<VectorDouble> detHVecMaster;
    boost::shared_ptr<MatrixDouble> invHMatMaster;

    boost::shared_ptr<MatrixDouble> hMatSlave;
    boost::shared_ptr<MatrixDouble> FMatSlave;
    boost::shared_ptr<MatrixDouble> HMatSlave;
    boost::shared_ptr<VectorDouble> detHVecSlave;
    boost::shared_ptr<MatrixDouble> invHMatSlave;

    DataForcesAndSourcesCore::EntData *faceRowDataMaster;
    DataForcesAndSourcesCore::EntData *faceRowDataSlave;

    double areaSlave;

    CommonDataSimpleContact(MoFEM::Interface &m_field) : mField(m_field) {
      positionAtGaussPtsMasterPtr = boost::make_shared<MatrixDouble>();
      positionAtGaussPtsSlavePtr = boost::make_shared<MatrixDouble>();
      lagMultAtGaussPtsPtr = boost::make_shared<VectorDouble>();
      gapPtr = boost::make_shared<VectorDouble>();
      normalSlaveLengthALEPtr = boost::make_shared<VectorDouble>();
      normalMasterLengthALEPtr = boost::make_shared<VectorDouble>();
      lagGapProdPtr = boost::make_shared<VectorDouble>();
      tildeCFunPtr = boost::make_shared<VectorDouble>();
      lambdaGapDiffProductPtr = boost::make_shared<VectorDouble>();
      normalVectorSlavePtr = boost::make_shared<VectorDouble>();
      normalVectorMasterPtr = boost::make_shared<VectorDouble>();

      hMatMaster = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      FMatMaster = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      HMatMaster = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      detHVecMaster = boost::shared_ptr<VectorDouble>(new VectorDouble());
      invHMatMaster = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      hMatSlave = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      FMatSlave = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      HMatSlave = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
      detHVecSlave = boost::shared_ptr<VectorDouble>(new VectorDouble());
      invHMatSlave = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

      faceRowDataMaster = nullptr;
      faceRowDataSlave = nullptr;
    }

  private:
    MoFEM::Interface &mField;
  };

  double rValue;
  double cnValue;
  bool newtonCotes;
  boost::shared_ptr<SimpleContactElement> feRhsSimpleContact;
  boost::shared_ptr<SimpleContactElement> feLhsSimpleContact;
  boost::shared_ptr<SimpleContactElement> fePostProcSimpleContact;
  boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  MoFEM::Interface &mField;

  SimpleContactProblem(MoFEM::Interface &m_field, double &r_value_regular,
                       double &cn_value, bool newton_cotes = false)
      : mField(m_field), rValue(r_value_regular), cnValue(cn_value),
        newtonCotes(newton_cotes) {
    commonDataSimpleContact =
        boost::make_shared<CommonDataSimpleContact>(mField);
    feRhsSimpleContact =
        boost::make_shared<SimpleContactElement>(mField, newtonCotes);
    feLhsSimpleContact =
        boost::make_shared<SimpleContactElement>(mField, newtonCotes);
    fePostProcSimpleContact =
        boost::make_shared<SimpleContactElement>(mField, newtonCotes);
  }

  struct OpGetTangentSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetTangentSlave(
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

  struct OpGetTangentMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetTangentMaster(
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

  /// \brief tangents t1 and t2 to face f4 at all gauss points
  struct OpGetNormalSlaveForSide
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetNormalSlaveForSide(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

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

  struct OpLoopSlaveForSide
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpLoopSlaveForSide(
        const string field_name,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          sideFe(side_fe), sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalMatForcesALEMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    VectorDouble vec_f;
    Vec F;
    OpCalMatForcesALEMaster(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalMatForcesALESlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalMatForcesALESlave(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpContactConstraintMatrixMasterSlaveForSide
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixMasterSlaveForSide(
        const string field_name, const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        std::string &side_fe_name, Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, lagrang_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name), Aij(aij) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  // Calculate displacements or spatial positions at Gauss points  (Note
  // OPCOL
  // here, which is dips/sp-pos here)
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

  // Calculate displacements or spatial positions at Gauss points  (Note
  // OPCOL
  // here, which is dips/sp-pos here)
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

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpGetGapSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    OpGetGapSlaveALE(
        const string field_name, // ign: does it matter??
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

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

  struct OpGetLagMulAtGaussPtsSlaveHdiv
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetLagMulAtGaussPtsSlaveHdiv(
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
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

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
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalFReConSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalFReConSlaveALE(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    VectorDouble vec_f;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalFReConMasterALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalFReConMasterALE(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    VectorDouble vec_f;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalTildeCFunSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    double r;  //@todo: ign: to become input parameter
    double cN; //@todo: ign: to become input parameter

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
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalIntTildeCFunSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalIntTildeCFunSlaveALE(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    VectorDouble vecR;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalIntTildeCFunSlaveHdiv
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalIntTildeCFunSlaveHdiv(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    VectorDouble vecR;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

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

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactConstraintMatrixSlaveSlave_dX
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixSlaveSlave_dX(
        const string field_name, const string mesh_nodes_field,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, mesh_nodes_field, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact), Aij(aij) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactConstraintMatrixMasterSlave_dX
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixMasterSlave_dX(
        const string field_name, const string mesh_nodes_field,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, mesh_nodes_field, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact), Aij(aij) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactConstraintMatrixMasterSlaveHdiv
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixMasterSlaveHdiv(
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

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

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

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactConstraintMatrixSlaveSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixSlaveSlaveALE(
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

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactConstraintMatrixMasterSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixMasterSlaveALE(
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

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactConstraintMatrixSlaveSlaveHdiv
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixSlaveSlaveHdiv(
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

  struct OpDerivativeBarTildeCFunOLambdaSlaveSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunOLambdaSlaveSlaveALE(
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

  struct OpDerivativeBarTildeCFunOLambdaSlaveSlaveHdiv
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunOLambdaSlaveSlaveHdiv(
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

  struct OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE(
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

  struct OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE_dX
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE_dX(
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

  struct OpDerivativeBarTildeCFunODisplacementsSlaveMasterHdiv
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunODisplacementsSlaveMasterHdiv(
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

  struct OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE(
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

  struct OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE_dX
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE_dX(
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

  struct OpDerivativeBarTildeCFunODisplacementsSlaveSlaveHdiv
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunODisplacementsSlaveSlaveHdiv(
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

  struct OpMakeVtkSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    MoFEM::Interface &mField;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    moab::Interface &moabOut;

    OpMakeVtkSlave(MoFEM::Interface &m_field, string field_name,
                   boost::shared_ptr<CommonDataSimpleContact> &common_data,
                   moab::Interface &moab_out)
        : MoFEM::ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          mField(m_field), commonDataSimpleContact(common_data),
          moabOut(moab_out) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  // setup operators for calculation of active set
  MoFEMErrorCode setContactOperatorsRhsOperators(string field_name,
                                                 string lagrang_field_name,
                                                 string side_fe_name,
                                                 Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      // feMatSideRhs->getOpPtrVector().push_back(
      //     new OpCalculateVectorFieldGradient<3, 3>(X_field,
      //     data_at_pts->HMat));
      // feMatSideRhs->getOpPtrVector().push_back(
      //     new OpCalculateVectorFieldGradient<3, 3>(x_field,
      //     data_at_pts->hMat));
      // feMatSideRhs->getOpPtrVector().push_back(
      //     new OpCalculateDeformation(X_field, data_at_pts, ho_geometry));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetNormalSlaveForSide(field_name, commonDataSimpleContact,
                                      feMatSideRhs, side_fe_name));

      // feRhsSimpleContact->getOpPtrVector().push_back(
      //     new OpContactConstraintMatrixMasterSlaveForSide(
      //         field_name, lagrang_field_name, commonDataSimpleContact,
      //         feMatSideRhs, side_fe_name));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpCalFReConMaster(field_name, commonDataSimpleContact, f_));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpCalFReConSlave(field_name, commonDataSimpleContact, f_));

      feRhsSimpleContact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, commonDataSimpleContact, rValue, cnValue));

      feRhsSimpleContact->getOpPtrVector().push_back(new OpCalIntTildeCFunSlave(
          lagrang_field_name, commonDataSimpleContact, f_));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsRhsOperatorsHdiv(string field_name,
                                                     string lagrang_field_name,
                                                     Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlaveHdiv(lagrang_field_name,
                                             commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpCalFReConMaster(field_name, commonDataSimpleContact, f_));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpCalFReConSlave(field_name, commonDataSimpleContact, f_));

      feRhsSimpleContact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, commonDataSimpleContact, rValue, cnValue));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpCalIntTildeCFunSlaveHdiv(lagrang_field_name,
                                         commonDataSimpleContact, f_));
    }
    MoFEMFunctionReturn(0);
  }

  // setup operators for calculation of active set
  MoFEMErrorCode setContactOperatorsRhsALE(const string field_name,
                                           const string mesh_node_field_name,
                                           const string lagrang_field_name,
                                           const string side_fe_name,
                                           Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideRhsMaster =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSideRhsMaster->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              mesh_node_field_name, commonDataSimpleContact->HMatMaster));
      feMatSideRhsMaster->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, commonDataSimpleContact->hMatMaster));

      feMatSideRhsMaster->getOpPtrVector().push_back(new OpCalculateDeformation(
          mesh_node_field_name, commonDataSimpleContact, false));

      feRhsSimpleContact->getOpPtrVector().push_back(new OpLoopMasterForSide(
          mesh_node_field_name, feMatSideRhsMaster, side_fe_name));

      // feRhsSimpleContact->getOpPtrVector().push_back(
      //     new OpContactConstraintMatrixMasterSlaveForSide(
      //         field_name, lagrang_field_name, commonDataSimpleContact,
      //         feMatSideRhs, side_fe_name));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetTangentSlave(mesh_node_field_name, commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
          mesh_node_field_name, commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetGapSlaveALE(field_name, commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpCalFReConMasterALE(field_name, commonDataSimpleContact, f_));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpCalFReConSlaveALE(field_name, commonDataSimpleContact, f_));

      feRhsSimpleContact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, commonDataSimpleContact, rValue, cnValue));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpCalIntTildeCFunSlaveALE(lagrang_field_name,
                                        commonDataSimpleContact, f_));

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpCalMatForcesALEMaster(mesh_node_field_name,
                                      commonDataSimpleContact));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsOperators(string field_name,
                                                 string lagrang_field_name,
                                                 Mat aij) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixMasterSlave(
              field_name, lagrang_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlave(
              field_name, lagrang_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, commonDataSimpleContact, rValue, cnValue));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunOLambdaSlaveSlave(
              lagrang_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveMaster(
              field_name, lagrang_field_name, cnValue, commonDataSimpleContact,
              aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlave(
              field_name, lagrang_field_name, cnValue, commonDataSimpleContact,
              aij));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsALE(const string field_name,
                                           const string mesh_node_field_name,
                                           const string lagrang_field_name,
                                           const string side_fe_name,
                                           Mat aij = NULL) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      feRhsSimpleContact->getOpPtrVector().push_back(
          new OpGetTangentSlave(mesh_node_field_name, commonDataSimpleContact));

      feRhsSimpleContact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
          mesh_node_field_name, commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetGapSlaveALE(field_name, commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixMasterSlaveALE(
              field_name, lagrang_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlaveALE(
              field_name, lagrang_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlave_dX(
              field_name, mesh_node_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixMasterSlave_dX(
              field_name, mesh_node_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, commonDataSimpleContact, rValue, cnValue));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunOLambdaSlaveSlaveALE(
              lagrang_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE(
              field_name, lagrang_field_name, cnValue, commonDataSimpleContact,
              aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE(
              field_name, lagrang_field_name, cnValue, commonDataSimpleContact,
              aij));

      // feLhsSimpleContact->getOpPtrVector().push_back(
      //     new OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE_dX(
      //         mesh_node_field_name, lagrang_field_name, cnValue,
      //         commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE_dX(
              mesh_node_field_name, lagrang_field_name, cnValue,
              commonDataSimpleContact, aij));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsOperatorsHdiv(string field_name,
                                                     string lagrang_field_name,
                                                     Mat aij) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlaveHdiv(lagrang_field_name,
                                             commonDataSimpleContact));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixMasterSlaveHdiv(
              field_name, lagrang_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlaveHdiv(
              field_name, lagrang_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, commonDataSimpleContact, rValue, cnValue));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunOLambdaSlaveSlaveHdiv(
              lagrang_field_name, commonDataSimpleContact, aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveMasterHdiv(
              field_name, lagrang_field_name, cnValue, commonDataSimpleContact,
              aij));

      feLhsSimpleContact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlaveHdiv(
              field_name, lagrang_field_name, cnValue, commonDataSimpleContact,
              aij));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsForPostProc(MoFEM::Interface &m_field,
                                                string field_name,
                                                string lagrang_field_name,
                                                moab::Interface &moab_out) {
    MoFEMFunctionBegin;
    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpLagGapProdGaussPtsSlave(lagrang_field_name,
                                        commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(new OpMakeVtkSlave(
          m_field, field_name, commonDataSimpleContact, moab_out));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsForPostProcHdiv(MoFEM::Interface &m_field,
                                                    string field_name,
                                                    string lagrang_field_name,
                                                    moab::Interface &moab_out) {
    MoFEMFunctionBegin;
    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlaveHdiv(lagrang_field_name,
                                             commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(
          new OpLagGapProdGaussPtsSlave(lagrang_field_name,
                                        commonDataSimpleContact));

      fePostProcSimpleContact->getOpPtrVector().push_back(new OpMakeVtkSlave(
          m_field, field_name, commonDataSimpleContact, moab_out));
    }
    MoFEMFunctionReturn(0);
  }
};

#endif