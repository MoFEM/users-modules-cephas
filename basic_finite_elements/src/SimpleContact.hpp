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

  struct CommonDataSimpleContact
      : public boost::enable_shared_from_this<CommonDataSimpleContact> {

    NonlinearElasticElement::CommonData elasticityCommonData;
    NonlinearElasticElement::CommonData elasticityCommonDataMaster;
    NonlinearElasticElement::CommonData elasticityCommonDataSlave;
    short int tAg;
    short int tAg2;

    //NonlinearElasticElement::BlockData elasticBlockData;

    std::map<std::string, std::vector<VectorDouble>> dataAtGaussPts;
    std::map<std::string, std::vector<MatrixDouble>> gradAtGaussPts;

    std::map<std::string, std::vector<VectorDouble>> dataAtGaussPtsMaster;
    std::map<std::string, std::vector<MatrixDouble>> gradAtGaussPtsMaster;

    std::map<std::string, std::vector<VectorDouble>> dataAtGaussPtsSlave;
    std::map<std::string, std::vector<MatrixDouble>> gradAtGaussPtsSlave;

    string spatialPositions;
    string meshPositions;
    std::vector<MatrixDouble3by3> sTress;
    std::vector<MatrixDouble> jacStress;


    vector<vector<VectorDouble>> tangentSlaveALE;
    vector<vector<VectorDouble>> tangentMasterALE;

    vector<VectorDouble> normalSlaveALE;
    vector<VectorDouble> normalMasterALE;

    boost::shared_ptr<VectorDouble> normalSlaveLengthALEPtr;
    boost::shared_ptr<VectorDouble> normalMasterLengthALEPtr;

    boost::shared_ptr<MatrixDouble> positionAtGaussPtsMasterPtr;
    boost::shared_ptr<MatrixDouble> positionAtGaussPtsSlavePtr;

    boost::shared_ptr<MatrixDouble> meshPositionAtGaussPtsMasterPtr;
    boost::shared_ptr<MatrixDouble> meshPositionAtGaussPtsSlavePtr;

    boost::shared_ptr<MatrixDouble> divPAtPts;

    boost::shared_ptr<VectorDouble> lagMultAtGaussPtsPtr;
    boost::shared_ptr<VectorDouble> lagMultAtGaussPtsT1Ptr;
    boost::shared_ptr<VectorDouble> lagMultAtGaussPtsT2Ptr;

    boost::shared_ptr<MatrixDouble> lagMatMultAtGaussPtsPtr;
    boost::shared_ptr<MatrixDouble> contactLagrangeHdivDivergencePtr;
    boost::shared_ptr<MatrixDouble> mGradPtr;
    boost::shared_ptr<MatrixDouble> contactTractionPtr;
    boost::shared_ptr<VectorDouble> contactBaseNormal;

    boost::shared_ptr<VectorDouble> projNormalStressAtMaster;
    boost::shared_ptr<VectorDouble> projNormalStressAtSlave;
    boost::shared_ptr<VectorDouble> relErrorLagNormalStressAtMaster;
    boost::shared_ptr<VectorDouble> relErrorLagNormalStressAtSlave;

    boost::shared_ptr<VectorDouble> diffNormalLagSlave;
    boost::shared_ptr<VectorDouble> diffNormalLagMaster;
    boost::shared_ptr<VectorDouble> gapPtr;
    boost::shared_ptr<VectorDouble> lagGapProdPtr;
    boost::shared_ptr<VectorDouble> tildeCFunPtr;
    boost::shared_ptr<VectorDouble> tildeCFunNitschePtr;
    boost::shared_ptr<VectorDouble> lambdaGapDiffProductPtr;
    boost::shared_ptr<VectorDouble> nitscheGapDiffProductPtr;

    boost::shared_ptr<VectorDouble> normalVectorSlavePtr;
    boost::shared_ptr<VectorDouble> normalVectorMasterPtr;

    boost::shared_ptr<VectorDouble> tangentOneVectorSlavePtr;
    boost::shared_ptr<VectorDouble> tangentTwoVectorSlavePtr;

     boost::shared_ptr<MatrixDouble> hMat;
     boost::shared_ptr<MatrixDouble> FMat;
     boost::shared_ptr<MatrixDouble> HMat;
     boost::shared_ptr<VectorDouble> detHVec;
     boost::shared_ptr<MatrixDouble> invHMat;

     boost::shared_ptr<MatrixDouble> xAtPts;
     boost::shared_ptr<MatrixDouble> xInitAtPts;

     boost::shared_ptr<MatrixDouble> lagMultAtGaussPtsPt3D;

     std::map<int, NonlinearElasticElement::BlockData> setOfMasterFacesData;
     std::map<int, NonlinearElasticElement::BlockData> setOfSlaveFacesData;

     boost::shared_ptr<std::vector<DataForcesAndSourcesCore::EntData *>>
         masterRowData;

    boost::shared_ptr<std::vector<DataForcesAndSourcesCore::EntData *>>
         slaveRowData;

     DataForcesAndSourcesCore::EntData *l2RowData;

     DataForcesAndSourcesCore::EntData *l2ColData;

     DataForcesAndSourcesCore::EntData *faceRowData;

     double areaSlave;
     double areaMaster;

     CommonDataSimpleContact(MoFEM::Interface &m_field) : mField(m_field) {
       positionAtGaussPtsMasterPtr = boost::make_shared<MatrixDouble>();
       meshPositionAtGaussPtsMasterPtr = boost::make_shared<MatrixDouble>();
       meshPositionAtGaussPtsSlavePtr = boost::make_shared<MatrixDouble>();

       positionAtGaussPtsSlavePtr = boost::make_shared<MatrixDouble>();
       lagMultAtGaussPtsPtr = boost::make_shared<VectorDouble>();
       lagMultAtGaussPtsT1Ptr = boost::make_shared<VectorDouble>();
       lagMultAtGaussPtsT2Ptr = boost::make_shared<VectorDouble>();

       lagMatMultAtGaussPtsPtr = boost::make_shared<MatrixDouble>();
       contactLagrangeHdivDivergencePtr = boost::make_shared<MatrixDouble>();
       mGradPtr = boost::make_shared<MatrixDouble>();
       contactTractionPtr = boost::make_shared<MatrixDouble>();
       contactBaseNormal = boost::make_shared<VectorDouble>();

       projNormalStressAtMaster = boost::make_shared<VectorDouble>();
       projNormalStressAtSlave = boost::make_shared<VectorDouble>();
       relErrorLagNormalStressAtMaster = boost::make_shared<VectorDouble>();
       relErrorLagNormalStressAtSlave = boost::make_shared<VectorDouble>();

       diffNormalLagSlave = boost::make_shared<VectorDouble>();
       diffNormalLagMaster = boost::make_shared<VectorDouble>();
       gapPtr = boost::make_shared<VectorDouble>();
       normalSlaveLengthALEPtr = boost::make_shared<VectorDouble>();
       normalMasterLengthALEPtr = boost::make_shared<VectorDouble>();
       lagGapProdPtr = boost::make_shared<VectorDouble>();
       tildeCFunPtr = boost::make_shared<VectorDouble>();
       tildeCFunNitschePtr = boost::make_shared<VectorDouble>();
       lambdaGapDiffProductPtr = boost::make_shared<VectorDouble>();
       nitscheGapDiffProductPtr = boost::make_shared<VectorDouble>();
       normalVectorSlavePtr = boost::make_shared<VectorDouble>();
       normalVectorMasterPtr = boost::make_shared<VectorDouble>();

       tangentOneVectorSlavePtr = boost::make_shared<VectorDouble>();
       tangentTwoVectorSlavePtr = boost::make_shared<VectorDouble>();

       divPAtPts = boost::make_shared<MatrixDouble>();
       lagMultAtGaussPtsPt3D = boost::make_shared<MatrixDouble>();

       hMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
       FMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
       HMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
       detHVec = boost::shared_ptr<VectorDouble>(new VectorDouble());
       invHMat = boost::shared_ptr<MatrixDouble>(new MatrixDouble());

       xAtPts = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
       xInitAtPts = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
       masterRowData = boost::make_shared<
           std::vector<DataForcesAndSourcesCore::EntData *>>();

       slaveRowData = boost::make_shared<
           std::vector<DataForcesAndSourcesCore::EntData *>>();

       faceRowData = nullptr;
       l2RowData = nullptr;
       l2ColData = nullptr;
    }

    Range forcesOnlyOnEntitiesRow;
    Range forcesOnlyOnEntitiesCol;

  private:
    MoFEM::Interface &mField;
  };

  double rValue;
  double cnValue;
  double omegaValue;
  int thetaSValue;
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
    omegaValue = 0.5;
    thetaSValue = -1;
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
        const std::string &side_fe_name)
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


  struct OpPassHdivToMasterNormal
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    OpPassHdivToMasterNormal(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
                    commonDataSimpleContact(common_data_contact) {}

          MoFEMErrorCode doWork(int side, EntityType type,
                                DataForcesAndSourcesCore::EntData &data);
        private:
          boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  };

  struct OpPassHdivToSlaveNormal
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    double cN;
    OpPassHdivToSlaveNormal(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        double cn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), cN(cn_value) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  };

  struct OpLoopSlaveForSideFace
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<FaceElementForcesAndSourcesCoreOnVolumeSide> faceSideFe;
    std::string sideFeName;

    OpLoopSlaveForSideFace(
        const string field_name,
        boost::shared_ptr<FaceElementForcesAndSourcesCoreOnVolumeSide> side_fe,
        const std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          faceSideFe(side_fe), sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  //   struct OpCalMatForcesALEMaster
  //       : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

  //     boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  //     VectorDouble vec_f;
  //     Vec F;
  //     OpCalMatForcesALEMaster(
  //         const string field_name,
  //         boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
  //         Vec f_ = PETSC_NULL)
  //         : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
  //               field_name, UserDataOperator::OPCOL,
  //               ContactPrismElementForcesAndSourcesCore::UserDataOperator::
  //                   FACEMASTER),
  //           commonDataSimpleContact(common_data_contact), F(f_) {}

  //     MoFEMErrorCode doWork(int side, EntityType type,
  //                           DataForcesAndSourcesCore::EntData &data);
  //   };

  /**
   * @brief RHS-operator for the pressure element (material configuration)
   *
   * Integrates pressure in the material configuration.
   *
   */
  struct OpCalMatForcesALEMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    VectorInt rowIndices;

    int nbRows;           ///< number of dofs on rows
    int nbIntegrationPts; ///< number of integration points

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    VectorDouble vec_f;
    Vec F;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data);
    MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data);

    OpCalMatForcesALEMaster(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    // OpCalMatForcesALEMaster(
    //     const string material_field,
    //     boost::shared_ptr<DataAtIntegrationPts> data_at_pts,
    //     boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
    //     std::string &side_fe_name, Vec f, bCPressure &data,
    //     bool ho_geometry = false)
    //     : UserDataOperator(material_field, UserDataOperator::OPROW),
    //       dataAtPts(data_at_pts), sideFe(side_fe), sideFeName(side_fe_name),
    //       F(f), dAta(data), hoGeometry(ho_geometry){};
  };

//   struct OpCalMatForcesALESlave
//       : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

//     boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
//     VectorDouble vec_f;
//     Vec F;
//     OpCalMatForcesALESlave(
//         const string field_name,
//         boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
//         Vec f_ = PETSC_NULL)
//         : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
//               field_name, UserDataOperator::OPCOL,
//               ContactPrismElementForcesAndSourcesCore::UserDataOperator::
//                   FACESLAVE),
//           commonDataSimpleContact(common_data_contact), F(f_) {}

//     MoFEMErrorCode doWork(int side, EntityType type,
//                           DataForcesAndSourcesCore::EntData &data);
//   };

  struct OpCalMatForcesALESlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    VectorInt rowIndices;

    int nbRows;           ///< number of dofs on rows
    int nbIntegrationPts; ///< number of integration points

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    VectorDouble vec_f;
    Vec F;

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data);
    MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data);

    OpCalMatForcesALESlave(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), F(f_) {}
  };

  struct OpLoopMasterForSideLhs
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpLoopMasterForSideLhs(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERMASTER),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpGetRowColDataForSideFace
      : public FaceElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetRowColDataForSideFace(
        const string lag_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : FaceElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator(
              lag_field_name, UserDataOperator::OPROWCOL),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpLoopMasterForSideLhsTest
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpLoopMasterForSideLhsTest(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERMASTER),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpLoopSlaveForSideLhsTest
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpLoopSlaveForSideLhsTest(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpLoopSlaveMasterForSideLhsTest
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpLoopSlaveMasterForSideLhsTest(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVEMASTER),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpLoopMasterSlaveForSideLhsTest
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpLoopMasterSlaveForSideLhsTest(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpLoopSlaveForSideLhs
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpLoopSlaveForSideLhs(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name) {}

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
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
        const std::string &side_fe_name, Mat aij = PETSC_NULL)
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
  struct OpGetMeshPositionAtGaussPtsMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetMeshPositionAtGaussPtsMaster(
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

  struct OpGetMeshPositionAtGaussPtsSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetMeshPositionAtGaussPtsSlave(
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

  // Calculate displacements or spatial positions at Gauss points  (Note
  // OPCOL
  // here, which is dips/sp-pos here)
  struct OpCalProjStressesAtGaussPtsMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalProjStressesAtGaussPtsMaster(
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

  struct OpCalNormalLagrangeMasterAndSlaveDifference
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalNormalLagrangeMasterAndSlaveDifference(
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

struct OpCalRelativeErrorNormalLagrangeMasterAndSlaveDifference
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalRelativeErrorNormalLagrangeMasterAndSlaveDifference(
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


  struct OpCalProjStressesAtGaussPtsSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpCalProjStressesAtGaussPtsSlave(
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

  struct OpGetLagMulAtGaussPtsSlaveHdiv3D
      : public VolumeElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetLagMulAtGaussPtsSlaveHdiv3D(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : VolumeElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPROW),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpGetLagMulAtGaussPtsSlaveL2
      : public FaceElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpGetLagMulAtGaussPtsSlaveL2(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : FaceElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPROW),
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

  struct OpCalPenaltyRhsMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    double cN;
    OpCalPenaltyRhsMaster(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        double &cn_value, Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          commonDataSimpleContact(common_data_contact), cN(cn_value), F(f_) {}

    VectorDouble vec_f;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalNitscheCStressRhsMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    double cN;
    double omegaVal;
    OpCalNitscheCStressRhsMaster(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        double &omega_val, Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTER),
          commonDataSimpleContact(common_data_contact), omegaVal(omega_val),
          F(f_) {}

    VectorDouble vec_f;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalNitscheCStressRhsSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    double cN;
    double omegaVal;
    OpCalNitscheCStressRhsSlave(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        double &omega_val, Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), omegaVal(omega_val),
          F(f_) {}

    VectorDouble vec_f;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalPenaltyRhsSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    double cN;
    OpCalPenaltyRhsSlave(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        double &cn_value, Vec f_ = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), cN(cn_value), F(f_) {}

    VectorDouble vec_f;
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

  struct OpCalTildeCFunNitscheSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    double r;  //@todo: ign: to become input parameter
    double cN; //@todo: ign: to become input parameter
    double omegaVal;

    OpCalTildeCFunNitscheSlave(
        const string lagrang_field_name, // ign: does it matter?
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        double &r_value, double &cn_value, double &omega_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), r(r_value),
          cN(cn_value), omegaVal(omega_value) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpCalIntTildeCFunSlaveL2
      : public FaceElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalIntTildeCFunSlaveL2(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : FaceElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPCOL),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    VectorDouble vecR;
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

  struct OpCalIntTildeCFunSlaveHdiv3D
      : public VolumeElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalIntTildeCFunSlaveHdiv3D(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : VolumeElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPCOL),
          commonDataSimpleContact(common_data_contact), F(f_) {}

    VectorDouble vecR;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpContactSimplePenaltyMasterMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become new input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactSimplePenaltyMasterMaster(
        const string field_name, const string field_name_2, double &cn_value,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, field_name_2, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERMASTER),
          cN(cn_value), commonDataSimpleContact(common_data_contact), Aij(aij) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactSimplePenaltyMasterSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become new input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactSimplePenaltyMasterSlave(
        const string field_name, const string field_name_2, double &cn_value,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, field_name_2, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          cN(cn_value), commonDataSimpleContact(common_data_contact), Aij(aij) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactSimplePenaltySlaveSlave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become new input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactSimplePenaltySlaveSlave(
        const string field_name, const string field_name_2, double &cn_value,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, field_name_2, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          cN(cn_value), commonDataSimpleContact(common_data_contact), Aij(aij) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpContactSimplePenaltySlaveMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    double cN; //@todo: ign: to become new input parameter
    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactSimplePenaltySlaveMaster(
        const string field_name, const string field_name_2, double &cn_value,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, field_name_2, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVEMASTER),
          cN(cn_value), commonDataSimpleContact(common_data_contact), Aij(aij) {
      sYmm = false; // This will make sure to loop over all intities (e.g.
                    // for order=2 it will make doWork to loop 16 time)
    }
    MatrixDouble NN;

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
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

  struct OpContactConstraintMatrixMasterSlaveH1L2
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixMasterSlaveH1L2(
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

  struct OpContactConstraintMatrixSlaveSlaveH1L2
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpContactConstraintMatrixSlaveSlaveH1L2(
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

  struct OpDerivativeBarTildeCFunOLambdaSlaveSlaveL2L2
      : public FaceElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator {

    Mat Aij;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    OpDerivativeBarTildeCFunOLambdaSlaveSlaveL2L2(
        const string lagrang_field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Mat aij = PETSC_NULL)
        : FaceElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator(
              lagrang_field_name, UserDataOperator::OPROWCOL),
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

  struct OpCalculateHdivDivergenceResidual
      : public VolumeElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalculateHdivDivergenceResidual(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          commonDataSimpleContact(common_data_contact), F(f_) {
      sYmm = false;
    }

    VectorDouble vec_f;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data); 

    
  };

  struct OpConstrainDomainLhs_dU
      : public VolumeElementForcesAndSourcesCore::UserDataOperator {
    OpConstrainDomainLhs_dU(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact)
        : VolumeElementForcesAndSourcesCore::UserDataOperator(
              row_field_name, col_field_name, UserDataOperator::OPROWCOL),
          commonDataSimpleContact(common_data_contact) {
      sYmm = false;
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    MatrixDouble locMat;
    MatrixDouble transLocMat;
    };

  struct OpConstrainDomainRhs
      : public VolumeElementForcesAndSourcesCore::UserDataOperator {
    OpConstrainDomainRhs(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          commonDataSimpleContact(common_data_contact) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData  &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  };

  struct OpInternalDomainContactRhs
      : public VolumeElementForcesAndSourcesCore::UserDataOperator {
    OpInternalDomainContactRhs(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          commonDataSimpleContact(common_data_contact) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  };

  struct OpConstrainBoundaryTraction
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    OpConstrainBoundaryTraction(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  };

  struct OpConstrainBoundaryRhs
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    double cN;
    OpConstrainBoundaryRhs(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        double &cn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name,
              UserDataOperator::OPCOL, ContactPrismElementForcesAndSourcesCore::
                  UserDataOperator::FACESLAVE),
          commonDataSimpleContact(common_data_contact), cN(cn_value) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    
  };

  struct OpConstrainBoundaryRhsMaster
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    double cN;
    OpConstrainBoundaryRhsMaster(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        double &cn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), cN(cn_value) {}
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
  };

  struct OpConstrainBoundaryLhs_dTraction : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    OpConstrainBoundaryLhs_dTraction(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        double &dn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              row_field_name, col_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact), cN(dn_value) {
      sYmm = false;
     }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                             EntityType col_type, DataForcesAndSourcesCore::EntData &row_data,
                             DataForcesAndSourcesCore::EntData &col_data);
    private :
      boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
      MatrixDouble locMat;
      double cN;
  };

  struct OpConstrainBoundaryLhs_dxdTraction
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    OpConstrainBoundaryLhs_dxdTraction(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        double &dn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              row_field_name, col_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact), cN(dn_value) {
      sYmm = false;
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    MatrixDouble locMat;
    double cN;
  };

  struct OpConstrainBoundaryLhs_dU_Slave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    OpConstrainBoundaryLhs_dU_Slave(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        double &dn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              row_field_name, col_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact), cN(dn_value) {
      sYmm = false;
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    MatrixDouble locMat;
    double cN;
  };

  struct OpConstrainBoundaryLhs_dU_Master
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    OpConstrainBoundaryLhs_dU_Master(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        double &dn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              row_field_name, col_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVEMASTER),
          commonDataSimpleContact(common_data_contact), cN(dn_value) {
      sYmm = false;
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    MatrixDouble locMat;
    MatrixDouble transLocMat;
    double cN;
  };

  struct OpConstrainBoundaryLhs_dU_dlambda_Master
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    OpConstrainBoundaryLhs_dU_dlambda_Master(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        double &dn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              row_field_name, col_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact), cN(dn_value) {
      sYmm = false;
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    MatrixDouble locMat;
    double cN;
  };

  struct OpConstrainBoundaryLhs_dU_dlambda_Slave
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {
    OpConstrainBoundaryLhs_dU_dlambda_Slave(
        const std::string row_field_name, const std::string col_field_name,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        double &dn_value)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              row_field_name, col_field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact), cN(dn_value) {
      sYmm = false;
    }
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

  private:
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    MatrixDouble locMat;
    double cN;
  };

  struct OpCalculateHdivDivergenceResidualOnDispl
      : public VolumeElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Vec F;
    OpCalculateHdivDivergenceResidualOnDispl(
        const string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        Vec f_ = PETSC_NULL)
        : VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL),
          commonDataSimpleContact(common_data_contact), F(f_) {
      sYmm = false;
    }

    VectorDouble vec_f;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);   
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

  // Material Forces begin
  struct OpContactMaterialVolOnMasterSideLhs
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator {

    MatrixDouble NN;

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Mat Aij;
    
    bool hoGeometry;

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
    virtual MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }
    MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialVolOnMasterSideLhs(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact, Mat aij,
        bool ho_geometry = false)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator(field_name_1, field_name_2,
                                    UserDataOperator::OPROWCOL),
          commonDataSimpleContact(common_data_contact), Aij(aij), hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };

  struct OpContactMaterialVolOnSlaveSideLhs
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
            UserDataOperator {

    MatrixDouble NN;

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    Mat Aij;

    bool hoGeometry;

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
    virtual MoFEMErrorCode
    iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
              DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }
    MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data,
                            DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialVolOnSlaveSideLhs(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact, Mat aij,
        bool ho_geometry = false)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
              UserDataOperator(field_name_1, field_name_2,
                               UserDataOperator::OPROWCOL),
          commonDataSimpleContact(common_data_contact), Aij(aij),
          hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };

  struct OpContactMaterialVolOnMasterSideLhs_dX_dx
      : public OpContactMaterialVolOnMasterSideLhs {

    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialVolOnMasterSideLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact, Mat aij,
        bool ho_geometry = false)
        : OpContactMaterialVolOnMasterSideLhs(field_name_1, field_name_2,
                                                common_data_contact, aij, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpContactMaterialVolOnMasterSideLhs_dX_dX
      : public OpContactMaterialVolOnMasterSideLhs {

    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialVolOnMasterSideLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact, Mat aij,
        bool ho_geometry = false)
        : OpContactMaterialVolOnMasterSideLhs(field_name_1, field_name_2,
                                                common_data_contact, aij, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpContactMaterialVolOnSlaveSideLhs_dX_dx
      : public OpContactMaterialVolOnSlaveSideLhs {

    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                             DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialVolOnSlaveSideLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact, Mat aij,
        bool ho_geometry = false)
        : OpContactMaterialVolOnSlaveSideLhs(field_name_1, field_name_2,
                                              common_data_contact, aij,
                                              ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpContactMaterialVolOnSlaveSideLhs_dX_dX
      : public OpContactMaterialVolOnSlaveSideLhs {

    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                             DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialVolOnSlaveSideLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact, Mat aij,
        bool ho_geometry = false)
        : OpContactMaterialVolOnSlaveSideLhs(field_name_1, field_name_2,
                                              common_data_contact, aij,
                                              ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  //   struct OpContactMaterialVolOnMasterSideLhs_dX_dLagmult
  //       : public OpContactMaterialVolOnMasterSideLhs {

  //     MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
  //                              DataForcesAndSourcesCore::EntData &col_data);

  //     OpContactMaterialVolOnMasterSideLhs_dX_dLagmult(
  //         const string field_name_1, const string lag_field,
  //         boost::shared_ptr<CommonDataSimpleContact> common_data_contact, Mat
  //         aij, bool ho_geometry = false) :
  //         OpContactMaterialVolOnMasterSideLhs(field_name_1, lag_field,
  //                                         common_data_contact, aij,
  //                                         ho_geometry) {
  //       sYmm = false; // This will make sure to loop over all entities
  //     };
  //   };

  struct OpContactMaterialMasterLhs
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;
    Mat Aij;
    bool hoGeometry;

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

    virtual MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialMasterLhs(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name_1, field_name_2, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERMASTER),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name), Aij(aij), hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };

  struct OpContactMaterialSlaveLhs
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;
    Mat Aij;
    bool hoGeometry;

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

    virtual MoFEMErrorCode
    iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
              DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data,
                            DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialSlaveLhs(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name_1, field_name_2, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::FACESLAVESLAVE),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name), Aij(aij), hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };

  struct OpContactMaterialMasterSlaveLhs
      : public ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> sideFe;
    std::string sideFeName;
    Mat Aij;
    bool hoGeometry;

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

    virtual MoFEMErrorCode
    iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
              DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;
      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode aSsemble(DataForcesAndSourcesCore::EntData &row_data,
                            DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialMasterSlaveLhs(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name_1, field_name_2, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact), sideFe(side_fe),
          sideFeName(side_fe_name), Aij(aij), hoGeometry(ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    }
  };
  

  struct OpContactMaterialMasterLhs_dX_dX
      : public OpContactMaterialMasterLhs {

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data, DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialMasterLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : OpContactMaterialMasterLhs(field_name_1, field_name_2,
                                       common_data_contact, side_fe,
                                       side_fe_name, aij, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpContactMaterialSlaveLhs_dX_dX : public OpContactMaterialSlaveLhs {

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                             DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialSlaveLhs_dX_dX(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : OpContactMaterialSlaveLhs(field_name_1, field_name_2,
                                     common_data_contact, side_fe, side_fe_name,
                                     aij, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpContactMaterialMasterSlaveLhs_dX_dLagmult : public OpContactMaterialMasterSlaveLhs {

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                             DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialMasterSlaveLhs_dX_dLagmult(
        const string field_name_1, const string lag_field,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : OpContactMaterialMasterSlaveLhs(field_name_1, lag_field,
                                     common_data_contact, side_fe, side_fe_name,
                                     aij, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpContactMaterialSlaveSlaveLhs_dX_dLagmult
      : public OpContactMaterialSlaveLhs {

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                             DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialSlaveSlaveLhs_dX_dLagmult(
        const string field_name_1, const string lag_field,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : OpContactMaterialSlaveLhs(field_name_1, lag_field,
                                          common_data_contact, side_fe,
                                          side_fe_name, aij, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpContactMaterialMasterSlaveLhs_dX_dX
      : public OpContactMaterialMasterSlaveLhs {

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode iNtegrate(DataForcesAndSourcesCore::EntData &row_data,
                             DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialMasterSlaveLhs_dX_dX(
        const string field_name_1, const string lag_field,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : OpContactMaterialMasterSlaveLhs(field_name_1, lag_field,
                                          common_data_contact, side_fe,
                                          side_fe_name, aij, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpContactMaterialMasterLhs_dX_dx
      : public OpContactMaterialMasterLhs {

    /*
     * Triggers loop over operators from the side volume
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialMasterLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide> side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : OpContactMaterialMasterLhs(field_name_1, field_name_2,
                                       common_data_contact, side_fe,
                                       side_fe_name, aij, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };

  struct OpContactMaterialSlaveLhs_dX_dx : public OpContactMaterialSlaveLhs {

    /*
     * Triggers loop over operators from the side volume
     *
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);

    OpContactMaterialSlaveLhs_dX_dx(
        const string field_name_1, const string field_name_2,
        boost::shared_ptr<CommonDataSimpleContact> common_data_contact,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            side_fe,
        const std::string &side_fe_name, Mat aij, bool ho_geometry = false)
        : OpContactMaterialSlaveLhs(field_name_1, field_name_2,
                                     common_data_contact, side_fe, side_fe_name,
                                     aij, ho_geometry) {
      sYmm = false; // This will make sure to loop over all entities
    };
  };
  struct OpStressDerivativeGapMaster_dx
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
            UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    NonlinearElasticElement::BlockData &dAta;
    NonlinearElasticElement::CommonData &commonData;
    int tAg;
    bool aLe;

    ublas::vector<int> rowIndices;
    ublas::vector<int> colIndices;
    VectorDouble vec_f;
    Vec F;
    double cN;
    double omegaVal;
    int thetaSVal;
    OpStressDerivativeGapMaster_dx(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        NonlinearElasticElement::BlockData &data,
        NonlinearElasticElement::CommonData &common_data, double &cn_value,
        double &omega_value, int &theta_s_value, Vec f_ = PETSC_NULL)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
              UserDataOperator(field_name, UserDataOperator::OPROW),
          commonDataSimpleContact(common_data_contact), dAta(data),
          commonData(common_data), aLe(false), cN(cn_value),
          omegaVal(omega_value), thetaSVal(theta_s_value), F(f_) {}

    MatrixDouble jac;

    /**
      \brief Directive of Piola Kirchhoff stress over spatial DOFs

      This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
      \f[
      \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
      F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
      \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
      function

    */
    virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    MoFEMErrorCode doWork(int row_side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpLoopForRowDataOnMaster
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
            UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    OpLoopForRowDataOnMaster(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
              UserDataOperator(field_name, UserDataOperator::OPROW),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int row_side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpLoopForRowDataOnSlave
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
            UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    OpLoopForRowDataOnSlave(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
              UserDataOperator(field_name, UserDataOperator::OPROW),
          commonDataSimpleContact(common_data_contact) {}

    MoFEMErrorCode doWork(int row_side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  //   struct OpLoopForRowDataOnMaster
  //       : public MoFEM::ContactPrismElementForcesAndSourcesCore::
  //             UserDataOperator {

  //     boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

  //     OpLoopForRowDataOnMaster(
  //         const std::string field_name,
  //         boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
  //         : MoFEM::ContactPrismElementForcesAndSourcesCore::UserDataOperator(
  //               field_name, UserDataOperator::OPROW,
  //               ContactPrismElementForcesAndSourcesCore::UserDataOperator::
  //                   FACEMASTER),
  //           commonDataSimpleContact(common_data_contact) {}

  //     MoFEMErrorCode doWork(int row_side, EntityType type,
  //                           DataForcesAndSourcesCore::EntData &data);
  //   };

  struct OpStressDerivativeGapSlave_dx
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
            UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    NonlinearElasticElement::BlockData &dAta;
    NonlinearElasticElement::CommonData &commonData;
    int tAg;
    bool aLe;

    ublas::vector<int> rowIndices;
    ublas::vector<int> colIndices;
    VectorDouble vec_f;
    Vec F;

    double cN;
    double omegaVal;
    int thetaSVal;
    OpStressDerivativeGapSlave_dx(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        NonlinearElasticElement::BlockData &data,
        NonlinearElasticElement::CommonData &common_data, double &cn_value,
        double &omega_value, int &theta_s_value, Vec f_ = PETSC_NULL)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
              UserDataOperator(field_name, UserDataOperator::OPROW),
          commonDataSimpleContact(common_data_contact), dAta(data),
          commonData(common_data), aLe(false), cN(cn_value),
          omegaVal(omega_value), thetaSVal(theta_s_value), F(f_) {}

    MatrixDouble jac;

    /**
      \brief Directive of Piola Kirchhoff stress over spatial DOFs

      This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
      \f[
      \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
      F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
      \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
      function

    */
    virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    MoFEMErrorCode doWork(int row_side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  struct OpStressDerivativeGapMasterMaster_dx
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
            UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    NonlinearElasticElement::BlockData &dAta;
    NonlinearElasticElement::CommonData &commonData;
    int tAg;
    bool aLe;

    ublas::vector<int> rowIndices;
    ublas::vector<int> colIndices;
    MatrixDouble NN;

    Mat Aij;

    double cN;
    double omegaVal;
    int thetaSVal;

    OpStressDerivativeGapMasterMaster_dx(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        NonlinearElasticElement::BlockData &data,
        NonlinearElasticElement::CommonData &common_data, double &c_n,
        double &omega_val, int &theta_s_val, Mat aij = PETSC_NULL)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
              UserDataOperator(field_name, UserDataOperator::OPROWCOL),
          commonDataSimpleContact(common_data_contact), dAta(data),
          commonData(common_data), aLe(false), cN(c_n), omegaVal(omega_val),
          thetaSVal(theta_s_val), Aij(aij) {
      sYmm = false;
    }

    MatrixDouble jac;
    MatrixDouble jac_row;

    /**
      \brief Directive of Piola Kirchhoff stress over spatial DOFs

      This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
      \f[
      \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
      F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
      \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
      function

    */
    virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    virtual MoFEMErrorCode getJacRow(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpStressDerivativeGapMasterSlave_dx
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
            UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    NonlinearElasticElement::BlockData &dAta;
    NonlinearElasticElement::CommonData &commonData;
    NonlinearElasticElement::CommonData &commonDataMaster;
    int tAg;
    bool aLe;

    ublas::vector<int> rowIndices;
    ublas::vector<int> colIndices;
    MatrixDouble NN, NNT;

    double cN;
    double omegaVal;
    int thetaSVal;

    Mat Aij;
    OpStressDerivativeGapMasterSlave_dx(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        NonlinearElasticElement::BlockData &data,
        NonlinearElasticElement::CommonData &common_data,
        NonlinearElasticElement::CommonData &common_data_master, double &c_n,
        double &omega_val, int &theta_s_val, Mat aij = PETSC_NULL)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
              UserDataOperator(field_name, UserDataOperator::OPCOL),
          commonDataSimpleContact(common_data_contact), dAta(data),
          commonData(common_data), commonDataMaster(common_data_master),
          cN(c_n), omegaVal(omega_val), thetaSVal(theta_s_val), aLe(false),
          Aij(aij) {}

    MatrixDouble jac;
    MatrixDouble jac_row;

    /**
      \brief Directive of Piola Kirchhoff stress over spatial DOFs

      This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
      \f[
      \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
      F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
      \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
      function

    */
    virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    virtual MoFEMErrorCode getJacRow(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    // MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
    //                       EntityType col_type,
    //                       DataForcesAndSourcesCore::EntData &row_data,
    //                       DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode doWork(int col_side,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpStressDerivativeGapMasterSlave_dxCorrect
      : public MoFEM::ContactPrismElementForcesAndSourcesCore::
            UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    NonlinearElasticElement::BlockData &dAta;
    NonlinearElasticElement::CommonData &commonData;
    int tAg;
    bool aLe;

    ublas::vector<int> rowIndices;
    ublas::vector<int> colIndices;
    MatrixDouble NN;

    Mat Aij;
    OpStressDerivativeGapMasterSlave_dxCorrect(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        NonlinearElasticElement::BlockData &data,
        NonlinearElasticElement::CommonData &common_data,
        Mat aij = PETSC_NULL)
        : MoFEM::ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROWCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACEMASTERSLAVE),
          commonDataSimpleContact(common_data_contact), dAta(data),
          commonData(common_data),
          aLe(false), Aij(aij) {}

    MatrixDouble jac;
    MatrixDouble jac_row;

    /**
      \brief Directive of Piola Kirchhoff stress over spatial DOFs

      This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
      \f[
      \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
      F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
      \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
      function

    */
    virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    virtual MoFEMErrorCode getJacRow(DataForcesAndSourcesCore::EntData &data,
                                     int gg);

    // MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
    //                       EntityType col_type,
    //                       DataForcesAndSourcesCore::EntData &row_data,
    //                       DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpStressDerivativeGapSlaveSlave_dx
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
            UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    NonlinearElasticElement::BlockData &dAta;
    NonlinearElasticElement::CommonData &commonData;
    int tAg;
    bool aLe;

    ublas::vector<int> rowIndices;
    ublas::vector<int> colIndices;
    MatrixDouble NN;

    double cN;
    double omegaVal;
    int thetaSVal;

    Mat Aij;
    OpStressDerivativeGapSlaveSlave_dx(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        NonlinearElasticElement::BlockData &data,
        NonlinearElasticElement::CommonData &common_data, double &c_n,
        double &omega_val, int &theta_s_val, Mat aij = PETSC_NULL)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
              UserDataOperator(field_name, UserDataOperator::OPROWCOL),
          commonDataSimpleContact(common_data_contact), dAta(data),
          commonData(common_data), aLe(false),
          cN(c_n), omegaVal(omega_val), thetaSVal(theta_s_val),
          Aij(aij) {
      sYmm = false;
          }

    MatrixDouble jac;
    MatrixDouble jac_row;

    /**
      \brief Directive of Piola Kirchhoff stress over spatial DOFs

      This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
      \f[
      \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
      F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
      \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
      function

    */
    virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    virtual MoFEMErrorCode getJacRow(DataForcesAndSourcesCore::EntData &data,
                                     int gg);

    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  struct OpLagMultOnVertex
      : public MoFEM::ContactPrismElementForcesAndSourcesCore::
            UserDataOperator {

    moab::Interface &moabOut;
    MoFEM::Interface &mField;

    OpLagMultOnVertex(MoFEM::Interface &m_field, string field_name,
                      moab::Interface &moab_out)
        : MoFEM::ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          mField(m_field), moabOut(moab_out) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data);
  };

  
//   struct OpStressDerivativeGapSlaveMaster_dx
//       : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
//             UserDataOperator {

//     boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

//     NonlinearElasticElement::BlockData &dAta;
//     NonlinearElasticElement::CommonData &commonData;
//     int tAg;
//     bool aLe;

//     ublas::vector<int> rowIndices;
//     ublas::vector<int> colIndices;
//     MatrixDouble NN;

//     Mat Aij;
//     OpStressDerivativeGapSlaveMaster_dx(
//         const std::string field_name,
//         boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
//         NonlinearElasticElement::BlockData &data,
//         NonlinearElasticElement::CommonData &common_data, Mat aij = PETSC_NULL)
//         : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::UserDataOperator(
//               field_name, UserDataOperator::OPROWCOL),
//           commonDataSimpleContact(common_data_contact), dAta(data),
//           commonData(common_data), aLe(false), Aij(aij) {}

//     MatrixDouble jac;

//     /**
//       \brief Directive of Piola Kirchhoff stress over spatial DOFs

//       This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
//       \f[
//       \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
//       F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
//       \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
//       function

//     */
//     virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
//                                   int gg);

//     MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
//                           EntityType col_type,
//                           DataForcesAndSourcesCore::EntData &row_data,
//                           DataForcesAndSourcesCore::EntData &col_data);
//   };

//   struct OpStressDerivativeGapSlaveMaster_dx
//       : public MoFEM::ContactPrismElementForcesAndSourcesCore::
//             UserDataOperator {

//     boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

//     NonlinearElasticElement::BlockData &dAta;
//     NonlinearElasticElement::CommonData &commonData;
//     NonlinearElasticElement::CommonData &commonDataMaster;
//     int tAg;
//     bool aLe;

//     ublas::vector<int> rowIndices;
//     ublas::vector<int> colIndices;
//     MatrixDouble NN;

//     Mat Aij;
//     OpStressDerivativeGapSlaveMaster_dx(
//         const std::string field_name,
//         boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
//         NonlinearElasticElement::BlockData &data,
//         NonlinearElasticElement::CommonData &common_data,
//         NonlinearElasticElement::CommonData &common_data_master,
//         Mat aij = PETSC_NULL)
//         : MoFEM::ContactPrismElementForcesAndSourcesCore::UserDataOperator(
//               field_name, UserDataOperator::OPCOL,
//               ContactPrismElementForcesAndSourcesCore::UserDataOperator::
//                   FACEMASTER),
//           commonDataSimpleContact(common_data_contact), dAta(data),
//           commonData(common_data), commonDataMaster(common_data_master),
//           aLe(false), Aij(aij) {}

//     MatrixDouble jac;
//     MatrixDouble jac_row;

//     /**
//       \brief Directive of Piola Kirchhoff stress over spatial DOFs

//       This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
//       \f[
//       \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
//       F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
//       \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
//       function

//     */
//     virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
//                                   int gg);

//     virtual MoFEMErrorCode getJacRow(DataForcesAndSourcesCore::EntData &data,
//                                      int gg);

//     // MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
//     //                       EntityType col_type,
//     //                       DataForcesAndSourcesCore::EntData &row_data,
//     //                       DataForcesAndSourcesCore::EntData &col_data);

//     MoFEMErrorCode doWork(int col_side, EntityType col_type,
//                           DataForcesAndSourcesCore::EntData &col_data);
//   };

  struct OpStressDerivativeGapSlaveMaster_dx
      : public MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
            UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    NonlinearElasticElement::BlockData &dAta;
    NonlinearElasticElement::CommonData &commonData;
    NonlinearElasticElement::CommonData &commonDataMaster;
    int tAg;
    bool aLe;

    ublas::vector<int> rowIndices;
    ublas::vector<int> colIndices;
    MatrixDouble NN, NNT;

    double cN;
    double omegaVal;
    int thetaSVal;

    Mat Aij;
    OpStressDerivativeGapSlaveMaster_dx(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        NonlinearElasticElement::BlockData &data,
        NonlinearElasticElement::CommonData &common_data,
        NonlinearElasticElement::CommonData &common_data_master, double &c_n,
        double &omega_val, int &theta_s_val, Mat aij = PETSC_NULL)
        : MoFEM::VolumeElementForcesAndSourcesCoreOnVolumeSide::
              UserDataOperator(field_name, UserDataOperator::OPCOL),
          commonDataSimpleContact(common_data_contact), dAta(data),
          commonData(common_data), commonDataMaster(common_data_master),
          aLe(false), cN(c_n), omegaVal(omega_val), thetaSVal(theta_s_val),
          Aij(aij) {}

    MatrixDouble jac;
    MatrixDouble jac_row;

    /**
      \brief Directive of Piola Kirchhoff stress over spatial DOFs

      This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
      \f[
      \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
      F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
      \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
      function

    */
    virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    virtual MoFEMErrorCode getJacRow(DataForcesAndSourcesCore::EntData &data,
                                     int gg);

    // MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
    //                       EntityType col_type,
    //                       DataForcesAndSourcesCore::EntData &row_data,
    //                       DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode doWork(int col_side, EntityType col_type,
                          DataForcesAndSourcesCore::EntData &col_data);
  };


  struct OpStressDerivativeGapSlaveMaster_dx2 : 
  public MoFEM::ContactPrismElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

    NonlinearElasticElement::BlockData &dAta;
    NonlinearElasticElement::CommonData &commonData;
    NonlinearElasticElement::CommonData &commonDataSlave;
    int tAg;
    bool aLe;

    ublas::vector<int> rowIndices;
    ublas::vector<int> colIndices;
    MatrixDouble NN, NNT;

    Mat Aij;
    OpStressDerivativeGapSlaveMaster_dx2(
        const std::string field_name,
        boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
        NonlinearElasticElement::BlockData &data,
        NonlinearElasticElement::CommonData &common_data,
        NonlinearElasticElement::CommonData &common_data_slave,
        Mat aij = PETSC_NULL)
        : MoFEM::ContactPrismElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPCOL,
              ContactPrismElementForcesAndSourcesCore::UserDataOperator::
                  FACESLAVE),
          commonDataSimpleContact(common_data_contact), dAta(data),
          commonData(common_data), commonDataSlave(common_data_slave),
          aLe(false), Aij(aij) {}

    MatrixDouble jac;
    MatrixDouble jac_row;

    /**
      \brief Directive of Piola Kirchhoff stress over spatial DOFs

      This project derivative \f$\frac{\partial P}{\partial F}\f$, that is
      \f[
      \frac{\partial P}{\partial x_\textrm{DOF}} =  \frac{\partial P}{\partial
      F}\frac{\partial F}{\partial x_\textrm{DOF}}, \f] where second therm
      \f$\frac{\partial F}{\partial x_\textrm{DOF}}\f$ is derivative of shape
      function

    */
    virtual MoFEMErrorCode getJac(DataForcesAndSourcesCore::EntData &data,
                                  int gg);

    virtual MoFEMErrorCode getJacRow(DataForcesAndSourcesCore::EntData &data,
                                     int gg);

    // MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
    //                       EntityType col_type,
    //                       DataForcesAndSourcesCore::EntData &row_data,
    //                       DataForcesAndSourcesCore::EntData &col_data);

    MoFEMErrorCode doWork(int col_side, EntityType col_type,
                          DataForcesAndSourcesCore::EntData &col_data);
  };

  // setup operators for calculation of active set
  MoFEMErrorCode setContactPenaltyRhsOperators(boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
  boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact, 
                                                 string field_name,
                                                 Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

//  boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
//       boost::make_shared<CommonDataSimpleContact>(mField);

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

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalPenaltyRhsMaster(
          field_name, common_data_simple_contact, cnValue, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalPenaltyRhsSlave(
          field_name, common_data_simple_contact, cnValue, f_));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactNitschePenaltyRhsOperators(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, const string mesh_node_position, string side_fe_name,
      NonlinearElasticElement::BlockData &master_block,
      NonlinearElasticElement::BlockData &slave_block, Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMaster(field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));

    //   fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalPenaltyRhsMaster(
    //       field_name, common_data_simple_contact, cnValue, f_));

    //   fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalPenaltyRhsSlave(
    //       field_name, common_data_simple_contact, cnValue, f_));

      common_data_simple_contact->tAg = 2;

      common_data_simple_contact->elasticityCommonDataMaster.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonDataMaster.meshPositions =
          mesh_node_position;

      common_data_simple_contact->elasticityCommonData.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonData.meshPositions =
          mesh_node_position;

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatMasterStressSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatMasterStressSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->dataAtGaussPtsMaster[field_name],
              common_data_simple_contact->gradAtGaussPtsMaster[field_name]));

      cerr << "Material data Master "
           << "E " << master_block.E << " Poisson " << master_block.PoissonRatio
           << " Element name " << side_fe_name << side_fe_name
           << " and number of tets " << master_block.tEts.size() <<"\n";

      feMatMasterStressSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name, common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterStressSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterStressSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, master_block,
              common_data_simple_contact->elasticityCommonDataMaster,
              common_data_simple_contact->tAg, false, false, false));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpLoopMasterForSide(field_name, feMatMasterStressSideRhs,
                                  side_fe_name));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalProjStressesAtGaussPtsMaster(field_name,
                                                common_data_simple_contact));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatMasterSideStressDerRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatMasterSideStressDerRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->dataAtGaussPtsMaster[field_name],
              common_data_simple_contact->gradAtGaussPtsMaster[field_name]));

      // again
      feMatMasterSideStressDerRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name, common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterSideStressDerRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterSideStressDerRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, master_block,
              common_data_simple_contact->elasticityCommonDataMaster,
              common_data_simple_contact->tAg, true, false, false));

      feMatMasterSideStressDerRhs->getOpPtrVector().push_back(
          new OpStressDerivativeGapMaster_dx(
              field_name, common_data_simple_contact, master_block,
              common_data_simple_contact->elasticityCommonDataMaster, cnValue,
              omegaValue, thetaSValue));

      //SLAVE
      // again
      cerr << "Material 1111 dasdasdasdaasdta Slave "
           << "E " << slave_block.E << " Poisson " << slave_block.PoissonRatio
           << " Element name " << side_fe_name << " and number of tets "
           <<slave_block.tEts.size()  <<"\n";

      cerr << " OMEGA   " << omegaValue
           << "\n";

cerr << " Theta   " << thetaSValue
           << "\n";

      //   cerr << "Material 1111 dasdasdasdaasdta Slave "
      //        << "E " << slave_block.E << " Poisson " <<
      //        slave_block.PoissonRatio
      //        << " Element name " << side_fe_name << "\n";

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveSideStressRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);
//
    //   feMatMasterStressSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetDataAtGaussPts(
    //           field_name,
    //           common_data_simple_contact->dataAtGaussPtsMaster[field_name],
    //           common_data_simple_contact->gradAtGaussPtsMaster[field_name]));

    //   feMatMasterStressSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
    //           field_name,
    //           common_data_simple_contact->elasticityCommonDataMaster));

    //   feMatMasterStressSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
    //           mesh_node_position,
    //           common_data_simple_contact->elasticityCommonDataMaster));

    //   feMatMasterStressSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
    //           field_name, master_block,
    //           common_data_simple_contact->elasticityCommonDataMaster,
    //           common_data_simple_contact->tAg, false, false, false));
//

      feMatSlaveSideStressRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->dataAtGaussPts[field_name],
              common_data_simple_contact->gradAtGaussPts[field_name]));

        feMatSlaveSideStressRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
                field_name,
                common_data_simple_contact->elasticityCommonData));

        feMatSlaveSideStressRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
                mesh_node_position,
                common_data_simple_contact->elasticityCommonData));

        feMatSlaveSideStressRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
                field_name, slave_block,
                common_data_simple_contact->elasticityCommonData,
                common_data_simple_contact->tAg, false, false, false));

        fe_rhs_simple_contact->getOpPtrVector().push_back(
            new OpLoopSlaveForSide(field_name, feMatSlaveSideStressRhs,
                                   side_fe_name));

        fe_rhs_simple_contact->getOpPtrVector().push_back(
            new OpCalProjStressesAtGaussPtsSlave(field_name,
                                                 common_data_simple_contact));


        //   //   // again
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            feMatSlaveSideStressDerRhs =
                boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                    mField);

        feMatSlaveSideStressDerRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetDataAtGaussPts(
                field_name,
                common_data_simple_contact->dataAtGaussPts[field_name],
                common_data_simple_contact->gradAtGaussPts[field_name]));

        feMatSlaveSideStressDerRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
                field_name,
                common_data_simple_contact->elasticityCommonData));

        feMatSlaveSideStressDerRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
                mesh_node_position,
                common_data_simple_contact->elasticityCommonData));

        feMatSlaveSideStressDerRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
                field_name, slave_block,
                common_data_simple_contact->elasticityCommonData,
                common_data_simple_contact->tAg, true, false, false));

        feMatSlaveSideStressDerRhs->getOpPtrVector().push_back(
            new OpStressDerivativeGapSlave_dx(
                field_name, common_data_simple_contact, slave_block,
                common_data_simple_contact->elasticityCommonData, cnValue,
                omegaValue, thetaSValue));

        fe_rhs_simple_contact->getOpPtrVector().push_back(
            new OpCalTildeCFunNitscheSlave(field_name,
                                           common_data_simple_contact, rValue,
                                           cnValue, omegaValue));

        fe_rhs_simple_contact->getOpPtrVector().push_back(
            new OpLoopSlaveForSide(field_name, feMatSlaveSideStressDerRhs,
            side_fe_name));

        fe_rhs_simple_contact->getOpPtrVector().push_back(
            new OpLoopMasterForSide(field_name, feMatMasterSideStressDerRhs,
                side_fe_name));

        fe_rhs_simple_contact->getOpPtrVector().push_back(
            new OpCalNitscheCStressRhsMaster(
                field_name, common_data_simple_contact, omegaValue, f_));

        fe_rhs_simple_contact->getOpPtrVector().push_back(
            new OpCalNitscheCStressRhsSlave(
                field_name, common_data_simple_contact, omegaValue, f_));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactNitschePenaltyLhsOperators(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, const string mesh_node_position, string side_fe_name,
      NonlinearElasticElement::BlockData &master_block,
      NonlinearElasticElement::BlockData &slave_block, Mat aij) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMaster(field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpContactSimplePenaltyMasterMaster(
    //           field_name, field_name, cnValue, common_data_simple_contact, aij));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpContactSimplePenaltyMasterSlave(field_name, field_name, cnValue,
    //                                             common_data_simple_contact, aij));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpContactSimplePenaltySlaveSlave(field_name, field_name, cnValue,
    //                                            common_data_simple_contact, aij));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpContactSimplePenaltySlaveMaster(field_name, field_name, cnValue,
    //                                             common_data_simple_contact, aij));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatMasterSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->dataAtGaussPtsMaster[field_name],
              common_data_simple_contact->gradAtGaussPtsMaster[field_name]));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->dataAtGaussPts[field_name],
              common_data_simple_contact->gradAtGaussPts[field_name]));

      cerr << "Material data Master "
           << "E " << master_block.E
           << " Poisson "
           << master_block.PoissonRatio
           << " Element name " << side_fe_name << "\n";
      common_data_simple_contact->tAg2 = 3;
      common_data_simple_contact->elasticityCommonData.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonData.meshPositions =
          mesh_node_position;

      common_data_simple_contact->elasticityCommonDataMaster.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonDataMaster.meshPositions =
          mesh_node_position;

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name, common_data_simple_contact->elasticityCommonDataMaster));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name,
              common_data_simple_contact->elasticityCommonData));

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonDataMaster));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonData));

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, master_block,
              common_data_simple_contact->elasticityCommonDataMaster,
              common_data_simple_contact->tAg2, false, false, false));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, slave_block,
              common_data_simple_contact->elasticityCommonData,
              common_data_simple_contact->tAg2, false, false, false));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpLoopMasterForSide(
          field_name, feMatMasterSideRhs, side_fe_name));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpLoopMasterForSideLhsTest(field_name, common_data_simple_contact,
    //                                      feMatMasterSideRhs, side_fe_name));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpCalProjStressesAtGaussPtsMaster(field_name,
                                                common_data_simple_contact));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpLoopSlaveForSideLhsTest(field_name, common_data_simple_contact,
    //                                     feMatSlaveSideRhs, side_fe_name));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpLoopSlaveForSide(
          field_name, feMatSlaveSideRhs, side_fe_name));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpCalProjStressesAtGaussPtsSlave(field_name,
                                               common_data_simple_contact));

      
      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpCalTildeCFunNitscheSlave(field_name, common_data_simple_contact,
                                         rValue, cnValue, omegaValue));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatMasterMasterSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatMasterMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->dataAtGaussPtsMaster[field_name],
              common_data_simple_contact->gradAtGaussPtsMaster[field_name]));

      feMatMasterMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name, common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonDataMaster));
      
      feMatMasterMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, master_block,
              common_data_simple_contact->elasticityCommonDataMaster,
              common_data_simple_contact->tAg, true, false, false));

      feMatMasterMasterSideRhs->getOpPtrVector().push_back(
          new OpStressDerivativeGapMasterMaster_dx(
              field_name, common_data_simple_contact, master_block,
              common_data_simple_contact->elasticityCommonDataMaster, cnValue, omegaValue, thetaSValue));

        

    //   feMatMasterMasterSideRhs->getOpPtrVector().push_back(
    //       new OpStressDerivativeGapSlaveMaster_dx(
    //           field_name, common_data_simple_contact,
    //           master_block,
    //           common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterMasterSideRhs->getOpPtrVector().push_back(
          new OpLoopForRowDataOnMaster(field_name, common_data_simple_contact));

      

    //   boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
    //       feMatSlaveMasterSideRhs =
    //           boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
    //               mField);
      
    //   feMatSlaveMasterSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetDataAtGaussPts(
    //           field_name,
    //           common_data_simple_contact->dataAtGaussPts[field_name],
    //           common_data_simple_contact->gradAtGaussPts[field_name]));

    //   feMatSlaveMasterSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
    //           field_name, common_data_simple_contact->elasticityCommonData));

    //   feMatSlaveMasterSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
    //           mesh_node_position,
    //           common_data_simple_contact->elasticityCommonData));
      
    //   feMatSlaveMasterSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
    //           field_name, slave_block,
    //           common_data_simple_contact->elasticityCommonData,
    //           common_data_simple_contact->tAg, true, false, false));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpLoopSlaveMasterForSideLhsTest(
    //           field_name, common_data_simple_contact, feMatSlaveMasterSideRhs,
    //           side_fe_name));

      //   feMatSlaveMasterSideRhs->getOpPtrVector().push_back(
      //       new OpStressDerivativeGapMasterMaster_dx(
      //           field_name, common_data_simple_contact, master_block,
      //           common_data_simple_contact->elasticityCommonData));

        // feMatSlaveMasterSideRhs->getOpPtrVector().push_back(
        //     new OpStressDerivativeGapSlaveMaster_dx(
        //         field_name, common_data_simple_contact,
        //         master_block,
        //         common_data_simple_contact->elasticityCommonData));

      

      //   fe_lhs_simple_contact->getOpPtrVector().push_back(
      //       new OpStressDerivativeGapMasterSlave_dx(
      //           field_name, common_data_simple_contact, master_block,
      //           common_data_simple_contact->elasticityCommonData,
      //           common_data_simple_contact->elasticityCommonData));

    //   cerr << "Material daasdta Slave "
    //        << "E " << slave_block.E << " Poisson " << slave_block.PoissonRatio
    //        << " Element name " << side_fe_name << "\n";

    //   boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
    //       feMatSlaveSideRhs =
    //           boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
    //               mField);

    //   feMatSlaveSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetDataAtGaussPts(
    //           field_name,
    //           common_data_simple_contact->dataAtGaussPts[field_name],
    //           common_data_simple_contact->gradAtGaussPts[field_name]));

    //   feMatSlaveSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
    //           field_name, common_data_simple_contact->elasticityCommonData));

    //   feMatSlaveSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
    //           mesh_node_position,
    //           common_data_simple_contact->elasticityCommonData));

    //   feMatSlaveSideRhs->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
    //           field_name, slave_block,
    //           common_data_simple_contact->elasticityCommonData,
    //           common_data_simple_contact->tAg, false, false, false));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpLoopSlaveForSide(field_name, feMatSlaveSideRhs, side_fe_name));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveSlaveSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSlaveSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->dataAtGaussPts[field_name],
              common_data_simple_contact->gradAtGaussPts[field_name]));

      feMatSlaveSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name, common_data_simple_contact->elasticityCommonData));

      feMatSlaveSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonData));

      feMatSlaveSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, slave_block,
              common_data_simple_contact->elasticityCommonData,
              common_data_simple_contact->tAg, true, false, false));

      feMatSlaveSlaveSideRhs->getOpPtrVector().push_back(
          new OpStressDerivativeGapSlaveSlave_dx(
              field_name, common_data_simple_contact, slave_block,
              common_data_simple_contact->elasticityCommonData, cnValue,
              omegaValue, thetaSValue));

      // 14/01/2020
      feMatSlaveSlaveSideRhs->getOpPtrVector().push_back(
          new OpLoopForRowDataOnSlave(field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpLoopSlaveForSide(
          field_name, feMatSlaveSlaveSideRhs, side_fe_name));

      //   fe_lhs_simple_contact->getOpPtrVector().push_back(
      //       new OpLoopSlaveForSideLhsTest(field_name,
      //       common_data_simple_contact,
      //                                     feMatSlaveSlaveSideRhs,
      //                                     side_fe_name));

    

      
      // 15/01/2020
    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpLoopMasterForSideLhsTest(field_name, common_data_simple_contact,
    //                                      feMatMasterMasterSideRhs,
    //                                      side_fe_name));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpLoopMasterForSide(
          field_name, feMatMasterMasterSideRhs, side_fe_name));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveMasterSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSlaveMasterSideRhs->getOpPtrVector().push_back(
          new OpStressDerivativeGapSlaveMaster_dx(
              field_name, common_data_simple_contact, slave_block,
              common_data_simple_contact->elasticityCommonData,
              common_data_simple_contact->elasticityCommonDataMaster, cnValue,
              omegaValue, thetaSValue));

      //   fe_lhs_simple_contact->getOpPtrVector().push_back(
      //       new OpLoopMasterForSideLhsTest(field_name,
      //       common_data_simple_contact,
      //                                      feMatSlaveMasterSideRhs,
      //                                      side_fe_name));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpLoopMasterForSide(
          field_name, feMatSlaveMasterSideRhs, side_fe_name));

      //   fe_lhs_simple_contact->getOpPtrVector().push_back(
      //       new OpStressDerivativeGapSlaveMaster_dx2(
      //           field_name, common_data_simple_contact, slave_block,
      //           common_data_simple_contact->elasticityCommonDataMaster,
      //           common_data_simple_contact->elasticityCommonData));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatMasterSlaveSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatMasterSlaveSideRhs->getOpPtrVector().push_back(
          new OpStressDerivativeGapMasterSlave_dx(
              field_name, common_data_simple_contact, master_block,
              common_data_simple_contact->elasticityCommonData,
              common_data_simple_contact->elasticityCommonDataMaster, cnValue,
              omegaValue, thetaSValue));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpLoopSlaveForSideLhsTest(field_name, common_data_simple_contact,
    //                                     feMatMasterSlaveSideRhs, side_fe_name));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpLoopSlaveForSide(
          field_name, feMatMasterSlaveSideRhs, side_fe_name));

      //   boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
      //       feMatMasterSlaveSideRhs =
      //           boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
      //               mField);

      //   feMatMasterSlaveSideRhs->getOpPtrVector().push_back(
      //       new NonlinearElasticElement::OpGetDataAtGaussPts(
      //           field_name,
      //           common_data_simple_contact->dataAtGaussPts[field_name],
      //           common_data_simple_contact->gradAtGaussPts[field_name]));

      //   feMatMasterSlaveSideRhs->getOpPtrVector().push_back(
      //       new NonlinearElasticElement::OpGetDataAtGaussPts(
      //           field_name,
      //           common_data_simple_contact->dataAtGaussPts[field_name],
      //           common_data_simple_contact->gradAtGaussPts[field_name]));

      //   feMatMasterSlaveSideRhs->getOpPtrVector().push_back(
      //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
      //           field_name,
      //           common_data_simple_contact->elasticityCommonData));

      //   feMatMasterSlaveSideRhs->getOpPtrVector().push_back(
      //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
      //           mesh_node_position,
      //           common_data_simple_contact->elasticityCommonData));

      //   feMatMasterSlaveSideRhs->getOpPtrVector().push_back(
      //       new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
      //           field_name, slave_block,
      //           common_data_simple_contact->elasticityCommonData,
      //           common_data_simple_contact->tAg, true, false, false));

      /*fe_lhs_simple_contact->getOpPtrVector().push_back(new
         OpLoopSlaveForSide( field_name, feMatMasterSlaveSideRhs,
         side_fe_name));*/

    //   boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact_2 =
    //       boost::make_shared<CommonDataSimpleContact>(mField);
    //   common_data_simple_contact_2->tAg = 2;
    //   boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
    //       feMatMasterSideForMasterSlave =
    //           boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
    //               mField);

    //   common_data_simple_contact_2->elasticityCommonData.spatialPositions =
    //       field_name;
    //   common_data_simple_contact_2->elasticityCommonData.meshPositions =
    //       mesh_node_position;

    //   common_data_simple_contact_2->elasticityCommonDataMaster
    //       .spatialPositions = field_name;
    //   common_data_simple_contact_2->elasticityCommonDataMaster.meshPositions =
    //       mesh_node_position;

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpGetNormalSlave(field_name, common_data_simple_contact_2));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpGetNormalMaster(field_name, common_data_simple_contact_2));

    //   feMatMasterSideForMasterSlave->getOpPtrVector().push_back(
    //       new OpLoopForRowDataOnMaster(field_name,
    //                                    common_data_simple_contact_2));

    //   feMatMasterSideForMasterSlave->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetDataAtGaussPts(
    //           field_name,
    //           common_data_simple_contact_2->dataAtGaussPtsMaster[field_name],
    //           common_data_simple_contact_2->gradAtGaussPtsMaster[field_name]));

    //   feMatMasterSideForMasterSlave->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
    //           field_name,
    //           common_data_simple_contact_2->elasticityCommonDataMaster));

    //   feMatMasterSideForMasterSlave->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
    //           mesh_node_position,
    //           common_data_simple_contact_2->elasticityCommonDataMaster));

    //   feMatMasterSideForMasterSlave->getOpPtrVector().push_back(
    //       new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
    //           field_name, master_block,
    //           common_data_simple_contact_2->elasticityCommonDataMaster,
    //           common_data_simple_contact_2->tAg, true, false, false));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(new OpLoopMasterForSide(
    //       field_name, feMatMasterSideForMasterSlave, side_fe_name));

      //  fe_lhs_simple_contact->getOpPtrVector().push_back(
      //      new OpStressDerivativeGapMasterSlave_dxCorrect(
      //          field_name, common_data_simple_contact_2, slave_block,
      //          common_data_simple_contact_2->elasticityCommonDataMaster));

      //    fe_lhs_simple_contact->getOpPtrVector().push_back(
      //        new OpStressDerivativeGapMasterSlave_dx(
      //            field_name, common_data_simple_contact_2, slave_block,
      //            common_data_simple_contact->elasticityCommonData,
      //            common_data_simple_contact_2->elasticityCommonDataMaster));

      //   fe_lhs_simple_contact->getOpPtrVector().push_back(
      //       new OpLoopMasterSlaveForSideLhsTest(field_name,
      //       common_data_simple_contact,
      //                                      feMatMasterSlaveSideRhs,
      //                                      side_fe_name));
    }
    MoFEMFunctionReturn(0);
  }

  // setup operators for calculation of active set
  MoFEMErrorCode setContactOperatorsRhsOperators(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
          string field_name,
      string lagrang_field_name, string side_fe_name, Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

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

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlaveForSide(field_name, common_data_simple_contact,
                                      feMatSideRhs, side_fe_name));

      // fe_rhs_simple_contact->getOpPtrVector().push_back(
      //     new OpContactConstraintMatrixMasterSlaveForSide(
      //         field_name, lagrang_field_name, common_data_simple_contact,
      //         feMatSideRhs, side_fe_name));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMaster(field_name, common_data_simple_contact));

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

      //   fe_rhs_simple_contact->getOpPtrVector().push_back(
      //       new OpPrintLagMulAtGaussPtsSlave(lagrang_field_name,
      //                                      common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConMaster(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConSlave(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalIntTildeCFunSlave(
          lagrang_field_name, common_data_simple_contact, f_));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsRhsOperatorsHdiv(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact, string
          field_name,
      string lagrang_field_name, Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
        boost::make_shared<CommonDataSimpleContact>(mField);

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
          new OpGetLagMulAtGaussPtsSlaveHdiv(lagrang_field_name,
                                             common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConMaster(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConSlave(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalIntTildeCFunSlaveHdiv(lagrang_field_name,
                                         common_data_simple_contact, f_));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsRhsOperatorsHdiv3D(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
      boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_hdiv_rhs_slave_tet,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name,
      string side_lagrange_volume_element, Vec f_ = PETSC_NULL) {
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
          new OpGetMeshPositionAtGaussPtsMaster("MESH_NODE_POSITIONS",
                                            common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetMeshPositionAtGaussPtsSlave("MESH_NODE_POSITIONS",
                                                common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          fe_hdiv_rhs_slave =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
          new OpCalculateHVecTensorField<3, 3>(
              lagrang_field_name,
              common_data_simple_contact->lagMatMultAtGaussPtsPtr));

      fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
          new OpCalculateHVecTensorDivergence<3, 3>(
              lagrang_field_name,
              common_data_simple_contact->contactLagrangeHdivDivergencePtr));

      fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, common_data_simple_contact->mGradPtr));

      fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<3>(
              field_name, common_data_simple_contact->xAtPts));

      fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<3>(
              "MESH_NODE_POSITIONS", common_data_simple_contact->xInitAtPts));

      fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
          new OpConstrainDomainRhs(lagrang_field_name,
                                   common_data_simple_contact));

      fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
          new OpInternalDomainContactRhs(field_name,
                                         common_data_simple_contact));
     
     //tets until here

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpConstrainBoundaryTraction(lagrang_field_name,
                                          common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpConstrainBoundaryRhs(lagrang_field_name,
                                     common_data_simple_contact, cnValue));

//=====
    //   fe_rhs_simple_contact->getOpPtrVector().push_back(
    //       new OpConstrainBoundaryRhsMaster(lagrang_field_name,
    //                                  common_data_simple_contact, cnValue));
//====

    //   fe_rhs_simple_contact->getOpPtrVector().push_back(
    //       new OpPassHdivToMasterNormal(field_name,
    //       common_data_simple_contact));

    //   fe_rhs_simple_contact->getOpPtrVector().push_back(
    //       new OpPassHdivToSlaveNormal(field_name, common_data_simple_contact, cnValue));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsOperatorsHdiv3D(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact_2,
      boost::shared_ptr<VolumeElementForcesAndSourcesCore>
          fe_hdiv_lhs_slave_tet,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrange_field_name,
      string side_lagrange_volume_element, Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(
    //       new OpGetMeshPositionAtGaussPtsMaster(field_name,
    //                                         common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));

      fe_hdiv_lhs_slave_tet->getOpPtrVector().push_back(
          new OpConstrainDomainLhs_dU(lagrange_field_name,
                                      field_name, common_data_simple_contact));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          fe_hdiv_lhs_slave =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      fe_lhs_simple_contact_2->getOpPtrVector().push_back(
          new OpConstrainBoundaryTraction(lagrange_field_name,
                                          common_data_simple_contact));

    //   fe_lhs_simple_contact_2->getOpPtrVector().push_back(
    //       new OpConstrainBoundaryLhs_dTractionMaster(
    //           lagrange_field_name, lagrange_field_name,
    //           common_data_simple_contact, cnValue));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpLoopSlaveForSide(
          lagrange_field_name, fe_hdiv_lhs_slave, side_lagrange_volume_element));

      fe_lhs_simple_contact_2->getOpPtrVector().push_back(
          new OpConstrainBoundaryLhs_dxdTraction(
              lagrange_field_name, lagrange_field_name,
              common_data_simple_contact, cnValue));

      fe_lhs_simple_contact_2->getOpPtrVector().push_back(
          new OpConstrainBoundaryLhs_dU_Slave(lagrange_field_name, field_name,
                                              common_data_simple_contact,
                                              cnValue));

    //   fe_lhs_simple_contact_2->getOpPtrVector().push_back(
    //       new OpConstrainBoundaryLhs_dU_Master(lagrange_field_name, field_name,
    //                                            common_data_simple_contact,
    //                                            cnValue));




    //   fe_lhs_simple_contact_2->getOpPtrVector().push_back(
    //       new OpConstrainBoundaryLhs_dU_dlambda_Master(
    //           field_name, lagrange_field_name, common_data_simple_contact,
    //           cnValue));

        //       fe_lhs_simple_contact_2->getOpPtrVector().push_back(
        //   new OpConstrainBoundaryLhs_dU_dlambda_Slave(
        //       field_name, lagrange_field_name, common_data_simple_contact,
        //       cnValue));




      //   fe_rhs_simple_contact->getOpPtrVector().push_back(
      //       new OpGetNormalSlave(field_name, common_data_simple_contact));

      //   fe_rhs_simple_contact->getOpPtrVector().push_back(
      //       new OpGetPositionAtGaussPtsMaster(field_name,
      //                                         common_data_simple_contact));

      //   fe_rhs_simple_contact->getOpPtrVector().push_back(
      //       new OpGetPositionAtGaussPtsSlave(field_name,
      //                                        common_data_simple_contact));

      //   fe_rhs_simple_contact->getOpPtrVector().push_back(
      //       new OpGetGapSlave(field_name, common_data_simple_contact));

      //   boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
      //       fe_hdiv_rhs_slave =
      //           boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
      //               mField);

      //   fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      //       new OpCalculateHVecTensorField<3, 3>(
      //           lagrang_field_name,
      //           common_data_simple_contact->lagMatMultAtGaussPtsPtr));

      //   fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      //       new OpCalculateHVecTensorDivergence<3, 3>(
      //           lagrang_field_name,
      //           common_data_simple_contact->contactLagrangeHdivDivergencePtr));

      //   fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      //       new OpCalculateVectorFieldGradient<3, 3>(
      //           field_name, common_data_simple_contact->mGradPtr));

      //   fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      //       new OpCalculateVectorFieldValues<3>(
      //           field_name, common_data_simple_contact->xAtPts));

      //   fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      //       new OpCalculateVectorFieldValues<3>(
      //           "MESH_NODE_POSITIONS",
      //           common_data_simple_contact->xInitAtPts));

      //   fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      //       new OpConstrainDomainRhs(lagrang_field_name,
      //                                common_data_simple_contact));

      //   fe_hdiv_rhs_slave_tet->getOpPtrVector().push_back(
      //       new OpInternalDomainContactRhs(field_name,
      //                                      common_data_simple_contact));
    }
    MoFEMFunctionReturn(0);
  }

  // setup operators for calculation of active set
  MoFEMErrorCode setContactOperatorsRhsALE(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      const string field_name, const string mesh_node_field_name,
      const string lagrang_field_name, const string side_fe_name,
      Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideRhsMaster =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSideRhsMaster->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              mesh_node_field_name, common_data_simple_contact->HMat));
      feMatSideRhsMaster->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, common_data_simple_contact->hMat));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideRhsSlave =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSideRhsSlave->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              mesh_node_field_name, common_data_simple_contact->HMat));
      feMatSideRhsSlave->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, common_data_simple_contact->hMat));

      // fe_rhs_simple_contact->getOpPtrVector().push_back(
      //     new OpContactConstraintMatrixMasterSlaveForSide(
      //         field_name, lagrang_field_name, common_data_simple_contact,
      //         feMatSideRhs, side_fe_name));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetTangentSlave(mesh_node_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
          mesh_node_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetTangentMaster(mesh_node_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalMasterALE(
          mesh_node_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlaveALE(field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConMasterALE(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConSlaveALE(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalIntTildeCFunSlaveALE(lagrang_field_name,
                                        common_data_simple_contact, f_));

      //this is the right order
      feMatSideRhsMaster->getOpPtrVector().push_back(new OpCalculateDeformation(
          mesh_node_field_name, common_data_simple_contact, false));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpLoopMasterForSide(
          mesh_node_field_name, feMatSideRhsMaster, side_fe_name));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalMatForcesALEMaster(mesh_node_field_name,
                                      common_data_simple_contact));

      feMatSideRhsSlave->getOpPtrVector().push_back(new OpCalculateDeformation(
          mesh_node_field_name, common_data_simple_contact, false));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpLoopSlaveForSide(
          mesh_node_field_name, feMatSideRhsSlave, side_fe_name));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalMatForcesALESlave(mesh_node_field_name,
                                      common_data_simple_contact));
    }
    MoFEMFunctionReturn(0);
  }

  // setup operators for calculation of active set
  MoFEMErrorCode setContactOperatorsRhsALESpatial(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact>
          common_data_simple_contact, const string field_name,
      const string mesh_node_field_name, const string lagrang_field_name,
      const string side_fe_name, Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetTangentSlave(
          mesh_node_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
          mesh_node_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlaveALE(field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConMasterALE(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConSlaveALE(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalIntTildeCFunSlaveALE(lagrang_field_name,
                                        common_data_simple_contact, f_));
    }
    MoFEMFunctionReturn(0);
  }

  // setup operators for calculation of active set
  MoFEMErrorCode setContactOperatorsRhsALEMaterial(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact>
          common_data_simple_contact, const string field_name,
      const string mesh_node_field_name, const string lagrang_field_name,
      const string side_fe_name, Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideRhsMaster =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSideRhsMaster->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              mesh_node_field_name, common_data_simple_contact->HMat));
      feMatSideRhsMaster->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, common_data_simple_contact->hMat));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideRhsSlave =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSideRhsSlave->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              mesh_node_field_name, common_data_simple_contact->HMat));
      feMatSideRhsSlave->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, common_data_simple_contact->hMat));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetTangentSlave(
          mesh_node_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
          mesh_node_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpGetTangentMaster(
          mesh_node_field_name, common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMasterALE(mesh_node_field_name,
                                   common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         common_data_simple_contact));

      // this is the right order
      feMatSideRhsMaster->getOpPtrVector().push_back(new OpCalculateDeformation(
          mesh_node_field_name, common_data_simple_contact, false));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpLoopMasterForSide(
          mesh_node_field_name, feMatSideRhsMaster, side_fe_name));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalMatForcesALEMaster(mesh_node_field_name,
                                      common_data_simple_contact));

      feMatSideRhsSlave->getOpPtrVector().push_back(new OpCalculateDeformation(
          mesh_node_field_name, common_data_simple_contact, false));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpLoopSlaveForSide(
          mesh_node_field_name, feMatSideRhsSlave, side_fe_name));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalMatForcesALESlave(mesh_node_field_name,
                                     common_data_simple_contact));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactPenaltyLhsOperators(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,  
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, Mat aij) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

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
          new OpContactSimplePenaltyMasterMaster(
              field_name, field_name, cnValue, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactSimplePenaltyMasterSlave(
              field_name, field_name, cnValue, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactSimplePenaltySlaveSlave(field_name, field_name, cnValue,
                                                common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactSimplePenaltySlaveMaster(field_name, field_name, cnValue,
                                                common_data_simple_contact, aij));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsOperators(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact, 
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name,
      string lagrang_field_name, Mat aij) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMaster(field_name, common_data_simple_contact));

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
              field_name, lagrang_field_name, cnValue, common_data_simple_contact,
              aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlave(
              field_name, lagrang_field_name, cnValue, common_data_simple_contact,
              aij));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsALE(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact, 
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact, const string
          field_name,
      const string mesh_node_field_name, const string lagrang_field_name,
      const string side_fe_name, Mat aij = NULL) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetTangentSlave(mesh_node_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
          mesh_node_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetTangentMaster(
          mesh_node_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalMasterALE(
          mesh_node_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlaveALE(field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixMasterSlaveALE(
              field_name, lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlaveALE(
              field_name, lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlave_dX(
              field_name, mesh_node_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixMasterSlave_dX(
              field_name, mesh_node_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunOLambdaSlaveSlaveALE(
              lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE(
              field_name, lagrang_field_name, cnValue, common_data_simple_contact,
              aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE(
              field_name, lagrang_field_name, cnValue, common_data_simple_contact,
              aij));

      // fe_lhs_simple_contact->getOpPtrVector().push_back(
      //     new OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE_dX(
      //         mesh_node_field_name, lagrang_field_name, cnValue,
      //         common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE_dX(
              mesh_node_field_name, lagrang_field_name, cnValue,
              common_data_simple_contact, aij));

      // Side volume element computes linearisation with spatial coordinates
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideLhs_dx =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);
      // Side volume element computes linearisation with material coordinates
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideLhs_dX =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideLhs_dLambda =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSideLhs_dx->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(mesh_node_field_name, common_data_simple_contact->HMat));
      feMatSideLhs_dx->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(field_name, common_data_simple_contact->hMat));

      // Master derivative over spatial
      feMatSideLhs_dx->getOpPtrVector().push_back(
          new OpCalculateDeformation(mesh_node_field_name, common_data_simple_contact, false));
      feMatSideLhs_dx->getOpPtrVector().push_back(
          new OpContactMaterialVolOnMasterSideLhs_dX_dx(
              mesh_node_field_name, field_name, common_data_simple_contact, aij, false));

    //   fe_lhs_simple_contact->getOpPtrVector().push_back(new OpLoopMasterForSideLhs(
    //       mesh_node_field_name, common_data_simple_contact, feMatSideLhs_dx,
    //       side_fe_name));

        fe_lhs_simple_contact->getOpPtrVector().push_back(
            new OpContactMaterialMasterLhs_dX_dx(
                mesh_node_field_name, field_name, common_data_simple_contact,
                feMatSideLhs_dx, side_fe_name, aij, false));

        // Master derivative over mesh_nodes
        feMatSideLhs_dX->getOpPtrVector().push_back(
            new OpCalculateVectorFieldGradient<3, 3>(
                mesh_node_field_name, common_data_simple_contact->HMat));

        feMatSideLhs_dX->getOpPtrVector().push_back(
            new OpCalculateVectorFieldGradient<3, 3>(
                field_name, common_data_simple_contact->hMat));

        feMatSideLhs_dX->getOpPtrVector().push_back(new OpCalculateDeformation(
            mesh_node_field_name, common_data_simple_contact, false));

        feMatSideLhs_dX->getOpPtrVector().push_back(
            new OpContactMaterialVolOnMasterSideLhs_dX_dX(
                mesh_node_field_name, mesh_node_field_name,
                common_data_simple_contact, aij, false));
        
        // fe_lhs_simple_contact->getOpPtrVector().push_back(
        //     new OpLoopMasterForSideLhs(mesh_node_field_name,
        //                                common_data_simple_contact, feMatSideLhs_dX,
        //                                side_fe_name));
       
        fe_lhs_simple_contact->getOpPtrVector().push_back(
            new OpContactMaterialMasterLhs_dX_dX(
                mesh_node_field_name, mesh_node_field_name,
                common_data_simple_contact, feMatSideLhs_dX, side_fe_name, aij,
                false));

        // feMatSideLhs_dLambda->getOpPtrVector().push_back(
        //     new OpContactMaterialVolOnMasterSideLhs_dX_dLagmult(
        //         mesh_node_field_name, lagrang_field_name,
        //         common_data_simple_contact, aij, false));


        fe_lhs_simple_contact->getOpPtrVector().push_back(
            new OpContactMaterialMasterSlaveLhs_dX_dLagmult(
                mesh_node_field_name, lagrang_field_name,
                common_data_simple_contact, feMatSideLhs_dLambda, side_fe_name,
                aij, false));

//Slave ALE!

        // Side volume element computes linearisation with spatial coordinates
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            feMatSideLhs_Slave_dx = boost::make_shared<
                VolumeElementForcesAndSourcesCoreOnVolumeSide>(mField);
        // Side volume element computes linearisation with material coordinates
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            feMatSideLhs_Slave_dX = boost::make_shared<
                VolumeElementForcesAndSourcesCoreOnVolumeSide>(mField);

        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            feMatSideLhs_Slave_dLambda = boost::make_shared<
                VolumeElementForcesAndSourcesCoreOnVolumeSide>(mField);

        feMatSideLhs_Slave_dx->getOpPtrVector().push_back(
            new OpCalculateVectorFieldGradient<3, 3>(
                mesh_node_field_name, common_data_simple_contact->HMat));
        feMatSideLhs_Slave_dx->getOpPtrVector().push_back(
            new OpCalculateVectorFieldGradient<3, 3>(
                field_name, common_data_simple_contact->hMat));

        // Master derivative over spatial
        feMatSideLhs_Slave_dx->getOpPtrVector().push_back(new OpCalculateDeformation(
            mesh_node_field_name, common_data_simple_contact, false));
        
        //error 1
        feMatSideLhs_Slave_dx->getOpPtrVector().push_back(
            new OpContactMaterialVolOnSlaveSideLhs_dX_dx(
                mesh_node_field_name, field_name, common_data_simple_contact, aij,
                false));

        //   fe_lhs_simple_contact->getOpPtrVector().push_back(new
        //   OpLoopMasterForSideLhs(
        //       mesh_node_field_name, common_data_simple_contact, feMatSideLhs_Slave_dx,
        //       side_fe_name));

//Not error?!
        fe_lhs_simple_contact->getOpPtrVector().push_back(
            new OpContactMaterialSlaveLhs_dX_dx(
                mesh_node_field_name, field_name, common_data_simple_contact,
                feMatSideLhs_Slave_dx, side_fe_name, aij, false));

        // Slave derivative over mesh_nodes
        feMatSideLhs_Slave_dX->getOpPtrVector().push_back(
            new OpCalculateVectorFieldGradient<3, 3>(
                mesh_node_field_name, common_data_simple_contact->HMat));

        feMatSideLhs_Slave_dX->getOpPtrVector().push_back(
            new OpCalculateVectorFieldGradient<3, 3>(
                field_name, common_data_simple_contact->hMat));

        feMatSideLhs_Slave_dX->getOpPtrVector().push_back(new OpCalculateDeformation(
            mesh_node_field_name, common_data_simple_contact, false));

//Error2!!!
        feMatSideLhs_Slave_dX->getOpPtrVector().push_back(
            new OpContactMaterialVolOnSlaveSideLhs_dX_dX(
                mesh_node_field_name, mesh_node_field_name,
                common_data_simple_contact, aij, false));

        // fe_lhs_simple_contact->getOpPtrVector().push_back(
        //     new OpLoopSlaveForSideLhs(mesh_node_field_name,
        //                                common_data_simple_contact,
        //                                feMatSideLhs_Slave_dX, side_fe_name));

        fe_lhs_simple_contact->getOpPtrVector().push_back(
            new OpContactMaterialSlaveLhs_dX_dX(
                mesh_node_field_name, mesh_node_field_name,
                common_data_simple_contact, feMatSideLhs_Slave_dX, side_fe_name, aij,
                false));

        fe_lhs_simple_contact->getOpPtrVector().push_back(
            new OpContactMaterialSlaveSlaveLhs_dX_dLagmult(
                mesh_node_field_name, lagrang_field_name,
                common_data_simple_contact, feMatSideLhs_Slave_dLambda, side_fe_name,
                aij, false));

                //End

        // feMatSideLhs_Slave_dLambda->getOpPtrVector().push_back(
        //     new OpContactMaterialVolOnSlaveSideLhs_dX_dLagmult(
        //         mesh_node_field_name, lagrang_field_name,
        //         commonDataSimpleContact, aij, false));


//todo !!!
        // feLhsSimpleContact->getOpPtrVector().push_back(
        //     new OpContactMaterialSlaveSlaveLhs_dX_dLagmult(
        //         mesh_node_field_name, lagrang_field_name,
        //         commonDataSimpleContact, feMatSideLhs_Slave_dLambda, side_fe_name,
        //         aij, false));

        // feLhsSimpleContact->getOpPtrVector().push_back(
        //     new OpContactMaterialMasterSlaveLhs_dX_dX(
        //         mesh_node_field_name, mesh_node_field_name,
        //         commonDataSimpleContact, feMatSideLhs_Slave_dLambda, side_fe_name,
        //         aij, false));

        // Slave
        //   feMatSideLhs_Slave_dx->getOpPtrVector().push_back(new
        //   OpCalculateDeformation(
        //       mesh_node_field_name, commonDataSimpleContact, false));
        //   //   feMatSideLhs_Slave_dx->getOpPtrVector().push_back(
        //   //       new OpContactMaterialVolOnSlaveSideLhs_dX_dx(
        //   //           mesh_node_field_name, field_name,
        //   commonDataSimpleContact,
        //   //           aij, mapPressure[ms_id], surface_pressure, false));
        //   feLhsSimpleContact->getOpPtrVector().push_back(new
        //   OpLoopSlaveForSideLhs(
        //       mesh_node_field_name, commonDataSimpleContact,
        //       feMatSideLhs_Slave_dx, side_fe_name));

        //   //   feMatLhs.getOpPtrVector().push_back(
        //   //       new OpContactMaterialMasterLhs_dX_dx(
        //   //           mesh_node_field_name, field_name,
        //   commonDataSimpleContact,
        //   //           feMatSideLhs_Slave_dx, side_fe_name, aij,
        //   mapPressure[ms_id],
        //   //           surface_pressure, false));




    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsALESpatial(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact>
          common_data_simple_contact, const string field_name,
      const string mesh_node_field_name, const string lagrang_field_name,
      const string side_fe_name, Mat aij = NULL) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetTangentSlave(
          mesh_node_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
          mesh_node_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlaveALE(field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixMasterSlaveALE(
              field_name, lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlaveALE(
              field_name, lagrang_field_name, common_data_simple_contact, aij));


      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunOLambdaSlaveSlaveALE(
              lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveMasterALE(
              field_name, lagrang_field_name, cnValue,
              common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE(
              field_name, lagrang_field_name, cnValue,
              common_data_simple_contact, aij));

    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsALEMaterial(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      const string field_name, const string mesh_node_field_name,
      const string lagrang_field_name, const string side_fe_name,
      Mat aij = NULL) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetTangentSlave(
          mesh_node_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetNormalSlaveALE(
          mesh_node_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpGetTangentMaster(
          mesh_node_field_name, common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMasterALE(mesh_node_field_name,
                                   common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                         common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlave_dX(
              field_name, mesh_node_field_name, common_data_simple_contact,
              aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixMasterSlave_dX(
              field_name, mesh_node_field_name, common_data_simple_contact,
              aij));

     
      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunODisplacementsSlaveSlaveALE_dX(
              mesh_node_field_name, lagrang_field_name, cnValue,
              common_data_simple_contact, aij));

      // Side volume element computes linearisation with spatial coordinates
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideLhs_dx =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);
      // Side volume element computes linearisation with material coordinates
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideLhs_dX =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideLhs_dLambda =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSideLhs_dx->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              mesh_node_field_name, common_data_simple_contact->HMat));
      feMatSideLhs_dx->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, common_data_simple_contact->hMat));

      // Master derivative over spatial
      feMatSideLhs_dx->getOpPtrVector().push_back(new OpCalculateDeformation(
          mesh_node_field_name, common_data_simple_contact, false));
      feMatSideLhs_dx->getOpPtrVector().push_back(
          new OpContactMaterialVolOnMasterSideLhs_dX_dx(
              mesh_node_field_name, field_name, common_data_simple_contact, aij,
              false));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactMaterialMasterLhs_dX_dx(
              mesh_node_field_name, field_name, common_data_simple_contact,
              feMatSideLhs_dx, side_fe_name, aij, false));

      // Master derivative over mesh_nodes
      feMatSideLhs_dX->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              mesh_node_field_name, common_data_simple_contact->HMat));

      feMatSideLhs_dX->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, common_data_simple_contact->hMat));

      feMatSideLhs_dX->getOpPtrVector().push_back(new OpCalculateDeformation(
          mesh_node_field_name, common_data_simple_contact, false));

      feMatSideLhs_dX->getOpPtrVector().push_back(
          new OpContactMaterialVolOnMasterSideLhs_dX_dX(
              mesh_node_field_name, mesh_node_field_name,
              common_data_simple_contact, aij, false));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactMaterialMasterLhs_dX_dX(
              mesh_node_field_name, mesh_node_field_name,
              common_data_simple_contact, feMatSideLhs_dX, side_fe_name, aij,
              false));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactMaterialMasterSlaveLhs_dX_dLagmult(
              mesh_node_field_name, lagrang_field_name,
              common_data_simple_contact, feMatSideLhs_dLambda, side_fe_name,
              aij, false));

      // Slave ALE!

      // Side volume element computes linearisation with spatial coordinates
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideLhs_Slave_dx =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);
      // Side volume element computes linearisation with material coordinates
      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideLhs_Slave_dX =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSideLhs_Slave_dLambda =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSideLhs_Slave_dx->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              mesh_node_field_name, common_data_simple_contact->HMat));
      feMatSideLhs_Slave_dx->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, common_data_simple_contact->hMat));

      // Master derivative over spatial
      feMatSideLhs_Slave_dx->getOpPtrVector().push_back(
          new OpCalculateDeformation(mesh_node_field_name,
                                     common_data_simple_contact, false));

      feMatSideLhs_Slave_dx->getOpPtrVector().push_back(
          new OpContactMaterialVolOnSlaveSideLhs_dX_dx(
              mesh_node_field_name, field_name, common_data_simple_contact, aij,
              false));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactMaterialSlaveLhs_dX_dx(
              mesh_node_field_name, field_name, common_data_simple_contact,
              feMatSideLhs_Slave_dx, side_fe_name, aij, false));

      // Slave derivative over mesh_nodes
      feMatSideLhs_Slave_dX->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              mesh_node_field_name, common_data_simple_contact->HMat));

      feMatSideLhs_Slave_dX->getOpPtrVector().push_back(
          new OpCalculateVectorFieldGradient<3, 3>(
              field_name, common_data_simple_contact->hMat));

      feMatSideLhs_Slave_dX->getOpPtrVector().push_back(
          new OpCalculateDeformation(mesh_node_field_name,
                                     common_data_simple_contact, false));

      feMatSideLhs_Slave_dX->getOpPtrVector().push_back(
          new OpContactMaterialVolOnSlaveSideLhs_dX_dX(
              mesh_node_field_name, mesh_node_field_name,
              common_data_simple_contact, aij, false));

 
      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactMaterialSlaveLhs_dX_dX(
              mesh_node_field_name, mesh_node_field_name,
              common_data_simple_contact, feMatSideLhs_Slave_dX, side_fe_name,
              aij, false));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactMaterialSlaveSlaveLhs_dX_dLagmult(
              mesh_node_field_name, lagrang_field_name,
              common_data_simple_contact, feMatSideLhs_Slave_dLambda,
              side_fe_name, aij, false));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsOperatorsHdiv(boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
  boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
                                                     string field_name,
                                                     string lagrang_field_name,
                                                     Mat aij) {
    MoFEMFunctionBegin;

   boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
       boost::make_shared<CommonDataSimpleContact>(mField);

   map<int, SimpleContactPrismsData>::iterator sit =
       setOfSimpleContactPrism.begin();
   for (; sit != setOfSimpleContactPrism.end(); sit++) {

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpGetNormalSlave(field_name, common_data_simple_contact));

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpGetPositionAtGaussPtsMaster(field_name,
                                           common_data_simple_contact));

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpGetPositionAtGaussPtsSlave(field_name, common_data_simple_contact));

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpGetGapSlave(field_name, common_data_simple_contact));

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpGetLagMulAtGaussPtsSlaveHdiv(lagrang_field_name,
                                            common_data_simple_contact));

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpContactConstraintMatrixMasterSlaveHdiv(
             field_name, lagrang_field_name, common_data_simple_contact, aij));

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpContactConstraintMatrixSlaveSlaveHdiv(
             field_name, lagrang_field_name, common_data_simple_contact, aij));

     fe_lhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
         field_name, common_data_simple_contact, rValue, cnValue));

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpDerivativeBarTildeCFunOLambdaSlaveSlaveHdiv(
             lagrang_field_name, common_data_simple_contact, aij));

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpDerivativeBarTildeCFunODisplacementsSlaveMasterHdiv(
             field_name, lagrang_field_name, cnValue, common_data_simple_contact,
             aij));

     fe_lhs_simple_contact->getOpPtrVector().push_back(
         new OpDerivativeBarTildeCFunODisplacementsSlaveSlaveHdiv(
             field_name, lagrang_field_name, cnValue, common_data_simple_contact,
             aij));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsForPostProc(
      boost::shared_ptr<SimpleContactElement> fe_post_proc_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      MoFEM::Interface &m_field, string field_name, string mesh_node_position,
      string lagrang_field_name, string side_fe_name, moab::Interface &moab_out,
    NonlinearElasticElement::BlockData &master_block,
      NonlinearElasticElement::BlockData &slave_block,
      bool lagrange_field = true) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMaster(field_name, common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatMasterSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);
      common_data_simple_contact->tAg = 2;
       
        feMatMasterSideRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetDataAtGaussPts(
                field_name,
                common_data_simple_contact->elasticityCommonDataMaster
                    .dataAtGaussPts[field_name],
                common_data_simple_contact
                    ->elasticityCommonDataMaster.gradAtGaussPts[field_name]));

      cerr << "Material data Master "
           << "E " << master_block.E << " Poisson "
           << master_block.PoissonRatio
           << " Element name " << side_fe_name << "\n";

      common_data_simple_contact->elasticityCommonDataMaster.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonDataMaster.meshPositions =
          mesh_node_position;

      common_data_simple_contact->elasticityCommonData.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonData.meshPositions =
          mesh_node_position;

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name, common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
                  field_name, master_block,
                  common_data_simple_contact->elasticityCommonDataMaster,
                  common_data_simple_contact->tAg, false, false, false));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpLoopMasterForSide(mesh_node_position, feMatMasterSideRhs,
                                  side_fe_name));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpCalProjStressesAtGaussPtsMaster(mesh_node_position,
                                              common_data_simple_contact));

      cerr << "Material data Slave "
           << "E " << slave_block.E
           << " Poisson "
           << slave_block.PoissonRatio
           << " Element name " << side_fe_name << "\n";

     
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
            feMatSlaveSideRhs =
                boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                    mField);

        feMatSlaveSideRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetDataAtGaussPts(
                field_name, common_data_simple_contact->dataAtGaussPts[field_name],
                common_data_simple_contact->gradAtGaussPts[field_name]));

        feMatSlaveSideRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
                field_name, common_data_simple_contact->elasticityCommonData));

        feMatSlaveSideRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
                mesh_node_position,
                common_data_simple_contact->elasticityCommonData));

        feMatSlaveSideRhs->getOpPtrVector().push_back(
            new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
                field_name, slave_block,
                common_data_simple_contact->elasticityCommonData,
                common_data_simple_contact->tAg, false, false, false));

        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpLoopSlaveForSide(mesh_node_position, feMatSlaveSideRhs,
                                    side_fe_name));

        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpCalProjStressesAtGaussPtsSlave(mesh_node_position,
                                                  common_data_simple_contact));

      //   if (mField.check_field(material_position_field_name)) {
      //     feRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
      //         material_position_field_name, commonData));
      //   }
      //   std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
      //   for (; sit != setOfBlocks.end(); sit++) {
      //     feRhs.getOpPtrVector().push_back(new
      //     OpJacobianPiolaKirchhoffStress(
      //         spatial_position_field_name, sit->second, commonData, tAg,
      //         false, ale, field_disp));
      //     feRhs.getOpPtrVector().push_back(new OpRhsPiolaKirchhoff(
      //         spatial_position_field_name, sit->second, commonData));
      //   }

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));
      
      if (lagrange_field) {
        cerr << "LAGRAAANGE!\n";
        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpGetLagMulAtGaussPtsSlave(lagrang_field_name,
                                           common_data_simple_contact));

        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpLagGapProdGaussPtsSlave(lagrang_field_name,
                                          common_data_simple_contact));
        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpCalNormalLagrangeMasterAndSlaveDifference(
                field_name, common_data_simple_contact));

        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpCalRelativeErrorNormalLagrangeMasterAndSlaveDifference(
                field_name, common_data_simple_contact));

        // fe_post_proc_simple_contact->getOpPtrVector().push_back(
        //     new OpLagMultOnVertex(mField, field_name, mb_post_vertex));
      }

    //   fe_post_proc_simple_contact->getOpPtrVector().push_back(
    //       new OpCalTildeCFunNitscheSlave(field_name, common_data_simple_contact,
    //                                      rValue, cnValue, omegaValue));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpMakeVtkSlave(m_field, field_name, common_data_simple_contact,
                             moab_out, lagrange_field));
    }
    MoFEMFunctionReturn(0);
  }

  /*
  boost::shared_ptr<SimpleContactElement> fe_post_proc_simple_contact,
        boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
        MoFEM::Interface &m_field, string field_name, string mesh_node_position,
        string lagrang_field_name, string side_fe_name, moab::Interface
  &moab_out, NonlinearElasticElement::BlockData &master_block,
        NonlinearElasticElement::BlockData &slave_block,
        bool lagrange_field = true
  */

  MoFEMErrorCode setContactOperatorsForPostProcHdiv(
      boost::shared_ptr<SimpleContactElement> fe_post_proc_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      MoFEM::Interface &m_field, string field_name, string mesh_node_position,
      string lagrang_field_name, string side_fe_name, moab::Interface &moab_out,
      NonlinearElasticElement::BlockData &master_block,
      NonlinearElasticElement::BlockData &slave_block,
      bool lagrange_field = true) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMaster(field_name, common_data_simple_contact));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatMasterSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);
      common_data_simple_contact->tAg = 2;

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->elasticityCommonDataMaster
                  .dataAtGaussPts[field_name],
              common_data_simple_contact->elasticityCommonDataMaster
                  .gradAtGaussPts[field_name]));

      cerr << "Material data Master "
           << "E " << master_block.E << " Poisson " << master_block.PoissonRatio
           << " Element name " << side_fe_name << "\n";

      common_data_simple_contact->elasticityCommonDataMaster.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonDataMaster.meshPositions =
          mesh_node_position;

      common_data_simple_contact->elasticityCommonData.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonData.meshPositions =
          mesh_node_position;

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name,
              common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, master_block,
              common_data_simple_contact->elasticityCommonDataMaster,
              common_data_simple_contact->tAg, false, false, false));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpLoopMasterForSide(mesh_node_position, feMatMasterSideRhs,
                                  side_fe_name));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpCalProjStressesAtGaussPtsMaster(mesh_node_position,
                                                common_data_simple_contact));

      cerr << "Material data Slave "
           << "E " << slave_block.E << " Poisson " << slave_block.PoissonRatio
           << " Element name " << side_fe_name << "\n";

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->dataAtGaussPts[field_name],
              common_data_simple_contact->gradAtGaussPts[field_name]));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name, common_data_simple_contact->elasticityCommonData));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonData));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, slave_block,
              common_data_simple_contact->elasticityCommonData,
              common_data_simple_contact->tAg, false, false, false));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpLoopSlaveForSide(mesh_node_position, feMatSlaveSideRhs,
                                 side_fe_name));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpCalProjStressesAtGaussPtsSlave(mesh_node_position,
                                               common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));

      if (lagrange_field) {
        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpGetLagMulAtGaussPtsSlaveHdiv(lagrang_field_name,
                                               common_data_simple_contact));

        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpLagGapProdGaussPtsSlave(lagrang_field_name,
                                          common_data_simple_contact));
        }
      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpCalNormalLagrangeMasterAndSlaveDifference(
              field_name, common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpCalRelativeErrorNormalLagrangeMasterAndSlaveDifference(
              field_name, common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(new OpMakeVtkSlave(
          m_field, field_name, common_data_simple_contact, moab_out));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsRhsOperatorsL2(
      boost::shared_ptr<SimpleContactElement> fe_rhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name, string side_fe_name,
      Vec f_ = PETSC_NULL) {
    MoFEMFunctionBegin;

    boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
        boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      cerr << "\nL2 L2 L2 L2 \n";

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

//create some loop for side element

      
      boost::shared_ptr<FaceElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveSideRhsL2 =
              boost::make_shared<FaceElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSlaveSideRhsL2->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlaveL2(lagrang_field_name,
                                           common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpLoopSlaveForSideFace(
          field_name, feMatSlaveSideRhsL2, side_fe_name));


      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConMaster(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpCalFReConSlave(field_name, common_data_simple_contact, f_));

      fe_rhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      boost::shared_ptr<FaceElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveSideRhsL2Second =
              boost::make_shared<FaceElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSlaveSideRhsL2Second->getOpPtrVector().push_back(
          new OpCalIntTildeCFunSlaveL2(lagrang_field_name,
                                       common_data_simple_contact));

      fe_rhs_simple_contact->getOpPtrVector().push_back(
          new OpLoopSlaveForSideFace(field_name, feMatSlaveSideRhsL2Second,
                                     side_fe_name));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsLhsOperatorsL2(
      boost::shared_ptr<SimpleContactElement> fe_lhs_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      string field_name, string lagrang_field_name, string side_fe_name,
      Mat aij) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

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

      boost::shared_ptr<FaceElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveSideRhsL2 =
              boost::make_shared<FaceElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSlaveSideRhsL2->getOpPtrVector().push_back(
          new OpGetLagMulAtGaussPtsSlaveL2(lagrang_field_name,
                                           common_data_simple_contact));

      feMatSlaveSideRhsL2->getOpPtrVector().push_back(
          new OpGetRowColDataForSideFace(lagrang_field_name,
                                         common_data_simple_contact));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpLoopSlaveForSideFace(field_name, feMatSlaveSideRhsL2,
                                     side_fe_name));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixMasterSlaveH1L2(
              field_name, field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpContactConstraintMatrixSlaveSlaveH1L2(
              field_name, field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(new OpCalTildeCFunSlave(
          field_name, common_data_simple_contact, rValue, cnValue));

      boost::shared_ptr<FaceElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveSideRhsL2L2 =
              boost::make_shared<FaceElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);
      // L2 - L2

      feMatSlaveSideRhsL2L2->getOpPtrVector().push_back(
          new OpDerivativeBarTildeCFunOLambdaSlaveSlaveL2L2(
              lagrang_field_name, common_data_simple_contact, aij));

      fe_lhs_simple_contact->getOpPtrVector().push_back(
          new OpLoopSlaveForSideFace(field_name, feMatSlaveSideRhsL2L2,
                                     side_fe_name));

      //   fe_lhs_simple_contact->getOpPtrVector().push_back(
      //       new OpDerivativeBarTildeCFunODisplacementsSlaveMaster(
      //           field_name, lagrang_field_name, cnValue,
      //           common_data_simple_contact, aij));

      //   fe_lhs_simple_contact->getOpPtrVector().push_back(
      //       new OpDerivativeBarTildeCFunODisplacementsSlaveSlave(
      //           field_name, lagrang_field_name, cnValue,
      //           common_data_simple_contact, aij));
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setContactOperatorsForPostProcL2(
      boost::shared_ptr<SimpleContactElement> fe_post_proc_simple_contact,
      boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact,
      MoFEM::Interface &m_field, string field_name, string mesh_node_position,
      string lagrang_field_name, string side_fe_name, string side_face_fe_name, moab::Interface &moab_out,
      NonlinearElasticElement::BlockData &master_block,
      NonlinearElasticElement::BlockData &slave_block,
      bool lagrange_field = true) {
    MoFEMFunctionBegin;

    // boost::shared_ptr<CommonDataSimpleContact> common_data_simple_contact =
    //     boost::make_shared<CommonDataSimpleContact>(mField);

    map<int, SimpleContactPrismsData>::iterator sit =
        setOfSimpleContactPrism.begin();
    for (; sit != setOfSimpleContactPrism.end(); sit++) {

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalMaster(field_name, common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetNormalSlave(field_name, common_data_simple_contact));

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatMasterSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);
      common_data_simple_contact->tAg = 2;

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->elasticityCommonDataMaster
                  .dataAtGaussPts[field_name],
              common_data_simple_contact->elasticityCommonDataMaster
                  .gradAtGaussPts[field_name]));

      cerr << "Material data Master "
           << "E " << master_block.E << " Poisson " << master_block.PoissonRatio
           << " Element name " << side_fe_name << "\n";

      common_data_simple_contact->elasticityCommonDataMaster.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonDataMaster.meshPositions =
          mesh_node_position;

      common_data_simple_contact->elasticityCommonData.spatialPositions =
          field_name;
      common_data_simple_contact->elasticityCommonData.meshPositions =
          mesh_node_position;

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name,
              common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonDataMaster));

      feMatMasterSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, master_block,
              common_data_simple_contact->elasticityCommonDataMaster,
              common_data_simple_contact->tAg, false, false, false));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpLoopMasterForSide(mesh_node_position, feMatMasterSideRhs,
                                  side_fe_name));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpCalProjStressesAtGaussPtsMaster(mesh_node_position,
                                                common_data_simple_contact));

      cerr << "Material data Slave "
           << "E " << slave_block.E << " Poisson " << slave_block.PoissonRatio
           << " Element name " << side_fe_name << "\n";

      boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnVolumeSide>
          feMatSlaveSideRhs =
              boost::make_shared<VolumeElementForcesAndSourcesCoreOnVolumeSide>(
                  mField);

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetDataAtGaussPts(
              field_name,
              common_data_simple_contact->dataAtGaussPts[field_name],
              common_data_simple_contact->gradAtGaussPts[field_name]));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              field_name, common_data_simple_contact->elasticityCommonData));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
              mesh_node_position,
              common_data_simple_contact->elasticityCommonData));

      feMatSlaveSideRhs->getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, slave_block,
              common_data_simple_contact->elasticityCommonData,
              common_data_simple_contact->tAg, false, false, false));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpLoopSlaveForSide(mesh_node_position, feMatSlaveSideRhs,
                                 side_fe_name));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpCalProjStressesAtGaussPtsSlave(mesh_node_position,
                                               common_data_simple_contact));

      //   if (mField.check_field(material_position_field_name)) {
      //     feRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
      //         material_position_field_name, commonData));
      //   }
      //   std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
      //   for (; sit != setOfBlocks.end(); sit++) {
      //     feRhs.getOpPtrVector().push_back(new
      //     OpJacobianPiolaKirchhoffStress(
      //         spatial_position_field_name, sit->second, commonData, tAg,
      //         false, ale, field_disp));
      //     feRhs.getOpPtrVector().push_back(new OpRhsPiolaKirchhoff(
      //         spatial_position_field_name, sit->second, commonData));
      //   }

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsMaster(field_name,
                                            common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetPositionAtGaussPtsSlave(field_name,
                                           common_data_simple_contact));

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpGetGapSlave(field_name, common_data_simple_contact));

      if (lagrange_field) {
        cerr << "LAGRAAANGE!\n";

        boost::shared_ptr<FaceElementForcesAndSourcesCoreOnVolumeSide>
            feMatSlaveSideRhsL2 =
                boost::make_shared<FaceElementForcesAndSourcesCoreOnVolumeSide>(
                    mField);

        feMatSlaveSideRhsL2->getOpPtrVector().push_back(
            new OpGetLagMulAtGaussPtsSlaveL2(lagrang_field_name,
                                             common_data_simple_contact));

        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpLoopSlaveForSideFace(field_name, feMatSlaveSideRhsL2,
                                       side_face_fe_name));

        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpLagGapProdGaussPtsSlave(lagrang_field_name,
                                          common_data_simple_contact));
        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpCalNormalLagrangeMasterAndSlaveDifference(
                field_name, common_data_simple_contact));

        fe_post_proc_simple_contact->getOpPtrVector().push_back(
            new OpCalRelativeErrorNormalLagrangeMasterAndSlaveDifference(
                field_name, common_data_simple_contact));

        // fe_post_proc_simple_contact->getOpPtrVector().push_back(
        //     new OpLagMultOnVertex(mField, field_name, mb_post_vertex));
      }

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new OpMakeVtkSlave(m_field, field_name, common_data_simple_contact,
                             moab_out, lagrange_field));
    }
    MoFEMFunctionReturn(0);
  }

};

#endif