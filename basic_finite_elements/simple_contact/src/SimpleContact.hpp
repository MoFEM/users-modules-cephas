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

#ifdef __cplusplus
extern "C" {
#endif
#include <cblas.h>
// #include <lapack_wrap.h>
// #include <gm_rule.h>
#include <quad.h>
#ifdef __cplusplus
}
#endif
#ifndef __SIMPLE_CONTACT__
#define __SIMPLE_CONTACT__
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

      SimpleContactElement(
          MoFEM::Interface &m_field)
          : MoFEM::ContactPrismElementForcesAndSourcesCore(m_field), mField(m_field) {   
            }

      ~SimpleContactElement() {} 

      int getRule(int order) { return 2 * (order - 1); };

};

      MoFEMErrorCode addContactElement(
          const string element_name, const string field_name,
          const string lagrang_field_name, Range &range_slave_master_prisms,
          bool lagrange_field = true,
          string material_position_field_name = "MESH_NODE_POSITIONS") {
        MoFEMFunctionBegin;

        CHKERR mField.add_finite_element(element_name, MF_ZERO);

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
          CHKERR mField.modify_finite_element_add_field_data(
              element_name, lagrang_field_name);
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

        MoFEMFunctionReturn(0);
      }

      struct CommonDataSimpleContact {

        boost::shared_ptr<MatrixDouble> dispAtGaussPtsMasterPtr;
        boost::shared_ptr<MatrixDouble> dispAtGaussPtsSlavePtr;
        boost::shared_ptr<VectorDouble> lagMultAtGaussPtsPtr;
        boost::shared_ptr<VectorDouble> gapPtr;
        boost::shared_ptr<VectorDouble> tildeCFunPtr;
        boost::shared_ptr<VectorDouble> lambdaGapDiffProductPtr;

        CommonDataSimpleContact() {
          dispAtGaussPtsMasterPtr = boost::make_shared<MatrixDouble>();
          dispAtGaussPtsSlavePtr = boost::make_shared<MatrixDouble>();
          lagMultAtGaussPtsPtr = boost::make_shared<VectorDouble>();
          gapPtr = boost::make_shared<VectorDouble>();
          tildeCFunPtr = boost::make_shared<VectorDouble>();
          lambdaGapDiffProductPtr = boost::make_shared<VectorDouble>();
        }
      };

      double rValue;
      double cnValue;
      boost::shared_ptr<SimpleContactElement> feRhsSimpleContact;
      boost::shared_ptr<SimpleContactElement> feLhsSimpleContact;
      boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
      MoFEM::Interface &mField;

      SimpleContactProblem(MoFEM::Interface &m_field, double &r_value_regular,
                           double &cn_value)
          : mField(m_field), rValue(r_value_regular), cnValue(cn_value) {
        commonDataSimpleContact = boost::make_shared<CommonDataSimpleContact>();
        feRhsSimpleContact =
            boost::make_shared<SimpleContactElement>(mField);
        feLhsSimpleContact =
            boost::make_shared<SimpleContactElement>(mField);
          }

          /// \brief tangents t1 and t2 to face f4 at all gauss points
          struct OpGetNormalSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            OpGetNormalSlave(
                const string field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      field_name, UserDataOperator::OPCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVE),
                  commonDataSimpleContact(common_data_contact) {}

            PetscErrorCode doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
              MoFEMFunctionBegin;

              if (data.getFieldData().size() == 0)
                PetscFunctionReturn(0);

              if (type == MBVERTEX) {

                auto get_tensor_vec = [](VectorDouble &n) {
                  return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
                };

                const double *normal_slave_ptr = &getNormalSlave()[0];

                // CommonDataSimpleContact->normalF3.resize(gp_gp);
                // CommonDataSimpleContact->normalF4.resize(gp_gp);

                

                // auto f4_length_ptr = getFTensor0FromVec(
                //     *CommonDataSimpleContact->spatialNormalF4LengthPtr);
              }

              MoFEMFunctionReturn(0);
            }
          };

          // setup operators for calculation of active set
          MoFEMErrorCode setContactOperatorsActiveSet(string field_name,
                                                      string lagrang_field_name,
                                                      Vec f_ = PETSC_NULL) {
            MoFEMFunctionBegin;
            cout << "Hi 1 from Simple Contact " << endl;

            map<int, SimpleContactPrismsData>::iterator sit =
                setOfSimpleContactPrism.begin();
            for (; sit != setOfSimpleContactPrism.end(); sit++) {
              cout << "HELP!!! Simple " << endl;
              // OpGetNormalSlave
              feRhsSimpleContact->getOpPtrVector().push_back(
                  new OpGetNormalSlave(field_name, commonDataSimpleContact));

              // feRhsSimpleContact->getOpPtrVector().push_back(
              //     new OpGetDispAtGaussPtsMaster(field_name,
              //     CommonDataSimpleContact));

              // feRhsSimpleContact->getOpPtrVector().push_back(
              //     new OpGetDispAtGaussPtsSlave(field_name,
              //     CommonDataSimpleContact));

              // feRhsSimpleContact->getOpPtrVector().push_back(
              //     new ContactProblemSmallDispNoFriction::OpGetGapSlave(
              //         field_name, CommonDataSimpleContact));

              // feRhsSimpleContact->getOpPtrVector().push_back(
              //     new
              //     ContactProblemSmallDispNoFriction::OpGetLagMulAtGaussPtsSlave(
              //         lagrang_field_name, CommonDataSimpleContact));

              // feRhsSimpleContact->getOpPtrVector().push_back(
              //     new ContactProblemSmallDispNoFriction::OpCalFReConMaster(
              //         field_name, CommonDataSimpleContact, f_));

              // feRhsSimpleContact->getOpPtrVector().push_back(
              //     new ContactProblemSmallDispNoFriction::OpCalFReConSlave(
              //         field_name, CommonDataSimpleContact, f_));

              // feRhsSimpleContact->getOpPtrVector().push_back(
              //     new ContactProblemSmallDispNoFriction::
              //         OpCalTildeCFunSlave( // Not
              //                              // tested
              //             field_name, CommonDataSimpleContact, rValue, cnValue));

              // feRhsSimpleContact->getOpPtrVector().push_back(
              //     new
              //     ContactProblemSmallDispNoFriction::OpCalIntTildeCFunSlave(
              //         lagrang_field_name, CommonDataSimpleContact, f_));
            }
            MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode
    setContactOperators(string field_name, string lagrang_field_name, Mat aij) {
      MoFEMFunctionBegin;
      cout << "Hi 1 from setContactOperators " << endl;
      cout << "setOfSimpleContactPrism[1].pRisms " << setOfSimpleContactPrism[1].pRisms
           << endl;

      map<int, SimpleContactPrismsData>::iterator sit = setOfSimpleContactPrism.begin();
      for (; sit != setOfSimpleContactPrism.end(); sit++) {

      }
      MoFEMFunctionReturn(0);
    }
};
#endif