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

  struct ContactPrismsData {
    Range pRisms; // All boundary surfaces
  };

  map<int, ContactPrismsData>
      setOfContactPrism; ///< maps side set id with appropriate FluxData

    struct SimpleContactElement
        : public MoFEM::ContactPrismElementForcesAndSourcesCore {

      MoFEM::Interface &mField;
      map<int, ContactPrismsData> &setOfContactPrismSpecial;

      
      SimpleContactElement(
          MoFEM::Interface &m_field,
          map<int, ContactPrismsData> &set_of_contact_prism_special)
          : MoFEM::ContactPrismElementForcesAndSourcesCore(m_field), mField(m_field),
            contactCommondataMultiIndex(contact_commondata_multi_index),
            setOfContactPrismSpecial(set_of_contact_prism_special) {
        
            }

      ~SimpleContactElement() {} 

      int getRule(int order) { return 2 * (order - 1); };

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

        // CHKERR mField.modify_finite_element_add_field_data(
        //   element_name, material_position_field_name);
        //;
        // ============================================================================================================

        setOfContactPrism[1].pRisms = range_slave_master_prisms;
        // Adding range_slave_master_prisms to Element element_name
        CHKERR mField.add_ents_to_finite_element_by_type(
            range_slave_master_prisms, MBPRISM, element_name);

        MoFEMFunctionReturn(0);
      }
    };

    double rValue;
    double cnValue;
    boost::shared_ptr<SimpleContactElement> feRhsSimpleContact;
    boost::shared_ptr<SimpleContactElement> feLhsSimpleContact;
    boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
    MoFEM::Interface &mField;

    SimpleContactProblem(MoFEM::Interface &m_field,
                         ContactProblemMultiIndex::ContactCommonData_multiIndex
                             &contact_commondata_multi_index,
                         double &r_value_regular, double &cn_value)
        : mField(m_field),
          contactCommondataMultiIndex(contact_commondata_multi_index),
          rValue(r_value_regular), cnValue(cn_value) {
      commonDataSimpleContact = boost::make_shared<CommonDataSimpleContact>();
      feRhsSimpleContact = boost::make_shared<SimpleContactElement>();
      feLhsSimpleContact = boost::make_shared<SimpleContactElement>();

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


};
#endif