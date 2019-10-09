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

  struct LoadScale : public MethodForForceScaling {

    static double scale;

    MoFEMErrorCode scaleNf(const FEMethod *fe, VectorDouble &nf) {
      MoFEMFunctionBegin;
      nf *= scale;
      MoFEMFunctionReturn(0);
    }
  };

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
          bool lagrange_field = true) {
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

        boost::shared_ptr<MatrixDouble> positionAtGaussPtsMasterPtr;
        boost::shared_ptr<MatrixDouble> positionAtGaussPtsSlavePtr;
        boost::shared_ptr<VectorDouble> lagMultAtGaussPtsPtr;
        boost::shared_ptr<VectorDouble> gapPtr;
        boost::shared_ptr<VectorDouble> tildeCFunPtr;
        boost::shared_ptr<VectorDouble> lambdaGapDiffProductPtr;
        boost::shared_ptr<VectorDouble> normalVectorPtr;
        double areaCommon;

        CommonDataSimpleContact() {
          positionAtGaussPtsMasterPtr = boost::make_shared<MatrixDouble>();
          positionAtGaussPtsSlavePtr = boost::make_shared<MatrixDouble>();
          lagMultAtGaussPtsPtr = boost::make_shared<VectorDouble>();
          gapPtr = boost::make_shared<VectorDouble>();
          tildeCFunPtr = boost::make_shared<VectorDouble>();
          lambdaGapDiffProductPtr = boost::make_shared<VectorDouble>();
          normalVectorPtr = boost::make_shared<VectorDouble>();
          //areaPtr = boost::make_shared<double>();
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

              if (type != MBVERTEX)
                PetscFunctionReturn(0);

                FTensor::Index<'i', 3> i;

                auto get_tensor_vec = [](VectorDouble &n) {
                  return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
                };

                const double *normal_slave_ptr = &getNormalSlave()[0];
                commonDataSimpleContact->normalVectorPtr.get()->resize(3);
                commonDataSimpleContact->normalVectorPtr.get()->clear();

                auto normal =
                    get_tensor_vec(commonDataSimpleContact->normalVectorPtr.get()[0]);
                for (int ii = 0; ii != 3; ++ii)
                  normal(ii) = normal_slave_ptr[ii];

                const double normal_length = pow (normal(i) * normal(i), 2);
                normal(i) = normal(i) / normal_length;
                commonDataSimpleContact->areaCommon = 0.5 * normal_length;
                

                MoFEMFunctionReturn(0);
            }
          };

          // Calculate displacements or spatial positions at Gauss points  (Note
          // OPCOL
          // here, which is dips/sp-pos here)
          struct OpGetPositionAtGaussPtsMaster
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            OpGetPositionAtGaussPtsMaster(
                const string field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      field_name, UserDataOperator::OPCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACEMASTER),
                  commonDataSimpleContact(common_data_contact) {}

            MoFEMErrorCode doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
              MoFEMFunctionBegin;
              const int nb_dofs = data.getFieldData().size();

              if (nb_dofs == 0)
                MoFEMFunctionReturnHot(0);

              const int nb_gauss_pts = data.getN().size1();
              
              if (type == MBVERTEX) {
                commonDataSimpleContact->positionAtGaussPtsMasterPtr.get()->resize(
                    3, nb_gauss_pts, false);

                commonDataSimpleContact->positionAtGaussPtsMasterPtr.get()->clear();
              }

              auto position_master = getFTensor1FromMat<3>(
                  *commonDataSimpleContact->positionAtGaussPtsMasterPtr);

              int nb_base_fun_col = data.getFieldData().size() / 3;

              FTensor::Index<'i', 3> i;

              for (int gg = 0; gg != nb_gauss_pts; gg++) {
                FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

                FTensor::Tensor1<double *, 3> t_field_data_master(
                    &data.getFieldData()[0], &data.getFieldData()[1],
                    &data.getFieldData()[2], 3);

                for (int bb = 0; bb != nb_base_fun_col; bb++) {
                  position_master(i) += t_base_master * t_field_data_master(i);

                  ++t_base_master;
                  ++t_field_data_master;
                }

                ++position_master;
              }

              MoFEMFunctionReturn(0);
            }
          };

          // Calculate displacements or spatial positions at Gauss points  (Note
          // OPCOL
          // here, which is dips/sp-pos here)
          struct OpGetPositionAtGaussPtsSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            OpGetPositionAtGaussPtsSlave(
                const string field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      field_name, UserDataOperator::OPCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVE),
                  commonDataSimpleContact(common_data_contact) {}

            MoFEMErrorCode doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
              MoFEMFunctionBegin;
              const int nb_dofs = data.getFieldData().size();

              if (nb_dofs == 0)
                MoFEMFunctionReturnHot(0);

              const int nb_gauss_pts = data.getN().size1();
              if (type == MBVERTEX) {
                commonDataSimpleContact->positionAtGaussPtsSlavePtr.get()->resize(
                    3, nb_gauss_pts, false);
                commonDataSimpleContact->positionAtGaussPtsSlavePtr.get()->clear();
              }

              auto position_slave = getFTensor1FromMat<3>(
                  *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

              int nb_base_fun_col = data.getFieldData().size() / 3;

              FTensor::Index<'i', 3> i;

              for (int gg = 0; gg != nb_gauss_pts; gg++) {
                FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));

                FTensor::Tensor1<double *, 3> t_field_data_slave(
                    &data.getFieldData()[0], &data.getFieldData()[1],
                    &data.getFieldData()[2], 3); // in-between

                for (int bb = 0; bb != nb_base_fun_col; bb++) {
                  position_slave(i) += t_base_slave * t_field_data_slave(i);

                  ++t_base_slave;
                  ++t_field_data_slave;
                }

                ++position_slave;
              }

              MoFEMFunctionReturn(0);
            }
          };

          struct OpGetGapSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;

            OpGetGapSlave(
                const string field_name, // ign: does it matter??
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      field_name, UserDataOperator::OPROW,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVE),
                  commonDataSimpleContact(common_data_contact) {}

            MoFEMErrorCode doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
              MoFEMFunctionBegin;

              if (data.getFieldData().size() == 0)
                PetscFunctionReturn(0);

              if (type != MBVERTEX)
                MoFEMFunctionReturnHot(0);

              auto get_tensor_vec = [](VectorDouble &n) {
                return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
              };

              const int nb_gauss_pts = data.getN().size1();

              commonDataSimpleContact->gapPtr.get()->resize(nb_gauss_pts);
              commonDataSimpleContact->gapPtr.get()->clear();

              FTensor::Index<'i', 3> i;

              auto position_master_gp = getFTensor1FromMat<3>(
                  *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
              auto position_slave_gp = getFTensor1FromMat<3>(
                  *commonDataSimpleContact->positionAtGaussPtsSlavePtr);

              auto gap_ptr = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

              auto normal_at_gp = get_tensor_vec(
                  commonDataSimpleContact->normalVectorPtr.get()[0]);
                  
              for (int gg = 0; gg != nb_gauss_pts; gg++) {
                gap_ptr -= normal_at_gp(i) *
                           (position_slave_gp(i) - position_master_gp(i));

                ++position_slave_gp;
                ++position_master_gp;
                ++gap_ptr;
              } // for gauss points
              
              MoFEMFunctionReturn(0);
            }
          };

          struct OpGetLagMulAtGaussPtsSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            OpGetLagMulAtGaussPtsSlave(
                const string lagrang_field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      lagrang_field_name, UserDataOperator::OPROW,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVE),
                  commonDataSimpleContact(common_data_contact) {}

            MoFEMErrorCode doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
              MoFEMFunctionBegin;

              const int nb_gauss_pts = data.getN().size1();

              if (type == MBVERTEX) {
                commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->resize(
                    nb_gauss_pts);
                commonDataSimpleContact->lagMultAtGaussPtsPtr.get()->clear();
              }

              int nb_base_fun_row = data.getFieldData().size();

              auto lagrange_slave =
                  getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

              for (int gg = 0; gg != nb_gauss_pts; gg++) {
                FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

                FTensor::Tensor0<double *> t_field_data_slave(
                    &data.getFieldData()[0]);
                for (int bb = 0; bb != nb_base_fun_row; bb++) {
                  lagrange_slave += t_base_lambda * t_field_data_slave;
                  ++t_base_lambda;
                  ++t_field_data_slave;
                }
                ++lagrange_slave;
              }
            
              MoFEMFunctionReturn(0);
            }
          };

          struct OpCalFReConMaster
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            Vec F;
            OpCalFReConMaster(
                const string field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
                Vec f_ = PETSC_NULL)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      field_name, UserDataOperator::OPCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACEMASTER),
                  commonDataSimpleContact(common_data_contact), F(f_) {}

            VectorDouble vec_f;
            PetscErrorCode doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
              PetscFunctionBegin;

              if (data.getIndices().size() == 0)
                PetscFunctionReturn(0);

              const int nb_gauss_pts = data.getN().size1();
              int nb_base_fun_col = data.getFieldData().size() / 3;

              vec_f.resize(3 * nb_base_fun_col,
                           false); // the last false in ublas
                                   // resize will destroy (not
                                   // preserved) the old
                                   // values
              vec_f.clear();

              const double area_m =
                  commonDataSimpleContact
                      ->areaCommon; // same area in master and slave

              auto get_tensor_vec = [](VectorDouble &n, const int r) {
                return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1),
                                                     &n(r + 2));
              };

              FTensor::Index<'i', 3> i;

              auto lagrange_slave =
                  getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

              auto const_unit_n = get_tensor_vec(
                  commonDataSimpleContact->normalVectorPtr.get()[0], 0);

              for (int gg = 0; gg != nb_gauss_pts; ++gg) {

                double val_m = getGaussPtsMaster()(2, gg) * area_m;

                FTensor::Tensor0<double *> t_base_master(&data.getN()(gg, 0));

                for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
                  const double m = val_m * t_base_master * lagrange_slave;
                  auto t_assemble_m = get_tensor_vec(vec_f, 3 * bbc);
                  t_assemble_m(i) -= m * const_unit_n(i);

                  ++t_base_master;
                }

                ++lagrange_slave;
              } // for gauss points

              const int nb_col = data.getIndices().size();

              CHKERR VecSetValues(getFEMethod()->snes_f, nb_col,
                                  &data.getIndices()[0], &vec_f[0], ADD_VALUES);
              PetscFunctionReturn(0);
            }
          };

          struct OpCalFReConSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            Vec F;
            OpCalFReConSlave(
                const string field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
                Vec f_ = PETSC_NULL)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      field_name, UserDataOperator::OPCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVE),
                  commonDataSimpleContact(common_data_contact), F(f_) {}

            VectorDouble vec_f;
            PetscErrorCode doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
              PetscFunctionBegin;

              if (data.getIndices().size() == 0)
                PetscFunctionReturn(0);

              const int nb_gauss_pts = data.getN().size1();
              int nb_base_fun_col = data.getFieldData().size() / 3;

              vec_f.resize(3 * nb_base_fun_col,
                           false); // the last false in ublas
                                   // resize will destroy (not
                                   // preserved) the old
                                   // values
              vec_f.clear();

              const double *normal_f4_ptr = &getNormalSlave()[0];

              const double area_s =
                  commonDataSimpleContact->areaCommon; // same area in master and slave
              
              auto get_tensor_vec = [](VectorDouble &n, const int r) {
                return FTensor::Tensor1<double *, 3>(&n(r + 0), &n(r + 1),
                                                     &n(r + 2));
              };

              FTensor::Index<'i', 3> i;
              auto lagrange_slave =
                  getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

              auto const_unit_n = get_tensor_vec(
                  commonDataSimpleContact->normalVectorPtr.get()[0], 0);

              for (int gg = 0; gg != nb_gauss_pts; ++gg) {

                double val_s = getGaussPtsSlave()(2, gg) * area_s;

                FTensor::Tensor0<double *> t_base_slave(&data.getN()(gg, 0));
  
                for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

                  const double s = val_s * t_base_slave * lagrange_slave;

                  auto t_assemble_s = get_tensor_vec(vec_f, 3 * bbc);

                  t_assemble_s(i) += s * const_unit_n(i);
                  ++t_base_slave;
                }

                ++lagrange_slave;
              } // for gauss points

              const int nb_col = data.getIndices().size();
          
              CHKERR VecSetValues(getFEMethod()->snes_f, nb_col,
                                  &data.getIndices()[0], &vec_f[0], ADD_VALUES);
              PetscFunctionReturn(0);
            }
          };

          struct OpCalTildeCFunSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            double r;  //@todo: ign: to become input parameter
            double cN; //@todo: ign: to become input parameter

            OpCalTildeCFunSlave(
                const string lagrang_field_name, // ign: does it matter?
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
                double &r_value, double &cn_value)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      lagrang_field_name, UserDataOperator::OPROW,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVE),
                  commonDataSimpleContact(common_data_contact), r(r_value),
                  cN(cn_value) {}

            MoFEMErrorCode doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
              MoFEMFunctionBegin;

              if (data.getFieldData().size() == 0)
                PetscFunctionReturn(0);

              if (type != MBVERTEX)
                MoFEMFunctionReturnHot(0);

              const int nb_gauss_pts = data.getN().size1();

              commonDataSimpleContact->tildeCFunPtr.get()->resize(nb_gauss_pts);
              commonDataSimpleContact->tildeCFunPtr.get()->clear();

              commonDataSimpleContact->lambdaGapDiffProductPtr.get()->resize(
                  nb_gauss_pts);
              commonDataSimpleContact->lambdaGapDiffProductPtr.get()->clear();

              auto lambda_gap_diff_prod = getFTensor0FromVec(
                  *commonDataSimpleContact->lambdaGapDiffProductPtr);

              auto lagrange_slave =
                  getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

              auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

              auto tilde_c_fun =
                  getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

              for (int gg = 0; gg != nb_gauss_pts; gg++) {
                const double cg = cN * gap_gp;

                const double lambda_gap_diff = lagrange_slave - cg;
                const double regular_abs = abs(lambda_gap_diff);

                tilde_c_fun = (lagrange_slave + cg -
                               pow(regular_abs, r) / r); // is lagMult Correct?

                const double exponent = r - 1.;

                double sign = 0.;
                sign =
                    (lambda_gap_diff == 0) ? 0 : (lambda_gap_diff < 0) ? -1 : 1;

                // if (lagrange_slave == 0. && cg == 0.)
                // sign = 1.;
                lambda_gap_diff_prod = sign * pow(regular_abs, exponent);
                ++lagrange_slave;
                ++gap_gp;
                ++lambda_gap_diff_prod;
                ++tilde_c_fun;
              }
              MoFEMFunctionReturn(0);
            }
          };

          struct OpCalIntTildeCFunSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            Vec F;
            OpCalIntTildeCFunSlave(
                const string lagrang_field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
                Vec f_ = PETSC_NULL)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      lagrang_field_name, UserDataOperator::OPCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVE),
                  commonDataSimpleContact(common_data_contact), F(f_) {}

            VectorDouble vecR;
            MoFEMErrorCode doWork(int side, EntityType type,
                                  DataForcesAndSourcesCore::EntData &data) {
              MoFEMFunctionBegin;

              if (data.getIndices().size() == 0)
                MoFEMFunctionReturnHot(0);

              const int nb_gauss_pts = data.getN().size1();

              int nb_base_fun_col = data.getFieldData().size();
              const double area_s =
                  commonDataSimpleContact
                      ->areaCommon; // same area in master and slave

              vecR.resize(nb_base_fun_col, false); // the last false in ublas
                                                   // resize will destroy (not
                                                   // preserved) the old values
              vecR.clear();

              auto tilde_c_fun =
                  getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

              for (int gg = 0; gg != nb_gauss_pts; gg++) {
                double val_s = getGaussPtsSlave()(2, gg) * area_s;
                
                FTensor::Tensor0<double *> t_base_lambda(&data.getN()(gg, 0));

                for (int bbr = 0; bbr != nb_base_fun_col; bbr++) {
                  const double s = val_s * t_base_lambda * tilde_c_fun;
                  vecR[bbr] += s;  // it is a plus since F always scales with -1
                  ++t_base_lambda; // update rows
                }
                ++tilde_c_fun;
              } // for gauss points

              CHKERR VecSetValues(getFEMethod()->snes_f, nb_base_fun_col,
                                  &data.getIndices()[0], &vecR[0], ADD_VALUES);
              MoFEMFunctionReturn(0);
            }
          };

          struct OpContactConstraintMatrixMasterSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            Mat Aij;
            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            OpContactConstraintMatrixMasterSlave(
                const string field_name, const string lagrang_field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
                Mat aij = PETSC_NULL)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      field_name, lagrang_field_name,
                      UserDataOperator::OPROWCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACEMASTERSLAVE),
                  commonDataSimpleContact(common_data_contact), Aij(aij) {
              sYmm =
                  false; // This will make sure to loop over all intities (e.g.
                         // for order=2 it will make doWork to loop 16 time)
            }
            MatrixDouble NN;

            MoFEMErrorCode doWork(int row_side, int col_side,
                                  EntityType row_type, EntityType col_type,
                                  DataForcesAndSourcesCore::EntData &row_data,
                                  DataForcesAndSourcesCore::EntData &col_data) {
              MoFEMFunctionBegin;

              // Both sides are needed since both sides contribute their shape
              // function to the stiffness matrix
              const int nb_row = row_data.getIndices().size();
              if (!nb_row)
                MoFEMFunctionReturnHot(0);
              const int nb_col = col_data.getIndices().size();
              if (!nb_col)
                MoFEMFunctionReturnHot(0);
              const int nb_gauss_pts = row_data.getN().size1();

              int nb_base_fun_row = row_data.getFieldData().size() / 3;
              int nb_base_fun_col = col_data.getFieldData().size();

              const double area_common =
                  commonDataSimpleContact
                      ->areaCommon; // same area in master and slave

              auto get_tensor_from_mat = [](MatrixDouble &m, const int r,
                                            const int c) {
                return FTensor::Tensor1<double *, 3>(
                    &m(r + 0, c + 0), &m(r + 1, c + 0), &m(r + 2, c + 0));
              };

              auto get_tensor_vec = [](VectorDouble &n) {
                return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
              };

              FTensor::Index<'i', 3> i;

              NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
              NN.clear();

              auto const_unit_n = get_tensor_vec(
                  commonDataSimpleContact->normalVectorPtr.get()[0]);

              for (int gg = 0; gg != nb_gauss_pts; gg++) {
                double val_m = getGaussPtsMaster()(2, gg) * area_common;
     
                FTensor::Tensor0<double *> t_base_lambda(
                    &col_data.getN()(gg, 0));

                for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
                  FTensor::Tensor0<double *> t_base_master(
                      &row_data.getN()(gg, 0));

                  for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
                    const double m = val_m * t_base_lambda * t_base_master;
                 
                    auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, bbc);
                    t_assemble_m(i) -= m * const_unit_n(i);

                    ++t_base_master; // update rows master
                  }
                  ++t_base_lambda; // update cols slave
                }
              }

              if (Aij == PETSC_NULL) {
                Aij = getFEMethod()->snes_B;
              }

              CHKERR MatSetValues(getFEMethod()->snes_B, nb_row,
                                  &row_data.getIndices()[0], nb_col,
                                  &col_data.getIndices()[0],
                                  &*NN.data().begin(), ADD_VALUES);

              MoFEMFunctionReturn(0);
            }
          };

          struct OpContactConstraintMatrixSlaveSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            Mat Aij;
            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            OpContactConstraintMatrixSlaveSlave(
                const string field_name, const string lagrang_field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
                Mat aij = PETSC_NULL)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      field_name, lagrang_field_name,
                      UserDataOperator::OPROWCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVESLAVE),
                  commonDataSimpleContact(common_data_contact), Aij(aij) {
              sYmm =
                  false; // This will make sure to loop over all intities (e.g.
                         // for order=2 it will make doWork to loop 16 time)
            }
            MatrixDouble NN;

            MoFEMErrorCode doWork(int row_side, int col_side,
                                  EntityType row_type, EntityType col_type,
                                  DataForcesAndSourcesCore::EntData &row_data,
                                  DataForcesAndSourcesCore::EntData &col_data) {
              MoFEMFunctionBegin;

              // Both sides are needed since both sides contribute their shape
              // function to the stiffness matrix
              const int nb_row = row_data.getIndices().size();
              if (!nb_row)
                MoFEMFunctionReturnHot(0);
              const int nb_col = col_data.getIndices().size();
              if (!nb_col)
                MoFEMFunctionReturnHot(0);
              const int nb_gauss_pts = row_data.getN().size1();

              int nb_base_fun_row = row_data.getFieldData().size() / 3;
              int nb_base_fun_col = col_data.getFieldData().size();
              
              const double area_common =
                  commonDataSimpleContact
                      ->areaCommon; // same area in master and slave

              auto get_tensor_from_mat = [](MatrixDouble &m, const int r,
                                            const int c) {
                return FTensor::Tensor1<double *, 3>(
                    &m(r + 0, c + 0), &m(r + 1, c + 0), &m(r + 2, c + 0));
              };

              auto get_tensor_vec = [](VectorDouble &n) {
                return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
              };

              FTensor::Index<'i', 3> i;

              NN.resize(3 * nb_base_fun_row, nb_base_fun_col, false);
              NN.clear();

              auto const_unit_n = get_tensor_vec(
                  commonDataSimpleContact->normalVectorPtr.get()[0]);

              for (int gg = 0; gg != nb_gauss_pts; gg++) {
                double val_s = getGaussPtsSlave()(2, gg) * area_common;

                FTensor::Tensor0<double *> t_base_lambda(
                    &col_data.getN()(gg, 0));
           
                for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {

                  FTensor::Tensor0<double *> t_base_slave(
                      &row_data.getN()(gg, 0));

                  for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
                    const double s = val_s * t_base_lambda * t_base_slave;

                    auto t_assemble_s = get_tensor_from_mat(NN, 3 * bbr, bbc);

                    t_assemble_s(i) += s * const_unit_n(i);

                    ++t_base_slave; // update rows
                  }
                  ++t_base_lambda; // update cols slave
                }
              }

              if (Aij == PETSC_NULL) {
                Aij = getFEMethod()->snes_B;
              }

              CHKERR MatSetValues(getFEMethod()->snes_B, nb_row,
                                  &row_data.getIndices()[0], nb_col,
                                  &col_data.getIndices()[0],
                                  &*NN.data().begin(), ADD_VALUES);

              MoFEMFunctionReturn(0);
            }
          };

          struct OpDerivativeBarTildeCFunOLambdaSlaveSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            Mat Aij;
            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            OpDerivativeBarTildeCFunOLambdaSlaveSlave(
                const string lagrang_field_name,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
                Mat aij = PETSC_NULL)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      lagrang_field_name, UserDataOperator::OPROWCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVESLAVE),
                  Aij(aij), commonDataSimpleContact(common_data_contact) {
              sYmm =
                  false; // This will make sure to loop over all entities (e.g.
                         // for order=2 it will make doWork to loop 16 time)
            }
            MatrixDouble NN;
            MoFEMErrorCode doWork(int row_side, int col_side,
                                  EntityType row_type, EntityType col_type,
                                  DataForcesAndSourcesCore::EntData &row_data,
                                  DataForcesAndSourcesCore::EntData &col_data) {
              MoFEMFunctionBegin;

              const int nb_row = row_data.getIndices().size();
              if (!nb_row)
                MoFEMFunctionReturnHot(0);
              const int nb_col = col_data.getIndices().size();
              if (!nb_col)
                MoFEMFunctionReturnHot(0);
              const int nb_gauss_pts = row_data.getN().size1();

              int nb_base_fun_row = row_data.getFieldData().size();

              int nb_base_fun_col = col_data.getFieldData().size();

              const double area_common =
                  commonDataSimpleContact
                      ->areaCommon; // same area in master and slave

              NN.resize(nb_base_fun_row, nb_base_fun_col,
                        false); // the last false in ublas resize will destroy
                                // (not preserved) the old values
              NN.clear();

              auto lagrange_slave =
                  getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);
              auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

              auto lambda_gap_diff_prod = getFTensor0FromVec(
                  *commonDataSimpleContact->lambdaGapDiffProductPtr);

              for (int gg = 0; gg != nb_gauss_pts; gg++) {
                const double val_s = getGaussPtsSlave()(2, gg) * area_common;

                FTensor::Tensor0<double *> t_base_lambda_col(
                    &col_data.getN()(gg, 0));

                for (int bbc = 0; bbc != nb_base_fun_col; bbc++) {
                  FTensor::Tensor0<double *> t_base_lambda_row(
                      &row_data.getN()(gg, 0));
                  for (int bbr = 0; bbr != nb_base_fun_row; bbr++) {
                    if (bbr == bbc) {
                      if (fabs(gap_gp) < 1.e-8 &&
                          fabs(lagrange_slave) < 1.e-8) {
                        NN(bbr, bbc) = 0;
                      } else {
                        NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                                        t_base_lambda_row * t_base_lambda_col;
                      }

                    } else {
                      if (fabs(gap_gp) < 1.e-8 &&
                          fabs(lagrange_slave) < 1.e-8) {
                      } else {
                        NN(bbr, bbc) += (1. - lambda_gap_diff_prod) * val_s *
                                        t_base_lambda_row * t_base_lambda_col;
                      }
                    }
                    ++t_base_lambda_row; // update rows
                  }
                  ++t_base_lambda_col; // update cols
                }
                ++lagrange_slave;
                ++gap_gp;
                ++lambda_gap_diff_prod;
              }

              CHKERR MatSetValues(
                  getFEMethod()->snes_B, nb_base_fun_row,
                  &row_data.getIndices()[0],
                  nb_base_fun_col, // ign: is shift row right here?
                  &col_data.getIndices()[0], &*NN.data().begin(), ADD_VALUES);

              MoFEMFunctionReturn(0);
            }
          };

          struct OpDerivativeBarTildeCFunODisplacementsSlaveMaster
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            double cN; //@todo: ign: to become input parameter
            Mat Aij;
            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            OpDerivativeBarTildeCFunODisplacementsSlaveMaster(
                const string field_name, const string lagrang_field_name,
                double &cn_value,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
                Mat aij = PETSC_NULL)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      lagrang_field_name, field_name,
                      UserDataOperator::OPROWCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVEMASTER),
                  cN(cn_value), Aij(aij),
                  commonDataSimpleContact(common_data_contact) {
              sYmm =
                  false; // This will make sure to loop over all entities (e.g.
                         // for order=2 it will make doWork to loop 16 time)
            }
            MatrixDouble NN;
            MoFEMErrorCode doWork(int row_side, int col_side,
                                  EntityType row_type, EntityType col_type,
                                  DataForcesAndSourcesCore::EntData &row_data,
                                  DataForcesAndSourcesCore::EntData &col_data) {
              MoFEMFunctionBegin;

              const int nb_row = row_data.getIndices().size();
              if (!nb_row)
                MoFEMFunctionReturnHot(0);
              const int nb_col = col_data.getIndices().size();
              if (!nb_col)
                MoFEMFunctionReturnHot(0);
              const int nb_gauss_pts = row_data.getN().size1();

              VectorDouble3 n4 = getNormalSlave();
              VectorDouble3 n4_unit(3);
              noalias(n4_unit) = n4 / norm_2(n4);

              VectorDouble3 n3 = getNormalMaster();
              VectorDouble3 n3_unit(3);
              noalias(n3_unit) = n3 / norm_2(n3);

              int nb_base_fun_row = row_data.getFieldData().size();

              int nb_base_fun_col = col_data.getFieldData().size() / 3;

              const double area_common =
                  commonDataSimpleContact
                      ->areaCommon; // same area in master and slave

              const double mesh_dot_prod =
                  cblas_ddot(3, &n4_unit[0], 1, &n3_unit[0], 1);

              NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
              NN.clear();
              
              auto get_tensor_from_vec = [](VectorDouble &n) {
                return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
              };

              auto get_vec_from_mat = [](MatrixDouble &m, const int r,
                                         const int c) {
                return FTensor::Tensor1<double *, 3>(
                    &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2));
              };

              FTensor::Index<'i', 3> i;
              
              auto lagrange_slave =
                  getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

              auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

              auto lambda_gap_diff_prod = getFTensor0FromVec(
                  *commonDataSimpleContact->lambdaGapDiffProductPtr);
              
              auto tilde_c_fun =
                  getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

              auto const_unit_n = get_tensor_from_vec(
                  commonDataSimpleContact->normalVectorPtr.get()[0]);

              for (int gg = 0; gg != nb_gauss_pts; ++gg) {
                double val_m = getGaussPtsMaster()(2, gg) * area_common;

               FTensor::Tensor0<double *> t_base_master(
                    &col_data.getN()(gg, 0));
                                
                FTensor::Tensor1<double *, 3> t_field_data_master(
                    &col_data.getFieldData()[0], &col_data.getFieldData()[1],
                    &col_data.getFieldData()[2], 3);

                for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
                  FTensor::Tensor0<double *> t_base_lambda(
                      &row_data.getN()(gg, 0));
                
                  for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
                    const double m = val_m * t_base_lambda * t_base_master;

                    auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

                    assemble_mat(i) += const_unit_n(i) * cN *
                                       (1. + lambda_gap_diff_prod) * m;

                    ++t_base_lambda; // update rows
                  }

                  ++t_base_master; // update cols master
                  
                  ++t_field_data_master;
                }

                ++tilde_c_fun;
                ++lagrange_slave;
                ++gap_gp;
                ++lambda_gap_diff_prod;
              }

              // Assemble NN to final Aij vector based on its global indices
              CHKERR MatSetValues(getFEMethod()->snes_B, nb_base_fun_row,
                                  &row_data.getIndices()[0], nb_col,
                                  &col_data.getIndices()[0],
                                  &*NN.data().begin(), ADD_VALUES);

              MoFEMFunctionReturn(0);
            }
          };

          struct OpDerivativeBarTildeCFunODisplacementsSlaveSlave
              : public ContactPrismElementForcesAndSourcesCore::
                    UserDataOperator {

            double cN; //@todo: ign: to become input parameter
            Mat Aij;
            boost::shared_ptr<CommonDataSimpleContact> commonDataSimpleContact;
            OpDerivativeBarTildeCFunODisplacementsSlaveSlave(
                const string field_name, const string lagrang_field_name,
                double &cn_value,
                boost::shared_ptr<CommonDataSimpleContact> &common_data_contact,
                Mat aij = PETSC_NULL)
                : ContactPrismElementForcesAndSourcesCore::UserDataOperator(
                      lagrang_field_name, field_name,
                      UserDataOperator::OPROWCOL,
                      ContactPrismElementForcesAndSourcesCore::
                          UserDataOperator::FACESLAVESLAVE),
                  cN(cn_value), Aij(aij),
                  commonDataSimpleContact(common_data_contact) {
              sYmm =
                  false; // This will make sure to loop over all entities (e.g.
                         // for order=2 it will make doWork to loop 16 time)
            }
            MatrixDouble NN;
            MoFEMErrorCode doWork(int row_side, int col_side,
                                  EntityType row_type, EntityType col_type,
                                  DataForcesAndSourcesCore::EntData &row_data,
                                  DataForcesAndSourcesCore::EntData &col_data) {
              MoFEMFunctionBegin;

              const int nb_row = row_data.getIndices().size();
              if (!nb_row)
                MoFEMFunctionReturnHot(0);
              const int nb_col = col_data.getIndices().size();
              if (!nb_col)
                MoFEMFunctionReturnHot(0);
              const int nb_gauss_pts = row_data.getN().size1();
              int nb_base_fun_row = row_data.getFieldData().size();
              int nb_base_fun_col = col_data.getFieldData().size() / 3;

              const double area_common =
                  commonDataSimpleContact
                      ->areaCommon; // same area in master and slave

              NN.resize(nb_base_fun_row, 3 * nb_base_fun_col, false);
              NN.clear();

              auto get_tensor_vec = [](VectorDouble &n) {
                return FTensor::Tensor1<double *, 3>(&n(0), &n(1), &n(2));
              };

              auto get_vec_from_mat = [](MatrixDouble &m, const int r,
                                         const int c) {
                return FTensor::Tensor1<double *, 3>(
                    &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2));
              };

              FTensor::Index<'i', 3> i;
              
              auto x_m = getFTensor1FromMat<3>(
                  *commonDataSimpleContact->positionAtGaussPtsMasterPtr);
              auto x_s = getFTensor1FromMat<3>(
                  *commonDataSimpleContact->positionAtGaussPtsSlavePtr);
              auto lagrange_slave =
                  getFTensor0FromVec(*commonDataSimpleContact->lagMultAtGaussPtsPtr);

              auto gap_gp = getFTensor0FromVec(*commonDataSimpleContact->gapPtr);

              auto lambda_gap_diff_prod = getFTensor0FromVec(
                  *commonDataSimpleContact->lambdaGapDiffProductPtr);

              auto tilde_c_fun =
                  getFTensor0FromVec(*commonDataSimpleContact->tildeCFunPtr);

              auto const_unit_n = get_tensor_vec(
                  commonDataSimpleContact->normalVectorPtr.get()[0]);

              for (int gg = 0; gg != nb_gauss_pts; ++gg) {
                double val_s = getGaussPtsSlave()(2, gg) * area_common;

                FTensor::Tensor0<double *> t_base_slave(
                    &col_data.getN()(gg, 0));

                for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
                  FTensor::Tensor0<double *> t_base_lambda(
                      &row_data.getN()(gg, 0));

                  for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {
                    const double s = val_s * t_base_lambda * t_base_slave;

                    auto assemble_mat = get_vec_from_mat(NN, bbr, 3 * bbc);

                    assemble_mat(i) -=
                        const_unit_n(i) * cN * (1. + lambda_gap_diff_prod) * s;
                    
                    ++t_base_lambda; // update rows
                  }

                  ++t_base_slave; // update cols slave
                }

                ++x_m;
                ++x_s;
                ++lagrange_slave;
                ++gap_gp;
                ++lambda_gap_diff_prod;
                ++tilde_c_fun;
              }

              // Assemble NN to final Aij vector based on its global indices
              CHKERR MatSetValues(getFEMethod()->snes_B, nb_base_fun_row,
                                  &row_data.getIndices()[0], nb_col,
                                  &col_data.getIndices()[0],
                                  &*NN.data().begin(), ADD_VALUES);

              MoFEMFunctionReturn(0);
            }
          };

          // setup operators for calculation of active set
          MoFEMErrorCode setContactOperatorsRhsOperators(string field_name,
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
                  new OpGetGapSlave(
                      field_name, commonDataSimpleContact));

              feRhsSimpleContact->getOpPtrVector().push_back(
                  new
                  OpGetLagMulAtGaussPtsSlave(
                      lagrang_field_name, commonDataSimpleContact));

              feRhsSimpleContact->getOpPtrVector().push_back(
                  new OpCalFReConMaster(
                      field_name, commonDataSimpleContact, f_));

              feRhsSimpleContact->getOpPtrVector().push_back(
                  new OpCalFReConSlave(
                      field_name, commonDataSimpleContact, f_));

              feRhsSimpleContact->getOpPtrVector().push_back(
                  new OpCalTildeCFunSlave(field_name, commonDataSimpleContact, rValue, cnValue));

              feRhsSimpleContact->getOpPtrVector().push_back(
                  new OpCalIntTildeCFunSlave(
                      lagrang_field_name, commonDataSimpleContact, f_));
            }
            MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode setContactOperatorsLhsOperators(string field_name,
                                                   string lagrang_field_name,
                                                   Mat aij) {
      MoFEMFunctionBegin;
      
      map<int, SimpleContactPrismsData>::iterator sit = setOfSimpleContactPrism.begin();
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
                field_name, lagrang_field_name, cnValue,
                commonDataSimpleContact, aij));

        feLhsSimpleContact->getOpPtrVector().push_back(
            new OpDerivativeBarTildeCFunODisplacementsSlaveSlave(
                field_name, lagrang_field_name, cnValue,
                commonDataSimpleContact, aij));
      }
      MoFEMFunctionReturn(0);
    }
};
#endif