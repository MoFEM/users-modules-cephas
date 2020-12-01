/** \file arbitrary_smooth_contact_surface.cpp
 * \example arbitrary_smooth_contact_surface.cpp
 *
 * Implementation of mortar contact between surfaces with matching meshes
 *
 **/

/* MoFEM is free software: you can redistribute it and/or modify it under
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
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>.
 */

#include <MoFEM.hpp>

using namespace MoFEM;

static char help[] = "...\n\n";

#include <BasicFiniteElements.hpp>

using EntData = DataForcesAndSourcesCore::EntData;
using FaceEle = FaceElementForcesAndSourcesCore;
using FaceEleOp = FaceEle::UserDataOperator;
using VolumeEle = VolumeElementForcesAndSourcesCore;
using VolumeEleOp = VolumeEle::UserDataOperator;

struct ArbitrarySmoothingProblem {

Vec volumeVec;
double cnVaule;

  struct SurfaceSmoothingElement : public FaceEle {
    MoFEM::Interface &mField;

    SurfaceSmoothingElement(MoFEM::Interface &m_field)
        : FaceEle(m_field), mField(m_field) {}

  int getRule(int order) { return 4 * (order - 1) ; };

    // Destructor
    ~SurfaceSmoothingElement() {}
  };

  struct VolumeSmoothingElement : public VolumeEle {
    MoFEM::Interface &mField;

    VolumeSmoothingElement(MoFEM::Interface &m_field)
        : VolumeEle(m_field), mField(m_field) {}

    // Destructor
    ~VolumeSmoothingElement() {}
  };

  struct CommonSurfaceSmoothingElement {

    boost::shared_ptr<MatrixDouble> nOrmal;
    boost::shared_ptr<MatrixDouble> tAngent1;
    boost::shared_ptr<MatrixDouble> tAngent2;

    boost::shared_ptr<MatrixDouble> tAngent1Unit;
    boost::shared_ptr<MatrixDouble> tAngent2Unit;

    boost::shared_ptr<MatrixDouble> nOrmalField;

    boost::shared_ptr<MatrixDouble> tAngentOneField;
    boost::shared_ptr<MatrixDouble> tAngentTwoField;
    
    boost::shared_ptr<MatrixDouble> tAnTanOneOneField;
    boost::shared_ptr<MatrixDouble> tAnTanOneTwoField;
    boost::shared_ptr<MatrixDouble> tAnTanTwoOneField;
    boost::shared_ptr<MatrixDouble> tAnTanTwoTwoField;

    boost::shared_ptr<MatrixDouble> curvatuersField;

    boost::shared_ptr<MatrixDouble> tAngentOneGradField;
    boost::shared_ptr<MatrixDouble> tAngentTwoGradField;
    boost::shared_ptr<MatrixDouble> metricTensorForField;
    boost::shared_ptr<VectorDouble> detOfMetricTensorForField;

    boost::shared_ptr<MatrixDouble> tanHdiv1;
    boost::shared_ptr<MatrixDouble> tanHdiv2;

    boost::shared_ptr<MatrixDouble> nOrmalHdiv;
    boost::shared_ptr<VectorDouble> divNormalHdiv;
    
    boost::shared_ptr<VectorDouble> divNormalField;

    boost::shared_ptr<VectorDouble> divTangentOneField;

    boost::shared_ptr<MatrixDouble> rotTangentOneField;

    boost::shared_ptr<VectorDouble> divTangentTwoField;

    boost::shared_ptr<MatrixDouble> matForDivTan2;
    
    boost::shared_ptr<MatrixDouble> matForDivTan1;
    boost::shared_ptr<MatrixDouble> lagMult;

    boost::shared_ptr<VectorDouble> areaNL;

    CommonSurfaceSmoothingElement(MoFEM::Interface &m_field) : mField(m_field) {
      nOrmal = boost::make_shared<MatrixDouble>();
      nOrmalHdiv = boost::make_shared<MatrixDouble>();
      tanHdiv1 = boost::make_shared<MatrixDouble>();
      tanHdiv2 = boost::make_shared<MatrixDouble>();
      
      divNormalHdiv = boost::make_shared<VectorDouble>();
      lagMult = boost::make_shared<MatrixDouble>();

      tAngent1 = boost::make_shared<MatrixDouble>();
      tAngent2 = boost::make_shared<MatrixDouble>();
      tAngent1Unit = boost::make_shared<MatrixDouble>();
      tAngent2Unit = boost::make_shared<MatrixDouble>();


      matForDivTan1 = boost::make_shared<MatrixDouble>();;
      matForDivTan2 = boost::make_shared<MatrixDouble>();;

      nOrmalField = boost::make_shared<MatrixDouble>();
      divNormalField = boost::make_shared<VectorDouble>();
      areaNL = boost::make_shared<VectorDouble>();

      tAngentOneField = boost::make_shared<MatrixDouble>();
      tAngentTwoField = boost::make_shared<MatrixDouble>();
      divTangentOneField = boost::make_shared<VectorDouble>();
      divTangentTwoField = boost::make_shared<VectorDouble>();
      tAngentOneGradField = boost::make_shared<MatrixDouble>();
      tAngentTwoGradField = boost::make_shared<MatrixDouble>();

      tAnTanOneOneField = boost::make_shared<MatrixDouble>();
      tAnTanOneTwoField = boost::make_shared<MatrixDouble>();
      tAnTanTwoOneField = boost::make_shared<MatrixDouble>();
      tAnTanTwoTwoField = boost::make_shared<MatrixDouble>();
      
      
      curvatuersField = boost::make_shared<MatrixDouble>();

      rotTangentOneField =  boost::make_shared<MatrixDouble>();
      metricTensorForField = boost::make_shared<MatrixDouble>();
      detOfMetricTensorForField = boost::make_shared<VectorDouble>();
    }

  private:
    MoFEM::Interface &mField;
  };

  MoFEMErrorCode addSurfaceSmoothingElement(const string element_name,
                                            const string field_name_position,
                                            const string field_name_tangent_one_field,
                                            const string field_name_tangent_two_field,
                                            const string field_lagrange,
                                            Range &range_smoothing_elements) {
    MoFEMFunctionBegin;

    CHKERR mField.add_finite_element(element_name, MF_ZERO);

    CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                      field_name_position);

    CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                      field_name_position);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_name_position);

    CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                      field_name_tangent_one_field);

    CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                      field_name_tangent_one_field);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_name_tangent_one_field);

    CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                      field_name_tangent_two_field);

    CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                      field_name_tangent_two_field);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_name_tangent_two_field);

    // CHKERR mField.modify_finite_element_add_field_col(element_name,
    //                                                   field_lagrange);

    // CHKERR mField.modify_finite_element_add_field_row(element_name,
    //                                                   field_lagrange);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_lagrange);

    mField.add_ents_to_finite_element_by_type(range_smoothing_elements, MBTRI,
                                              element_name);

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode addVolumeElement(const string element_name,
                                  const string field_name,
                                  Range &range_volume_elements) {
    MoFEMFunctionBegin;

    CHKERR mField.add_finite_element(element_name, MF_ZERO);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_name);

    mField.add_ents_to_finite_element_by_type(range_volume_elements, MBTET,
                                              element_name);

    MoFEMFunctionReturn(0);
  }

  /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetTangentForSmoothSurf : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetTangentForSmoothSurf(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;
      FTensor::Index<'i', 3> i;
      if (type == MBVERTEX) {
        commonSurfaceSmoothingElement->tAngent1->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAngent1->clear();

        commonSurfaceSmoothingElement->tAngent2->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAngent2->clear();
      }

      auto t_1 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
      auto t_2 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);
        // auto t_dof = data.getFTensor1FieldData<3>();
        FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
          
        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
          t_1(i) += t_dof(i) * t_N(0);
          t_2(i) += t_dof(i) * t_N(1);
          ++t_dof;
          ++t_N;
        }
        ++t_1;
        ++t_2;
      }

      MoFEMFunctionReturn(0);
    }
  };

  /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetNormalForSmoothSurf : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetNormalForSmoothSurf(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;


      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      commonSurfaceSmoothingElement->nOrmal->resize(3, ngp, false);
      commonSurfaceSmoothingElement->nOrmal->clear();
      commonSurfaceSmoothingElement->areaNL->resize(ngp, false);
      commonSurfaceSmoothingElement->areaNL->clear();
      
      commonSurfaceSmoothingElement->tAngent1Unit->resize(3, ngp, false);
      commonSurfaceSmoothingElement->tAngent1Unit->clear();
      commonSurfaceSmoothingElement->tAngent2Unit->resize(3, ngp, false);
      commonSurfaceSmoothingElement->tAngent2Unit->clear();


      auto t_1 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
      auto t_2 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);

      auto t_1_unit =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1Unit);
      auto t_2_unit =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2Unit);

      auto t_n = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);
      auto t_area =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);
      for (unsigned int gg = 0; gg != ngp; ++gg) {
        t_n(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);
        const double n_mag = sqrt(t_n(i) * t_n(i));
        t_n(i) /= n_mag;
        const double t_1_mag = sqrt(t_1(i) * t_1(i));
        t_1_unit(i) = t_1(i)/t_1_mag;
        t_2_unit(i) = FTensor::levi_civita(i, j, k) * t_n(j) * t_1_unit(k);

        t_area = 0.5 * n_mag;
        ++t_area;
        ++t_n;
        ++t_1;
        ++t_2;
        ++t_1_unit;
        ++t_2_unit;
      }

      MoFEMFunctionReturn(0);
    }
  };
  
  struct OpCalNormalFieldTwo : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpCalNormalFieldTwo(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;


      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      commonSurfaceSmoothingElement->nOrmalField->resize(3, ngp, false);
      commonSurfaceSmoothingElement->nOrmalField->clear();

      auto t_1 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);
      auto t_2 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);
      auto t_n = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);
      for (unsigned int gg = 0; gg != ngp; ++gg) {
        t_n(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);
        const double n_mag = sqrt(t_n(i) * t_n(i));
        t_n(i) /= n_mag;
        // t_area = 0.5 * n_mag;
        ++t_n;
        ++t_1;
        ++t_2;
      }

      MoFEMFunctionReturn(0);
    }
  };


    struct OpCalPrincipalCurvatures : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpCalPrincipalCurvatures(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;


      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      commonSurfaceSmoothingElement->curvatuersField->resize(2, ngp, false);
      commonSurfaceSmoothingElement->curvatuersField->clear();

      auto t_1 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);
      auto t_2 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

      auto t_tan_tan_one_one = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanOneOneField);

      auto t_tan_tan_one_two = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanOneTwoField);


      auto t_tan_tan_two_one = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanTwoOneField);

      auto t_tan_tan_two_two =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAnTanTwoTwoField);

      auto t_curvature =
          getFTensor1FromMat<2>(*commonSurfaceSmoothingElement->curvatuersField);
      auto t_n = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);
      for (unsigned int gg = 0; gg != ngp; ++gg) {


        const double fund_e = t_1(i) * t_1(i);
        const double fund_f = t_1(i) * t_2(i);
        const double fund_g = t_2(i) * t_2(i);
        const double fund_l = t_tan_tan_one_one(i) * t_n(i);
        
        const double fund_m_1 = t_tan_tan_one_two(i)* t_n(i);
        const double fund_m_2 = t_tan_tan_two_one(i)* t_n(i);
        
        const double fund_n = t_tan_tan_two_two(i) * t_n(i);

        double value_k = (fund_l * fund_n - fund_m_1 * fund_m_2)/(fund_e * fund_g - fund_f * fund_f);
        double value_h = 0.5 *
                         (fund_e * fund_n + fund_g * fund_l -
                          fund_f * fund_m_1 - fund_f * fund_m_2) /
                         (fund_e * fund_g - fund_f * fund_f);
        double val_discriminant = value_h * value_h - value_k;
        t_curvature(0) = value_h + sqrt(val_discriminant);
        t_curvature(1) = value_h - sqrt(val_discriminant);

        ++t_tan_tan_two_one;
        ++t_tan_tan_one_one;
        ++t_tan_tan_two_two;
        ++t_tan_tan_one_two;
        ++t_curvature;
        ++t_1;
        ++t_2;
        ++t_n;
      }

      MoFEMFunctionReturn(0);
    }
  };

  struct OpCalMetricTensorAndDeterminant : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpCalMetricTensorAndDeterminant(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;


      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      
      commonSurfaceSmoothingElement->metricTensorForField->resize(4, ngp, false);
      commonSurfaceSmoothingElement->metricTensorForField->clear();

      commonSurfaceSmoothingElement->detOfMetricTensorForField->resize(ngp, false);
      commonSurfaceSmoothingElement->detOfMetricTensorForField->clear();

      auto t_1 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);
      auto t_2 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

      auto t_metric_tensor = getFTensor2FromMat<2, 2>(
          *commonSurfaceSmoothingElement->metricTensorForField);
      auto t_det =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->detOfMetricTensorForField);


      for (unsigned int gg = 0; gg != ngp; ++gg) {
        t_metric_tensor(0, 0) = t_1(i) * t_1(i);
        t_metric_tensor(1, 1) = t_2(i) * t_2(i);
        t_metric_tensor(0, 1) = t_1(i) * t_2(i);
        t_metric_tensor(1, 0) = t_2(i) * t_1(i);
        t_det = t_metric_tensor(0, 0) * t_metric_tensor(1, 1) -
                t_metric_tensor(0, 1) * t_metric_tensor(1, 0);
        ++t_metric_tensor;
        ++t_det;
        ++t_1;
        ++t_2;
      }

      MoFEMFunctionReturn(0);
    }
  };
  


/// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpPrintNormals : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpPrintNormals(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;


      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      if (type != MBVERTEX)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      auto t_tangent_1 = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);
      auto t_tangent_1_tot =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
      auto t_tangent_2_tot =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
      // auto t_tangent_2 =
      // getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

      auto t_n = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

      FTensor::Tensor1<double, 3> t_normal_field;
      
      for (unsigned int gg = 0; gg != ngp; ++gg) {
        
        t_normal_field(i) = FTensor::levi_civita(i, j, k) * t_tangent_1_tot(j) * t_tangent_2_tot(k);
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "X Normal = %e, %e, %e\n\n",
                       t_normal_field(0), t_normal_field(1), t_normal_field(2));
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Field Normal = %e, %e, %e\n\n",
                       t_tangent_1(0), t_tangent_1(1), t_tangent_1(2));

        ++t_tangent_1;
        // ++t_tangent_2;
        ++t_n;
        ++t_tangent_1_tot;
        ++t_tangent_2_tot;
      }

      MoFEMFunctionReturn(0);
    }
  };
  

    /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetNormalField : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetNormalField(
        const string normal_field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(normal_field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      FTensor::Index<'q', 2> q;


        auto t_1 = getFTensor1Tangent1();        
        auto t_2 = getFTensor1Tangent2();        

        const double size_1 =  sqrt(t_1(i) * t_1(i));
        const double size_2 =  sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;

        FTensor::Tensor1<double, 3> t_normal;

        t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);

      // FTensor::Tensor2<const double *, 3, 3> t_m(
      //     &t_1(0), &t_2(0), &t_normal(0),

      //     &t_1(1), &t_2(1), &t_normal(1),

      //     &t_1(2), &t_2(2), &t_normal(2),

      //     3);

      FTensor::Tensor2<const double *, 3, 3> t_m(
          &t_1(0), &t_1(1), &t_1(2),

          &t_2(0), &t_2(1), &t_2(2),

          &t_normal(0), &t_normal(1), &t_normal(2),

          3);

      double det;
      FTensor::Tensor2<double, 3, 3> t_inv_m;
      CHKERR determinantTensor3by3(t_m, det);
      CHKERR invertTensor3by3(t_m, det, t_inv_m);

      if (type == MBVERTEX) {
        commonSurfaceSmoothingElement->nOrmalField->resize(3, ngp, false);
        commonSurfaceSmoothingElement->nOrmalField->clear();

        commonSurfaceSmoothingElement->divNormalField->resize(ngp, false);
        commonSurfaceSmoothingElement->divNormalField->clear();
      }

      auto t_n_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

      auto t_div_normal =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->divNormalField);
      FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 2> t_container_N(
          &t_inv_m(0, 0), &t_inv_m(0, 1), &t_inv_m(1, 0),

          &t_inv_m(1, 1), &t_inv_m(2, 0), &t_inv_m(2, 1));

      FTensor::Tensor1<double, 3> t_transformed_N;

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);
        // auto t_dof = data.getFTensor1FieldData<3>();
        FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
        FTensor::Tensor0<double *> t_base_normal_field(&data.getN()(gg, 0));

        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
          // t_container_N(0,0) = t_container_N(1,0) = t_container_N(2,0) = t_N(0);
          // t_container_N(2,0) = t_container_N(2,1) = t_container_N(2,1) = t_N(1);
          t_transformed_N(i) = t_N(q) * t_container_N(i, q);

          t_n_field(i) += t_dof(i) * t_base_normal_field;
          t_div_normal += t_dof(i) * t_transformed_N(i);
          ++t_dof;
          ++t_N;
          ++t_base_normal_field;
        }
        ++t_n_field;
        ++t_div_normal;

      }

      MoFEMFunctionReturn(0);
    }
  };


  struct OpGetTangentOneField : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetTangentOneField(
        const string normal_field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(normal_field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      FTensor::Index<'q', 2> q;


        auto t_1 = getFTensor1Tangent1();        
        auto t_2 = getFTensor1Tangent2();        

        const double size_1 =  sqrt(t_1(i) * t_1(i));
        const double size_2 =  sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;

        FTensor::Tensor1<double, 3> t_normal;

        t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);

      // FTensor::Tensor2<const double *, 3, 3> t_m(
      //     &t_1(0), &t_2(0), &t_normal(0),

      //     &t_1(1), &t_2(1), &t_normal(1),

      //     &t_1(2), &t_2(2), &t_normal(2),

      //     3);

      FTensor::Tensor2<const double *, 3, 3> t_m(
          &t_1(0), &t_1(1), &t_1(2),

          &t_2(0), &t_2(1), &t_2(2),

          &t_normal(0), &t_normal(1), &t_normal(2),

          3);

      double det;
      FTensor::Tensor2<double, 3, 3> t_inv_m;
      CHKERR determinantTensor3by3(t_m, det);
      CHKERR invertTensor3by3(t_m, det, t_inv_m);

      if (type == MBVERTEX) {
        commonSurfaceSmoothingElement->tAngentOneField->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAngentOneField->clear();
        
        commonSurfaceSmoothingElement->tAnTanOneOneField->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAnTanOneOneField->clear();

        commonSurfaceSmoothingElement->tAnTanOneTwoField->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAnTanOneTwoField->clear();
        }

      auto t_tan_1_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);

      auto t_tan_tan_one_one = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanOneOneField);

      auto t_tan_tan_one_two = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanOneTwoField);


      FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 2> t_container_N(
          &t_inv_m(0, 0), &t_inv_m(0, 1), &t_inv_m(1, 0),

          &t_inv_m(1, 1), &t_inv_m(2, 0), &t_inv_m(2, 1));

      FTensor::Tensor1<double, 3> t_transformed_N;

      auto t_tan_1 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);
        // auto t_dof = data.getFTensor1FieldData<3>();
        FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
        FTensor::Tensor0<double *> t_base_normal_field(&data.getN()(gg, 0));

        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
          // t_container_N(0,0) = t_container_N(1,0) = t_container_N(2,0) = t_N(0);
          // t_container_N(2,0) = t_container_N(2,1) = t_container_N(2,1) = t_N(1);
          t_transformed_N(i) = t_N(q) * t_container_N(i, q);
//FTensor::levi_civita(i, j, k)
          t_tan_1_field(i) += t_dof(i) * t_base_normal_field;
          // t_div_tan_1 += FTensor::levi_civita(i, j, k) * t_dof(j) * t_transformed_N(i) * ;
          t_tan_tan_one_one(i) += t_N(0) * t_dof(i);
          t_tan_tan_one_two(i) += t_N(1) * t_dof(i);
          ++t_dof;
          ++t_N;
          ++t_base_normal_field;
        }
        if(t_tan_1_field(i) * t_tan_1_field(i) < 1.e-8){
        // t_tan_1_field(i) =  1.e-3;
        // t_tan_tan_one_one(i) = 1.e-3;
        // t_tan_tan_one_two(i) = 1.e-3;
        } else {
        // cerr << "t_tan_1_field   " << t_tan_1_field << "\n"; 
        // cerr << "t_tan_tan_one_one   " << t_tan_tan_one_one << "\n"; 
        // cerr << "t_tan_tan_one_two   " << t_tan_tan_one_two << "\n"; 
        }

        ++t_tan_tan_one_two;
        ++t_tan_tan_one_one;  
        ++t_tan_1_field;
        ++t_tan_1;
      }

      MoFEMFunctionReturn(0);
    }
  };


  struct OpGetTangentTwoField : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetTangentTwoField(
        const string normal_field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(normal_field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      FTensor::Index<'q', 2> q;


        auto t_1 = getFTensor1Tangent1();        
        auto t_2 = getFTensor1Tangent2();        

        const double size_1 =  sqrt(t_1(i) * t_1(i));
        const double size_2 =  sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;

        FTensor::Tensor1<double, 3> t_normal;

        t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);

      // FTensor::Tensor2<const double *, 3, 3> t_m(
      //     &t_1(0), &t_2(0), &t_normal(0),

      //     &t_1(1), &t_2(1), &t_normal(1),

      //     &t_1(2), &t_2(2), &t_normal(2),

      //     3);

      FTensor::Tensor2<const double *, 3, 3> t_m(
          &t_1(0), &t_1(1), &t_1(2),

          &t_2(0), &t_2(1), &t_2(2),

          &t_normal(0), &t_normal(1), &t_normal(2),

          3);

      double det;
      FTensor::Tensor2<double, 3, 3> t_inv_m;
      CHKERR determinantTensor3by3(t_m, det);
      CHKERR invertTensor3by3(t_m, det, t_inv_m);

      if (type == MBVERTEX) {
        commonSurfaceSmoothingElement->tAngentTwoField->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAngentTwoField->clear();
        
        commonSurfaceSmoothingElement->tAnTanTwoOneField->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAnTanTwoOneField->clear();

        commonSurfaceSmoothingElement->tAnTanTwoTwoField->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tAnTanTwoTwoField->clear();


      }

      auto t_n_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

      auto t_tan_tan_two_one =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAnTanTwoOneField);

      auto t_tan_tan_two_two =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAnTanTwoTwoField);


      FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 2> t_container_N(
          &t_inv_m(0, 0), &t_inv_m(0, 1), &t_inv_m(1, 0),

          &t_inv_m(1, 1), &t_inv_m(2, 0), &t_inv_m(2, 1));

      FTensor::Tensor1<double, 3> t_transformed_N;
      auto t_tan_2 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);
        // auto t_dof = data.getFTensor1FieldData<3>();
        FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
        FTensor::Tensor0<double *> t_base_normal_field(&data.getN()(gg, 0));

        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
          // t_container_N(0,0) = t_container_N(1,0) = t_container_N(2,0) = t_N(0);
          // t_container_N(2,0) = t_container_N(2,1) = t_container_N(2,1) = t_N(1);
          t_transformed_N(i) = t_N(q) * t_container_N(i, q);

          t_n_field(i) += t_dof(i) * t_base_normal_field;

          t_tan_tan_two_one(i) += t_N(0) * t_dof(i);
          t_tan_tan_two_two(i) += t_N(1) * t_dof(i);

          ++t_dof;
          ++t_N;
          ++t_base_normal_field;
        }
      if(t_n_field(i)* t_n_field(i) < 1.e-8){
        // t_n_field(0) = 1.e-3;
        // t_n_field(1) = 1.e-4;
        // t_n_field(2) = 1.e-5;
        
        // t_tan_tan_two_one(i) = t_n_field(i);
        // t_tan_tan_two_two(i) = t_n_field(i);
      }
        ++t_tan_tan_two_one;
        ++t_tan_tan_two_two;
        ++t_n_field;
        ++t_tan_2;
        
      }

      MoFEMFunctionReturn(0);
    }
  };



  struct OpGetDivOneField : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetDivOneField(
        const string normal_field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(normal_field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      FTensor::Index<'q', 2> q;


        auto t_1 = getFTensor1Tangent1();        
        auto t_2 = getFTensor1Tangent2();        

        const double size_1 =  sqrt(t_1(i) * t_1(i));
        const double size_2 =  sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;

        FTensor::Tensor1<double, 3> t_normal;

        t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);

      // FTensor::Tensor2<const double *, 3, 3> t_m(
      //     &t_1(0), &t_2(0), &t_normal(0),

      //     &t_1(1), &t_2(1), &t_normal(1),

      //     &t_1(2), &t_2(2), &t_normal(2),

      //     3);

      FTensor::Tensor2<const double *, 3, 3> t_m(
          &t_1(0), &t_1(1), &t_1(2),

          &t_2(0), &t_2(1), &t_2(2),

          &t_normal(0), &t_normal(1), &t_normal(2),

          3);

      double det;
      FTensor::Tensor2<double, 3, 3> t_inv_m;
      CHKERR determinantTensor3by3(t_m, det);
      CHKERR invertTensor3by3(t_m, det, t_inv_m);

      if (type == MBVERTEX) {
        commonSurfaceSmoothingElement->divTangentOneField->resize(ngp, false);
        commonSurfaceSmoothingElement->divTangentOneField->clear();
        commonSurfaceSmoothingElement->tAngentOneGradField->resize(9, ngp, false);
        commonSurfaceSmoothingElement->tAngentOneGradField->clear();
        commonSurfaceSmoothingElement->rotTangentOneField->resize(3, ngp, false);
        commonSurfaceSmoothingElement->rotTangentOneField->clear();
      }

      // auto t_tangent_two_field =
      //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

      auto t_div_tan_1 =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->divTangentOneField);

      FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 2> t_container_N(
          &t_inv_m(0, 0), &t_inv_m(0, 1), &t_inv_m(1, 0),

          &t_inv_m(1, 1), &t_inv_m(2, 0), &t_inv_m(2, 1));

      FTensor::Tensor1<double, 3> t_transformed_N;

      auto t_tangent_one_grad = getFTensor2FromMat<3, 3>(*commonSurfaceSmoothingElement->tAngentOneGradField);
      auto t_rot_normal = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->rotTangentOneField);
  
      for (unsigned int gg = 0; gg != ngp; ++gg) {
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);
        // auto t_dof = data.getFTensor1FieldData<3>();
        FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
        FTensor::Tensor0<double *> t_base_normal_field(&data.getN()(gg, 0));

        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
          // t_container_N(0,0) = t_container_N(1,0) = t_container_N(2,0) = t_N(0);
          // t_container_N(2,0) = t_container_N(2,1) = t_container_N(2,1) = t_N(1);
          t_transformed_N(i) = t_N(q) * t_container_N(i, q);
//FTensor::levi_civita(i, j, k)
          // t_n_field(i) += t_dof(i) * t_base_normal_field;
          t_div_tan_1 += t_dof(i) * t_transformed_N(i);
          // t_div_tan_1 += FTensor::levi_civita(i, j, k) * t_dof(j) * t_transformed_N(i) * t_tangent_two_field(k);
          t_tangent_one_grad(i, j) += t_dof(i) * t_transformed_N(j);
          t_rot_normal(i) += FTensor::levi_civita(i, j, k) * t_transformed_N(j) * t_dof(k);
          ++t_dof;
          ++t_N;
          ++t_base_normal_field;
        }
        // ++t_n_field;
        // ++t_tangent_two_field;
        
        
        ++t_div_tan_1;


        ++t_tangent_one_grad;
      }

      MoFEMFunctionReturn(0);
    }
  };

    struct OpGetDivTwoField : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetDivTwoField(
        const string normal_field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(normal_field_name, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      FTensor::Index<'q', 2> q;


        auto t_1 = getFTensor1Tangent1();        
        auto t_2 = getFTensor1Tangent2();        

        const double size_1 =  sqrt(t_1(i) * t_1(i));
        const double size_2 =  sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;

        FTensor::Tensor1<double, 3> t_normal;

        t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);

      // FTensor::Tensor2<const double *, 3, 3> t_m(
      //     &t_1(0), &t_2(0), &t_normal(0),

      //     &t_1(1), &t_2(1), &t_normal(1),

      //     &t_1(2), &t_2(2), &t_normal(2),

      //     3);

      FTensor::Tensor2<const double *, 3, 3> t_m(
          &t_1(0), &t_1(1), &t_1(2),

          &t_2(0), &t_2(1), &t_2(2),

          &t_normal(0), &t_normal(1), &t_normal(2),

          3);

      double det;
      FTensor::Tensor2<double, 3, 3> t_inv_m;
      CHKERR determinantTensor3by3(t_m, det);
      CHKERR invertTensor3by3(t_m, det, t_inv_m);

      if (type == MBVERTEX) {
        commonSurfaceSmoothingElement->divTangentTwoField->resize(ngp, false);
        commonSurfaceSmoothingElement->divTangentTwoField->clear();
        commonSurfaceSmoothingElement->tAngentTwoGradField->resize(9, ngp, false);
        commonSurfaceSmoothingElement->tAngentTwoGradField->clear();
      }

      auto t_tangent_one_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);

      auto t_div_tan_2 =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->divTangentTwoField);
      FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 2> t_container_N(
          &t_inv_m(0, 0), &t_inv_m(0, 1), &t_inv_m(1, 0),

          &t_inv_m(1, 1), &t_inv_m(2, 0), &t_inv_m(2, 1));

      FTensor::Tensor1<double, 3> t_transformed_N;

      auto t_tangent_two_grad = getFTensor2FromMat<3, 3>(*commonSurfaceSmoothingElement->tAngentTwoGradField);
  

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);
        // auto t_dof = data.getFTensor1FieldData<3>();
        FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
        FTensor::Tensor0<double *> t_base_normal_field(&data.getN()(gg, 0));

        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
          // t_container_N(0,0) = t_container_N(1,0) = t_container_N(2,0) = t_N(0);
          // t_container_N(2,0) = t_container_N(2,1) = t_container_N(2,1) = t_N(1);
          t_transformed_N(i) = t_N(q) * t_container_N(i, q);
//FTensor::levi_civita(i, j, k)
          // t_n_field(i) += t_dof(i) * t_base_normal_field;
          t_div_tan_2 += FTensor::levi_civita(i, j, k) * t_dof(k) * t_transformed_N(i) * t_tangent_one_field(j);
          t_tangent_two_grad(i, j) += t_dof(i) * t_transformed_N(j);
          ++t_dof;
          ++t_N;
          ++t_base_normal_field;
        }
        // ++t_n_field;
        ++t_tangent_one_field;
        ++t_div_tan_2;
        ++t_tangent_two_grad;

      }

      MoFEMFunctionReturn(0);
    }
  };

    /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetLagrangeField : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetLagrangeField(
        const string lagrange_field,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(lagrange_field, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      if (data.getFieldData().size() == 0)
        PetscFunctionReturn(0);

      int ngp = data.getN().size1();

      unsigned int nb_dofs = data.getFieldData().size() / 3;
      FTensor::Index<'i', 3> i;
      if (type == MBVERTEX) {
        commonSurfaceSmoothingElement->lagMult->resize(3, ngp, false);
        commonSurfaceSmoothingElement->lagMult->clear();
      }

      auto t_lag_mult =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        // auto t_dof = data.getFTensor1FieldData<3>();
        FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
        FTensor::Tensor0<double *> t_base_lagrange_field(&data.getN()(gg, 0));

        for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
          t_lag_mult(i) += t_dof(i) * t_base_lagrange_field;
          ++t_dof;
          ++t_base_lagrange_field;
        }
        ++t_lag_mult;
      }

      MoFEMFunctionReturn(0);
    }
  };

  struct OpCalSmoothingXRhs : public FaceEleOp {

    OpCalSmoothingXRhs(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs / 3;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;

        auto t_n =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);
        auto t_n_h_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalHdiv);

        auto t_w = getFTensor0IntegrationWeight();

        for (int gg = 0; gg != nb_gauss_pts; ++gg) {

          double val_m = t_w * area_m;

          FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
              &vecF[0], &vecF[1], &vecF[2]};
          auto t_base = data.getFTensor0N(gg, 0);          
          for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
            t_assemble_m(i) += (t_n(i) - t_n_h_div(i)) * t_base * val_m;
            
            // cerr <<"Vec 2   " << t_n << "\n";

            ++t_assemble_m;
            ++t_base;
          }
          ++t_n;
          ++t_n_h_div;
          ++t_w;
        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
  };


  struct OpCalLagMultRhs : public FaceEleOp {

    OpCalLagMultRhs(
        const string lagrange_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(lagrange_name, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs / 3;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;

        auto t_n =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);
        auto t_n_field =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

        auto t_w = getFTensor0IntegrationWeight();
      auto t_area =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);
        for (int gg = 0; gg != nb_gauss_pts; ++gg) {

          double val_m = t_w * area_m;
        // double val_m = t_w * t_area;
          FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
              &vecF[0], &vecF[1], &vecF[2]};
          auto t_base = data.getFTensor0N(gg, 0);          
          for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
            t_assemble_m(i) += (t_n(i) - t_n_field(i)) * t_base * val_m;
            
            // cerr <<"Vec 2   " << t_n << "\n";

            ++t_assemble_m;
            ++t_base;
          }
          // cerr << "t_n_field   " << t_n_field <<"\n";
          // cerr << "t_n   " << t_n <<"\n";
          ++t_n;
          ++t_n_field;
          ++t_w;
          ++t_area;
        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
  };

struct OpCalPositionsRhs : public FaceEleOp {

    OpCalPositionsRhs(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing), cnValue(cn_value) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs / 3;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;

        auto make_vec_der = [&](FTensor::Tensor1<double *, 2> t_N,
                                FTensor::Tensor1<double *, 3> t_1,
                                FTensor::Tensor1<double *, 3> t_2,
                                FTensor::Tensor1<double *, 3> t_normal) {
          FTensor::Tensor2<double, 3, 3> t_n;
          FTensor::Tensor2<double, 3, 3> t_n_container;
          const double n_norm = sqrt(t_normal(i) * t_normal(i));
          t_n(i, j) = 0;
          t_n_container(i, j) = 0;
          t_n_container(i, j) +=
              FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
          t_n_container(i, j) -=
              FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
          t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
                                                         t_n_container(j, k) /
                                                         pow(n_norm, 3);
          return t_n;
        };

        auto t_1 =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
        auto t_2 =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
        auto t_normal =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

        auto t_normal_field =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

        auto t_lag_mult =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

        auto t_w = getFTensor0IntegrationWeight();
      auto t_area =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);
        for (int gg = 0; gg != nb_gauss_pts; ++gg) {

          double val_m = t_w * area_m;
          // double val_m = t_w * t_area;

          FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
              &vecF[0], &vecF[1], &vecF[2]};
          auto t_base = data.getFTensor0N(gg, 0); 
          auto t_N = data.getFTensor1DiffN<2>(gg, 0);
         
          for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
            auto t_d_n = make_vec_der(t_N, t_1, t_2, t_normal);

            // t_assemble_m(j) += (2 *(t_normal(i) - t_normal_field(i)) + t_lag_mult(i)) * t_d_n(i, j) * val_m;

            t_assemble_m(j) += (cnValue *(t_normal(i) - t_normal_field(i)) /*+ t_lag_mult(i)*/) * t_d_n(i, j) * val_m;
            
            
            // t_assemble_m(j) -= (t_normal_field(i) - t_normal(i)) * t_d_n(i, j) * val_m;

            // t_assemble_m(i) += (t_normal_field(j) * t_normal(j) - 1. )* t_base * val_m;
            ++t_assemble_m;
            ++t_base;
            ++t_N;
          }
          ++t_normal;
          ++t_w;
          ++t_1;
          ++t_2;
          ++t_lag_mult;
          ++t_area;
          ++t_normal_field;
        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
    double cnValue;
  };

struct OpCalPositionsForTanRhs : public FaceEleOp {

    OpCalPositionsForTanRhs(
        const string field_name,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing), cnValue(cn_value) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs / 3;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;
        FTensor::Index<'q', 3> q;

        auto make_vec_der = [&](FTensor::Tensor1<double *, 2> t_N,
                                FTensor::Tensor1<double *, 3> t_1,
                                FTensor::Tensor1<double *, 3> t_2,
                                FTensor::Tensor1<double *, 3> t_normal) {
          FTensor::Tensor2<double, 3, 3> t_n;
          FTensor::Tensor2<double, 3, 3> t_n_container;
          const double n_norm = sqrt(t_normal(i) * t_normal(i));
          t_n(i, j) = 0;
          t_n_container(i, j) = 0;
          t_n_container(i, j) +=
              FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
          t_n_container(i, j) -=
              FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
          t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
                                                         t_n_container(j, k) /
                                                         pow(n_norm, 3);
          return t_n;
        };

        auto make_vec_der_1 = [&](double t_N,
                                FTensor::Tensor1<double *, 3> t_1,
                                FTensor::Tensor1<double *, 3> t_2,
                                FTensor::Tensor1<double *, 3> t_normal) {
          FTensor::Tensor2<double, 3, 3> t_n;
          FTensor::Tensor2<double, 3, 3> t_n_container;
          const double n_norm = sqrt(t_normal(i) * t_normal(i));
          t_n(i, j) = 0;
          t_n_container(i, j) = 0;
          t_n_container(i, j) +=
              FTensor::levi_civita(i, j, k) * t_2(k) * t_N;
          t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
                                                         t_n_container(j, k) /
                                                         pow(n_norm, 3);
          return t_n;
        };
        

        auto t_1 =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
        auto t_2 =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
        
        
        auto t_normal =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

        auto t_tan_1_field =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);

        auto t_tan_2_field =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

        // auto t_normal_field =
        //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);


        // auto t_lag_mult =
        //   getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

        auto t_w = getFTensor0IntegrationWeight();
      auto t_area =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);

      // auto t_1_unit =
      //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1Unit);
      // auto t_2_unit =
      //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2Unit);

        for (int gg = 0; gg != nb_gauss_pts; ++gg) {

          double val_m = t_w * area_m;
          // double val_m = t_w * t_area;

          FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
              &vecF[0], &vecF[1], &vecF[2]};
          auto t_base = data.getFTensor0N(gg, 0); 
          auto t_N = data.getFTensor1DiffN<2>(gg, 0);
         FTensor::Tensor1<double *, 3> t_dof(
              &data.getFieldData()[0], &data.getFieldData()[1], &data.getFieldData()[2], 3);
        
          for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
            auto t_d_n = make_vec_der(t_N, t_1, t_2, t_normal);

            // t_assemble_m(j) += (2 *(t_normal(i) - t_normal_field(i)) + t_lag_mult(i)) * t_d_n(i, j) * val_m;
            double mag_tan_1 = sqrt(t_1(i) * t_1(i));
            double mag_tan_2 = sqrt(t_2(i) * t_2(i));
            double mag_t_1 = sqrt(t_1(i) * t_1(i));
            constexpr auto t_kd = FTensor::Kronecker_Delta<int>();
            
            // t_assemble_m(q) += ( cnValue * (t_1_unit(i) - t_tan_1_field(i)) ) * FTensor::levi_civita(i, j, k) * t_normal(j) * ( t_kd(k, q)/mag_t_1 - t_1(k) * t_1(q)/pow(mag_t_1, 3) )  * t_N(0) * val_m;
            // t_assemble_m(q) += ( cnValue * (t_2_unit(i) - t_tan_2_field(i)) ) * FTensor::levi_civita(i, j, k) * t_d_n(j, q) * t_1_unit(k) * val_m;
            
            
            // t_assemble_m(q) += ( cnValue * (t_1_unit(i) - t_tan_1_field(i)) ) * FTensor::levi_civita(i, j, k) * t_normal(j) * ( t_kd(k, q)/mag_t_1 - t_1(k) * t_1(q)/pow(mag_t_1, 3) )  * t_N(0) * val_m;
            // t_assemble_m(q) += ( cnValue * (t_2_unit(i) - t_tan_2_field(i)) ) * FTensor::levi_civita(i, j, k) * t_d_n(j, q) * t_1_unit(k) * val_m;
            
            
            // t_assemble_m(j) += cnValue * ( t_normal(i) - t_tan_1_field(i) ) * t_d_n(i, j) * val_m;
            // t_assemble_m(i) += cnValue * (t_tan_1_field(j) * (t_1(j) + t_2(j) )  ) * t_tan_1_field(i) * (t_N(0) + t_N(1)) * val_m;
            // t_assemble_m(k) += cnValue * (t_normal(j) * t_tan_1_field(j)  - 1. ) * t_tan_1_field(i) * t_d_n(i, k) * val_m;

            t_assemble_m(j) += cnValue * (t_1(j) - t_tan_1_field(j)) * t_N(0);
            t_assemble_m(j) += cnValue * (t_2(j) - t_tan_2_field(j)) * t_N(1);
// cerr <<"t_assemble_m " << t_assemble_m << "\n";
            // t_assemble_m(j) += cnValue * t_1(j);
            // t_assemble_m(j) += cnValue * t_N(1);

            // t_assemble_m(q) += ( cnValue * (t_2_unit(i) - t_tan_2_field(i)) ) * FTensor::levi_civita(i, j, k) * t_normal(j) * ( t_kd(k, q)/mag_t_1 - t_1(k) * t_1(q)/pow(mag_t_1, 3) )  * t_N(0) * val_m;
            // t_assemble_m(q) += ( cnValue * (t_1_unit(i) - t_tan_1_field(i)) ) * FTensor::levi_civita(i, j, k) * t_d_n(j, q) * t_1_unit(k) * val_m;




            // t_assemble_m(i) += (t_normal_field(j) * t_normal(j) - 1. )* t_base * val_m;
            ++t_assemble_m;
            ++t_base;
            ++t_N;
            ++t_dof;
          }
          ++t_normal;
          ++t_w;
          ++t_1;
          ++t_2;
          // ++t_lag_mult;
          ++t_area;
          ++t_tan_1_field;
          ++t_tan_2_field;
          // ++t_normal_field;
          // ++t_1_unit;
          // ++t_2_unit;
        } // for gauss points
        
        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
    double cnValue;
  };


  /// \brief Computes, for reference configuration, normal to slave face that is
  /// common to all gauss points
  struct OpGetTangentHdivAtGaussPoints : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetTangentHdivAtGaussPoints(
        const string field_name_h_div,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name_h_div, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        int ngp = data.getN().size1();

        commonSurfaceSmoothingElement->divNormalHdiv.get()->resize(ngp);
        commonSurfaceSmoothingElement->divNormalHdiv.get()->clear();

        commonSurfaceSmoothingElement->tanHdiv1->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tanHdiv1->clear();
        commonSurfaceSmoothingElement->tanHdiv2->resize(3, ngp, false);
        commonSurfaceSmoothingElement->tanHdiv2->clear();
        commonSurfaceSmoothingElement->matForDivTan1->resize(6, ngp, false);
        commonSurfaceSmoothingElement->matForDivTan1->clear();
        commonSurfaceSmoothingElement->matForDivTan2->resize(6, ngp, false);
        commonSurfaceSmoothingElement->matForDivTan2->clear();

        auto t_t_1_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv1);

        auto t_t_2_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv2);

        auto t_divergence_n_h_div =
            getFTensor0FromVec(*commonSurfaceSmoothingElement->divNormalHdiv);

        auto t_div_tan_1 = getFTensor2FromMat<3, 2>(*commonSurfaceSmoothingElement->matForDivTan1);
        auto t_div_tan_2 = getFTensor2FromMat<3, 2>(*commonSurfaceSmoothingElement->matForDivTan2);

        auto t_1 = getFTensor1Tangent1();        
        auto t_2 = getFTensor1Tangent2();        

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;
        FTensor::Index<'m', 3> m;
        FTensor::Index<'l', 3> l;
        FTensor::Index<'q', 2> q;
        const double size_1 =  sqrt(t_1(i) * t_1(i));
        const double size_2 =  sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;

        FTensor::Tensor2<double, 3, 2> t_grad, t_grad_1, t_grad_2;
        FTensor::Tensor1<double, 3> t_vec_1, t_vec_2;
        FTensor::Tensor2<double, 3, 3> t_mat_1, t_mat_2;

        int nb_base_fun_row = data.getFieldData().size() / 2;

        auto t_diff_base_fun = data.getFTensor2DiffN<3, 2>();
        for (unsigned int gg = 0; gg != ngp; ++gg) {
          FTensor::Tensor1<double *, 2> t_field_data(
              &data.getFieldData()[0], &data.getFieldData()[1], 2);
          auto t_base = data.getFTensor1N<3>(gg, 0);
          

          for (int bb = 0; bb != nb_base_fun_row; ++bb) {
            t_vec_1(j) = t_field_data(0) * t_base(m) * t_1(m) * t_1(j);
            t_vec_2(j) = t_field_data(1) * t_base(m) * t_2(m) * t_2(j);
            t_t_1_div(j) += t_vec_1(j);
            t_t_2_div(j) += t_vec_2(j);
            t_grad_1(j, q) = t_diff_base_fun(m, q) * t_1(m) * t_1(j);
            t_grad_2(j, q) = t_diff_base_fun(m, q) * t_2(m) * t_2(j);

            t_div_tan_1(j, q) += t_field_data(0) * t_grad_1(j, q);
            t_div_tan_2(j, q) += t_field_data(0) * t_grad_2(j, q);

            t_mat_1(i, j) =  FTensor::levi_civita(i, j, k) * t_vec_2(k);
            t_mat_2(i, k) =  FTensor::levi_civita(i, j, k) * t_vec_1(j);

            t_grad(i, q) =
                (t_grad_1(j, q) * t_mat_1(i, j) + t_grad_1(k, q) * t_mat_2(i, k));

            t_divergence_n_h_div += t_grad(0, 0) + t_grad(1, 1);
            ++t_base;
            ++t_diff_base_fun;
            ++t_field_data;
          }
          ++t_divergence_n_h_div;
          ++t_t_1_div;
          ++t_t_2_div;
          ++t_div_tan_1;
          ++t_div_tan_2;
        }
      }
      MoFEMFunctionReturn(0);
    }
  };

 struct OpGetNormalHdivAtGaussPoints : public FaceEleOp {

    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    OpGetNormalHdivAtGaussPoints(
        const string field_name_h_div,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name_h_div, UserDataOperator::OPCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        int ngp = data.getN().size1();

        commonSurfaceSmoothingElement->nOrmalHdiv->resize(3, ngp, false);
        commonSurfaceSmoothingElement->nOrmalHdiv->clear();
        
        auto t_n_h_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalHdiv);

        auto t_t_1_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv1);

        auto t_t_2_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv2);

        auto t_divergence_n_h_div =
            getFTensor0FromVec(*commonSurfaceSmoothingElement->divNormalHdiv);
          
        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;

        
        for (unsigned int gg = 0; gg != ngp; ++gg) {
          t_n_h_div(i) = FTensor::levi_civita(i, j, k) * t_t_1_div(j)  * t_t_2_div(k);

          ++t_n_h_div;
          ++t_t_1_div;
          ++t_t_2_div;
        }
      }
      MoFEMFunctionReturn(0);
    }
  };

  struct OpCalSmoothingNormalHdivRhs : public FaceEleOp {

    OpCalSmoothingNormalHdivRhs(
        const string field_name_h_div,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name_h_div, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing) {
            sYmm = false; // This will make sure to loop over all entities
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;
        FTensor::Index<'m', 3> m;
        FTensor::Index<'l', 3> l;
        FTensor::Index<'q', 2> q;
        auto t_divergence_n_h_div =
            getFTensor0FromVec(*commonSurfaceSmoothingElement->divNormalHdiv);

        auto t_w = getFTensor0IntegrationWeight();

        auto t_1 = getFTensor1Tangent1();        
        auto t_2 = getFTensor1Tangent2();        

        const double size_1 =  sqrt(t_1(i) * t_1(i));
        const double size_2 =  sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;

        auto t_t_1_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv1);

        auto t_t_2_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv2);

        auto t_div_tan_1 = getFTensor2FromMat<3, 2>(*commonSurfaceSmoothingElement->matForDivTan1);
        auto t_div_tan_2 = getFTensor2FromMat<3, 2>(*commonSurfaceSmoothingElement->matForDivTan2);


        FTensor::Tensor2<double, 3, 2> t_div_1, t_div_2;
        FTensor::Tensor2<double, 3, 3> t_mat_1, t_mat_2;

      // auto t_1 =
      //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
      // auto t_2 =
      //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);

        auto t_diff_base_fun = data.getFTensor2DiffN<3, 2>();
          
        for (int gg = 0; gg != nb_gauss_pts; ++gg) {
          // const double norm_1 = sqrt(t_1(i) * t_1(i));
          // const double norm_2 = sqrt(t_2(i) * t_2(i));
          // t_1(i) /= norm_1;
          // t_2(i) /= norm_2;
          
          double val_m = t_w * area_m;
          auto t_base = data.getFTensor1N<3>(gg, 0);
          FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_assemble_m{
          &vecF[0], &vecF[1]};

          for (int bbc = 0; bbc != nb_base_fun_col / 2; ++bbc) {
            // vecF[bbc] += t_row_diff_base(i, i) * t_divergence_n_h_div * val_m;
            t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_2_div(k);
            t_div_1(i, q) =
                t_mat_1(i, j) * t_diff_base_fun(m, q) * t_1(m) * t_1(j);
            t_assemble_m(0) +=
                (t_div_1(0, 0) + t_div_1(1, 1)) * t_divergence_n_h_div * val_m;
            
            t_mat_1(i, k) = FTensor::levi_civita(i, j, k) * (t_base(m) * t_1(m) * t_1(j));
            t_div_1(i, q) =
                t_mat_1(i, k) * t_div_tan_2(k,q);
            t_assemble_m(0) +=
                (t_div_1(0, 0) + t_div_1(1, 1)) * t_divergence_n_h_div * val_m;

            t_mat_2(i, k) = FTensor::levi_civita(i, j, k) * t_t_1_div(j);
            t_div_2(i, q) =
                t_mat_2(i, k) * t_diff_base_fun(m, q) * t_2(m) * t_2(k);
            t_assemble_m(1) +=
                (t_div_2(0, 0) + t_div_2(1, 1)) * t_divergence_n_h_div * val_m;

            t_mat_2(i, j) = FTensor::levi_civita(i, j, k) * (t_base(m) * t_2(m) * t_2(k));
            t_div_2(i, q) =
                t_mat_2(i, j) * t_div_tan_1(j,q);
            t_assemble_m(1) +=
                (t_div_2(0, 0) + t_div_2(1, 1)) * t_divergence_n_h_div * val_m;
            
            // if(abs(t_assemble_m(0)) > 1.e-20 || abs(t_assemble_m(1)) > 1.e-20 )
            // cerr <<"Vec 1   " << t_assemble_m << "\n";

            ++t_assemble_m;
            ++t_diff_base_fun;
            ++t_base;
          }
          ++t_divergence_n_h_div;
          ++t_w;
          ++t_t_1_div;
          ++t_t_2_div;
          ++t_div_tan_1;
          ++t_div_tan_2;
          // ++t_1;
          // ++t_2;
        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
  };


  struct OpCalNormalFieldRhs : public FaceEleOp {

    OpCalNormalFieldRhs(
        const string field_name_h_div,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name_h_div, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing), cnValue(cn_value) {
            sYmm = false; // This will make sure to loop over all entities
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;

         auto t_n_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

         auto t_n =
             getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

         auto t_div_normal =
             getFTensor0FromVec(*commonSurfaceSmoothingElement->divNormalField);

         auto t_lag_mult =
             getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

         auto t_w = getFTensor0IntegrationWeight();

         ///
         FTensor::Index<'k', 3> k;
         FTensor::Index<'q', 2> q;

         auto t_1 = getFTensor1Tangent1();
         auto t_2 = getFTensor1Tangent2();

         const double size_1 = sqrt(t_1(i) * t_1(i));
         const double size_2 = sqrt(t_2(i) * t_2(i));
         t_1(i) /= size_1;
         t_2(i) /= size_2;

         FTensor::Tensor1<double, 3> t_normal;

         t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);

         // FTensor::Tensor2<const double *, 3, 3> t_m(
         //     &t_1(0), &t_2(0), &t_normal(0),

         //     &t_1(1), &t_2(1), &t_normal(1),

         //     &t_1(2), &t_2(2), &t_normal(2),

         //     3);

         FTensor::Tensor2<const double *, 3, 3> t_m(&t_1(0), &t_1(1), &t_1(2),

                                                    &t_2(0), &t_2(1), &t_2(2),

                                                    &t_normal(0), &t_normal(1),
                                                    &t_normal(2),

                                                    3);

         double det;
         FTensor::Tensor2<double, 3, 3> t_inv_m;
         CHKERR determinantTensor3by3(t_m, det);
         CHKERR invertTensor3by3(t_m, det, t_inv_m);
         // FTensor::Tensor2<double, 3, 2> t_container_N;
         // FTensor::Tensor2<double, 3, 2> t_transformed_N;

         // auto t_1 =
         //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
         // auto t_2 =
         //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
         auto t_area =
             getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);

         FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 2> t_container_N(
             &t_inv_m(0, 0), &t_inv_m(0, 1), &t_inv_m(1, 0),

             &t_inv_m(1, 1), &t_inv_m(2, 0), &t_inv_m(2, 1));

         FTensor::Tensor1<double, 3> t_transformed_N;

         for (int gg = 0; gg != nb_gauss_pts; ++gg) {
           // const double norm_1 = sqrt(t_1(i) * t_1(i));
           // const double norm_2 = sqrt(t_2(i) * t_2(i));
           // t_1(i) /= norm_1;
           // t_2(i) /= norm_2;
           auto t_N = data.getFTensor1DiffN<2>(gg, 0);
           double val_m = t_w * area_m;
           // double val_m = t_w * t_area;
           auto t_base = data.getFTensor0N(gg, 0);

           FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
               &vecF[0], &vecF[1], &vecF[2]};

           for (int bbc = 0; bbc != nb_base_fun_col / 3; ++bbc) {

             // t_container_N(0,0) = t_container_N(1,0) = t_container_N(2,0) =
             // t_N(0); t_container_N(2,0) = t_container_N(2,1) =
             // t_container_N(2,1) = t_N(1);
             t_transformed_N(i) = t_N(q) * t_container_N(i, q);
             // t_transformed_N(i,k) = t_inv_m(j, i) * t_container_N(j, k);
             // t_assemble_m(0) += t_transformed_N(0, 0) * t_div_normal * val_m;
             // t_assemble_m(1) += t_transformed_N(1, 1) * t_div_normal * val_m;
           
           //Divergence
            //  t_assemble_m(i) +=  t_transformed_N(i) * t_div_normal * val_m;
           
          //  cerr << cnValue <<"\n";
             t_assemble_m(i) -= (cnValue * (t_n(i) - t_n_field(i)) /*+ t_lag_mult(i)*/ ) * t_base * val_m;
             
            //  t_assemble_m(i) += (t_n_field(i) - t_n(i) )* t_base * val_m;

             

             ++t_assemble_m;
             ++t_base;
             ++t_N;
           }

           ++t_w;
           ++t_n_field;
           ++t_div_normal;
           ++t_lag_mult;
           // ++t_1;
           // ++t_2;
           ++t_n;
           ++t_area;
        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
    double cnValue;
  };

struct OpCalTangentOneFieldRhs : public FaceEleOp {

    OpCalTangentOneFieldRhs(
        const string field_name_h_div,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name_h_div, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing), cnValue(cn_value) {
            sYmm = false; // This will make sure to loop over all entities
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;
        FTensor::Index<'l', 3> l;

         auto t_tan_1_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);

        //  auto t_tan_2_field =
        //   getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

         auto t_tan_1 =
             getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);

        auto t_tan_2 =
             getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);

        //  auto t_div_tan_1 =
        //      getFTensor0FromVec(*commonSurfaceSmoothingElement->divTangentOneField);

        // auto t_div_tan_2 =
        //      getFTensor0FromVec(*commonSurfaceSmoothingElement->divTangentTwoField);

        auto make_vec_der_1 = [&](double t_N,
                                FTensor::Tensor1<double *, 3> t_1,
                                FTensor::Tensor1<double *, 3> t_2,
                                FTensor::Tensor1<double *, 3> t_normal) {
          FTensor::Tensor2<double, 3, 3> t_n;
          FTensor::Tensor2<double, 3, 3> t_n_container;
          const double n_norm = sqrt(t_normal(i) * t_normal(i));
          t_n(i, j) = 0;
          t_n_container(i, j) = 0;
          t_n_container(i, j) +=
              FTensor::levi_civita(i, j, k) * t_2(k) * t_N;
          // t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
          //                                                t_n_container(j, k) /
          //                                                pow(n_norm, 3);
          return t_n_container;
        };
        auto t_w = getFTensor0IntegrationWeight();

        ///
        // FTensor::Index<'k', 3> k;
        FTensor::Index<'q', 2> q;

        auto t_1 = getFTensor1Tangent1();
        auto t_2 = getFTensor1Tangent2();

        const double size_1 = sqrt(t_1(i) * t_1(i));
        const double size_2 = sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;

        FTensor::Tensor1<double, 3> t_normal;

        t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);

        // FTensor::Tensor2<const double *, 3, 3> t_m(
        //     &t_1(0), &t_2(0), &t_normal(0),

        //     &t_1(1), &t_2(1), &t_normal(1),

        //     &t_1(2), &t_2(2), &t_normal(2),

        //     3);

        FTensor::Tensor2<const double *, 3, 3> t_m(&t_1(0), &t_1(1), &t_1(2),

                                                   &t_2(0), &t_2(1), &t_2(2),

                                                   &t_normal(0), &t_normal(1),
                                                   &t_normal(2),

                                                   3);

        double det;
        FTensor::Tensor2<double, 3, 3> t_inv_m;
        CHKERR determinantTensor3by3(t_m, det);
        CHKERR invertTensor3by3(t_m, det, t_inv_m);
        // FTensor::Tensor2<double, 3, 2> t_container_N;
        // FTensor::Tensor2<double, 3, 2> t_transformed_N;

        // auto t_1 =
        //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
        // auto t_2 =
        //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
        auto t_area =
            getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);

        FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 2> t_container_N(
            &t_inv_m(0, 0), &t_inv_m(0, 1), &t_inv_m(1, 0),

            &t_inv_m(1, 1), &t_inv_m(2, 0), &t_inv_m(2, 1));

        FTensor::Tensor1<double, 3> t_transformed_N;
        // auto t_tangent_one_grad = getFTensor2FromMat<3, 3>(*commonSurfaceSmoothingElement->tAngentOneGradField);
  
        auto t_n =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);
        // auto t_n_field =
        //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

      // auto t_1_unit =
      //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1Unit);
      // auto t_2_unit =
      //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2Unit);

      auto t_div_tan_1 =
      getFTensor0FromVec(*commonSurfaceSmoothingElement->divTangentOneField);
      auto t_rot_normal = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->rotTangentOneField);


      auto t_1_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);
      auto t_2_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

      auto t_tan_tan_one_one = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanOneOneField);

      auto t_tan_tan_one_two = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanOneTwoField);

      auto t_tan_tan_two_one = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanTwoOneField);

      auto t_tan_tan_two_two = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanTwoTwoField);

      auto t_curvature =
          getFTensor1FromMat<2>(*commonSurfaceSmoothingElement->curvatuersField);
      auto t_n_field = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);
      
      for (int gg = 0; gg != nb_gauss_pts; ++gg) {
        // const double norm_1 = sqrt(t_1(i) * t_1(i));
        // const double norm_2 = sqrt(t_2(i) * t_2(i));
        // t_1(i) /= norm_1;
        // t_2(i) /= norm_2;
        auto t_N = data.getFTensor1DiffN<2>(gg, 0);
        double val_m = t_w * area_m;
        //  double val_m = t_w * t_area;
        auto t_base = data.getFTensor0N(gg, 0);

        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
            &vecF[0], &vecF[1], &vecF[2]};

        const double fund_e = t_1_field(i) * t_1_field(i);
        const double fund_f = t_1_field(i) * t_2_field(i);
        const double fund_g = t_2_field(i) * t_2_field(i);
        const double fund_l = t_tan_tan_one_one(i) * t_n_field(i);
        const double fund_m_1 = t_tan_tan_one_two(i) * t_n_field(i);
        const double fund_m_2 = t_tan_tan_two_one(i) * t_n_field(i);
        
        const double fund_n = t_tan_tan_two_two(i) * t_n_field(i);

// cerr << "~~~~~~~~~~~~~~~~~~~~~" << "\n";
// cerr << "l " << fund_l << "\n";
// cerr << "n " << fund_n << "\n";
// cerr << "m1 " << fund_m_1 << "\n";
// cerr << "m2 " << fund_m_2 << "\n";
// cerr << "e " << fund_e << "\n";
// cerr << "g " << fund_g << "\n";
// cerr << "f " << fund_f << "\n";
// cerr << "~~~~~~~~~~~~~~~~~~~~~" << "\n";


        const double minor_k_l =
            (0.5 * fund_g) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (-(fund_n / (-pow(fund_f, 2) + fund_e * fund_g)) +
             (0.5 * fund_g *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_n =
            (0.5 * fund_e) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (-(fund_l / (-pow(fund_f, 2) + fund_e * fund_g)) +
             (0.5 * fund_e *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_m1 =
            (-0.5 * fund_f) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (fund_m_2 / (-pow(fund_f, 2) + fund_e * fund_g) -
             (0.5 * fund_f *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_m2 =
            (-0.5 * fund_f) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (fund_m_1 / (-pow(fund_f, 2) + fund_e * fund_g) -
             (0.5 * fund_f *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_e =
            (0.5 * fund_n) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (0.5 * fund_g *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
            ((0.5 * fund_n *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
             (0.5 * fund_g *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) +
             (fund_g * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_g =
            (0.5 * fund_l) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (0.5 * fund_e *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
            ((0.5 * fund_l *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
             (0.5 * fund_e *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) +
             (fund_e * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_f =
            (0.5 * (-fund_m_1 - fund_m_2)) /
                (-pow(fund_f, 2) + fund_e * fund_g) +
            (1. * fund_f *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
            ((0.5 * (-fund_m_1 - fund_m_2) *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
             (1. * fund_f *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) -
             (2 * fund_f * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_l =
            (0.5 * fund_g) / (-pow(fund_f, 2) + fund_e * fund_g) +
            (-(fund_n / (-pow(fund_f, 2) + fund_e * fund_g)) +
             (0.5 * fund_g *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_n =
            (0.5 * fund_e) / (-pow(fund_f, 2) + fund_e * fund_g) +
            (-(fund_l / (-pow(fund_f, 2) + fund_e * fund_g)) +
             (0.5 * fund_e *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_m1 =
            (-0.5 * fund_f) / (-pow(fund_f, 2) + fund_e * fund_g) +
            (fund_m_2 / (-pow(fund_f, 2) + fund_e * fund_g) -
             (0.5 * fund_f *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_m2 =
            (-0.5 * fund_f) / (-pow(fund_f, 2) + fund_e * fund_g) +
            (fund_m_1 / (-pow(fund_f, 2) + fund_e * fund_g) -
             (0.5 * fund_f *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_e =
            (0.5 * fund_n) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (0.5 * fund_g *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
            ((0.5 * fund_n *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
             (0.5 * fund_g *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) +
             (fund_g * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_g =
            (0.5 * fund_l) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (0.5 * fund_e *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
            ((0.5 * fund_l *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
             (0.5 * fund_e *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) +
             (fund_e * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_f =
            (0.5 * (-fund_m_1 - fund_m_2)) /
                (-pow(fund_f, 2) + fund_e * fund_g) +
            (1. * fund_f *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
            ((0.5 * (-fund_m_1 - fund_m_2) *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
             (1. * fund_f *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) -
             (2 * fund_f * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));


// cerr << "l " << minor_k_l<< "\n";
// cerr << "n " << minor_k_n<< "\n";
// cerr << "m1 " << minor_k_m1<< "\n";
// cerr << "m2 " << minor_k_m2<< "\n";
// cerr << "e " << minor_k_e<< "\n";
// cerr << "g " << minor_k_g<< "\n";
// cerr << "f " << minor_k_f<< "\n";

        for (int bbc = 0; bbc != nb_base_fun_col / 3; ++bbc) {

          // double mag_tan_1 = sqrt(t_tan_1(i) * t_tan_1(i));
          // t_assemble_m(i) -=
          //     cnValue * (t_n_field(i) - t_tan_1_field(i)) * t_base * val_m;
          // t_assemble_m(i) += cnValue *
          //                    (t_tan_1_field(j) * (t_tan_1(j) + t_tan_2(j))) *
          //                    (t_tan_1(i) + t_tan_2(i)) * t_base * val_m;

          // t_assemble_m(i) += cnValue * (t_n_field(j) * t_tan_1_field(j) - 1.) *
          //                    t_n_field(i) * t_base * val_m;

          //tangent
          t_assemble_m(i) -= cnValue * (t_tan_1(i) - t_1_field(i)) *
                              t_base * val_m;

          //curvature
          if(t_1_field(i) * t_1_field(i) > 1.e-8){


//             cerr << "~~~~~~~~~~~~~~~~~~~~~   " << t_1_field(i) * t_1_field(i) << "\n";
// cerr << "l " << fund_l << "\n";
// cerr << "n " << fund_n << "\n";
// cerr << "m1 " << fund_m_1 << "\n";
// cerr << "m2 " << fund_m_2 << "\n";
// cerr << "e " << fund_e << "\n";
// cerr << "g " << fund_g << "\n";
// cerr << "f " << fund_f << "\n";
// cerr << "~~~~~~~~~~~~~~~~~~~~~" << "\n";

          t_assemble_m(i) += ( major_k_l * t_curvature(0) + minor_k_l * t_curvature(1) ) * (
              t_N(0) * t_n_field(i) + t_base * t_tan_tan_one_one(l) *
                                          FTensor::levi_civita(l, i, k) *
                                          t_2_field(k)) * val_m;

          t_assemble_m(i) +=
              (major_k_m1 * t_curvature(0) + minor_k_m1 * t_curvature(1)) *
              (t_N(1) * t_n_field(i) + t_base * t_tan_tan_two_one(l) *
                                           FTensor::levi_civita(l, i, k) *
                                           t_2_field(k)) * val_m;

          t_assemble_m(i) +=
              (major_k_m2 * t_curvature(0) + minor_k_m2 * t_curvature(1)) *
              t_base * t_tan_tan_one_two(l) * FTensor::levi_civita(l, i, k) *
              t_2_field(k) * val_m;

          t_assemble_m(i) +=
              (major_k_n * t_curvature(0) + minor_k_n * t_curvature(1)) *
              t_base * t_tan_tan_two_two(l) * FTensor::levi_civita(l, i, k) *
              t_2_field(k) * val_m;

          t_assemble_m(i) +=
              2 * (major_k_e * t_curvature(0) + minor_k_e * t_curvature(1)) *
              t_base *
              t_1_field(i)* val_m;

          t_assemble_m(i) +=
              (major_k_e * t_curvature(0) + minor_k_e * t_curvature(1)) *
              t_base *
              t_2_field(i)* val_m;
          }

          //  t_assemble_m(i) -= cnValue * (t_1_unit(i) - t_tan_1_field(i)) *
          //  t_base * val_m; t_assemble_m(i) += t_tan_2_field(i) * t_base *
          //  val_m; t_assemble_m(j) -= cnValue * (t_n(i) - t_n_field(i)) *
          //  t_d_n(i, j) * val_m;
          // t_assemble_m(i) -= cnValue * (t_2_unit(i) - t_tan_2_field(i)) *
          // t_base * val_m;
          //  cerr << "t_n   " << t_n << "\n";
          //  cerr << "t_d_n   " << t_d_n << "\n";

          //  t_assemble_m(i) += (t_tan_1_field(i) - t_tan_1(i) )* t_base *
          //  val_m;

          ++t_assemble_m;
          ++t_base;
          ++t_N;
        }

        ++t_w;
        ++t_tan_1_field;
        //  ++t_tan_2_field;
        //  ++t_div_tan_1;
        //  ++t_div_tan_2;
        //  ++t_lag_mult;
        // ++t_1;
        // ++t_2;
        ++t_tan_1;
        ++t_tan_2;
        ++t_area;
        //  ++t_tangent_one_grad;
        //  ++t_n_field;
        ++t_n;
        //  ++t_1_unit;
        //  ++t_2_unit;
        ++t_div_tan_1;
        ++t_rot_normal;

        ++t_tan_tan_one_one;
        ++t_tan_tan_two_one;
        ++t_tan_tan_one_two;
        ++t_n_field;
        ++t_tan_tan_two_two;
        ++t_curvature;
        ++t_1_field;
        ++t_2_field;

        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
    double cnValue;
  };


struct OpCalTangentTwoFieldRhs : public FaceEleOp {

    OpCalTangentTwoFieldRhs(
        const string field_name_h_div,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing, double cn_value)
        : FaceEleOp(field_name_h_div, UserDataOperator::OPROW),
          commonSurfaceSmoothingElement(common_data_smoothing), cnValue(cn_value) {
            sYmm = false; // This will make sure to loop over all entities
          }

    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;

      const int nb_dofs = data.getIndices().size();
      if (nb_dofs) {

        const int nb_gauss_pts = data.getN().size1();
        int nb_base_fun_col = nb_dofs;

        vecF.resize(nb_dofs, false);
        vecF.clear();

        const double area_m = getMeasure(); // same area in master and slave

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;
        FTensor::Index<'l', 3> l;

         auto t_tan_2_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

         auto t_tan_1_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);

         auto t_tan_2 =
             getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);

        //  auto t_div_tan_1 =
        //      getFTensor0FromVec(*commonSurfaceSmoothingElement->divTangentOneField);

        //  auto t_div_tan_2 =
        //      getFTensor0FromVec(*commonSurfaceSmoothingElement->divTangentTwoField);

          auto t_n =
             getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);



          auto make_vec_der_2 = [&](double t_N,
                                FTensor::Tensor1<double *, 3> t_1,
                                FTensor::Tensor1<double *, 3> t_2,
                                FTensor::Tensor1<double *, 3> t_normal) {
          FTensor::Tensor2<double, 3, 3> t_n;
          FTensor::Tensor2<double, 3, 3> t_n_container;
          const double n_norm = sqrt(t_normal(i) * t_normal(i));
          t_n(i, j) = 0;
          t_n_container(i, j) = 0;
          t_n_container(i, k) +=
              FTensor::levi_civita(i, j, k) * t_1(j) * t_N;
          // t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
          //                                                t_n_container(j, k) /
          //                                                pow(n_norm, 3);
          return t_n_container;
        };

        //  auto t_lag_mult =
        //      getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->lagMult);

         auto t_w = getFTensor0IntegrationWeight();

         ///
        //  FTensor::Index<'k', 3> k;
         FTensor::Index<'q', 2> q;

         auto t_1 = getFTensor1Tangent1();
         auto t_2 = getFTensor1Tangent2();

         const double size_1 = sqrt(t_1(i) * t_1(i));
         const double size_2 = sqrt(t_2(i) * t_2(i));
         t_1(i) /= size_1;
         t_2(i) /= size_2;

         FTensor::Tensor1<double, 3> t_normal;

         t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);

         // FTensor::Tensor2<const double *, 3, 3> t_m(
         //     &t_1(0), &t_2(0), &t_normal(0),

         //     &t_1(1), &t_2(1), &t_normal(1),

         //     &t_1(2), &t_2(2), &t_normal(2),

         //     3);

         FTensor::Tensor2<const double *, 3, 3> t_m(&t_1(0), &t_1(1), &t_1(2),

                                                    &t_2(0), &t_2(1), &t_2(2),

                                                    &t_normal(0), &t_normal(1),
                                                    &t_normal(2),

                                                    3);

         double det;
         FTensor::Tensor2<double, 3, 3> t_inv_m;
         CHKERR determinantTensor3by3(t_m, det);
         CHKERR invertTensor3by3(t_m, det, t_inv_m);
         // FTensor::Tensor2<double, 3, 2> t_container_N;
         // FTensor::Tensor2<double, 3, 2> t_transformed_N;

         // auto t_1 =
         //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
         // auto t_2 =
         //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
         auto t_area =
             getFTensor0FromVec(*commonSurfaceSmoothingElement->areaNL);

         FTensor::Tensor2<FTensor::PackPtr<double *, 6>, 3, 2> t_container_N(
             &t_inv_m(0, 0), &t_inv_m(0, 1), &t_inv_m(1, 0),

             &t_inv_m(1, 1), &t_inv_m(2, 0), &t_inv_m(2, 1));

        //  FTensor::Tensor1<double, 3> t_transformed_N;
        // auto t_tangent_two_grad = getFTensor2FromMat<3, 3>(*commonSurfaceSmoothingElement->tAngentTwoGradField);

        auto t_2_unit =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2Unit);
        auto t_1_unit =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1Unit);




      auto t_1_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentOneField);
      auto t_2_field =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngentTwoField);

      auto t_tan_tan_one_one = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanOneOneField);

      auto t_tan_tan_one_two = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanOneTwoField);

      auto t_tan_tan_two_one = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanTwoOneField);

      auto t_tan_tan_two_two = getFTensor1FromMat<3>(
          *commonSurfaceSmoothingElement->tAnTanTwoTwoField);

      auto t_curvature =
          getFTensor1FromMat<2>(*commonSurfaceSmoothingElement->curvatuersField);
      auto t_n_field = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmalField);

      FTensor::Tensor1<double, 3> t_tan_tan_2_1;

        for (int gg = 0; gg != nb_gauss_pts; ++gg) {

        const double fund_e = t_1_field(i) * t_1_field(i);
        const double fund_f = t_1_field(i) * t_2_field(i);
        const double fund_g = t_2_field(i) * t_2_field(i);
        const double fund_l = t_tan_tan_one_one(i) * t_n_field(i);
        const double fund_m_1 = t_tan_tan_one_two(i) * t_n_field(i);
        const double fund_m_2 = t_tan_tan_two_one(i) * t_n_field(i);
        
        const double fund_n = t_tan_tan_two_two(i) * t_n_field(i);

        const double minor_k_l =
            (0.5 * fund_g) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (-(fund_n / (-pow(fund_f, 2) + fund_e * fund_g)) +
             (0.5 * fund_g *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_n =
            (0.5 * fund_e) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (-(fund_l / (-pow(fund_f, 2) + fund_e * fund_g)) +
             (0.5 * fund_e *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_m1 =
            (-0.5 * fund_f) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (fund_m_2 / (-pow(fund_f, 2) + fund_e * fund_g) -
             (0.5 * fund_f *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_m2 =
            (-0.5 * fund_f) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (fund_m_1 / (-pow(fund_f, 2) + fund_e * fund_g) -
             (0.5 * fund_f *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_e =
            (0.5 * fund_n) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (0.5 * fund_g *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
            ((0.5 * fund_n *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
             (0.5 * fund_g *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) +
             (fund_g * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_g =
            (0.5 * fund_l) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (0.5 * fund_e *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
            ((0.5 * fund_l *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
             (0.5 * fund_e *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) +
             (fund_e * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double minor_k_f =
            (0.5 * (-fund_m_1 - fund_m_2)) /
                (-pow(fund_f, 2) + fund_e * fund_g) +
            (1. * fund_f *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
            ((0.5 * (-fund_m_1 - fund_m_2) *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
             (1. * fund_f *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) -
             (2 * fund_f * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_l =
            (0.5 * fund_g) / (-pow(fund_f, 2) + fund_e * fund_g) +
            (-(fund_n / (-pow(fund_f, 2) + fund_e * fund_g)) +
             (0.5 * fund_g *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_n =
            (0.5 * fund_e) / (-pow(fund_f, 2) + fund_e * fund_g) +
            (-(fund_l / (-pow(fund_f, 2) + fund_e * fund_g)) +
             (0.5 * fund_e *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_m1 =
            (-0.5 * fund_f) / (-pow(fund_f, 2) + fund_e * fund_g) +
            (fund_m_2 / (-pow(fund_f, 2) + fund_e * fund_g) -
             (0.5 * fund_f *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_m2 =
            (-0.5 * fund_f) / (-pow(fund_f, 2) + fund_e * fund_g) +
            (fund_m_1 / (-pow(fund_f, 2) + fund_e * fund_g) -
             (0.5 * fund_f *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_e =
            (0.5 * fund_n) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (0.5 * fund_g *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
            ((0.5 * fund_n *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
             (0.5 * fund_g *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) +
             (fund_g * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_g =
            (0.5 * fund_l) / (-pow(fund_f, 2) + fund_e * fund_g) -
            (0.5 * fund_e *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
            ((0.5 * fund_l *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
             (0.5 * fund_e *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) +
             (fund_e * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));

        const double major_k_f =
            (0.5 * (-fund_m_1 - fund_m_2)) /
                (-pow(fund_f, 2) + fund_e * fund_g) +
            (1. * fund_f *
             (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
              fund_e * fund_n)) /
                pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
            ((0.5 * (-fund_m_1 - fund_m_2) *
              (fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
               fund_e * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2) +
             (1. * fund_f *
              pow(fund_g * fund_l - fund_f * fund_m_1 - fund_f * fund_m_2 +
                      fund_e * fund_n,
                  2)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 3) -
             (2 * fund_f * (-(fund_m_1 * fund_m_2) + fund_l * fund_n)) /
                 pow(-pow(fund_f, 2) + fund_e * fund_g, 2)) /
                (2. * sqrt((0.25 * pow(fund_g * fund_l - fund_f * fund_m_1 -
                                           fund_f * fund_m_2 + fund_e * fund_n,
                                       2)) /
                               pow(-pow(fund_f, 2) + fund_e * fund_g, 2) -
                           (-(fund_m_1 * fund_m_2) + fund_l * fund_n) /
                               (-pow(fund_f, 2) + fund_e * fund_g)));



          auto t_N = data.getFTensor1DiffN<2>(gg, 0);
          double val_m = t_w * area_m;
          // double val_m = t_w * t_area;
          auto t_base = data.getFTensor0N(gg, 0);

          FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_assemble_m{
              &vecF[0], &vecF[1], &vecF[2]};

          for (int bbc = 0; bbc != nb_base_fun_col / 3; ++bbc) {

            // t_container_N(0,0) = t_container_N(1,0) = t_container_N(2,0) =
            // t_N(0); t_container_N(2,0) = t_container_N(2,1) =
            // t_container_N(2,1) = t_N(1);
            
            //for divergence
            // t_transformed_N(i) = t_N(q) * t_container_N(i, q);
            
            
            // t_transformed_N(i,k) = t_inv_m(j, i) * t_container_N(j, k);
            // t_assemble_m(0) += t_transformed_N(0, 0) * t_div_normal * val_m;
            // t_assemble_m(1) += t_transformed_N(1, 1) * t_div_normal * val_m;

            // Divergence
            //  t_assemble_m(i) +=  t_transformed_N(i) * t_div_tan_2 * val_m;
            //  t_assemble_m(k) +=  FTensor::levi_civita(i, j, k) *
            //  (t_transformed_N(i) *t_tan_1_field(j) + t_tangent_two_grad(j,i)
            //  * t_N(1)) * (t_div_tan_1 + t_div_tan_2) * val_m;

            // auto t_d_n =
            //     make_vec_der_2(t_base, t_tan_1_field, t_tan_2_field, t_n_field);
            //  cerr << cnValue <<"\n";
            // double mag_tan_2 = sqrt(t_tan_2(i) * t_tan_2(i));
            
            // t_assemble_m(i) += t_tan_1_field(i) * t_base * val_m;

            //  t_assemble_m(j) -= cnValue * (t_n(i) - t_n_field(i)) * t_d_n(i, j) * t_base * val_m;

            // t_assemble_m(i) -=
            //     cnValue *
            //      (t_1_unit(i) - t_tan_1_field(i)) *
            //     t_base * val_m;


            //  t_assemble_m(i) += (t_tan_2_field(i) - t_tan_2(i) )* t_base * val_m;
            
            //Tangent comparisons
            t_assemble_m(i) -=
                cnValue * (t_tan_2(i) - t_tan_2_field(i)) * t_base * val_m;

            //curvature
if(t_tan_2_field(i) * t_tan_2_field(i) > 1.e-8) {

//               cerr << "~~~~~~~~~~~~~~~~~~~~~   " << t_1_field(i) * t_1_field(i) << "\n";
// cerr << "l " << fund_l << "\n";
// cerr << "n " << fund_n << "\n";
// cerr << "m1 " << fund_m_1 << "\n";
// cerr << "m2 " << fund_m_2 << "\n";
// cerr << "e " << fund_e << "\n";
// cerr << "g " << fund_g << "\n";
// cerr << "f " << fund_f << "\n";
// cerr << "~~~~~~~~~~~~~~~~~~~~~" << "\n";


            t_assemble_m(i) +=
                (major_k_l * t_curvature(0) + minor_k_l * t_curvature(1)) *
                t_base * t_tan_tan_one_one(l) * FTensor::levi_civita(l, k, i) *
                                             t_1_field(k) *
                val_m;

            t_assemble_m(i) +=
                (major_k_m1 * t_curvature(0) + minor_k_m1 * t_curvature(1)) *
                t_base * t_tan_tan_one_two(l) * FTensor::levi_civita(l, k, i) *
                t_1_field(k) * val_m;


            t_assemble_m(i) +=
                (major_k_m2 * t_curvature(0) + minor_k_m2 * t_curvature(1)) *
                (t_N(0) * t_n_field(i) + t_base * t_tan_tan_two_one(l) *
                                             FTensor::levi_civita(l, k, i) *
                                             t_1_field(k)) *
                val_m;


            t_assemble_m(i) +=
                (major_k_n * t_curvature(0) + minor_k_n * t_curvature(1)) *
                (t_N(1) * t_n_field(i) + t_base * t_tan_tan_two_two(l) *
                                             FTensor::levi_civita(l, i, k) *
                                             t_1_field(k)) *
                val_m;

            t_assemble_m(i) +=
                (major_k_f * t_curvature(0) + minor_k_f * t_curvature(1)) *
                t_base * t_1_field(i) * val_m;

            t_assemble_m(i) +=
                2 * (major_k_g * t_curvature(0) + minor_k_g * t_curvature(1)) *
                t_base * t_2_field(i) * val_m;

          }

            ++t_assemble_m;
            ++t_base;
            ++t_N;
           }

           ++t_w;
           ++t_tan_1_field;
           ++t_tan_2_field;
          //  ++t_div_tan_2;
          //  ++t_div_tan_1;
          //  ++t_lag_mult;
           // ++t_1;
           // ++t_2;
           ++t_tan_2;
           ++t_area;
          //  ++t_tangent_two_grad;
           ++t_n;
           ++t_2_unit;
           ++t_1_unit;

        ++t_tan_tan_one_one;
        ++t_tan_tan_two_one;

        ++t_tan_tan_one_two;
        ++t_n_field;
        ++t_tan_tan_two_two;
        ++t_curvature;
        ++t_1_field;
        ++t_2_field;
        } // for gauss points

        CHKERR VecSetValues(getSNESf(), data, &*vecF.begin(), ADD_VALUES);
      }
      MoFEMFunctionReturn(0);
    }

  private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    VectorDouble vecF;
    double cnValue;
  };

  struct OpCalSmoothingNormalHdivHdivLhs : public FaceEleOp {

    OpCalSmoothingNormalHdivHdivLhs(
        const string field_name_h_div,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name_h_div, UserDataOperator::OPROWCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
            sYmm = false; // This will make sure to loop over all entities
          }

  MoFEMErrorCode doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

    const int nb_gauss_pts = row_data.getN().size1();
    int nb_base_fun_row = row_data.getFieldData().size() / 2;
    int nb_base_fun_col = col_data.getFieldData().size() / 2;

    const double area = getMeasure();

    NN.resize(2 * nb_base_fun_row, 2 * nb_base_fun_col, false);
    NN.clear();

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 2, 2>(&m(r + 0, c + 0), &m(r + 0, c + 1),
                                           &m(r + 1, c + 0), &m(r + 1, c + 1));
    };

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;
        FTensor::Index<'m', 3> m;
        FTensor::Index<'l', 3> l;
        FTensor::Index<'q', 2> q;


///////


        // FTensor::Index<'i', 3> i;
        // FTensor::Index<'j', 3> j;
        // FTensor::Index<'k', 3> k;
        // FTensor::Index<'m', 3> m;
        // FTensor::Index<'l', 3> l;
        // FTensor::Index<'q', 2> q;
        // auto t_divergence_n_h_div =
        //     getFTensor0FromVec(*commonSurfaceSmoothingElement->divNormalHdiv);

        // auto t_w = getFTensor0IntegrationWeight();

        // auto t_1 = getFTensor1Tangent1();        
        // auto t_2 = getFTensor1Tangent2();        

        // const double size_1 =  sqrt(t_1(i) * t_1(i));
        // const double size_2 =  sqrt(t_2(i) * t_2(i));
        // t_1(i) /= size_1;
        // t_2(i) /= size_2;

        // auto t_t_1_div =
        //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv1);

        // auto t_t_2_div =
        //     getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv2);

        // FTensor::Tensor2<double, 3, 2> t_div_1, t_div_2;
        // FTensor::Tensor2<double, 3, 3> t_mat_1, t_mat_2;

        // for (int gg = 0; gg != nb_gauss_pts; ++gg) {

        //   double val_m = t_w * area_m;

        //   auto t_diff_base_fun = data.getFTensor2DiffN<3, 2>();
        //   FTensor::Tensor1<FTensor::PackPtr<double *, 2>, 2> t_assemble_m{
        //   &vecF[0], &vecF[1]};

        //   for (int bbc = 0; bbc != nb_base_fun_col / 2; ++bbc) {
        //     // vecF[bbc] += t_row_diff_base(i, i) * t_divergence_n_h_div * val_m;
        //     t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_1_div(k);
        //     t_div_1(i, q) =
        //         t_mat_1(i, j) * t_diff_base_fun(m, q) * t_1(m) * t_1(j);
        //     t_assemble_m(0) +=
        //         (t_div_1(0, 0) + t_div_1(1, 1)) * t_divergence_n_h_div * val_m;
        //           t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_1_div(k);

///////


        auto t_t_1_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv1);

        auto t_t_2_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv2);

        auto t_div_tan_1 = getFTensor2FromMat<3, 2>(*commonSurfaceSmoothingElement->matForDivTan1);
        auto t_div_tan_2 = getFTensor2FromMat<3, 2>(*commonSurfaceSmoothingElement->matForDivTan2);

        auto t_divergence_n_h_div =
            getFTensor0FromVec(*commonSurfaceSmoothingElement->divNormalHdiv);

        auto t_1 = getFTensor1Tangent1();        
        auto t_2 = getFTensor1Tangent2();        

        const double size_1 =  sqrt(t_1(i) * t_1(i));
        const double size_2 =  sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;

        FTensor::Tensor2<double, 3, 2> t_div_1_row, t_div_1_col, t_div_2_row, t_div_2_col;
        FTensor::Tensor2<double, 3, 3> t_mat_1, t_mat_2;


    auto t_w = getFTensor0IntegrationWeight();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      const double val_m = t_w * area;
      auto t_row_diff_base = row_data.getFTensor2DiffN<3, 2>();
      auto t_base_row = row_data.getFTensor1N<3>(gg, 0);

      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_col_diff_base = col_data.getFTensor2DiffN<3, 2>();
        auto t_base_col = col_data.getFTensor1N<3>(gg, 0);

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {

        auto t_assemble_m = get_tensor_from_mat(NN, 2 * bbr, 2 * bbc);
          
          t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_2_div(k);
          t_div_1_row(i, q) =
          t_mat_1(i, j) * t_row_diff_base(m, q) * t_1(m) * t_1(j);

          t_mat_1(i, k) =
              FTensor::levi_civita(i, j, k) * (t_base_row(m) * t_1(m) * t_1(j));
          t_div_1_row(i, q) += t_mat_1(i, k) * t_div_tan_2(k, q);

          t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_2_div(k);
          t_div_1_col(i, q) =
          t_mat_1(i, j) * t_col_diff_base(m, q) * t_1(m) * t_1(j);

          t_mat_1(i, k) =
              FTensor::levi_civita(i, j, k) * (t_base_col(m) * t_1(m) * t_1(j));
          t_div_1_col(i, q) += t_mat_1(i, k) * t_div_tan_2(k, q);

          t_assemble_m(0, 0) +=
                (t_div_1_row(0, 0) + t_div_1_row(1, 1)) * (t_div_1_col(0, 0) + t_div_1_col(1, 1)) * val_m;

            t_mat_2(i, k) = FTensor::levi_civita(i, j, k) * t_t_1_div(j);
            t_div_2_row(i, q) =
                t_mat_2(i, k) * t_row_diff_base(m, q) * t_2(m) * t_2(k);
            
            t_mat_2(i, j) = FTensor::levi_civita(i, j, k) * (t_base_row(m) * t_2(m) * t_2(k));
            t_div_2_row(i, q) +=
                t_mat_2(i, j) * t_div_tan_1(j,q);

            t_mat_2(i, k) = FTensor::levi_civita(i, j, k) * t_t_1_div(j);
            t_div_2_col(i, q) =
                t_mat_2(i, k) * t_col_diff_base(m, q) * t_2(m) * t_2(k);
            
            t_mat_2(i, j) = FTensor::levi_civita(i, j, k) * (t_base_col(m) * t_2(m) * t_2(k));
            t_div_2_col(i, q) +=
                t_mat_2(i, j) * t_div_tan_1(j,q);

            t_assemble_m(1, 1) +=
                (t_div_2_row(0, 0) + t_div_2_row(1, 1)) * (t_div_2_col(0, 0) + t_div_2_col(1, 1)) * val_m;


          ++t_col_diff_base; // update cols
          ++t_base_col;
        }
        ++t_row_diff_base; // update rows
        ++t_base_row;
      }
      ++t_t_1_div;
      ++t_t_2_div;
      ++t_div_tan_1;
      ++t_div_tan_2;
      ++t_divergence_n_h_div;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

 private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    MatrixDouble NN;
  };

struct OpCalSmoothingX_dnLhs : public FaceEleOp {

    OpCalSmoothingX_dnLhs(
        const string field_name_position, const string field_name_h_div,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name_position, field_name_h_div, UserDataOperator::OPROWCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {
            sYmm = false; // This will make sure to loop over all entities
          }

  MoFEMErrorCode doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {
  MoFEMFunctionBegin;

  const int nb_row = row_data.getIndices().size();
  const int nb_col = col_data.getIndices().size();

  if (nb_row && nb_col) {

    const int nb_gauss_pts = row_data.getN().size1();
    int nb_base_fun_row = row_data.getFieldData().size() / 3;
    int nb_base_fun_col = col_data.getFieldData().size() / 2;

    const double area = getMeasure();

    NN.resize(3 * nb_base_fun_row, 2 * nb_base_fun_col, false);
    NN.clear();

    auto get_tensor_from_mat = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor1<double *, 3>(&m(r + 0, c), &m(r + 1, c),
                                           &m(r + 2, c));
    };

        auto t_1 = getFTensor1Tangent1();        
        auto t_2 = getFTensor1Tangent2();        
        auto t_t_1_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv1);

        auto t_t_2_div =
            getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tanHdiv2);

    FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;
        FTensor::Index<'m', 3> m;

        const double size_1 =  sqrt(t_1(i) * t_1(i));
        const double size_2 =  sqrt(t_2(i) * t_2(i));
        t_1(i) /= size_1;
        t_2(i) /= size_2;
        
    FTensor::Tensor2<double, 3, 3> t_mat_1, t_mat_2;
    FTensor::Tensor1<double, 3> t_vec_1, t_vec_2;

    auto t_w = getFTensor0IntegrationWeight();
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
     
      const double val_m = t_w * area;

      auto t_base_row_X = row_data.getFTensor0N(gg, 0);
      for (int bbr = 0; bbr != nb_base_fun_row; ++bbr) {

        auto t_base_col_hdiv = col_data.getFTensor1N<3>(gg, 0);
        const double m = val_m * t_base_row_X;

        for (int bbc = 0; bbc != nb_base_fun_col; ++bbc) {
          auto t_assemble_m = get_tensor_from_mat(NN, 3 * bbr, 2 * bbc);
          t_mat_1(i, j) = FTensor::levi_civita(i, j, k) * t_t_1_div(k);
           t_vec_1(j) = t_base_col_hdiv(m) * t_1(m) * t_1(j); 
          t_assemble_m(i) -=  m * t_mat_1(i, j) * t_vec_1(j);

          auto t_assemble_m_2 = get_tensor_from_mat(NN, 3 * bbr, 2 * bbc + 1);
          t_mat_2(i, j) = FTensor::levi_civita(i, j, k) * t_t_2_div(k);
           t_vec_2(j) = t_base_col_hdiv(m) * t_2(m) * t_2(j); 
          t_assemble_m_2(i) -=  m * t_mat_2(i, j) * t_vec_2(j);

          ++t_base_col_hdiv; // update rows
        }
        ++t_base_row_X; // update cols master
      }
      ++t_t_1_div;
      ++t_t_2_div;
      ++t_w;
    }

    CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}
private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    MatrixDouble NN;
};


struct OpCalSmoothingX_dXLhs : public FaceEleOp {

    OpCalSmoothingX_dXLhs(
        const string field_name_position,
        boost::shared_ptr<CommonSurfaceSmoothingElement> common_data_smoothing)
        : FaceEleOp(field_name_position, UserDataOperator::OPROWCOL),
          commonSurfaceSmoothingElement(common_data_smoothing) {}


  MoFEMErrorCode doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntData &row_data, EntData &col_data) {

  MoFEMFunctionBegin;


  const int row_nb_dofs = row_data.getIndices().size();
  if (!row_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int col_nb_dofs = col_data.getIndices().size();
  if (!col_nb_dofs)
    MoFEMFunctionReturnHot(0);
  const int nb_gauss_pts = row_data.getN().size1();

  const int nb_base_fun_row = row_data.getFieldData().size() / 3;
  const int nb_base_fun_col = col_data.getFieldData().size() / 3;

  NN.resize(3 * nb_base_fun_row, 3 * nb_base_fun_col, false);
  NN.clear();

  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;

  auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
    return FTensor::Tensor2<double *, 3, 3>(
        &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2), &m(r + 1, c + 0),
        &m(r + 1, c + 1), &m(r + 1, c + 2), &m(r + 2, c + 0), &m(r + 2, c + 1),
        &m(r + 2, c + 2));
  };

  auto make_vec_der = [&](FTensor::Tensor1<double *, 2> t_N,
                          FTensor::Tensor1<double *, 3> t_1,
                          FTensor::Tensor1<double *, 3> t_2,
                          FTensor::Tensor1<double *, 3> t_normal) {
    FTensor::Tensor2<double, 3, 3> t_n;
    FTensor::Tensor2<double, 3, 3> t_n_container;
    const double n_norm = sqrt(t_normal(i) * t_normal(i));
    t_n(i, j) = 0;
    t_n_container(i, j) = 0;
    t_n_container(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n_container(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    t_n(i, k) = t_n_container(i, k) / n_norm - t_normal(i) * t_normal(j) *
                                                   t_n_container(j, k) /
                                                   pow(n_norm, 3);
    return t_n;
  };

  const double area = getMeasure();
  auto t_w = getFTensor0IntegrationWeight();
  auto t_1 = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
  auto t_2 = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
  auto t_normal = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

  for (int gg = 0; gg != nb_gauss_pts; ++gg) {

    double val = t_w * area;

    auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);

    int bbc = 0;
    for (; bbc != nb_base_fun_col; ++bbc) {

      auto t_base = row_data.getFTensor0N(gg, 0);

      int bbr = 0;
      for (; bbr != nb_base_fun_row; ++bbr) {

        auto t_d_n = make_vec_der(t_N, t_1, t_2, t_normal);

        auto t_assemble = get_tensor2(NN, 3 * bbr, 3 * bbc);
        // TODO: handle hoGeometry

        t_assemble(i, k) +=
            val * t_base * t_d_n(i, k);

        ++t_base;
      }
      ++t_N;
    }
    ++t_w;
    ++t_1;
    ++t_2;
    ++t_normal;
  }

CHKERR MatSetValues(getSNESB(), row_data, col_data, &*NN.data().begin(),
                        ADD_VALUES);

  MoFEMFunctionReturn(0);
}
private:
    boost::shared_ptr<CommonSurfaceSmoothingElement>
        commonSurfaceSmoothingElement;
    MatrixDouble NN;
};


  MoFEMErrorCode setSmoothFaceOperatorsRhs(
      boost::shared_ptr<SurfaceSmoothingElement> fe_rhs_smooth_element,
      boost::shared_ptr<CommonSurfaceSmoothingElement>
          common_data_smooth_element,
      string field_name_position, string field_name_tangent_one_field, string field_name_tangent_two_field, string lagrange_field_name) {
    MoFEMFunctionBegin;
    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpGetTangentForSmoothSurf(field_name_position,
                                      common_data_smooth_element));
    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpGetNormalForSmoothSurf(field_name_position,
                                     common_data_smooth_element));
    // fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetLagrangeField(
    //     lagrange_field_name, common_data_smooth_element));


    // fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetNormalField(
    //     field_name_normal_field, common_data_smooth_element));

    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetTangentOneField(
        field_name_tangent_one_field, common_data_smooth_element));

    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetTangentTwoField(
        field_name_tangent_two_field, common_data_smooth_element));

     fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalNormalFieldTwo(
         field_name_tangent_one_field, common_data_smooth_element));
         
     fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalPrincipalCurvatures(
         field_name_tangent_one_field, common_data_smooth_element));

///New
    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetDivOneField(
        field_name_tangent_one_field, common_data_smooth_element));

    // fe_rhs_smooth_element->getOpPtrVector().push_back(new OpGetDivTwoField(
    //     field_name_tangent_two_field, common_data_smooth_element));


    // fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalNormalFieldRhs(
    //     field_name_normal_field, common_data_smooth_element, cnVaule));


    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalTangentOneFieldRhs(
        field_name_tangent_one_field, common_data_smooth_element, cnVaule));

    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalTangentTwoFieldRhs(
        field_name_tangent_two_field, common_data_smooth_element, cnVaule));

    // fe_rhs_smooth_element->getOpPtrVector().push_back(
    //     new OpGetNormalFT(field_name_tangent_one_field,
    //                                  common_data_smooth_element));
    

    // fe_rhs_smooth_element->getOpPtrVector().push_back(
    //     new OpCalLagMultRhs(lagrange_field_name, common_data_smooth_element));

    // fe_rhs_smooth_element->getOpPtrVector().push_back(
    //     new OpCalPositionsRhs(field_name_position, common_data_smooth_element, cnVaule));
    


    
    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpCalPositionsForTanRhs(field_name_position, common_data_smooth_element, cnVaule));
    
    
    /////
    // fe_rhs_smooth_element->getOpPtrVector().push_back(
    //     new OpGetTangentHdivAtGaussPoints(field_name_normal_hdiv,
    //                                      common_data_smooth_element));

    // fe_rhs_smooth_element->getOpPtrVector().push_back(
    //     new OpGetNormalHdivAtGaussPoints(field_name_normal_hdiv,
    //                                      common_data_smooth_element));


    // Rhs
    // fe_rhs_smooth_element->getOpPtrVector().push_back(
    //     new OpCalSmoothingNormalHdivRhs(field_name_normal_hdiv,
    //                                     common_data_smooth_element));
    // fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalSmoothingXRhs(
    //     field_name_position, common_data_smooth_element));

    MoFEMFunctionReturn(0);
  }


  MoFEMErrorCode setSmoothFacePostProc(
      boost::shared_ptr<SurfaceSmoothingElement> fe_smooth_post_proc,
      boost::shared_ptr<CommonSurfaceSmoothingElement>
          common_data_smooth_element,
      string field_name_position, string field_name_tangent_one_field, string field_name_tangent_two_field, string lagrange_field_name) {
    MoFEMFunctionBegin;
    fe_smooth_post_proc->getOpPtrVector().push_back(
        new OpGetTangentForSmoothSurf(field_name_position,
                                      common_data_smooth_element));
    fe_smooth_post_proc->getOpPtrVector().push_back(
        new OpGetNormalForSmoothSurf(field_name_position,
                                     common_data_smooth_element));

    fe_smooth_post_proc->getOpPtrVector().push_back(new OpGetTangentOneField(
        field_name_tangent_one_field, common_data_smooth_element));

    // fe_smooth_post_proc->getOpPtrVector().push_back(new OpGetTangentTwoField(
    //     field_name_tangent_two_field, common_data_smooth_element));
    // fe_smooth_post_proc->getOpPtrVector().push_back(new OpGetNormalField(
    //     field_name_normal_field, common_data_smooth_element));

    fe_smooth_post_proc->getOpPtrVector().push_back(new OpPrintNormals(
        field_name_position, common_data_smooth_element));

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setGlobalVolumeEvaluator(
      boost::shared_ptr<VolumeSmoothingElement> fe_smooth_volume_element,
      string field_name_position) {
    MoFEMFunctionBegin;

    fe_smooth_volume_element->getOpPtrVector().push_back(
        new VolumeCalculation(field_name_position, volumeVec));

    MoFEMFunctionReturn(0);
  }


  MoFEM::Interface &mField;
  ArbitrarySmoothingProblem(MoFEM::Interface &m_field, double c_n) : mField(m_field), cnVaule(c_n) {}
};

// double MortarContactProblem::LoadScale::lAmbda = 1;
int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_divergence_tolerance 0 \n"
                                 "-snes_max_it 50 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-ksp_monitor \n"
                                 "-snes_converged_reason \n"
                                 "-my_order 2 \n"
                                 "-my_cn_value 1\n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  enum arbitrary_smoothing_tests { EIGHT_CUBE = 1, LAST_TEST };

  // Initialize MoFEM
  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  // Create mesh database
  moab::Core mb_instance;              // create database
  moab::Interface &moab = mb_instance; // create interface to database

  try {
    PetscBool flg_file;

    char mesh_file_name[255];
    PetscInt order = 2;
    PetscInt order_tangent_one = order;
    PetscInt order_tangent_two = order;
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool is_newton_cotes = PETSC_FALSE;
    PetscInt test_num = 0;
    PetscBool print_contact_state = PETSC_FALSE;
    PetscReal cn_value = 1.;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order",
                           "approximation order of spatial positions", "", 1,
                           &order, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_order_tangent_one",
                           "approximation order of spatial positions", "", 1,
                           &order_tangent_one, PETSC_NULL);
    CHKERR PetscOptionsInt("-my_order_tangent_two",
                           "approximation order of spatial positions", "", 1,
                           &order_tangent_two, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsReal("-my_cn_value", "default regularisation cn value",
                            "", 1., &cn_value, PETSC_NULL);

    CHKERR PetscOptionsInt("-my_test_num", "test number", "", 0, &test_num,
                           PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    // Check if mesh file was provided
    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    if (is_partitioned == PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "Partitioned mesh is not supported");
    }

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    MeshsetsManager *mmanager_ptr;
    CHKERR m_field.getInterface(mmanager_ptr);
    CHKERR mmanager_ptr->printDisplacementSet();
    CHKERR mmanager_ptr->printForceSet();
    // print block sets with materials
    CHKERR mmanager_ptr->printMaterialsSet();

    Range master_tris, slave_tris, all_tris_for_smoothing, all_tets;
    std::vector<BitRefLevel> bit_levels;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, 13, "MORTAR_MASTER") == 0) {
        CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTRI,
                                                       master_tris, true);
      }
    }

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, 12, "MORTAR_SLAVE") == 0) {
        rval = m_field.get_moab().get_entities_by_type(it->meshset, MBTRI,
                                                       slave_tris, true);
        CHKERRQ_MOAB(rval);
      }
    }

    all_tris_for_smoothing.merge(master_tris);
    all_tris_for_smoothing.merge(slave_tris);

    bit_levels.push_back(BitRefLevel().set(0));
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_levels.back());

    Range meshset_level0;
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
        bit_levels.back(), BitRefLevel().set(), meshset_level0);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "meshset_level0 %d\n",
                            meshset_level0.size());
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

// CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
//         bit_levels.back(), BitRefLevel().set(), MBTET, all_tets);

    EntityHandle meshset_surf_slave;
    CHKERR m_field.get_moab().create_meshset(MESHSET_SET, meshset_surf_slave);
    CHKERR m_field.get_moab().add_entities(meshset_surf_slave, slave_tris);

    CHKERR m_field.get_moab().write_mesh("surf_slave.vtk", &meshset_surf_slave,
                                         1);
    EntityHandle meshset_tri_slave;
    CHKERR m_field.get_moab().create_meshset(MESHSET_SET, meshset_tri_slave);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1,
                             AINSWORTH_LEGENDRE_BASE, 3, MB_TAG_SPARSE,
                             MF_ZERO);

    CHKERR m_field.add_field("TANGENT_ONE_FIELD", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("TANGENT_TWO_FIELD", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);


    CHKERR m_field.add_field("LAGMULT", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    Range fixed_vertex;
    CHKERR m_field.get_moab().get_connectivity(all_tris_for_smoothing,
                                               fixed_vertex);

    // Declare problem add entities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET,
                                             "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS",
                                   1);

    if (!slave_tris.empty()) {
      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "MESH_NODE_POSITIONS");
      CHKERR m_field.set_field_order(slave_tris, "MESH_NODE_POSITIONS", order);
      Range edges_smoothed;
      CHKERR moab.get_adjacencies(slave_tris, 1, false, edges_smoothed,
                              moab::Interface::UNION);
      CHKERR m_field.set_field_order(edges_smoothed, "MESH_NODE_POSITIONS", order);
      Range vertices_smoothed;
      CHKERR m_field.get_moab().get_connectivity(slave_tris,
                                               vertices_smoothed);
      CHKERR m_field.set_field_order(vertices_smoothed, "MESH_NODE_POSITIONS", order);

      //Normal
      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "TANGENT_ONE_FIELD");
      CHKERR m_field.set_field_order(slave_tris, "TANGENT_ONE_FIELD",  order_tangent_one );
      CHKERR m_field.set_field_order(edges_smoothed, "TANGENT_ONE_FIELD",  order_tangent_one );
      CHKERR m_field.set_field_order(vertices_smoothed, "TANGENT_ONE_FIELD",  order_tangent_one );

      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "TANGENT_TWO_FIELD");
      CHKERR m_field.set_field_order(slave_tris, "TANGENT_TWO_FIELD",  order_tangent_two );
      CHKERR m_field.set_field_order(edges_smoothed, "TANGENT_TWO_FIELD",  order_tangent_two );
      CHKERR m_field.set_field_order(vertices_smoothed, "TANGENT_TWO_FIELD",  order_tangent_two );

      //Lagmult
      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "LAGMULT");
      CHKERR m_field.set_field_order(slave_tris, "LAGMULT", 1);
      CHKERR m_field.set_field_order(edges_smoothed, "LAGMULT", 1);
      CHKERR m_field.set_field_order(vertices_smoothed, "LAGMULT", 1);

    }
    if (!master_tris.empty()) {
      CHKERR m_field.add_ents_to_field_by_type(master_tris, MBTRI,
                                               "MESH_NODE_POSITIONS");
      CHKERR m_field.set_field_order(master_tris, "MESH_NODE_POSITIONS", order);
      Range edges_smoothed;
      CHKERR moab.get_adjacencies(master_tris, 1, false, edges_smoothed,
                              moab::Interface::UNION);
      CHKERR m_field.set_field_order(edges_smoothed, "MESH_NODE_POSITIONS", order);
      Range vertices_smoothed;
      CHKERR m_field.get_moab().get_connectivity(master_tris,
                                               vertices_smoothed);
      CHKERR m_field.set_field_order(vertices_smoothed, "MESH_NODE_POSITIONS", order);

      //Normal
      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "TANGENT_ONE_FIELD");
      CHKERR m_field.set_field_order(slave_tris, "TANGENT_ONE_FIELD",  order_tangent_one );
      CHKERR m_field.set_field_order(edges_smoothed, "TANGENT_ONE_FIELD",  order_tangent_one );
      CHKERR m_field.set_field_order(vertices_smoothed, "TANGENT_ONE_FIELD",  order_tangent_one );

      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "TANGENT_TWO_FIELD");
      CHKERR m_field.set_field_order(slave_tris, "TANGENT_TWO_FIELD",  order_tangent_two );
      CHKERR m_field.set_field_order(edges_smoothed, "TANGENT_TWO_FIELD",  order_tangent_two );
      CHKERR m_field.set_field_order(vertices_smoothed, "TANGENT_TWO_FIELD",  order_tangent_two );

      //Lagmult
      CHKERR m_field.add_ents_to_field_by_type(master_tris, MBTRI,
                                               "LAGMULT");
      CHKERR m_field.set_field_order(master_tris, "LAGMULT", 1);
      CHKERR m_field.set_field_order(edges_smoothed, "LAGMULT", 1);
      CHKERR m_field.set_field_order(vertices_smoothed, "LAGMULT", 1);
    }

    

    // build field
    CHKERR m_field.build_fields();

    // Projection on "X" field
    {
      Projection10NodeCoordsOnField ent_method(m_field,
                                               "MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);
    }

    auto make_arbitrary_smooth_element_face = [&]() {
      return boost::make_shared<
          ArbitrarySmoothingProblem::SurfaceSmoothingElement>(m_field);
    };

    auto make_arbitrary_smooth_element_volume = [&]() {
      return boost::make_shared<
          ArbitrarySmoothingProblem::VolumeSmoothingElement>(m_field);
    };

    auto make_smooth_element_common_data = [&]() {
      return boost::make_shared<
          ArbitrarySmoothingProblem::CommonSurfaceSmoothingElement>(m_field);
    };

    auto get_smooth_face_rhs = [&](auto smooth_problem, auto make_element) {
      auto fe_rhs_smooth_face = make_element();
      auto common_data_smooth_elements = make_smooth_element_common_data();
      smooth_problem->setSmoothFaceOperatorsRhs(
          fe_rhs_smooth_face, common_data_smooth_elements,
          "MESH_NODE_POSITIONS", "TANGENT_ONE_FIELD", "TANGENT_TWO_FIELD", "LAGMULT");
      return fe_rhs_smooth_face;
    };

    auto get_smooth_volume_element_operators = [&](auto smooth_problem,
                                                   auto make_element) {
      auto fe_smooth_volumes = make_element();
      smooth_problem->setGlobalVolumeEvaluator(fe_smooth_volumes,
                                               "MESH_NODE_POSITIONS");
      return fe_smooth_volumes;
    };

    auto get_smooth_face_lhs = [&](auto smooth_problem, auto make_element) {
      auto fe_lhs_smooth_face = make_element();
      auto common_data_smooth_elements = make_smooth_element_common_data();
      // smooth_problem->setSmoothFaceOperatorsLhs(fe_lhs_smooth_face,
      //                                           common_data_smooth_elements,
      //                                           "MESH_NODE_POSITIONS", "NORMAL_HDIV");
      return fe_lhs_smooth_face;
    };

    auto get_smooth_face_post_prov = [&](auto smooth_problem, auto make_element) {
      auto fe_smooth_face_post_proc = make_element();
      auto common_data_smooth_elements = make_smooth_element_common_data();
      smooth_problem->setSmoothFacePostProc(
          fe_smooth_face_post_proc, common_data_smooth_elements,
          "MESH_NODE_POSITIONS", "TANGENT_ONE_FIELD", "TANGENT_TWO_FIELD", "LAGMULT");
      return fe_smooth_face_post_proc;
    };

    auto smooth_problem =
        boost::make_shared<ArbitrarySmoothingProblem>(m_field, cn_value);

    // add fields to the global matrix by adding the element

    smooth_problem->addSurfaceSmoothingElement(
        "SURFACE_SMOOTHING_ELEM", "MESH_NODE_POSITIONS", "TANGENT_ONE_FIELD", "TANGENT_TWO_FIELD", "LAGMULT",
        all_tris_for_smoothing);

    //addVolumeElement
    smooth_problem->addVolumeElement("VOLUME_SMOOTHING_ELEM",
                                     "MESH_NODE_POSITIONS", all_tets);

    CHKERR MetaSpringBC::addSpringElements(m_field, "MESH_NODE_POSITIONS",
                                           "MESH_NODE_POSITIONS");

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_levels.back());

    // define problems
    CHKERR m_field.add_problem("SURFACE_SMOOTHING_PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("SURFACE_SMOOTHING_PROB",
                                                    bit_levels.back());

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    SmartPetscObj<DM> dm;
    dm = createSmartDM(m_field.get_comm(), dm_name);

    // create dm instance
    CHKERR DMSetType(dm, dm_name);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "SURFACE_SMOOTHING_PROB",
                              bit_levels.back());
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "SURFACE_SMOOTHING_ELEM");
    CHKERR DMMoFEMAddElement(dm, "VOLUME_SMOOTHING_ELEM");
    CHKERR DMMoFEMAddElement(dm, "SPRING");
    CHKERR DMSetUp(dm);

    // Vector of DOFs and the RHS
    auto D = smartCreateDMVector(dm);
    auto F = smartVectorDuplicate(D);

    // Stiffness matrix
    auto Aij = smartCreateDMMatrix(dm);

    CHKERR VecZeroEntries(D);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);
    CHKERR MatZeroEntries(Aij);

    // Dirichlet BC
    // boost::shared_ptr<FEMethod> dirichlet_bc_ptr =
    //     boost::shared_ptr<FEMethod>(new DirichletSpatialPositionsBc(
    //         m_field, "MESH_NODE_POSITIONS", Aij, D, F));

    Range fixed_boundary;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      if (it->getName().compare(0, 15, "CONSTRAIN_SHAPE") == 0) {
        CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBVERTEX,
                                                       fixed_boundary, true);
      }
    }

    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr(
        new FaceElementForcesAndSourcesCore(m_field));

    CHKERR MetaSpringBC::setSpringOperators(
        m_field, fe_spring_lhs_ptr, fe_spring_rhs_ptr, "MESH_NODE_POSITIONS",
        "MESH_NODE_POSITIONS");

    boost::shared_ptr<DirichletFixFieldAtEntitiesBc> dirichlet_bc_ptr =
        boost::shared_ptr<DirichletFixFieldAtEntitiesBc>(
            new DirichletFixFieldAtEntitiesBc(
                m_field, "MESH_NODE_POSITIONS", fixed_vertex));
dirichlet_bc_ptr->fieldNames.push_back("TANGENT_ONE_FIELD");
dirichlet_bc_ptr->fieldNames.push_back("TANGENT_TWO_FIELD");

    dirichlet_bc_ptr->snes_ctx = SnesMethod::CTX_SNESNONE;
    dirichlet_bc_ptr->snes_x = D;

    CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
    
    // PetscViewer viewer;
    // CHKERR
    // PetscViewerASCIIOpen(PETSC_COMM_WORLD,"forces_and_sources_thermal_stress_elem.txt",&viewer);
    // CHKERR VecChop(F,1e-4);
    // CHKERR VecView(D,viewer);
    // CHKERR VecView(D,PETSC_VIEWER_STDOUT_WORLD);

    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL,
                                  dirichlet_bc_ptr.get(), NULL);

    CHKERR DMMoFEMSNESSetFunction(
        dm, "SURFACE_SMOOTHING_ELEM",
        get_smooth_face_rhs(smooth_problem, make_arbitrary_smooth_element_face),
        PETSC_NULL, PETSC_NULL);

    CHKERR DMMoFEMSNESSetFunction(dm, "SPRING", fe_spring_rhs_ptr, PETSC_NULL,
                                  PETSC_NULL);

    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL, NULL,
                                  dirichlet_bc_ptr.get());

    boost::shared_ptr<FEMethod> fe_null;
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, dirichlet_bc_ptr,
                                  fe_null);

    CHKERR DMMoFEMSNESSetJacobian(
        dm, "SURFACE_SMOOTHING_ELEM",
        get_smooth_face_lhs(smooth_problem, make_arbitrary_smooth_element_face),
        PETSC_NULL, PETSC_NULL);

    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, fe_null,
                                  dirichlet_bc_ptr);

    if (test_num) {
      char testing_options[] = "-ksp_type fgmres "
                               "-pc_type lu "
                               "-pc_factor_mat_solver_type mumps "
                               "-snes_type newtonls "
                               "-snes_linesearch_type basic "
                               "-snes_max_it 20 "
                               "-snes_atol 1e-8 "
                               "-snes_rtol 1e-8 ";
      CHKERR PetscOptionsInsertString(NULL, testing_options);
    }

    auto snes = MoFEM::createSNES(m_field.get_comm());
    CHKERR SNESSetDM(snes, dm);
    SNESConvergedReason snes_reason;
    SnesCtx *snes_ctx;
    // create snes nonlinear solver
    {
      CHKERR SNESSetDM(snes, dm);
      CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
      CHKERR SNESSetFunction(snes, F, SnesRhs, snes_ctx);
      CHKERR SNESSetJacobian(snes, Aij, Aij, SnesMat, snes_ctx);
      CHKERR SNESSetFromOptions(snes);
    }
      
    CHKERR SNESSolve(snes, PETSC_NULL, D);

    CHKERR SNESGetConvergedReason(snes, &snes_reason);

    int its;
    CHKERR SNESGetIterationNumber(snes, &its);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "number of Newton iterations = %D\n\n",
                       its);

    // save on mesh
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    ///get_smooth_face_post_prov

     CHKERR DMoFEMLoopFiniteElements(dm, "SURFACE_SMOOTHING_ELEM",
                                      get_smooth_face_post_prov(smooth_problem, make_arbitrary_smooth_element_face));
   
    PetscPrintf(PETSC_COMM_WORLD, "Loop post proc\n");
    PetscPrintf(PETSC_COMM_WORLD, "Loop Volume\n");
    VecCreate(PETSC_COMM_WORLD,&smooth_problem->volumeVec);
    PetscInt n = 1;
    VecSetSizes(smooth_problem->volumeVec,PETSC_DECIDE,n);
    VecSetUp(smooth_problem->volumeVec);
    CHKERR DMoFEMLoopFiniteElements(dm, "VOLUME_SMOOTHING_ELEM",
                                      get_smooth_volume_element_operators(smooth_problem, make_arbitrary_smooth_element_volume));
    // PetscInt nloc;
    // VecGetLocalSize(smooth_problem->volumeVec,&nloc);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "Total Volume: ");

    // cerr << smooth_problem->volumeVec;
    CHKERR VecView(smooth_problem->volumeVec,PETSC_VIEWER_STDOUT_WORLD);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "\n");
    
    // moab_instance
    moab::Core mb_post;                   // create database
    moab::Interface &moab_proc = mb_post; // create interface to database

    mb_post.delete_mesh();
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}