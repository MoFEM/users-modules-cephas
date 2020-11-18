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

  struct SurfaceSmoothingElement : public FaceEle {
    MoFEM::Interface &mField;

    SurfaceSmoothingElement(MoFEM::Interface &m_field)
        : FaceEle(m_field), mField(m_field) {}

  int getRule(int order) { return 2 * order + 1; };

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

    boost::shared_ptr<MatrixDouble> tAngent1;
    boost::shared_ptr<MatrixDouble> tAngent2;
    boost::shared_ptr<MatrixDouble> nOrmal;

    boost::shared_ptr<MatrixDouble> nOrmalHdiv;
    boost::shared_ptr<MatrixDouble> tanHdiv1;
    boost::shared_ptr<MatrixDouble> tanHdiv2;
    
    boost::shared_ptr<VectorDouble> divNormalHdiv;

    boost::shared_ptr<MatrixDouble> matForDivTan1;
    boost::shared_ptr<MatrixDouble> matForDivTan2;

    CommonSurfaceSmoothingElement(MoFEM::Interface &m_field) : mField(m_field) {
      tAngent1 = boost::make_shared<MatrixDouble>();
      tAngent2 = boost::make_shared<MatrixDouble>();
      nOrmal = boost::make_shared<MatrixDouble>();
      nOrmalHdiv = boost::make_shared<MatrixDouble>();
      tanHdiv1 = boost::make_shared<MatrixDouble>();
      tanHdiv2 = boost::make_shared<MatrixDouble>();
      matForDivTan1 = boost::make_shared<MatrixDouble>();
      matForDivTan2 = boost::make_shared<MatrixDouble>();

      divNormalHdiv = boost::make_shared<VectorDouble>();
    }

  private:
    MoFEM::Interface &mField;
  };

  MoFEMErrorCode addSurfaceSmoothingElement(const string element_name,
                                            const string field_name_position,
                                            const string field_name_normal_hdiv,
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
                                                      field_name_normal_hdiv);

    CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                      field_name_normal_hdiv);

    CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                       field_name_normal_hdiv);

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

      auto t_1 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent1);
      auto t_2 =
          getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->tAngent2);
      auto t_n = getFTensor1FromMat<3>(*commonSurfaceSmoothingElement->nOrmal);

      for (unsigned int gg = 0; gg != ngp; ++gg) {
        t_n(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);
        const double n_mag = sqrt(t_n(i) * t_n(i));
        t_n(i) /= n_mag;
        ++t_n;
        ++t_1;
        ++t_2;
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
      string field_name_position, string field_name_normal_hdiv) {
    MoFEMFunctionBegin;
    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpGetTangentForSmoothSurf(field_name_position,
                                      common_data_smooth_element));
    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpGetNormalForSmoothSurf(field_name_position,
                                     common_data_smooth_element));
    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpGetTangentHdivAtGaussPoints(field_name_normal_hdiv,
                                         common_data_smooth_element));

    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpGetNormalHdivAtGaussPoints(field_name_normal_hdiv,
                                         common_data_smooth_element));


    // Rhs
    fe_rhs_smooth_element->getOpPtrVector().push_back(
        new OpCalSmoothingNormalHdivRhs(field_name_normal_hdiv,
                                        common_data_smooth_element));
    fe_rhs_smooth_element->getOpPtrVector().push_back(new OpCalSmoothingXRhs(
        field_name_position, common_data_smooth_element));

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setSmoothFaceOperatorsLhs(
      boost::shared_ptr<SurfaceSmoothingElement> fe_lhs_smooth_element,
      boost::shared_ptr<CommonSurfaceSmoothingElement>
          common_data_smooth_element,
      string field_name_position, string field_name_normal_hdiv) {
    MoFEMFunctionBegin;

    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpGetTangentForSmoothSurf(field_name_position, common_data_smooth_element));
    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpGetNormalForSmoothSurf(field_name_position, common_data_smooth_element));
    
    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpGetTangentHdivAtGaussPoints(field_name_normal_hdiv,
                                         common_data_smooth_element));

    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpGetNormalHdivAtGaussPoints(field_name_normal_hdiv,
                                         common_data_smooth_element));

    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpCalSmoothingNormalHdivHdivLhs(field_name_normal_hdiv, common_data_smooth_element));

    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpCalSmoothingX_dnLhs(field_name_position, field_name_normal_hdiv, common_data_smooth_element));

    fe_lhs_smooth_element->getOpPtrVector().push_back(
        new OpCalSmoothingX_dXLhs(field_name_position, common_data_smooth_element));

    MoFEMFunctionReturn(0);
  }


  MoFEM::Interface &mField;
  ArbitrarySmoothingProblem(MoFEM::Interface &m_field) : mField(m_field) {}
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
                                 "-my_order 2 \n";

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
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool is_newton_cotes = PETSC_FALSE;
    PetscInt test_num = 0;
    PetscBool print_contact_state = PETSC_FALSE;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order",
                           "approximation order of spatial positions", "", 1,
                           &order, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_partitioned, PETSC_NULL);

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

    CHKERR m_field.add_field("CONTACT_MESH_NODE_POSITIONS", H1,
                             AINSWORTH_LEGENDRE_BASE, 3, MB_TAG_SPARSE,
                             MF_ZERO);

    CHKERR m_field.add_field("NORMAL_HDIV", HCURL, AINSWORTH_LEGENDRE_BASE, 2,
                             MB_TAG_SPARSE, MF_ZERO);

    Range fixed_vertex;
    CHKERR m_field.get_moab().get_connectivity(all_tris_for_smoothing,
                                               fixed_vertex);

    // Declare problem add entities (by tets) to the field
    // CHKERR m_field.add_ents_to_field_by_type(0, MBTET,
    //                                          "CONTACT_MESH_NODE_POSITIONS");
    // CHKERR m_field.set_field_order(0, MBTET, "CONTACT_MESH_NODE_POSITIONS", 1);
    // CHKERR m_field.set_field_order(0, MBTRI, "CONTACT_MESH_NODE_POSITIONS", 1);
    // CHKERR m_field.set_field_order(0, MBEDGE, "CONTACT_MESH_NODE_POSITIONS", 1);
    // CHKERR m_field.set_field_order(0, MBVERTEX, "CONTACT_MESH_NODE_POSITIONS",
    //                                1);

    if (!slave_tris.empty()) {
      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "CONTACT_MESH_NODE_POSITIONS");
      CHKERR m_field.set_field_order(slave_tris, "CONTACT_MESH_NODE_POSITIONS", order);
      Range edges_smoothed;
      CHKERR moab.get_adjacencies(slave_tris, 1, false, edges_smoothed,
                              moab::Interface::UNION);
      CHKERR m_field.set_field_order(edges_smoothed, "CONTACT_MESH_NODE_POSITIONS", order);
      Range vertices_smoothed;
      CHKERR m_field.get_moab().get_connectivity(slave_tris,
                                               vertices_smoothed);
      CHKERR m_field.set_field_order(vertices_smoothed, "CONTACT_MESH_NODE_POSITIONS", order);

    }
    if (!master_tris.empty()) {
      CHKERR m_field.add_ents_to_field_by_type(master_tris, MBTRI,
                                               "CONTACT_MESH_NODE_POSITIONS");
      CHKERR m_field.set_field_order(master_tris, "CONTACT_MESH_NODE_POSITIONS", order);
      Range edges_smoothed;
      CHKERR moab.get_adjacencies(master_tris, 1, false, edges_smoothed,
                              moab::Interface::UNION);
      CHKERR m_field.set_field_order(edges_smoothed, "CONTACT_MESH_NODE_POSITIONS", order);
      Range vertices_smoothed;
      CHKERR m_field.get_moab().get_connectivity(master_tris,
                                               vertices_smoothed);
      CHKERR m_field.set_field_order(vertices_smoothed, "CONTACT_MESH_NODE_POSITIONS", order);
    }

    if (!slave_tris.empty()) {
      Range edges_hcurl_slave;
      CHKERR moab.get_adjacencies(slave_tris, 1, false, edges_hcurl_slave,
                              moab::Interface::UNION);
      CHKERR m_field.set_field_order(edges_hcurl_slave, "NORMAL_HDIV", order - 1);

      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "NORMAL_HDIV");
      CHKERR m_field.set_field_order(slave_tris, "NORMAL_HDIV", order - 1);
    }
    if (!master_tris.empty()) {
      Range edges_hcurl_master;
      CHKERR moab.get_adjacencies(master_tris, 1, false, edges_hcurl_master,
                              moab::Interface::UNION);
      CHKERR m_field.set_field_order(edges_hcurl_master, "NORMAL_HDIV", order - 1);

      CHKERR m_field.add_ents_to_field_by_type(master_tris, MBTRI,
                                               "NORMAL_HDIV");
      CHKERR m_field.set_field_order(master_tris, "NORMAL_HDIV", order - 1);
    }

    // build field
    CHKERR m_field.build_fields();

    // Projection on "X" field
    {
      Projection10NodeCoordsOnField ent_method(m_field,
                                               "CONTACT_MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("CONTACT_MESH_NODE_POSITIONS", ent_method);
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
          "CONTACT_MESH_NODE_POSITIONS", "NORMAL_HDIV");
      return fe_rhs_smooth_face;
    };

    auto get_smooth_volume_element_operators = [&](auto smooth_problem,
                                                   auto make_element) {
      auto fe_smooth_volumes = make_element();
      auto common_data_smooth_elements = make_smooth_element_common_data();
      smooth_problem->setGlobalVolumeEvaluator(fe_smooth_volumes,
                                               common_data_smooth_elements,
                                               "CONTACT_MESH_NODE_POSITIONS");
      return fe_smooth_volumes;
    };

    auto get_smooth_face_lhs = [&](auto smooth_problem, auto make_element) {
      auto fe_lhs_smooth_face = make_element();
      auto common_data_smooth_elements = make_smooth_element_common_data();
      smooth_problem->setSmoothFaceOperatorsLhs(fe_lhs_smooth_face,
                                                common_data_smooth_elements,
                                                "CONTACT_MESH_NODE_POSITIONS", "NORMAL_HDIV");
      return fe_lhs_smooth_face;
    };

    auto smooth_problem =
        boost::make_shared<ArbitrarySmoothingProblem>(m_field);

    // add fields to the global matrix by adding the element

    smooth_problem->addSurfaceSmoothingElement(
        "SURFACE_SMOOTHING_ELEM", "CONTACT_MESH_NODE_POSITIONS", "NORMAL_HDIV",
        all_tris_for_smoothing);

    // addVolumeElement
    // smooth_problem->addVolumeElement("VOLUME_SMOOTHING_ELEM",
    //                                  "CONTACT_MESH_NODE_POSITIONS", all_tets);

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
    // CHKERR DMMoFEMAddElement(dm, "VOLUME_SMOOTHING_ELEM");
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
    //         m_field, "CONTACT_MESH_NODE_POSITIONS", Aij, D, F));

    boost::shared_ptr<FEMethod> dirichlet_bc_ptr =
        boost::shared_ptr<DirichletFixFieldAtEntitiesBc>(
            new DirichletFixFieldAtEntitiesBc(
                m_field, "CONTACT_MESH_NODE_POSITIONS", fixed_vertex));

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
  // CHKERR DMoFEMLoopFiniteElements(dm, "SURFACE_SMOOTHING_ELEM",
  //                                     get_smooth_face_rhs(smooth_problem, make_arbitrary_smooth_element_face));
      
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

    PetscPrintf(PETSC_COMM_WORLD, "Loop post proc\n");
    PetscPrintf(PETSC_COMM_WORLD, "Loop Volume\n");
    
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