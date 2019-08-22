// /**
//  * \file reaction_diffusion_equation.cpp
//  * \example reaction_diffusion_equation.cpp
//  *
//  **/
// /* This file is part of MoFEM.
//  * MoFEM is free software: you can redistribute it and/or modify it under
//  * the terms of the GNU Lesser General Public License as published by the
//  * Free Software Foundation, either version 3 of the License, or (at your
//  * option) any later version.
//  *
//  * MoFEM is distributed in the hope that it will be useful, but WITHOUT
//  * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
//  * License for more details.
//  *
//  * You should have received a copy of the GNU Lesser General Public
//  * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */
#include <stdlib.h>
#include <BasicFiniteElements.hpp>
using namespace MoFEM;
static char help[] = "...\n\n";
namespace ReactionDiffusionEquation {
using Ele = FaceElementForcesAndSourcesCore;
using OpEle = FaceElementForcesAndSourcesCore::UserDataOperator;



using EntData = DataForcesAndSourcesCore::EntData;

const double D = 1e-2; ///< diffusivity
const double Dn = 0.;   // 1e-1; ///< non-linear diffusivity
// const double r = 1;    ///< rate factor
// const double k = 1;    ///< caring capacity
const double u0 = 0.5; ///< inital vale on blocksets
const int order = 2;   ///< approximation order
const int save_every_nth_step = 2;

double a1, a2, a3, r;

const double au1 = 1.0;
const double au2 = 2.0;
const double au3 = 7.0;

const double av1 = 7.0;
const double av2 = 1.0;
const double av3 = 2.0;

const double aw1 = 2.0;
const double aw2 = 7.0;
const double aw3 = 1.0;

const double ru = 1.0;
const double rv = 1.0;
const double rw = 1.0;

/**
 * @brief Common data
 *
 * Common data are used to keep and pass data between elements
 *
 */
struct CommonData {
  MatrixDouble grad_u;    ///< Gradients of field "u" at integration points
  VectorDouble val_u;     ///< Values of field "u" at integration points
  VectorDouble dot_val_u; ///< Rate of values of field "u" at integration points

  MatrixDouble grad_v;    ///< Gradients of field "v" at integration points
  VectorDouble val_v;     ///< Values of field "v" at integration points
  VectorDouble dot_val_v; ///< Rate of values of field "v" at integration points

  MatrixDouble grad_w;    ///< Gradients of field "w" at integration points
  VectorDouble val_w;     ///< Values of field "w" at integration points
  VectorDouble dot_val_w; ///< Rate of values of field "w" at integration points

  MatrixDouble Grad_U;
  MatrixDouble FMat; // first calculate grad_u and add 1. to diagonal
  MatrixDouble PullBackMat;

  MatrixDouble invJac; ///< Inverse of element jacobian

  CommonData() {}

  // SmartPetscObj<Mat> M;   ///< Mass matrix
  // SmartPetscObj<KSP> ksp; ///< Linear solver
};
/**
 * @brief Assemble mass matrix
 */
// template <int DIM>

auto wrap_matrix2_ftensor = [](MatrixDouble &m){
        return FTensor::Tensor2<FTensor::PackPtr<double *, 2>, 2, 2>(
                                       &m(0, 0), &m(0, 1),
                                       &m(1, 0), &m(1, 1));
}; 

auto wrap_matrix3_ftensor = [](MatrixDouble &m){
        return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
                                      &m(0, 0), &m(0, 1), &m(0, 2),
                                      &m(1, 0), &m(1, 1), &m(1, 2),
                                      &m(2, 0), &m(2, 1), &m(2, 2));
};




struct OpAssembleMass : OpEle {
  OpAssembleMass(std::string fieldu, std::string fieldv, SmartPetscObj<Mat> m)
      : OpEle(fieldu, fieldv, OpEle::OPROWCOL), M(m)

  {
    sYmm = true;
  }
  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    if (nb_row_dofs && nb_col_dofs) {
      const int nb_integration_pts = getGaussPts().size2();
      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      auto t_row_base = row_data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = t_w * vol;
        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            mat(rr, cc) += a * t_row_base * t_col_base;
            ++t_col_base;
          }
          ++t_row_base;
        }
        ++t_w;
      }
      CHKERR MatSetValues(M, row_data, col_data, &mat(0, 0), ADD_VALUES);
      if (row_side != col_side || row_type != col_type) {
        transMat.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transMat) = trans(mat);
        CHKERR MatSetValues(M, col_data, row_data, &transMat(0, 0), ADD_VALUES);
      }
    }
    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble mat, transMat;
  SmartPetscObj<Mat> M;
};
/**
 * @brief Assemble slow part
 *
 * Solve problem \f$ F(t,u,\dot{u}) = G(t,u) \f$ where here the right hand side
 * \f$ G(t,u) \f$ is implemented.
 *
 */
struct OpAssembleSlowRhs : OpEle {
  OpAssembleSlowRhs(std::string field, boost::shared_ptr<CommonData> &data)
      : OpEle(field, OpEle::OPROW), commonData(data), field(field) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    if (field == "u") {
      r = ru;
      a1 = au1;
      a2 = au2;
      a3 = au3;
    }
    if (field == "v") {
      r = rv;
      a1 = av2;
      a2 = av1;
      a3 = av3;
    }
    if (field == "w") {
      r = rw;
      a1 = aw3;
      a2 = aw1;
      a3 = aw2;
    }
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      vecF.resize(nb_dofs, false);
      vecF.clear();
      const int nb_integration_pts = getGaussPts().size2();

      boost::shared_ptr<VectorDouble> val_vec_ptr1;
      boost::shared_ptr<VectorDouble> val_vec_ptr2;
      boost::shared_ptr<VectorDouble> val_vec_ptr3;

      if (field == "u") {
        val_vec_ptr1 =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_u);
        val_vec_ptr2 =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_v);
        val_vec_ptr3 =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_w);
      }
      if (field == "v") {
        val_vec_ptr1 =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_v);
        val_vec_ptr2 =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_u);
        val_vec_ptr3 =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_w);
      }
      if (field == "w") {
        val_vec_ptr1 =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_w);
        val_vec_ptr2 =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_u);
        val_vec_ptr3 =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_v);
      }

      auto t_val_u = getFTensor0FromVec(*val_vec_ptr1);
      auto t_val_v = getFTensor0FromVec(*val_vec_ptr2);
      auto t_val_w = getFTensor0FromVec(*val_vec_ptr3);

      auto t_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double f = a * r * t_val_u *
                         (1.0 - a1 * t_val_u - a2 * t_val_v - a3 * t_val_w);
        for (int rr = 0; rr != nb_dofs; ++rr) {
          const double b = f * t_base;
          vecF[rr] += b;
          ++t_base;
        }
        ++t_val_u;
        ++t_val_v;
        ++t_val_w;
        ++t_w;
      }
      CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
                          PETSC_TRUE);
      CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
                          ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonData;
  VectorDouble vecF;
  std::string field;
};
/**
 * @brief Assemble stiff part
 *
 * Solve problem \f$ F(t,u,\dot{u}) = G(t,u) \f$ where here the right hand side
 * \f$ F(t,u,\dot{u}) \f$ is implemented.
 *
 */
template <int DIM> struct OpAssembleStiffRhs : OpEle {
  OpAssembleStiffRhs(std::string field, boost::shared_ptr<CommonData> &data)
      : OpEle(field, OpEle::OPROW), commonData(data), field(field) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      vecF.resize(nb_dofs, false);
      vecF.clear();
      const int nb_integration_pts = getGaussPts().size2();
      boost::shared_ptr<VectorDouble> dot_val_vec_ptr;
      boost::shared_ptr<VectorDouble> val_vec_ptr;
      boost::shared_ptr<MatrixDouble> grad_matrix_ptr;

      if (field == "u") {
        dot_val_vec_ptr =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->dot_val_u);

        val_vec_ptr =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_u);

        grad_matrix_ptr =
            boost::shared_ptr<MatrixDouble>(commonData, &commonData->grad_u);
      } else if (field == "v") {
        dot_val_vec_ptr =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->dot_val_v);
        val_vec_ptr =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_v);
        grad_matrix_ptr =
            boost::shared_ptr<MatrixDouble>(commonData, &commonData->grad_v);
      } else if (field == "w") {
        dot_val_vec_ptr =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->dot_val_w);
        val_vec_ptr =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_w);
        grad_matrix_ptr =
            boost::shared_ptr<MatrixDouble>(commonData, &commonData->grad_w);
      }


      auto t_dot_val = getFTensor0FromVec(*dot_val_vec_ptr);
      auto t_val = getFTensor0FromVec(*val_vec_ptr);
      auto t_grad = getFTensor1FromMat<DIM>(*grad_matrix_ptr);

      auto t_base = data.getFTensor0N();
      auto t_diff_base = data.getFTensor1DiffN<DIM>();
      auto t_w = getFTensor0IntegrationWeight();
    //   double time = getTStime();
    //   cerr << "time : " << time << "\n";
      FTensor::Index<'i', DIM> i;
      FTensor::Index<'j', DIM> j;
      FTensor::Index<'k', DIM> k;

      auto pull_back = getFTensor2FromMat<DIM, DIM>(commonData->PullBackMat);
      MatrixDouble d_mat;
      d_mat.resize(2, 2);
      //FTensor::Tensor2<FTensor::PackPtr<double *, DIM>, DIM, DIM> t_D();
    // FTensor::Tensor2<double, DIM, DIM> t_D;
      auto t_D = wrap_matrix2_ftensor(d_mat);
      
      const double vol = getMeasure();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        d_mat.clear();
        d_mat(0, 0) = d_mat(1, 1) = D + Dn * t_val;
        for (int rr = 0; rr != nb_dofs; ++rr) {
           vecF[rr] += a * (t_base * t_dot_val +
                          pull_back(i,j) * t_D(j,k) * t_diff_base(k) * t_grad(i));
            // vecF[rr] += a * (t_base * t_dot_val + (D + Dn * t_val) *
            // t_diff_base(i) * t_grad(i));
          ++t_diff_base;
          ++t_base;
        }
        ++t_dot_val;
        ++t_grad;
        ++t_val;
        ++t_w;
        ++pull_back;
      }
      CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
                          PETSC_TRUE);
      CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
                          ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonData;
  VectorDouble vecF;
  std::string field;
};
/**
 * @brief Assemble stiff part tangent
 *
 * Solve problem \f$ F(t,u,\dot{u}) = G(t,u) \f$ where here the right hand side
 * \f$ \frac{\textrm{d} F}{\textrm{d} u^n} = a F_{\dot{u}}(t,u,\textrm{u}) +
 * F_{u}(t,u,\textrm{u}) \f$ is implemented.
 *
 */
template <int DIM> struct OpAssembleStiffLhs : OpEle {

  OpAssembleStiffLhs(std::string                   fieldu, 
                     std::string                   fieldv,
                     boost::shared_ptr<CommonData> &data)
  : OpEle(fieldu, fieldv, OpEle::OPROWCOL), 
    commonData(data),
    field(fieldu) 
    {
      sYmm = true;
    }
  MoFEMErrorCode doWork(int           row_side, 
                        int           col_side, 
                        EntityType    row_type,
                        EntityType    col_type, 
                        EntData       &row_data,
                        EntData       &col_data) {
    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    // cerr << "In doWork() : (row, col) = (" << nb_row_dofs << ", " << nb_col_dofs << ")" << endl; 
    if (nb_row_dofs && nb_col_dofs) {
      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_row_base = row_data.getFTensor0N();

      boost::shared_ptr<VectorDouble> val_vec_ptr;
      boost::shared_ptr<MatrixDouble> grad_matrix_ptr;

      if (field == "u") {
        val_vec_ptr =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_u);
        grad_matrix_ptr =
            boost::shared_ptr<MatrixDouble>(commonData, &commonData->grad_u);
      }
      if (field == "v") {
        val_vec_ptr =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_v);
        grad_matrix_ptr =
            boost::shared_ptr<MatrixDouble>(commonData, &commonData->grad_v);
      }
      if (field == "w") {
        val_vec_ptr =
            boost::shared_ptr<VectorDouble>(commonData, &commonData->val_w);
        grad_matrix_ptr =
            boost::shared_ptr<MatrixDouble>(commonData, &commonData->grad_w);
      }

      auto t_val = getFTensor0FromVec(*val_vec_ptr);
      auto t_grad = getFTensor1FromMat<DIM>(*grad_matrix_ptr);

      auto t_row_diff_base = row_data.getFTensor1DiffN<DIM>();
      auto t_w = getFTensor0IntegrationWeight();
      const double ts_a = getFEMethod()->ts_a;
      const double vol = getMeasure();

      FTensor::Index<'i', DIM> i;
      FTensor::Index<'j', DIM> j;
      FTensor::Index<'k', DIM> k;


      auto pull_back = getFTensor2FromMat<DIM, DIM>(commonData->PullBackMat);
      MatrixDouble d_mat;
      MatrixDouble d_diff_mat;
      d_mat.resize(2, 2);
      d_diff_mat.resize(2, 2);
      //FTensor::Tensor2<FTensor::PackPtr<double *, DIM>, DIM, DIM> t_D();
      //FTensor::Tensor2<double, DIM, DIM> t_D;

      auto t_D = wrap_matrix2_ftensor(d_mat);
      auto t_diff_D = wrap_matrix2_ftensor(d_diff_mat);

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        d_mat.clear(); d_diff_mat.clear();
        d_mat(0, 0) = d_mat(1, 1) = D + Dn * t_val;
        d_diff_mat(0, 0) = d_diff_mat(1, 1) = Dn;
        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_base = col_data.getFTensor0N(gg, 0);
          auto t_col_diff_base = col_data.getFTensor1DiffN<DIM>(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            
            
            mat(rr, cc) +=
                a *
                (t_row_base * t_col_base * ts_a +
                pull_back(i,j) * t_D(j,k)  * t_row_diff_base(k) * t_col_diff_base(i) +
                 t_col_base * pull_back(i,j) * t_diff_D(j, k)  * t_row_diff_base(k) * t_grad(i));

            // mat(rr, cc) +=
            //     a * (t_row_base * t_col_base * ts_a + (D + Dn * t_val) * t_row_diff_base(i) * t_col_diff_base(i) 
            //           + Dn * t_col_base * t_grad(i) * t_row_diff_base(i));


            ++t_col_base;
            ++t_col_diff_base;
          }
          ++t_row_base;
          ++t_row_diff_base;
        }
        ++t_w;
        ++t_val;
        ++t_grad;
      }
      CHKERR MatSetValues(getFEMethod()->ts_B, row_data, col_data, &mat(0, 0),
                          ADD_VALUES);
      if (row_side != col_side || row_type != col_type) {
        transMat.resize(nb_col_dofs, nb_row_dofs, false);
        noalias(transMat) = trans(mat);
        CHKERR MatSetValues(getFEMethod()->ts_B, col_data, row_data,
                            &transMat(0, 0), ADD_VALUES);
      }
    }
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commonData;
  MatrixDouble mat, transMat;
  std::string field;
};
/**
 * @brief Monitor solution
 *
 * This functions is called by TS solver at the end of each step. It is used
 * to output results to the hard drive.
 */
struct Monitor : public FEMethod {
  Monitor(SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcFaceOnRefinedMesh> &post_proc)
      : dM(dm), postProc(post_proc){};
  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
      CHKERR postProc->writeFile(
          "out_level_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
    }
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProc;
};

template <int DIM> struct OpCalcPullBackTensor : public OpEle {
  OpCalcPullBackTensor(std::string field, boost::shared_ptr<CommonData> &data)
      : OpEle("U", OpEle::OPROW), commData(data) {
          doEdges = false;
      doQuads = false;
      doTris = false;
      doTets = false;
      doPrisms = false;
      }

  MoFEMErrorCode doWork(int row_side, EntityType row_type, EntData &row_data) {
    MoFEMFunctionBegin;
    FTensor::Index<'i', 2> i;
    FTensor::Index<'j', 2> j;
    FTensor::Index<'k', 2> k;

    const int nb_integration_pts = getGaussPts().size2();

    commData->PullBackMat.resize(4, nb_integration_pts, false);
    auto t_p_b = getFTensor2FromMat<2, 2>(commData->PullBackMat);
    auto t_F = getFTensor2FromMat<2, 2>(commData->FMat);
    double time = getTStime();

    // cerr << "time : " << time << "\n";

    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      double det = 0.;
      
      t_F(0, 0) += 1.;
      t_F(1, 1) += 1.;
      CHKERR determinantTensor2by2(t_F, det);
      CHKERR invertTensor2by2(t_F, det, t_p_b);
      t_p_b(i, k) = t_p_b(i, j) * t_p_b(k, j);
      //cerr<< t_p_b <<"\n";
      //cerr<< det <<"\n";
      ++t_F;
      ++t_p_b;
    }

    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<CommonData> commData;
};
}; // namespace ReactionDiffusionEquation

using namespace ReactionDiffusionEquation;
int main(int argc, char *argv[]) {
  // initialize petsc
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);
  try {

    SmartPetscObj<Mat> local_M;
    SmartPetscObj<KSP> local_Ksp;

    // Create moab and mofem instances
    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;
    // Register DM Manager
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);
    // Simple interface
    Simple *simple_interface;
    CHKERR m_field.getInterface(simple_interface);
    CHKERR simple_interface->getOptions();
    CHKERR simple_interface->loadFile();
    // add fields
    CHKERR simple_interface->addDomainField("u", H1, AINSWORTH_LEGENDRE_BASE,
                                            1);
    CHKERR simple_interface->addDomainField("v", H1, AINSWORTH_LEGENDRE_BASE,
                                            1);
    CHKERR simple_interface->addDomainField("w", H1, AINSWORTH_LEGENDRE_BASE,
                                            1);

    CHKERR simple_interface->addDataField("U", H1, AINSWORTH_LEGENDRE_BASE, 2);

    // set fields order
    CHKERR simple_interface->setFieldOrder("u", order);
    CHKERR simple_interface->setFieldOrder("v", order);
    CHKERR simple_interface->setFieldOrder("w", order);

    CHKERR simple_interface->setFieldOrder("U", 1);

    // setup problem
    CHKERR simple_interface->setUp();

    FieldBlas *field_blas;
    int num = 1;
    double magnitude = 2;
    double length = 2.;
    double inv_length = 1. / length;
    CHKERR m_field.getInterface(field_blas);
    // auto set_displacement_field = [&](VectorAdaptor &field_data, double *xcoord,
    //                                   double *ycoord, double *zcoord) {
    //   MoFEMFunctionBegin;

    //   field_data[0] = 0.;//0.5 * magnitude * ((*xcoord) * inv_length + 1.);
    //   //field_data[0] = std::sin((*xcoord));
    //   field_data[1] = 0.;

    //   MoFEMFunctionReturn(0);
    // };
    // CHKERR field_blas->setVertexDofs(set_displacement_field, "U");

    // Create common data structure
    boost::shared_ptr<CommonData> data(new CommonData());

    /// Alias pointers to data in common data structure
    auto val_ptr_u = boost::shared_ptr<VectorDouble>(data, &data->val_u);
    auto dot_val_ptr_u =
        boost::shared_ptr<VectorDouble>(data, &data->dot_val_u);
    auto grad_ptr_u = boost::shared_ptr<MatrixDouble>(data, &data->grad_u);

    auto val_ptr_v = boost::shared_ptr<VectorDouble>(data, &data->val_v);
    auto dot_val_ptr_v =
        boost::shared_ptr<VectorDouble>(data, &data->dot_val_v);
    auto grad_ptr_v = boost::shared_ptr<MatrixDouble>(data, &data->grad_v);

    auto val_ptr_w = boost::shared_ptr<VectorDouble>(data, &data->val_w);
    auto dot_val_ptr_w =
        boost::shared_ptr<VectorDouble>(data, &data->dot_val_w);
    auto grad_ptr_w = boost::shared_ptr<MatrixDouble>(data, &data->grad_w);

    auto F_mat_ptr = boost::shared_ptr<MatrixDouble>(data, &data->FMat);

    // Create finite element instances to integrate the right-hand side of slow
    // and stiff vector, and the tangent left-hand side for stiff part.
    boost::shared_ptr<Ele> vol_ele_slow_rhs(new Ele(m_field));
    boost::shared_ptr<Ele> vol_ele_stiff_rhs(new Ele(m_field));
    boost::shared_ptr<Ele> vol_ele_stiff_lhs(new Ele(m_field));
    // Push operators to integrate the slow right-hand side vector

    vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("u", val_ptr_u));
    vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("v", val_ptr_v));
    vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("w", val_ptr_w));

    vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpAssembleSlowRhs("u", data));



    vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpAssembleSlowRhs("v", data));



    vol_ele_slow_rhs->getOpPtrVector().push_back(
        new OpAssembleSlowRhs("w", data));

    // PETSc IMEX and Explicit solver demans that g = M^-1 G is provided. So
    // when the slow right-hand side vector (G) is assembled is solved for g
    // vector.
    auto solve_for_g = [&]() {
      MoFEMFunctionBegin;
      if (vol_ele_slow_rhs->vecAssembleSwitch) {
        CHKERR VecGhostUpdateBegin(vol_ele_slow_rhs->ts_F, ADD_VALUES,
                                   SCATTER_REVERSE);
        CHKERR VecGhostUpdateEnd(vol_ele_slow_rhs->ts_F, ADD_VALUES,
                                 SCATTER_REVERSE);
        CHKERR VecAssemblyBegin(vol_ele_slow_rhs->ts_F);
        CHKERR VecAssemblyEnd(vol_ele_slow_rhs->ts_F);
        *vol_ele_slow_rhs->vecAssembleSwitch = false;
      }
      CHKERR KSPSolve(local_Ksp, vol_ele_slow_rhs->ts_F,
                      vol_ele_slow_rhs->ts_F);
      MoFEMFunctionReturn(0);
    };
    // Add hook to the element to calculate g.
    vol_ele_slow_rhs->postProcessHook = solve_for_g;

    // Add operators to calculate the stiff right-hand side
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->invJac));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(data->invJac));

    // F tensor calculation
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<2, 2>("U", F_mat_ptr));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalcPullBackTensor<2>("U", data));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("u", val_ptr_u));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarValuesDot("u", dot_val_ptr_u));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("u", grad_ptr_u));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("v", val_ptr_v));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarValuesDot("v", dot_val_ptr_v));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("v", grad_ptr_v));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("w", val_ptr_w));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarValuesDot("w", dot_val_ptr_w));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("w", grad_ptr_w));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpAssembleStiffRhs<2>("u", data));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpAssembleStiffRhs<2>("v", data));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpAssembleStiffRhs<2>("w", data));

    // Add operators to calculate the stiff left-hand side

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->invJac));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpSetInvJacH1ForFace(data->invJac));

    // F tensor calculation
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<2, 2>("U", F_mat_ptr));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalcPullBackTensor<2>("U", data));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("u", grad_ptr_u));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("u", val_ptr_u));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("v", grad_ptr_v));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("v", val_ptr_v));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldGradient<2>("w", grad_ptr_v));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("w", val_ptr_v));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpAssembleStiffLhs<2>("u", "u", data));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpAssembleStiffLhs<2>("v", "v", data));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpAssembleStiffLhs<2>("w", "w", data));

    // Set integration rules
    auto vol_rule = [](int, int, int p) -> int { return 2 * p; };
    vol_ele_slow_rhs->getRuleHook = vol_rule;
    vol_ele_stiff_rhs->getRuleHook = vol_rule;
    vol_ele_stiff_lhs->getRuleHook = vol_rule;
    // Crate element for post-processing
    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc =
        boost::shared_ptr<PostProcFaceOnRefinedMesh>(
            new PostProcFaceOnRefinedMesh(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> null;
    // Genarte post-processing mesh
    post_proc->generateReferenceElementMesh();
    // Postprocess only field values
    post_proc->addFieldValuesPostProc("u");
    post_proc->addFieldValuesPostProc("v");
    post_proc->addFieldValuesPostProc("w");
    post_proc->addFieldValuesPostProc("U");

    // Get PETSc discrete manager
    auto dm = simple_interface->getDM();
    // Get surface entities form blockset, set initial values in those
    // blocksets. To keep it simple is assumed that inital values are on

    // blockset 1
    const int block_id1 = 1;
    if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id1,
                                                              BLOCKSET)) {
      Range inner_surface;
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          block_id1, BLOCKSET, 2, inner_surface, true);
      if (!inner_surface.empty()) {
        Range inner_surface_verts;
        CHKERR moab.get_connectivity(inner_surface, inner_surface_verts, false);
        CHKERR m_field.getInterface<FieldBlas>()->setField(
            ((double) rand() / (RAND_MAX)), MBVERTEX, inner_surface_verts, "u");
      }
    }
    // blockset 2
    const int block_id2 = 2;
    if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id2,
                                                              BLOCKSET)) {
      Range inner_surface;
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          block_id2, BLOCKSET, 2, inner_surface, true);
      if (!inner_surface.empty()) {
        Range inner_surface_verts;
        CHKERR moab.get_connectivity(inner_surface, inner_surface_verts, false);
        CHKERR m_field.getInterface<FieldBlas>()->setField(
            ((double) rand() / (RAND_MAX)), MBVERTEX, inner_surface_verts, "v");
      }
    }
    cerr << "rand :" << ((double) rand() / (RAND_MAX)) << "\n";
    // // blockset 3
    // const int block_id3 = 3;
    // if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id3,
    //                                                           BLOCKSET)) {
    //   Range inner_surface;
    //   CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
    //       block_id3, BLOCKSET, 2, inner_surface, true);
    //   if (!inner_surface.empty()) {
    //     Range inner_surface_verts;
    //     CHKERR moab.get_connectivity(inner_surface, inner_surface_verts, false);
    //     CHKERR m_field.getInterface<FieldBlas>()->setField(
    //         rand() % 1, MBVERTEX, inner_surface_verts, "w");
    //   }
    // }



    // Create mass matrix, calculate and assemble
    CHKERR DMCreateMatrix_MoFEM(dm, local_M);
    CHKERR MatZeroEntries(local_M);
    boost::shared_ptr<Ele> vol_mass_ele(new Ele(m_field));
    vol_mass_ele->getOpPtrVector().push_back(
        new OpAssembleMass("v", "v", local_M));
    vol_mass_ele->getOpPtrVector().push_back(
        new OpAssembleMass("u", "u", local_M));
    vol_mass_ele->getOpPtrVector().push_back(
        new OpAssembleMass("w", "w", local_M));
    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    vol_mass_ele);
    CHKERR MatAssemblyBegin(local_M, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(local_M, MAT_FINAL_ASSEMBLY);
    // Create and septup KSP (linear solver), we need this to calculate g(t,u) =
    // M^-1G(t,u)
    local_Ksp = createKSP(m_field.get_comm());
    CHKERR KSPSetOperators(local_Ksp, local_M, local_M);
    CHKERR KSPSetFromOptions(local_Ksp);
    CHKERR KSPSetUp(local_Ksp);
    // Create and setup TS solver
    auto ts = createTS(m_field.get_comm());
    // Use IMEX solver, i.e. implicit/explicit solver
    CHKERR TSSetType(ts, TSARKIMEX);
    CHKERR TSARKIMEXSetType(ts, TSARKIMEXA2);
    // Add element to calculate lhs of stiff part
    CHKERR DMMoFEMTSSetIJacobian(dm, simple_interface->getDomainFEName(),
                                 vol_ele_stiff_lhs, null, null);
    // Add element to calculate rhs of stiff part
    CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getDomainFEName(),
                                 vol_ele_stiff_rhs, null, null);
    // Add element to calculate rhs of slow (nonlinear) part
    CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getDomainFEName(),
                                   vol_ele_slow_rhs, null, null);
    // Add monitor to time solver
    boost::shared_ptr<Monitor> monitor_ptr(new Monitor(dm, post_proc));
    CHKERR DMMoFEMTSSetMonitor(dm, ts, simple_interface->getDomainFEName(),
                               monitor_ptr, null, null);
    // Create solution vector
    SmartPetscObj<Vec> X;
    CHKERR DMCreateGlobalVector_MoFEM(dm, X);
    CHKERR DMoFEMMeshToLocalVector(dm, X, INSERT_VALUES, SCATTER_FORWARD);
    // Solve problem
    double ftime = 1;
    CHKERR TSSetDM(ts, dm);
    CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
    CHKERR TSSetSolution(ts, X);
    CHKERR TSSetFromOptions(ts);
    CHKERR TSSolve(ts, X);
  }
  CATCH_ERRORS;
  // finish work cleaning memory, getting statistics, etc.
  MoFEM::Core::Finalize();
  return 0;
}

