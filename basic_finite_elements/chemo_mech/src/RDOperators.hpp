#ifndef __RDOPERATORS_HPP__
#define __RDOPERATORS_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

namespace ReactionDiffusion {

using FaceEle = MoFEM::FaceElementForcesAndSourcesCoreSwitch<
    FaceElementForcesAndSourcesCore::NO_HO_GEOMETRY |
    FaceElementForcesAndSourcesCore::NO_CONTRAVARIANT_TRANSFORM_HDIV |
    FaceElementForcesAndSourcesCore::NO_COVARIANT_TRANSFORM_HCURL>;

using BoundaryEle = MoFEM::EdgeElementForcesAndSourcesCoreSwitch<
    EdgeElementForcesAndSourcesCore::NO_HO_GEOMETRY |
    EdgeElementForcesAndSourcesCore::NO_COVARIANT_TRANSFORM_HCURL>;

using OpFaceEle = FaceEle::UserDataOperator;
using OpBoundaryEle = BoundaryEle::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;

const double a11 = 1; 
const double a12 = 0; 
const double a13 = 0;
const double a21 = 1;
const double a22 = 0;
const double a23 = 0;
const double a31 = 1;
const double a32 = 0;
const double a33 = 0;
const double r = 1;

const double B = 0.0;
const double B0 = 1e-3;
const double B_epsilon = 0.0;


const int save_every_nth_step = 4;
const int order = 3; ///< approximation order
const double natural_bc_values = 0.0;
const double essential_bc_values = 0.0;
// const int dim = 3;
FTensor::Index<'i', 3> i;

struct PreviousData {
  MatrixDouble flux_values;
  VectorDouble flux_divs;

  VectorDouble mass_dots;
  VectorDouble mass_values;

  VectorDouble slow_values;

  MatrixDouble jac;
  MatrixDouble inv_jac;

  PreviousData(){
    jac.resize(2, 2, false);
    inv_jac.resize(2, 2, false);
  }
};






struct NaturalBoundaryValues {
  NaturalBoundaryValues(EntityHandle fe_entity, VectorDouble &gauss_point)
  : fE_entity(fe_entity)
  , gPoint(gauss_point)
  {}

  MoFEMErrorCode set_value(double &out_value) { 
    MoFEMFunctionBegin;
    out_value = 0;
    MoFEMFunctionReturn(0);
  }
  EntityHandle fE_entity;
  VectorDouble gPoint;
};

struct EssentialBoundaryValues {
  EssentialBoundaryValues(EntityHandle fe_entity, VectorDouble &gauss_point)
      : fE_entity(fe_entity), gPoint(gauss_point) {}

  MoFEMErrorCode set_value(double &out_value) {
    MoFEMFunctionBegin;
    out_value = 0;
    MoFEMFunctionReturn(0);
  }
  EntityHandle fE_entity;
  VectorDouble gPoint;
}; 

struct InitialValues {
  InitialValues(VectorDouble &gauss_point)
  : gPoint(gauss_point)
  {}
  MoFEMErrorCode set_value(double &out_value) {
    MoFEMFunctionBegin;
    out_value = 0.5;
    MoFEMFunctionReturn(0);
  }

  VectorDouble gPoint;
};

struct OpComputeSlowValue : public OpFaceEle {
  OpComputeSlowValue(std::string mass_field,
                     boost::shared_ptr<PreviousData> &data1,
                     boost::shared_ptr<PreviousData> &data2,
                     boost::shared_ptr<PreviousData> &data3)
      : OpFaceEle(mass_field, OpFaceEle::OPROW), commonData1(data1),
        commonData2(data2), commonData3(data3), massField(mass_field) {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    boost::shared_ptr<VectorDouble> slow_value_ptr1(commonData1,
                                                    &commonData1->slow_values);
    boost::shared_ptr<VectorDouble> slow_value_ptr2(commonData2,
                                                    &commonData2->slow_values);
    boost::shared_ptr<VectorDouble> slow_value_ptr3(commonData3,
                                                    &commonData3->slow_values);

    VectorDouble &vec1 = *slow_value_ptr1;
    VectorDouble &vec2 = *slow_value_ptr2;
    VectorDouble &vec3 = *slow_value_ptr3;
    const int nb_integration_pts = getGaussPts().size2();
    if (type == MBVERTEX) {
      vec1.resize(nb_integration_pts, false);
      vec2.resize(nb_integration_pts, false);
      vec3.resize(nb_integration_pts, false);
      vec1.clear();
      vec2.clear();
      vec3.clear();
    }
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      const int nb_integration_pts = getGaussPts().size2();

      auto t_slow_values1 = getFTensor0FromVec(vec1);
      auto t_slow_values2 = getFTensor0FromVec(vec2);
      auto t_slow_values3 = getFTensor0FromVec(vec3);

      auto t_mass_values1 = getFTensor0FromVec(commonData1->mass_values);
      auto t_mass_values2 = getFTensor0FromVec(commonData2->mass_values);
      auto t_mass_values3 = getFTensor0FromVec(commonData3->mass_values);
      for (int gg = 0; gg != nb_integration_pts; ++gg) {

        t_slow_values1 = r * t_mass_values1 *
                         (1.0 - a11 * t_mass_values1 - a12 * t_mass_values2 -
                          a13 * t_mass_values3);
        t_slow_values2 = r * t_mass_values2 *
                         (1.0 - a21 * t_mass_values1 - a22 * t_mass_values2 -
                          a23 * t_mass_values3);

        t_slow_values3 = r * t_mass_values3 *
                         (1.0 - a31 * t_mass_values1 - a32 * t_mass_values2 -
                          a33 * t_mass_values3);
        ++t_slow_values1;
        ++t_slow_values2;
        ++t_slow_values3;

        ++t_mass_values1;
        ++t_mass_values2;
        ++t_mass_values3;
      }
    }
    MoFEMFunctionReturn(0);
  }

private:
  std::string massField;
  boost::shared_ptr<PreviousData> commonData1;
  boost::shared_ptr<PreviousData> commonData2;
  boost::shared_ptr<PreviousData> commonData3;
};

struct OpEssentialBC : public OpBoundaryEle {
  OpEssentialBC(const std::string &flux_field, Range &essential_bd_ents)
      : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW),
        essential_bd_ents(essential_bd_ents) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    EntityType type1 = type;
    int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      EntityHandle fe_ent = getFEEntityHandle();
      bool is_essential =
          (essential_bd_ents.find(fe_ent) != essential_bd_ents.end());
      if (is_essential) {
        int nb_gauss_pts = getGaussPts().size2();
        int size2 = data.getN().size2();
        if (3 * nb_dofs != static_cast<int>(data.getN().size2()))
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                  "wrong number of dofs");
        nN.resize(nb_dofs, nb_dofs, false);
        nF.resize(nb_dofs, false);
        nN.clear();
        nF.clear();

        auto t_row_tau = data.getFTensor1N<3>();

        auto dir = getDirection();
        double len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);

        FTensor::Tensor1<double, 3> t_normal(-dir[1] / len, dir[0] / len,
                                             dir[2] / len);

        auto t_w = getFTensor0IntegrationWeight();
        const double vol = getMeasure();

        for (int gg = 0; gg < nb_gauss_pts; gg++) {
          const double a = t_w * vol;
          const double ess_bc_value_gg = essential_bc_values;
          for (int rr = 0; rr != nb_dofs; rr++) {
            auto t_col_tau = data.getFTensor1N<3>(gg, 0);
            nF[rr] += a * ess_bc_value_gg * t_row_tau(i) * t_normal(i);
            for (int cc = 0; cc != nb_dofs; cc++) {
              nN(rr, cc) += a * (t_row_tau(i) * t_normal(i)) *
                            (t_col_tau(i) * t_normal(i));
              ++t_col_tau;
            }
            ++t_row_tau;
          }
          ++t_w;
        }

        cholesky_decompose(nN);
        cholesky_solve(nN, nF, ublas::lower());

        for (auto &dof : data.getFieldDofs()) {
          dof->getFieldData() = nF[dof->getEntDofIdx()];

          // this is only to check
          // data.getFieldData()[dof->getEntDofIdx()] = nF[dof->getEntDofIdx()];
        }
      }
    }
    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble nN;
  VectorDouble nF;
  Range &essential_bd_ents;
};

// Assembly of system mass matrix
// //***********************************************

// Mass matrix corresponding to the flux equation.
// 01. Note that it is an identity matrix

struct OpInitialMass : public OpFaceEle {
  OpInitialMass(const std::string &mass_field, Range &inner_surface)
      : OpFaceEle(mass_field, OpFaceEle::OPROW), innerSurface(inner_surface) {
    cerr << "OpInitialMass()" << endl;
  }
  MatrixDouble nN;
  VectorDouble nF;
  Range &innerSurface;
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    int nb_dofs = data.getFieldData().size();
    if (nb_dofs) {
      EntityHandle fe_ent = getFEEntityHandle();
      bool is_inner_side = (innerSurface.find(fe_ent) != innerSurface.end());
      if (is_inner_side) {
        int nb_gauss_pts = getGaussPts().size2();
        if (nb_dofs != static_cast<int>(data.getN().size2()))
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                  "wrong number of dofs");
        nN.resize(nb_dofs, nb_dofs, false);
        nF.resize(nb_dofs, false);
        nN.clear();
        nF.clear();

        auto t_row_mass = data.getFTensor0N();
        auto t_w = getFTensor0IntegrationWeight();
        const double vol = getMeasure();

        for (int gg = 0; gg < nb_gauss_pts; gg++) {
          const double a = t_w * vol;
          const double init_val_at_gg = 0.5;
          for (int rr = 0; rr != nb_dofs; rr++) {
            auto t_col_mass = data.getFTensor0N(gg, 0);
            nF[rr] += a * init_val_at_gg * t_row_mass;
            for (int cc = 0; cc != nb_dofs; cc++) {
              nN(rr, cc) += a * t_row_mass * t_col_mass;
              ++t_col_mass;
            }
            ++t_row_mass;
          }
          ++t_w;
        }

        cholesky_decompose(nN);
        cholesky_solve(nN, nF, ublas::lower());

        for (auto &dof : data.getFieldDofs()) {
          dof->getFieldData() = nF[dof->getEntDofIdx()];

          // this is only to check
          // data.getFieldData()[dof->getEntDofIdx()] = nF[dof->getEntDofIdx()];
        }
      }
    }
    MoFEMFunctionReturn(0);
  }
};

// Assembly of RHS for explicit (slow)
// part//**************************************

// 2. RHS for explicit part of the mass balance equation
struct OpAssembleSlowRhsV : OpFaceEle // R_V
{
  OpAssembleSlowRhsV(std::string mass_field,
                     boost::shared_ptr<PreviousData> &common_data)
      : OpFaceEle(mass_field, OpFaceEle::OPROW), commonData(common_data) {
    cerr << "OpAssembleSlowRhsV()" << endl;
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    // cerr << "In OpAssembleSlowRhsV...." << endl;
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      // cerr << "In SlowRhsV..." << endl;
      if (nb_dofs != static_cast<int>(data.getN().size2()))
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "wrong number of dofs");
      vecF.resize(nb_dofs, false);
      mat.resize(nb_dofs, nb_dofs, false);
      vecF.clear();
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_mass_value = getFTensor0FromVec(commonData->mass_values);
      auto t_slow_value = getFTensor0FromVec(commonData->slow_values);
      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        // const double f = a * r * t_mass_value * (1 - t_mass_value);
        for (int rr = 0; rr != nb_dofs; ++rr) {
          auto t_col_v_base = data.getFTensor0N(gg, 0);
          vecF[rr] += a * t_slow_value * t_row_v_base;
          for (int cc = 0; cc != nb_dofs; ++cc) {
            mat(rr, cc) += a * t_row_v_base * t_col_v_base;
            ++t_col_v_base;
          }
          ++t_row_v_base;
        }
        ++t_mass_value;
        ++t_slow_value;
        ++t_w;
      }
      cholesky_decompose(mat);
      cholesky_solve(mat, vecF, ublas::lower());

      CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
                          PETSC_TRUE);
      CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
                          ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<PreviousData> commonData;
  VectorDouble vecF;
  MatrixDouble mat;
};

// 5. RHS contribution of the natural boundary condition
struct OpAssembleNaturalBCRhsTau : OpBoundaryEle // R_tau_2
{
  OpAssembleNaturalBCRhsTau(std::string flux_field,
                            Range &natural_bd_ents)
      : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW), natural_bd_ents(natural_bd_ents) {
    cerr << "OpAssembleNaturalBCRhsTau()" << endl;
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();

    if (nb_dofs) {
      EntityHandle row_side_ent = data.getFieldDofs()[0]->getEnt();

      bool is_natural =
          (natural_bd_ents.find(row_side_ent) != natural_bd_ents.end());
      if (is_natural) {
        // cerr << "In NaturalBCRhsTau..." << endl;
        vecF.resize(nb_dofs, false);
        vecF.clear();
        const int nb_integration_pts = getGaussPts().size2();
        auto t_tau_base = data.getFTensor1N<3>();

        auto dir = getDirection();
        FTensor::Tensor1<double, 3> t_normal(-dir[1], dir[0], dir[2]);

        auto t_w = getFTensor0IntegrationWeight();

        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          const double a = t_w;
          for (int rr = 0; rr != nb_dofs; ++rr) {
            vecF[rr] += (t_tau_base(i) * t_normal(i) * natural_bc_values) * a;
            ++t_tau_base;
          }
          ++t_w;
        }
        CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
                            PETSC_TRUE);
        CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
                            ADD_VALUES);
      }
    }
    MoFEMFunctionReturn(0);
  }

private:
  VectorDouble vecF;
  Range natural_bd_ents;
};

// Assembly of RHS for the implicit (stiff) part excluding the essential
// boundary //**********************************
// 3. Assembly of F_tau excluding the essential boundary condition
template <int dim>
struct OpAssembleStiffRhsTau : OpFaceEle //  F_tau_1
{
  OpAssembleStiffRhsTau(std::string flux_field,
                        boost::shared_ptr<PreviousData> &data)
      : OpFaceEle(flux_field, OpFaceEle::OPROW), commonData(data) {
    cerr << "OpAssembleStiffRhsTau()" << endl;
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {

      vecF.resize(nb_dofs, false);
      vecF.clear();

      const int nb_integration_pts = getGaussPts().size2();
      auto t_flux_value = getFTensor1FromMat<3>(commonData->flux_values);
      auto t_mass_value = getFTensor0FromVec(commonData->mass_values);
      auto t_tau_base = data.getFTensor1N<3>();

      auto t_tau_grad = data.getFTensor2DiffN<3, 2>();

      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      for (int gg = 0; gg < nb_integration_pts; ++gg) {

        const double K = B_epsilon + (B0 + B * t_mass_value);
        const double K_inv = 1. / K;
        const double a = vol * t_w;
        for (int rr = 0; rr < nb_dofs; ++rr) {
          double div_base = t_tau_grad(0, 0) + t_tau_grad(1, 1);
          vecF[rr] += (K_inv * t_tau_base(i) * t_flux_value(i) -
                       div_base * t_mass_value) *
                      a;
          ++t_tau_base;
          ++t_tau_grad;
        }
        ++t_flux_value;
        ++t_mass_value;
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
  boost::shared_ptr<PreviousData> commonData;
  VectorDouble vecF;
};
// 4. Assembly of F_v
template <int dim>
struct OpAssembleStiffRhsV : OpFaceEle // F_V
{
  OpAssembleStiffRhsV(std::string flux_field,
                      boost::shared_ptr<PreviousData> &data)
      : OpFaceEle(flux_field, OpFaceEle::OPROW), commonData(data) {

    cerr << "OpAssembleStiffRhsV()" << endl;
  }

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    // cerr << "In StiffRhsV ..." << endl;
    if (nb_dofs) {
      vecF.resize(nb_dofs, false);
      vecF.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_mass_dot = getFTensor0FromVec(commonData->mass_dots);
      auto t_flux_div = getFTensor0FromVec(commonData->flux_divs);
      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      for (int gg = 0; gg < nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        for (int rr = 0; rr < nb_dofs; ++rr) {
          vecF[rr] += (-t_row_v_base * (t_mass_dot + t_flux_div)) * a;
          ++t_row_v_base;
        }
        ++t_mass_dot;
        ++t_flux_div;
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
  boost::shared_ptr<PreviousData> commonData;
  VectorDouble vecF;
};

// Tangent operator
// //**********************************************
// 7. Tangent assembly for F_tautau excluding the essential boundary condition
template <int dim>
struct OpAssembleLhsTauTau : OpFaceEle // A_TauTau_1
{
  OpAssembleLhsTauTau(std::string flux_field,
                      boost::shared_ptr<PreviousData> &commonData)
      : OpFaceEle(flux_field, flux_field, OpFaceEle::OPROWCOL),
        // essential_bd_ents(essential_bd_ents),
        commonData(commonData) {
    sYmm = true;
    cerr << "OpAssembleLhsTauTau()" << endl;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    if (nb_row_dofs && nb_col_dofs) {
      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_mass_value = getFTensor0FromVec(commonData->mass_values);

      auto t_row_tau_base = row_data.getFTensor1N<3>();

      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double K = B_epsilon + (B0 + B * t_mass_value);
        const double K_inv = 1. / K;
        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_tau_base = col_data.getFTensor1N<3>(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            mat(rr, cc) += (K_inv * t_row_tau_base(i) * t_col_tau_base(i)) * a;
            ++t_col_tau_base;
          }
          ++t_row_tau_base;
        }
        ++t_mass_value;
        ++t_w;
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
  boost::shared_ptr<PreviousData> commonData;
  MatrixDouble mat, transMat;
  Range essential_bd_ents;
};

// 9. Assembly of tangent for F_tau_v excluding the essential bc
template <int dim>
struct OpAssembleLhsTauV : OpFaceEle // E_TauV
{
  OpAssembleLhsTauV(std::string flux_field, std::string mass_field,
                    boost::shared_ptr<PreviousData> &data)
      : OpFaceEle(flux_field, mass_field, OpFaceEle::OPROWCOL),
        commonData(data) {
    sYmm = false;
    cerr << "OpAssembleLhsTauV()" << endl;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    if (nb_row_dofs && nb_col_dofs) {
      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_w = getFTensor0IntegrationWeight();
      auto t_row_tau_base = row_data.getFTensor1N<3>();

      auto t_row_tau_grad = row_data.getFTensor2DiffN<3, 2>();
      auto t_mass_value = getFTensor0FromVec(commonData->mass_values);
      auto t_flux_value = getFTensor1FromMat<3>(commonData->flux_values);
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double K = B_epsilon + (B0 + B * t_mass_value);
        const double K_inv = 1. / K;
        const double K_diff = B;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_v_base = col_data.getFTensor0N(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            double div_row_base = t_row_tau_grad(0, 0) + t_row_tau_grad(1, 1);
            mat(rr, cc) += (-(t_row_tau_base(i) * t_flux_value(i) * K_inv *
                              K_inv * K_diff * t_col_v_base) -
                            (div_row_base * t_col_v_base)) *
                           a;
            ++t_col_v_base;
          }
          ++t_row_tau_base;
          ++t_row_tau_grad;
        }
        ++t_w;
        ++t_mass_value;
        ++t_flux_value;
      }
      CHKERR MatSetValues(getFEMethod()->ts_B, row_data, col_data, &mat(0, 0),
                          ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<PreviousData> commonData;
  MatrixDouble mat;
};

// 10. Assembly of tangent for F_v_tau
struct OpAssembleLhsVTau : OpFaceEle // C_VTau
{
  OpAssembleLhsVTau(std::string mass_field, std::string flux_field)
      : OpFaceEle(mass_field, flux_field, OpFaceEle::OPROWCOL) {
    sYmm = false;
    cerr << "OpAssembleLhsVTau()" << endl;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    if (nb_row_dofs && nb_col_dofs) {
      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_w = getFTensor0IntegrationWeight();
      auto t_row_v_base = row_data.getFTensor0N();
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_tau_grad = col_data.getFTensor2DiffN<3, 2>(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            double div_col_base = t_col_tau_grad(0, 0) + t_col_tau_grad(1, 1);
            mat(rr, cc) += -(t_row_v_base * div_col_base) * a;
            ++t_col_tau_grad;
          }
          ++t_row_v_base;
        }
        ++t_w;
      }
      CHKERR MatSetValues(getFEMethod()->ts_B, row_data, col_data, &mat(0, 0),
                          ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

private:
  MatrixDouble mat;
};

// 11. Assembly of tangent for F_v_v
struct OpAssembleLhsVV : OpFaceEle // D
{
  OpAssembleLhsVV(std::string mass_field)
      : OpFaceEle(mass_field, mass_field, OpFaceEle::OPROWCOL) {
    sYmm = true;
    cerr << "OpAssembleLhsVV()" << endl;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;

    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    if (nb_row_dofs && nb_col_dofs) {

      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();

      auto t_row_v_base = row_data.getFTensor0N();

      auto t_w = getFTensor0IntegrationWeight();
      const double ts_a = getFEMethod()->ts_a;
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_v_base = col_data.getFTensor0N(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            mat(rr, cc) += -(ts_a * t_row_v_base * t_col_v_base) * a;

            ++t_col_v_base;
          }
          ++t_row_v_base;
        }
        ++t_w;
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
  MatrixDouble mat, transMat;
};

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
    //cerr << "ts_step : " << ts_step << endl; 
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProc;
};

}; // namespace ReactionDiffusion

#endif //__RDOPERATORS_HPP__

// #include <stdlib.h>
// #include <BasicFiniteElements.hpp>
// #include <RDOperators.hpp>

// using namespace MoFEM;
// using namespace ReactionDiffusion;

// static char help[] = "...\n\n";

// struct RDProblem {
// public:
//   RDProblem(MoFEM::Core &core, const int order) : m_field(core), order(order) {
//     vol_ele_slow_rhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));
//     natural_bdry_ele_slow_rhs =
//         boost::shared_ptr<BoundaryEle>(new BoundaryEle(m_field));
//     vol_ele_stiff_rhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));
//     vol_ele_stiff_lhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));
//     post_proc = boost::shared_ptr<PostProcFaceOnRefinedMesh>(
//         new PostProcFaceOnRefinedMesh(m_field));

//     data1 = boost::shared_ptr<PreviousData>(new PreviousData());
//     data2 = boost::shared_ptr<PreviousData>(new PreviousData());
//     data3 = boost::shared_ptr<PreviousData>(new PreviousData());

//     flux_values_ptr1 =
//         boost::shared_ptr<MatrixDouble>(data1, &data1->flux_values);
//     flux_divs_ptr1 = boost::shared_ptr<VectorDouble>(data1, &data1->flux_divs);
//     mass_values_ptr1 =
//         boost::shared_ptr<VectorDouble>(data1, &data1->mass_values);
//     mass_dots_ptr1 = boost::shared_ptr<VectorDouble>(data1, &data1->mass_dots);

//     flux_values_ptr2 =
//         boost::shared_ptr<MatrixDouble>(data2, &data2->flux_values);
//     flux_divs_ptr2 = boost::shared_ptr<VectorDouble>(data2, &data2->flux_divs);
//     mass_values_ptr2 =
//         boost::shared_ptr<VectorDouble>(data2, &data2->mass_values);
//     mass_dots_ptr2 = boost::shared_ptr<VectorDouble>(data2, &data2->mass_dots);

//     flux_values_ptr3 =
//         boost::shared_ptr<MatrixDouble>(data3, &data3->flux_values);
//     flux_divs_ptr3 = boost::shared_ptr<VectorDouble>(data3, &data3->flux_divs);
//     mass_values_ptr3 =
//         boost::shared_ptr<VectorDouble>(data3, &data3->mass_values);
//     mass_dots_ptr3 = boost::shared_ptr<VectorDouble>(data3, &data3->mass_dots);
//   }

//   // RDProblem(const int order) : order(order){}
//   void run_analysis();

// private:
//   void setup_system();
//   void add_fe(std::string mass_field, std::string flux_field);
//   void extract_bd_ents(std::string ESSENTIAL, std::string NATURAL);
//   void extract_initial_ents(int block_id, Range &surface);
//   void update_slow_rhs(std::string mass_fiedl,
//                        boost::shared_ptr<VectorDouble> &mass_ptr);
//   void push_slow_rhs(std::string mass_field, std::string flux_field,
//                      boost::shared_ptr<PreviousData> &data);
//   void update_vol_fe(boost::shared_ptr<FaceEle> &vol_ele,
//                      boost::shared_ptr<PreviousData> &data);
//   void update_stiff_rhs(std::string mass_field, std::string flux_field,
//                         boost::shared_ptr<VectorDouble> &mass_ptr,
//                         boost::shared_ptr<MatrixDouble> &flux_ptr,
//                         boost::shared_ptr<VectorDouble> &mass_dot_ptr,
//                         boost::shared_ptr<VectorDouble> &flux_div_ptr);
//   void push_stiff_rhs(std::string mass_field, std::string flux_field,
//                       boost::shared_ptr<PreviousData> &data);
//   void update_stiff_lhs(std::string mass_fiedl, std::string flux_field,
//                         boost::shared_ptr<VectorDouble> &mass_ptr,
//                         boost::shared_ptr<MatrixDouble> &flux_ptr);
//   void push_stiff_lhs(std::string mass_field, std::string flux_field,
//                       boost::shared_ptr<PreviousData> &data);

//   void set_integration_rule();
//   void apply_IC(boost::shared_ptr<FaceEle> &initial_ele);
//   void apply_BC(std::string flux_field);
//   void loop_fe();
//   void post_proc_fields(std::string mass_field, std::string flux_field);
//   void output_result();
//   void solve();

//   MoFEM::Interface &m_field;
//   Simple *simple_interface;
//   SmartPetscObj<DM> dm;
//   SmartPetscObj<TS> ts;

//   Range essential_bdry_ents;
//   Range natural_bdry_ents;

//   Range inner_surface1; // nb_species times
//   Range inner_surface2;
//   Range inner_surface3;

//   int order;

//   boost::shared_ptr<FaceEle> vol_ele_slow_rhs;
//   boost::shared_ptr<FaceEle> vol_ele_stiff_rhs;
//   boost::shared_ptr<FaceEle> vol_ele_stiff_lhs;
//   boost::shared_ptr<BoundaryEle> natural_bdry_ele_slow_rhs;
//   boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc;
//   boost::shared_ptr<Monitor> monitor_ptr;

//   boost::shared_ptr<PreviousData> data1; // nb_species times
//   boost::shared_ptr<PreviousData> data2;
//   boost::shared_ptr<PreviousData> data3;

//   boost::shared_ptr<MatrixDouble> flux_values_ptr1; // nb_species times
//   boost::shared_ptr<MatrixDouble> flux_values_ptr2;
//   boost::shared_ptr<MatrixDouble> flux_values_ptr3;

//   boost::shared_ptr<VectorDouble> flux_divs_ptr1; // nb_species times
//   boost::shared_ptr<VectorDouble> flux_divs_ptr2;
//   boost::shared_ptr<VectorDouble> flux_divs_ptr3;

//   boost::shared_ptr<VectorDouble> mass_values_ptr1; // nb_species times
//   boost::shared_ptr<VectorDouble> mass_values_ptr2;
//   boost::shared_ptr<VectorDouble> mass_values_ptr3;

//   boost::shared_ptr<VectorDouble> mass_dots_ptr1; // nb_species times
//   boost::shared_ptr<VectorDouble> mass_dots_ptr2;
//   boost::shared_ptr<VectorDouble> mass_dots_ptr3;

//   boost::shared_ptr<ForcesAndSourcesCore> null;
// };

// void RDProblem::setup_system() {
//   CHKERR m_field.getInterface(simple_interface);
//   CHKERR simple_interface->getOptions();
//   CHKERR simple_interface->loadFile();
// }

// void RDProblem::add_fe(std::string mass_field, std::string flux_field) {
//   CHKERR simple_interface->addDomainField(mass_field, L2,
//                                           AINSWORTH_LEGENDRE_BASE, 1);

//   CHKERR simple_interface->addDomainField(flux_field, HCURL,
//                                           AINSWORTH_LEGENDRE_BASE, 1);

//   CHKERR simple_interface->addBoundaryField(flux_field, HCURL,
//                                             DEMKOWICZ_JACOBI_BASE, 1);

//   CHKERR simple_interface->setFieldOrder(mass_field, order - 1);
//   CHKERR simple_interface->setFieldOrder(flux_field, order);
// }
// void RDProblem::extract_bd_ents(std::string essential, std::string natural) {
//   for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
//     string name = it->getName();
//     if (name.compare(0, 14, natural) == 0) {

//       CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), 1,
//                                                  natural_bdry_ents, true);
//     } else if (name.compare(0, 14, essential) == 0) {
//       CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), 1,
//                                                  essential_bdry_ents, true);
//     }
//   }
// }

// void RDProblem::extract_initial_ents(int block_id, Range &surface) {
//   if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id,
//                                                             BLOCKSET)) {
//     CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
//         block_id, BLOCKSET, 2, surface, true);
//   }
// }
// void RDProblem::update_slow_rhs(std::string mass_field,
//                                 boost::shared_ptr<VectorDouble> &mass_ptr) {
//   vol_ele_slow_rhs->getOpPtrVector().push_back(
//       new OpCalculateScalarFieldValues(mass_field, mass_ptr));
// }

// void RDProblem::push_slow_rhs(std::string mass_field, std::string flux_field,
//                               boost::shared_ptr<PreviousData> &data) {

//   vol_ele_slow_rhs->getOpPtrVector().push_back(
//       new OpAssembleSlowRhsV(mass_field, data));

//   natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
//       new OpSetContrariantPiolaTransformOnEdge());

//   natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
//       new OpAssembleNaturalBCRhsTau(flux_field, natural_bdry_ents));

//   natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
//       new OpEssentialBC(flux_field, essential_bdry_ents));
// }

// void RDProblem::update_vol_fe(boost::shared_ptr<FaceEle> &vol_ele,
//                               boost::shared_ptr<PreviousData> &data) {
//   vol_ele->getOpPtrVector().push_back(new OpCalculateJacForFace(data->jac));
//   vol_ele->getOpPtrVector().push_back(
//       new OpCalculateInvJacForFace(data->inv_jac));
//   vol_ele->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());

//   vol_ele->getOpPtrVector().push_back(
//       new OpSetContravariantPiolaTransformFace(data->jac));

//   vol_ele->getOpPtrVector().push_back(new OpSetInvJacHcurlFace(data->inv_jac));
// }
// void RDProblem::update_stiff_rhs(
//     std::string mass_field, std::string flux_field,
//     boost::shared_ptr<VectorDouble> &mass_ptr,
//     boost::shared_ptr<MatrixDouble> &flux_ptr,
//     boost::shared_ptr<VectorDouble> &mass_dot_ptr,
//     boost::shared_ptr<VectorDouble> &flux_div_ptr) {

//   vol_ele_stiff_rhs->getOpPtrVector().push_back(
//       new OpCalculateScalarFieldValues(mass_field, mass_ptr));

//   vol_ele_stiff_rhs->getOpPtrVector().push_back(
//       new OpCalculateHdivVectorField<3>(flux_field, flux_ptr));

//   vol_ele_stiff_rhs->getOpPtrVector().push_back(
//       new OpCalculateScalarValuesDot(mass_field, mass_dot_ptr));

//   vol_ele_stiff_rhs->getOpPtrVector().push_back(
//       new OpCalculateHdivVectorDivergence<3, 2>(flux_field, flux_div_ptr));
// }

// void RDProblem::push_stiff_rhs(std::string mass_field, std::string flux_field,
//                                boost::shared_ptr<PreviousData> &data) {
//   vol_ele_stiff_rhs->getOpPtrVector().push_back(
//       new OpAssembleStiffRhsTau<3>(flux_field, data));

//   vol_ele_stiff_rhs->getOpPtrVector().push_back(
//       new OpAssembleStiffRhsV<3>(mass_field, data));
// }

// void RDProblem::update_stiff_lhs(std::string mass_field, std::string flux_field,
//                                  boost::shared_ptr<VectorDouble> &mass_ptr,
//                                  boost::shared_ptr<MatrixDouble> &flux_ptr) {
//   vol_ele_stiff_lhs->getOpPtrVector().push_back(
//       new OpCalculateScalarFieldValues(mass_field, mass_ptr));

//   vol_ele_stiff_lhs->getOpPtrVector().push_back(
//       new OpCalculateHdivVectorField<3>(flux_field, flux_ptr));
// }

// void RDProblem::push_stiff_lhs(std::string mass_field, std::string flux_field,
//                                boost::shared_ptr<PreviousData> &data) {

//   vol_ele_stiff_lhs->getOpPtrVector().push_back(
//       new OpAssembleLhsTauTau<3>(flux_field, data));

//   vol_ele_stiff_lhs->getOpPtrVector().push_back(
//       new OpAssembleLhsVV(mass_field));

//   vol_ele_stiff_lhs->getOpPtrVector().push_back(
//       new OpAssembleLhsTauV<3>(flux_field, mass_field, data));

//   vol_ele_stiff_lhs->getOpPtrVector().push_back(
//       new OpAssembleLhsVTau(mass_field, flux_field));
// }

// void RDProblem::set_integration_rule() {
//   auto vol_rule = [](int, int, int p) -> int { return 2 * p; };
//   vol_ele_slow_rhs->getRuleHook = vol_rule;
//   natural_bdry_ele_slow_rhs->getRuleHook = vol_rule;

//   vol_ele_stiff_rhs->getRuleHook = vol_rule;

//   vol_ele_stiff_lhs->getRuleHook = vol_rule;
// }

// void RDProblem::apply_IC(boost::shared_ptr<FaceEle> &initial_ele) {

//   CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
//                                   initial_ele);
// }

// void RDProblem::apply_BC(std::string flux_field) {

//   CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
//       "SimpleProblem", flux_field, essential_bdry_ents);
// }
// void RDProblem::loop_fe() {

//   CHKERR TSSetType(ts, TSARKIMEX);
//   CHKERR TSARKIMEXSetType(ts, TSARKIMEXA2);

//   CHKERR DMMoFEMTSSetIJacobian(dm, simple_interface->getDomainFEName(),
//                                vol_ele_stiff_lhs, null, null);

//   CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getDomainFEName(),
//                                vol_ele_stiff_rhs, null, null);

//   CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getDomainFEName(),
//                                  vol_ele_slow_rhs, null, null);
//   CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getBoundaryFEName(),
//                                  natural_bdry_ele_slow_rhs, null, null);
// }

// void RDProblem::post_proc_fields(std::string mass_field,
//                                  std::string flux_field) {
//   post_proc->addFieldValuesPostProc("MASS");
//   post_proc->addFieldValuesPostProc("FLUX");
// }

// void RDProblem::output_result() {

//   CHKERR DMMoFEMTSSetMonitor(dm, ts, simple_interface->getDomainFEName(),
//                              monitor_ptr, null, null);
// }
// void RDProblem::solve() {
//   // Create solution vector
//   SmartPetscObj<Vec> X;
//   CHKERR DMCreateGlobalVector_MoFEM(dm, X);
//   CHKERR DMoFEMMeshToLocalVector(dm, X, INSERT_VALUES, SCATTER_FORWARD);
//   // Solve problem
//   double ftime = 1;
//   CHKERR TSSetDM(ts, dm);
//   CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
//   CHKERR TSSetSolution(ts, X);
//   CHKERR TSSetFromOptions(ts);
//   CHKERR TSSolve(ts, X);
// }

// void RDProblem::run_analysis() {
//   setup_system();           // only once
//   add_fe("MASS1", "FLUX1"); // nb_species times
//   // add_fe("MASS2", "FLUX2");
//   // add_fe("MASS3", "FLUX3");

//   CHKERR simple_interface->setUp();
//   extract_bd_ents("ESSENTIAL", "NATURAL"); // nb_species times

//   extract_initial_ents(2, inner_surface1);
//   // extract_initial_ents(3, inner_surface2);
//   // extract_initial_ents(4, inner_surface3);

//   update_slow_rhs("MASS1", mass_values_ptr1);
//   // update_slow_rhs("MASS2", mass_values_ptr2);
//   // update_slow_rhs("MASS3", mass_values_ptr3);

//   vol_ele_slow_rhs->getOpPtrVector().push_back(
//       new OpComputeSlowValue("MASS1", data1, data2, data3));
//   // vol_ele_slow_rhs->getOpPtrVector().push_back(
//   //     new OpComputeSlowValue("MASS2", data1, data2, data3));
//   // vol_ele_slow_rhs->getOpPtrVector().push_back(
//   //     new OpComputeSlowValue("MASS3", data1, data2, data3));

//   push_slow_rhs("MASS1", "FLUX1", data1); // nb_species times
//   // push_slow_rhs("MASS2", "FLUX2", data2);
//   // push_slow_rhs("MASS3", "FLUX3", data3);

//   update_vol_fe(vol_ele_stiff_rhs, data1);

//   update_stiff_rhs("MASS1", "FLUX1", mass_values_ptr1, flux_values_ptr1,
//                    mass_dots_ptr1, flux_divs_ptr1);
//   // update_stiff_rhs("MASS2", "FLUX2", mass_values_ptr2, flux_values_ptr2,
//   //                  mass_dots_ptr2, flux_divs_ptr2);
//   // update_stiff_rhs("MASS3", "FLUX3", mass_values_ptr3, flux_values_ptr3,
//   //                  mass_dots_ptr3, flux_divs_ptr3);

//   push_stiff_rhs("MASS1", "FLUX1", data1); // nb_species times
//   // push_stiff_rhs("MASS2", "FLUX2", data2);
//   // push_stiff_rhs("MASS3", "FLUX3", data3);

//   update_vol_fe(vol_ele_stiff_lhs, data1);

//   update_stiff_lhs("MASS1", "FLUX1", mass_values_ptr1, flux_values_ptr1);
//   // update_stiff_lhs("MASS2", "FLUX2", mass_values_ptr2, flux_values_ptr2);
//   // update_stiff_lhs("MASS3", "FLUX3", mass_values_ptr3, flux_values_ptr3);

//   push_stiff_lhs("MASS1", "FLUX1", data1); // nb_species times
//   // push_stiff_lhs("MASS2", "FLUX2", data2);
//   // push_stiff_lhs("MASS3", "FLUX3", data3);

//   set_integration_rule();

//   dm = simple_interface->getDM();
//   ts = createTS(m_field.get_comm());

//   boost::shared_ptr<FaceEle> initial_mass_ele(new FaceEle(m_field));

//   initial_mass_ele->getOpPtrVector().push_back(
//       new OpInitialMass("MASS1", inner_surface1));
//   // initial_mass_ele->getOpPtrVector().push_back(
//   //     new OpInitialMass("MASS2", inner_surface2));
//   // initial_mass_ele->getOpPtrVector().push_back(
//   //     new OpInitialMass("MASS3", inner_surface3));

//   apply_IC(initial_mass_ele); // only once

//   apply_BC("FLUX1"); // nb_species times
//   // apply_BC("FLUX2");
//   // apply_BC("FLUX3");

//   loop_fe();                                 // only once
//   post_proc->generateReferenceElementMesh(); // only once

//   post_proc_fields("MASS1", "FLUX1");
//   // post_proc_fields("MASS2", "FLUX2");
//   // post_proc_fields("MASS3", "FLUX3");

//   auto dm = simple_interface->getDM();
//   monitor_ptr = boost::shared_ptr<Monitor>(
//       new Monitor(dm, post_proc)); // nb_species times

//   output_result(); // only once
//   solve();         // only once
// }

// int main(int argc, char *argv[]) {
//   const char param_file[] = "param_file.petsc";
//   MoFEM::Core::Initialize(&argc, &argv, param_file, help);
//   try {
//     moab::Core mb_instance;
//     moab::Interface &moab = mb_instance;
//     MoFEM::Core core(moab);
//     DMType dm_name = "DMMOFEM";
//     CHKERR DMRegister_MoFEM(dm_name);

//     int order = 3;
//     RDProblem reac_diff_problem(core, order);
//     reac_diff_problem.run_analysis();
//   }
//   CATCH_ERRORS;
//   MoFEM::Core::Finalize();
//   return 0;
// }