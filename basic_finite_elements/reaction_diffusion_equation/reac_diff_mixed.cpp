/**
 * \file reaction_diffusion_equation.cpp
 * \example reaction_diffusion_equation.cpp
 *
 **/
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

#include <stdlib.h>
#include <BasicFiniteElements.hpp>
using namespace MoFEM;
static char help[] = "...\n\n";

namespace SoftTissue {
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

const double B = 1e-2;
const double B0 = 1e-3;
const double B_epsilon = 0.0;

const double r = 1.0;

const int save_every_nth_step = 4;
const int order = 3; ///< approximation order
const double natural_bc_values = 0.0;
const double essential_bc_values = 0.0;
// const int dim = 3;
FTensor::Index<'i', 3> i;

auto wrap_matrix3_ftensor = [](MatrixDouble &m) {
  return FTensor::Tensor2<FTensor::PackPtr<double *, 3>, 3, 3>(
      &m(0, 0), &m(0, 1), &m(0, 2), &m(1, 0), &m(1, 1), &m(1, 2), &m(2, 0),
      &m(2, 1), &m(2, 2));
};

struct CommonData {
  MatrixDouble flux_values; ///< flux values at integration points
  VectorDouble flux_divs;   ///< flux divergences at integration points
  VectorDouble mass_values; ///< mass values at integration points
  MatrixDouble mass_grads;
  VectorDouble mass_dots; ///< mass rates at integration points

  MatrixDouble jac;
  MatrixDouble inv_jac; ///< Inverse of element jacobian at integration points

  CommonData() {
    jac.resize(2,2,false);
    inv_jac.resize(2,2,false);
  }
};

struct OpEssentialBC : public OpBoundaryEle {
  OpEssentialBC(const std::string &flux_field, Range &essential_bd_ents)
      : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW),
        essential_bd_ents(essential_bd_ents) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data){
    MoFEMFunctionBegin;
    EntityType type1 = type;
    int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      EntityHandle fe_ent = getFEEntityHandle();
      bool is_essential = (essential_bd_ents.find(fe_ent) != essential_bd_ents.end());
      if (is_essential) {
        int nb_gauss_pts = getGaussPts().size2();
        int size2 =
            data.getN().size2(); 
            if (3*nb_dofs != static_cast<int>(data.getN().size2()))
                SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                        "wrong number of dofs");
        nN.resize(nb_dofs, nb_dofs, false);
        nF.resize(nb_dofs, false);
        nN.clear();
        nF.clear();

        auto t_row_tau = data.getFTensor1N<3>();

        auto dir = getDirection();
        double len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);

        FTensor::Tensor1<double, 3>
                t_normal(-dir[1]/len, dir[0]/len, dir[2]/len);

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
  : OpFaceEle(mass_field, OpFaceEle::OPROW), innerSurface(inner_surface)
  {
    cerr << "OpInitialMass()" << endl;
  }
  MatrixDouble nN;
  VectorDouble nF;
  Range &innerSurface;
  MoFEMErrorCode doWork(int side, EntityType type,
                        EntData &data) {
    MoFEMFunctionBegin;
    int nb_dofs = data.getFieldData().size();
      if (nb_dofs) {
        EntityHandle fe_ent = getFEEntityHandle();
        bool is_inner_side =
            (innerSurface.find(fe_ent) != innerSurface.end());
        if(is_inner_side){
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

          for(int gg = 0; gg < nb_gauss_pts; gg++){
            const double a = t_w * vol;
            const double init_val_at_gg = 0.5;
            for(int rr = 0; rr != nb_dofs; rr++){
              auto t_col_mass = data.getFTensor0N(gg, 0);
              nF[rr] += a * init_val_at_gg * t_row_mass;
              for(int cc = 0; cc != nb_dofs; cc++){
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
                     boost::shared_ptr<CommonData> &common_data)
      : OpFaceEle(mass_field, OpFaceEle::OPROW), commonData(common_data)
      {
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
      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double f = a * r * t_mass_value * (1 - t_mass_value);
        for (int rr = 0; rr != nb_dofs; ++rr) {
          auto t_col_v_base = data.getFTensor0N(gg, 0);
          vecF[rr] +=  f * t_row_v_base;
          for (int cc = 0; cc != nb_dofs; ++cc) {
            mat(rr, cc) += a * t_row_v_base * t_col_v_base;
            ++t_col_v_base;
          }
            ++t_row_v_base;
        }
        ++t_mass_value;
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
  boost::shared_ptr<CommonData> commonData;
  VectorDouble vecF;
  MatrixDouble mat;
};

// 5. RHS contribution of the natural boundary condition
struct OpAssembleNaturalBCRhsTau : OpBoundaryEle // R_tau_2
{
  OpAssembleNaturalBCRhsTau(std::string flux_field,
                            boost::shared_ptr<CommonData> &common_data,
                            Range &natural_bd_ents)
      : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW),
        commonData(common_data),
        natural_bd_ents(natural_bd_ents) {
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
  boost::shared_ptr<CommonData> commonData;
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
                        boost::shared_ptr<CommonData> &data)
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
                      div_base * t_mass_value) * a;
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
  boost::shared_ptr<CommonData> commonData;
  VectorDouble vecF;
};
// 4. Assembly of F_v
template <int dim>
struct OpAssembleStiffRhsV : OpFaceEle // F_V
{
  OpAssembleStiffRhsV(std::string flux_field,
                      boost::shared_ptr<CommonData> &data)
      : OpFaceEle(flux_field, OpFaceEle::OPROW),
        commonData(data){

            cerr << "OpAssembleStiffRhsV()" << endl;}

        MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    //cerr << "In StiffRhsV ..." << endl;
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
  boost::shared_ptr<CommonData> commonData;
  VectorDouble vecF;
};



// Tangent operator
// //**********************************************
// 7. Tangent assembly for F_tautau excluding the essential boundary condition
template <int dim>
struct OpAssembleLhsTauTau : OpFaceEle // A_TauTau_1
{
  OpAssembleLhsTauTau(std::string flux_field,
                      boost::shared_ptr<CommonData> &commonData,
                      Range &essential_bd_ents)
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
  boost::shared_ptr<CommonData> commonData;
  MatrixDouble mat, transMat;
  Range essential_bd_ents;
};


// 9. Assembly of tangent for F_tau_v excluding the essential bc
template <int dim>
struct OpAssembleLhsTauV : OpFaceEle // E_TauV
{
  OpAssembleLhsTauV(std::string flux_field, std::string mass_field,
                    boost::shared_ptr<CommonData> &data)
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
  boost::shared_ptr<CommonData> commonData;
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
      : OpFaceEle(mass_field, mass_field, OpFaceEle::OPROWCOL)
    {
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
    MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProc;
};

}; // namespace SoftTissue

using namespace SoftTissue;

int main(int argc, char *argv[]) {
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);
  try {

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
    CHKERR simple_interface->addDomainField("MASS", L2, AINSWORTH_LEGENDRE_BASE,
                                            1);

    CHKERR simple_interface->addDomainField("FLUX", HCURL,
                                            AINSWORTH_LEGENDRE_BASE, 1);

    CHKERR simple_interface->addBoundaryField("FLUX", HCURL,
                                              DEMKOWICZ_JACOBI_BASE, 1);

    CHKERR simple_interface->setFieldOrder("MASS", order - 1);
    CHKERR simple_interface->setFieldOrder("FLUX", order);

    CHKERR simple_interface->setUp();

    
    Range essential_bdry_ents;
    Range natural_bdry_ents;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      string name = it->getName();
      if (name.compare(0, 14, "NATURAL") == 0) {

        CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), 1,
                                                   natural_bdry_ents, true);
      } else if (name.compare(0, 14, "ESSENTIAL") == 0) {
        CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), 1,
                                                   essential_bdry_ents, true);
      }
    }

    // blockset 1
    const int block_id1 = 2;
    Range inner_surface;

    if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id1,
                                                              BLOCKSET)) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          block_id1, BLOCKSET, 2, inner_surface, true);
    }

    boost::shared_ptr<CommonData> data(new CommonData());

    auto flux_values_ptr =
        boost::shared_ptr<MatrixDouble>(data, &data->flux_values);
    auto flux_divs_ptr =
        boost::shared_ptr<VectorDouble>(data, &data->flux_divs);
    auto mass_values_ptr =
        boost::shared_ptr<VectorDouble>(data, &data->mass_values);
    auto mass_grads_ptr =
        boost::shared_ptr<MatrixDouble>(data, &data->mass_grads);
    auto mass_dots_ptr =
        boost::shared_ptr<VectorDouble>(data, &data->mass_dots);

    // ************* creating finite element instances for the lhs and rhs *****
    boost::shared_ptr<FaceEle> vol_ele_slow_rhs(new FaceEle(m_field));
    boost::shared_ptr<BoundaryEle> natural_bdry_ele_slow_rhs(
        new BoundaryEle(m_field));

    boost::shared_ptr<FaceEle> vol_ele_stiff_rhs(new FaceEle(m_field));
    // boost::shared_ptr<BoundaryEle> essential_bdry_ele_stiff_rhs(
    //     new BoundaryEle(m_field));

    boost::shared_ptr<FaceEle> vol_ele_stiff_lhs(new FaceEle(m_field));
    // boost::shared_ptr<BoundaryEle> essential_bdry_ele_stiff_lhs(
    //     new BoundaryEle(m_field));

    // ************* pushing operators for slow rhs system vector (G)

    vol_ele_slow_rhs->getOpPtrVector().push_back(
    new OpCalculateScalarFieldValues("MASS", mass_values_ptr));

    vol_ele_slow_rhs->getOpPtrVector().push_back(
    new OpAssembleSlowRhsV("MASS", data));
  

    natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
        new OpSetContrariantPiolaTransformOnEdge());

    natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
        new OpAssembleNaturalBCRhsTau("FLUX", data, natural_bdry_ents));

    natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
        new OpEssentialBC("FLUX", essential_bdry_ents));

    //************ push operators for stiff rhs system vector (F)
    //***************************
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateJacForFace(data->jac));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->inv_jac));
    vol_ele_stiff_rhs->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpSetContravariantPiolaTransformFace(data->jac));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpSetInvJacHcurlFace(data->inv_jac));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("MASS", mass_values_ptr));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateHdivVectorField<3>("FLUX", flux_values_ptr));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarValuesDot("MASS", mass_dots_ptr));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateHdivVectorDivergence<3, 2>("FLUX", flux_divs_ptr));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpAssembleStiffRhsTau<3>("FLUX", data));

    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpAssembleStiffRhsV<3>("MASS", data));



    // **************** pushing operators for the stiff Lhs system matrix (DF)
    // *********

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateJacForFace(data->jac));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateInvJacForFace(data->inv_jac));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpSetContravariantPiolaTransformFace(data->jac));
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpSetInvJacHcurlFace(data->inv_jac));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("MASS", mass_values_ptr));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateHdivVectorField<3>("FLUX", flux_values_ptr));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpAssembleLhsTauTau<3>("FLUX", data, essential_bdry_ents));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(new OpAssembleLhsVV("MASS"));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpAssembleLhsTauV<3>("FLUX", "MASS", data));

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpAssembleLhsVTau("MASS", "FLUX"));

    // essential_bdry_ele_stiff_lhs->getOpPtrVector().push_back(
    //     new OpSetContrariantPiolaTransformOnEdge());
    // essential_bdry_ele_stiff_lhs->getOpPtrVector().push_back(
    //     new OpAssembleLhsEssentialBCTauTau<3>("FLUX", essential_bdry_ents));

    // set integration rules for the slow rhs, stiff rhs, stiff Lhs

    auto vol_rule = [](int, int, int p) -> int { return 2 * p; };
    vol_ele_slow_rhs->getRuleHook = vol_rule;
    natural_bdry_ele_slow_rhs->getRuleHook = vol_rule;

    vol_ele_stiff_rhs->getRuleHook = vol_rule;
    // essential_bdry_ele_stiff_rhs->getRuleHook = vol_rule;

    vol_ele_stiff_lhs->getRuleHook = vol_rule;
    // essential_bdry_ele_stiff_lhs->getRuleHook = vol_rule;

    // Create element for post-processing
    boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc =
        boost::shared_ptr<PostProcFaceOnRefinedMesh>(
            new PostProcFaceOnRefinedMesh(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> null;

    // Genarte post-processing mesh
    post_proc->generateReferenceElementMesh();

    post_proc->addFieldValuesPostProc("MASS");
    post_proc->addFieldValuesPostProc("FLUX");

    // Get PETSc discrete manager
    auto dm = simple_interface->getDM();

    

    boost::shared_ptr<FaceEle> initial_mass_ele(new FaceEle(m_field));
    

    initial_mass_ele->getOpPtrVector().push_back(
        new OpInitialMass("MASS", inner_surface));

   

    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    initial_mass_ele);
    
    CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
        "SimpleProblem", "FLUX", essential_bdry_ents);

    

    auto ts = createTS(m_field.get_comm());
    // Use IMEX solver, i.e. implicit/explicit solver
    CHKERR TSSetType(ts, TSARKIMEX);
    CHKERR TSARKIMEXSetType(ts, TSARKIMEXA2);
    // Add element to calculate lhs of stiff part
    CHKERR DMMoFEMTSSetIJacobian(dm, simple_interface->getDomainFEName(),
                                 vol_ele_stiff_lhs, null, null);
    // CHKERR DMMoFEMTSSetIJacobian(dm, simple_interface->getBoundaryFEName(),
    //                              essential_bdry_ele_stiff_lhs, null, null);

    // Add element to calculate rhs of stiff part
    CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getDomainFEName(),
                                 vol_ele_stiff_rhs, null, null);
    // CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getBoundaryFEName(),
    //                              essential_bdry_ele_stiff_rhs, null, null);

    // Add element to calculate rhs of slow (nonlinear) part
    CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getDomainFEName(),
                                   vol_ele_slow_rhs, null, null);
    CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getBoundaryFEName(),
                                   natural_bdry_ele_slow_rhs, null, null);

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

// struct OpAssembleMassTauTau : OpFaceEle {
//   OpAssembleMassTauTau(std::string flux_field, SmartPetscObj<Mat> mass_matrix)
//       : OpFaceEle(flux_field, flux_field, OpFaceEle::OPROWCOL), M(mass_matrix) {
//     sYmm = true;
//   }

//   MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
//                         EntityType col_type, EntData &row_data,
//                         EntData &col_data) {
//     MoFEMFunctionBegin;

//     const int nb_row_dofs = row_data.getIndices().size();
//     const int nb_col_dofs = col_data.getIndices().size();
//     if (nb_row_dofs && nb_col_dofs && row_type == col_type &&
//         row_side == col_side) {
//       // cerr << "In mass tau tau ..." << endl;
//       mat.resize(nb_row_dofs, nb_col_dofs, false);
//       mat.clear();
//       for (int rr = 0; rr != nb_row_dofs; ++rr) {
//         mat(rr, rr) = 1.0;
//       }
//       CHKERR MatSetValues(M, row_data, col_data, &mat(0, 0), INSERT_VALUES);
//     }
//     MoFEMFunctionReturn(0);
//   }

// private:
//   MatrixDouble mat, transMat;
//   SmartPetscObj<Mat> M;
// };

// // Mass matrix corresponding to the balance of mass equation where
// // 02. there is a time derivative.
// struct OpAssembleMassVV : OpFaceEle {
//   OpAssembleMassVV(std::string mass_field, SmartPetscObj<Mat> mass_matrix)
//       : OpFaceEle(mass_field, mass_field, OpFaceEle::OPROWCOL), M(mass_matrix) {
//     sYmm = true;
//   }
//   MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
//                         EntityType col_type, EntData &row_data,
//                         EntData &col_data) {
//     MoFEMFunctionBegin;
//     const int nb_row_dofs = row_data.getIndices().size();
//     const int nb_col_dofs = col_data.getIndices().size();
//     if (nb_row_dofs && nb_col_dofs) {

//       const int nb_integration_pts = getGaussPts().size2();
//       mat.resize(nb_row_dofs, nb_col_dofs, false);
//       mat.clear();
//       auto t_row_base = row_data.getFTensor0N();
//       auto t_w = getFTensor0IntegrationWeight();
//       const double vol = getMeasure();
//       for (int gg = 0; gg != nb_integration_pts; ++gg) {
//         const double a = t_w * vol;
//         for (int rr = 0; rr != nb_row_dofs; ++rr) {
//           auto t_col_base = col_data.getFTensor0N(gg, 0);
//           for (int cc = 0; cc != nb_col_dofs; ++cc) {
//             mat(rr, cc) += a * t_row_base * t_col_base;
//             ++t_col_base;
//           }
//           ++t_row_base;
//         }
//         ++t_w;
//       }
//       CHKERR MatSetValues(M, row_data, col_data, &mat(0, 0), ADD_VALUES);
//       if (row_side != col_side || row_type != col_type) {
//         transMat.resize(nb_col_dofs, nb_row_dofs, false);
//         noalias(transMat) = trans(mat);
//         CHKERR MatSetValues(M, col_data, row_data, &transMat(0, 0), ADD_VALUES);
//       }
//     }
//     MoFEMFunctionReturn(0);
//   }

// private:
//   MatrixDouble mat, transMat;
//   SmartPetscObj<Mat> M;
// };

// // 1. RHS for explicit part of the flux equation excluding the essential
// // boundary
// template <int dim>
// struct OpAssembleSlowRhsTau : OpFaceEle // R_tau_1
// {
//   OpAssembleSlowRhsTau(std::string flux_field,
//                        boost::shared_ptr<CommonData> &common_data,
//                        Range &essential_bd_ents)
//       : OpFaceEle(flux_field, OpFaceEle::OPROW), commonData(common_data),
//         field(flux_field), essential_bd_ents(essential_bd_ents) {}

//   MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
//     MoFEMFunctionBegin;
//     const int nb_dofs = data.getIndices().size();

//     if (nb_dofs) {
//       EntityHandle row_side_ent = data.getFieldDofs()[0]->getEnt();
//       bool is_essential =
//           (essential_bd_ents.find(row_side_ent) != essential_bd_ents.end());
//       if (!is_essential) {
//         vecF.resize(nb_dofs, false);
//         vecF.clear();
//         const int nb_integration_pts = getGaussPts().size2();
//         auto t_mass_grad = getFTensor1FromMat<dim>(commonData->mass_grads);
//         auto t_mass_value = getFTensor0FromVec(commonData->mass_values);
//         auto t_tau_base = data.getFTensor1N<dim>();

//         auto t_w = getFTensor0IntegrationWeight();
//         const double vol = getMeasure();
//         for (int gg = 0; gg < nb_integration_pts; ++gg) {
//           const double K = B_epsilon + (B0 + B * t_mass_value);
//           const double K_inv = 1. / K;
//           const double a = vol * t_w;
//           for (int rr = 0; rr < nb_dofs; ++rr) {
//             vecF[rr] +=
//                 -(B_epsilon * K_inv * t_tau_base(i) * t_mass_grad(i)) * a;
//             ++t_tau_base;
//           }
//           ++t_mass_grad;
//           ++t_mass_value;
//         }
//       }
//       CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
//                           PETSC_TRUE);
//       CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
//                           ADD_VALUES);
//     }
//     MoFEMFunctionReturn(0);
//   }

// private:
//   boost::shared_ptr<CommonData> commonData;
//   VectorDouble vecF;
//   std::string field;
//   Range &essential_bd_ents;
// };

// // 8. Assembly of tangent matrix for the essential boundary condition
// template <int dim>
// struct OpAssembleLhsEssentialBCTauTau : OpBoundaryEle // A_TauTau_2
// {
//   OpAssembleLhsEssentialBCTauTau(std::string flux_field,
//                                  Range &essential_bd_ents)
//       : OpBoundaryEle(flux_field, flux_field, OpBoundaryEle::OPROWCOL),
//         essential_bd_ents(essential_bd_ents) {
//     sYmm = true;
//     cerr << "OpAssembleLhsEssentialBCTauTau()" << endl;
//   }

//   MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
//                         EntityType col_type, EntData &row_data,
//                         EntData &col_data) {
//     MoFEMFunctionBegin;
//     const int nb_row_dofs = row_data.getIndices().size();
//     const int nb_col_dofs = col_data.getIndices().size();

//     EntityHandle row_side_ent = getFEEntityHandle();

//     bool is_essential =
//         (essential_bd_ents.find(row_side_ent) != essential_bd_ents.end());

//     if (nb_row_dofs && nb_col_dofs) {
//       // cerr << "EssentialBCTauTau" << endl;

//       if (is_essential) {
//         // cerr << "is essential" << endl;

//         mat.resize(nb_row_dofs, nb_col_dofs, false);
//         mat.clear();
//         const int nb_integration_pts = getGaussPts().size2();
//         auto t_w = getFTensor0IntegrationWeight();
//         auto t_row_tau_base = row_data.getFTensor1N<3>();

//         auto dir = getDirection();
//         double len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
//         FTensor::Tensor1<double, 3> t_normal(-dir[1] / len, dir[0] / len,
//                                              dir[2] / len);

//         const double vol = getMeasure();
//         for (int gg = 0; gg != nb_integration_pts; ++gg) {
//           const double a = vol * t_w;
//           for (int rr = 0; rr != nb_row_dofs; ++rr) {
//             auto t_col_tau_base = col_data.getFTensor1N<3>(gg, 0);

//             for (int cc = 0; cc != nb_col_dofs; ++cc) {
//               mat(rr, cc) += ((t_row_tau_base(i) * t_normal(i)) *
//                               (t_col_tau_base(i) * t_normal(i))) *
//                              a;

//               ++t_col_tau_base;
//             }
//             ++t_row_tau_base;
//           }
//           ++t_w;
//         }
//         CHKERR MatSetValues(getFEMethod()->ts_B, row_data, col_data, &mat(0, 0),
//                             ADD_VALUES);
//         if (row_side != col_side || row_type != col_type) {
//           transMat.resize(nb_col_dofs, nb_row_dofs, false);
//           noalias(transMat) = trans(mat);
//           CHKERR MatSetValues(getFEMethod()->ts_B, col_data, row_data,
//                               &transMat(0, 0), ADD_VALUES);
//         }
//       }
//     }
//     MoFEMFunctionReturn(0);
//   }

// private:
//   MatrixDouble mat, transMat;
//   Range essential_bd_ents;
// // };

// // 6 RHS contribution of the essential boundary condition
// struct OpAssembleEssentialBCRhsTau : OpBoundaryEle // F_tau_2
// {
//   OpAssembleEssentialBCRhsTau(std::string flux_field,
//                               boost::shared_ptr<CommonData> &common_data,
//                               Range &essential_bd_ents)
//       : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW),
//         commonData(common_data), essential_bd_ents(essential_bd_ents) {
//     cerr << "OpAssembleEssentialBCRhsTau()" << endl;
//   }

//   MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
//     MoFEMFunctionBegin;
//     const int nb_dofs = data.getIndices().size();
//     if (nb_dofs) {
//       EntityHandle row_side_ent = data.getFieldDofs()[0]->getEnt();
//       bool is_essential =
//           (essential_bd_ents.find(row_side_ent) != essential_bd_ents.end());
//       // cerr << "In EssentialBCRhsTau" << endl;
//       // cerr << essential_bd_ents.size() << endl;
//       if (is_essential) {
//         // cerr << "is_essential: " << is_essential << endl;
//         vecF.resize(nb_dofs, false);
//         vecF.clear();
//         const int nb_integration_pts = getGaussPts().size2();
//         auto t_flux_value = getFTensor1FromMat<3>(commonData->flux_values);
//         // double tmp = t_flux_value(0);
//         // t_flux_value(0) = -t_flux_value(1);
//         // t_flux_value(0) = tmp;
//         auto t_tau_base = data.getFTensor1N<3>();

//         auto dir = getDirection();
//         double len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
//         FTensor::Tensor1<double, 3> t_normal(-dir[1] / len, dir[0] / len,
//                                              dir[2] / len);

//         // auto t_normal = getFTensor1FromVec(getDirection());
//         auto t_w = getFTensor0IntegrationWeight();
//         const double vol = getMeasure();
//         for (int gg = 0; gg < nb_integration_pts; ++gg) {
//           const double a = vol * t_w;
//           // double tmp = t_flux_value(0);
//           // t_flux_value(0) = -t_flux_value(1);
//           // t_flux_value(0) = tmp;
//           for (int rr = 0; rr < nb_dofs; ++rr) {
//             vecF[rr] +=
//                 (t_tau_base(i) * t_normal(i) *
//                  (t_flux_value(i) * t_normal(i) - essential_bc_values)) *
//                 a;
//             ++t_tau_base;
//           }
//           ++t_flux_value;
//           ++t_w;
//         }
//         CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
//                             PETSC_TRUE);
//         CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
//                             ADD_VALUES);
//       }
//     }

//     MoFEMFunctionReturn(0);
//   }

// private:
//   boost::shared_ptr<CommonData> commonData;
//   VectorDouble vecF;
//   Range essential_bd_ents;
// };
