#ifndef __ELECPHYSOPERATORS_HPP__
#define __ELECPHYSOPERATORS_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

namespace ElectroPhysiology {

using VolEle = MoFEM::VolumeElementForcesAndSourcesCore;

using FaceEle = MoFEM::FaceElementForcesAndSourcesCore;

using OpVolEle = VolEle::UserDataOperator;
using OpFaceEle = FaceEle::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;



const int save_every_nth_step = 1;

const double essen_value = 0;

FTensor::Index<'i', 3> i;


// problem parameters
const double alpha = 0.01; 
const double gma = 0.002; 
const double b = 0.15;
const double c = 8.00;
const double mu1 = 0.20;
const double mu2 = 0.30;

const double D_tilde = 0.01;



struct PreviousData {
  MatrixDouble flux_values;
  VectorDouble flux_divs;

  VectorDouble mass_dots;
  VectorDouble mass_values;

  VectorDouble slow_values;

  MatrixDouble jac;
  MatrixDouble inv_jac;

  PreviousData() {
    jac.resize(3, 3, false);
    inv_jac.resize(3, 3, false);
  }
};


struct OpEssentialBC : public OpFaceEle {
  OpEssentialBC(std::string flux_field, Range &essential_bd_ents)
      : OpFaceEle(flux_field, OpFaceEle::OPROW), essential_bd_ents(essential_bd_ents) {
  }
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
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

        double *normal_ptr;
        if (getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
          // HO geometry
          normal_ptr = &getNormalsAtGaussPts(0)[0];
        } else {
          // Linear geometry, i.e. constant normal on face
          normal_ptr = &getNormal()[0];
        }
        // set tensor from pointer
        FTensor::Tensor1<const double *, 3> t_normal(normal_ptr, &normal_ptr[1],
                                                     &normal_ptr[2], 3);

        auto t_w = getFTensor0IntegrationWeight();
        const double vol = getMeasure();
        double nrm2 = 0;
        for (int gg = 0; gg < nb_gauss_pts; gg++) {
          if (gg == 0) {
            nrm2 = sqrt(t_normal(i) * t_normal(i));
          }
          const double a = t_w * vol;
          for (int rr = 0; rr != nb_dofs; rr++) {
            FTensor::Tensor1<const double *, 3> t_col_tau(
                &data.getVectorN<3>(gg)(0, HVEC0),
                &data.getVectorN<3>(gg)(0, HVEC1),
                &data.getVectorN<3>(gg)(0, HVEC2), 3);
            nF[rr] += a * essen_value * t_row_tau(i) * t_normal(i) / nrm2;
            for (int cc = 0; cc != nb_dofs; cc++) {
              nN(rr, cc) += a * (t_row_tau(i) * t_normal(i)) *
                            (t_col_tau(i) * t_normal(i)) / (nrm2 * nrm2);
              ++t_col_tau;
            }
            ++t_row_tau;
          }
          // If HO geometry increment t_normal to next integration point
          if (getNormalsAtGaussPts().size1() == (unsigned int)nb_gauss_pts) {
            ++t_normal;
            nrm2 = sqrt(t_normal(i) * t_normal(i));
          }
          ++t_w;
        }

        cholesky_decompose(nN);
        cholesky_solve(nN, nF, ublas::lower());

        for (auto &dof : data.getFieldDofs()) {
          dof->getFieldData() = nF[dof->getEntDofIdx()];
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

struct OpInitialMass : public OpVolEle {
  OpInitialMass(const std::string &mass_field, Range &inner_surface,
                double init_val)
      : OpVolEle(mass_field, OpVolEle::OPROW), innerSurface(inner_surface),
        initVal(init_val) {}
  MatrixDouble nN;
  VectorDouble nF;
  Range &innerSurface;
  double initVal;
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
          // double r = ((double)rand() / (RAND_MAX));
          for (int rr = 0; rr != nb_dofs; rr++) {
            auto t_col_mass = data.getFTensor0N(gg, 0);
            nF[rr] += a * initVal * t_row_mass;
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


struct OpSolveRecovery : public OpVolEle {
  typedef boost::function<double(const double, const double, const double)>
      Method;
  OpSolveRecovery(const std::string &mass_field, 
                boost::shared_ptr<PreviousData> &data_u,
                boost::shared_ptr<PreviousData> &data_v,
                Method runge_kutta4)
      : OpVolEle(mass_field, OpVolEle::OPROW)
      , dataU(data_u) 
      , dataV(data_v)
      , rungeKutta4(runge_kutta4)
         {}
  boost::shared_ptr<PreviousData> dataU;
  boost::shared_ptr<PreviousData> dataV;
  Method rungeKutta4;

  MatrixDouble nN;
  VectorDouble nF;
  double initVal;
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    int nb_dofs = data.getFieldData().size();
    if (nb_dofs) {
        int nb_gauss_pts = getGaussPts().size2();
        if (nb_dofs != static_cast<int>(data.getN().size2()))
          SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                  "wrong number of dofs");
        nN.resize(nb_dofs, nb_dofs, false);
        nF.resize(nb_dofs, false);
        nN.clear();
        nF.clear();

        auto t_val_u = getFTensor0FromVec(dataU->mass_values);
        auto t_val_v = getFTensor0FromVec(dataV->mass_values);

        double dt;
        CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);

        auto t_row_mass = data.getFTensor0N();
        auto t_w = getFTensor0IntegrationWeight();
        const double vol = getMeasure();

        for (int gg = 0; gg < nb_gauss_pts; gg++) {
          const double a = t_w * vol;
          const double vn = rungeKutta4(t_val_u, t_val_v, dt);
          for (int rr = 0; rr != nb_dofs; rr++) {
            auto t_col_mass = data.getFTensor0N(gg, 0);
            nF[rr] += a * vn * t_row_mass;
            for (int cc = 0; cc != nb_dofs; cc++) {
              nN(rr, cc) += a * t_row_mass * t_col_mass;
              ++t_col_mass;
            }
            ++t_row_mass;
          }
          ++t_w;
          ++t_val_u;
          ++t_val_v;
        }

        cholesky_decompose(nN);
        cholesky_solve(nN, nF, ublas::lower());

        for (auto &dof : data.getFieldDofs()) {
          dof->getFieldData() = nF[dof->getEntDofIdx()];

          // this is only to check
          // data.getFieldData()[dof->getEntDofIdx()] = nF[dof->getEntDofIdx()];
        }
  
    }
    MoFEMFunctionReturn(0);
  }
};

// Assembly of RHS for explicit (slow)
// part//**************************************

// 2. RHS for explicit part of the mass balance equation
struct OpAssembleSlowRhsV : OpVolEle // R_V
{
  typedef boost::function<double(const double, const double)>
      Feval_u;
  OpAssembleSlowRhsV(std::string mass_field,
                     boost::shared_ptr<PreviousData> &data_u,
                     boost::shared_ptr<PreviousData> &data_v,
                     Feval_u rhs_u)
      : OpVolEle(mass_field, OpVolEle::OPROW)
      , dataU(data_u)
      , dataV(data_v)
      , rhsU(rhs_u) {}

  Feval_u rhsU;
  FTensor::Number<0> NX;
  FTensor::Number<1> NY;
  FTensor::Number<2> NZ;

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
   
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      if (nb_dofs != static_cast<int>(data.getN().size2()))
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "wrong number of dofs");
      vecF.resize(nb_dofs, false);
      mat.resize(nb_dofs, nb_dofs, false);
      vecF.clear();
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_val_u = getFTensor0FromVec(dataU->mass_values);
      auto t_val_v = getFTensor0FromVec(dataV->mass_values);


      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        
        double rhs = rhsU(t_val_u, t_val_v);
        const double a = vol * t_w;

        for (int rr = 0; rr != nb_dofs; ++rr) {
          auto t_col_v_base = data.getFTensor0N(gg, 0);
          vecF[rr] += a * rhs * t_row_v_base;  
          for (int cc = 0; cc != nb_dofs; ++cc) {
            mat(rr, cc) += a * t_row_v_base * t_col_v_base;
            ++t_col_v_base;
          }
          ++t_row_v_base;
        }
        ++t_val_u;
        ++t_val_v;
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
  boost::shared_ptr<PreviousData> dataU;
  boost::shared_ptr<PreviousData> dataV;
  VectorDouble vecF;
  MatrixDouble mat;
  
};

// // 5. RHS contribution of the natural boundary condition
// struct OpAssembleNaturalBCRhsTau : OpFaceEle // R_tau_2
// {
//   OpAssembleNaturalBCRhsTau(std::string flux_field, Range &natural_bd_ents)
//       : OpFaceEle(flux_field, OpFaceEle::OPROW),
//         natural_bd_ents(natural_bd_ents) {}

//   MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
//     MoFEMFunctionBegin;
//     const int nb_dofs = data.getIndices().size();

//     if (nb_dofs) {
//       EntityHandle row_side_ent = data.getFieldDofs()[0]->getEnt();

//       bool is_natural =
//           (natural_bd_ents.find(row_side_ent) != natural_bd_ents.end());
//       if (is_natural) {
//         // cerr << "In NaturalBCRhsTau..." << endl;
//         vecF.resize(nb_dofs, false);
//         vecF.clear();
//         const int nb_integration_pts = getGaussPts().size2();
//         auto t_tau_base = data.getFTensor1N<3>();

//         auto dir = getDirection();
//         FTensor::Tensor1<double, 3> t_normal(-dir[1], dir[0], dir[2]);

//         auto t_w = getFTensor0IntegrationWeight();

//         for (int gg = 0; gg != nb_integration_pts; ++gg) {
//           const double a = t_w;
//           for (int rr = 0; rr != nb_dofs; ++rr) {
//             vecF[rr] += (t_tau_base(i) * t_normal(i) * natu_value) * a;
//             ++t_tau_base;
//           }
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
//   VectorDouble vecF;
//   Range natural_bd_ents;
// };

// Assembly of RHS for the implicit (stiff) part excluding the essential
// boundary //**********************************
// 3. Assembly of F_tau excluding the essential boundary condition
template <int dim>
struct OpAssembleStiffRhsTau : OpVolEle //  F_tau_1
{
  OpAssembleStiffRhsTau(std::string flux_field,
                        boost::shared_ptr<PreviousData> &data)
      : OpVolEle(flux_field, OpVolEle::OPROW), commonData(data) {}

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

      auto t_tau_grad = data.getFTensor2DiffN<3, 3>();

      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      for (int gg = 0; gg < nb_integration_pts; ++gg) {

        const double K_inv = 1. / D_tilde;
        const double a = vol * t_w;
        for (int rr = 0; rr < nb_dofs; ++rr) {
          double div_base =
              t_tau_grad(0, 0) + t_tau_grad(1, 1) + t_tau_grad(2, 2);
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
struct OpAssembleStiffRhsV : OpVolEle // F_V
{
  OpAssembleStiffRhsV(std::string mass_field,
                      boost::shared_ptr<PreviousData> &data_u,
                      Range &stim_region)
      : OpVolEle(mass_field, OpVolEle::OPROW)
      , dataU(data_u)
      , stimRegion(stim_region) {}

  Range &stimRegion;

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    // cerr << "In StiffRhsV ..." << endl;
    if (nb_dofs) {

      vecF.resize(nb_dofs, false);
      vecF.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_dot_u = getFTensor0FromVec(dataU->mass_dots);


      auto t_div_u = getFTensor0FromVec(dataU->flux_divs);
      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      const double c_time = getFEMethod()->ts_t;

      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);

      double stim = 0.0;

      double T = 0;
      double duration = 1.0;

      if (T-dt < c_time && c_time <= T + duration) {
        EntityHandle stim_ent = getFEEntityHandle();
        if (stimRegion.find(stim_ent) != stimRegion.end()){
          stim = 30.0;
        } else {
            stim = 0.0;
        }
      }
      
      for (int gg = 0; gg < nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        for (int rr = 0; rr < nb_dofs; ++rr) {
            vecF[rr] += t_row_v_base * (t_dot_u + t_div_u - stim) * a; 
          ++t_row_v_base;
        }
        ++t_dot_u;
        ++t_div_u;
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
  boost::shared_ptr<PreviousData> dataU;
  boost::shared_ptr<PreviousData> dataV;
  VectorDouble vecF;
};

// Tangent operator
// //**********************************************
// 7. Tangent assembly for F_tautau excluding the essential boundary condition
template <int dim>
struct OpAssembleLhsTauTau : OpVolEle // A_TauTau_1
{
  OpAssembleLhsTauTau(std::string flux_field)
      : OpVolEle(flux_field, flux_field, OpVolEle::OPROWCOL) {
    sYmm = true;
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

      auto t_row_tau_base = row_data.getFTensor1N<3>();

      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double K_inv = 1. / D_tilde;
        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_tau_base = col_data.getFTensor1N<3>(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            mat(rr, cc) += (K_inv * t_row_tau_base(i) * t_col_tau_base(i)) * a;
            ++t_col_tau_base;
          }
          ++t_row_tau_base;
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

// 9. Assembly of tangent for F_tau_v excluding the essential bc
template <int dim>
struct OpAssembleLhsTauV : OpVolEle // E_TauV
{
  OpAssembleLhsTauV(std::string flux_field, std::string mass_field)
      : OpVolEle(flux_field, mass_field, OpVolEle::OPROWCOL) {
    sYmm = false;
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

      auto t_row_tau_grad = row_data.getFTensor2DiffN<3, 3>();
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;

        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_v_base = col_data.getFTensor0N(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            double div_row_base = t_row_tau_grad(0, 0) + t_row_tau_grad(1, 1) +
                                  t_row_tau_grad(2, 2);
            mat(rr, cc) +=  -(div_row_base * t_col_v_base) * a;
            ++t_col_v_base;
          }
          ++t_row_tau_base;
          ++t_row_tau_grad;
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

// 10. Assembly of tangent for F_v_tau
struct OpAssembleLhsVTau : OpVolEle // C_VTau
{
  OpAssembleLhsVTau(std::string mass_field, std::string flux_field)
      : OpVolEle(mass_field, flux_field, OpVolEle::OPROWCOL) {
    sYmm = false;
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
          auto t_col_tau_grad = col_data.getFTensor2DiffN<3, 3>(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            double div_col_base = t_col_tau_grad(0, 0) + t_col_tau_grad(1, 1) + t_col_tau_grad(2, 2);
            mat(rr, cc) += (t_row_v_base * div_col_base) * a;
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
struct OpAssembleLhsVV : OpVolEle // D
{
  OpAssembleLhsVV(std::string mass_field)
      : OpVolEle(mass_field, mass_field, OpVolEle::OPROWCOL) {
    sYmm = true;
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
            mat(rr, cc) += (ts_a * t_row_v_base * t_col_v_base) * a;

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

// struct OpError : public OpFaceEle {
//   typedef boost::function<double(const double, const double, const double)>
//       FVal;
//   typedef boost::function<FTensor::Tensor1<double, 3>(
//       const double, const double, const double)>
//       FGrad;
//   double &eRror;
//   OpError(FVal exact_value, FVal exact_lap, FGrad exact_grad,
//           boost::shared_ptr<PreviousData> &prev_data,
//           std::map<int, BlockData> &block_map, double &err)
//       : OpFaceEle("ERROR", OpFaceEle::OPROW), exactVal(exact_value),
//         exactLap(exact_lap), exactGrad(exact_grad), prevData(prev_data),
//         setOfBlock(block_map), eRror(err) {}
//   MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
//     MoFEMFunctionBegin;
//     const int nb_dofs = data.getFieldData().size();
//     // cout << "nb_error_dofs : " << nb_dofs << endl;
//     if (nb_dofs) {
//       auto find_block_data = [&]() {
//         EntityHandle fe_ent = getFEEntityHandle();
//         BlockData *block_raw_ptr = nullptr;
//         for (auto &m : setOfBlock) {
//           if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
//             block_raw_ptr = &m.second;
//             break;
//           }
//         }
//         return block_raw_ptr;
//       };

//       auto block_data_ptr = find_block_data();
//       if (!block_data_ptr)
//         SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");

//       auto &block_data = *block_data_ptr;

//       auto t_flux_value = getFTensor1FromMat<3>(prevData->flux_values);
//       // auto t_mass_dot = getFTensor0FromVec(prevData->mass_dots);
//       auto t_mass_value = getFTensor0FromVec(prevData->mass_values);
//       // auto t_flux_div = getFTensor0FromVec(prevData->flux_divs);
//       data.getFieldData().clear();
//       const double vol = getMeasure();
//       const int nb_integration_pts = getGaussPts().size2();
//       auto t_w = getFTensor0IntegrationWeight();
//       double dt;
//       CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
//       double ct = getFEMethod()->ts_t - dt;
//       auto t_coords = getFTensor1CoordsAtGaussPts();

//       FTensor::Tensor1<double, 3> t_exact_flux, t_flux_error;

//       for (int gg = 0; gg != nb_integration_pts; ++gg) {
//         const double a = vol * t_w;
//         double mass_exact = exactVal(t_coords(NX), t_coords(NY), ct);
//         // double flux_lap = - block_data.B0 * exactLap(t_coords(NX),
//         // t_coords(NY), ct);
//         t_exact_flux(i) =
//             -block_data.B0 * exactGrad(t_coords(NX), t_coords(NY), ct)(i);
//         t_flux_error(0) = t_flux_value(0) - t_exact_flux(0);
//         t_flux_error(1) = t_flux_value(1) - t_exact_flux(1);
//         t_flux_error(2) = t_flux_value(2) - t_exact_flux(2);
//         double local_error = pow(mass_exact - t_mass_value, 2) +
//                              t_flux_error(i) * t_flux_error(i);
//         // cout << "flux_div : " << t_flux_div << "   flux_exact : " <<
//         // flux_exact << endl;
//         data.getFieldData()[0] += a * local_error;
//         eRror += a * local_error;

//         ++t_w;
//         ++t_mass_value;
//         // ++t_flux_div;
//         ++t_flux_value;
//         // ++t_mass_dot;
//         ++t_coords;
//       }

//       data.getFieldDofs()[0]->getFieldData() = data.getFieldData()[0];
//     }
//     MoFEMFunctionReturn(0);
//   }

// private:
//   FVal exactVal;
//   FVal exactLap;
//   FGrad exactGrad;
//   boost::shared_ptr<PreviousData> prevData;
//   std::map<int, BlockData> setOfBlock;

//   FTensor::Number<0> NX;
//   FTensor::Number<1> NY;
// };



struct Monitor : public FEMethod {
  Monitor(MPI_Comm &comm, const int &rank, SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcVolumeOnRefinedMesh> &post_proc)
      : cOmm(comm), rAnk(rank), dM(dm), postProc(post_proc){};
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

  boost::shared_ptr<PostProcVolumeOnRefinedMesh> postProc;
  MPI_Comm cOmm;
  const int rAnk;
};

}; // namespace ReactionDiffusion

#endif //__ELECPHYSOPERATORS_HPP__