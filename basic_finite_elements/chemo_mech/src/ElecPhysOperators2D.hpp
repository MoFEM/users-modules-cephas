#ifndef __ELECPHYSOPERATORS2D_HPP__
#define __ELECPHYSOPERATORS2D_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

namespace ElectroPhysiology {

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

using PostProc = PostProcFaceOnRefinedMesh;

double factor = 1.0/ 12.9;

const double B = 0.0;
const double B_epsilon = 0.0;

const int save_every_nth_step = 16;

const double essen_value = 0;

FTensor::Index<'i', 3> i;

// problem parameters
const double alpha = 0.01;
const double gma = 0.002;
const double b = 0.15;
const double c = 8.00;
const double mu1 = 0.20;
const double mu2 = 0.30;

struct BlockData {
  int block_id;

  Range block_ents;

  double B0; // species mobility

  BlockData()
      : 
        B0(0.2) {}
};

BlockData block;

const double D_tilde = 1e-2;

struct PreviousData {
  MatrixDouble flux_values;
  VectorDouble flux_divs;

  VectorDouble mass_dots;
  VectorDouble mass_values;


  MatrixDouble jac;
  MatrixDouble inv_jac;

  PreviousData() {
    jac.resize(2, 2, false);
    inv_jac.resize(2, 2, false);
  }
};
struct OpEssentialBC : public OpBoundaryEle {
  OpEssentialBC(const std::string &flux_field, Range &essential_bd_ents)
      : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW),
        essential_bd_ents(essential_bd_ents) {}

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

        auto dir = getDirection();
        double len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);

        FTensor::Tensor1<double, 3> t_normal(-dir[1] / len, dir[0] / len,
                                             dir[2] / len);

        auto t_w = getFTensor0IntegrationWeight();
        const double vol = getMeasure();

        for (int gg = 0; gg < nb_gauss_pts; gg++) {
          const double a = t_w * vol;
          for (int rr = 0; rr != nb_dofs; rr++) {
            auto t_col_tau = data.getFTensor1N<3>(gg, 0);
            nF[rr] += a * essen_value * t_row_tau(i) * t_normal(i);
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
  OpInitialMass(const std::string &mass_field, Range &inner_surface, double &init_val)
      : OpFaceEle(mass_field, OpFaceEle::OPROW), innerSurface(inner_surface), initVal(init_val) {}
  MatrixDouble nN;
  VectorDouble nF;
  double &initVal;
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
          double r = initVal;
          for (int rr = 0; rr != nb_dofs; rr++) {
            auto t_col_mass = data.getFTensor0N(gg, 0);
            nF[rr] += a * r * t_row_mass;
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

struct OpSolveRecovery : public OpFaceEle {
  typedef boost::function<double(const double, const double, const double)>
      Method;
  OpSolveRecovery(const std::string &mass_field,
                  boost::shared_ptr<PreviousData> &data_u,
                  boost::shared_ptr<PreviousData> &data_v, Method runge_kutta4)
      : OpFaceEle(mass_field, OpFaceEle::OPROW), dataU(data_u), dataV(data_v),
        rungeKutta4(runge_kutta4) {}
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
struct OpAssembleSlowRhsV : OpFaceEle // R_V
{
  typedef boost::function<double(const double, const double)>
      FUVal;
  OpAssembleSlowRhsV(std::string mass_field,
                     boost::shared_ptr<PreviousData> &common_datau,
                     boost::shared_ptr<PreviousData> &common_datav, FUVal rhs_u)
      : OpFaceEle(mass_field, OpFaceEle::OPROW), commonDatau(common_datau),
        commonDatav(common_datav), rhsU(rhs_u) {}

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
      auto t_u_value = getFTensor0FromVec(commonDatau->mass_values);
      auto t_v_value = getFTensor0FromVec(commonDatav->mass_values);
      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      const double ct = getFEMethod()->ts_t - 0.01;
      auto t_coords = getFTensor1CoordsAtGaussPts();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;

        // double u_dot = exactDot(t_coords(NX), t_coords(NY), ct);
        // double u_lap = exactLap(t_coords(NX), t_coords(NY), ct);

        // double f = u_dot - u_lap;
        double f = t_u_value * (1.0 - t_u_value);
        for (int rr = 0; rr != nb_dofs; ++rr) {
          double rhs = rhsU(t_u_value, t_v_value);
          auto t_col_v_base = data.getFTensor0N(gg, 0);
          vecF[rr] += a * rhs * t_row_v_base;
          // vecF[rr] +=  a * f * t_row_v_base;
          for (int cc = 0; cc != nb_dofs; ++cc) {
            mat(rr, cc) += a * t_row_v_base * t_col_v_base;
            ++t_col_v_base;
          }
          ++t_row_v_base;
        }
        ++t_u_value;
        ++t_v_value;
        ++t_w;
        ++t_coords;
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
  boost::shared_ptr<PreviousData> commonDatau;
  boost::shared_ptr<PreviousData> commonDatav;
  VectorDouble vecF;
  MatrixDouble mat;
  FUVal rhsU;


  FTensor::Number<0> NX;
  FTensor::Number<1> NY;
  FTensor::Number<2> NZ;
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
struct OpAssembleStiffRhsTau : OpFaceEle //  F_tau_1
{
  OpAssembleStiffRhsTau(std::string flux_field,
                        boost::shared_ptr<PreviousData> &data,
                        std::map<int, BlockData> &block_map)
      : OpFaceEle(flux_field, OpFaceEle::OPROW), commonData(data),
        setOfBlock(block_map) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      // auto find_block_data = [&]() {
      //   EntityHandle fe_ent = getFEEntityHandle();
      //   BlockData *block_raw_ptr = nullptr;
      //   for (auto &m : setOfBlock) {
      //     if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
      //       block_raw_ptr = &m.second;
      //       break;
      //     }
      //   }
      //   return block_raw_ptr;
      // };

      // auto block_data_ptr = find_block_data();
      // if (!block_data_ptr)
      //   SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");
      // auto &block_data = *block_data_ptr;

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

        const double K = B_epsilon + (block.B0 + B * t_mass_value);
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
  std::map<int, BlockData> setOfBlock;
};
// 4. Assembly of F_v
template <int dim>
struct OpAssembleStiffRhsV : OpFaceEle // F_V
{
  typedef boost::function<double(const double, const double)>
      FUval;
  OpAssembleStiffRhsV(std::string flux_field,
                      boost::shared_ptr<PreviousData> &datau,
                      boost::shared_ptr<PreviousData> &datav, FUval rhs_u,
                      std::map<int, BlockData> &block_map, Range &stim_region)
      : OpFaceEle(flux_field, OpFaceEle::OPROW), commonDatau(datau),
        commonDatav(datav), setOfBlock(block_map), rhsU(rhs_u),
        stimRegion(stim_region) {}

  Range &stimRegion;
  FUval rhsU;
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    // cerr << "In StiffRhsV ..." << endl;
    if (nb_dofs) {
      // auto find_block_data = [&]() {
      //   EntityHandle fe_ent = getFEEntityHandle();
      //   BlockData *block_raw_ptr = nullptr;
      //   for (auto &m : setOfBlock) {
      //     if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
      //       block_raw_ptr = &m.second;
      //       break;
      //     }
      //   }
      //   return block_raw_ptr;
      // };

      // auto block_data_ptr = find_block_data();
      // if (!block_data_ptr)
      //   SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");
      // auto &block_data = *block_data_ptr;

      vecF.resize(nb_dofs, false);
      vecF.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_u_value = getFTensor0FromVec(commonDatau->mass_values);
      auto t_v_value = getFTensor0FromVec(commonDatav->mass_values);

      auto t_mass_dot = getFTensor0FromVec(commonDatau->mass_dots);
      auto t_flux_div = getFTensor0FromVec(commonDatau->flux_divs);
      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();
      auto t_coords = getFTensor1CoordsAtGaussPts();

      const double c_time = getFEMethod()->ts_t;

      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);

      double stim = 0.0;

      double T = 627.0;
      double duration = 5.0;

      if (T - dt < c_time && c_time <= T + duration) {
        EntityHandle stim_ent = getFEEntityHandle();
        if (stimRegion.find(stim_ent) != stimRegion.end()) {
          stim = 40.0;
        } else {
          stim = 0.0;
        }
      }
      for (int gg = 0; gg < nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        // double u_dot = exactDot(t_coords(NX), t_coords(NY), ct);
        // double u_lap = exactLap(t_coords(NX), t_coords(NY), ct);

        // double f = u_dot - block_data.B0 * u_lap;
        double rhsu = rhsU(t_u_value, t_v_value);
        for (int rr = 0; rr < nb_dofs; ++rr) {
          vecF[rr] += (t_row_v_base * (t_mass_dot + t_flux_div - 0*rhsu - factor * stim)) * a;
          ++t_row_v_base;
        }
        ++t_mass_dot;
        ++t_flux_div;
        ++t_u_value;
        ++t_v_value;
        ++t_w;
        ++t_coords;
      }
      CHKERR VecSetOption(getFEMethod()->ts_F, VEC_IGNORE_NEGATIVE_INDICES,
                          PETSC_TRUE);
      CHKERR VecSetValues(getFEMethod()->ts_F, data, &*vecF.begin(),
                          ADD_VALUES);
    }
    MoFEMFunctionReturn(0);
  }

private:
  boost::shared_ptr<PreviousData> commonDatau;
  boost::shared_ptr<PreviousData> commonDatav;
  VectorDouble vecF;
  std::map<int, BlockData> setOfBlock;

  // FVal exactValue;
  // FVal exactDot;
  // FVal exactLap;

  FTensor::Number<0> NX;
  FTensor::Number<1> NY;
};

// Tangent operator
// //**********************************************
// 7. Tangent assembly for F_tautau excluding the essential boundary condition
template <int dim>
struct OpAssembleLhsTauTau : OpFaceEle // A_TauTau_1
{
  OpAssembleLhsTauTau(std::string flux_field,
                      boost::shared_ptr<PreviousData> &commonData,
                      std::map<int, BlockData> &block_map)
      : OpFaceEle(flux_field, flux_field, OpFaceEle::OPROWCOL),
        setOfBlock(block_map), commonData(commonData) {
    sYmm = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    if (nb_row_dofs && nb_col_dofs) {
      // auto find_block_data = [&]() {
      //   EntityHandle fe_ent = getFEEntityHandle();
      //   BlockData *block_raw_ptr = nullptr;
      //   for (auto &m : setOfBlock) {
      //     if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
      //       block_raw_ptr = &m.second;
      //       break;
      //     }
      //   }
      //   return block_raw_ptr;
      // };

      // auto block_data_ptr = find_block_data();
      // if (!block_data_ptr)
      //   SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");
      // auto &block_data = *block_data_ptr;

      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_mass_value = getFTensor0FromVec(commonData->mass_values);

      auto t_row_tau_base = row_data.getFTensor1N<3>();

      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double K = B_epsilon + (block.B0 + B * t_mass_value);
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
  std::map<int, BlockData> setOfBlock;
};

// 9. Assembly of tangent for F_tau_v excluding the essential bc
template <int dim>
struct OpAssembleLhsTauV : OpFaceEle // E_TauV
{
  OpAssembleLhsTauV(std::string flux_field, std::string mass_field,
                    boost::shared_ptr<PreviousData> &data,
                    std::map<int, BlockData> &block_map)
      : OpFaceEle(flux_field, mass_field, OpFaceEle::OPROWCOL),
        commonData(data), setOfBlock(block_map) {
    sYmm = false;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    if (nb_row_dofs && nb_col_dofs) {
      // auto find_block_data = [&]() {
      //   EntityHandle fe_ent = getFEEntityHandle();
      //   BlockData *block_raw_ptr = nullptr;
      //   for (auto &m : setOfBlock) {
      //     if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
      //       block_raw_ptr = &m.second;
      //       break;
      //     }
      //   }
      //   return block_raw_ptr;
      // };

      // auto block_data_ptr = find_block_data();
      // if (!block_data_ptr)
      //   SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");
      // auto &block_data = *block_data_ptr;
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
        const double K = B_epsilon + (block.B0 + B * t_mass_value);
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
  std::map<int, BlockData> setOfBlock;
};

// 10. Assembly of tangent for F_v_tau
struct OpAssembleLhsVTau : OpFaceEle // C_VTau
{
  OpAssembleLhsVTau(std::string mass_field, std::string flux_field)
      : OpFaceEle(mass_field, flux_field, OpFaceEle::OPROWCOL) {
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
          auto t_col_tau_grad = col_data.getFTensor2DiffN<3, 2>(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            double div_col_base = t_col_tau_grad(0, 0) + t_col_tau_grad(1, 1);
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
struct OpAssembleLhsVV : OpFaceEle // D
{
  typedef boost::function<double(const double, const double)> FUval;
  OpAssembleLhsVV(std::string mass_field,
                  boost::shared_ptr<PreviousData> &datau,
                  boost::shared_ptr<PreviousData> &datav,
                  FUval Drhs_u)
      : OpFaceEle(mass_field, mass_field, OpFaceEle::OPROWCOL) 
      , DRhs_u(Drhs_u)
      , dataU(datau)
      , dataV(datav){
    sYmm = true;
  }

  boost::shared_ptr<PreviousData> &dataU;
  boost::shared_ptr<PreviousData> &dataV;

  FUval DRhs_u;

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
      auto t_u_value = getFTensor0FromVec(dataU->mass_values);
      auto t_v_value = getFTensor0FromVec(dataV->mass_values);

      auto t_w = getFTensor0IntegrationWeight();
      const double ts_a = getFEMethod()->ts_a;
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        double dfu = DRhs_u(t_u_value, t_v_value);
        for (int rr = 0; rr != nb_row_dofs; ++rr) {
          auto t_col_v_base = col_data.getFTensor0N(gg, 0);
          for (int cc = 0; cc != nb_col_dofs; ++cc) {
            mat(rr, cc) += ((ts_a - 0*dfu) * t_row_v_base * t_col_v_base) * a;

            ++t_col_v_base;
          }
          ++t_row_v_base;
          
        }
        ++t_w;
        ++t_u_value;
        ++t_v_value;
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
//           if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end())
//           {
//             block_raw_ptr = &m.second;
//             break;
//           }
//         }
//         return block_raw_ptr;
//       };

//       auto block_data_ptr = find_block_data();
//       if (!block_data_ptr)
//         SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not
//         found");

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
          boost::shared_ptr<PostProc> &post_proc)
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

  boost::shared_ptr<PostProc> postProc;
  MPI_Comm cOmm;
  const int rAnk;
};

}; // namespace ElectroPhysiology

#endif //__ELECPHYSOPERATORS_HPP__