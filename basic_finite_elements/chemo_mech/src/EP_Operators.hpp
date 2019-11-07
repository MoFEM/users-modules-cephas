#ifndef __EPOPERATORS_HPP__
#define __EPOPERATORS_HPP__

#include <stdlib.h>
#include <BasicFiniteElements.hpp>

namespace ElecPhys{
using VolEle = MoFEM::VolumeElementForcesAndSourcesCore;
using OpVolEle = VolEle::UserDataOperator;

using FaceEle = MoFEM::FaceElementForcesAndSourcesCore;
using OpFaceEle = FaceEle::UserDataOperator;

using EntData = DataForcesAndSourcesCore::EntData;

const double essen_value = 0.0;
const int save_every_nth_step = 4;

FTensor::Index<'i', 3> i;

double Heaviside(const double pt) {
  if (pt >= 0.0) {
    return 1;
  } else {
    return 0;
  }
}
struct Params {
  double U_0, U_u, theta_v, theta_w, theta_vMinus, theta_0, tau_v1Minus,
      tau_v2Minus, tau_vPlus, tau_w1Minus, tau_w2Minus, K_wMinus, U_wMinus,
      tau_wPlus, tau_fi, tau_01, tau_02, tau_s01, tau_s02, K_s0, U_s0, tau_s1,
      tau_s2, K_s, U_s, tau_si, tau_wInf, W_infStar;

  double D_tilde; // species mobility

Params()
      : U_0(0.0), U_u(1.58), theta_v(0.3), theta_w(0.015), theta_vMinus(0.015), theta_0(0.006), tau_v1Minus(60),
        tau_v2Minus(1150.0), tau_vPlus(1.4506), tau_w1Minus(70.0), tau_w2Minus(20.0), K_wMinus(65.0), U_wMinus(0.03), tau_wPlus(280),
        tau_fi(0.11), tau_01(6.0), tau_02(6.0), tau_s01(43.0), tau_s02(0.2), K_s0(2.0), U_s0(0.65), tau_s1(2.7342), tau_s2(3),
        K_s(2.0994), U_s(0.9087), tau_si(2.8723), tau_wInf(0.07), W_infStar(0.94), D_tilde() {}

double get_v_inf(const double u) const {
    if (u < theta_vMinus) {
      return 1;
    } else {
      return 0;
    }
  }


double get_w_inf(const double u) const {
    return (1 - Heaviside(u - theta_0)) * (1 - u / tau_wInf) +
           Heaviside(u - theta_0) * W_infStar;
  }

double get_tau_vMinus (const double u) const {
    return (1 - Heaviside(u - theta_vMinus)) * tau_v1Minus +
           Heaviside(u - theta_vMinus) * tau_v2Minus;
  }


double get_tau_wMinus(const double u) const {
    return tau_w1Minus +
           (tau_w2Minus - tau_w1Minus) *
               (1 + tanh(K_wMinus * (u - U_wMinus))) * 0.5;
  }

double get_tau_so(const double u) const {
    return tau_s01 + (tau_s02 - tau_s01) *
                                (1 + tanh(K_s0 * (u - U_s0))) *
                                0.5;
  }

double get_tau_s(const double u) const {
    return (1 - Heaviside(u - theta_w)) * tau_s1 +
           Heaviside(u - theta_w) * tau_s2;
  }

double get_tau_0(const double u) const {
    return (1 - Heaviside(u - theta_0)) * tau_s2 +
           Heaviside(u - theta_0) * tau_02;
  }

double get_J_fi(const double u, const double v, const double w,
                    const double s) const {
    return -v * Heaviside(u - theta_v) * (u - theta_v) *
           (U_u - u) / tau_fi;
  }

double get_J_so(const double u, const double v, const double w,
                    const double s) const {
    return (u - U_0) * (1 - Heaviside(u - theta_w)) / get_tau_0(u) +
           Heaviside(u - theta_w) / get_tau_so(u);
  }

double get_J_si(const double u, const double v, const double w,
                    const double s) const {
    return -Heaviside(u - theta_w) * w * s / tau_si;
  }

double get_rhs_u(const double u, const double v, const double w,
                    const double s) const {
    return -(get_J_fi(u, v, w, s) + get_J_so(u, v, w, s) + get_J_si(u, v, w, s));
  }

double get_rhs_v(const double u, const double v, const double w,
                    const double s) const {
    return (1 - Heaviside(u - theta_v)) * (get_v_inf(u) - v) /
               get_tau_vMinus(u) -
           Heaviside(u - theta_v) * v / tau_vPlus;
  }

double get_rhs_w(const double u, const double v, const double w,
                    const double s) const {
    return (1 - Heaviside(u - theta_w)) * (get_w_inf(u) - w) /
               get_tau_wMinus(u) -
           Heaviside(u - theta_w) * w / tau_wPlus;
  }

double get_rhs_s(const double u, const double v, const double w,
                    const double s) const {
    return ((1 + tanh(K_s * (u - U_s))) * 0.5 - s) / get_tau_s(u);
  }
};

struct PreviousData {
  MatrixDouble flux_values;
  VectorDouble flux_divs;

  VectorDouble mass_dots;
  VectorDouble mass_values;

  VectorDouble rhs_values;

  MatrixDouble jac;
  MatrixDouble inv_jac;

  PreviousData() {
    jac.resize(2, 2, false);
    inv_jac.resize(2, 2, false);
  }
};

Params params;





// struct OpComputeSlowValue : public OpVolEle {
//   OpComputeSlowValue(boost::shared_ptr<PreviousData> &datau,
//                      boost::shared_ptr<PreviousData> &datav,
//                      boost::shared_ptr<PreviousData> &dataw,
//                      boost::shared_ptr<PreviousData> &datas)
//       : OpVolEle("u", OpFaceEle::OPROW), commonDatau(datau), commonDatav(datav),
//         commonDataw(dataw), commonDatas(datas){}

// private:
//   std::string massField;
//   boost::shared_ptr<PreviousData> commonDatau;
//   boost::shared_ptr<PreviousData> commonDatav;
//   boost::shared_ptr<PreviousData> commonDataw;
//   boost::shared_ptr<PreviousData> commonDatas;
// };

struct OpEssentialBC : public OpFaceEle {
  OpEssentialBC(Range &essential_bd_ents)
      : OpFaceEle("f", OpFaceEle::OPROW),
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

struct OpInitialMass : public OpVolEle {
  OpInitialMass(std::vector<Range> &inner_surfaces, std::vector<double> &inits)
      : OpVolEle("u", OpVolEle::OPROW), innerSurfaces(inner_surfaces), iNits(inits) {}
  MatrixDouble nN;
  VectorDouble nF;
  Range innerSurface;
  std::vector<Range> &innerSurfaces;
  std::vector<double> &iNits;
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    int nb_dofs = data.getFieldData().size();
    if (nb_dofs) {
      unsigned int nstep = getFEMethod()->ts_step;
      if ( nstep < 5){
        CHKERR solveInitial(nstep, data);
      } 
    }
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode solveInitial(const int nstep, EntData &data) {
    MoFEMFunctionBegin;
    int nb_dofs = data.getFieldData().size();
    EntityHandle fe_ent = getFEEntityHandle();
    innerSurface = innerSurfaces[nstep-1];
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
      double init_value = iNits[nstep];
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        const double a = t_w * vol;
        for (int rr = 0; rr != nb_dofs; rr++) {
          auto t_col_mass = data.getFTensor0N(gg, 0);
          nF[rr] += a * init_value * t_row_mass;
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
      }
    }
    MoFEMFunctionReturn(0);
  }
};

struct OpAssembleSlowRhsV : OpVolEle // R_V
{
  OpAssembleSlowRhsV(boost::shared_ptr<PreviousData> &data_u,
                     boost::shared_ptr<PreviousData> &data_v,
                     boost::shared_ptr<PreviousData> &data_w,
                     boost::shared_ptr<PreviousData> &data_s)
      : OpVolEle("u", OpVolEle::OPROW), dataU(data_u)
      , dataV(data_v), dataW(data_w), dataS(data_s) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    // cerr << "In OpAssembleSlowRhsV...." << endl;
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      // cerr << "In SlowRhsV..." << endl;
      if (nb_dofs != static_cast<int>(data.getN().size2()))
        SETERRQ(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                "wrong number of dofs");

      unsigned int nstep = getFEMethod()->ts_step;
      vecF.resize(nb_dofs, false);
      mat.resize(nb_dofs, nb_dofs, false);
      vecF.clear();
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_val_u = getFTensor0FromVec(dataU->mass_values);
      auto t_val_v = getFTensor0FromVec(dataV->mass_values);
      auto t_val_w = getFTensor0FromVec(dataW->mass_values);
      auto t_val_s = getFTensor0FromVec(dataS->mass_values);

      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      auto t_coords = getFTensor1CoordsAtGaussPts();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;

        double rhs_u = params.get_rhs_u(t_val_u, t_val_v, t_val_w, t_val_s);
        double rhs_v = params.get_rhs_v(t_val_u, t_val_v, t_val_w, t_val_s);
        double rhs_w = params.get_rhs_w(t_val_u, t_val_v, t_val_w, t_val_s);
        double rhs_s = params.get_rhs_s(t_val_u, t_val_v, t_val_w, t_val_s);
        for (int rr = 0; rr != nb_dofs; ++rr) {
          auto t_col_v_base = data.getFTensor0N(gg, 0);
          if (((nstep + 3) % 4) == 0) {
            vecF[rr] += a * rhs_u * t_row_v_base;
          } else if (((nstep + 2) % 4) == 0) {
            vecF[rr] += a * rhs_v * t_row_v_base;
          } else if (((nstep + 1) % 4) == 0) {
            vecF[rr] += a * rhs_w * t_row_v_base;
          } else if ((nstep % 4) == 0) {
            vecF[rr] += a * rhs_s * t_row_v_base;
          }
          for (int cc = 0; cc != nb_dofs; ++cc) {
            mat(rr, cc) += a * t_row_v_base * t_col_v_base;
            ++t_col_v_base;
          }
          ++t_row_v_base;
        }
        ++t_val_u;
        ++t_val_v;
        ++t_val_w;
        ++t_val_s;
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
  boost::shared_ptr<PreviousData> dataW;
  boost::shared_ptr<PreviousData> dataS;
  VectorDouble vecF;
  MatrixDouble mat;

  FTensor::Number<0> NX;
  FTensor::Number<1> NY;
  FTensor::Number<2> NZ;
};

template <int dim>
struct OpAssembleStiffRhsTau : OpFaceEle //  F_tau_1
{
  OpAssembleStiffRhsTau(boost::shared_ptr<PreviousData> &data_u)
      : OpFaceEle("f", OpFaceEle::OPROW), dataU(data_u) {}

  // VectorDouble div_base;

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    unsigned int nstep = getFEMethod()->ts_step;

    const int nb_dofs = data.getIndices().size();
    if (nb_dofs && ((nstep + 3) % 4) == 0) {
      vecF.resize(nb_dofs, false);
      vecF.clear();
      // div_base.resize(data.getN().size2() / 3, 0);
      // if (div_base.size() != data.getIndices().size()) {
      //   SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
      //           "data inconsistency");
      // }

      const int nb_integration_pts = getGaussPts().size2();
      auto t_flux_u = getFTensor1FromMat<3>(dataU->flux_values);
      auto t_val_u = getFTensor0FromVec(dataU->mass_values);

      auto t_tau_base = data.getFTensor1N<3>();

      auto t_tau_grad = data.getFTensor2DiffN<3, 3>();

      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      for (int gg = 0; gg < nb_integration_pts; ++gg) {

        // double w = getGaussPts()(3, gg) * getVolume();
        // if (getHoGaussPtsDetJac().size() > 0) {
        //   w *= getHoGaussPtsDetJac()(gg);
        // }

        const double K_inv = 1. / params.D_tilde;
        const double a = vol * t_w;

        // CHKERR getDivergenceOfHDivBaseFunctions(side, type, data, gg, div_base);

        for (int rr = 0; rr < nb_dofs; ++rr) {
          double div_base = t_tau_grad(0, 0) + t_tau_grad(1, 1) + t_tau_grad(2, 2);
          vecF[rr] += (K_inv * t_tau_base(i) * t_flux_u(i) -
                       div_base * t_val_u) * a;
          ++t_tau_base;
          ++t_tau_grad;
        }
        ++t_flux_u;
        ++t_val_u;
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
  VectorDouble vecF;
};

template <int dim>
struct OpAssembleStiffRhsV : OpVolEle // F_V
{
  OpAssembleStiffRhsV(boost::shared_ptr<PreviousData> &data_u,
                      boost::shared_ptr<PreviousData> &data_v,
                      boost::shared_ptr<PreviousData> &data_w,
                      boost::shared_ptr<PreviousData> &data_s)
      : OpVolEle("f", OpVolEle::OPROW), dataU(data_u), dataV(data_v), dataW(data_w),
        dataS(data_s) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    unsigned int nstep = getFEMethod()->ts_step;
    const int nb_dofs = data.getIndices().size();
    // cerr << "In StiffRhsV ..." << endl;
    if (nb_dofs) {
     

      vecF.resize(nb_dofs, false);
      vecF.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_dot_u = getFTensor0FromVec(dataU->mass_dots);
      auto t_dot_v = getFTensor0FromVec(dataV->mass_dots);
      auto t_dot_w = getFTensor0FromVec(dataW->mass_dots);
      auto t_dot_s = getFTensor0FromVec(dataS->mass_dots);

      auto t_div_u = getFTensor0FromVec(dataU->flux_divs);
      auto t_div_v = getFTensor0FromVec(dataV->flux_divs);
      auto t_div_w = getFTensor0FromVec(dataW->flux_divs);
      auto t_div_s = getFTensor0FromVec(dataS->flux_divs);

      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      auto t_coords = getFTensor1CoordsAtGaussPts();
      for (int gg = 0; gg < nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        
        for (int rr = 0; rr < nb_dofs; ++rr) {

          if (((nstep + 3) % 4) == 0) {
            vecF[rr] += (t_row_v_base * (t_dot_u + t_div_u)) * a;
          } else if (((nstep + 2) % 4) == 0) {
            vecF[rr] += (t_row_v_base * (t_dot_v + t_div_v)) * a;
          } else if (((nstep + 1) % 4) == 0) {
            vecF[rr] += (t_row_v_base * (t_dot_w + t_div_w)) * a;
          } else if ((nstep % 4) == 0) {
            vecF[rr] += (t_row_v_base * (t_dot_s + t_div_s)) * a;
          }
          ++t_row_v_base;
        }
        ++t_dot_u;
        ++t_dot_v;
        ++t_dot_w;
        ++t_dot_s;

        ++t_div_u;
        ++t_div_v;
        ++t_div_w;
        ++t_div_s;
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
  boost::shared_ptr<PreviousData> dataW;
  boost::shared_ptr<PreviousData> dataS;
  VectorDouble vecF;

};

template <int dim>
struct OpAssembleLhsTauTau : OpVolEle // A_TauTau_1
{
  OpAssembleLhsTauTau()
      : OpVolEle("f", "f", OpVolEle::OPROWCOL){
    sYmm = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    unsigned int nstep = getFEMethod()->ts_step;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();
    
    bool is_u = ((nstep + 3) % 4) == 0;

    if (nb_row_dofs && nb_col_dofs && is_u) {
      
      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();

      auto t_row_tau_base = row_data.getFTensor1N<3>();

      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double K_inv = 1. / params.D_tilde;
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

template <int dim>
struct OpAssembleLhsTauV : OpVolEle // E_TauV
{
  OpAssembleLhsTauV(boost::shared_ptr<PreviousData> &data)
      : OpVolEle("f", "u", OpVolEle::OPROWCOL){
    sYmm = false;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    unsigned int nstep = getFEMethod()->ts_step;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    bool is_u = ((nstep + 3) % 4) == 0;

    if (nb_row_dofs && nb_col_dofs && is_u) {
      
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
            double div_row_base = t_row_tau_grad(0, 0) + t_row_tau_grad(1, 1) + t_row_tau_grad(2, 2);
            mat(rr, cc) += - (div_row_base * t_col_v_base) * a;
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

struct OpAssembleLhsVTau : OpVolEle // C_VTau
{
  OpAssembleLhsVTau()
      : OpVolEle("u", "f", OpVolEle::OPROWCOL) {
    sYmm = false;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    unsigned int nstep = getFEMethod()->ts_step;
    bool is_u = ((nstep + 3) % 4) == 0;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    if (nb_row_dofs && nb_col_dofs && is_u) {
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
            double div_col_base =
                t_col_tau_grad(0, 0) + t_col_tau_grad(1, 1) + t_col_tau_grad(2, 2);
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

struct OpAssembleLhsVV : OpVolEle // D
{
  OpAssembleLhsVV()
      : OpVolEle("u", "u", OpVolEle::OPROWCOL) {
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

struct Monitor : public FEMethod {
  Monitor(MPI_Comm &comm, const int &rank, SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcFaceOnRefinedMesh> &post_proc)
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

  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProc;
  MPI_Comm cOmm;
  const int rAnk;
};
};

#endif //__EPOPERATORS_HPP__