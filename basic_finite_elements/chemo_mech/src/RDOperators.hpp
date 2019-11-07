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



const double B = 0.0;
const double B_epsilon = 0.0;

const int save_every_nth_step = 4;
// const int order = 3; ///< approximation order
const double init_value = 0.5;
const double essen_value = 0;
const double natu_value = 0;
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

  PreviousData() {
    jac.resize(2, 2, false);
    inv_jac.resize(2, 2, false);
  }
};

struct BlockData {
  int block_id;
  double a11, a12, a13, a21, a22, a23, a31, a32, a33;

  double r1, r2, r3;

  Range block_ents;

  double B0; // species mobility

  BlockData()
      : a11(1), a12(2), a13(7), 
        a21(7), a22(1), a23(2), 
        a31(2), a32(7), a33(1),
        B0(2e-3), r1(1), r2(1), r3(1) {}
};

double compute_init_val(const double x, const double y, const double z) {
  return 0.0;
}

double compute_essen_bc(const double x, const double y, const double z) {
  return 0.0;
  }

  double compute_natu_bc(const double x, const double y, const double z){
    return 0.0;
  }

  

struct OpComputeSlowValue : public OpFaceEle {
    OpComputeSlowValue(std::string mass_field,
                       boost::shared_ptr<PreviousData> &data1,
                       boost::shared_ptr<PreviousData> &data2,
                       boost::shared_ptr<PreviousData> &data3,
                       std::map<int, BlockData> &block_map)
        : OpFaceEle(mass_field, OpFaceEle::OPROW), commonData1(data1),
          commonData2(data2), commonData3(data3), massField(mass_field),
          setOfBlock(block_map) {}
    MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
      MoFEMFunctionBegin;
      boost::shared_ptr<VectorDouble> slow_value_ptr1(
          commonData1, &commonData1->slow_values);
      boost::shared_ptr<VectorDouble> slow_value_ptr2(
          commonData2, &commonData2->slow_values);
      boost::shared_ptr<VectorDouble> slow_value_ptr3(
          commonData3, &commonData3->slow_values);

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
        auto find_block_data = [&]() {
          EntityHandle fe_ent = getFEEntityHandle();
          BlockData *block_raw_ptr = nullptr;
          for (auto &m : setOfBlock) {
            if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
              block_raw_ptr = &m.second;
              break;
            }
          }
          return block_raw_ptr;
        };

        auto block_data_ptr = find_block_data();
        if (!block_data_ptr)
          SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");

        auto &block_data = *block_data_ptr;

        const int nb_integration_pts = getGaussPts().size2();

        auto t_slow_values1 = getFTensor0FromVec(vec1);
        auto t_slow_values2 = getFTensor0FromVec(vec2);
        auto t_slow_values3 = getFTensor0FromVec(vec3);

        auto t_mass_values1 = getFTensor0FromVec(commonData1->mass_values);
        auto t_mass_values2 = getFTensor0FromVec(commonData2->mass_values);
        auto t_mass_values3 = getFTensor0FromVec(commonData3->mass_values);
        // cout << "r1 : " << block_data.r1 << endl;
        for (int gg = 0; gg != nb_integration_pts; ++gg) {
          t_slow_values1 = block_data.r1 * t_mass_values1 *
                           (1.0 - block_data.a11 * t_mass_values1 -
                            block_data.a12 * t_mass_values2 -
                            block_data.a13 * t_mass_values3);
          t_slow_values2 = block_data.r2 * t_mass_values2 *
                           (1.0 - block_data.a21 * t_mass_values1 -
                            block_data.a22 * t_mass_values2 -
                            block_data.a23 * t_mass_values3);

          t_slow_values3 = block_data.r3 * t_mass_values3 *
                           (1.0 - block_data.a31 * t_mass_values1 -
                            block_data.a32 * t_mass_values2 -
                            block_data.a33 * t_mass_values3);
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
    std::map<int, BlockData> setOfBlock;
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
  OpInitialMass(const std::string &mass_field, Range &inner_surface)
      : OpFaceEle(mass_field, OpFaceEle::OPROW), innerSurface(inner_surface) {
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
  typedef boost::function<double(const double, const double, const double)>
      FVal;
  typedef boost::function<FTensor::Tensor1<double, 3>(
      const double, const double, const double)>
      FGrad;
  OpAssembleSlowRhsV(std::string mass_field,
                     boost::shared_ptr<PreviousData> &common_data, 
                     FVal exact_value, 
                     FVal exact_dot, 
                     FVal exact_lap
                     )
      : OpFaceEle(mass_field, OpFaceEle::OPROW)
      , commonData(common_data)
      , exactValue(exact_value)
      , exactDot(exact_dot)
      , exactLap(exact_lap)
  {}

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
      
      const double ct = getFEMethod()->ts_t - 0.01;
      auto t_coords = getFTensor1CoordsAtGaussPts();
      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        
        double u_dot = exactDot(t_coords(NX), t_coords(NY), ct);
        double u_lap = exactLap(t_coords(NX), t_coords(NY), ct);

        // double f = u_dot - u_lap;
        // double f = 0;
        for (int rr = 0; rr != nb_dofs; ++rr) {
          auto t_col_v_base = data.getFTensor0N(gg, 0);
          vecF[rr] += a * t_slow_value * t_row_v_base;
          // vecF[rr] +=  a * f * t_row_v_base;
          for (int cc = 0; cc != nb_dofs; ++cc) {
            mat(rr, cc) += a * t_row_v_base * t_col_v_base;
            ++t_col_v_base;
          }
          ++t_row_v_base;
        }
        ++t_mass_value;
        ++t_slow_value;
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
  boost::shared_ptr<PreviousData> commonData;
  VectorDouble vecF;
  MatrixDouble mat;
  FVal exactValue;
  FVal exactDot;
  FVal exactLap;

  FTensor::Number<0> NX;
  FTensor::Number<1> NY;
  FTensor::Number<2> NZ;
};

// 5. RHS contribution of the natural boundary condition
struct OpAssembleNaturalBCRhsTau : OpBoundaryEle // R_tau_2
{
  OpAssembleNaturalBCRhsTau(std::string flux_field, Range &natural_bd_ents)
      : OpBoundaryEle(flux_field, OpBoundaryEle::OPROW),
        natural_bd_ents(natural_bd_ents) {}

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
            vecF[rr] += (t_tau_base(i) * t_normal(i) * natu_value) * a;
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
                        boost::shared_ptr<PreviousData> &data,
                        std::map<int, BlockData> &block_map)
      : OpFaceEle(flux_field, OpFaceEle::OPROW), commonData(data),
        setOfBlock(block_map) {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;

    const int nb_dofs = data.getIndices().size();
    if (nb_dofs) {
      auto find_block_data = [&]() {
        EntityHandle fe_ent = getFEEntityHandle();
        BlockData *block_raw_ptr = nullptr;
        for (auto &m : setOfBlock) {
          if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
            block_raw_ptr = &m.second;
            break;
          }
        }
        return block_raw_ptr;
      };

      auto block_data_ptr = find_block_data();
      if (!block_data_ptr)
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");
      auto &block_data = *block_data_ptr;

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

        const double K = B_epsilon + (block_data.B0 + B * t_mass_value);
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
  typedef boost::function<double(const double, const double, const double)>
      FVal;
  OpAssembleStiffRhsV(std::string flux_field,
                      boost::shared_ptr<PreviousData> &data,
                      std::map<int, BlockData> &block_map, 
                      FVal exact_value,
                      FVal exact_dot, 
                      FVal exact_lap)
      : OpFaceEle(flux_field, OpFaceEle::OPROW)
      , commonData(data) 
      , setOfBlock(block_map)
      , exactValue(exact_value)
      , exactDot(exact_dot)
      , exactLap(exact_lap)
      {}

  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getIndices().size();
    // cerr << "In StiffRhsV ..." << endl;
    if (nb_dofs) {
      auto find_block_data = [&]() {
        EntityHandle fe_ent = getFEEntityHandle();
        BlockData *block_raw_ptr = nullptr;
        for (auto &m : setOfBlock) {
          if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
            block_raw_ptr = &m.second;
            break;
          }
        }
        return block_raw_ptr;
      };

      auto block_data_ptr = find_block_data();
      if (!block_data_ptr)
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");
      auto &block_data = *block_data_ptr;


      vecF.resize(nb_dofs, false);
      vecF.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_mass_dot = getFTensor0FromVec(commonData->mass_dots);
      auto t_flux_div = getFTensor0FromVec(commonData->flux_divs);
      auto t_row_v_base = data.getFTensor0N();
      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      const double ct = getFEMethod()->ts_t;
      auto t_coords = getFTensor1CoordsAtGaussPts();
      for (int gg = 0; gg < nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        double u_dot = exactDot(t_coords(NX), t_coords(NY), ct);
        double u_lap = exactLap(t_coords(NX), t_coords(NY), ct);

        double f = u_dot - block_data.B0 * u_lap;
        for (int rr = 0; rr < nb_dofs; ++rr) {
          vecF[rr] += (t_row_v_base * (t_mass_dot + t_flux_div)) * a;
          ++t_row_v_base;
        }
        ++t_mass_dot;
        ++t_flux_div;
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
  boost::shared_ptr<PreviousData> commonData;
  VectorDouble vecF;
  std::map<int, BlockData> setOfBlock;

  FVal exactValue;
  FVal exactDot;
  FVal exactLap;

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
        setOfBlock(block_map), commonData(commonData) 
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
      auto find_block_data = [&]() {
        EntityHandle fe_ent = getFEEntityHandle();
        BlockData *block_raw_ptr = nullptr;
        for (auto &m : setOfBlock) {
          if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
            block_raw_ptr = &m.second;
            break;
          }
        }
        return block_raw_ptr;
      };

      auto block_data_ptr = find_block_data();
      if (!block_data_ptr)
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");
      auto &block_data = *block_data_ptr;

      mat.resize(nb_row_dofs, nb_col_dofs, false);
      mat.clear();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_mass_value = getFTensor0FromVec(commonData->mass_values);

      auto t_row_tau_base = row_data.getFTensor1N<3>();

      auto t_w = getFTensor0IntegrationWeight();
      const double vol = getMeasure();

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        const double K = B_epsilon + (block_data.B0 + B * t_mass_value);
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
        commonData(data), setOfBlock(block_map)
  {
    sYmm = false;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type, EntData &row_data,
                        EntData &col_data) {
    MoFEMFunctionBegin;
    const int nb_row_dofs = row_data.getIndices().size();
    const int nb_col_dofs = col_data.getIndices().size();

    if (nb_row_dofs && nb_col_dofs) {
      auto find_block_data = [&]() {
        EntityHandle fe_ent = getFEEntityHandle();
        BlockData *block_raw_ptr = nullptr;
        for (auto &m : setOfBlock) {
          if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
            block_raw_ptr = &m.second;
            break;
          }
        }
        return block_raw_ptr;
      };

      auto block_data_ptr = find_block_data();
      if (!block_data_ptr)
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");
      auto &block_data = *block_data_ptr;
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
        const double K = B_epsilon + (block_data.B0 + B * t_mass_value);
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
      : OpFaceEle(mass_field, flux_field, OpFaceEle::OPROWCOL) 
  {
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
  OpAssembleLhsVV(std::string mass_field)
      : OpFaceEle(mass_field, mass_field, OpFaceEle::OPROWCOL) 
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

struct OpError : public OpFaceEle {
  typedef boost::function<double(const double, const double, const double)>
      FVal;
  typedef boost::function<FTensor::Tensor1<double, 3>(
      const double, const double, const double)>
      FGrad;
  double &eRror;
  OpError(FVal exact_value, 
          FVal exact_lap, FGrad exact_grad,
          boost::shared_ptr<PreviousData> &prev_data, 
          std::map<int, BlockData> &block_map,
          double &err)
      : OpFaceEle("ERROR", OpFaceEle::OPROW)
      , exactVal(exact_value)
      , exactLap(exact_lap)
      , exactGrad(exact_grad)
      , prevData(prev_data)
      , setOfBlock(block_map)
      , eRror(err)
      {}
  MoFEMErrorCode doWork(int side, EntityType type, EntData &data) {
    MoFEMFunctionBegin;
    const int nb_dofs = data.getFieldData().size();
    // cout << "nb_error_dofs : " << nb_dofs << endl;
    if (nb_dofs) {
      auto find_block_data = [&]() {
          EntityHandle fe_ent = getFEEntityHandle();
          BlockData *block_raw_ptr = nullptr;
          for (auto &m : setOfBlock) {
            if (m.second.block_ents.find(fe_ent) != m.second.block_ents.end()) {
              block_raw_ptr = &m.second;
              break;
            }
          }
          return block_raw_ptr;
        };

      auto block_data_ptr = find_block_data();
      if (!block_data_ptr)
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY, "Block not found");

      auto &block_data = *block_data_ptr;



      auto t_flux_value = getFTensor1FromMat<3>(prevData->flux_values);
      // auto t_mass_dot = getFTensor0FromVec(prevData->mass_dots);
      auto t_mass_value = getFTensor0FromVec(prevData->mass_values);
      // auto t_flux_div = getFTensor0FromVec(prevData->flux_divs);
      data.getFieldData().clear();
      const double vol = getMeasure();
      const int nb_integration_pts = getGaussPts().size2();
      auto t_w = getFTensor0IntegrationWeight();
      double dt;
      CHKERR TSGetTimeStep(getFEMethod()->ts, &dt);
      double ct = getFEMethod()->ts_t - dt;
      auto t_coords = getFTensor1CoordsAtGaussPts();

      FTensor::Tensor1<double, 3> t_exact_flux, t_flux_error;

      for (int gg = 0; gg != nb_integration_pts; ++gg) {
        const double a = vol * t_w;
        double mass_exact =  exactVal(t_coords(NX), t_coords(NY), ct);
        // double flux_lap = - block_data.B0 * exactLap(t_coords(NX), t_coords(NY), ct);
        t_exact_flux(i) = - block_data.B0 * exactGrad(t_coords(NX), t_coords(NY), ct)(i);
        t_flux_error(0) = t_flux_value(0) - t_exact_flux(0);
        t_flux_error(1) = t_flux_value(1) - t_exact_flux(1);
        t_flux_error(2) = t_flux_value(2) - t_exact_flux(2);
        double local_error = pow(mass_exact - t_mass_value, 2) + t_flux_error(i) * t_flux_error(i); 
        // cout << "flux_div : " << t_flux_div << "   flux_exact : " << flux_exact << endl;
        data.getFieldData()[0] += a * local_error;
        eRror += a * local_error;

        ++t_w;
        ++t_mass_value;
        // ++t_flux_div;
        ++t_flux_value;
        // ++t_mass_dot;
        ++t_coords;
      }

      data.getFieldDofs()[0]->getFieldData() = data.getFieldData()[0];  
    }
      MoFEMFunctionReturn(0);
  }

  private:
    FVal exactVal;
    FVal exactLap;
    FGrad exactGrad;
    boost::shared_ptr<PreviousData> prevData;
    std::map<int, BlockData> setOfBlock;

    FTensor::Number<0> NX;
    FTensor::Number<1> NY;
};

// struct ExactMass : public OpFaceEle {
//   typedef boost::function<double(const double, const double, const double)>
//       FVal;

//   ExacMass(FVal exact_value)
//       : OpFaceEle("EXACT_M", OpFaceEle::OPROWCOL), exactVal(exact_value),
//         exactLap(exact_lap), exactGrad(exact_grad), prevData(prev_data),
//         setOfBlock(block_map), eRror(err) {}

// }

struct Monitor : public FEMethod {
  double &eRror;
  Monitor(MPI_Comm &comm, const int &rank, SmartPetscObj<DM> &dm,
          boost::shared_ptr<PostProcFaceOnRefinedMesh> &post_proc, double &err)
      : cOmm(comm)
      , rAnk(rank)
      , dM(dm)
      , postProc(post_proc)
      , eRror(err)
      {};
  MoFEMErrorCode preProcess() { return 0; }
  MoFEMErrorCode operator()() { return 0; }
  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    if (ts_step % save_every_nth_step == 0) {
      CHKERR DMoFEMLoopFiniteElements(dM, "dFE", postProc);
      CHKERR postProc->writeFile(
          "out_level_" + boost::lexical_cast<std::string>(ts_step) + ".h5m");
    }
  Vec error_per_proc;
  CHKERR VecCreateMPI(cOmm, 1, PETSC_DECIDE, &error_per_proc);
  auto get_global_error = [&]() {
  MoFEMFunctionBegin;  
  CHKERR VecSetValue(error_per_proc, rAnk, eRror, INSERT_VALUES);
  MoFEMFunctionReturn(0);
  };
  CHKERR get_global_error();
  CHKERR VecAssemblyBegin(error_per_proc);
  CHKERR VecAssemblyEnd(error_per_proc);
  double error_sum;
  CHKERR VecSum(error_per_proc, &error_sum);
  CHKERR PetscPrintf(PETSC_COMM_WORLD, "Error : %3.4e \n",
                    error_sum);
  eRror = 0;
  // PetscPrintf(PETSC_COMM_SELF, "global_error : %3.4e\n", eRror);
  // eRror = 0;
  MoFEMFunctionReturn(0);
  }

private:
  SmartPetscObj<DM> dM;

  boost::shared_ptr<PostProcFaceOnRefinedMesh> postProc;
  MPI_Comm cOmm;
  const int rAnk;
};

}; // namespace ReactionDiffusion

#endif //__RDOPERATORS_HPP__