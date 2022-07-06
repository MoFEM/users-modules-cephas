/** \file ConvectiveMassElement.cpp
 * \brief Operators and data structures for mass and convective mass element
 * \ingroup convective_mass_elem
 *
 */

/* Implementation of convective mass element
 *
 * This file is part of MoFEM.
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

#include <MoFEM.hpp>
using namespace MoFEM;

#include <Projection10NodeCoordsOnField.hpp>

#include <adolc/adolc.h>
#include <MethodForForceScaling.hpp>
#include <DirichletBC.hpp>
#include <MethodForForceScaling.hpp>
#include <ConvectiveMassElement.hpp>

#ifndef WITH_ADOL_C
#error "MoFEM need to be compiled with ADOL-C"
#endif

ConvectiveMassElement::MyVolumeFE::MyVolumeFE(MoFEM::Interface &m_field)
    : VolumeElementForcesAndSourcesCore(m_field), A(PETSC_NULL), F(PETSC_NULL) {
  meshPositionsFieldName = "NoNE";

  auto create_vec = [&]() {
    constexpr int ghosts[] = {0};
    if (mField.get_comm_rank() == 0) {
      return createSmartVectorMPI(mField.get_comm(), 1,1);
    } else {
      return createSmartVectorMPI(mField.get_comm(), 0, 1);
    }
  };

  V = create_vec();
}

int ConvectiveMassElement::MyVolumeFE::getRule(int order) { return 2 * order; };

MoFEMErrorCode ConvectiveMassElement::MyVolumeFE::preProcess() {
  MoFEMFunctionBeginHot;

  CHKERR VolumeElementForcesAndSourcesCore::preProcess();

  switch (ts_ctx) {
  case CTX_TSNONE:
    CHKERR VecZeroEntries(V);
    break;
  default:
    break;
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::MyVolumeFE::postProcess() {
  MoFEMFunctionBeginHot;

  CHKERR VolumeElementForcesAndSourcesCore::postProcess();

  const double *array;
  switch (ts_ctx) {
  case CTX_TSNONE:
    CHKERR VecAssemblyBegin(V);
    CHKERR VecAssemblyEnd(V);
    CHKERR VecSum(V, &eNergy);
    break;
  default:
    break;
  }

  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::ConvectiveMassElement(MoFEM::Interface &m_field,
                                             short int tag)
    : feMassRhs(m_field), feMassLhs(m_field), feMassAuxLhs(m_field),
      feVelRhs(m_field), feVelLhs(m_field), feTRhs(m_field), feTLhs(m_field),
      feEnergy(m_field), mField(m_field), tAg(tag) {}

ConvectiveMassElement::OpGetDataAtGaussPts::OpGetDataAtGaussPts(
    const std::string field_name,
    std::vector<VectorDouble> &values_at_gauss_pts,
    std::vector<MatrixDouble> &gardient_at_gauss_pts)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
      valuesAtGaussPts(values_at_gauss_pts),
      gradientAtGaussPts(gardient_at_gauss_pts), zeroAtType(MBVERTEX) {}

MoFEMErrorCode ConvectiveMassElement::OpGetDataAtGaussPts::doWork(
    int side, EntityType type, EntitiesFieldData::EntData &data) {
  MoFEMFunctionBeginHot;

  int nb_dofs = data.getFieldData().size();
  if (nb_dofs == 0) {
    MoFEMFunctionReturnHot(0);
  }
  int nb_gauss_pts = data.getN().size1();
  int nb_base_functions = data.getN().size2();

  // initialize
  // VectorDouble& values = data.getFieldData();
  valuesAtGaussPts.resize(nb_gauss_pts);
  gradientAtGaussPts.resize(nb_gauss_pts);
  for (int gg = 0; gg < nb_gauss_pts; gg++) {
    valuesAtGaussPts[gg].resize(3);
    gradientAtGaussPts[gg].resize(3, 3);
  }

  if (type == zeroAtType) {
    for (int gg = 0; gg < nb_gauss_pts; gg++) {
      valuesAtGaussPts[gg].clear();
      gradientAtGaussPts[gg].clear();
    }
  }

  auto base_function = data.getFTensor0N();
  auto diff_base_functions = data.getFTensor1DiffN<3>();
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;

  for (int gg = 0; gg != nb_gauss_pts; gg++) {
    auto field_data = data.getFTensor1FieldData<3>();
    FTensor::Tensor1<double *, 3> values(&valuesAtGaussPts[gg][0],
                                         &valuesAtGaussPts[gg][1],
                                         &valuesAtGaussPts[gg][2]);
    FTensor::Tensor2<double *, 3, 3> gradient(
        &gradientAtGaussPts[gg](0, 0), &gradientAtGaussPts[gg](0, 1),
        &gradientAtGaussPts[gg](0, 2), &gradientAtGaussPts[gg](1, 0),
        &gradientAtGaussPts[gg](1, 1), &gradientAtGaussPts[gg](1, 2),
        &gradientAtGaussPts[gg](2, 0), &gradientAtGaussPts[gg](2, 1),
        &gradientAtGaussPts[gg](2, 2));
    int bb = 0;
    for (; bb != nb_dofs / 3; bb++) {
      values(i) += base_function * field_data(i);
      gradient(i, j) += field_data(i) * diff_base_functions(j);
      ++diff_base_functions;
      ++base_function;
      ++field_data;
    }
    for (; bb != nb_base_functions; bb++) {
      ++diff_base_functions;
      ++base_function;
    }
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpGetCommonDataAtGaussPts::OpGetCommonDataAtGaussPts(
    const std::string field_name, CommonData &common_data)
    : OpGetDataAtGaussPts(field_name, common_data.dataAtGaussPts[field_name],
                          common_data.gradAtGaussPts[field_name]) {}

ConvectiveMassElement::OpMassJacobian::OpMassJacobian(
    const std::string field_name, BlockData &data, CommonData &common_data,
    boost::ptr_vector<MethodForForceScaling> &methods_op, int tag,
    bool jacobian)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
      dAta(data), commonData(common_data), tAg(tag), jAcobian(jacobian),
      lInear(commonData.lInear), fieldDisp(false), methodsOp(methods_op) {}

MoFEMErrorCode ConvectiveMassElement::OpMassJacobian::doWork(
    int row_side, EntityType row_type,
    EntitiesFieldData::EntData &row_data) {
  MoFEMFunctionBeginHot;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  // do it only once, no need to repeat this for edges,faces or tets
  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  int nb_dofs = row_data.getIndices().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  {

    if (a.size() != 3) {
      a.resize(3, false);
      dot_W.resize(3, false);
      a_res.resize(3, false);
      g.resize(3, 3, false);
      G.resize(3, 3, false);
      h.resize(3, 3, false);
      H.resize(3, 3, false);
      invH.resize(3, 3, false);
      F.resize(3, 3, false);
    }

    dot_W.clear();
    H.clear();
    invH.clear();
    for (int dd = 0; dd < 3; dd++) {
      H(dd, dd) = 1;
      invH(dd, dd) = 1;
    }

    int nb_gauss_pts = row_data.getN().size1();
    commonData.valMass.resize(nb_gauss_pts);
    commonData.jacMassRowPtr.resize(nb_gauss_pts);
    commonData.jacMass.resize(nb_gauss_pts);

    const std::vector<VectorDouble> &dot_spacial_vel =
        commonData.dataAtGaussPts["DOT_" + commonData.spatialVelocities];

    const std::vector<MatrixDouble> &spatial_positions_grad =
        commonData.gradAtGaussPts[commonData.spatialPositions];

    const std::vector<MatrixDouble> &spatial_velocities_grad =
        commonData.gradAtGaussPts[commonData.spatialVelocities];

    const std::vector<VectorDouble> &meshpos_vel =
        commonData.dataAtGaussPts["DOT_" + commonData.meshPositions];

    const std::vector<MatrixDouble> &mesh_positions_gradient =
        commonData.gradAtGaussPts[commonData.meshPositions];

    int nb_active_vars = 0;
    for (int gg = 0; gg < nb_gauss_pts; gg++) {

      if (gg == 0) {

        trace_on(tAg);

        for (int nn1 = 0; nn1 < 3; nn1++) { // 0
          // commonData.dataAtGaussPts["DOT_"+commonData.spatialVelocities]
          a[nn1] <<= dot_spacial_vel[gg][nn1];
          nb_active_vars++;
        }
        for (int nn1 = 0; nn1 < 3; nn1++) { // 3
          for (int nn2 = 0; nn2 < 3; nn2++) {
            // commonData.gradAtGaussPts[commonData.spatialPositions][gg]
            h(nn1, nn2) <<= spatial_positions_grad[gg](nn1, nn2);
            if (fieldDisp) {
              if (nn1 == nn2) {
                h(nn1, nn2) += 1;
              }
            }
            nb_active_vars++;
          }
        }
        if (commonData.dataAtGaussPts["DOT_" + commonData.meshPositions]
                .size() > 0) {
          for (int nn1 = 0; nn1 < 3; nn1++) { // 3+9=12
            for (int nn2 = 0; nn2 < 3; nn2++) {
              // commonData.gradAtGaussPts[commonData.spatialVelocities]
              g(nn1, nn2) <<= spatial_velocities_grad[gg](nn1, nn2);
              nb_active_vars++;
            }
          }
          for (int nn1 = 0; nn1 < 3; nn1++) { // 3+9+9=21
            // commonData.dataAtGaussPts["DOT_"+commonData.meshPositions]
            dot_W(nn1) <<= meshpos_vel[gg][nn1];
            nb_active_vars++;
          }
          for (int nn1 = 0; nn1 < 3; nn1++) { // 3+9+9+3=24
            for (int nn2 = 0; nn2 < 3; nn2++) {
              // commonData.gradAtGaussPts[commonData.meshPositions][gg]
              H(nn1, nn2) <<= mesh_positions_gradient[gg](nn1, nn2);
              nb_active_vars++;
            }
          }
        }

        auto a0 = dAta.a0;
        CHKERR MethodForForceScaling::applyScale(getFEMethod(), methodsOp, a0);

        auto t_a_res =
            FTensor::Tensor1<adouble *, 3>{&a_res[0], &a_res[1], &a_res[2]};
        auto t_a = FTensor::Tensor1<adouble *, 3>{&a[0], &a[1], &a[2]};
        auto t_a0 = FTensor::Tensor1<double *, 3>{&a0[0], &a0[1], &a0[2]};
        auto t_dotW =
            FTensor::Tensor1<adouble *, 3>{&dot_W[0], &dot_W[1], &dot_W[2]};
        auto t_g = getFTensor2FromArray3by3(g, FTensor::Number<0>(), 0);
        auto t_G = getFTensor2FromArray3by3(G, FTensor::Number<0>(), 0);
        auto t_invH = getFTensor2FromArray3by3(invH, FTensor::Number<0>(), 0);
        auto t_F = getFTensor2FromArray3by3(F, FTensor::Number<0>(), 0);
        auto t_h = getFTensor2FromArray3by3(h, FTensor::Number<0>(), 0);

        const double rho0 = dAta.rho0;

        adouble detH = determinantTensor3by3(H);
        CHKERR invertTensor3by3(H, detH, invH);

        t_G(i, j) = t_g(i, k) * t_invH(k, j);
        t_a_res(i) = t_a(i) - t_a0(i) + t_G(i, j) * t_dotW(j);

        //FIXME: there is error somewhere for nonlinear case
        // test dam example with -is_linear 0
        if (!lInear) {

          t_F(i,j) = t_h(i,k)*t_invH(k,j);
          t_a_res(i) *= rho0 * detH;
          t_a_res(i) *= determinantTensor3by3(t_F);

        } else {

          t_a_res(i) *= rho0 * detH;

        }

        // dependant
        VectorDouble &res = commonData.valMass[gg];
        res.resize(3);
        for (int rr = 0; rr < 3; rr++) {
          a_res[rr] >>= res[rr];
        }

        trace_off();
      }

      active.resize(nb_active_vars);
      int aa = 0;
      for (int nn1 = 0; nn1 < 3; nn1++) { // 0
        active[aa++] = dot_spacial_vel[gg][nn1];
      }
      for (int nn1 = 0; nn1 < 3; nn1++) { // 3
        for (int nn2 = 0; nn2 < 3; nn2++) {
          if (fieldDisp && nn1 == nn2) {
            active[aa++] = spatial_positions_grad[gg](nn1, nn2) + 1;
          } else {
            active[aa++] = spatial_positions_grad[gg](nn1, nn2);
          }
        }
      }
      if (commonData.dataAtGaussPts["DOT_" + commonData.meshPositions].size() >
          0) {
        for (int nn1 = 0; nn1 < 3; nn1++) { // 3+9=12
          for (int nn2 = 0; nn2 < 3; nn2++) {
            active[aa++] = spatial_velocities_grad[gg](nn1, nn2);
          }
        }
        for (int nn1 = 0; nn1 < 3; nn1++) { // 3+9+9=21
          active[aa++] = meshpos_vel[gg][nn1];
        }
        for (int nn1 = 0; nn1 < 3; nn1++) { // 3+9+9+3=24
          for (int nn2 = 0; nn2 < 3; nn2++) {
            active[aa++] = mesh_positions_gradient[gg](nn1, nn2);
          }
        }
      }

      if (!jAcobian) {
        VectorDouble &res = commonData.valMass[gg];
        if (gg > 0) {
          res.resize(3);
          int r;
          r = ::function(tAg, 3, nb_active_vars, &active[0], &res[0]);
          if (r != 3) { // function is locally analytic
            SETERRQ1(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                     "ADOL-C function evaluation with error r = %d", r);
          }
        }
        double val = getVolume() * getGaussPts()(3, gg);
        res *= val;
        // cout << "my res " << res << endl;
      } else {
        commonData.jacMassRowPtr[gg].resize(3);
        commonData.jacMass[gg].resize(3, nb_active_vars);
        for (int nn1 = 0; nn1 < 3; nn1++) {
          (commonData.jacMassRowPtr[gg])[nn1] =
              &(commonData.jacMass[gg](nn1, 0));
        }
        int r;
        r = jacobian(tAg, 3, nb_active_vars, &active[0],
                     &(commonData.jacMassRowPtr[gg])[0]);
        if (r != 3) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                  "ADOL-C function evaluation with error");
        }
        double val = getVolume() * getGaussPts()(3, gg);
        commonData.jacMass[gg] *= val;
      }
    }
  }

  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpMassRhs::OpMassRhs(const std::string field_name,
                                            BlockData &data,
                                            CommonData &common_data)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
      dAta(data), commonData(common_data) {}

MoFEMErrorCode ConvectiveMassElement::OpMassRhs::doWork(
    int row_side, EntityType row_type,
    EntitiesFieldData::EntData &row_data) {
  MoFEMFunctionBeginHot;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }
  if (row_data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  int nb_dofs = row_data.getIndices().size();

  auto base = row_data.getFTensor0N();
  int nb_base_functions = row_data.getN().size2();

  {

    nf.resize(nb_dofs);
    nf.clear();

    FTensor::Index<'i', 3> i;

    for (unsigned int gg = 0; gg < row_data.getN().size1(); gg++) {
      FTensor::Tensor1<double *, 3> t_nf(&nf[0], &nf[1], &nf[2], 3);
      FTensor::Tensor1<double *, 3> res(&commonData.valMass[gg][0],
                                        &commonData.valMass[gg][1],
                                        &commonData.valMass[gg][2]);
      int dd = 0;
      for (; dd < nb_dofs / 3; dd++) {
        t_nf(i) += base * res(i);
        ++base;
        ++t_nf;
      }
      for (; dd != nb_base_functions; dd++) {
        ++base;
      }
    }

    if ((unsigned int)nb_dofs > 3 * row_data.getN().size2()) {
      SETERRQ(PETSC_COMM_SELF, 1, "data inconsistency");
    }
    CHKERR VecSetValues(getFEMethod()->ts_F, nb_dofs, &row_data.getIndices()[0],
                        &nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpMassLhs_dM_dv::OpMassLhs_dM_dv(
    const std::string vel_field, const std::string field_name, BlockData &data,
    CommonData &common_data, Range *forcesonlyonentities_ptr)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          vel_field, field_name,
          ForcesAndSourcesCore::UserDataOperator::OPROWCOL),
      dAta(data), commonData(common_data) {
  sYmm = false;
  if (forcesonlyonentities_ptr != NULL) {
    forcesOnlyOnEntities = *forcesonlyonentities_ptr;
  }
}

MoFEMErrorCode ConvectiveMassElement::OpMassLhs_dM_dv::getJac(
    EntitiesFieldData::EntData &col_data, int gg) {
  MoFEMFunctionBeginHot;
  int nb_col = col_data.getIndices().size();
  jac.clear();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Tensor2<double *, 3, 3> t_jac(&jac(0, 0), &jac(0, 1), &jac(0, 2),
                                         &jac(1, 0), &jac(1, 1), &jac(1, 2),
                                         &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
  FTensor::Tensor2<double *, 3, 3> t_mass1(
      &commonData.jacMass[gg](0, 0), &commonData.jacMass[gg](0, 1),
      &commonData.jacMass[gg](0, 2), &commonData.jacMass[gg](1, 0),
      &commonData.jacMass[gg](1, 1), &commonData.jacMass[gg](1, 2),
      &commonData.jacMass[gg](2, 0), &commonData.jacMass[gg](2, 1),
      &commonData.jacMass[gg](2, 2));
  double *base_ptr = const_cast<double *>(&col_data.getN(gg)[0]);
  FTensor::Tensor0<double *> base(base_ptr, 1);
  if (commonData.dataAtGaussPts["DOT_" + commonData.meshPositions].size() ==
      0) {
    for (int dd = 0; dd < nb_col / 3; dd++) {
      t_jac(i, j) += t_mass1(i, j) * base * getFEMethod()->ts_a;
      ++base;
      ++t_jac;
    }
  } else {
    const int s = 3 + 9;
    FTensor::Tensor3<double *, 3, 3, 3> t_mass3(
        // T* d000, T* d001, T* d002,
        // T* d010, T* d011, T* d012,
        // T* d020, T* d021, T* d022,
        // T* d100, T* d101, T* d102,
        // T* d110, T* d111, T* d112,
        // T* d120, T* d121, T* d122,
        // T* d200, T* d201, T* d202,
        // T* d210, T* d211, T* d212,
        // T* d220, T* d221, T* d222,
        &commonData.jacMass[gg](0, s + 0), &commonData.jacMass[gg](0, s + 1),
        &commonData.jacMass[gg](0, s + 2), &commonData.jacMass[gg](0, s + 3),
        &commonData.jacMass[gg](0, s + 4), &commonData.jacMass[gg](0, s + 5),
        &commonData.jacMass[gg](0, s + 6), &commonData.jacMass[gg](0, s + 7),
        &commonData.jacMass[gg](0, s + 8), &commonData.jacMass[gg](1, s + 0),
        &commonData.jacMass[gg](1, s + 1), &commonData.jacMass[gg](1, s + 2),
        &commonData.jacMass[gg](1, s + 3), &commonData.jacMass[gg](1, s + 4),
        &commonData.jacMass[gg](1, s + 5), &commonData.jacMass[gg](1, s + 6),
        &commonData.jacMass[gg](1, s + 7), &commonData.jacMass[gg](1, s + 8),
        &commonData.jacMass[gg](2, s + 0), &commonData.jacMass[gg](2, s + 1),
        &commonData.jacMass[gg](2, s + 2), &commonData.jacMass[gg](2, s + 3),
        &commonData.jacMass[gg](2, s + 4), &commonData.jacMass[gg](2, s + 5),
        &commonData.jacMass[gg](2, s + 6), &commonData.jacMass[gg](2, s + 7),
        &commonData.jacMass[gg](2, s + 8));

    double *diff_ptr =
        const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
    FTensor::Tensor1<double *, 3> diff(diff_ptr, &diff_ptr[1], &diff_ptr[2], 3);
    for (int dd = 0; dd < nb_col / 3; dd++) {
      t_jac(i, j) += t_mass1(i, j) * base * getFEMethod()->ts_a;
      t_jac(i, j) += t_mass3(i, j, k) * diff(k);
      ++base;
      ++diff;
      ++t_jac;
    }
  }
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::OpMassLhs_dM_dv::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBeginHot;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  int nb_row = row_data.getIndices().size();
  int nb_col = col_data.getIndices().size();
  if (nb_row == 0)
    MoFEMFunctionReturnHot(0);
  if (nb_col == 0)
    MoFEMFunctionReturnHot(0);

  auto base = row_data.getFTensor0N();
  int nb_base_functions = row_data.getN().size2();

  {

    k.resize(nb_row, nb_col);
    k.clear();
    jac.resize(3, nb_col);

    for (unsigned int gg = 0; gg < row_data.getN().size1(); gg++) {

      try {
        CHKERR getJac(col_data, gg);
      } catch (const std::exception &ex) {
        std::ostringstream ss;
        ss << "throw in method: " << ex.what() << std::endl;
        SETERRQ(PETSC_COMM_SELF, 1, ss.str().c_str());
      }

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;

      {
        int dd1 = 0;
        // integrate element stiffness matrix
        for (; dd1 < nb_row / 3; dd1++) {
          FTensor::Tensor2<double *, 3, 3> t_jac(
              &jac(0, 0), &jac(0, 1), &jac(0, 2), &jac(1, 0), &jac(1, 1),
              &jac(1, 2), &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
          for (int dd2 = 0; dd2 < nb_col / 3; dd2++) {
            FTensor::Tensor2<double *, 3, 3> t_k(
                &k(3 * dd1 + 0, 3 * dd2 + 0), &k(3 * dd1 + 0, 3 * dd2 + 1),
                &k(3 * dd1 + 0, 3 * dd2 + 2), &k(3 * dd1 + 1, 3 * dd2 + 0),
                &k(3 * dd1 + 1, 3 * dd2 + 1), &k(3 * dd1 + 1, 3 * dd2 + 2),
                &k(3 * dd1 + 2, 3 * dd2 + 0), &k(3 * dd1 + 2, 3 * dd2 + 1),
                &k(3 * dd1 + 2, 3 * dd2 + 2));
            t_k(i, j) += base * t_jac(i, j);
            ++t_jac;
          }
          ++base;
          // for(int rr1 = 0;rr1<3;rr1++) {
          //   for(int dd2 = 0;dd2<nb_col;dd2++) {
          //     k(3*dd1+rr1,dd2) += row_data.getN()(gg,dd1)*jac(rr1,dd2);
          //   }
          // }
        }
        for (; dd1 != nb_base_functions; dd1++) {
          ++base;
        }
      }
    }

    if (!forcesOnlyOnEntities.empty()) {
      VectorInt indices = row_data.getIndices();
      VectorDofs &dofs = row_data.getFieldDofs();
      VectorDofs::iterator dit = dofs.begin();
      for (int ii = 0; dit != dofs.end(); dit++, ii++) {
        if (forcesOnlyOnEntities.find((*dit)->getEnt()) ==
            forcesOnlyOnEntities.end()) {
          indices[ii] = -1;
        }
      }
      CHKERR MatSetValues(getFEMethod()->ts_B, nb_row, &indices[0], nb_col,
                          &col_data.getIndices()[0], &k(0, 0), ADD_VALUES);
    } else {
      CHKERR MatSetValues(getFEMethod()->ts_B, nb_row,
                          &row_data.getIndices()[0], nb_col,
                          &col_data.getIndices()[0], &k(0, 0), ADD_VALUES);
    }
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpMassLhs_dM_dx::OpMassLhs_dM_dx(
    const std::string field_name, const std::string col_field, BlockData &data,
    CommonData &common_data)
    : OpMassLhs_dM_dv(field_name, col_field, data, common_data) {}

MoFEMErrorCode ConvectiveMassElement::OpMassLhs_dM_dx::getJac(
    EntitiesFieldData::EntData &col_data, int gg) {
  MoFEMFunctionBeginHot;
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  int nb_col = col_data.getIndices().size();
  jac.clear();
  FTensor::Tensor2<double *, 3, 3> t_jac(&jac(0, 0), &jac(0, 1), &jac(0, 2),
                                         &jac(1, 0), &jac(1, 1), &jac(1, 2),
                                         &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
  const int s = 3;
  FTensor::Tensor3<double *, 3, 3, 3> t_mass3(
      // T* d000, T* d001, T* d002,
      // T* d010, T* d011, T* d012,
      // T* d020, T* d021, T* d022,
      // T* d100, T* d101, T* d102,
      // T* d110, T* d111, T* d112,
      // T* d120, T* d121, T* d122,
      // T* d200, T* d201, T* d202,
      // T* d210, T* d211, T* d212,
      // T* d220, T* d221, T* d222,
      &commonData.jacMass[gg](0, s + 0), &commonData.jacMass[gg](0, s + 1),
      &commonData.jacMass[gg](0, s + 2), &commonData.jacMass[gg](0, s + 3),
      &commonData.jacMass[gg](0, s + 4), &commonData.jacMass[gg](0, s + 5),
      &commonData.jacMass[gg](0, s + 6), &commonData.jacMass[gg](0, s + 7),
      &commonData.jacMass[gg](0, s + 8), &commonData.jacMass[gg](1, s + 0),
      &commonData.jacMass[gg](1, s + 1), &commonData.jacMass[gg](1, s + 2),
      &commonData.jacMass[gg](1, s + 3), &commonData.jacMass[gg](1, s + 4),
      &commonData.jacMass[gg](1, s + 5), &commonData.jacMass[gg](1, s + 6),
      &commonData.jacMass[gg](1, s + 7), &commonData.jacMass[gg](1, s + 8),
      &commonData.jacMass[gg](2, s + 0), &commonData.jacMass[gg](2, s + 1),
      &commonData.jacMass[gg](2, s + 2), &commonData.jacMass[gg](2, s + 3),
      &commonData.jacMass[gg](2, s + 4), &commonData.jacMass[gg](2, s + 5),
      &commonData.jacMass[gg](2, s + 6), &commonData.jacMass[gg](2, s + 7),
      &commonData.jacMass[gg](2, s + 8));
  double *diff_ptr =
      const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
  FTensor::Tensor1<double *, 3> diff(diff_ptr, &diff_ptr[1], &diff_ptr[2], 3);
  for (int dd = 0; dd < nb_col / 3; dd++) {
    t_jac(i, j) += t_mass3(i, j, k) * diff(k);
    ++diff;
    ++t_jac;
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpMassLhs_dM_dX::OpMassLhs_dM_dX(
    const std::string field_name, const std::string col_field, BlockData &data,
    CommonData &common_data)
    : OpMassLhs_dM_dv(field_name, col_field, data, common_data) {}

MoFEMErrorCode ConvectiveMassElement::OpMassLhs_dM_dX::getJac(
    EntitiesFieldData::EntData &col_data, int gg) {
  MoFEMFunctionBeginHot;
  int nb_col = col_data.getIndices().size();
  jac.clear();
  double *base_ptr = const_cast<double *>(&col_data.getN(gg)[0]);
  FTensor::Tensor0<double *> base(base_ptr, 1);
  double *diff_ptr =
      const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
  FTensor::Tensor1<double *, 3> diff(diff_ptr, &diff_ptr[1], &diff_ptr[2], 3);
  FTensor::Tensor2<double *, 3, 3> t_jac(&jac(0, 0), &jac(0, 1), &jac(0, 2),
                                         &jac(1, 0), &jac(1, 1), &jac(1, 2),
                                         &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
  const int u = 3 + 9 + 9;
  FTensor::Tensor2<double *, 3, 3> t_mass1(
      &commonData.jacMass[gg](0, u + 0), &commonData.jacMass[gg](0, u + 1),
      &commonData.jacMass[gg](0, u + 2), &commonData.jacMass[gg](1, u + 0),
      &commonData.jacMass[gg](1, u + 1), &commonData.jacMass[gg](1, u + 2),
      &commonData.jacMass[gg](2, u + 0), &commonData.jacMass[gg](2, u + 1),
      &commonData.jacMass[gg](2, u + 2));
  const int s = 3 + 9 + 9 + 3;
  FTensor::Tensor3<double *, 3, 3, 3> t_mass3(
      // T* d000, T* d001, T* d002,
      // T* d010, T* d011, T* d012,
      // T* d020, T* d021, T* d022,
      // T* d100, T* d101, T* d102,
      // T* d110, T* d111, T* d112,
      // T* d120, T* d121, T* d122,
      // T* d200, T* d201, T* d202,
      // T* d210, T* d211, T* d212,
      // T* d220, T* d221, T* d222,
      &commonData.jacMass[gg](0, s + 0), &commonData.jacMass[gg](0, s + 1),
      &commonData.jacMass[gg](0, s + 2), &commonData.jacMass[gg](0, s + 3),
      &commonData.jacMass[gg](0, s + 4), &commonData.jacMass[gg](0, s + 5),
      &commonData.jacMass[gg](0, s + 6), &commonData.jacMass[gg](0, s + 7),
      &commonData.jacMass[gg](0, s + 8), &commonData.jacMass[gg](1, s + 0),
      &commonData.jacMass[gg](1, s + 1), &commonData.jacMass[gg](1, s + 2),
      &commonData.jacMass[gg](1, s + 3), &commonData.jacMass[gg](1, s + 4),
      &commonData.jacMass[gg](1, s + 5), &commonData.jacMass[gg](1, s + 6),
      &commonData.jacMass[gg](1, s + 7), &commonData.jacMass[gg](1, s + 8),
      &commonData.jacMass[gg](2, s + 0), &commonData.jacMass[gg](2, s + 1),
      &commonData.jacMass[gg](2, s + 2), &commonData.jacMass[gg](2, s + 3),
      &commonData.jacMass[gg](2, s + 4), &commonData.jacMass[gg](2, s + 5),
      &commonData.jacMass[gg](2, s + 6), &commonData.jacMass[gg](2, s + 7),
      &commonData.jacMass[gg](2, s + 8));
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  for (int dd = 0; dd < nb_col / 3; dd++) {
    t_jac(i, j) += t_mass1(i, j) * base * getFEMethod()->ts_a;
    t_jac(i, j) += t_mass3(i, j, k) * diff(k);
    ++base_ptr;
    ++diff_ptr;
    ++t_jac;
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpEnergy::OpEnergy(const std::string field_name,
                                          BlockData &data,
                                          CommonData &common_data,
                                          SmartPetscObj<Vec> v)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
      dAta(data), commonData(common_data), V(v, true),
      lInear(commonData.lInear) {}

MoFEMErrorCode ConvectiveMassElement::OpEnergy::doWork(
    int row_side, EntityType row_type,
    EntitiesFieldData::EntData &row_data) {
  MoFEMFunctionBeginHot;

  if (row_type != MBVERTEX) {
    MoFEMFunctionReturnHot(0);
  }
  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  {
    double energy = 0;
    for (unsigned int gg = 0; gg < row_data.getN().size1(); gg++) {
      double val = getVolume() * getGaussPts()(3, gg);
      double rho0 = dAta.rho0;
      double rho;
      if (lInear) {
        rho = rho0;
      } else {
        h.resize(3, 3);
        noalias(h) =
            (commonData.gradAtGaussPts[commonData.spatialPositions][gg]);
        if (commonData.dataAtGaussPts["DOT_" + commonData.meshPositions]
                .size() > 0) {
          H.resize(3, 3);
          noalias(H) =
              (commonData.gradAtGaussPts[commonData.meshPositions][gg]);
          auto detH = determinantTensor3by3(H);
          invH.resize(3, 3);
          CHKERR invertTensor3by3(H, detH, invH);
          F.resize(3, 3);
          noalias(F) = prod(h, invH);
        } else {
          F.resize(3, 3);
          noalias(F) = h;
        }
        double detF = determinantTensor3by3(F);
        rho = detF * rho0;
      }
      v.resize(3);
      noalias(v) = commonData.dataAtGaussPts[commonData.spatialVelocities][gg];
      energy += 0.5 * (rho * val) * inner_prod(v, v);
    }
    CHKERR VecSetValue(V, 0, energy, ADD_VALUES);
  }

  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpVelocityJacobian::OpVelocityJacobian(
    const std::string field_name, BlockData &data, CommonData &common_data,
    int tag, bool jacobian)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
      dAta(data), commonData(common_data), tAg(tag), jAcobian(jacobian),
      fieldDisp(false) {}

MoFEMErrorCode ConvectiveMassElement::OpVelocityJacobian::doWork(
    int row_side, EntityType row_type,
    EntitiesFieldData::EntData &row_data) {
  MoFEMFunctionBeginHot;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  // do it only once, no need to repeat this for edges,faces or tets
  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  int nb_dofs = row_data.getIndices().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  {

    v.resize(3);
    dot_w.resize(3);
    h.resize(3, 3);
    h.clear();
    F.resize(3, 3);
    dot_W.resize(3);
    dot_W.clear();
    H.resize(3, 3);
    H.clear();
    invH.resize(3, 3);
    invH.clear();
    dot_u.resize(3);
    for (int dd = 0; dd < 3; dd++) {
      H(dd, dd) = 1;
      invH(dd, dd) = 1;
    }

    a_res.resize(3);
    int nb_gauss_pts = row_data.getN().size1();
    commonData.valVel.resize(nb_gauss_pts);
    commonData.jacVelRowPtr.resize(nb_gauss_pts);
    commonData.jacVel.resize(nb_gauss_pts);

    int nb_active_vars = 0;
    for (int gg = 0; gg < nb_gauss_pts; gg++) {

      if (gg == 0) {

        trace_on(tAg);

        for (int nn1 = 0; nn1 < 3; nn1++) { // 0
          v[nn1] <<=
              commonData.dataAtGaussPts[commonData.spatialVelocities][gg][nn1];
          nb_active_vars++;
        }
        for (int nn1 = 0; nn1 < 3; nn1++) { // 3
          dot_w[nn1] <<=
              commonData.dataAtGaussPts["DOT_" + commonData.spatialPositions]
                                       [gg][nn1];
          nb_active_vars++;
        }
        if (commonData.dataAtGaussPts["DOT_" + commonData.meshPositions]
                .size() > 0) {
          for (int nn1 = 0; nn1 < 3; nn1++) { // 3+3 = 6
            for (int nn2 = 0; nn2 < 3; nn2++) {
              h(nn1, nn2) <<=
                  commonData.gradAtGaussPts[commonData.spatialPositions][gg](
                      nn1, nn2);
              if (fieldDisp) {
                if (nn1 == nn2) {
                  h(nn1, nn2) += 1;
                }
              }
              nb_active_vars++;
            }
          }
          for (int nn1 = 0; nn1 < 3; nn1++) { // 3+3+9
            dot_W[nn1] <<=
                commonData
                    .dataAtGaussPts["DOT_" + commonData.meshPositions][gg][nn1];
            nb_active_vars++;
          }
        }
        if (commonData.gradAtGaussPts[commonData.meshPositions].size() > 0) {
          for (int nn1 = 0; nn1 < 3; nn1++) { // 3+3+9+3
            for (int nn2 = 0; nn2 < 3; nn2++) {
              H(nn1, nn2) <<=
                  commonData.gradAtGaussPts[commonData.meshPositions][gg](nn1,
                                                                          nn2);
              nb_active_vars++;
            }
          }
        }
        detH = 1;

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;

        auto t_F = getFTensor2FromArray3by3(F, FTensor::Number<0>(), 0);
        auto t_h = getFTensor2FromArray3by3(h, FTensor::Number<0>(), 0);
        auto t_H = getFTensor2FromArray3by3(H, FTensor::Number<0>(), 0);
        auto t_invH = getFTensor2FromArray3by3(invH, FTensor::Number<0>(), 0);
        auto t_dot_u =
            FTensor::Tensor1<adouble *, 3>{&dot_u[0], &dot_u[1], &dot_u[2]};
        auto t_dot_w =
            FTensor::Tensor1<adouble *, 3>{&dot_w[0], &dot_w[1], &dot_w[2]};
        auto t_dot_W =
            FTensor::Tensor1<adouble *, 3>{&dot_W[0], &dot_W[1], &dot_W[2]};
        auto t_v = FTensor::Tensor1<adouble *, 3>{&v[0], &v[1], &v[2]};
        auto t_a_res =
            FTensor::Tensor1<adouble *, 3>{&a_res[0], &a_res[1], &a_res[2]};

        if (commonData.gradAtGaussPts[commonData.meshPositions].size() > 0) {
          detH = determinantTensor3by3(H);
          CHKERR invertTensor3by3(H, detH, invH);
          t_F(i, j) = t_h(i, k) * t_invH(k, j);
        } else {
          t_F(i, j) = t_h(i, j);
        }

        t_dot_u(i) = t_dot_w(i) + t_F(i, j) * t_dot_W(j);
        t_a_res(i) = t_v(i) - t_dot_u(i);
        t_a_res(i) *= detH;

        // dependant
        VectorDouble &res = commonData.valVel[gg];
        res.resize(3);
        for (int rr = 0; rr < 3; rr++) {
          a_res[rr] >>= res[rr];
        }
        trace_off();
      }

      active.resize(nb_active_vars);
      int aa = 0;
      for (int nn1 = 0; nn1 < 3; nn1++) {
        active[aa++] =
            commonData.dataAtGaussPts[commonData.spatialVelocities][gg][nn1];
      }
      for (int nn1 = 0; nn1 < 3; nn1++) {
        active[aa++] =
            commonData
                .dataAtGaussPts["DOT_" + commonData.spatialPositions][gg][nn1];
      }
      if (commonData.dataAtGaussPts["DOT_" + commonData.meshPositions].size() >
          0) {
        for (int nn1 = 0; nn1 < 3; nn1++) {
          for (int nn2 = 0; nn2 < 3; nn2++) {
            if (fieldDisp && nn1 == nn2) {
              active[aa++] =
                  commonData.gradAtGaussPts[commonData.spatialPositions][gg](
                      nn1, nn2) +
                  1;
            } else {
              active[aa++] =
                  commonData.gradAtGaussPts[commonData.spatialPositions][gg](
                      nn1, nn2);
            }
          }
        }
        for (int nn1 = 0; nn1 < 3; nn1++) {
          active[aa++] =
              commonData
                  .dataAtGaussPts["DOT_" + commonData.meshPositions][gg][nn1];
        }
      }
      if (commonData.gradAtGaussPts[commonData.meshPositions].size() > 0) {
        for (int nn1 = 0; nn1 < 3; nn1++) {
          for (int nn2 = 0; nn2 < 3; nn2++) {
            active[aa++] =
                commonData.gradAtGaussPts[commonData.meshPositions][gg](nn1,
                                                                        nn2);
          }
        }
      }

      if (!jAcobian) {
        VectorDouble &res = commonData.valVel[gg];
        if (gg > 0) {
          res.resize(3);
          int r;
          r = ::function(tAg, 3, nb_active_vars, &active[0], &res[0]);
          if (r != 3) {
            SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                    "ADOL-C function evaluation with error");
          }
        }
        double val = getVolume() * getGaussPts()(3, gg);
        res *= val;
      } else {
        commonData.jacVelRowPtr[gg].resize(3);
        commonData.jacVel[gg].resize(3, nb_active_vars);
        for (int nn1 = 0; nn1 < 3; nn1++) {
          (commonData.jacVelRowPtr[gg])[nn1] = &(commonData.jacVel[gg](nn1, 0));
        }
        int r;
        r = jacobian(tAg, 3, nb_active_vars, &active[0],
                     &(commonData.jacVelRowPtr[gg])[0]);
        if (r != 3) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                  "ADOL-C function evaluation with error");
        }
        double val = getVolume() * getGaussPts()(3, gg);
        commonData.jacVel[gg] *= val;
        // std::cerr << gg << " : " << commonData.jacVel[gg] << std::endl;
      }
    }
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpVelocityRhs::OpVelocityRhs(
    const std::string field_name, BlockData &data, CommonData &common_data)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
      dAta(data), commonData(common_data) {}

MoFEMErrorCode ConvectiveMassElement::OpVelocityRhs::doWork(
    int row_side, EntityType row_type,
    EntitiesFieldData::EntData &row_data) {
  MoFEMFunctionBeginHot;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }
  int nb_dofs = row_data.getIndices().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  auto base = row_data.getFTensor0N();
  int nb_base_functions = row_data.getN().size2();
  FTensor::Index<'i', 3> i;

  {

    nf.resize(nb_dofs);
    nf.clear();

    for (unsigned int gg = 0; gg < row_data.getN().size1(); gg++) {
      FTensor::Tensor1<double *, 3> t_nf(&nf[0], &nf[1], &nf[2], 3);
      FTensor::Tensor1<double *, 3> res(&commonData.valVel[gg][0],
                                        &commonData.valVel[gg][1],
                                        &commonData.valVel[gg][2]);
      int dd = 0;
      for (; dd < nb_dofs / 3; dd++) {
        t_nf(i) += base * res(i);
        ++base;
        ++t_nf;
      }
      for (; dd != nb_base_functions; dd++) {
        ++base;
      }
    }

    if (row_data.getIndices().size() > 3 * row_data.getN().size2()) {
      SETERRQ(PETSC_COMM_SELF, 1, "data inconsistency");
    }
    CHKERR VecSetValues(getFEMethod()->ts_F, row_data.getIndices().size(),
                        &row_data.getIndices()[0], &nf[0], ADD_VALUES);
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpVelocityLhs_dV_dv::OpVelocityLhs_dV_dv(
    const std::string vel_field, const std::string field_name, BlockData &data,
    CommonData &common_data)
    : OpMassLhs_dM_dv(vel_field, field_name, data, common_data) {}

MoFEMErrorCode ConvectiveMassElement::OpVelocityLhs_dV_dv::getJac(
    EntitiesFieldData::EntData &col_data, int gg) {
  MoFEMFunctionBeginHot;
  int nb_col = col_data.getIndices().size();
  jac.clear();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  double *base_ptr = const_cast<double *>(&col_data.getN(gg)[0]);
  FTensor::Tensor0<double *> base(base_ptr, 1);
  FTensor::Tensor2<double *, 3, 3> t_jac(&jac(0, 0), &jac(0, 1), &jac(0, 2),
                                         &jac(1, 0), &jac(1, 1), &jac(1, 2),
                                         &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
  FTensor::Tensor2<double *, 3, 3> t_mass1(
      &commonData.jacVel[gg](0, 0), &commonData.jacVel[gg](0, 1),
      &commonData.jacVel[gg](0, 2), &commonData.jacVel[gg](1, 0),
      &commonData.jacVel[gg](1, 1), &commonData.jacVel[gg](1, 2),
      &commonData.jacVel[gg](2, 0), &commonData.jacVel[gg](2, 1),
      &commonData.jacVel[gg](2, 2));
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  for (int dd = 0; dd < nb_col / 3; dd++) {
    t_jac(i, j) += t_mass1(i, j) * base;
    ++base_ptr;
    ++t_jac;
  }

  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpVelocityLhs_dV_dx::OpVelocityLhs_dV_dx(
    const std::string vel_field, const std::string field_name, BlockData &data,
    CommonData &common_data)
    : OpVelocityLhs_dV_dv(vel_field, field_name, data, common_data) {}

MoFEMErrorCode ConvectiveMassElement::OpVelocityLhs_dV_dx::getJac(
    EntitiesFieldData::EntData &col_data, int gg) {
  MoFEMFunctionBeginHot;
  int nb_col = col_data.getIndices().size();
  jac.clear();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  double *base_ptr = const_cast<double *>(&col_data.getN(gg)[0]);
  FTensor::Tensor0<double *> base(base_ptr, 1);
  FTensor::Tensor2<double *, 3, 3> t_jac(&jac(0, 0), &jac(0, 1), &jac(0, 2),
                                         &jac(1, 0), &jac(1, 1), &jac(1, 2),
                                         &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
  const int u = 3;
  FTensor::Tensor2<double *, 3, 3> t_mass1(
      &commonData.jacVel[gg](0, u + 0), &commonData.jacVel[gg](0, u + 1),
      &commonData.jacVel[gg](0, u + 2), &commonData.jacVel[gg](1, u + 0),
      &commonData.jacVel[gg](1, u + 1), &commonData.jacVel[gg](1, u + 2),
      &commonData.jacVel[gg](2, u + 0), &commonData.jacVel[gg](2, u + 1),
      &commonData.jacVel[gg](2, u + 2));
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  if (commonData.dataAtGaussPts["DOT_" + commonData.meshPositions].size() ==
      0) {

    for (int dd = 0; dd < nb_col / 3; dd++) {
      t_jac(i, j) += t_mass1(i, j) * base * getFEMethod()->ts_a;
      ++base_ptr;
      ++t_jac;
    }
  } else {
    double *diff_ptr =
        const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
    FTensor::Tensor1<double *, 3> diff(diff_ptr, &diff_ptr[1], &diff_ptr[2], 3);
    const int s = 3 + 3;
    FTensor::Tensor3<double *, 3, 3, 3> t_mass3(
        // T* d000, T* d001, T* d002,
        // T* d010, T* d011, T* d012,
        // T* d020, T* d021, T* d022,
        // T* d100, T* d101, T* d102,
        // T* d110, T* d111, T* d112,
        // T* d120, T* d121, T* d122,
        // T* d200, T* d201, T* d202,
        // T* d210, T* d211, T* d212,
        // T* d220, T* d221, T* d222,
        &commonData.jacVel[gg](0, s + 0), &commonData.jacVel[gg](0, s + 1),
        &commonData.jacVel[gg](0, s + 2), &commonData.jacVel[gg](0, s + 3),
        &commonData.jacVel[gg](0, s + 4), &commonData.jacVel[gg](0, s + 5),
        &commonData.jacVel[gg](0, s + 6), &commonData.jacVel[gg](0, s + 7),
        &commonData.jacVel[gg](0, s + 8), &commonData.jacVel[gg](1, s + 0),
        &commonData.jacVel[gg](1, s + 1), &commonData.jacVel[gg](1, s + 2),
        &commonData.jacVel[gg](1, s + 3), &commonData.jacVel[gg](1, s + 4),
        &commonData.jacVel[gg](1, s + 5), &commonData.jacVel[gg](1, s + 6),
        &commonData.jacVel[gg](1, s + 7), &commonData.jacVel[gg](1, s + 8),
        &commonData.jacVel[gg](2, s + 0), &commonData.jacVel[gg](2, s + 1),
        &commonData.jacVel[gg](2, s + 2), &commonData.jacVel[gg](2, s + 3),
        &commonData.jacVel[gg](2, s + 4), &commonData.jacVel[gg](2, s + 5),
        &commonData.jacVel[gg](2, s + 6), &commonData.jacVel[gg](2, s + 7),
        &commonData.jacVel[gg](2, s + 8));
    for (int dd = 0; dd < nb_col / 3; dd++) {
      t_jac(i, j) += t_mass1(i, j) * base * getFEMethod()->ts_a;
      t_jac(i, j) += t_mass3(i, j, k) * diff(k);
      ++base_ptr;
      ++diff_ptr;
      ++t_jac;
    }
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpVelocityLhs_dV_dX::OpVelocityLhs_dV_dX(
    const std::string vel_field, const std::string field_name, BlockData &data,
    CommonData &common_data)
    : OpVelocityLhs_dV_dv(vel_field, field_name, data, common_data) {}

MoFEMErrorCode ConvectiveMassElement::OpVelocityLhs_dV_dX::getJac(
    EntitiesFieldData::EntData &col_data, int gg) {
  MoFEMFunctionBeginHot;
  int nb_col = col_data.getIndices().size();
  jac.clear();
  if (!nb_col)
    MoFEMFunctionReturnHot(0);
  double *base_ptr = const_cast<double *>(&col_data.getN(gg)[0]);
  FTensor::Tensor0<double *> base(base_ptr, 1);
  double *diff_ptr =
      const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
  FTensor::Tensor1<double *, 3> diff(diff_ptr, &diff_ptr[1], &diff_ptr[2], 3);
  FTensor::Tensor2<double *, 3, 3> t_jac(&jac(0, 0), &jac(0, 1), &jac(0, 2),
                                         &jac(1, 0), &jac(1, 1), &jac(1, 2),
                                         &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
  const int u = 3 + 3 + 9;
  FTensor::Tensor2<double *, 3, 3> t_mass1(
      &commonData.jacVel[gg](0, u + 0), &commonData.jacVel[gg](0, u + 1),
      &commonData.jacVel[gg](0, u + 2), &commonData.jacVel[gg](1, u + 0),
      &commonData.jacVel[gg](1, u + 1), &commonData.jacVel[gg](1, u + 2),
      &commonData.jacVel[gg](2, u + 0), &commonData.jacVel[gg](2, u + 1),
      &commonData.jacVel[gg](2, u + 2));
  const int s = 3 + 3 + 9 + 3;
  FTensor::Tensor3<double *, 3, 3, 3> t_mass3(
      // T* d000, T* d001, T* d002,
      // T* d010, T* d011, T* d012,
      // T* d020, T* d021, T* d022,
      // T* d100, T* d101, T* d102,
      // T* d110, T* d111, T* d112,
      // T* d120, T* d121, T* d122,
      // T* d200, T* d201, T* d202,
      // T* d210, T* d211, T* d212,
      // T* d220, T* d221, T* d222,
      &commonData.jacVel[gg](0, s + 0), &commonData.jacVel[gg](0, s + 1),
      &commonData.jacVel[gg](0, s + 2), &commonData.jacVel[gg](0, s + 3),
      &commonData.jacVel[gg](0, s + 4), &commonData.jacVel[gg](0, s + 5),
      &commonData.jacVel[gg](0, s + 6), &commonData.jacVel[gg](0, s + 7),
      &commonData.jacVel[gg](0, s + 8), &commonData.jacVel[gg](1, s + 0),
      &commonData.jacVel[gg](1, s + 1), &commonData.jacVel[gg](1, s + 2),
      &commonData.jacVel[gg](1, s + 3), &commonData.jacVel[gg](1, s + 4),
      &commonData.jacVel[gg](1, s + 5), &commonData.jacVel[gg](1, s + 6),
      &commonData.jacVel[gg](1, s + 7), &commonData.jacVel[gg](1, s + 8),
      &commonData.jacVel[gg](2, s + 0), &commonData.jacVel[gg](2, s + 1),
      &commonData.jacVel[gg](2, s + 2), &commonData.jacVel[gg](2, s + 3),
      &commonData.jacVel[gg](2, s + 4), &commonData.jacVel[gg](2, s + 5),
      &commonData.jacVel[gg](2, s + 6), &commonData.jacVel[gg](2, s + 7),
      &commonData.jacVel[gg](2, s + 8));
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  for (int dd = 0; dd < nb_col / 3; dd++) {
    t_jac(i, j) += t_mass1(i, j) * base * getFEMethod()->ts_a;
    t_jac(i, j) += t_mass3(i, j, k) * diff(k);
    ++base_ptr;
    ++diff_ptr;
    ++t_jac;
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumJacobian::
    OpEshelbyDynamicMaterialMomentumJacobian(const std::string field_name,
                                             BlockData &data,
                                             CommonData &common_data, int tag,
                                             bool jacobian)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
      dAta(data), commonData(common_data), tAg(tag), jAcobian(jacobian),
      fieldDisp(false) {}

MoFEMErrorCode
ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumJacobian::doWork(
    int row_side, EntityType row_type,
    EntitiesFieldData::EntData &row_data) {
  MoFEMFunctionBeginHot;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  // do it only once, no need to repeat this for edges,faces or tets
  if (row_type != MBVERTEX)
    MoFEMFunctionReturnHot(0);

  int nb_dofs = row_data.getIndices().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  try {

    a.resize(3);
    v.resize(3);
    g.resize(3, 3);
    G.resize(3, 3);
    h.resize(3, 3);
    F.resize(3, 3);
    H.resize(3, 3);
    H.clear();
    invH.resize(3, 3);
    invH.clear();
    for (int dd = 0; dd < 3; dd++) {
      H(dd, dd) = 1;
      invH(dd, dd) = 1;
    }

    int nb_gauss_pts = row_data.getN().size1();
    commonData.valT.resize(nb_gauss_pts);
    commonData.jacTRowPtr.resize(nb_gauss_pts);
    commonData.jacT.resize(nb_gauss_pts);

    int nb_active_vars = 0;
    for (int gg = 0; gg < nb_gauss_pts; gg++) {

      if (gg == 0) {

        trace_on(tAg);

        for (int nn1 = 0; nn1 < 3; nn1++) { // 0
          a[nn1] <<=
              commonData.dataAtGaussPts["DOT_" + commonData.spatialVelocities]
                                       [gg][nn1];
          nb_active_vars++;
        }

        for (int nn1 = 0; nn1 < 3; nn1++) { // 3
          v[nn1] <<=
              commonData.dataAtGaussPts[commonData.spatialVelocities][gg][nn1];
          nb_active_vars++;
        }
        for (int nn1 = 0; nn1 < 3; nn1++) { // 3+3
          for (int nn2 = 0; nn2 < 3; nn2++) {
            g(nn1, nn2) <<=
                commonData.gradAtGaussPts[commonData.spatialVelocities][gg](
                    nn1, nn2);
            nb_active_vars++;
          }
        }
        for (int nn1 = 0; nn1 < 3; nn1++) { // 3+3+9
          for (int nn2 = 0; nn2 < 3; nn2++) {
            h(nn1, nn2) <<=
                commonData.gradAtGaussPts[commonData.spatialPositions][gg](nn1,
                                                                           nn2);
            nb_active_vars++;
            if (fieldDisp) {
              if (nn1 == nn2) {
                h(nn1, nn2) += 1;
              }
            }
          }
        }
        if (commonData.gradAtGaussPts[commonData.meshPositions].size() > 0) {
          for (int nn1 = 0; nn1 < 3; nn1++) { // 3+3+9+9
            for (int nn2 = 0; nn2 < 3; nn2++) {
              H(nn1, nn2) <<=
                  commonData.gradAtGaussPts[commonData.meshPositions][gg](nn1,
                                                                          nn2);
              nb_active_vars++;
            }
          }
        }
        adouble detH;
        detH = 1;
        if (commonData.gradAtGaussPts[commonData.meshPositions].size() > 0) {
          detH = determinantTensor3by3(H);
        }
        CHKERR invertTensor3by3(H, detH, invH);

        FTensor::Index<'i', 3> i;
        FTensor::Index<'j', 3> j;
        FTensor::Index<'k', 3> k;

        a_T.resize(3);

        auto t_h = getFTensor2FromArray3by3(h, FTensor::Number<0>(), 0);
        auto t_invH = getFTensor2FromArray3by3(invH, FTensor::Number<0>(), 0);
        auto t_F = getFTensor2FromArray3by3(F, FTensor::Number<0>(), 0);
        auto t_g = getFTensor2FromArray3by3(g, FTensor::Number<0>(), 0);
        auto t_G = getFTensor2FromArray3by3(G, FTensor::Number<0>(), 0);

        auto t_a = FTensor::Tensor1<adouble *, 3>{&a[0], &a[1], &a[2]};
        auto t_v = FTensor::Tensor1<adouble *, 3>{&v[0], &v[1], &v[2]};
        auto t_a_T = FTensor::Tensor1<adouble *, 3>{&a_T[0], &a_T[1], &a_T[2]};

        t_F(i, j) = t_h(i, k) * t_invH(k, j);
        t_G(i, j) = t_g(i, k) * t_invH(k, j);
        t_a_T(i) = t_F(k, i) * t_a(k) + t_G(k, i) * t_v(k);
        const auto rho0 = dAta.rho0;
        t_a_T(i) = -rho0 * detH;

        commonData.valT[gg].resize(3);
        for (int nn = 0; nn < 3; nn++) {
          a_T[nn] >>= (commonData.valT[gg])[nn];
        }
        trace_off();
      }

      active.resize(nb_active_vars);
      int aa = 0;
      for (int nn1 = 0; nn1 < 3; nn1++) { // 0
        active[aa++] =
            commonData
                .dataAtGaussPts["DOT_" + commonData.spatialVelocities][gg][nn1];
      }

      for (int nn1 = 0; nn1 < 3; nn1++) { // 3
        active[aa++] =
            commonData.dataAtGaussPts[commonData.spatialVelocities][gg][nn1];
      }
      for (int nn1 = 0; nn1 < 3; nn1++) { // 3+3
        for (int nn2 = 0; nn2 < 3; nn2++) {
          active[aa++] =
              commonData.gradAtGaussPts[commonData.spatialVelocities][gg](nn1,
                                                                          nn2);
        }
      }
      for (int nn1 = 0; nn1 < 3; nn1++) { // 3+3+9
        for (int nn2 = 0; nn2 < 3; nn2++) {
          if (fieldDisp && nn1 == nn2) {
            active[aa++] =
                commonData.gradAtGaussPts[commonData.spatialPositions][gg](
                    nn1, nn2) +
                1;
          } else {
            active[aa++] =
                commonData.gradAtGaussPts[commonData.spatialPositions][gg](nn1,
                                                                           nn2);
          }
        }
      }
      if (commonData.gradAtGaussPts[commonData.meshPositions].size() > 0) {
        for (int nn1 = 0; nn1 < 3; nn1++) { // 3+3+9+9
          for (int nn2 = 0; nn2 < 3; nn2++) {
            active[aa++] =
                commonData.gradAtGaussPts[commonData.meshPositions][gg](nn1,
                                                                        nn2);
          }
        }
      }

      if (!jAcobian) {
        VectorDouble &res = commonData.valT[gg];
        if (gg > 0) {
          res.resize(3);
          int r;
          r = ::function(tAg, 3, nb_active_vars, &active[0], &res[0]);
          if (r != 3) { // function is locally analytic
            SETERRQ1(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                     "ADOL-C function evaluation with error r = %d", r);
          }
        }
        double val = getVolume() * getGaussPts()(3, gg);
        res *= val;
      } else {
        commonData.jacTRowPtr[gg].resize(3);
        commonData.jacT[gg].resize(3, nb_active_vars);
        for (int nn1 = 0; nn1 < 3; nn1++) {
          (commonData.jacTRowPtr[gg])[nn1] = &(commonData.jacT[gg](nn1, 0));
        }
        int r;
        r = jacobian(tAg, 3, nb_active_vars, &active[0],
                     &(commonData.jacTRowPtr[gg])[0]);
        if (r != 3) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_OPERATION_UNSUCCESSFUL,
                  "ADOL-C function evaluation with error");
        }
        double val = getVolume() * getGaussPts()(3, gg);
        commonData.jacT[gg] *= val;
      }
    }

  } catch (const std::exception &ex) {
    std::ostringstream ss;
    ss << "throw in method: " << ex.what() << std::endl;
    SETERRQ(PETSC_COMM_SELF, 1, ss.str().c_str());
  }

  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumRhs::
    OpEshelbyDynamicMaterialMomentumRhs(const std::string field_name,
                                        BlockData &data,
                                        CommonData &common_data,
                                        Range *forcesonlyonentities_ptr)
    : VolumeElementForcesAndSourcesCore::UserDataOperator(
          field_name, ForcesAndSourcesCore::UserDataOperator::OPROW),
      dAta(data), commonData(common_data) {
  if (forcesonlyonentities_ptr != NULL) {
    forcesOnlyOnEntities = *forcesonlyonentities_ptr;
  }
}

MoFEMErrorCode
ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumRhs::doWork(
    int row_side, EntityType row_type,
    EntitiesFieldData::EntData &row_data) {
  MoFEMFunctionBeginHot;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }
  int nb_dofs = row_data.getIndices().size();
  if (nb_dofs == 0)
    MoFEMFunctionReturnHot(0);

  try {

    nf.resize(nb_dofs);
    nf.clear();

    auto base = row_data.getFTensor0N();
    int nb_base_functions = row_data.getN().size2();
    FTensor::Index<'i', 3> i;

    for (unsigned int gg = 0; gg < row_data.getN().size1(); gg++) {
      FTensor::Tensor1<double *, 3> t_nf(&nf[0], &nf[1], &nf[2], 3);
      FTensor::Tensor1<double *, 3> res(&commonData.valT[gg][0],
                                        &commonData.valT[gg][1],
                                        &commonData.valT[gg][2]);
      int dd = 0;
      for (; dd < nb_dofs / 3; dd++) {
        t_nf(i) += base * res(i);
        ++base;
        ++t_nf;
      }
      for (; dd != nb_base_functions; dd++) {
        ++base;
      }
    }

    if (row_data.getIndices().size() > 3 * row_data.getN().size2()) {
      SETERRQ(PETSC_COMM_SELF, 1, "data inconsistency");
    }
    if (!forcesOnlyOnEntities.empty()) {
      VectorInt indices = row_data.getIndices();
      VectorDofs &dofs = row_data.getFieldDofs();
      VectorDofs::iterator dit = dofs.begin();
      for (int ii = 0; dit != dofs.end(); dit++, ii++) {
        if (forcesOnlyOnEntities.find((*dit)->getEnt()) ==
            forcesOnlyOnEntities.end()) {
          // std::cerr << **dit << std::endl;
          indices[ii] = -1;
        }
      }
      // std::cerr << indices << std::endl;
      CHKERR VecSetValues(getFEMethod()->ts_F, indices.size(), &indices[0],
                          &nf[0], ADD_VALUES);
    } else {
      CHKERR VecSetValues(getFEMethod()->ts_F, row_data.getIndices().size(),
                          &row_data.getIndices()[0], &nf[0], ADD_VALUES);
    }

  } catch (const std::exception &ex) {
    std::ostringstream ss;
    ss << "throw in method: " << ex.what() << std::endl;
    SETERRQ(PETSC_COMM_SELF, 1, ss.str().c_str());
  }

  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumLhs_dv::
    OpEshelbyDynamicMaterialMomentumLhs_dv(const std::string vel_field,
                                           const std::string field_name,
                                           BlockData &data,
                                           CommonData &common_data,
                                           Range *forcesonlyonentities_ptr)
    : ConvectiveMassElement::OpMassLhs_dM_dv(
          vel_field, field_name, data, common_data, forcesonlyonentities_ptr) {}

MoFEMErrorCode
ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumLhs_dv::getJac(
    EntitiesFieldData::EntData &col_data, int gg) {
  MoFEMFunctionBeginHot;
  int nb_col = col_data.getIndices().size();
  jac.clear();
  double *base_ptr = const_cast<double *>(&col_data.getN(gg)[0]);
  FTensor::Tensor0<double *> base(base_ptr, 1);
  double *diff_ptr =
      const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
  FTensor::Tensor1<double *, 3> diff(diff_ptr, &diff_ptr[1], &diff_ptr[2], 3);
  FTensor::Tensor2<double *, 3, 3> t_jac(&jac(0, 0), &jac(0, 1), &jac(0, 2),
                                         &jac(1, 0), &jac(1, 1), &jac(1, 2),
                                         &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
  const int u = 3;
  FTensor::Tensor2<double *, 3, 3> t_mass1(
      &commonData.jacT[gg](0, u + 0), &commonData.jacT[gg](0, u + 1),
      &commonData.jacT[gg](0, u + 2), &commonData.jacT[gg](1, u + 0),
      &commonData.jacT[gg](1, u + 1), &commonData.jacT[gg](1, u + 2),
      &commonData.jacT[gg](2, u + 0), &commonData.jacT[gg](2, u + 1),
      &commonData.jacT[gg](2, u + 2));
  const int s = 3 + 3;
  FTensor::Tensor3<double *, 3, 3, 3> t_mass3(
      // T* d000, T* d001, T* d002,
      // T* d010, T* d011, T* d012,
      // T* d020, T* d021, T* d022,
      // T* d100, T* d101, T* d102,
      // T* d110, T* d111, T* d112,
      // T* d120, T* d121, T* d122,
      // T* d200, T* d201, T* d202,
      // T* d210, T* d211, T* d212,
      // T* d220, T* d221, T* d222,
      &commonData.jacT[gg](0, s + 0), &commonData.jacT[gg](0, s + 1),
      &commonData.jacT[gg](0, s + 2), &commonData.jacT[gg](0, s + 3),
      &commonData.jacT[gg](0, s + 4), &commonData.jacT[gg](0, s + 5),
      &commonData.jacT[gg](0, s + 6), &commonData.jacT[gg](0, s + 7),
      &commonData.jacT[gg](0, s + 8), &commonData.jacT[gg](1, s + 0),
      &commonData.jacT[gg](1, s + 1), &commonData.jacT[gg](1, s + 2),
      &commonData.jacT[gg](1, s + 3), &commonData.jacT[gg](1, s + 4),
      &commonData.jacT[gg](1, s + 5), &commonData.jacT[gg](1, s + 6),
      &commonData.jacT[gg](1, s + 7), &commonData.jacT[gg](1, s + 8),
      &commonData.jacT[gg](2, s + 0), &commonData.jacT[gg](2, s + 1),
      &commonData.jacT[gg](2, s + 2), &commonData.jacT[gg](2, s + 3),
      &commonData.jacT[gg](2, s + 4), &commonData.jacT[gg](2, s + 5),
      &commonData.jacT[gg](2, s + 6), &commonData.jacT[gg](2, s + 7),
      &commonData.jacT[gg](2, s + 8));
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  for (int dd = 0; dd < nb_col / 3; dd++) {
    t_jac(i, j) += t_mass1(i, j) * base * getFEMethod()->ts_a;
    t_jac(i, j) += t_mass3(i, j, k) * diff(k);
    ++base_ptr;
    ++diff_ptr;
    ++t_jac;
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumLhs_dx::
    OpEshelbyDynamicMaterialMomentumLhs_dx(const std::string vel_field,
                                           const std::string field_name,
                                           BlockData &data,
                                           CommonData &common_data,
                                           Range *forcesonlyonentities_ptr)
    : ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumLhs_dv(
          vel_field, field_name, data, common_data, forcesonlyonentities_ptr) {}

MoFEMErrorCode
ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumLhs_dx::getJac(
    EntitiesFieldData::EntData &col_data, int gg) {
  MoFEMFunctionBeginHot;
  int nb_col = col_data.getIndices().size();
  jac.clear();
  double *diff_ptr =
      const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
  FTensor::Tensor1<double *, 3> diff(diff_ptr, &diff_ptr[1], &diff_ptr[2], 3);
  FTensor::Tensor2<double *, 3, 3> t_jac(&jac(0, 0), &jac(0, 1), &jac(0, 2),
                                         &jac(1, 0), &jac(1, 1), &jac(1, 2),
                                         &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
  const int s = 3 + 3 + 9;
  FTensor::Tensor3<double *, 3, 3, 3> t_mass3(
      // T* d000, T* d001, T* d002,
      // T* d010, T* d011, T* d012,
      // T* d020, T* d021, T* d022,
      // T* d100, T* d101, T* d102,
      // T* d110, T* d111, T* d112,
      // T* d120, T* d121, T* d122,
      // T* d200, T* d201, T* d202,
      // T* d210, T* d211, T* d212,
      // T* d220, T* d221, T* d222,
      &commonData.jacT[gg](0, s + 0), &commonData.jacT[gg](0, s + 1),
      &commonData.jacT[gg](0, s + 2), &commonData.jacT[gg](0, s + 3),
      &commonData.jacT[gg](0, s + 4), &commonData.jacT[gg](0, s + 5),
      &commonData.jacT[gg](0, s + 6), &commonData.jacT[gg](0, s + 7),
      &commonData.jacT[gg](0, s + 8), &commonData.jacT[gg](1, s + 0),
      &commonData.jacT[gg](1, s + 1), &commonData.jacT[gg](1, s + 2),
      &commonData.jacT[gg](1, s + 3), &commonData.jacT[gg](1, s + 4),
      &commonData.jacT[gg](1, s + 5), &commonData.jacT[gg](1, s + 6),
      &commonData.jacT[gg](1, s + 7), &commonData.jacT[gg](1, s + 8),
      &commonData.jacT[gg](2, s + 0), &commonData.jacT[gg](2, s + 1),
      &commonData.jacT[gg](2, s + 2), &commonData.jacT[gg](2, s + 3),
      &commonData.jacT[gg](2, s + 4), &commonData.jacT[gg](2, s + 5),
      &commonData.jacT[gg](2, s + 6), &commonData.jacT[gg](2, s + 7),
      &commonData.jacT[gg](2, s + 8));
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  for (int dd = 0; dd < nb_col / 3; dd++) {
    t_jac(i, j) += t_mass3(i, j, k) * diff(k);
    ++diff_ptr;
    ++t_jac;
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumLhs_dX::
    OpEshelbyDynamicMaterialMomentumLhs_dX(const std::string vel_field,
                                           const std::string field_name,
                                           BlockData &data,
                                           CommonData &common_data,
                                           Range *forcesonlyonentities_ptr)
    : ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumLhs_dv(
          vel_field, field_name, data, common_data, forcesonlyonentities_ptr) {}

MoFEMErrorCode
ConvectiveMassElement::OpEshelbyDynamicMaterialMomentumLhs_dX::getJac(
    EntitiesFieldData::EntData &col_data, int gg) {
  MoFEMFunctionBeginHot;
  int nb_col = col_data.getIndices().size();
  jac.clear();
  double *diff_ptr =
      const_cast<double *>(&(col_data.getDiffN(gg, nb_col / 3)(0, 0)));
  FTensor::Tensor1<double *, 3> diff(diff_ptr, &diff_ptr[1], &diff_ptr[2], 3);
  FTensor::Tensor2<double *, 3, 3> t_jac(&jac(0, 0), &jac(0, 1), &jac(0, 2),
                                         &jac(1, 0), &jac(1, 1), &jac(1, 2),
                                         &jac(2, 0), &jac(2, 1), &jac(2, 2), 3);
  const int s = 3 + 3 + 9 + 9;
  FTensor::Tensor3<double *, 3, 3, 3> t_mass3(
      // T* d000, T* d001, T* d002,
      // T* d010, T* d011, T* d012,
      // T* d020, T* d021, T* d022,
      // T* d100, T* d101, T* d102,
      // T* d110, T* d111, T* d112,
      // T* d120, T* d121, T* d122,
      // T* d200, T* d201, T* d202,
      // T* d210, T* d211, T* d212,
      // T* d220, T* d221, T* d222,
      &commonData.jacT[gg](0, s + 0), &commonData.jacT[gg](0, s + 1),
      &commonData.jacT[gg](0, s + 2), &commonData.jacT[gg](0, s + 3),
      &commonData.jacT[gg](0, s + 4), &commonData.jacT[gg](0, s + 5),
      &commonData.jacT[gg](0, s + 6), &commonData.jacT[gg](0, s + 7),
      &commonData.jacT[gg](0, s + 8), &commonData.jacT[gg](1, s + 0),
      &commonData.jacT[gg](1, s + 1), &commonData.jacT[gg](1, s + 2),
      &commonData.jacT[gg](1, s + 3), &commonData.jacT[gg](1, s + 4),
      &commonData.jacT[gg](1, s + 5), &commonData.jacT[gg](1, s + 6),
      &commonData.jacT[gg](1, s + 7), &commonData.jacT[gg](1, s + 8),
      &commonData.jacT[gg](2, s + 0), &commonData.jacT[gg](2, s + 1),
      &commonData.jacT[gg](2, s + 2), &commonData.jacT[gg](2, s + 3),
      &commonData.jacT[gg](2, s + 4), &commonData.jacT[gg](2, s + 5),
      &commonData.jacT[gg](2, s + 6), &commonData.jacT[gg](2, s + 7),
      &commonData.jacT[gg](2, s + 8));
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  for (int dd = 0; dd < nb_col / 3; dd++) {
    t_jac(i, j) += t_mass3(i, j, k) * diff(k);
    ++diff_ptr;
    ++t_jac;
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::UpdateAndControl::UpdateAndControl(
    MoFEM::Interface &m_field, TS _ts, const std::string velocity_field,
    const std::string spatial_position_field)
    : mField(m_field), tS(_ts), velocityField(velocity_field),
      spatialPositionField(spatial_position_field), jacobianLag(-1) {}

MoFEMErrorCode ConvectiveMassElement::UpdateAndControl::preProcess() {
  MoFEMFunctionBeginHot;

  switch (ts_ctx) {
  case CTX_TSSETIFUNCTION: {
    snes_f = ts_F;
    // FIXME: This global scattering because Kuu problem and Dynamic problem
    // not share partitions. Both problem should use the same partitioning to
    // resolve this problem.
    CHKERR mField.getInterface<VecManager>()->setGlobalGhostVector(
        problemPtr, COL, ts_u, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR mField.getInterface<VecManager>()->setOtherGlobalGhostVector(
        problemPtr, velocityField, "DOT_" + velocityField, COL, ts_u_t,
        INSERT_VALUES, SCATTER_REVERSE);
    CHKERR mField.getInterface<VecManager>()->setOtherGlobalGhostVector(
        problemPtr, spatialPositionField, "DOT_" + spatialPositionField, COL,
        ts_u_t, INSERT_VALUES, SCATTER_REVERSE);
    break;
  }
  case CTX_TSSETIJACOBIAN: {
    snes_B = ts_B;
    break;
  }
  default:
    break;
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::UpdateAndControl::postProcess() {
  MoFEMFunctionBeginHot;
  //
  // SNES snes;
  // CHKERR TSGetSNES(tS,&snes);
  // CHKERR SNESSetLagJacobian(snes,jacobianLag);
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::setBlocks() {
  MoFEMFunctionBeginHot;

  Range added_tets;
  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
           mField, BLOCKSET | BODYFORCESSET, it)) {
    int id = it->getMeshsetId();
    EntityHandle meshset = it->getMeshset();
    CHKERR mField.get_moab().get_entities_by_type(meshset, MBTET,
                                                  setOfBlocks[id].tEts, true);
    added_tets.merge(setOfBlocks[id].tEts);
    Block_BodyForces mydata;
    CHKERR it->getAttributeDataStructure(mydata);
    setOfBlocks[id].rho0 = mydata.data.density;
    setOfBlocks[id].a0.resize(3);
    setOfBlocks[id].a0[0] = mydata.data.acceleration_x;
    setOfBlocks[id].a0[1] = mydata.data.acceleration_y;
    setOfBlocks[id].a0[2] = mydata.data.acceleration_z;
    // std::cerr << setOfBlocks[id].tEts << std::endl;
  }

  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
           mField, BLOCKSET | MAT_ELASTICSET, it)) {
    Mat_Elastic mydata;
    CHKERR it->getAttributeDataStructure(mydata);
    if (mydata.data.User1 == 0)
      continue;
    Range tets;
    EntityHandle meshset = it->getMeshset();
    CHKERR mField.get_moab().get_entities_by_type(meshset, MBTET, tets, true);
    tets = subtract(tets, added_tets);
    if (tets.empty())
      continue;
    int id = it->getMeshsetId();
    setOfBlocks[-id].tEts = tets;
    setOfBlocks[-id].rho0 = mydata.data.User1;
    setOfBlocks[-id].a0.resize(3);
    setOfBlocks[-id].a0[0] = mydata.data.User2;
    setOfBlocks[-id].a0[1] = mydata.data.User3;
    setOfBlocks[-id].a0[2] = mydata.data.User4;
    // std::cerr << setOfBlocks[id].tEts << std::endl;
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::setBlocks(
    MoFEM::Interface &m_field,
    boost::shared_ptr<map<int, BlockData>> &block_sets_ptr) {
  MoFEMFunctionBegin;

  if (!block_sets_ptr)
    SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Pointer to block of sets is null");

  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
           m_field, BLOCKSET | BODYFORCESSET, it)) {
    Block_BodyForces mydata;
    CHKERR it->getAttributeDataStructure(mydata);
    int id = it->getMeshsetId();
    auto &block_data = (*block_sets_ptr)[id];
    EntityHandle meshset = it->getMeshset();
    CHKERR m_field.get_moab().get_entities_by_dimension(meshset, 3,
                                                        block_data.tEts, true);
    block_data.rho0 = mydata.data.density;
    block_data.a0.resize(3);
    block_data.a0[0] = mydata.data.acceleration_x;
    block_data.a0[1] = mydata.data.acceleration_y;
    block_data.a0[2] = mydata.data.acceleration_z;
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ConvectiveMassElement::addConvectiveMassElement(
    string element_name, string velocity_field_name,
    string spatial_position_field_name, string material_position_field_name,
    bool ale, BitRefLevel bit) {
  MoFEMFunctionBeginHot;

  //

  CHKERR mField.add_finite_element(element_name, MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                    velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                    velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                     velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_row(
      element_name, spatial_position_field_name);
  CHKERR mField.modify_finite_element_add_field_col(
      element_name, spatial_position_field_name);
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, spatial_position_field_name);
  if (mField.check_field(material_position_field_name)) {
    if (ale) {
      CHKERR mField.modify_finite_element_add_field_row(
          element_name, material_position_field_name);
      CHKERR mField.modify_finite_element_add_field_col(
          element_name, material_position_field_name);
      CHKERR mField.modify_finite_element_add_field_data(
          element_name, "DOT_" + material_position_field_name);
    }
    CHKERR mField.modify_finite_element_add_field_data(
        element_name, material_position_field_name);
  }
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, "DOT_" + velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, "DOT_" + spatial_position_field_name);

  Range tets;
  if (bit.any()) {
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit, BitRefLevel().set(), MBTET, tets);
  }

  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    Range add_tets = sit->second.tEts;
    if (!tets.empty()) {
      add_tets = intersect(add_tets, tets);
    }
    CHKERR mField.add_ents_to_finite_element_by_type(add_tets, MBTET,
                                                     element_name);
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::addVelocityElement(
    string element_name, string velocity_field_name,
    string spatial_position_field_name, string material_position_field_name,
    bool ale, BitRefLevel bit) {
  MoFEMFunctionBeginHot;

  //

  CHKERR mField.add_finite_element(element_name, MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_row(element_name,
                                                    velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                    velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                     velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_col(
      element_name, spatial_position_field_name);
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, spatial_position_field_name);
  if (mField.check_field(material_position_field_name)) {
    if (ale) {
      CHKERR mField.modify_finite_element_add_field_col(
          element_name, material_position_field_name);
      CHKERR mField.modify_finite_element_add_field_data(
          element_name, "DOT_" + material_position_field_name);
    }
    CHKERR mField.modify_finite_element_add_field_data(
        element_name, material_position_field_name);
  }
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, "DOT_" + velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, "DOT_" + spatial_position_field_name);

  Range tets;
  if (bit.any()) {
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit, BitRefLevel().set(), MBTET, tets);
  }

  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    Range add_tets = sit->second.tEts;
    if (!tets.empty()) {
      add_tets = intersect(add_tets, tets);
    }
    CHKERR mField.add_ents_to_finite_element_by_type(add_tets, MBTET,
                                                     element_name);
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::addEshelbyDynamicMaterialMomentum(
    string element_name, string velocity_field_name,
    string spatial_position_field_name, string material_position_field_name,
    bool ale, BitRefLevel bit, Range *intersected) {
  MoFEMFunctionBeginHot;

  //

  CHKERR mField.add_finite_element(element_name, MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_col(element_name,
                                                    velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_data(element_name,
                                                     velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_col(
      element_name, spatial_position_field_name);
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, spatial_position_field_name);
  if (mField.check_field(material_position_field_name)) {
    if (ale) {
      CHKERR mField.modify_finite_element_add_field_row(
          element_name, material_position_field_name);
      CHKERR mField.modify_finite_element_add_field_col(
          element_name, material_position_field_name);
      CHKERR mField.modify_finite_element_add_field_data(
          element_name, "DOT_" + material_position_field_name);
    }
    CHKERR mField.modify_finite_element_add_field_data(
        element_name, material_position_field_name);
  }
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, "DOT_" + velocity_field_name);
  CHKERR mField.modify_finite_element_add_field_data(
      element_name, "DOT_" + spatial_position_field_name);

  Range tets;
  if (bit.any()) {
    CHKERR mField.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit, BitRefLevel().set(), MBTET, tets);
  }
  if (intersected != NULL) {
    if (tets.empty()) {
      tets = *intersected;
    } else {
      tets = intersect(*intersected, tets);
    }
  }

  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    Range add_tets = sit->second.tEts;
    if (!tets.empty()) {
      add_tets = intersect(add_tets, tets);
    }
    CHKERR mField.add_ents_to_finite_element_by_type(add_tets, MBTET,
                                                     element_name);
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::setConvectiveMassOperators(
    string velocity_field_name, string spatial_position_field_name,
    string material_position_field_name, bool ale, bool linear) {
  MoFEMFunctionBeginHot;

  commonData.spatialPositions = spatial_position_field_name;
  commonData.meshPositions = material_position_field_name;
  commonData.spatialVelocities = velocity_field_name;
  commonData.lInear = linear;

  // Rhs
  feMassRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feMassRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  feMassRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts("DOT_" + velocity_field_name, commonData));
  feMassRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
      "DOT_" + spatial_position_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feMassRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
    if (ale) {
      feMassRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
          "DOT_" + material_position_field_name, commonData));
    } else {
      feMassRhs.meshPositionsFieldName = material_position_field_name;
    }
  }
  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feMassRhs.getOpPtrVector().push_back(
        new OpMassJacobian(spatial_position_field_name, sit->second, commonData,
                           methodsOp, tAg, false));
    feMassRhs.getOpPtrVector().push_back(
        new OpMassRhs(spatial_position_field_name, sit->second, commonData));
  }

  // Lhs
  feMassLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feMassLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  feMassLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts("DOT_" + velocity_field_name, commonData));
  feMassLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
      "DOT_" + spatial_position_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feMassLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
    if (ale) {
      feMassLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
          "DOT_" + material_position_field_name, commonData));
    } else {
      feMassLhs.meshPositionsFieldName = material_position_field_name;
    }
  }
  sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feMassLhs.getOpPtrVector().push_back(
        new OpMassJacobian(spatial_position_field_name, sit->second, commonData,
                           methodsOp, tAg, true));
    feMassLhs.getOpPtrVector().push_back(
        new OpMassLhs_dM_dv(spatial_position_field_name, velocity_field_name,
                            sit->second, commonData));
    feMassLhs.getOpPtrVector().push_back(new OpMassLhs_dM_dx(
        spatial_position_field_name, spatial_position_field_name, sit->second,
        commonData));
    if (mField.check_field(material_position_field_name)) {
      if (ale) {
        feMassLhs.getOpPtrVector().push_back(new OpMassLhs_dM_dX(
            spatial_position_field_name, material_position_field_name,
            sit->second, commonData));
      } else {
        feMassLhs.meshPositionsFieldName = material_position_field_name;
      }
    }
  }

  // Energy
  feEnergy.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feEnergy.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feEnergy.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
    feEnergy.meshPositionsFieldName = material_position_field_name;
  }
  sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feEnergy.getOpPtrVector().push_back(new OpEnergy(
        spatial_position_field_name, sit->second, commonData, feEnergy.V));
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::setVelocityOperators(
    string velocity_field_name, string spatial_position_field_name,
    string material_position_field_name, bool ale) {
  MoFEMFunctionBeginHot;

  commonData.spatialPositions = spatial_position_field_name;
  commonData.meshPositions = material_position_field_name;
  commonData.spatialVelocities = velocity_field_name;

  // Rhs
  feVelRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feVelRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  feVelRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts("DOT_" + velocity_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feVelRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        "DOT_" + spatial_position_field_name, commonData));
    feVelRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
    if (ale) {
      feVelRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
          "DOT_" + material_position_field_name, commonData));
    } else {
      feVelRhs.meshPositionsFieldName = material_position_field_name;
    }
  }
  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feVelRhs.getOpPtrVector().push_back(new OpVelocityJacobian(
        velocity_field_name, sit->second, commonData, tAg, false));
    feVelRhs.getOpPtrVector().push_back(
        new OpVelocityRhs(velocity_field_name, sit->second, commonData));
  }

  // Lhs
  feVelLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feVelLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  feVelLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts("DOT_" + velocity_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feVelLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        "DOT_" + spatial_position_field_name, commonData));
    feVelLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
    if (ale) {
      feVelLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
          "DOT_" + material_position_field_name, commonData));
    } else {
      feVelLhs.meshPositionsFieldName = material_position_field_name;
    }
  }
  sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feVelLhs.getOpPtrVector().push_back(new OpVelocityJacobian(
        velocity_field_name, sit->second, commonData, tAg));
    feVelLhs.getOpPtrVector().push_back(new OpVelocityLhs_dV_dv(
        velocity_field_name, velocity_field_name, sit->second, commonData));
    feVelLhs.getOpPtrVector().push_back(new OpVelocityLhs_dV_dx(
        velocity_field_name, spatial_position_field_name, sit->second,
        commonData));
    if (mField.check_field(material_position_field_name)) {
      if (ale) {
        feVelLhs.getOpPtrVector().push_back(new OpVelocityLhs_dV_dX(
            velocity_field_name, material_position_field_name, sit->second,
            commonData));
      } else {
        feVelLhs.meshPositionsFieldName = material_position_field_name;
      }
    }
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::setKinematicEshelbyOperators(
    string velocity_field_name, string spatial_position_field_name,
    string material_position_field_name, Range *forces_on_entities_ptr) {
  MoFEMFunctionBeginHot;

  commonData.spatialPositions = spatial_position_field_name;
  commonData.meshPositions = material_position_field_name;
  commonData.spatialVelocities = velocity_field_name;

  // Rhs
  feTRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feTRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  feTRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(material_position_field_name, commonData));
  feTRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts("DOT_" + velocity_field_name, commonData));

  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feTRhs.getOpPtrVector().push_back(
        new OpEshelbyDynamicMaterialMomentumJacobian(
            material_position_field_name, sit->second, commonData, tAg, false));
    feTRhs.getOpPtrVector().push_back(new OpEshelbyDynamicMaterialMomentumRhs(
        material_position_field_name, sit->second, commonData,
        forces_on_entities_ptr));
  }

  // Lhs
  feTLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feTLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  feTLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts("DOT_" + velocity_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feTLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
  }
  sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feTLhs.getOpPtrVector().push_back(
        new OpEshelbyDynamicMaterialMomentumJacobian(
            material_position_field_name, sit->second, commonData, tAg));
    feTLhs.getOpPtrVector().push_back(
        new OpEshelbyDynamicMaterialMomentumLhs_dv(
            material_position_field_name, velocity_field_name, sit->second,
            commonData, forces_on_entities_ptr));
    feTLhs.getOpPtrVector().push_back(
        new OpEshelbyDynamicMaterialMomentumLhs_dx(
            material_position_field_name, spatial_position_field_name,
            sit->second, commonData, forces_on_entities_ptr));
    feTLhs.getOpPtrVector().push_back(
        new OpEshelbyDynamicMaterialMomentumLhs_dX(
            material_position_field_name, material_position_field_name,
            sit->second, commonData, forces_on_entities_ptr));
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::setShellMatrixMassOperators(
    string velocity_field_name, string spatial_position_field_name,
    string material_position_field_name, bool linear) {
  MoFEMFunctionBeginHot;

  commonData.spatialPositions = spatial_position_field_name;
  commonData.meshPositions = material_position_field_name;
  commonData.spatialVelocities = velocity_field_name;
  commonData.lInear = linear;

  // Rhs
  feMassRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feMassRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  feMassRhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts("DOT_" + velocity_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feMassRhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
    feMassRhs.meshPositionsFieldName = material_position_field_name;
  }
  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feMassRhs.getOpPtrVector().push_back(
        new OpMassJacobian(spatial_position_field_name, sit->second, commonData,
                           methodsOp, tAg, false));
    feMassRhs.getOpPtrVector().push_back(
        new OpMassRhs(spatial_position_field_name, sit->second, commonData));
  }

  // Lhs
  feMassLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feMassLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  feMassLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts("DOT_" + velocity_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feMassLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
    feMassLhs.meshPositionsFieldName = material_position_field_name;
  }
  sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feMassLhs.getOpPtrVector().push_back(
        new OpMassJacobian(spatial_position_field_name, sit->second, commonData,
                           methodsOp, tAg, true));
    feMassLhs.getOpPtrVector().push_back(new OpMassLhs_dM_dv(
        spatial_position_field_name, spatial_position_field_name, sit->second,
        commonData));
    if (mField.check_field(material_position_field_name)) {
      feMassLhs.meshPositionsFieldName = material_position_field_name;
    }
  }

  // Aux Lhs
  feMassAuxLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feMassAuxLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  feMassAuxLhs.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts("DOT_" + velocity_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feMassAuxLhs.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
    feMassAuxLhs.meshPositionsFieldName = material_position_field_name;
  }
  sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feMassAuxLhs.getOpPtrVector().push_back(
        new OpMassJacobian(spatial_position_field_name, sit->second, commonData,
                           methodsOp, tAg, true));
    feMassAuxLhs.getOpPtrVector().push_back(new OpMassLhs_dM_dx(
        spatial_position_field_name, spatial_position_field_name, sit->second,
        commonData));
    if (mField.check_field(material_position_field_name)) {
      feMassAuxLhs.meshPositionsFieldName = material_position_field_name;
    }
  }

  // Energy E=0.5*rho*v*v
  feEnergy.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(velocity_field_name, commonData));
  feEnergy.getOpPtrVector().push_back(
      new OpGetCommonDataAtGaussPts(spatial_position_field_name, commonData));
  if (mField.check_field(material_position_field_name)) {
    feEnergy.getOpPtrVector().push_back(new OpGetCommonDataAtGaussPts(
        material_position_field_name, commonData));
    feEnergy.meshPositionsFieldName = material_position_field_name;
  }
  sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    feEnergy.getOpPtrVector().push_back(new OpEnergy(
        spatial_position_field_name, sit->second, commonData, feEnergy.V));
  }

  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::MatShellCtx::MatShellCtx() : iNitialized(false) {}
ConvectiveMassElement::MatShellCtx::~MatShellCtx() {
  if (iNitialized) {

    CHKERR dEstroy();
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }
}

MoFEMErrorCode ConvectiveMassElement::MatShellCtx::iNit() {
  MoFEMFunctionBeginHot;
  if (!iNitialized) {

#if PETSC_VERSION_GE(3, 5, 3)
    CHKERR MatCreateVecs(K, &u, &Ku);
    CHKERR MatCreateVecs(M, &v, &Mv);
#else
    CHKERR MatGetVecs(K, &u, &Ku);
    CHKERR MatGetVecs(M, &v, &Mv);
#endif
    CHKERR MatDuplicate(K, MAT_SHARE_NONZERO_PATTERN, &barK);
    iNitialized = true;
  }
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::MatShellCtx::dEstroy() {
  MoFEMFunctionBeginHot;
  if (iNitialized) {

    CHKERR VecDestroy(&u);
    CHKERR VecDestroy(&Ku);
    CHKERR VecDestroy(&v);
    CHKERR VecDestroy(&Mv);
    CHKERR MatDestroy(&barK);
    iNitialized = false;
  }
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::PCShellCtx::iNit() {
  MoFEMFunctionBeginHot;

  if (!initPC) {
    MPI_Comm comm;
    CHKERR PetscObjectGetComm((PetscObject)shellMat, &comm);
    CHKERR PCCreate(comm, &pC);
    initPC = true;
  }
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ConvectiveMassElement::PCShellCtx::dEstroy() {
  MoFEMFunctionBeginHot;

  if (initPC) {
    CHKERR PCDestroy(&pC);
    initPC = false;
  }
  MoFEMFunctionReturnHot(0);
}

ConvectiveMassElement::ShellResidualElement::ShellResidualElement(
    MoFEM::Interface &m_field)
    : mField(m_field) {}

MoFEMErrorCode ConvectiveMassElement::ShellResidualElement::preProcess() {
  MoFEMFunctionBeginHot;

  if (ts_ctx != CTX_TSSETIFUNCTION) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "It is used to residual of velocities");
  }
  if (!shellMatCtx->iNitialized) {
    CHKERR shellMatCtx->iNit();
  }
  // Note velocities calculate from displacements are stroed in shellMatCtx->u
  CHKERR VecScatterBegin(shellMatCtx->scatterU, ts_u_t, shellMatCtx->u,
                         INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecScatterEnd(shellMatCtx->scatterU, ts_u_t, shellMatCtx->u,
                       INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecScatterBegin(shellMatCtx->scatterV, ts_u, shellMatCtx->v,
                         INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecScatterEnd(shellMatCtx->scatterV, ts_u, shellMatCtx->v,
                       INSERT_VALUES, SCATTER_FORWARD);
  CHKERR VecAXPY(shellMatCtx->v, -1, shellMatCtx->u);
  CHKERR VecScatterBegin(shellMatCtx->scatterV, shellMatCtx->v, ts_F,
                         ADD_VALUES, SCATTER_REVERSE);
  CHKERR VecScatterEnd(shellMatCtx->scatterV, shellMatCtx->v, ts_F, ADD_VALUES,
                       SCATTER_REVERSE);
  // VecView(shellMatCtx->v,PETSC_VIEWER_STDOUT_WORLD);

  MoFEMFunctionReturnHot(0);
}


#ifdef __DIRICHLET_HPP__

ConvectiveMassElement::ShellMatrixElement::ShellMatrixElement(
    MoFEM::Interface &m_field)
    : mField(m_field) {}

MoFEMErrorCode ConvectiveMassElement::ShellMatrixElement::preProcess() {
  MoFEMFunctionBeginHot;

  if (ts_ctx != CTX_TSSETIJACOBIAN) {
    SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
            "It is used to calculate shell matrix only");
  }

  shellMatCtx->ts_a = ts_a;
  DirichletBcPtr->copyTs(*((TSMethod *)this)); // copy context for TSMethod

  DirichletBcPtr->dIag = 1;
  DirichletBcPtr->ts_B = shellMatCtx->K;
  CHKERR MatZeroEntries(shellMatCtx->K);
  CHKERR mField.problem_basic_method_preProcess(problemName, *DirichletBcPtr);
  LoopsToDoType::iterator itk = loopK.begin();
  for (; itk != loopK.end(); itk++) {
    itk->second->copyTs(*((TSMethod *)this));
    itk->second->ts_B = shellMatCtx->K;
    CHKERR mField.loop_finite_elements(problemName, itk->first, *itk->second);
  }
  LoopsToDoType::iterator itam = loopAuxM.begin();
  for (; itam != loopAuxM.end(); itam++) {
    itam->second->copyTs(*((TSMethod *)this));
    itam->second->ts_B = shellMatCtx->K;
    CHKERR mField.loop_finite_elements(problemName, itam->first, *itam->second);
  }
  CHKERR mField.problem_basic_method_postProcess(problemName, *DirichletBcPtr);
  CHKERR MatAssemblyBegin(shellMatCtx->K, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(shellMatCtx->K, MAT_FINAL_ASSEMBLY);

  DirichletBcPtr->dIag = 0;
  DirichletBcPtr->ts_B = shellMatCtx->M;
  CHKERR MatZeroEntries(shellMatCtx->M);
  // CHKERR mField.problem_basic_method_preProcess(problemName,*DirichletBcPtr);
  LoopsToDoType::iterator itm = loopM.begin();
  for (; itm != loopM.end(); itm++) {
    itm->second->copyTs(*((TSMethod *)this));
    itm->second->ts_B = shellMatCtx->M;
    CHKERR mField.loop_finite_elements(problemName, itm->first, *itm->second);
  }
  CHKERR mField.problem_basic_method_postProcess(problemName, *DirichletBcPtr);
  CHKERR MatAssemblyBegin(shellMatCtx->M, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(shellMatCtx->M, MAT_FINAL_ASSEMBLY);

  // barK
  CHKERR MatZeroEntries(shellMatCtx->barK);
  CHKERR MatCopy(shellMatCtx->K, shellMatCtx->barK, SAME_NONZERO_PATTERN);
  CHKERR MatAXPY(shellMatCtx->barK, ts_a, shellMatCtx->M, SAME_NONZERO_PATTERN);
  CHKERR MatAssemblyBegin(shellMatCtx->barK, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(shellMatCtx->barK, MAT_FINAL_ASSEMBLY);

  // Matrix View
  // MatView(shellMatCtx->barK,PETSC_VIEWER_DRAW_WORLD);//PETSC_VIEWER_STDOUT_WORLD);
  // std::string wait;
  // std::cin >> wait;

  MoFEMFunctionReturnHot(0);
}

#endif //__DIRICHLET_HPP__
