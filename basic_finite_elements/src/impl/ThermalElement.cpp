/** \file ThermalElement.cpp
  \ingroup mofem_thermal_elem
*/



#include <MoFEM.hpp>
using namespace MoFEM;
#include <ThermalElement.hpp>

using namespace boost::numeric;

MoFEMErrorCode ThermalElement::OpGetGradAtGaussPts::doWork(
    int side, EntityType type, EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  int nb_dofs = data.getFieldData().size();
  int nb_gauss_pts = data.getN().size1();

  // initialize
  commonData.gradAtGaussPts.resize(nb_gauss_pts, 3);
  if (type == MBVERTEX) {
    std::fill(commonData.gradAtGaussPts.data().begin(),
              commonData.gradAtGaussPts.data().end(), 0);
  }

  for (int gg = 0; gg < nb_gauss_pts; gg++) {
    ublas::noalias(commonData.getGradAtGaussPts(gg)) +=
        prod(trans(data.getDiffN(gg, nb_dofs)), data.getFieldData());
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ThermalElement::OpThermalRhs::doWork(int side, EntityType type,
                                     EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end())
    MoFEMFunctionReturnHot(0);

  int nb_row_dofs = data.getIndices().size();
  Nf.resize(nb_row_dofs);
  Nf.clear();

  for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

    MatrixDouble val =
        dAta.cOnductivity_mat * getVolume() * getGaussPts()(3, gg);

    // ublas
    ublas::noalias(Nf) += prod(prod(data.getDiffN(gg, nb_row_dofs), val),
                               commonData.getGradAtGaussPts(gg));
  }

  if (useTsF) {
    CHKERR VecSetValues(getFEMethod()->ts_F, data.getIndices().size(),
                        &data.getIndices()[0], &Nf[0], ADD_VALUES);
  } else {
    CHKERR VecSetValues(F, data.getIndices().size(), &data.getIndices()[0],
                        &Nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::OpThermalLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tEts.end()) {
    MoFEMFunctionReturnHot(0);
  }

  if (row_data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (col_data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  int nb_row = row_data.getN().size2();
  int nb_col = col_data.getN().size2();
  K.resize(nb_row, nb_col);
  K.clear();
  for (unsigned int gg = 0; gg < row_data.getN().size1(); gg++) {

    MatrixDouble val =
        dAta.cOnductivity_mat * getVolume() * getGaussPts()(3, gg);

    // ublas
    MatrixDouble K1 = prod(row_data.getDiffN(gg, nb_row), val);
    noalias(K) += prod(K1, trans(col_data.getDiffN(gg, nb_col)));
  }

  if (!useTsB) {
    const_cast<FEMethod *>(getFEMethod())->ts_B = A;
  }
  CHKERR MatSetValues((getFEMethod()->ts_B), nb_row, &row_data.getIndices()[0],
                      nb_col, &col_data.getIndices()[0], &K(0, 0), ADD_VALUES);
  if (row_side != col_side || row_type != col_type) {
    transK.resize(nb_col, nb_row);
    noalias(transK) = trans(K);
    CHKERR MatSetValues((getFEMethod()->ts_B), nb_col,
                        &col_data.getIndices()[0], nb_row,
                        &row_data.getIndices()[0], &transK(0, 0), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::OpHeatCapacityRhs::doWork(
    int side, EntityType type, EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  int nb_row = data.getN().size2();
  Nf.resize(nb_row);
  Nf.clear();
  for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {
    double val = getGaussPts()(3, gg);
    val *= commonData.temperatureRateAtGaussPts[gg];
    ////////////
    // cblas
    // cblas_daxpy(nb_row,val,&data.getN()(gg,0),1,&*Nf.data().begin(),1);
    // ublas
    ublas::noalias(Nf) += val * data.getN(gg);
  }
  Nf *= getVolume() * dAta.cApacity;

  CHKERR VecSetValues(getFEMethod()->ts_F, data.getIndices().size(),
                      &data.getIndices()[0], &Nf[0], ADD_VALUES);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::OpHeatCapacityLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (col_data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  int nb_row = row_data.getN().size2();
  int nb_col = col_data.getN().size2();
  M.resize(nb_row, nb_col);
  M.clear();

  for (unsigned int gg = 0; gg < row_data.getN().size1(); gg++) {

    double val = getGaussPts()(3, gg);

    // ublas
    noalias(M) +=
        val * outer_prod(row_data.getN(gg, nb_row), col_data.getN(gg, nb_col));
  }

  M *= getVolume() * dAta.cApacity * getFEMethod()->ts_a;

  CHKERR MatSetValues((getFEMethod()->ts_B), nb_row, &row_data.getIndices()[0],
                      nb_col, &col_data.getIndices()[0], &M(0, 0), ADD_VALUES);
  if (row_side != col_side || row_type != col_type) {
    transM.resize(nb_col, nb_row);
    noalias(transM) = trans(M);
    CHKERR MatSetValues((getFEMethod()->ts_B), nb_col,
                        &col_data.getIndices()[0], nb_row,
                        &row_data.getIndices()[0], &transM(0, 0), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ThermalElement::OpHeatFlux::doWork(int side, EntityType type,
                                   EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tRis.end())
    MoFEMFunctionReturnHot(0);

  const auto &dof_ptr = data.getFieldDofs()[0];
  int rank = dof_ptr->getNbOfCoeffs();

  int nb_dofs = data.getIndices().size() / rank;

  Nf.resize(data.getIndices().size());
  Nf.clear();

  for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

    double val = getGaussPts()(2, gg);
    double flux;
    if (hoGeometry) {
      const double area = norm_2(getNormalsAtGaussPts(gg)) * 0.5;
      flux = dAta.dAta.data.value1 * area; 
    } else {
      flux = dAta.dAta.data.value1 * getArea();
    }
    ublas::noalias(Nf) += val * flux * data.getN(gg, nb_dofs);
  }

  if (useTsF || F == PETSC_NULL) {
    CHKERR VecSetValues(getFEMethod()->ts_F, data.getIndices().size(),
                        &data.getIndices()[0], &Nf[0], ADD_VALUES);
  } else {
    CHKERR VecSetValues(F, data.getIndices().size(), &data.getIndices()[0],
                        &Nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::OpRadiationLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
  MoFEMFunctionBegin;

  if (row_data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (col_data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  int nb_row = row_data.getN().size2();
  int nb_col = col_data.getN().size2();

  N.resize(nb_row, nb_col);
  N.clear();

  for (unsigned int gg = 0; gg < row_data.getN().size1(); gg++) {
    double T3_at_Gauss_pt = pow(commonData.temperatureAtGaussPts[gg], 3.0);

    double radiationConst;
    if (hoGeometry) {
      double area = norm_2(getNormalsAtGaussPts(gg)) * 0.5;
      radiationConst = dAta.sIgma * dAta.eMissivity * area;
    } else {
      radiationConst = dAta.sIgma * dAta.eMissivity * getArea();
    }
    const double fOur = 4.0;
    double val = fOur * getGaussPts()(2, gg) * radiationConst * T3_at_Gauss_pt;
    noalias(N) +=
        val * outer_prod(row_data.getN(gg, nb_row), col_data.getN(gg, nb_col));
  }

  if (!useTsB) {
    const_cast<FEMethod *>(getFEMethod())->ts_B = A;
  }
  CHKERR MatSetValues((getFEMethod()->ts_B), nb_row, &row_data.getIndices()[0],
                      nb_col, &col_data.getIndices()[0], &N(0, 0), ADD_VALUES);
  if (row_side != col_side || row_type != col_type) {
    transN.resize(nb_col, nb_row);
    noalias(transN) = trans(N);
    CHKERR MatSetValues((getFEMethod()->ts_B), nb_col,
                        &col_data.getIndices()[0], nb_row,
                        &row_data.getIndices()[0], &transN(0, 0), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::OpRadiationRhs::doWork(
    int side, EntityType type, EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tRis.end())
    MoFEMFunctionReturnHot(0);

  const auto &dof_ptr = data.getFieldDofs()[0];
  int rank = dof_ptr->getNbOfCoeffs();
  int nb_row_dofs = data.getIndices().size() / rank;

  Nf.resize(data.getIndices().size());
  Nf.clear();

  for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

    double T4_at_Gauss_pt = pow(commonData.temperatureAtGaussPts[gg], 4.0);
    double ambientTemp = pow(dAta.aMbienttEmp, 4.0);
    double tEmp = 0;

    if (ambientTemp > 0) {
      tEmp = -ambientTemp + T4_at_Gauss_pt;
    }

    double val = getGaussPts()(2, gg);
    double radiationConst;

    if (hoGeometry) {
      double area = norm_2(getNormalsAtGaussPts(gg)) * 0.5;
      radiationConst = dAta.sIgma * dAta.eMissivity * tEmp * area;
    } else {
      radiationConst = dAta.sIgma * dAta.eMissivity * tEmp * getArea();
    }
    ublas::noalias(Nf) += val * radiationConst * data.getN(gg, nb_row_dofs);
  }

  if (useTsF) {
    CHKERR VecSetValues(getFEMethod()->ts_F, data.getIndices().size(),
                        &data.getIndices()[0], &Nf[0], ADD_VALUES);
  } else {
    CHKERR VecSetValues(F, data.getIndices().size(), &data.getIndices()[0],
                        &Nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::OpConvectionRhs::doWork(
    int side, EntityType type, EntitiesFieldData::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tRis.end())
    MoFEMFunctionReturnHot(0);

  const auto &dof_ptr = data.getFieldDofs()[0];
  int rank = dof_ptr->getNbOfCoeffs();

  int nb_row_dofs = data.getIndices().size() / rank;

  Nf.resize(data.getIndices().size());
  Nf.clear();

  for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

    double T_at_Gauss_pt = commonData.temperatureAtGaussPts[gg];
    double convectionConst;
    if (hoGeometry) {
      double area = norm_2(getNormalsAtGaussPts(gg)) * 0.5;
      convectionConst =
          dAta.cOnvection * area * (T_at_Gauss_pt - dAta.tEmperature);
    } else {
      convectionConst =
          dAta.cOnvection * getArea() * (T_at_Gauss_pt - dAta.tEmperature);
    }
    double val = getGaussPts()(2, gg) * convectionConst;
    ublas::noalias(Nf) += val * data.getN(gg, nb_row_dofs);
  }

  if (useTsF) {
    CHKERR VecSetValues(getFEMethod()->ts_F, data.getIndices().size(),
                        &data.getIndices()[0], &Nf[0], ADD_VALUES);
  } else {
    CHKERR VecSetValues(F, data.getIndices().size(), &data.getIndices()[0],
                        &Nf[0], ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::OpConvectionLhs::doWork(
    int row_side, int col_side, EntityType row_type, EntityType col_type,
    EntitiesFieldData::EntData &row_data,
    EntitiesFieldData::EntData &col_data) {
    MoFEMFunctionBegin;
if (row_data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);
  if (col_data.getIndices().size() == 0)
    MoFEMFunctionReturnHot(0);

  int nb_row = row_data.getN().size2();
  int nb_col = col_data.getN().size2();
  K.resize(nb_row, nb_col);
  K.clear();

  for (unsigned int gg = 0; gg < row_data.getN().size1(); gg++) {

    double convectionConst;
    if (hoGeometry) {
      double area = norm_2(getNormalsAtGaussPts(gg)) * 0.5;
      convectionConst = dAta.cOnvection * area;
    } else {
      convectionConst = dAta.cOnvection * getArea();
    }
    double val = getGaussPts()(2, gg) * convectionConst;
    noalias(K) +=
        val * outer_prod(row_data.getN(gg, nb_row), col_data.getN(gg, nb_col));
  }

  if (!useTsB) {
    const_cast<FEMethod *>(getFEMethod())->ts_B = A;
  }
  CHKERR MatSetValues((getFEMethod()->ts_B), nb_row, &row_data.getIndices()[0],
                      nb_col, &col_data.getIndices()[0], &K(0, 0), ADD_VALUES);
  if (row_side != col_side || row_type != col_type) {
    transK.resize(nb_col, nb_row);
    noalias(transK) = trans(K);
    CHKERR MatSetValues((getFEMethod()->ts_B), nb_col,
                        &col_data.getIndices()[0], nb_row,
                        &row_data.getIndices()[0], &transK(0, 0), ADD_VALUES);
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::UpdateAndControl::preProcess() {
  MoFEMFunctionBegin;
  CHKERR mField.getInterface<VecManager>()->setOtherLocalGhostVector(
      problemPtr, tempName, rateName, ROW, ts_u_t, INSERT_VALUES,
      SCATTER_REVERSE);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::UpdateAndControl::postProcess() {
  MoFEMFunctionBeginHot;
  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ThermalElement::TimeSeriesMonitor::postProcess() {
  MoFEMFunctionBegin;

  CHKERR mField.getInterface<VecManager>()->setGlobalGhostVector(
      problemPtr, ROW, ts_u, INSERT_VALUES, SCATTER_REVERSE);

  BitRefLevel proble_bit_level = problemPtr->getBitRefLevel();

  SeriesRecorder *recorder_ptr = NULL;
  CHKERR mField.getInterface(recorder_ptr);
  CHKERR recorder_ptr->record_begin(seriesName);
  CHKERR recorder_ptr->record_field(seriesName, tempName, proble_bit_level,
                                    mask);
  CHKERR recorder_ptr->record_end(seriesName, ts_t);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ThermalElement::addThermalElements(const std::string field_name,
                                   const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  CHKERR mField.add_finite_element("THERMAL_FE", MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_row("THERMAL_FE", field_name);
  CHKERR mField.modify_finite_element_add_field_col("THERMAL_FE", field_name);
  CHKERR mField.modify_finite_element_add_field_data("THERMAL_FE", field_name);
  if (mField.check_field(mesh_nodals_positions)) {
    CHKERR mField.modify_finite_element_add_field_data("THERMAL_FE",
                                                       mesh_nodals_positions);
  }

  // takes skin of block of entities
  // Skinner skin(&mField.get_moab());
  // loop over all blocksets and get data which name is FluidPressure
  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
           mField, BLOCKSET | MAT_THERMALSET, it)) {

    Mat_Thermal temp_data;
    ierr = it->getAttributeDataStructure(temp_data);

    setOfBlocks[it->getMeshsetId()].cOnductivity_mat.resize(
        3, 3); //(3X3) conductivity matrix
    setOfBlocks[it->getMeshsetId()].cOnductivity_mat.clear();
    setOfBlocks[it->getMeshsetId()].cOnductivity_mat(0, 0) =
        temp_data.data.Conductivity;
    setOfBlocks[it->getMeshsetId()].cOnductivity_mat(1, 1) =
        temp_data.data.Conductivity;
    setOfBlocks[it->getMeshsetId()].cOnductivity_mat(2, 2) =
        temp_data.data.Conductivity;
    // setOfBlocks[it->getMeshsetId()].cOnductivity =
    // temp_data.data.Conductivity;
    setOfBlocks[it->getMeshsetId()].cApacity = temp_data.data.HeatCapacity;
    if (temp_data.data.User2 != 0) {
      setOfBlocks[it->getMeshsetId()].initTemp = temp_data.data.User2;
    }
    CHKERR mField.get_moab().get_entities_by_type(
        it->meshset, MBTET, setOfBlocks[it->getMeshsetId()].tEts, true);
    CHKERR mField.add_ents_to_finite_element_by_type(
        setOfBlocks[it->getMeshsetId()].tEts, MBTET, "THERMAL_FE");
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ThermalElement::addThermalFluxElement(const std::string field_name,
                                      const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  CHKERR mField.add_finite_element("THERMAL_FLUX_FE", MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_row("THERMAL_FLUX_FE",
                                                    field_name);
  CHKERR mField.modify_finite_element_add_field_col("THERMAL_FLUX_FE",
                                                    field_name);
  CHKERR mField.modify_finite_element_add_field_data("THERMAL_FLUX_FE",
                                                     field_name);
  if (mField.check_field(mesh_nodals_positions)) {
    CHKERR mField.modify_finite_element_add_field_data("THERMAL_FLUX_FE",
                                                       mesh_nodals_positions);
  }

  for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(mField, SIDESET | HEATFLUXSET,
                                                  it)) {
    CHKERR it->getBcDataStructure(setOfFluxes[it->getMeshsetId()].dAta);
    CHKERR mField.get_moab().get_entities_by_type(
        it->meshset, MBTRI, setOfFluxes[it->getMeshsetId()].tRis, true);
    CHKERR mField.add_ents_to_finite_element_by_type(
        setOfFluxes[it->getMeshsetId()].tRis, MBTRI, "THERMAL_FLUX_FE");
  }

  // this is alternative method for setting boundary conditions, to bypass bu
  // in cubit file reader. not elegant, but good enough
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
    if (it->getName().compare(0, 9, "HEAT_FLUX") == 0) {
      std::vector<double> data;
      CHKERR it->getAttributes(data);
      if (data.size() != 1) {
        SETERRQ(PETSC_COMM_SELF, 1, "Data inconsistency");
      }
      strcpy(setOfFluxes[it->getMeshsetId()].dAta.data.name, "HeatFlu");
      setOfFluxes[it->getMeshsetId()].dAta.data.flag1 = 1;
      setOfFluxes[it->getMeshsetId()].dAta.data.value1 = data[0];
      CHKERR mField.get_moab().get_entities_by_type(
          it->meshset, MBTRI, setOfFluxes[it->getMeshsetId()].tRis, true);
      CHKERR mField.add_ents_to_finite_element_by_type(
          setOfFluxes[it->getMeshsetId()].tRis, MBTRI, "THERMAL_FLUX_FE");
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::addThermalConvectionElement(
    const std::string field_name, const std::string mesh_nodals_positions) {
  MoFEMFunctionBeginHot;

  CHKERR mField.add_finite_element("THERMAL_CONVECTION_FE", MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_row("THERMAL_CONVECTION_FE",
                                                    field_name);
  CHKERR mField.modify_finite_element_add_field_col("THERMAL_CONVECTION_FE",
                                                    field_name);
  CHKERR mField.modify_finite_element_add_field_data("THERMAL_CONVECTION_FE",
                                                     field_name);
  if (mField.check_field(mesh_nodals_positions)) {
    CHKERR mField.modify_finite_element_add_field_data("THERMAL_CONVECTION_FE",
                                                       mesh_nodals_positions);
  }

  // this is alternative method for setting boundary conditions, to bypass bu
  // in cubit file reader. not elegant, but good enough
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
    if (it->getName().compare(0, 10, "CONVECTION") == 0) {

      std::vector<double> data;
      CHKERR it->getAttributes(data);
      if (data.size() != 2) {
        SETERRQ(PETSC_COMM_SELF, 1, "Data inconsistency");
      }
      setOfConvection[it->getMeshsetId()].cOnvection = data[0];
      setOfConvection[it->getMeshsetId()].tEmperature = data[1];
      CHKERR mField.get_moab().get_entities_by_type(
          it->meshset, MBTRI, setOfConvection[it->getMeshsetId()].tRis, true);
      CHKERR mField.add_ents_to_finite_element_by_type(
          setOfConvection[it->getMeshsetId()].tRis, MBTRI,
          "THERMAL_CONVECTION_FE");
    }
  }

  MoFEMFunctionReturnHot(0);
}

MoFEMErrorCode ThermalElement::addThermalRadiationElement(
    const std::string field_name, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  CHKERR mField.add_finite_element("THERMAL_RADIATION_FE", MF_ZERO);
  CHKERR mField.modify_finite_element_add_field_row("THERMAL_RADIATION_FE",
                                                    field_name);
  CHKERR mField.modify_finite_element_add_field_col("THERMAL_RADIATION_FE",
                                                    field_name);
  CHKERR mField.modify_finite_element_add_field_data("THERMAL_RADIATION_FE",
                                                     field_name);
  if (mField.check_field(mesh_nodals_positions)) {
    CHKERR mField.modify_finite_element_add_field_data("THERMAL_RADIATION_FE",
                                                       mesh_nodals_positions);
  }

  // this is alternative method for setting boundary conditions, to bypass bu
  // in cubit file reader. not elegant, but good enough
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, it)) {
    if (it->getName().compare(0, 9, "RADIATION") == 0) {
      std::vector<double> data;
      ierr = it->getAttributes(data);
      if (data.size() != 3) {
        SETERRQ(PETSC_COMM_SELF, 1, "Data inconsistency");
      }
      setOfRadiation[it->getMeshsetId()].sIgma = data[0];
      setOfRadiation[it->getMeshsetId()].eMissivity = data[1];
      setOfRadiation[it->getMeshsetId()].aMbienttEmp = data[2];
      CHKERR mField.get_moab().get_entities_by_type(
          it->meshset, MBTRI, setOfRadiation[it->getMeshsetId()].tRis, true);
      CHKERR mField.add_ents_to_finite_element_by_type(
          setOfRadiation[it->getMeshsetId()].tRis, MBTRI,
          "THERMAL_RADIATION_FE");
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ThermalElement::setThermalFiniteElementRhsOperators(string field_name, Vec &F) {
  MoFEMFunctionBegin;
  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  feRhs.getOpPtrVector().push_back(
      new OpGetGradAtGaussPts(field_name, commonData));
  for (; sit != setOfBlocks.end(); sit++) {
    // add finite element
    feRhs.getOpPtrVector().push_back(
        new OpThermalRhs(field_name, F, sit->second, commonData));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ThermalElement::setThermalFiniteElementLhsOperators(string field_name, Mat A) {
  MoFEMFunctionBegin;
  std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
  for (; sit != setOfBlocks.end(); sit++) {
    // add finite elemen
    feLhs.getOpPtrVector().push_back(
        new OpThermalLhs(field_name, A, sit->second, commonData));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::setThermalFluxFiniteElementRhsOperators(
    string field_name, Vec &F, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;
  bool hoGeometry = false;
  if (mField.check_field(mesh_nodals_positions)) {
    hoGeometry = true;
  }
  std::map<int, FluxData>::iterator sit = setOfFluxes.begin();
  for (; sit != setOfFluxes.end(); sit++) {
    // add finite element
    feFlux.getOpPtrVector().push_back(
        new OpHeatFlux(field_name, F, sit->second, hoGeometry));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::setThermalConvectionFiniteElementRhsOperators(
    string field_name, Vec &F, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;
  bool hoGeometry = false;
  if (mField.check_field(mesh_nodals_positions)) {
    hoGeometry = true;
  }
  std::map<int, ConvectionData>::iterator sit = setOfConvection.begin();
  for (; sit != setOfConvection.end(); sit++) {
    // add finite element
    feConvectionRhs.getOpPtrVector().push_back(
        new OpGetTriTemperatureAtGaussPts(field_name, commonData));
    feConvectionRhs.getOpPtrVector().push_back(new OpConvectionRhs(
        field_name, F, sit->second, commonData, hoGeometry));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::setThermalConvectionFiniteElementLhsOperators(
    string field_name, Mat A, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;
  bool hoGeometry = false;
  if (mField.check_field(mesh_nodals_positions)) {
    hoGeometry = true;
  }
  std::map<int, ConvectionData>::iterator sit = setOfConvection.begin();
  for (; sit != setOfConvection.end(); sit++) {
    // add finite element
    feConvectionLhs.getOpPtrVector().push_back(
        new OpConvectionLhs(field_name, A, sit->second, hoGeometry));
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::setTimeSteppingProblem(
    string field_name, string rate_name,
    const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  bool hoGeometry = false;
  if (mField.check_field(mesh_nodals_positions)) {
    hoGeometry = true;
  }

  {
    std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
    for (; sit != setOfBlocks.end(); sit++) {
      // add finite element
      // those methods are to calculate matrices on Lhs
      //  feLhs.getOpPtrVector().push_back(new
      //  OpGetTetTemperatureAtGaussPts(field_name,commonData));
      feLhs.getOpPtrVector().push_back(
          new OpThermalLhs(field_name, sit->second, commonData));
      feLhs.getOpPtrVector().push_back(
          new OpHeatCapacityLhs(field_name, sit->second, commonData));
      // those methods are to calculate vectors on Rhs
      feRhs.getOpPtrVector().push_back(
          new OpGetTetTemperatureAtGaussPts(field_name, commonData));
      feRhs.getOpPtrVector().push_back(
          new OpGetTetRateAtGaussPts(rate_name, commonData));
      feRhs.getOpPtrVector().push_back(
          new OpGetGradAtGaussPts(field_name, commonData));
      feRhs.getOpPtrVector().push_back(
          new OpThermalRhs(field_name, sit->second, commonData));
      feRhs.getOpPtrVector().push_back(
          new OpHeatCapacityRhs(field_name, sit->second, commonData));
    }
  }

  // Flux
  {
    std::map<int, FluxData>::iterator sit = setOfFluxes.begin();
    for (; sit != setOfFluxes.end(); sit++) {
      feFlux.getOpPtrVector().push_back(
          new OpHeatFlux(field_name, sit->second, hoGeometry));
    }
  }

  // Convection
  {
    std::map<int, ConvectionData>::iterator sit = setOfConvection.begin();
    for (; sit != setOfConvection.end(); sit++) {
      feConvectionRhs.getOpPtrVector().push_back(
          new OpGetTriTemperatureAtGaussPts(field_name, commonData));
      feConvectionRhs.getOpPtrVector().push_back(
          new OpConvectionRhs(field_name, sit->second, commonData, hoGeometry));
    }
  }
  {
    std::map<int, ConvectionData>::iterator sit = setOfConvection.begin();
    for (; sit != setOfConvection.end(); sit++) {
      feConvectionLhs.getOpPtrVector().push_back(
          new OpConvectionLhs(field_name, sit->second, hoGeometry));
    }
  }

  // Radiation
  {
    std::map<int, RadiationData>::iterator sit = setOfRadiation.begin();
    for (; sit != setOfRadiation.end(); sit++) {
      feRadiationRhs.getOpPtrVector().push_back(
          new OpGetTriTemperatureAtGaussPts(field_name, commonData));
      feRadiationRhs.getOpPtrVector().push_back(
          new OpRadiationRhs(field_name, sit->second, commonData, hoGeometry));
    }
  }
  {
    std::map<int, RadiationData>::iterator sit = setOfRadiation.begin();
    for (; sit != setOfRadiation.end(); sit++) {
      feRadiationLhs.getOpPtrVector().push_back(
          new OpGetTriTemperatureAtGaussPts(field_name, commonData));
      feRadiationLhs.getOpPtrVector().push_back(
          new OpRadiationLhs(field_name, sit->second, commonData, hoGeometry));
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ThermalElement::setTimeSteppingProblem(
    TsCtx &ts_ctx, string field_name, string rate_name,
    const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  CHKERR setTimeSteppingProblem(field_name, rate_name, mesh_nodals_positions);

  // rhs
  TsCtx::FEMethodsSequence &loops_to_do_Rhs =
      ts_ctx.getLoopsIFunction();
  loops_to_do_Rhs.push_back(TsCtx::PairNameFEMethodPtr("THERMAL_FE", &feRhs));
  loops_to_do_Rhs.push_back(
      TsCtx::PairNameFEMethodPtr("THERMAL_FLUX_FE", &feFlux));
  if (mField.check_finite_element("THERMAL_CONVECTION_FE"))
    loops_to_do_Rhs.push_back(
        TsCtx::PairNameFEMethodPtr("THERMAL_CONVECTION_FE", &feConvectionRhs));
  if (mField.check_finite_element("THERMAL_RADIATION_FE"))
    loops_to_do_Rhs.push_back(
        TsCtx::PairNameFEMethodPtr("THERMAL_RADIATION_FE", &feRadiationRhs));

  // lhs
  TsCtx::FEMethodsSequence &loops_to_do_Mat =
      ts_ctx.getLoopsIJacobian();
  loops_to_do_Mat.push_back(TsCtx::PairNameFEMethodPtr("THERMAL_FE", &feLhs));
  if (mField.check_finite_element("THERMAL_CONVECTION_FE"))
    loops_to_do_Mat.push_back(
        TsCtx::PairNameFEMethodPtr("THERMAL_CONVECTION_FE", &feConvectionLhs));
  if (mField.check_finite_element("THERMAL_RADIATION_FE"))
    loops_to_do_Mat.push_back(
        TsCtx::PairNameFEMethodPtr("THERMAL_RADIATION_FE", &feRadiationLhs));

  MoFEMFunctionReturn(0);
}
