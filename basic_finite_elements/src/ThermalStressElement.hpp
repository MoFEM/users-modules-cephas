/** \file ThermalStressElement.hpp
  \ingroup mofem_thermal_elem
  \brief Implementation of thermal stresses element

*/

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

#ifndef __THERMALSTRESSELEMENT_HPP
#define __THERMALSTRESSELEMENT_HPP

/** \brief Implentation of thermal stress element
\ingroup mofem_thermal_elem
*/
struct ThermalStressElement {

  struct MyVolumeFE : public MoFEM::VolumeElementForcesAndSourcesCore {
    MyVolumeFE(MoFEM::Interface &m_field)
        : MoFEM::VolumeElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return 2 * (order - 1); };
  };

  MyVolumeFE feThermalStressRhs;
  MyVolumeFE &getLoopThermalStressRhs() { return feThermalStressRhs; }

  MoFEM::Interface &mField;
  ThermalStressElement(MoFEM::Interface &m_field)
      : feThermalStressRhs(m_field), mField(m_field) {}

  struct BlockData {
    double youngModulus;
    double poissonRatio;
    double thermalExpansion;
    double refTemperature;
    BlockData() : refTemperature(0) {}
    Range tEts;
  };
  std::map<int, BlockData> setOfBlocks;

  struct CommonData {
    VectorDouble temperatureAtGaussPts;
  };
  CommonData commonData;

  struct OpGetTemperatureAtGaussPts
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    CommonData &commonData;
    int verb;
    OpGetTemperatureAtGaussPts(const std::string field_name,
                               CommonData &common_data, int _verb = 0)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          commonData(common_data), verb(_verb) {}

    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;
      if (data.getFieldData().size() == 0)
        MoFEMFunctionReturnHot(0);
      int nb_dofs = data.getFieldData().size();
      int nb_gauss_pts = data.getN().size1();
      // initialize
      commonData.temperatureAtGaussPts.resize(nb_gauss_pts);
      if (type == MBVERTEX) {
        std::fill(commonData.temperatureAtGaussPts.begin(),
                  commonData.temperatureAtGaussPts.end(), 0);
      }
      for (int gg = 0; gg < nb_gauss_pts; gg++) {
        commonData.temperatureAtGaussPts[gg] +=
            inner_prod(data.getN(gg, nb_dofs), data.getFieldData());
      }
      MoFEMFunctionReturn(0);
    }
  };

  struct OpThermalStressRhs
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    Vec F;
    BlockData &dAta;
    CommonData &commonData;
    int verb;
    OpThermalStressRhs(const std::string field_name, Vec _F, BlockData &data,
                       CommonData &common_data, int _verb = 0)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              field_name, UserDataOperator::OPROW),
          F(_F), dAta(data), commonData(common_data), verb(_verb) {}

    VectorDouble Nf;
    MoFEMErrorCode doWork(int side, EntityType type,
                          DataForcesAndSourcesCore::EntData &data) {
      MoFEMFunctionBegin;

      if (data.getIndices().size() == 0)
        MoFEMFunctionReturnHot(0);
      if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
          dAta.tEts.end())
        MoFEMFunctionReturnHot(0);

      const auto dof_ptr = data.getFieldDofs()[0].lock().get();
      int rank = dof_ptr->getNbOfCoeffs();
      if (rank != 3) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "data inconsistency");
      }

      unsigned int nb_dofs = data.getIndices().size();
      if (nb_dofs % rank != 0) {
        SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                "data inconsistency");
      }
      if (data.getN().size2() < nb_dofs / rank) {
        SETERRQ3(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                 "data inconsistency N.size2 %d nb_dof %d rank %d",
                 data.getN().size2(), nb_dofs, rank);
      }

      Nf.resize(nb_dofs);
      bzero(&*Nf.data().begin(), nb_dofs * sizeof(FieldData));

      if (verb > VERBOSE) {
        if (type == MBVERTEX) {
          cout << commonData.temperatureAtGaussPts << std::endl;
          cout << "thermal expansion " << dAta.thermalExpansion << std::endl;
        }
      }

      for (unsigned int gg = 0; gg < data.getN().size1(); gg++) {

        if (dAta.refTemperature != dAta.refTemperature) {
          SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA, "invalid data");
        }

        double phi =
            (commonData.temperatureAtGaussPts[gg] - dAta.refTemperature);
        double eps_thermal = dAta.thermalExpansion * phi;
        double sig_thermal =
            -eps_thermal * (dAta.youngModulus / (1. - 2 * dAta.poissonRatio));

        double val = sig_thermal * getVolume() * getGaussPts()(3, gg);

        double *diff_N;
        diff_N = &data.getDiffN()(gg, 0);
        cblas_daxpy(nb_dofs, val, diff_N, 1, &Nf[0], 1);
      }

      CHKERR VecSetValues(F, data.getIndices().size(), &data.getIndices()[0],
                          &Nf[0], ADD_VALUES);

      MoFEMFunctionReturn(0);
    }
  };

  MoFEMErrorCode addThermalStressElement(
      const std::string fe_name, const std::string field_name,
      const std::string thermal_field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS") {
    MoFEMFunctionBegin;
    if (mField.check_field(thermal_field_name)) {

      CHKERR mField.add_finite_element(fe_name, MF_ZERO);
      CHKERR mField.modify_finite_element_add_field_row(fe_name, field_name);
      CHKERR mField.modify_finite_element_add_field_col(fe_name, field_name);
      CHKERR mField.modify_finite_element_add_field_data(fe_name, field_name);
      CHKERR mField.modify_finite_element_add_field_data(fe_name,
                                                         thermal_field_name);
      if (mField.check_field(mesh_nodals_positions)) {
        CHKERR mField.modify_finite_element_add_field_data(
            fe_name, mesh_nodals_positions);
      }
      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
               mField, BLOCKSET | MAT_ELASTICSET, it)) {
        Mat_Elastic mydata;
        CHKERR it->getAttributeDataStructure(mydata);
        setOfBlocks[it->getMeshsetId()].youngModulus = mydata.data.Young;
        setOfBlocks[it->getMeshsetId()].poissonRatio = mydata.data.Poisson;
        setOfBlocks[it->getMeshsetId()].thermalExpansion =
            mydata.data.ThermalExpansion;
        CHKERR mField.get_moab().get_entities_by_type(
            it->meshset, MBTET, setOfBlocks[it->getMeshsetId()].tEts, true);
        CHKERR mField.add_ents_to_finite_element_by_type(
            setOfBlocks[it->getMeshsetId()].tEts, MBTET, fe_name);
        double ref_temp;
        PetscBool flg;
        CHKERR PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-my_ref_temp",
                                   &ref_temp, &flg);
        if (flg == PETSC_TRUE) {
          PetscPrintf(mField.get_comm(), "set refernce temperature %3.2f\n",
                      ref_temp);
          setOfBlocks[it->getMeshsetId()].refTemperature = ref_temp;
        }
      }
    }
    MoFEMFunctionReturn(0);
  }

  /// \deprecated Do not use this fiction with spelling mistake
  DEPRECATED inline MoFEMErrorCode addThermalSterssElement(
      const std::string fe_name, const std::string field_name,
      const std::string thermal_field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS") {
    return addThermalStressElement(fe_name, field_name, thermal_field_name,
                                   mesh_nodals_positions);
  }

  MoFEMErrorCode setThermalStressRhsOperators(string field_name,
                                              string thermal_field_name, Vec &F,
                                              int verb = 0) {
    MoFEMFunctionBegin;
    if (mField.check_field(thermal_field_name)) {
      std::map<int, BlockData>::iterator sit = setOfBlocks.begin();
      for (; sit != setOfBlocks.end(); sit++) {
        // add finite elemen
        feThermalStressRhs.getOpPtrVector().push_back(
            new OpGetTemperatureAtGaussPts(thermal_field_name, commonData,
                                           verb));
        feThermalStressRhs.getOpPtrVector().push_back(new OpThermalStressRhs(
            field_name, F, sit->second, commonData, verb));
      }
    }
    MoFEMFunctionReturn(0);
  }
};

#endif //__THERMALSTRESSELEMENT_HPP
