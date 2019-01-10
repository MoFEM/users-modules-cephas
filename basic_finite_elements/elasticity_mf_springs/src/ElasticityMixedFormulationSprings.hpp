/**
 * \file ElasticityMixedFormulationSprings.hpp
 * \example ElasticityMixedFormulationSprings.hpp
 *
 * \brief Operator implementation of U-P (mixed) finite element with springs.
 *
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

#ifndef __ELASTICITYMIXEDFORMULATIONSPRINGS_HPP__
#define __ELASTICITYMIXEDFORMULATIONSPRINGS_HPP__

struct BlockData {
  int iD;
  int oRder;
  double yOung;
  double pOisson;

  double springStiffness0; // Spring stiffness
  double springStiffness1;
  double springStiffness2;

  Range tEts;
  Range tRis;   //TODO: Should generalise for edges and vertices
  BlockData()
      : oRder(-1), yOung(-1), pOisson(-2), springStiffness0(-1),
        springStiffness1(-1), springStiffness2(-1) {}
};
struct DataAtIntegrationPts {

  boost::shared_ptr<MatrixDouble> gradDispPtr;
  boost::shared_ptr<VectorDouble> pPtr;
  FTensor::Ddg<double, 3, 3> tD;

  double pOisson;
  double yOung;
  double lAmbda;
  double mU;

  double springStiffness0; // Spring stiffness
  double springStiffness1;
  double springStiffness2;

  std::map<int, BlockData> mapElastic;
  std::map<int, BlockData> mapSpring;

  DataAtIntegrationPts(MoFEM::Interface &m_field) : mField(m_field) {

    // Setting default values for coeffcients
    gradDispPtr = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    pPtr = boost::shared_ptr<VectorDouble>(new VectorDouble());

    ierr = setBlocks();
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  MoFEMErrorCode getParameters() {
    MoFEMFunctionBegin; // They will be overwriten by BlockData
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Problem", "none");

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode getBlockData(BlockData &data) {
    MoFEMFunctionBegin;

    yOung = data.yOung;
    pOisson = data.pOisson;
    lAmbda = (yOung * pOisson) / ((1. + pOisson) * (1. - 2. * pOisson));
    mU = yOung / (2. * (1. + pOisson));

    springStiffness0 = data.springStiffness0;
    springStiffness1 = data.springStiffness1;
    springStiffness2 = data.springStiffness2;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Index<'l', 3> l;

    tD(i, j, k, l) = 0.;

    tD(0, 0, 0, 0) = 2 * mU;
    tD(0, 1, 0, 1) = mU;
    tD(0, 1, 1, 0) = mU;
    tD(0, 2, 0, 2) = mU;
    tD(0, 2, 2, 0) = mU;
    tD(1, 0, 0, 1) = mU;
    tD(1, 0, 1, 0) = mU;
    tD(1, 1, 1, 1) = 2 * mU;
    tD(1, 2, 1, 2) = mU;
    tD(1, 2, 2, 1) = mU;
    tD(2, 0, 0, 2) = mU;
    tD(2, 0, 2, 0) = mU;
    tD(2, 1, 1, 2) = mU;
    tD(2, 1, 2, 1) = mU;
    tD(2, 2, 2, 2) = 2 * mU;

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             mField, BLOCKSET | MAT_ELASTICSET, it)) {
      Mat_Elastic mydata;
      CHKERR it->getAttributeDataStructure(mydata);
      int id = it->getMeshsetId();
      EntityHandle meshset = it->getMeshset();
      CHKERR mField.get_moab().get_entities_by_type(
          meshset, MBTET, mapElastic[id].tEts, true);
      mapElastic[id].iD = id;
      mapElastic[id].yOung = mydata.data.Young;
      mapElastic[id].pOisson = mydata.data.Poisson;
    }

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {
      
        //CHKERR bit->getAttributeDataStructure(mydata);
        const int id = bit->getMeshsetId();
        CHKERR mField.get_moab().get_entities_by_type(
            bit->getMeshset(), MBTRI, mapSpring[id].tRis, true);
        
        EntityHandle out_meshset;
        CHKERR mField.get_moab().create_meshset(MESHSET_SET, out_meshset);
        CHKERR mField.get_moab().add_entities(out_meshset, mapSpring[id].tRis);
        CHKERR mField.get_moab().write_file("error.vtk", "VTK", "",
                                            &out_meshset, 1);

        std::vector<double> attributes;
        bit->getAttributes(attributes);
        if (attributes.size() != 3) {
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                   "should be 3 attributes but is %d", attributes.size());
        }
        mapSpring[id].iD = id;
        mapSpring[id].springStiffness0 = attributes[0];
        mapSpring[id].springStiffness1 = attributes[1];
        mapSpring[id].springStiffness2 = attributes[2];
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MoFEM::Interface &mField;
};

/** * @brief Assemble P *
 * \f[
 * {\bf{P}} =  - \int\limits_\Omega  {{\bf{N}}_p^T\frac{1}{\lambda
        }{{\bf{N}}_p}d\Omega }
 * \f]
 *
 */
struct OpAssembleP
    : public VolumeElementForcesAndSourcesCore::UserDataOperator {

  DataAtIntegrationPts &commonData;
  MatrixDouble locP;
  MatrixDouble translocP;
  BlockData &dAta;

  OpAssembleP(DataAtIntegrationPts &common_data, BlockData &data)
      : VolumeElementForcesAndSourcesCore::UserDataOperator(
            "P", "P", UserDataOperator::OPROWCOL),
        commonData(common_data), dAta(data) {
    sYmm = true;
  }

  PetscErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data) {

    MoFEMFunctionBegin;
    const int row_nb_dofs = row_data.getIndices().size();
    if (!row_nb_dofs)
      MoFEMFunctionReturnHot(0);
    const int col_nb_dofs = col_data.getIndices().size();
    if (!col_nb_dofs)
      MoFEMFunctionReturnHot(0);

    if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tEts.end()) {
      MoFEMFunctionReturnHot(0);
    }
    CHKERR commonData.getBlockData(dAta);
    // Set size can clear local tangent matrix
    locP.resize(row_nb_dofs, col_nb_dofs, false);
    locP.clear();

    const int row_nb_gauss_pts = row_data.getN().size1();
    if (!row_nb_gauss_pts)
      MoFEMFunctionReturnHot(0);
    const int row_nb_base_functions = row_data.getN().size2();
    auto row_base_functions = row_data.getFTensor0N();

    // get data
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    const double lambda = commonData.lAmbda;
    const double mu = commonData.mU;
    // const double st1 = commonData.springStiffness0;
    // const double st2 = commonData.springStiffness1;

    double coefficient = commonData.pOisson == 0.5 ? 0. : 1 / lambda;

    // Print test
    // std::cout << "Value of string stiffness: " << st1 + st2 << endl;

    // integration
    if (coefficient != 0.) {
      for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

        // Get volume and integration weight
        double w = getVolume() * getGaussPts()(3, gg);
        if (getHoGaussPtsDetJac().size() > 0) {
          w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
        }

        // INTEGRATION
        int row_bb = 0;
        for (; row_bb != row_nb_dofs; row_bb++) {
          auto col_base_functions = col_data.getFTensor0N(gg, 0);
          for (int col_bb = 0; col_bb != col_nb_dofs; col_bb++) {

            locP(row_bb, col_bb) -=
                w * row_base_functions * col_base_functions * coefficient;

            ++col_base_functions;
          }
          ++row_base_functions;
        }
        for (; row_bb != row_nb_base_functions; row_bb++) {
          ++row_base_functions;
        }
      }
    }
    

    CHKERR MatSetValues(
        getFEMethod()->ksp_B, row_nb_dofs, &*row_data.getIndices().begin(),
        col_nb_dofs, &*col_data.getIndices().begin(), &locP(0, 0), ADD_VALUES);

    // is symmetric
    if (row_side != col_side || row_type != col_type) {
      translocP.resize(col_nb_dofs, row_nb_dofs, false);
      noalias(translocP) = trans(locP);
      CHKERR MatSetValues(getFEMethod()->ksp_B, col_nb_dofs,
                          &*col_data.getIndices().begin(), row_nb_dofs,
                          &*row_data.getIndices().begin(), &translocP(0, 0),
                          ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }
};

/** * @brief Assemble G *
 * \f[
 * {\bf{G}} =  - \int\limits_\Omega  {{{\bf{B}}^T}{\bf m}{{\bf{N}}_p}d\Omega }
 * \f]
 *
 */
struct OpAssembleG
    : public VolumeElementForcesAndSourcesCore::UserDataOperator {

  DataAtIntegrationPts &commonData;
  MatrixDouble locG;
  BlockData &dAta;

  OpAssembleG(DataAtIntegrationPts &common_data, BlockData &data)
      : VolumeElementForcesAndSourcesCore::UserDataOperator(
            "U", "P", UserDataOperator::OPROWCOL),
        commonData(common_data), dAta(data) {
    sYmm = false;
  }

  PetscErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data) {

    MoFEMFunctionBegin;

    const int row_nb_dofs = row_data.getIndices().size();
    if (!row_nb_dofs)
      MoFEMFunctionReturnHot(0);
    const int col_nb_dofs = col_data.getIndices().size();
    if (!col_nb_dofs)
      MoFEMFunctionReturnHot(0);

    if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tEts.end()) {
      MoFEMFunctionReturnHot(0);
    }
    commonData.getBlockData(dAta);

    // Set size can clear local tangent matrix
    locG.resize(row_nb_dofs, col_nb_dofs, false);
    locG.clear();
    const int row_nb_gauss_pts = row_data.getN().size1();
    if (!row_nb_gauss_pts)
      MoFEMFunctionReturnHot(0);
    const int row_nb_base_functions = row_data.getN().size2();
    auto row_diff_base_functions = row_data.getFTensor1DiffN<3>();

    const double lambda = commonData.lAmbda;
    const double mu = commonData.mU;

    FTensor::Tensor1<double, 3> t1;
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    // INTEGRATION
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

      // Get volume and integration weight
      double w = getVolume() * getGaussPts()(3, gg);
      if (getHoGaussPtsDetJac().size() > 0) {
        w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
      }

      int row_bb = 0;
      for (; row_bb != row_nb_dofs / 3; row_bb++) {

        t1(i) = w * row_diff_base_functions(i);

        auto col_base_functions = col_data.getFTensor0N(gg, 0);
        for (int col_bb = 0; col_bb != col_nb_dofs; col_bb++) {

          FTensor::Tensor1<double *, 3> k(&locG(3 * row_bb + 0, col_bb),
                                          &locG(3 * row_bb + 1, col_bb),
                                          &locG(3 * row_bb + 2, col_bb));

          k(i) += t1(i) * col_base_functions;
          ++col_base_functions;
        }
        ++row_diff_base_functions;
      }
      for (; row_bb != row_nb_base_functions; row_bb++) {
        ++row_diff_base_functions;
      }
    }

    CHKERR MatSetValues(getFEMethod()->ksp_B, row_nb_dofs,
                        &*row_data.getIndices().begin(), col_nb_dofs,
                        &*col_data.getIndices().begin(), &*locG.data().begin(),
                        ADD_VALUES);

    // ASSEMBLE THE TRANSPOSE
    locG = trans(locG);
    CHKERR MatSetValues(getFEMethod()->ksp_B, col_nb_dofs,
                        &*col_data.getIndices().begin(), row_nb_dofs,
                        &*row_data.getIndices().begin(), &*locG.data().begin(),
                        ADD_VALUES);
    MoFEMFunctionReturn(0);
  }
};

/** * @brief Assemble K *
 * \f[
 * {\bf{K}} = \int\limits_\Omega  {{{\bf{B}}^T}{{\bf{D}}_d}{\bf{B}}d\Omega }
 * \f]
 *
 */
struct OpAssembleK
    : public VolumeElementForcesAndSourcesCore::UserDataOperator {

  MatrixDouble locK;
  MatrixDouble translocK;
  BlockData &dAta;
  FTensor::Tensor2<double, 3, 3> diffDiff;

  DataAtIntegrationPts &commonData;

  OpAssembleK(DataAtIntegrationPts &common_data, BlockData &data, bool symm = true)
      : VolumeElementForcesAndSourcesCore::UserDataOperator("U", "U", OPROWCOL,
                                                            symm),
        commonData(common_data), dAta(data) {}

  PetscErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;

    auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(r + 0, c + 0), &m(r + 0, c + 1), &m(r + 0, c + 2),
          &m(r + 1, c + 0), &m(r + 1, c + 1), &m(r + 1, c + 2),
          &m(r + 2, c + 0), &m(r + 2, c + 1), &m(r + 2, c + 2));
    };

    const int row_nb_dofs = row_data.getIndices().size();
    if (!row_nb_dofs)
      MoFEMFunctionReturnHot(0);
    const int col_nb_dofs = col_data.getIndices().size();
    if (!col_nb_dofs)
      MoFEMFunctionReturnHot(0);
    if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tEts.end()) {
      MoFEMFunctionReturnHot(0);
    }
    commonData.getBlockData(dAta);

    const bool diagonal_block =
        (row_type == col_type) && (row_side == col_side);
    // get number of integration points
    // Set size can clear local tangent matrix
    locK.resize(row_nb_dofs, col_nb_dofs, false);
    locK.clear();

    const int row_nb_gauss_pts = row_data.getN().size1();
    const int row_nb_base_functions = row_data.getN().size2();

    auto row_diff_base_functions = row_data.getFTensor1DiffN<3>();

    const double mu = commonData.mU;
    const double lambda = commonData.lAmbda;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Index<'l', 3> l;

    // integrate local matrix for entity block
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

      // Get volume and integration weight
      double w = getVolume() * getGaussPts()(3, gg);
      if (getHoGaussPtsDetJac().size() > 0) {
        w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
      }

      int row_bb = 0;
      for (; row_bb != row_nb_dofs / 3; row_bb++) {

        auto col_diff_base_functions = col_data.getFTensor1DiffN<3>(gg, 0);
        const int final_bb = diagonal_block ? row_bb + 1 : col_nb_dofs / 3;
        int col_bb = 0;
        for (; col_bb != final_bb; col_bb++) {

          auto t_assemble = get_tensor2(locK, 3 * row_bb, 3 * col_bb);

          diffDiff(j, l) =
              w * row_diff_base_functions(j) * col_diff_base_functions(l);

          t_assemble(i, k) += diffDiff(j, l) * commonData.tD(i, j, k, l);
          // Next base function for column
          ++col_diff_base_functions;
        }

        ++row_diff_base_functions;
      }
      for (; row_bb != row_nb_base_functions; row_bb++) {
        ++row_diff_base_functions;
      }
    }

    if (diagonal_block) {
      for (int row_bb = 0; row_bb != row_nb_dofs / 3; row_bb++) {
        int col_bb = 0;
        for (; col_bb != row_bb + 1; col_bb++) {
          auto t_assemble = get_tensor2(locK, 3 * row_bb, 3 * col_bb);
          auto t_off_side = get_tensor2(locK, 3 * col_bb, 3 * row_bb);
          t_off_side(i, k) = t_assemble(k, i);
        }
      }
    }

    const int *row_ind = &*row_data.getIndices().begin();
    const int *col_ind = &*col_data.getIndices().begin();
    Mat B = getFEMethod()->ksp_B != PETSC_NULL ? getFEMethod()->ksp_B
                                               : getFEMethod()->ksp_B;
    CHKERR MatSetValues(B, row_nb_dofs, row_ind, col_nb_dofs, col_ind,
                        &locK(0, 0), ADD_VALUES);

    if (row_type != col_type || row_side != col_side) {
      translocK.resize(col_nb_dofs, row_nb_dofs, false);
      noalias(translocK) = trans(locK);
      CHKERR MatSetValues(B, col_nb_dofs, col_ind, row_nb_dofs, row_ind,
                          &translocK(0, 0), ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }
};

struct OpSpringKs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  DataAtIntegrationPts &commonData;
  MatrixDouble locKs;
  MatrixDouble transLocKs;
  BlockData &dAta;

  OpSpringKs(DataAtIntegrationPts &common_data, BlockData &data)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
            "U", "U", OPROWCOL),
        commonData(common_data), dAta(data) {
    sYmm = true;
  }

  MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        DataForcesAndSourcesCore::EntData &row_data,
                        DataForcesAndSourcesCore::EntData &col_data) {
    MoFEMFunctionBegin;
    
    // check if the volumes have associated degrees of freedom
    const int row_nb_dofs = row_data.getIndices().size();
    if (!row_nb_dofs)
      MoFEMFunctionReturnHot(0);

    const int col_nb_dofs = col_data.getIndices().size();
    if (!col_nb_dofs)
      MoFEMFunctionReturnHot(0);

    std::cout << dAta.tRis << endl;
    if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tRis.end()) {
      MoFEMFunctionReturnHot(0);  // TODO:This never gets executed
    }
    // std::cout << "End: " << dAta.tRis.end() << endl;

    CHKERR commonData.getBlockData(dAta);
    // size associated to the entity
    locKs.resize(row_nb_dofs, col_nb_dofs, false);
    locKs.clear();

    // get number of Gauss points
    const int row_nb_gauss_pts = row_data.getN().size1();
    if (!row_nb_gauss_pts) // check if number of Gauss point <> 0
      MoFEMFunctionReturnHot(0);

    const int row_nb_base_functions = row_data.getN().size2();
    auto row_base_functions = row_data.getFTensor0N();

    vector <double> spring_stiffness;    // spring_stiffness[0]
    spring_stiffness.push_back(commonData.springStiffness0);
    spring_stiffness.push_back(commonData.springStiffness1);
    spring_stiffness.push_back(commonData.springStiffness2);
    // FTensor::Tensor1<double, 3> spring_stiffness;   //spring_stiffness(0)
    // spring_stiffness(0) = commonData.springStiffness0;
    // spring_stiffness(1) = commonData.springStiffness1;
    // spring_stiffness(2) = commonData.springStiffness2;


    // FTensor::Tensor1<double, 3> t1;
    FTensor::Index<'i', 3> i;
    // FieldSpace space;

    // loop over all Gauss point of the volume
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {
      // get area and integration weight
      double w = getArea() * getGaussPts()(2, gg);
      // TODO: w includes Jacobian due to OpSetInvJacH1ForFace(inv_jac)?
      
      for (int row_index = 0; row_index != row_nb_dofs / 3; row_index++) {
        // t1(i) = w * row_base_functions(i) * spring_stiffness(i);
        auto col_base_functions = col_data.getFTensor0N(gg, 0);
        for (int col_index = 0; col_index != col_nb_dofs / 3; col_index++) {
          locKs(row_index, col_index) += w * row_base_functions *
                                         spring_stiffness[col_index % 3] *
                                         col_base_functions;
          ++col_base_functions;
        }
        ++row_base_functions;
      }
    }

    // Add computed values of spring stiffness to the global LHS matrix
    CHKERR MatSetValues(
        getFEMethod()->ksp_B, row_nb_dofs, &*row_data.getIndices().begin(),
        col_nb_dofs, &*col_data.getIndices().begin(), &locKs(0, 0), ADD_VALUES);

    // is symmetric
    if (row_side != col_side || row_type != col_type) {
      transLocKs.resize(col_nb_dofs, row_nb_dofs, false);
      noalias(transLocKs) = trans(locKs);
      CHKERR MatSetValues(getFEMethod()->ksp_B, col_nb_dofs,
                          &*col_data.getIndices().begin(), row_nb_dofs,
                          &*row_data.getIndices().begin(), &transLocKs(0, 0),
                          ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }
};

struct OpSpringFs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  DataAtIntegrationPts &commonData;
  BlockData &dAta;

  OpSpringFs(DataAtIntegrationPts &common_data, BlockData &data)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
            "U", OPROW),
        commonData(common_data), dAta(data) {}

  // vector used to store force vector for each degree of freedom
  VectorDouble nF;

  FTensor::Index<'i', 3> i;

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {

    MoFEMFunctionBegin;
    // check that the faces have associated degrees of freedom
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs == 0)
      MoFEMFunctionReturnHot(0);

    std::cout << dAta.tRis << endl;
    if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tRis.end()) {
      MoFEMFunctionReturnHot(0);
    }

    // size of force vector associated to the entity
    // set equal to the number of degrees of freedom of associated with the
    // entity
    nF.resize(nb_dofs, false);
    nF.clear();

    // get number of gauss points
    const int nb_gauss_pts = data.getN().size1();

    // create a 3d vector to be used as the normal to the face with length equal
    // to the face area
    auto t_normal = getFTensor1Normal();

    // get intergration weights
    auto t_w = getFTensor0IntegrationWeight();

    // vector of base functions
    auto base_functions = data.getFTensor0N();

    // get spring stiffness
    vector<double> spring_stiffness; // spring_stiffness[0]
    spring_stiffness.push_back(commonData.springStiffness0);
    spring_stiffness.push_back(commonData.springStiffness1);
    spring_stiffness.push_back(commonData.springStiffness2);

    // ***** double val = data.getFieldData()[dd];

    // loop over all gauss points of the face
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      // weight of gg gauss point
      double w = 0.5 * t_w;
      // FIXME: w includes Jacobian due to OpCalculateInvJacForFace()?

      // create a vector t_nf whose pointer points an array of 3 pointers
      // pointing to nF  memory location of components
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf(&nF[0], &nF[1],
                                                              &nF[2]);
      for (int col_index = 0; col_index != nb_dofs / 3; ++col_index) {
        // scale the three components of t_normal and pass them to the t_nf
        // (hence to nF)
        // Here: *u
    // FTensor::Tensor0<double *> t_field_data_slave(&data.getFieldData()[3]);
    // FIXME: Operator for shape functions at Gauss point & nodal solutions
        t_nf(i) += (w * spring_stiffness[col_index % 3] * base_functions) *
                   t_normal(i);
        // move the pointer to next element of t_nf
        ++t_nf;
        // move to next base function
        ++base_functions;
      }

      // move to next integration weight
      ++t_w;
    }

    // add computed values of pressure in the global right hand side vector
    CHKERR VecSetValues(getFEMethod()->ksp_f, nb_dofs, &data.getIndices()[0],
                        &nF[0], ADD_VALUES);

    MoFEMFunctionReturn(0);
  }
};

struct OpPostProcStress
    : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {
  DataAtIntegrationPts &commonData;
  moab::Interface &postProcMesh;
  std::vector<EntityHandle> &mapGaussPts;
  BlockData &dAta;

  OpPostProcStress(moab::Interface &post_proc_mesh,
                   std::vector<EntityHandle> &map_gauss_pts,
                   DataAtIntegrationPts &common_data, BlockData &data)
      : VolumeElementForcesAndSourcesCore::UserDataOperator(
            "U", UserDataOperator::OPROW),
        commonData(common_data), postProcMesh(post_proc_mesh),
        mapGaussPts(map_gauss_pts), dAta(data) {
    doVertices = true;
    doEdges = false;
    doQuads = false;
    doTris = false;
    doTets = false;
    doPrisms = false;
  }

  PetscErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {
    MoFEMFunctionBegin;
    if (type != MBVERTEX)
      PetscFunctionReturn(9);
    double def_VAL[9];
    bzero(def_VAL, 9 * sizeof(double));

    if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tEts.end()) {
      MoFEMFunctionReturnHot(0);
    }
    commonData.getBlockData(dAta);

    Tag th_stress;
    CHKERR postProcMesh.tag_get_handle("STRESS", 9, MB_TYPE_DOUBLE, th_stress,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    Tag th_strain;
    CHKERR postProcMesh.tag_get_handle("STRAIN", 9, MB_TYPE_DOUBLE, th_strain,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
    Tag th_psi;
    CHKERR postProcMesh.tag_get_handle("ENERGY", 1, MB_TYPE_DOUBLE, th_psi,
                                       MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

    auto grad = getFTensor2FromMat<3, 3>(*commonData.gradDispPtr);
    auto p = getFTensor0FromVec(*commonData.pPtr);

    const int nb_gauss_pts = commonData.gradDispPtr->size2();
    const int nb_gauss_pts2 = commonData.pPtr->size();

    const double lambda = commonData.lAmbda;
    const double mu = commonData.mU;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Tensor2<double, 3, 3> strain;
    FTensor::Tensor2<double, 3, 3> stress;

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      strain(i, j) = 0.5 * (grad(i, j) + grad(j, i));
      double trace = strain(i, i);
      double psi = 0.5 * p * p + mu * strain(i, j) * strain(i, j);

      stress(i, j) = 2 * mu * strain(i, j);
      stress(1, 1) -= p;
      stress(0, 0) -= p;
      stress(2, 2) -= p;

      CHKERR postProcMesh.tag_set_data(th_psi, &mapGaussPts[gg], 1, &psi);
      CHKERR postProcMesh.tag_set_data(th_strain, &mapGaussPts[gg], 1,
                                       &strain(0, 0));
      CHKERR postProcMesh.tag_set_data(th_stress, &mapGaussPts[gg], 1,
                                       &stress(0, 0));
      ++p;
      ++grad;
    }

    MoFEMFunctionReturn(0);
  }
};

#endif //__ELASTICITYMIXEDFORMULATION_HPP__
