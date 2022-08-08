/**
 * \file ElasticityMixedFormulation.hpp
 * \example ElasticityMixedFormulation.hpp
 *
 * \brief Operator implementation of U-P (mixed) finite element.
 *
 */

#ifndef __ELASTICITYMIXEDFORMULATION_HPP__
#define __ELASTICITYMIXEDFORMULATION_HPP__

struct BlockData {
  int iD;
  int oRder;
  double yOung;
  double pOisson;
  Range tEts;
  BlockData() : oRder(-1), yOung(-1), pOisson(-2) {}
};
struct DataAtIntegrationPts {

  boost::shared_ptr<MatrixDouble> gradDispPtr;
  boost::shared_ptr<VectorDouble> pPtr;
  FTensor::Ddg<double, 3, 3> tD;

  double pOisson;
  double yOung;
  double lAmbda;
  double mU;

  std::map<int, BlockData> setOfBlocksData;

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
          meshset, MBTET, setOfBlocksData[id].tEts, true);
      setOfBlocksData[id].iD = id;
      setOfBlocksData[id].yOung = mydata.data.Young;
      setOfBlocksData[id].pOisson = mydata.data.Poisson;
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
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {

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

    double coefficient = commonData.pOisson == 0.5 ? 0. : 1 / lambda;

    // integration
    if (coefficient != 0.) {
      for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

        // Get volume and integration weight
        double w = getVolume() * getGaussPts()(3, gg);

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
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {

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

    FTensor::Tensor1<double, 3> t1;
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;

    // INTEGRATION
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

      // Get volume and integration weight
      double w = getVolume() * getGaussPts()(3, gg);

      int row_bb = 0;
      for (; row_bb != row_nb_dofs / 3; row_bb++) {

        t1(i) = w * row_diff_base_functions(i);

        auto base_functions = col_data.getFTensor0N(gg, 0);
        for (int col_bb = 0; col_bb != col_nb_dofs; col_bb++) {

          FTensor::Tensor1<double *, 3> k(&locG(3 * row_bb + 0, col_bb),
                                          &locG(3 * row_bb + 1, col_bb),
                                          &locG(3 * row_bb + 2, col_bb));

          k(i) += t1(i) * base_functions;
          ++base_functions;
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
  FTensor::Tensor2<double, 3, 3> diffDiff;

  DataAtIntegrationPts &commonData;
  BlockData &dAta;

  OpAssembleK(DataAtIntegrationPts &common_data, BlockData &data,
              bool symm = true)
      : VolumeElementForcesAndSourcesCore::UserDataOperator("U", "U", OPROWCOL,
                                                            symm),
        commonData(common_data), dAta(data) {}

  PetscErrorCode doWork(int row_side, int col_side, EntityType row_type,
                        EntityType col_type,
                        EntitiesFieldData::EntData &row_data,
                        EntitiesFieldData::EntData &col_data) {
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

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    FTensor::Index<'l', 3> l;

    // integrate local matrix for entity block
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {

      // Get volume and integration weight
      double w = getVolume() * getGaussPts()(3, gg);

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
                        EntitiesFieldData::EntData &data) {
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
    const double mu = commonData.mU;

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Tensor2<double, 3, 3> strain;
    FTensor::Tensor2<double, 3, 3> stress;

    for (int gg = 0; gg != nb_gauss_pts; gg++) {
      strain(i, j) = 0.5 * (grad(i, j) + grad(j, i));
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
