/**
 * \file NavierStokesOperators.hpp
 * \example NavierStokesOperators.hpp
 *
 * \brief Implementation of operators for a finite element 
 * for Navier-Stokes equations
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

#ifndef __NAVIERSTOKESOPERATORS_HPP__
#define __NAVIERSTOKESOPERATORS_HPP__

struct BlockData {
  int iD;
  int oRder;
  double fluidViscosity;
  double fluidDensity;
  Range tEts;
  BlockData() : oRder(-1), fluidViscosity(-1), fluidDensity(-2) {}
};

struct DataAtIntegrationPts {

  boost::shared_ptr<MatrixDouble> gradDispPtr;
  boost::shared_ptr<VectorDouble> pPtr;

  double fluidViscosity;
  double fluidDensity;

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

    fluidViscosity = data.fluidViscosity;
    fluidDensity = data.fluidDensity;

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
        if (bit->getName().compare(0, 5, "FLUID") == 0) { 
          const int id = bit->getMeshsetId();
          CHKERR mField.get_moab().get_entities_by_type(
              bit->getMeshset(), MBTET, setOfBlocksData[id].tEts, true);

          std::vector<double> attributes;
          bit->getAttributes(attributes);
          if (attributes.size() != 2) {
            SETERRQ1(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                      "should be 2 attributes but is %d", attributes.size());
          }
          setOfBlocksData[id].iD = id;
          setOfBlocksData[id].fluidViscosity = attributes[0];
          setOfBlocksData[id].fluidDensity = attributes[1];
        }
      }
   


    MoFEMFunctionReturn(0);
  }

private:
  MoFEM::Interface &mField;
};

/**
 * \brief Set integration rule to volume elements
 *
 * This rule is used to integrate \f$\nabla v \cdot \nabla u\f$, thus
 * if the approximation field and the testing field are polynomials of order "p",
 * then the rule for the exact integration is 2*(p-1).
 *
 * Integration rule is order of polynomial which is calculated exactly. Finite
 * element selects integration method based on return of this function.
 *
 */
struct VolRule {
  int operator()(int order_row, int order_col, int order_data) const {
     return order_row < order_col ? 2 * (order_col - 1) 
                       : 2 * (order_row - 1);
     //return 2 * (p - 1); 
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

    //const double lambda = commonData.lAmbda;
    //const double mu = commonData.mU;

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

        auto base_functions = col_data.getFTensor0N(gg, 0);
        for (int col_bb = 0; col_bb != col_nb_dofs; col_bb++) {

          FTensor::Tensor1<double *, 3> k(&locG(3 * row_bb + 0, col_bb),
                                          &locG(3 * row_bb + 1, col_bb),
                                          &locG(3 * row_bb + 2, col_bb));

          k(i) -= t1(i) * base_functions;
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
      PetscPrintf(PETSC_COMM_WORLD, "PROBLEM");
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

    //const double mu = commonData.mU;
    //const double lambda = commonData.lAmbda;

    // integrate local matrix for entity block
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {
    
      //const MatrixAdaptor &diffN = row_data.getDiffN(gg,nb_dofs/3);
         
      // Get volume and integration weight
      double w = getVolume() * getGaussPts()(3, gg) * commonData.fluidViscosity;
      //int itest = getHoGaussPtsDetJac().size();
      if (getHoGaussPtsDetJac().size() > 0) {
         //double test = getHoGaussPtsDetJac()[gg];
         w *= getHoGaussPtsDetJac()[gg]; ///< higher order geometry
      }

      int row_bb = 0;
      for (; row_bb != row_nb_dofs / 3; row_bb++) {

        auto col_diff_base_functions = col_data.getFTensor1DiffN<3>(gg, 0);
        const int final_bb = diagonal_block ? row_bb + 1 : col_nb_dofs / 3;
        int col_bb = 0;

        //MatrixDouble mat = col_data.getDiffN();

        for (; col_bb != final_bb; col_bb++) {

          auto t_assemble = get_tensor2(locK, 3 * row_bb, 3 * col_bb);
          
          for (int i : {0,1,2}) {
            for (int j : {0,1,2}) {
              t_assemble(i, i) += w * row_diff_base_functions(j) * col_diff_base_functions(j);
              t_assemble(i, j) += w * row_diff_base_functions(j) * col_diff_base_functions(i);
            }            
          } 

          // Next base function for column
          ++col_diff_base_functions;
        }

        // Next base function for row
        ++row_diff_base_functions;
      }
      for (; row_bb != row_nb_base_functions; row_bb++) {
        ++row_diff_base_functions;
      }
    }

    if (diagonal_block) {
      FTensor::Index<'i', 3> i;
      //FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      //FTensor::Index<'l', 3> l;
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

// struct OpPostProcStress
//     : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {
//   DataAtIntegrationPts &commonData;
//   moab::Interface &postProcMesh;
//   std::vector<EntityHandle> &mapGaussPts;
//   BlockData &dAta;

//   OpPostProcStress(moab::Interface &post_proc_mesh,
//                    std::vector<EntityHandle> &map_gauss_pts,
//                    DataAtIntegrationPts &common_data, BlockData &data)
//       : VolumeElementForcesAndSourcesCore::UserDataOperator(
//             "U", UserDataOperator::OPROW),
//         commonData(common_data), postProcMesh(post_proc_mesh),
//         mapGaussPts(map_gauss_pts), dAta(data) {
//     doVertices = true;
//     doEdges = false;
//     doQuads = false;
//     doTris = false;
//     doTets = false;
//     doPrisms = false;
//   }

//   PetscErrorCode doWork(int side, EntityType type,
//                         DataForcesAndSourcesCore::EntData &data) {
//     MoFEMFunctionBegin;
//     if (type != MBVERTEX)
//       PetscFunctionReturn(9);
//     double def_VAL[9];
//     bzero(def_VAL, 9 * sizeof(double));

//     if (dAta.tEts.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
//         dAta.tEts.end()) {
//       MoFEMFunctionReturnHot(0);
//     }
//     commonData.getBlockData(dAta);

//     Tag th_stress;
//     CHKERR postProcMesh.tag_get_handle("STRESS", 9, MB_TYPE_DOUBLE, th_stress,
//                                        MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
//     Tag th_strain;
//     CHKERR postProcMesh.tag_get_handle("STRAIN", 9, MB_TYPE_DOUBLE, th_strain,
//                                        MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);
//     Tag th_psi;
//     CHKERR postProcMesh.tag_get_handle("ENERGY", 1, MB_TYPE_DOUBLE, th_psi,
//                                        MB_TAG_CREAT | MB_TAG_SPARSE, def_VAL);

//     auto grad = getFTensor2FromMat<3, 3>(*commonData.gradDispPtr);
//     auto p = getFTensor0FromVec(*commonData.pPtr);

//     const int nb_gauss_pts = commonData.gradDispPtr->size2();
//     const int nb_gauss_pts2 = commonData.pPtr->size();

//     //const double lambda = commonData.lAmbda;
//     //const double mu = commonData.mU;

//     FTensor::Index<'i', 3> i;
//     FTensor::Index<'j', 3> j;
//     FTensor::Tensor2<double, 3, 3> strain;
//     FTensor::Tensor2<double, 3, 3> stress;

//     for (int gg = 0; gg != nb_gauss_pts; gg++) {
//       strain(i, j) = 0.5 * (grad(i, j) + grad(j, i));
//       double trace = strain(i, i);
//       double psi = 0.5 * p * p + mu * strain(i, j) * strain(i, j);

//       stress(i, j) = 2 * mu * strain(i, j);
//       stress(1, 1) -= p;
//       stress(0, 0) -= p;
//       stress(2, 2) -= p;

//       CHKERR postProcMesh.tag_set_data(th_psi, &mapGaussPts[gg], 1, &psi);
//       CHKERR postProcMesh.tag_set_data(th_strain, &mapGaussPts[gg], 1,
//                                        &strain(0, 0));
//       CHKERR postProcMesh.tag_set_data(th_stress, &mapGaussPts[gg], 1,
//                                        &stress(0, 0));
//       ++p;
//       ++grad;
//     }

//     MoFEMFunctionReturn(0);
//   }
// };

#endif //__NAVIERSTOKESOPERATORS_HPP__
