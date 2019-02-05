/** \file SpringElements.hpp
  \ingroup Header file for spring elements implementation
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

#ifndef __SPRINGELEMENT_HPP__
#define __SPRINGELEMENT_HPP__

struct BlockOptionDataSprings {
  int iD;

  double springStiffness0; // Spring stiffness
  double springStiffness1;
  double springStiffness2;

  Range tRis;

  BlockOptionDataSprings()
      : springStiffness0(-1), springStiffness1(-1), springStiffness2(-1) {}
};

struct DataAtIntegrationPtsSprings {

  boost::shared_ptr<MatrixDouble> gradDispPtr =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> xAtPts =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> xInitAtPts =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());

  double springStiffness0; // Spring stiffness
  double springStiffness1;
  double springStiffness2;

  std::map<int, BlockOptionDataSprings> mapSpring;
  ~DataAtIntegrationPtsSprings(){
  }
  DataAtIntegrationPtsSprings(MoFEM::Interface &m_field)
      : mField(m_field) {

    ierr = setBlocks();
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  MoFEMErrorCode getParameters() {
    MoFEMFunctionBegin; // They will be overwritten by BlockData
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Problem", "none");

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode getBlockData(BlockOptionDataSprings &data) {
    MoFEMFunctionBegin;

    springStiffness0 = data.springStiffness0;
    springStiffness1 = data.springStiffness1;
    springStiffness2 = data.springStiffness2;

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {

        const int id = bit->getMeshsetId();
        CHKERR mField.get_moab().get_entities_by_type(bit->getMeshset(), MBTRI,
                                                      mapSpring[id].tRis, true);

        // EntityHandle out_meshset;
        // CHKERR mField.get_moab().create_meshset(MESHSET_SET, out_meshset);
        // CHKERR mField.get_moab().add_entities(out_meshset,
        // mapSpring[id].tRis); CHKERR mField.get_moab().write_file("error.vtk",
        // "VTK", "",
        //                                     &out_meshset, 1);

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

/** * @brief Assemble contribution of spring to RHS *
 * \f[
 * {K^s} = \int\limits_\Omega ^{} {{\psi ^T}{k_s}\psi d\Omega }
 * \f]
 *
 */
struct OpSpringKs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  DataAtIntegrationPtsSprings &commonData;
  BlockOptionDataSprings &dAta;

  MatrixDouble locKs;
  MatrixDouble transLocKs;

  OpSpringKs(DataAtIntegrationPtsSprings &common_data,
             BlockOptionDataSprings &data)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
            "SPATIAL_POSITION", "SPATIAL_POSITION", OPROWCOL),
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

    if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tRis.end()) {
      MoFEMFunctionReturnHot(0);
    }

    CHKERR commonData.getBlockData(dAta);
    // size associated to the entity
    locKs.resize(row_nb_dofs, col_nb_dofs, false);
    locKs.clear();

    // get number of Gauss points
    const int row_nb_gauss_pts = row_data.getN().size1();
    if (!row_nb_gauss_pts) // check if number of Gauss point <> 0
      MoFEMFunctionReturnHot(0);

    const int row_nb_base_functions = row_data.getN().size2();

    // get intergration weights
    auto t_w = getFTensor0IntegrationWeight();

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(3 * r + 0, 3 * c + 0), &m(3 * r + 0, 3 * c + 1),
          &m(3 * r + 0, 3 * c + 2), &m(3 * r + 1, 3 * c + 0),
          &m(3 * r + 1, 3 * c + 1), &m(3 * r + 1, 3 * c + 2),
          &m(3 * r + 2, 3 * c + 0), &m(3 * r + 2, 3 * c + 1),
          &m(3 * r + 2, 3 * c + 2));
    };

    FTensor::Tensor2<double, 3, 3> linear_spring(
        commonData.springStiffness0, 0., 0., 0., commonData.springStiffness1,
        0., 0., 0., commonData.springStiffness2);

    // loop over the Gauss points
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {
      // get area and integration weight
      double w = t_w * getArea();

      auto row_base_functions = row_data.getFTensor0N(gg, 0);

      for (int row_index = 0; row_index != row_nb_dofs / 3; row_index++) {
        auto col_base_functions = col_data.getFTensor0N(gg, 0);
        for (int col_index = 0; col_index != col_nb_dofs / 3; col_index++) {
          auto assemble_m = get_tensor2(locKs, row_index, col_index);
          assemble_m(i, j) +=
              w * row_base_functions * col_base_functions * linear_spring(i, j);
          ++col_base_functions;
        }
        ++row_base_functions;
      }
      // move to next integration weight
      ++t_w;
    }

    // Add computed values of spring stiffness to the global LHS matrix
    CHKERR MatSetValues(
        getFEMethod()->snes_B, row_nb_dofs, &*row_data.getIndices().begin(),
        col_nb_dofs, &*col_data.getIndices().begin(), &locKs(0, 0), ADD_VALUES);

    // is symmetric
    if (row_side != col_side || row_type != col_type) {
      transLocKs.resize(col_nb_dofs, row_nb_dofs, false);
      noalias(transLocKs) = trans(locKs);
      CHKERR MatSetValues(getFEMethod()->snes_B, col_nb_dofs,
                          &*col_data.getIndices().begin(), row_nb_dofs,
                          &*row_data.getIndices().begin(), &transLocKs(0, 0),
                          ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }
};

/** * @brief Assemble contribution of springs to LHS *
 * \f[
 * f_s =  \int\limits_{\partial \Omega }^{} {{\psi ^T}{F^s}\left( u
 * \right)d\partial \Omega }  = \int\limits_{\partial \Omega }^{} {{\psi
 * ^T}{k_s}ud\partial \Omega }
 * \f]
 *
 */
struct OpSpringFs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  boost::shared_ptr<DataAtIntegrationPtsSprings> commonData;
  BlockOptionDataSprings &dAta;

  OpSpringFs(boost::shared_ptr<DataAtIntegrationPtsSprings> &common_data,
             BlockOptionDataSprings &data)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
            "SPATIAL_POSITION", OPROW),
        commonData(common_data), dAta(data) {}

  // vector used to store force vector for each degree of freedom
  VectorDouble nF;

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {

    MoFEMFunctionBegin;
    // check that the faces have associated degrees of freedom
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs == 0)
      MoFEMFunctionReturnHot(0);

    if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tRis.end()) {
      MoFEMFunctionReturnHot(0);
    }

    CHKERR commonData->getBlockData(dAta);
    // CHKERR commonData->getBlockData(*dAta);

    // size of force vector associated to the entity
    // set equal to the number of degrees of freedom of associated with the
    // entity
    nF.resize(nb_dofs, false);
    nF.clear();

    // get number of gauss points
    const int nb_gauss_pts = data.getN().size1();

    // get intergration weights
    auto t_w = getFTensor0IntegrationWeight();

    FTensor::Tensor1<double, 3> t_spring_stiffness(commonData->springStiffness0,
                                                   commonData->springStiffness1,
                                                   commonData->springStiffness2);

    // for nonlinear elasticity, solution is spatial position
    auto t_spatial_position_at_gauss_points =
        getFTensor1FromMat<3>(*commonData->xAtPts);
    auto t_init_spatial_position_at_gauss_points =
        getFTensor1FromMat<3>(*commonData->xInitAtPts);

    // loop over all gauss points of the face
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      double w = t_w * getArea();

      auto base_functions = data.getFTensor0N(gg, 0);

      // create a vector t_nf whose pointer points an array of 3 pointers
      // pointing to nF  memory location of components
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf(&nF[0], &nF[1],
                                                              &nF[2]);
      for (int row_index = 0; row_index != nb_dofs / 3;
           ++row_index) { // loop over the nodes
        for (int ii = 0; ii != 3; ++ii) {
          t_nf(ii) += (w * base_functions * t_spring_stiffness(ii)) *
                      (t_spatial_position_at_gauss_points(ii) -
                       t_init_spatial_position_at_gauss_points(ii));
        }
        // move to next base function
        ++base_functions;
        // move the pointer to next element of t_nf
        ++t_nf;
      }
      // move to next integration weight
      ++t_w;
      // move to the solutions at the next Gauss point
      ++t_spatial_position_at_gauss_points;
      ++t_init_spatial_position_at_gauss_points;
    }

    // add computed values of pressure in the global right hand side vector
    CHKERR VecSetValues(getFEMethod()->snes_f, nb_dofs, &data.getIndices()[0],
                        &nF[0], ADD_VALUES);
    MoFEMFunctionReturn(0);
  }
};

/** \brief Set of high-level function declaring elements and setting operators
 * to apply spring boundary condition
 */
struct MetaSpringBC{

  /**
   * \brief Declare spring element
   *
   * Search cubit sidesets and blocksets with spring bc and declare surface
   * element

   * Blockset has to have name “SPRING_BC”. The first three attributes of the
   * blockset are spring stiffness value.

   *
   * @param  m_field               Interface insurance
   * @param  field_name            Field name (e.g. DISPLACEMENT)
   * @param  mesh_nodals_positions Name of field on which ho-geometry is defined
   * @param  intersect_ptr         Pointer to range to interect meshset entities
   * @return                       Error code
   */
   static MoFEMErrorCode addSpringElements(
      MoFEM::Interface &m_field, const std::string field_name,
      const std::string mesh_nodals_positions = "MESH_NODE_POSITIONS");


  /**
   * \brief Implementation of spring element. Set operators to calculate LHS and
   * RHS
   *
   * @param  m_field               Interface insurance
   * @return                       Error code
   */
   static MoFEMErrorCode setSpringOperators(
       MoFEM::Interface &m_field,
       boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr,
       boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr,
       const std::string field_name, const std::string mesh_nodals_positions =
           "MESH_NODE_POSITIONS");
};


#endif //__SPRINGELEMENT_HPP__