/* \file SpringElement.cpp
  \brief Implementation of spring boundary condition on triangular surfaces
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

#include <MoFEM.hpp>
using namespace MoFEM;

#include <SpringElement.hpp>
using namespace boost::numeric;

struct BlockOptionDataSprings {
  int iD;

  double springStiffnessNormal;
  double springStiffnessTangent;

  Range tRis;

  BlockOptionDataSprings()
      : springStiffnessNormal(-1), springStiffnessTangent(-1) {}
};

struct DataAtIntegrationPtsSprings {

  boost::shared_ptr<MatrixDouble> gradDispPtr =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> xAtPts =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> xInitAtPts =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());

  double springStiffnessNormal;
  double springStiffnessTangent;

  std::map<int, BlockOptionDataSprings> mapSpring;
  //   ~DataAtIntegrationPtsSprings() {}
  DataAtIntegrationPtsSprings(MoFEM::Interface &m_field) : mField(m_field) {

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

    springStiffnessNormal = data.springStiffnessNormal;
    springStiffnessTangent = data.springStiffnessTangent;

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {

        const int id = bit->getMeshsetId();
        mapSpring[id].tRis.clear();
        CHKERR mField.get_moab().get_entities_by_type(bit->getMeshset(), MBTRI,
                                                      mapSpring[id].tRis, true);

        std::vector<double> attributes;
        bit->getAttributes(attributes);
        if (attributes.size() < 2) {
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                   "Springs should have 2 attributes but there is %d",
                   attributes.size());
        }
        mapSpring[id].iD = id;
        mapSpring[id].springStiffnessNormal = attributes[0];
        mapSpring[id].springStiffnessTangent = attributes[1];

        // Print spring blocks after being read
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "\nSpring block %d\n", id);
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "\tNormal stiffness %3.4g\n",
                           attributes[0]);
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "\tTangent stiffness %3.4g\n",
                           attributes[1]);
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MoFEM::Interface &mField;
};

/** * @brief Assemble contribution of springs to RHS *
 * \f[
 * f_s =  \int\limits_{\partial \Omega }^{} {{\psi ^T}{F^s}\left( u
 * \right)d\partial \Omega }  = \int\limits_{\partial \Omega }^{} {{\psi
 * ^T}{k_s}ud\partial \Omega }
 * \f]
 *
 */
struct OpSpringFs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  // vector used to store force vector for each degree of freedom
  VectorDouble nF;

  boost::shared_ptr<DataAtIntegrationPtsSprings> commonDataPtr;
  BlockOptionDataSprings &dAta;
  bool is_spatial_position = true;

  OpSpringFs(boost::shared_ptr<DataAtIntegrationPtsSprings> &common_data_ptr,
             BlockOptionDataSprings &data, const std::string field_name)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
            field_name.c_str(), OPROW),
        commonDataPtr(common_data_ptr), dAta(data) {
    if (field_name.compare(0, 16, "SPATIAL_POSITION") != 0)
      is_spatial_position = false;
  }

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {

    MoFEMFunctionBegin;

    // check that the faces have associated degrees of freedom
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs == 0)
      MoFEMFunctionReturnHot(0);

    if (dAta.tRis.find(getFEEntityHandle()) == dAta.tRis.end()) {
      MoFEMFunctionReturnHot(0);
    }

    CHKERR commonDataPtr->getBlockData(dAta);

    // size of force vector associated to the entity
    // set equal to the number of degrees of freedom of associated with the
    // entity
    nF.resize(nb_dofs, false);
    nF.clear();

    // get number of Gauss points
    const int nb_gauss_pts = data.getN().size1();

    // get integration weights
    auto t_w = getFTensor0IntegrationWeight();

    // FTensor indices
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    FTensor::Tensor2<double, 3, 3> t_spring_local(
        commonDataPtr->springStiffnessNormal, 0., 0., 0.,
        commonDataPtr->springStiffnessTangent, 0., 0., 0.,
        commonDataPtr->springStiffnessTangent);
    // create a 3d vector to be used as the normal to the face with length equal
    // to the face area
    auto t_normal_ptr = getFTensor1Normal();

    FTensor::Tensor1<double, 3> t_normal;
    t_normal(i) = t_normal_ptr(i);

    // First tangent vector
    auto t_tangent1_ptr = getFTensor1Tangent1AtGaussPts();
    FTensor::Tensor1<double, 3> t_tangent1;
    t_tangent1(i) = t_tangent1_ptr(i);

    // Second tangent vector, such that t_n = t_t1 x t_t2 | t_t2 = t_n x t_t1
    FTensor::Tensor1<double, 3> t_tangent2;
    t_tangent2(i) = FTensor::levi_civita(i, j, k) * t_normal(j) * t_tangent1(k);

    // Spring stiffness in global coordinate
    FTensor::Tensor2<double, 3, 3> t_spring_global;
    t_spring_global = MetaSpringBC::transformLocalToGlobal(
        t_normal, t_tangent1, t_tangent2, t_spring_local);

    // Extract solution at Gauss points
    auto t_solution_at_gauss_point =
        getFTensor1FromMat<3>(*commonDataPtr->xAtPts);
    auto t_init_solution_at_gauss_point =
        getFTensor1FromMat<3>(*commonDataPtr->xInitAtPts);
    FTensor::Tensor1<double, 3> t_displacement_at_gauss_point;

    // loop over all Gauss points of the face
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      // Calculate the displacement at the Gauss point
      if (is_spatial_position) { // "SPATIAL_POSITION"
        t_displacement_at_gauss_point(i) =
            t_solution_at_gauss_point(i) - t_init_solution_at_gauss_point(i);
      } else { // e.g. "DISPLACEMENT" or "U"
        t_displacement_at_gauss_point(i) = t_solution_at_gauss_point(i);
      }

      double w = t_w * getArea();

      auto t_base_func = data.getFTensor0N(gg, 0);

      // create a vector t_nf whose pointer points an array of 3 pointers
      // pointing to nF  memory location of components
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf(&nF[0], &nF[1],
                                                              &nF[2]);

      for (int rr = 0; rr != nb_dofs / 3; ++rr) { // loop over the nodes
        t_nf(i) += w * t_base_func * t_spring_global(i, j) *
                   t_displacement_at_gauss_point(j);

        // move to next base function
        ++t_base_func;
        // move the pointer to next element of t_nf
        ++t_nf;
      }
      // move to next integration weight
      ++t_w;
      // move to the solutions at the next Gauss point
      ++t_solution_at_gauss_point;
      ++t_init_solution_at_gauss_point;
    }
    // add computed values of spring in the global right hand side vector
    Vec f = getFEMethod()->ksp_f != PETSC_NULL ? getFEMethod()->ksp_f
                                               : getFEMethod()->snes_f;
    CHKERR VecSetValues(f, nb_dofs, &data.getIndices()[0], &nF[0], ADD_VALUES);
    MoFEMFunctionReturn(0);
  }
};

/** * @brief Assemble contribution of spring to LHS *
 * \f[
 * {K^s} = \int\limits_\Omega ^{} {{\psi ^T}{k_s}\psi d\Omega }
 * \f]
 *
 */
struct OpSpringKs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  boost::shared_ptr<DataAtIntegrationPtsSprings> commonDataPtr;
  BlockOptionDataSprings &dAta;

  MatrixDouble locKs;
  MatrixDouble transLocKs;

  OpSpringKs(boost::shared_ptr<DataAtIntegrationPtsSprings> &common_data_ptr,
             BlockOptionDataSprings &data, const std::string field_name)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
            field_name.c_str(), field_name.c_str(), OPROWCOL),
        commonDataPtr(common_data_ptr), dAta(data) {}

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

    if (dAta.tRis.find(getFEEntityHandle()) == dAta.tRis.end()) {
      MoFEMFunctionReturnHot(0);
    }

    CHKERR commonDataPtr->getBlockData(dAta);
    // size associated to the entity
    locKs.resize(row_nb_dofs, col_nb_dofs, false);
    locKs.clear();

    // get number of Gauss points
    const int row_nb_gauss_pts = row_data.getN().size1();
    if (!row_nb_gauss_pts) // check if number of Gauss point <> 0
      MoFEMFunctionReturnHot(0);

    // get intergration weights
    auto t_w = getFTensor0IntegrationWeight();

    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;
    auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(3 * r + 0, 3 * c + 0), &m(3 * r + 0, 3 * c + 1),
          &m(3 * r + 0, 3 * c + 2), &m(3 * r + 1, 3 * c + 0),
          &m(3 * r + 1, 3 * c + 1), &m(3 * r + 1, 3 * c + 2),
          &m(3 * r + 2, 3 * c + 0), &m(3 * r + 2, 3 * c + 1),
          &m(3 * r + 2, 3 * c + 2));
    };

    FTensor::Tensor2<double, 3, 3> t_spring_local(
        commonDataPtr->springStiffnessNormal, 0., 0., 0.,
        commonDataPtr->springStiffnessTangent, 0., 0., 0.,
        commonDataPtr->springStiffnessTangent);
    // create a 3d vector to be used as the normal to the face with length equal
    // to the face area
    auto t_normal_ptr = getFTensor1Normal();

    FTensor::Tensor1<double, 3> t_normal;
    t_normal(i) = t_normal_ptr(i);

    // First tangent vector
    auto t_tangent1_ptr = getFTensor1Tangent1AtGaussPts();
    FTensor::Tensor1<double, 3> t_tangent1;
    t_tangent1(i) = t_tangent1_ptr(i);

    // Second tangent vector, such that t_n = t_t1 x t_t2 | t_t2 = t_n x t_t1
    FTensor::Tensor1<double, 3> t_tangent2;
    t_tangent2(i) = FTensor::levi_civita(i, j, k) * t_normal(j) * t_tangent1(k);

    // Spring stiffness in global coordinate
    FTensor::Tensor2<double, 3, 3> t_spring_global;
    t_spring_global = MetaSpringBC::transformLocalToGlobal(
        t_normal, t_tangent1, t_tangent2, t_spring_local);

    // loop over the Gauss points
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {
      // get area and integration weight
      double w = t_w * getArea();

      auto t_row_base_func = row_data.getFTensor0N(gg, 0);

      for (int rr = 0; rr != row_nb_dofs / 3; rr++) {
        auto t_col_base_func = col_data.getFTensor0N(gg, 0);
        for (int cc = 0; cc != col_nb_dofs / 3; cc++) {
          auto assemble_m = get_tensor2(locKs, rr, cc);
          assemble_m(i, j) +=
              w * t_row_base_func * t_col_base_func * t_spring_global(i, j);
          ++t_col_base_func;
        }
        ++t_row_base_func;
      }
      // move to next integration weight
      ++t_w;
    }

    // Add computed values of spring stiffness to the global LHS matrix
    CHKERR MatSetValues(getKSPB(), row_data, col_data, &locKs(0, 0),
                        ADD_VALUES);

    // is symmetric
    if (row_side != col_side || row_type != col_type) {
      transLocKs.resize(col_nb_dofs, row_nb_dofs, false);
      noalias(transLocKs) = trans(locKs);

      CHKERR MatSetValues(getKSPB(), col_data, row_data, &transLocKs(0, 0),
                          ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }
};

MoFEMErrorCode
MetaSpringBC::addSpringElements(MoFEM::Interface &m_field,
                                const std::string field_name,
                                const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  // Define boundary element that operates on rows, columns and data of a
  // given field
  CHKERR m_field.add_finite_element("SPRING", MF_ZERO);
  CHKERR m_field.modify_finite_element_add_field_row("SPRING", field_name);
  CHKERR m_field.modify_finite_element_add_field_col("SPRING", field_name);
  CHKERR m_field.modify_finite_element_add_field_data("SPRING", field_name);
  if (m_field.check_field(mesh_nodals_positions)) {
    CHKERR m_field.modify_finite_element_add_field_data("SPRING",
                                                        mesh_nodals_positions);
  }
  // Add entities to that element, here we add all triangles with SPRING_BC
  // from cubit
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {
      CHKERR m_field.add_ents_to_finite_element_by_type(bit->getMeshset(),
                                                        MBTRI, "SPRING");
    }
  }

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode MetaSpringBC::setSpringOperators(
    MoFEM::Interface &m_field,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr,
    const std::string field_name, const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  // Push operators to instances for springs
  // loop over blocks
  boost::shared_ptr<DataAtIntegrationPtsSprings> commonDataPtr =
      boost::make_shared<DataAtIntegrationPtsSprings>(m_field);
  CHKERR commonDataPtr->getParameters();

  for (auto &sitSpring : commonDataPtr->mapSpring) {
    fe_spring_lhs_ptr->getOpPtrVector().push_back(
        new OpSpringKs(commonDataPtr, sitSpring.second, field_name));

    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(field_name, commonDataPtr->xAtPts));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(mesh_nodals_positions,
                                            commonDataPtr->xInitAtPts));
    fe_spring_rhs_ptr->getOpPtrVector().push_back(
        new OpSpringFs(commonDataPtr, sitSpring.second, field_name));
  }
  //   cerr << "commonDataPtr has been used!!! " << commonDataPtr.use_count() <<
  //   " times" << endl;
  MoFEMFunctionReturn(0);
}

FTensor::Tensor2<double, 3, 3> MetaSpringBC::transformLocalToGlobal(
    FTensor::Tensor1<double, 3> t_normal_local,
    FTensor::Tensor1<double, 3> t_tangent1_local,
    FTensor::Tensor1<double, 3> t_tangent2_local,
    FTensor::Tensor2<double, 3, 3> t_spring_local) {

  // FTensor indices
  FTensor::Index<'i', 3> i;
  FTensor::Index<'j', 3> j;
  FTensor::Index<'k', 3> k;
  FTensor::Index<'l', 3> l;

  // Global base vectors
  FTensor::Tensor1<double, 3> t_e1(1., 0., 0.);
  FTensor::Tensor1<double, 3> t_e2(0., 1., 0.);
  FTensor::Tensor1<double, 3> t_e3(0., 0., 1.);

  // Direction cosines
  auto get_cosine = [&](auto x, auto xp) {
    return (x(i) * xp(i)) / (sqrt(x(i) * x(i)) * sqrt(xp(i) * xp(i)));
  };

  // Transformation matrix (tensor)
  FTensor::Tensor2<double, 3, 3> t_transformation_matrix(
      get_cosine(t_e1, t_normal_local), get_cosine(t_e1, t_tangent1_local),
      get_cosine(t_e1, t_tangent2_local), get_cosine(t_e2, t_normal_local),
      get_cosine(t_e2, t_tangent1_local), get_cosine(t_e2, t_tangent2_local),
      get_cosine(t_e3, t_normal_local), get_cosine(t_e3, t_tangent1_local),
      get_cosine(t_e3, t_tangent2_local));

  // Spring stiffness in global coordinate, Q*ls*Q^T
  FTensor::Tensor2<double, 3, 3> t_spring_global;
  t_spring_global(i, j) = t_transformation_matrix(i, k) * t_spring_local(k, l) *
                          t_transformation_matrix(j, l);

  return t_spring_global;
};
