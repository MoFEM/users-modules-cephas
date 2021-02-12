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

struct DataAtIntegrationPtsSprings : public boost::enable_shared_from_this<DataAtIntegrationPtsSprings> {

  boost::shared_ptr<MatrixDouble> gradDispPtr =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> xAtPts =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> xInitAtPts =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());

  boost::shared_ptr<MatrixDouble> hMat =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> FMat =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> HMat =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<MatrixDouble> invHMat =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());
  boost::shared_ptr<VectorDouble> detHVec =
      boost::shared_ptr<VectorDouble>(new VectorDouble());

  MatrixDouble tangent1;
  MatrixDouble tangent2;

  double springStiffnessNormal;
  double springStiffnessTangent;

    Range forcesOnlyOnEntitiesRow;
    Range forcesOnlyOnEntitiesCol;

  DataForcesAndSourcesCore::EntData *faceRowData;

  std::map<int, BlockOptionDataSprings> mapSpring;
  //   ~DataAtIntegrationPtsSprings() {}
  DataAtIntegrationPtsSprings(MoFEM::Interface &m_field) : mField(m_field), faceRowData(nullptr) {

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
    // if (field_name.compare(0, 16, "SPATIAL_POSITION") != 0)
    //   is_spatial_position = false;
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

    // FTensor::Tensor2<double, 3, 3> t_spring_local(
    //     commonDataPtr->springStiffnessNormal, 0., 0., 0., 0., 0., 0., 0., 0.);
        //     FTensor::Tensor2<double, 3, 3> t_spring_local(
        // 1., 0., 0., 0., 0., 0., 0., 0., 0.);
    // FTensor::Tensor2<double, 3, 3> t_spring_local(
    //     commonDataPtr->springStiffnessNormal, 0., 0., 0.,
    //     commonDataPtr->springStiffnessTangent, 0., 0., 0.,
    //     commonDataPtr->springStiffnessTangent);

  FTensor::Tensor2<double, 3, 3> t_spring_local(
        0., 0., 0., 0.,
        commonDataPtr->springStiffnessTangent, 0., 0., 0.,
        0.);

    // create a 3d vector to be used as the normal to the face with length equal
    // to the face area
    auto t_normal_ptr = getFTensor1Normal();
    FTensor::Tensor1<double, 3> t_normal;
    t_normal(i) = t_normal_ptr(i);

    // First tangent vector
    auto t_tangent1_ptr = getFTensor1Tangent1AtGaussPts();
    FTensor::Tensor1<double, 3> t_tangent1;
    t_tangent1(i) = t_tangent1_ptr(i);

    auto t_tangent2_ptr = getFTensor1Tangent2AtGaussPts();
    FTensor::Tensor1<double, 3> t_tangent2_help;
    t_tangent2_help(i) = t_tangent2_ptr(i);

    t_normal(i) = FTensor::levi_civita(i, j, k) * t_tangent1(j) * t_tangent2_help(k);
    const double area_h = sqrt(t_normal(k)* t_normal(k));
    t_normal(i) = t_normal(i) / area_h;
    // t_normal(i) *= 0.5;

    // Second tangent vector, such that t_n = t_t1 x t_t2 | t_t2 = t_n x t_t1
    FTensor::Tensor1<double, 3> t_tangent2;
    t_tangent2(i) = FTensor::levi_civita(i, j, k) * t_normal(j) * t_tangent1(k);
// t_tangent2(i) = t_tangent2_help(i);
    // Spring stiffness in global coordinate
    FTensor::Tensor2<double, 3, 3> t_spring_global;
    t_spring_global = MetaSpringBC::transformLocalToGlobal(
        t_normal, t_tangent1, t_tangent2, t_spring_local);
    FTensor::Tensor2<double, 3, 3> t_normal_projection;
    FTensor::Tensor2<double, 3, 3> t_tangent_projection;
    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;
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

      double w = t_w;//area was constant

      auto t_base_func = data.getFTensor0N(gg, 0);

      // create a vector t_nf whose pointer points an array of 3 pointers
      // pointing to nF  memory location of components
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf(&nF[0], &nF[1],
                                                              &nF[2]);

      for (int rr = 0; rr != nb_dofs / 3; ++rr) { // loop over the nodes
        // t_nf(i) += w * 0.5 * area_h * t_base_func * t_spring_global(i, j) *
        //            t_displacement_at_gauss_point(j);

        t_nf(i) += w * 0.5 * area_h * t_base_func * commonDataPtr->springStiffnessNormal * t_normal_projection(i, j) *
                   t_displacement_at_gauss_point(j);
        t_nf(i) -= w * 0.5 * area_h * t_base_func * commonDataPtr->springStiffnessTangent * t_tangent_projection(i, j) *
                   t_displacement_at_gauss_point(j);


        // t_normal_projection
        // t_tangent_projection
        //  t_nf(i) += w * t_base_func * t_normal(i) * t_normal(j) *
        //  t_displacement_at_gauss_point(j)/sqrt(t_normal(k)* t_normal(k));

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
        commonDataPtr(common_data_ptr), dAta(data) {
          // sYmm = false;
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

    // FTensor::Tensor2<double, 3, 3> t_spring_local(
    //     commonDataPtr->springStiffnessNormal, 0., 0., 0., 0., 0., 0., 0., 0.);

    // FTensor::Tensor2<double, 3, 3> t_spring_local(
    //     commonDataPtr->springStiffnessNormal, 0., 0., 0.,
    //     commonDataPtr->springStiffnessTangent, 0., 0., 0.,
    //     commonDataPtr->springStiffnessTangent);

    FTensor::Tensor2<double, 3, 3> t_spring_local(
        0., 0., 0., 0., commonDataPtr->springStiffnessTangent, 0., 0., 0., 0.);
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

    auto t_tangent2_ptr = getFTensor1Tangent2AtGaussPts();
    t_normal(i) = FTensor::levi_civita(i, j, k) * t_tangent1(j) * t_tangent2_ptr(k);
    const double area_h = sqrt(t_normal(k)* t_normal(k));
    t_normal(i) = t_normal(i) / area_h;
    // t_normal(i) *= 0.5;

    // Spring stiffness in global coordinate
    FTensor::Tensor2<double, 3, 3> t_spring_global;
    t_spring_global = MetaSpringBC::transformLocalToGlobal(
        t_normal, t_tangent1, t_tangent2, t_spring_local);

      FTensor::Tensor2<double, 3, 3> t_normal_projection;
    FTensor::Tensor2<double, 3, 3> t_tangent_projection;
    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;
    // loop over the Gauss points
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {
      // get area and integration weight
      double w = 0.5 *  t_w * area_h;// * getArea();

      auto t_row_base_func = row_data.getFTensor0N(gg, 0);

      for (int rr = 0; rr != row_nb_dofs / 3; rr++) {
        auto t_col_base_func = col_data.getFTensor0N(gg, 0);
        for (int cc = 0; cc != col_nb_dofs / 3; cc++) {
          auto assemble_m = get_tensor2(locKs, rr, cc);
          // assemble_m(i, j) +=
          //     w * t_row_base_func * t_col_base_func * t_spring_global(i, j);
               assemble_m(i, j) +=
              w * t_row_base_func * t_col_base_func * commonDataPtr->springStiffnessNormal * t_normal_projection(i, j);
               assemble_m(i, j) -=
              w * t_row_base_func * t_col_base_func * commonDataPtr->springStiffnessTangent * t_tangent_projection(i, j);
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


/** * @brief Assemble contribution of spring to LHS *
 * \f[
 * {K^s} = \int\limits_\Omega ^{} {{\psi ^T}{k_s}\psi d\Omega }
 * \f]
 *
 */
struct OpSpringKs_dX : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  boost::shared_ptr<DataAtIntegrationPtsSprings> commonDataPtr;
  BlockOptionDataSprings &dAta;

  MatrixDouble locKs;
  MatrixDouble transLocKs;

  OpSpringKs_dX(boost::shared_ptr<DataAtIntegrationPtsSprings> &common_data_ptr,
             BlockOptionDataSprings &data, const std::string field_name_1, const std::string field_name_2)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
            field_name_1.c_str(), field_name_2.c_str(), OPROWCOL),
        commonDataPtr(common_data_ptr), dAta(data) {
          sYmm = false;
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
    FTensor::Index<'l', 3> l;
    auto get_tensor2 = [](MatrixDouble &m, const int r, const int c) {
      return FTensor::Tensor2<double *, 3, 3>(
          &m(3 * r + 0, 3 * c + 0), &m(3 * r + 0, 3 * c + 1),
          &m(3 * r + 0, 3 * c + 2), &m(3 * r + 1, 3 * c + 0),
          &m(3 * r + 1, 3 * c + 1), &m(3 * r + 1, 3 * c + 2),
          &m(3 * r + 2, 3 * c + 0), &m(3 * r + 2, 3 * c + 1),
          &m(3 * r + 2, 3 * c + 2));
    };

  auto make_vec_der = [&](FTensor::Tensor1<double *, 2> t_N,
                         FTensor::Tensor1<double *, 3> t_1,
                         FTensor::Tensor1<double *, 3> t_2) {
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    return t_n;
  };

    auto make_vec_der_2 = [&](FTensor::Tensor1<double *, 2> t_N,
                         FTensor::Tensor1<double *, 3> t_1,
                         FTensor::Tensor1<double *, 3> t_2) {
    FTensor::Tensor1<double, 3> t_normal;
    t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);
    const double normal_norm = sqrt(t_normal(i) * t_normal(i));
    FTensor::Tensor2<double, 3, 3> t_n;
    t_n(i, j) = 0;
    t_n(i, j) += FTensor::levi_civita(i, j, k) * t_2(k) * t_N(0);
    t_n(i, j) -= FTensor::levi_civita(i, j, k) * t_1(k) * t_N(1);
    FTensor::Tensor2<double, 3, 3> t_result;
    t_result(i, j) = 0;
    t_result(i, j) = t_n(i, j) / normal_norm - t_normal(i) * t_n(k, j) * t_normal(k)/ (normal_norm * normal_norm * normal_norm);
    return t_result;
  };


    FTensor::Tensor2<double, 3, 3> t_spring_local(
        commonDataPtr->springStiffnessNormal, 0., 0., 0.,
        commonDataPtr->springStiffnessTangent, 0., 0., 0.,
        commonDataPtr->springStiffnessTangent);

    const double normal_stiffness = commonDataPtr->springStiffnessNormal; 
    const double tangent_stiffness = commonDataPtr->springStiffnessTangent; 

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
    auto t_tangent2_ptr = getFTensor1Tangent2AtGaussPts();
    FTensor::Tensor1<double, 3> t_tangent2;
    // t_tangent2(i) = FTensor::levi_civita(i, j, k) * t_normal(j) * t_tangent1(k);
    t_tangent2(i) = t_tangent2_ptr(i);

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
    auto t_1 = getFTensor1FromMat<3>(commonDataPtr->tangent1);
    auto t_2 = getFTensor1FromMat<3>(commonDataPtr->tangent2);

    constexpr auto t_kd = FTensor::Kronecker_Delta<int>();


    // loop over the Gauss points
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {
      // get area and integration weight
      t_normal(i) = FTensor::levi_civita(i, j, k) * t_1(j) * t_2(k);
    
    const double material_area = sqrt(t_normal(k)* t_normal(k));
    t_normal(i) = t_normal(i) / material_area;
    
    FTensor::Tensor2<double, 3, 3> t_normal_projection;
    FTensor::Tensor2<double, 3, 3> t_tangent_projection;
    t_normal_projection(i, j) = t_normal(i) * t_normal(j);
    t_tangent_projection(i, j) = t_normal_projection(i, j);
    t_tangent_projection(0, 0) -= 1;
    t_tangent_projection(1, 1) -= 1;
    t_tangent_projection(2, 2) -= 1;
    
      double w = t_w;

      t_displacement_at_gauss_point(i) =
          t_solution_at_gauss_point(i) - t_init_solution_at_gauss_point(i);

      auto t_row_base_func = row_data.getFTensor0N(gg, 0);

      for (int rr = 0; rr != row_nb_dofs / 3; rr++) {
        auto t_col_base_func = col_data.getFTensor0N(gg, 0);
        auto t_N = col_data.getFTensor1DiffN<2>(gg, 0);
        for (int cc = 0; cc != col_nb_dofs / 3; cc++) {
          auto t_d_n = make_vec_der(t_N, t_1, t_2);
          auto t_d_n_2 = make_vec_der_2(t_N, t_1, t_2);
          auto assemble_m = get_tensor2(locKs, rr, cc);
          // assemble_m(i, j) = 1.;
          // const double normal_norm =  sqrt(t_normal(i) * t_normal(i));

          // assemble_m(i, j) +=
          //     w * 0.5 * t_row_base_func * normal_stiffness *
          //     (t_d_n(i, j) * t_normal(k) + t_d_n(k, j) * t_normal(i)) *
          //     t_displacement_at_gauss_point(k) / normal_norm;

          // assemble_m(i, j) -= w * 0.5 * t_normal(i) *  t_row_base_func * normal_stiffness *
          //                     (t_normal(l) * t_d_n(l, j)) *
          //                     (t_normal(k) * t_displacement_at_gauss_point(k)) /
          //                     (normal_norm * normal_norm * normal_norm);

          // assemble_m(i, j) -=
          //     w * t_col_base_func *  normal_stiffness * t_row_base_func * t_normal(i) * t_normal(j) / normal_norm;

        const double length_tang_1 = sqrt(t_tangent1(k)*t_tangent1(k));

          // assemble_m(i, j) +=
          //     normal_norm * w *  tangent_stiffness *
          //     t_row_base_func * t_N(0) * ( t_kd(i, j) * t_tangent1(k)* t_displacement_at_gauss_point(k) + t_tangent1(i) * t_displacement_at_gauss_point(j) ) ;

          // assemble_m(i, j) -=
          //     normal_norm * w * t_col_base_func *  tangent_stiffness *
          //     t_row_base_func * t_tangent1(i) * t_tangent1(j) ;

          assemble_m(i, j) -=
              0.5 * w * material_area * t_col_base_func *  normal_stiffness * t_row_base_func * t_normal_projection(i, j);


              assemble_m(i, j) +=
              0.5 * w * material_area * t_col_base_func *  tangent_stiffness * t_row_base_func * t_tangent_projection(i, j);

              assemble_m(i, j) +=
                  0.5 * w * (normal_stiffness - tangent_stiffness) *
                  t_row_base_func *
                  (t_d_n(i, j) *
                       (t_normal(k) * t_displacement_at_gauss_point(k)) +
                   material_area * t_normal(i) *
                       (t_d_n_2(k, j) * t_displacement_at_gauss_point(k)));
constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
              assemble_m(i, j) += 0.5 * w * (t_d_n(l, j) * t_normal(l) )
                                  * tangent_stiffness *
                                  t_row_base_func * t_kd(i, k) *
                                  t_displacement_at_gauss_point(k);
                                  
              ++t_col_base_func;
              ++t_N;
        }
        ++t_row_base_func;
      }
      // move to next integration weight
      ++t_w;
      ++t_solution_at_gauss_point;
      ++t_init_solution_at_gauss_point;
      ++t_1;
      ++t_2;
    }

    // Add computed values of spring stiffness to the global LHS matrix
    CHKERR MatSetValues(getKSPB(), row_data, col_data, &locKs(0, 0),
                        ADD_VALUES);

    // // is symmetric
    // if (row_side != col_side || row_type != col_type) {
    //   transLocKs.resize(col_nb_dofs, row_nb_dofs, false);
    //   noalias(transLocKs) = trans(locKs);

    //   CHKERR MatSetValues(getKSPB(), col_data, row_data, &transLocKs(0, 0),
    //                       ADD_VALUES);
    // }

    MoFEMFunctionReturn(0);
  }
};


/**
   * @brief Operator for computing deformation gradients in side volumes
   *
   */
struct OpCalculateDeformation
    : public VolumeElementForcesAndSourcesCoreOnContactPrismSide::
          UserDataOperator {

  boost::shared_ptr<DataAtIntegrationPtsSprings> commonDataPtr;

  bool hoGeometry;
  OpCalculateDeformation(
      const string field_name,
      boost::shared_ptr<DataAtIntegrationPtsSprings> common_data_ptr,
      bool ho_geometry = false)
      : VolumeElementForcesAndSourcesCoreOnContactPrismSide::UserDataOperator(
            field_name, UserDataOperator::OPROW),
        commonDataPtr(common_data_ptr), hoGeometry(ho_geometry) {
    doEdges = false;
    doQuads = false;
    doTris = false;
    doTets = false;
    doPrisms = false;
    sYmm = false;
  };

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &row_data) {

    MoFEMFunctionBegin;
    FTensor::Index<'i', 3> i;
    FTensor::Index<'j', 3> j;
    FTensor::Index<'k', 3> k;

    // get number of integration points
    const int nb_integration_pts = getGaussPts().size2();

    auto t_h = getFTensor2FromMat<3, 3>(*commonDataPtr->hMat);
    auto t_H = getFTensor2FromMat<3, 3>(*commonDataPtr->HMat);

    commonDataPtr->detHVec->resize(nb_integration_pts, false);
    commonDataPtr->invHMat->resize(9, nb_integration_pts, false);
    commonDataPtr->FMat->resize(9, nb_integration_pts, false);

    commonDataPtr->detHVec->clear();
    commonDataPtr->invHMat->clear();
    commonDataPtr->FMat->clear();

    auto t_detH = getFTensor0FromVec(*commonDataPtr->detHVec);
    auto t_invH = getFTensor2FromMat<3, 3>(*commonDataPtr->invHMat);
    auto t_F = getFTensor2FromMat<3, 3>(*commonDataPtr->FMat);

    for (int gg = 0; gg != nb_integration_pts; ++gg) {
      CHKERR determinantTensor3by3(t_H, t_detH);
      CHKERR invertTensor3by3(t_H, t_detH, t_invH);
      t_F(i, j) = t_h(i, k) * t_invH(k, j);
      ++t_h;
      ++t_H;
      ++t_detH;
      ++t_invH;
      ++t_F;
    }

    MoFEMFunctionReturn(0);
  }
};

  /**
   * @brief RHS-operator for the pressure element (material configuration)
   *
   * Integrates pressure in the material configuration.
   *
   */
  struct OpSpringFsMaterial : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<DataAtIntegrationPtsSprings> dataAtPts;
    boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> sideFe;
    std::string sideFeName;
    Vec F;
    BlockOptionDataSprings &dAta;
    bool hoGeometry;

    VectorDouble nF;

    VectorInt rowIndices;

    int nbRows;           ///< number of dofs on rows
    int nbIntegrationPts; ///< number of integration points

    /**
     * @brief Integrate pressure in the material configuration.
     *
     * Virtual work of the surface pressure corresponding to a test function
     * of the material configuration \f$(\delta\mathbf{X})\f$:
     *
     * \f[
     * \delta W^\text{material}_p(\mathbf{x}, \mathbf{X}, \delta\mathbf{X}) =
     * -\int\limits_\mathcal{T} p\left\{\mathbf{F}^{\intercal}\cdot
     * \mathbf{N}(\mathbf{X}) \right\} \cdot \delta\mathbf{X}\,
     * \textrm{d}\mathcal{T} =
     * -\int\limits_{\mathcal{T}_{\xi}} p\left\{\mathbf{F}^{\intercal}\cdot
     * \left(\frac{\partial\mathbf{X}}{\partial\xi}\times\frac{\partial
     * \mathbf{X}} {\partial\eta}\right) \right\} \cdot \delta\mathbf{X}\,
     * \textrm{d}\xi\textrm{d}\eta
     *  \f]
     *
     * where \f$p\f$ is pressure, \f$\mathbf{N}\f$ is a normal to the face
     * in the material configuration, \f$\xi, \eta\f$ are coordinates in the
     * parent space
     * \f$(\mathcal{T}_\xi)\f$ and \f$\mathbf{F}\f$ is the deformation gradient:
     *
     * \f[
     * \mathbf{F} = \mathbf{h}(\mathbf{x})\,\mathbf{H}(\mathbf{X})^{-1} =
     * \frac{\partial\mathbf{x}}{\partial\boldsymbol{\chi}}
     * \frac{\partial\boldsymbol{\chi}}{\partial\mathbf{X}}
     * \f]
     *
     * where \f$\mathbf{h}\f$ and \f$\mathbf{H}\f$ are the gradients of the
     * spatial and material maps, respectively, and \f$\boldsymbol{\chi}\f$ are
     * the reference coordinates.
     *
     */

    OpSpringFsMaterial(
        const string material_field,
        boost::shared_ptr<DataAtIntegrationPtsSprings> data_at_pts,
        boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> side_fe,
        std::string &side_fe_name, BlockOptionDataSprings &data)
        : UserDataOperator(material_field, UserDataOperator::OPROW),
          dataAtPts(data_at_pts), sideFe(side_fe), sideFeName(side_fe_name),
          dAta(data){};
  

MoFEMErrorCode doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &row_data) {

  MoFEMFunctionBegin;

  if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
      dAta.tRis.end()) {
    MoFEMFunctionReturnHot(0);
  }

  // get number of dofs on row
  nbRows = row_data.getIndices().size();
  // if no dofs on row, exit that work, nothing to do here
  if (!nbRows)
    MoFEMFunctionReturnHot(0);

  nF.resize(nbRows, false);
  nF.clear();

  // get number of integration points
  nbIntegrationPts = getGaussPts().size2();

  // integrate local matrix for entity block
  CHKERR iNtegrate(row_data);

  // assemble local matrix
  CHKERR aSsemble(row_data);

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode iNtegrate(
    DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  auto get_tensor1 = [](VectorDouble &v, const int r) {
    return FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3>(
        &v(r + 0), &v(r + 1), &v(r + 2));
  };

  CHKERR loopSideVolumes(sideFeName, *sideFe);

   // check that the faces have associated degrees of freedom
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs == 0)
      MoFEMFunctionReturnHot(0);

    if (dAta.tRis.find(getFEEntityHandle()) == dAta.tRis.end()) {
      MoFEMFunctionReturnHot(0);
    }

    CHKERR dataAtPts->getBlockData(dAta);

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
        dataAtPts->springStiffnessNormal, 0., 0., 0.,
        dataAtPts->springStiffnessTangent, 0., 0., 0.,
        dataAtPts->springStiffnessTangent);
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
        getFTensor1FromMat<3>(*dataAtPts->xAtPts);
    auto t_init_solution_at_gauss_point =
        getFTensor1FromMat<3>(*dataAtPts->xInitAtPts);
    FTensor::Tensor1<double, 3> t_displacement_at_gauss_point;

    auto t_F = getFTensor2FromMat<3, 3>(*dataAtPts->FMat);
    // loop over all Gauss points of the face
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {

      // Calculate the displacement at the Gauss point
        t_displacement_at_gauss_point(i) =
            t_solution_at_gauss_point(i) - t_init_solution_at_gauss_point(i);
      

      double w = t_w * getArea();

      auto t_base_func = data.getFTensor0N(gg, 0);

      // create a vector t_nf whose pointer points an array of 3 pointers
      // pointing to nF  memory location of components
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf(&nF[0], &nF[1],
                                                              &nF[2]);

      for (int rr = 0; rr != nb_dofs / 3; ++rr) { // loop over the nodes
        t_nf(i) -= w * t_base_func * t_F(k, i) * t_spring_global(k, j) *
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
      ++t_F;
    }
    // add computed values of spring in the global right hand side vector
    // Vec f = getFEMethod()->ksp_f != PETSC_NULL ? getFEMethod()->ksp_f
    //                                            : getFEMethod()->snes_f;
    // CHKERR VecSetValues(f, nb_dofs, &data.getIndices()[0], &nF[0], ADD_VALUES);
    MoFEMFunctionReturn(0);
}

MoFEMErrorCode aSsemble(
    DataForcesAndSourcesCore::EntData &row_data) {
  MoFEMFunctionBegin;

  // get pointer to first global index on row
  const int *row_indices = &*row_data.getIndices().data().begin();

  auto &data = *dataAtPts;
  if (!data.forcesOnlyOnEntitiesRow.empty()) {
    rowIndices.resize(nbRows, false);
    noalias(rowIndices) = row_data.getIndices();
    row_indices = &rowIndices[0];
    VectorDofs &dofs = row_data.getFieldDofs();
    VectorDofs::iterator dit = dofs.begin();
    for (int ii = 0; dit != dofs.end(); ++dit, ++ii) {
      if (data.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) ==
          data.forcesOnlyOnEntitiesRow.end()) {
        rowIndices[ii] = -1;
      }
    }
  }

    CHKERR VecSetValues(getSNESf(), nbRows, row_indices, &*nF.data().begin(),
                      ADD_VALUES);

  // auto get_f = [&]() {
  //   if (F == PETSC_NULL)
  //     return getKSPf();
  //   return F;
  // };

  // auto vec_assemble = [&](Vec my_f) {
  //   MoFEMFunctionBegin;
  //   CHKERR VecSetOption(my_f, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);
  //   CHKERR VecSetValues(my_f, nbRows, row_indices, &*nF.data().begin(),
  //                       ADD_VALUES);
    MoFEMFunctionReturn(0);
  };

  // assemble local matrix
  // CHKERR vec_assemble(get_f());

//   MoFEMFunctionReturn(0);
// }
};


// /**
//    * @brief Operator for computing tangent vectors
//    *
//    */
  struct OpGetTangentSpEle : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    boost::shared_ptr<DataAtIntegrationPtsSprings> dataAtIntegrationPts;
    OpGetTangentSpEle(const string field_name,
                 boost::shared_ptr<DataAtIntegrationPtsSprings> dataAtIntegrationPts)
        : UserDataOperator(field_name, UserDataOperator::OPCOL),
          dataAtIntegrationPts(dataAtIntegrationPts) {}

    int ngp;

MoFEMErrorCode doWork(
    int side, EntityType type, DataForcesAndSourcesCore::EntData &data) {
  MoFEMFunctionBegin;

  if (data.getFieldData().size() == 0)
    PetscFunctionReturn(0);

  ngp = data.getN().size1();

  unsigned int nb_dofs = data.getFieldData().size() / 3;
  FTensor::Index<'i', 3> i;
  if (type == MBVERTEX) {
    dataAtIntegrationPts->tangent1.resize(3, ngp, false);
    dataAtIntegrationPts->tangent1.clear();

    dataAtIntegrationPts->tangent2.resize(3, ngp, false);
    dataAtIntegrationPts->tangent2.clear();
  }

  auto t_1 = getFTensor1FromMat<3>(dataAtIntegrationPts->tangent1);
  auto t_2 = getFTensor1FromMat<3>(dataAtIntegrationPts->tangent2);

  for (unsigned int gg = 0; gg != ngp; ++gg) {
    auto t_N = data.getFTensor1DiffN<2>(gg, 0);
    auto t_dof = data.getFTensor1FieldData<3>();

    for (unsigned int dd = 0; dd != nb_dofs; ++dd) {
      t_1(i) += t_dof(i) * t_N(0);
      t_2(i) += t_dof(i) * t_N(1);
      ++t_dof;
      ++t_N;
    }
    ++t_1;
    ++t_2;
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


MoFEMErrorCode
MetaSpringBC::addSpringElementsALE(MoFEM::Interface &m_field,
                                const std::string field_name,
                                const std::string mesh_nodals_positions) {
  MoFEMFunctionBegin;

  // if (field_name != "SPATIAL_POSITONS" &&
  //     mesh_nodals_positions != "MESH_NODE_POSITIONS") {
  //   SETERRQ2(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
  //            "Input fields for ALE spring elements has to be SPATIAL_POSITONS "
  //            "and MESH_NODE_POSITIONS, but instead they are %s and %s",
  //            field_name.c_str(), mesh_nodals_positions.c_str());
  // }

  // Define boundary element that operates on rows, columns and data of a
  // given field
  CHKERR m_field.add_finite_element("SPRING_ALE", MF_ZERO);
  CHKERR m_field.modify_finite_element_add_field_row("SPRING_ALE", field_name);
  CHKERR m_field.modify_finite_element_add_field_col("SPRING_ALE", field_name);
  CHKERR m_field.modify_finite_element_add_field_data("SPRING_ALE", field_name);
  CHKERR m_field.modify_finite_element_add_field_row("SPRING_ALE", mesh_nodals_positions);
  CHKERR m_field.modify_finite_element_add_field_col("SPRING_ALE", mesh_nodals_positions);
  CHKERR m_field.modify_finite_element_add_field_data("SPRING_ALE",
                                                      mesh_nodals_positions);
  // Add entities to that element, here we add all triangles with SPRING_BC
  // from cubit
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
    if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {
      CHKERR m_field.add_ents_to_finite_element_by_type(bit->getMeshset(),
                                                        MBTRI, "SPRING_ALE");
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

MoFEMErrorCode MetaSpringBC::setSpringOperatorsMaterial(
    MoFEM::Interface &m_field,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr,
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr,
    const std::string field_name, const std::string mesh_nodals_positions, std::string side_fe_name) {
  MoFEMFunctionBegin;

  // Push operators to instances for springs
  // loop over blocks
  boost::shared_ptr<DataAtIntegrationPtsSprings> commonDataPtr =
      boost::make_shared<DataAtIntegrationPtsSprings>(m_field);
  CHKERR commonDataPtr->getParameters();

  for (auto &sitSpring : commonDataPtr->mapSpring) {

  boost::shared_ptr<VolumeElementForcesAndSourcesCoreOnSide> feMatSideRhs =
      boost::make_shared<VolumeElementForcesAndSourcesCoreOnSide>(m_field);

  // feMatSideRhs->getOpPtrVector().push_back(
  //     new OpCalculateVectorFieldGradient<3, 3>(mesh_nodals_positions,
  //                                              commonDataPtr->HMat));
  // feMatSideRhs->getOpPtrVector().push_back(
  //     new OpCalculateVectorFieldGradient<3, 3>(field_name,
  //                                              commonDataPtr->hMat));

  // feMatSideRhs->getOpPtrVector().push_back(
  //     new OpCalculateDeformation(mesh_nodals_positions, commonDataPtr));

  //   fe_spring_rhs_ptr->getOpPtrVector().push_back(
  //       new OpCalculateVectorFieldValues<3>(field_name, commonDataPtr->xAtPts));
  //   fe_spring_rhs_ptr->getOpPtrVector().push_back(
  //       new OpCalculateVectorFieldValues<3>(mesh_nodals_positions,
  //                                           commonDataPtr->xInitAtPts));

  // fe_spring_rhs_ptr->getOpPtrVector().push_back(new OpSpringFsMaterial(
  //     mesh_nodals_positions, commonDataPtr, feMatSideRhs, side_fe_name, sitSpring.second));

    fe_spring_lhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(field_name,
        commonDataPtr->xAtPts));
    fe_spring_lhs_ptr->getOpPtrVector().push_back(
        new OpCalculateVectorFieldValues<3>(mesh_nodals_positions,
                                            commonDataPtr->xInitAtPts));
    fe_spring_lhs_ptr->getOpPtrVector().push_back(new OpGetTangentSpEle(
        mesh_nodals_positions, commonDataPtr));
    feMatSideRhs->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(mesh_nodals_positions,
                                                 commonDataPtr->HMat));
    feMatSideRhs->getOpPtrVector().push_back(
        new OpCalculateVectorFieldGradient<3, 3>(field_name,
                                                 commonDataPtr->hMat));
    fe_spring_lhs_ptr->getOpPtrVector().push_back(new OpSpringKs_dX(
        commonDataPtr, sitSpring.second, field_name, mesh_nodals_positions));

    // fe_spring_rhs_ptr->getOpPtrVector().push_back(
    //     new OpSpringFs(commonDataPtr, sitSpring.second, field_name));
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
