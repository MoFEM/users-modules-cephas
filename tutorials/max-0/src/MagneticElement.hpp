/** \file MagneticElement.hpp
 * \brief Implementation of magnetic element
 * \ingroup maxwell_element
 * \example MagneticElement.hpp
 *
 */

/*
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

#ifndef __MAGNETICELEMENT_HPP__
#define __MAGNETICELEMENT_HPP__

/**
 * \brief Implementation of magneto-static problem (basic Implementation)
 * \ingroup maxwell_element
 *
 *  Look for theory and details here:
 *
 *  \cite ivanyshyn2013computation
 *  <www.hpfem.jku.at/publications/szthesis.pdf>
 *  <https://pdfs.semanticscholar.org/0574/e5763d9b64bff16f908e9621f23af3b3dc86.pdf>
 * 
 * Election file and all other problem related file are here \ref
 * maxwell_element.
 *
 * \todo Extension for mix formulation
 * \todo Use appropriate pre-conditioner for large problems
 *
 */
struct MagneticElement {

  MoFEM::Interface &mField;

  /// \brief  definition of volume element
  struct VolumeFE : public MoFEM::VolumeElementForcesAndSourcesCore {
    VolumeFE(MoFEM::Interface &m_field)
        : MoFEM::VolumeElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return 2 * order + 1; };
  };

  // /// \brief  definition of volume element
  // struct VolumeFEReducedIntegration: public
  // MoFEM::VolumeElementForcesAndSourcesCore {
  //   VolumeFEReducedIntegration(MoFEM::Interface &m_field):
  //   MoFEM::VolumeElementForcesAndSourcesCore(m_field) {}
  //   int getRule(int order) { return 2*order+1; };
  // };

  /** \brief define surface element
   *
   */
  struct TriFE : public MoFEM::FaceElementForcesAndSourcesCore {
    TriFE(MoFEM::Interface &m_field)
        : MoFEM::FaceElementForcesAndSourcesCore(m_field) {}
    int getRule(int order) { return 2 * order; };
  };

  MagneticElement(MoFEM::Interface &m_field) : mField(m_field) {}
  virtual ~MagneticElement() = default;

  /**
   * \brief data structure storing material constants, model parameters,
   * matrices, etc.
   *
   */
  struct BlockData {

    // field
    const string fieldName;
    const string feName;
    const string feNaturalBCName;

    // material parameters
    double mU;      ///< magnetic constant  N / A2
    double ePsilon; ///< regularization paramater

    // Natural boundary conditions
    Range naturalBc;

    // Essential boundary conditions
    Range essentialBc;

    int oRder; ///< approximation order

    // Petsc data
    DM dM;
    Mat A;
    Vec D, F;

    BlockData()
        : fieldName("MAGNETIC_POTENTIAL"), feName("MAGNETIC"),
          feNaturalBCName("MAGENTIC_NATURAL_BC"), mU(1), ePsilon(0.1) {}
    ~BlockData() {}
  };

  BlockData blockData;

  /**
   * \brief get natural boundary conditions
   * \ingroup maxwell_element
   * @return      error code
   */
  MoFEMErrorCode getNaturalBc() {
    MoFEMFunctionBegin;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 9, "NATURALBC") == 0) {
        Range faces;
        CHKERR mField.get_moab().get_entities_by_type(bit->meshset, MBTRI,
                                                      faces, true);
        CHKERR mField.get_moab().get_adjacencies(
            faces, 1, true, blockData.naturalBc, moab::Interface::UNION);
        blockData.naturalBc.merge(faces);
      }
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief get essential boundary conditions (only homogenous case is
   * considered) \ingroup maxwell_element
   * @return      error code
   */
  MoFEMErrorCode getEssentialBc() {
    ParallelComm *pcomm =
        ParallelComm::get_pcomm(&mField.get_moab(), MYPCOMM_INDEX);
    MoFEMFunctionBegin;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 10, "ESSENTIALBC") == 0) {
        Range faces;
        CHKERR mField.get_moab().get_entities_by_type(bit->meshset, MBTRI,
                                                      faces, true);
        CHKERR mField.get_moab().get_adjacencies(
            faces, 1, true, blockData.essentialBc, moab::Interface::UNION);
        blockData.essentialBc.merge(faces);
      }
    }
    if (blockData.essentialBc.empty()) {
      Range tets;
      CHKERR mField.get_moab().get_entities_by_type(0, MBTET, tets);
      Skinner skin(&mField.get_moab());
      Range skin_faces; // skin faces from 3d ents
      CHKERR skin.find_skin(0, tets, false, skin_faces);
      skin_faces = subtract(skin_faces, blockData.naturalBc);
      Range proc_skin;
      CHKERR pcomm->filter_pstatus(skin_faces,
                                   PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                   PSTATUS_NOT, -1, &proc_skin);
      CHKERR mField.get_moab().get_adjacencies(
          proc_skin, 1, true, blockData.essentialBc, moab::Interface::UNION);
      blockData.essentialBc.merge(proc_skin);
    }
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief build problem data structures
   * \ingroup maxwell_element
   * @return error code
   */
  MoFEMErrorCode createFields() {
    MoFEMFunctionBegin;

    // Set entities bit level. each entity has bit level depending for example
    // on refinement level. In this case we do not refine mesh or not do
    // topological changes, simply set refinement level to zero on all entities.

    CHKERR mField.getInterface<BitRefManager>()->setBitRefLevelByDim(
      0, 3,BitRefLevel().set(0));

    // add fields
    CHKERR mField.add_field(blockData.fieldName, HCURL, DEMKOWICZ_JACOBI_BASE,
                            1);
    CHKERR mField.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                            3);
    // meshset consisting all entities in mesh
    EntityHandle root_set = mField.get_moab().get_root_set();
    // add entities to field
    CHKERR mField.add_ents_to_field_by_type(root_set, MBTET,
                                            blockData.fieldName);

    // // The higher-order gradients can be gauged by locally skipping the
    // // corresponding degrees of freedom and basis functions in the
    // higher-order
    // // edge-based, face-based and cell-based finite element subspaces.
    //
    // Range tris,edges;
    // CHKERR mField.get_moab().get_entities_by_type(root_set,MBTRI,tris,true);
    // mField.get_moab().get_entities_by_type(root_set,MBEDGE,edges,true);
    //
    // // Set order in volume
    // Range bc_ents = unite(blockData.naturalBc,blockData.essentialBc);
    // Range vol_ents = subtract(unite(tris,edges),bc_ents);
    // CHKERR
    // mField.set_field_order(vol_ents,blockData.fieldName,blockData.oRder); int
    // gauged_order = 1; CHKERR
    // mField.set_field_order(bc_ents,blockData.fieldName,gauged_order);

    // Set order on tets
    CHKERR mField.set_field_order(root_set, MBTET, blockData.fieldName,
                                  blockData.oRder);
    CHKERR mField.set_field_order(root_set, MBTRI, blockData.fieldName,
                                  blockData.oRder);
    CHKERR mField.set_field_order(root_set, MBEDGE, blockData.fieldName,
                                  blockData.oRder);

    // Set geometry approximation ordered
    CHKERR mField.add_ents_to_field_by_type(root_set, MBTET,
                                            "MESH_NODE_POSITIONS");
    CHKERR mField.set_field_order(root_set, MBTET, "MESH_NODE_POSITIONS", 2);
    CHKERR mField.set_field_order(root_set, MBTRI, "MESH_NODE_POSITIONS", 2);
    CHKERR mField.set_field_order(root_set, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR mField.set_field_order(root_set, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    // build field
    CHKERR mField.build_fields();

    // get HO geometry for 10 node tets
    // This method takes coordinates form edges mid nodes in 10 node tet and
    // project values on 2nd order hierarchical basis used to approx. geometry.
    Projection10NodeCoordsOnField ent_method_material(mField,
                                                      "MESH_NODE_POSITIONS");
    CHKERR mField.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);

    MoFEMFunctionReturn(0);
  }

  /**
   * \brief create finite elements
   * \ingroup maxwell_element
   *
   * Create volume and surface element. Surface element is used to integrate
   * natural boundary conditions.
   *
   * @return error code
   */
  MoFEMErrorCode createElements() {
    MoFEMFunctionBegin;
    // //Elements
    CHKERR mField.add_finite_element(blockData.feName);
    CHKERR mField.modify_finite_element_add_field_row(blockData.feName,
                                                      blockData.fieldName);
    CHKERR mField.modify_finite_element_add_field_col(blockData.feName,
                                                      blockData.fieldName);
    CHKERR mField.modify_finite_element_add_field_data(blockData.feName,
                                                       blockData.fieldName);
    CHKERR mField.modify_finite_element_add_field_data(blockData.feName,
                                                       "MESH_NODE_POSITIONS");
    CHKERR mField.add_finite_element(blockData.feNaturalBCName);
    CHKERR mField.modify_finite_element_add_field_row(blockData.feNaturalBCName,
                                                      blockData.fieldName);
    CHKERR mField.modify_finite_element_add_field_col(blockData.feNaturalBCName,
                                                      blockData.fieldName);
    CHKERR mField.modify_finite_element_add_field_data(
        blockData.feNaturalBCName, blockData.fieldName);
    CHKERR mField.modify_finite_element_add_field_data(
        blockData.feNaturalBCName, "MESH_NODE_POSITIONS");
    CHKERR mField.add_ents_to_finite_element_by_type(0, MBTET,
                                                     blockData.feName);
    CHKERR mField.add_ents_to_finite_element_by_type(blockData.naturalBc, MBTRI,
                                                     blockData.feNaturalBCName);
    // build finite elemnts
    CHKERR mField.build_finite_elements();
    // build adjacencies
    CHKERR mField.build_adjacencies(BitRefLevel().set(0));
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief create problem
   * \ingroup maxwell_element
   *
   * Problem is collection of finite elements. With the information on which
   * fields finite elements operates the matrix and  left and right hand side
   * vector could be created.
   *
   * Here we use Distributed mesh manager from PETSc as a inteface.
   *
   * @return error code
   */
  MoFEMErrorCode createProblem() {
    MoFEMFunctionBegin;
    // set up DM
    CHKERR DMRegister_MoFEM("DMMOFEM");
    CHKERR DMCreate(PETSC_COMM_WORLD, &blockData.dM);
    CHKERR DMSetType(blockData.dM, "DMMOFEM");
    CHKERR DMMoFEMCreateMoFEM(blockData.dM, &mField, "MAGNETIC_PROBLEM",
                              BitRefLevel().set(0));
    CHKERR DMSetFromOptions(blockData.dM);
    CHKERR DMMoFEMSetIsPartitioned(blockData.dM, PETSC_TRUE);
    // add elements to blockData.dM
    CHKERR DMMoFEMAddElement(blockData.dM, blockData.feName.c_str());
    CHKERR DMMoFEMAddElement(blockData.dM, blockData.feNaturalBCName.c_str());
    CHKERR DMSetUp(blockData.dM);

    // remove essential DOFs
    const MoFEM::Problem *problem_ptr;
    CHKERR DMMoFEMGetProblemPtr(blockData.dM, &problem_ptr);
    CHKERR mField.getInterface<ProblemsManager>()->removeDofsOnEntities(
        problem_ptr->getName(), blockData.fieldName, blockData.essentialBc);
    
    MoFEMFunctionReturn(0);
  }

  /**
   * \brief destroy Distributed mesh manager
   * \ingroup maxwell_element
   * @return [description]
   */
  MoFEMErrorCode destroyProblem() {
    MoFEMFunctionBegin;
    CHKERR DMDestroy(&blockData.dM);
    MoFEMFunctionReturn(0);
  }

  /**  \brief solve problem
   * \ingroup maxwell_element
   *
   * Create matrices; integrate over elements; solve linear system of equations
   *
   */
  MoFEMErrorCode solveProblem(const bool regression_test = false) {
    MoFEMFunctionBegin;

    VolumeFE vol_fe(mField);
    vol_fe.getOpPtrVector().push_back(new OpCurlCurl(blockData));
    vol_fe.getOpPtrVector().push_back(new OpStab(blockData));
    TriFE tri_fe(mField);
    tri_fe.getOpPtrVector().push_back(new OpNaturalBC(blockData));

    // create matrices and vectors
    CHKERR DMCreateGlobalVector(blockData.dM, &blockData.D);
    // CHKERR DMCreateMatrix(blockData.dM, &blockData.A);
    CHKERR DMCreateGlobalVector(blockData.dM, &blockData.F);
    CHKERR VecDuplicate(blockData.F, &blockData.D);

    const MoFEM::Problem *problem_ptr;
    CHKERR DMMoFEMGetProblemPtr(blockData.dM, &problem_ptr);
    CHKERR mField.getInterface<MatrixManager>()
        ->createMPIAIJ<PetscGlobalIdx_mi_tag>(problem_ptr->getName(),
                                              &blockData.A);

    CHKERR MatZeroEntries(blockData.A);
    CHKERR VecZeroEntries(blockData.F);
    CHKERR VecGhostUpdateBegin(blockData.F, ADD_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(blockData.F, ADD_VALUES, SCATTER_FORWARD);

    CHKERR DMoFEMLoopFiniteElements(blockData.dM, blockData.feName.c_str(),
                                    &vol_fe);
    CHKERR DMoFEMLoopFiniteElements(blockData.dM,
                                    blockData.feNaturalBCName.c_str(), &tri_fe);

    CHKERR MatAssemblyBegin(blockData.A, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(blockData.A, MAT_FINAL_ASSEMBLY);
    CHKERR VecGhostUpdateBegin(blockData.F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(blockData.F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(blockData.F);
    CHKERR VecAssemblyEnd(blockData.F);

    KSP solver;
    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);
    CHKERR KSPSetOperators(solver, blockData.A, blockData.A);
    CHKERR KSPSetFromOptions(solver);
    CHKERR KSPSetUp(solver);
    CHKERR KSPSolve(solver, blockData.F, blockData.D);
    CHKERR KSPDestroy(&solver);

    CHKERR VecGhostUpdateBegin(blockData.D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(blockData.D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(blockData.dM, blockData.D, INSERT_VALUES,
                                   SCATTER_REVERSE);

    if (regression_test) {
      // This test is for order = 1 only
      double nrm2;
      CHKERR VecNorm(blockData.D, NORM_2, &nrm2);
      const double expected_value = 4.6772e+01;
      const double tol = 1e-4;
      if ((nrm2 - expected_value) / expected_value > tol) {
        SETERRQ2(PETSC_COMM_WORLD, MOFEM_DATA_INCONSISTENCY,
                 "Regression test failed %6.4e != %6.4e", nrm2, expected_value);
      }
    }

    CHKERR VecDestroy(&blockData.D);
    CHKERR VecDestroy(&blockData.F);
    CHKERR MatDestroy(&blockData.A);

    MoFEMFunctionReturn(0);
  }

  /**
   * \brief post-process results, i.e. save solution on the mesh
   * \ingroup maxwell_element
   * @return [description]
   */
  MoFEMErrorCode postProcessResults() {

    MoFEMFunctionBegin;
    PostProcVolumeOnRefinedMesh post_proc(mField);
    CHKERR post_proc.generateReferenceElementMesh();
    CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
    CHKERR post_proc.addFieldValuesPostProc(blockData.fieldName);
    post_proc.getOpPtrVector().push_back(new OpPostProcessCurl(
        blockData, post_proc.postProcMesh, post_proc.mapGaussPts));
    CHKERR DMoFEMLoopFiniteElements(blockData.dM, blockData.feName.c_str(),
                                    &post_proc);
    CHKERR post_proc.writeFile("out_values.h5m");
    MoFEMFunctionReturn(0);
  }

  /** \brief calculate and assemble CurlCurl matrix
   * \ingroup maxwell_element

  \f[
  \mathbf{A} = \int_\Omega \mu^{-1} \left( \nabla \times \mathbf{u}  \cdot
  \nabla \times \mathbf{v} \right) \textrm{d}\Omega \f] where \f[ \mathbf{u} =
  \nabla \times \mathbf{B} \f] where \f$\mathbf{B}\f$ is magnetic flux and
  \f$\mu\f$ is magnetic permeability.

  For more details pleas look to \cite ivanyshyn2013computation

  */
  struct OpCurlCurl
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &blockData;
    OpCurlCurl(BlockData &data)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              data.fieldName, UserDataOperator::OPROWCOL),
          blockData(data) {
      sYmm = true;
    }

    MatrixDouble entityLocalMatrix;

    /**
     * \brief integrate matrix
     * @param  row_side local number of entity on element for row of the matrix
     * @param  col_side local number of entity on element for col of the matrix
     * @param  row_type type of row entity (EDGE/TRIANGLE/TETRAHEDRON)
     * @param  col_type type of col entity (EDGE/TRIANGLE/TETRAHEDRON)
     * @param  row_data structure of data, like base functions and associated
     * methods to access those data on rows
     * @param  col_data structure of data, like base functions and associated
     * methods to access those data on rows
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;

      if (row_type == MBVERTEX)
        MoFEMFunctionReturnHot(0);
      if (col_type == MBVERTEX)
        MoFEMFunctionReturnHot(0);

      const int nb_row_dofs = row_data.getN().size2() / 3;
      if (nb_row_dofs == 0)
        MoFEMFunctionReturnHot(0);
      const int nb_col_dofs = col_data.getN().size2() / 3;
      if (nb_col_dofs == 0)
        MoFEMFunctionReturnHot(0);
      entityLocalMatrix.resize(nb_row_dofs, nb_col_dofs, false);
      entityLocalMatrix.clear();

      if (nb_row_dofs != static_cast<int>(row_data.getFieldData().size())) {
        SETERRQ2(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                 "Number of base functions and DOFs on entity is different on "
                 "rows %d!=%d",
                 nb_row_dofs, row_data.getFieldData().size());
      }
      if (nb_col_dofs != static_cast<int>(col_data.getFieldData().size())) {
        SETERRQ2(
            PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Number of base functions and DOFs on entity is different on cols",
            nb_col_dofs, col_data.getFieldData().size());
      }

      MatrixDouble row_curl_mat, col_curl_mat;
      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;

      const double c0 = 1. / blockData.mU;
      const int nb_gauss_pts = row_data.getN().size1();
      auto t_row_curl_base = row_data.getFTensor2DiffN<3, 3>();

      for (int gg = 0; gg != nb_gauss_pts; gg++) {

        // get integration weight scaled by volume
        double w = getGaussPts()(3, gg) * getVolume();
        // if ho geometry is given
        w *= getHoGaussPtsDetJac()(gg);

        FTensor::Tensor1<double, 3> t_row_curl;
        for (int aa = 0; aa != nb_row_dofs; aa++) {
          t_row_curl(i) = levi_civita(j, i, k) * t_row_curl_base(j, k);

          FTensor::Tensor0<double *> t_local_mat(&entityLocalMatrix(aa, 0), 1);
          FTensor::Tensor1<double, 3> t_col_curl;

          auto t_col_curl_base = col_data.getFTensor2DiffN<3, 3>(gg, 0);
          for (int bb = 0; bb != nb_col_dofs; bb++) {
            t_col_curl(i) = levi_civita(j, i, k) * t_col_curl_base(j, k);
            t_local_mat += c0 * w * t_row_curl(i) * t_col_curl(i);
            ++t_local_mat;
            ++t_col_curl_base;
          }
          
          ++t_row_curl_base;
        }
      }

      CHKERR MatSetValues(blockData.A, nb_row_dofs, &row_data.getIndices()[0],
                          nb_col_dofs, &col_data.getIndices()[0],
                          &entityLocalMatrix(0, 0), ADD_VALUES);

      if (row_side != col_side || row_type != col_type) {
        entityLocalMatrix = trans(entityLocalMatrix);
        CHKERR MatSetValues(blockData.A, nb_col_dofs, &col_data.getIndices()[0],
                            nb_row_dofs, &row_data.getIndices()[0],
                            &entityLocalMatrix(0, 0), ADD_VALUES);
      }

      MoFEMFunctionReturn(0);
    }
  };

  /** \brief calculate and assemble stabilization matrix
  * \ingroup maxwell_element

  \f[
  \mathbf{A} = \int_\Omega \epsilon \mathbf{u}  \cdot \mathbf{v}
  \textrm{d}\Omega \f] where \f$\epsilon\f$ is regularization parameter.

  */
  struct OpStab
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &blockData;
    OpStab(BlockData &data)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              data.fieldName, UserDataOperator::OPROWCOL),
          blockData(data) {
      sYmm = true;
    }

    MatrixDouble entityLocalMatrix;

    /**
     * \brief integrate matrix
     * @param  row_side local number of entity on element for row of the matrix
     * @param  col_side local number of entity on element for col of the matrix
     * @param  row_type type of row entity (EDGE/TRIANGLE/TETRAHEDRON)
     * @param  col_type type of col entity (EDGE/TRIANGLE/TETRAHEDRON)
     * @param  row_data structure of data, like base functions and associated
     * methods to access those data on rows
     * @param  col_data structure of data, like base functions and associated
     * methods to access those data on rows
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, int col_side, EntityType row_type,
                          EntityType col_type,
                          DataForcesAndSourcesCore::EntData &row_data,
                          DataForcesAndSourcesCore::EntData &col_data) {
      MoFEMFunctionBegin;

      if (row_type == MBVERTEX)
        MoFEMFunctionReturnHot(0);
      if (col_type == MBVERTEX)
        MoFEMFunctionReturnHot(0);

      const int nb_row_dofs = row_data.getN().size2() / 3;
      if (nb_row_dofs == 0)
        MoFEMFunctionReturnHot(0);
      const int nb_col_dofs = col_data.getN().size2() / 3;
      if (nb_col_dofs == 0)
        MoFEMFunctionReturnHot(0);
      entityLocalMatrix.resize(nb_row_dofs, nb_col_dofs, false);
      entityLocalMatrix.clear();

      if (nb_row_dofs != static_cast<int>(row_data.getFieldData().size())) {
        SETERRQ2(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                 "Number of base functions and DOFs on entity is different on "
                 "rows %d!=%d",
                 nb_row_dofs, row_data.getFieldData().size());
      }
      if (nb_col_dofs != static_cast<int>(col_data.getFieldData().size())) {
        SETERRQ2(
            PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
            "Number of base functions and DOFs on entity is different on cols",
            nb_col_dofs, col_data.getFieldData().size());
      }

      MatrixDouble row_curl_mat, col_curl_mat;
      FTensor::Index<'i', 3> i;

      const double c0 = 1. / blockData.mU;
      const double c1 = blockData.ePsilon * c0;
      const int nb_gauss_pts = row_data.getN().size1();

      for (int gg = 0; gg != nb_gauss_pts; gg++) {

        // get integration weight scaled by volume
        double w = getGaussPts()(3, gg) * getVolume();
        // if ho geometry is given
        w *= getHoGaussPtsDetJac()(gg);

        FTensor::Tensor1<const double *, 3> t_row_base(
            &row_data.getVectorN<3>(gg)(0, HVEC0),
            &row_data.getVectorN<3>(gg)(0, HVEC1),
            &row_data.getVectorN<3>(gg)(0, HVEC2), 3);

        for (int aa = 0; aa != nb_row_dofs; aa++) {
          FTensor::Tensor0<double *> t_local_mat(&entityLocalMatrix(aa, 0), 1);
          FTensor::Tensor1<const double *, 3> t_col_base(
              &col_data.getVectorN<3>(gg)(0, HVEC0),
              &col_data.getVectorN<3>(gg)(0, HVEC1),
              &col_data.getVectorN<3>(gg)(0, HVEC2), 3);
          for (int bb = 0; bb != nb_col_dofs; bb++) {
            t_local_mat += c1 * w * t_row_base(i) * t_col_base(i);
            ++t_col_base;
            ++t_local_mat;
          }
          ++t_row_base;
        }
      }

      // cerr << entityLocalMatrix << endl;
      // cerr << endl;

      CHKERR MatSetValues(blockData.A, nb_row_dofs, &row_data.getIndices()[0],
                          nb_col_dofs, &col_data.getIndices()[0],
                          &entityLocalMatrix(0, 0), ADD_VALUES);

      if (row_side != col_side || row_type != col_type) {
        entityLocalMatrix = trans(entityLocalMatrix);
        CHKERR MatSetValues(blockData.A, nb_col_dofs, &col_data.getIndices()[0],
                            nb_row_dofs, &row_data.getIndices()[0],
                            &entityLocalMatrix(0, 0), ADD_VALUES);
      }

      MoFEMFunctionReturn(0);
    }
  };

  /** \brief calculate essential boundary conditions
    * \ingroup maxwell_element

    \f[
    \mathbf{F} = \int_{\partial\Omega} \ \mathbf{u}  \cdot \mathbf{j}_i
    \textrm{d}{\partial\Omega} \f] where \f$\mathbf{j}_i\f$ is current density
    function.

    Here simple current on coil is hard coded. In future more general
    implementation is needed.

    */
  struct OpNaturalBC
      : public MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

    BlockData &blockData;

    OpNaturalBC(BlockData &data)
        : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
              data.fieldName, UserDataOperator::OPROW),
          blockData(data) {}

    VectorDouble naturalBC;

    /**
     * \brief integrate matrix
     * \ingroup maxwell_element
     * @param  row_side local number of entity on element for row of the matrix
     * @param  row_type type of row entity (EDGE/TRIANGLE/TETRAHEDRON)
     * @param  row_data structure of data, like base functions and associated
     * methods to access those data on rows
     * @return          error code
     */
    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          DataForcesAndSourcesCore::EntData &row_data) {
      MoFEMFunctionBegin;

      if (row_type == MBVERTEX)
        MoFEMFunctionReturnHot(0);

      const int nb_row_dofs = row_data.getN().size2() / 3;
      if (nb_row_dofs == 0)
        MoFEMFunctionReturnHot(0);
      naturalBC.resize(nb_row_dofs, false);
      naturalBC.clear();

      FTensor::Index<'i', 3> i;

      const int nb_gauss_pts = row_data.getN().size1();
      auto t_row_base = row_data.getFTensor1N<3>();

      for (int gg = 0; gg != nb_gauss_pts; gg++) {

        // get integration weight scaled by volume
        double area;
        area = norm_2(getNormalsAtGaussPts(gg)) * 0.5;
        double w = getGaussPts()(2, gg) * area;

        // Current is on surface where natural bc are applied. It is set that
        // current is in XY plane, circular, around the coil.
        const double x = getHoCoordsAtGaussPts()(gg, 0);
        const double y = getHoCoordsAtGaussPts()(gg, 1);
        const double r = sqrt(x * x + y * y);
        FTensor::Tensor1<double, 3> t_j;
        t_j(0) = -y / r;
        t_j(1) = +x / r;
        t_j(2) = 0;

        //double a = t_j(i) * t_tangent1(i);
        //double b = t_j(i) * t_tangent2(i);
        //t_j(i) = a * t_tangent1(i) + b * t_tangent2(i);

        // ++t_tangent1;
        // ++t_tangent2;

        FTensor::Tensor0<double *> t_f(&naturalBC[0]);
        for (int aa = 0; aa != nb_row_dofs; aa++) {
          t_f += w * t_row_base(i) * t_j(i);
          ++t_row_base;
          ++t_f;
        }
      }

      CHKERR VecSetValues(blockData.F, row_data.getIndices().size(),
                          &row_data.getIndices()[0], &naturalBC[0], ADD_VALUES);

      MoFEMFunctionReturn(0);
    }
  };

  /** \brief calculate and assemble CurlCurl matrix
   * \ingroup maxwell_element
   */
  struct OpPostProcessCurl
      : public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {

    BlockData &blockData;
    moab::Interface &postProcMesh;
    std::vector<EntityHandle> &mapGaussPts;

    OpPostProcessCurl(BlockData &data, moab::Interface &post_proc_mesh,
                      std::vector<EntityHandle> &map_gauss_pts)
        : MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator(
              data.fieldName, UserDataOperator::OPROW),
          blockData(data), postProcMesh(post_proc_mesh),
          mapGaussPts(map_gauss_pts) {}

    MoFEMErrorCode doWork(int row_side, EntityType row_type,
                          DataForcesAndSourcesCore::EntData &row_data) {
      MoFEMFunctionBegin;

      if (row_type == MBVERTEX)
        MoFEMFunctionReturnHot(0);

      Tag th;
      double def_val[] = {0, 0, 0};
      CHKERR postProcMesh.tag_get_handle("MAGNETIC_INDUCTION_FIELD", 3,
                                         MB_TYPE_DOUBLE, th,
                                         MB_TAG_CREAT | MB_TAG_SPARSE, def_val);
      const int nb_row_dofs = row_data.getN().size2() / 3;
      if (nb_row_dofs == 0)
        MoFEMFunctionReturnHot(0);
      const void *tags_ptr[mapGaussPts.size()];

      if(nb_row_dofs != row_data.getFieldData().size())
        SETERRQ2(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                 "Wrong number of base functions and DOFs %d != %d",
                 nb_row_dofs, row_data.getFieldData().size());

      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      const int nb_gauss_pts = row_data.getN().size1();
      if (nb_gauss_pts != static_cast<int>(mapGaussPts.size())) {
        SETERRQ2(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                 "Inconsistency number of dofs %d!=%d", nb_gauss_pts,
                 mapGaussPts.size());
      }
      CHKERR postProcMesh.tag_get_by_ptr(th, &mapGaussPts[0],
                                         mapGaussPts.size(), tags_ptr);
      auto t_curl_base = row_data.getFTensor2DiffN<3, 3>();
      for (int gg = 0; gg != nb_gauss_pts; gg++) {
        // get pointer to tag values on entity (i.e. vertex on refined
        // post-processing mesh)
        double *ptr = &((double *)tags_ptr[gg])[0];
        FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_curl(ptr, &ptr[1],
                                                                  &ptr[2]);
        // calculate curl value
        auto t_dof = row_data.getFTensor0FieldData();
        for (int aa = 0; aa != nb_row_dofs; aa++) {
          t_curl(i) += t_dof * (levi_civita(j, i, k) * t_curl_base(j, k));
          ++t_curl_base;
          ++t_dof;
        }
      }
      MoFEMFunctionReturn(0);
    }
  };
};

#endif //__MAGNETICELEMENT_HPP__

/******************************************************************************
 * \defgroup maxwell_element Magnetic/Maxwell element
 * \ingroup user_modules
 ******************************************************************************/
