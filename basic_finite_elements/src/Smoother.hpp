

#ifndef __SMOOTHER_HPP__
#define __SMOOTHER_HPP__

#ifndef WITH_ADOL_C
#error "MoFEM need to be compiled with ADOL-C"
#endif

struct Smoother {

  struct SmootherBlockData {

    bool sTabilised;
    Vec frontF;
    Vec tangentFrontF;
    bool ownVectors;

    SmootherBlockData()
        : sTabilised(false), frontF(PETSC_NULL), tangentFrontF(PETSC_NULL),
          ownVectors(false) {
      ierr = getOptions();
      CHKERRABORT(PETSC_COMM_SELF, ierr);
    }

    MoFEMErrorCode getOptions() {
      MoFEMFunctionBegin;
      ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "",
                               "Get stabilisation element options", "none");
      CHKERRG(ierr);
      PetscBool smoothing_on = sTabilised ? PETSC_TRUE : PETSC_FALSE;
      CHKERR PetscOptionsBool("-smoothing_stabilise",
                              "all nodes controlled by smoothing element", "",
                              smoothing_on, &smoothing_on, PETSC_NULL);
      sTabilised = (smoothing_on == PETSC_TRUE) ? true : false;
      ierr = PetscOptionsEnd();
      CHKERRG(ierr);
      MoFEMFunctionReturn(0);
    }

    virtual ~SmootherBlockData() {
      if (ownVectors) {
        ierr = VecDestroy(&frontF);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        ierr = VecDestroy(&tangentFrontF);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
      }
    }
  };
  SmootherBlockData smootherData;

  std::map<int, NonlinearElasticElement::BlockData> setOfBlocks;
  NonlinearElasticElement::CommonData commonData;

  struct MyVolumeFE : public NonlinearElasticElement::MyVolumeFE {

    SmootherBlockData &smootherData;

    MyVolumeFE(MoFEM::Interface &m_field, SmootherBlockData &smoother_data)
        : NonlinearElasticElement::MyVolumeFE(m_field),
          smootherData(smoother_data) {}

    MoFEMErrorCode preProcess() {
      MoFEMFunctionBegin;

      CHKERR VolumeElementForcesAndSourcesCore::preProcess();

      if (A != PETSC_NULL)
        snes_B = A;

      if (F != PETSC_NULL)
        snes_f = F;
      switch (snes_ctx) {
      case CTX_SNESSETFUNCTION: {
        if (smootherData.frontF) {
          CHKERR VecZeroEntries(smootherData.frontF);
          CHKERR VecGhostUpdateBegin(smootherData.frontF, INSERT_VALUES,
                                     SCATTER_FORWARD);
          CHKERR VecGhostUpdateEnd(smootherData.frontF, INSERT_VALUES,
                                   SCATTER_FORWARD);
        }
      } break;
      default:
        break;
      }

      MoFEMFunctionReturn(0);
    }

    MoFEMErrorCode postProcess() {
      MoFEMFunctionBegin;

      switch (snes_ctx) {
      case CTX_SNESSETFUNCTION: {
        if (smootherData.frontF) {
          CHKERR VecAssemblyBegin(smootherData.frontF);
          CHKERR VecAssemblyEnd(smootherData.frontF);
          CHKERR VecGhostUpdateBegin(smootherData.frontF, ADD_VALUES,
                                     SCATTER_REVERSE);
          CHKERR VecGhostUpdateEnd(smootherData.frontF, ADD_VALUES,
                                   SCATTER_REVERSE);
          CHKERR VecGhostUpdateBegin(smootherData.frontF, INSERT_VALUES,
                                     SCATTER_FORWARD);
          CHKERR VecGhostUpdateEnd(smootherData.frontF, INSERT_VALUES,
                                   SCATTER_FORWARD);
        }
        break;
      default:
        break;
      }
      }

      CHKERR VolumeElementForcesAndSourcesCore::postProcess();
      MoFEMFunctionReturn(0);
    }
  };

  boost::shared_ptr<MyVolumeFE> feRhsPtr;
  boost::shared_ptr<MyVolumeFE> feLhsPtr;

  MyVolumeFE &feRhs; ///< calculate right hand side for tetrahedral elements
  MyVolumeFE &getLoopFeRhs() { return feRhs; } ///< get rhs volume element
  MyVolumeFE &feLhs; //< calculate left hand side for tetrahedral elements
  MyVolumeFE &getLoopFeLhs() { return feLhs; } ///< get lhs volume element

  Smoother(MoFEM::Interface &m_field)
      : feRhsPtr(new MyVolumeFE(m_field, smootherData)),
        feLhsPtr(new MyVolumeFE(m_field, smootherData)), feRhs(*feRhsPtr),
        feLhs(*feLhsPtr) {}

  struct OpJacobianSmoother
      : public NonlinearElasticElement::OpJacobianPiolaKirchhoffStress {

    OpJacobianSmoother(const std::string field_name,
                       NonlinearElasticElement::BlockData &data,
                       NonlinearElasticElement::CommonData &common_data,
                       int tag, bool jacobian)
        : NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
              field_name, data, common_data, tag, jacobian, false, false) {}

    MoFEMErrorCode calculateStress(const int gg) {
      MoFEMFunctionBegin;

      CHKERR dAta.materialAdoublePtr->calculateP_PiolaKirchhoffI(
          dAta, getNumeredEntFiniteElementPtr());

      commonData.sTress[gg].resize(3, 3, false);
      for (int dd1 = 0; dd1 < 3; dd1++) {
        for (int dd2 = 0; dd2 < 3; dd2++) {
          dAta.materialAdoublePtr->P(dd1, dd2) >>=
              (commonData.sTress[gg])(dd1, dd2);
        }
      }

      MoFEMFunctionReturn(0);
    }
  };

  struct OpRhsSmoother : public NonlinearElasticElement::OpRhsPiolaKirchhoff {

    SmootherBlockData &smootherData;

    OpRhsSmoother(const std::string field_name,
                  NonlinearElasticElement::BlockData &data,
                  NonlinearElasticElement::CommonData &common_data,
                  SmootherBlockData &smoother_data)
        : NonlinearElasticElement::OpRhsPiolaKirchhoff(field_name, data,
                                                       common_data),
          smootherData(smoother_data) {}

    ublas::vector<int> frontIndices;

    MoFEMErrorCode aSemble(int row_side, EntityType row_type,
                           EntitiesFieldData::EntData &row_data) {
      MoFEMFunctionBegin;

      int nb_dofs = row_data.getIndices().size();
      int *indices_ptr = &row_data.getIndices()[0];

      if (!dAta.forcesOnlyOnEntitiesRow.empty()) {
        iNdices.resize(nb_dofs, false);
        noalias(iNdices) = row_data.getIndices();
        if (!smootherData.sTabilised) {
          indices_ptr = &iNdices[0];
        }
        frontIndices.resize(nb_dofs, false);
        noalias(frontIndices) = row_data.getIndices();
        VectorDofs &dofs = row_data.getFieldDofs();
        VectorDofs::iterator dit = dofs.begin();
        for (int ii = 0; dit != dofs.end(); dit++, ii++) {
          if (dAta.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) !=
              dAta.forcesOnlyOnEntitiesRow.end()) {
            iNdices[ii] = -1;
          } else {
            frontIndices[ii] = -1;
          }
        }
        if (smootherData.frontF) {
          CHKERR VecSetValues(smootherData.frontF, nb_dofs, &frontIndices[0],
                              &nf[0], ADD_VALUES);
        }
      }

      CHKERR VecSetOption(getFEMethod()->snes_f, VEC_IGNORE_NEGATIVE_INDICES,
                          PETSC_TRUE);
      CHKERR VecSetValues(getFEMethod()->snes_f, nb_dofs, indices_ptr, &nf[0],
                          ADD_VALUES);

      MoFEMFunctionReturn(0);
    }
  };

  struct OpLhsSmoother
      : public NonlinearElasticElement::OpLhsPiolaKirchhoff_dx {

    SmootherBlockData &smootherData;
    const std::string fieldCrackAreaTangentConstrains;

    OpLhsSmoother(
        const std::string vel_field, const std::string field_name,
        NonlinearElasticElement::BlockData &data,
        NonlinearElasticElement::CommonData &common_data,
        SmootherBlockData &smoother_data,
        const std::string
            crack_area_tangent_constrains // = "LAMBDA_CRACK_TANGENT_CONSTRAIN"
        )
        : NonlinearElasticElement::OpLhsPiolaKirchhoff_dx(vel_field, field_name,
                                                          data, common_data),
          smootherData(smoother_data),
          fieldCrackAreaTangentConstrains(crack_area_tangent_constrains) {}

    ublas::vector<int> rowFrontIndices;

    MoFEMErrorCode aSemble(int row_side, int col_side, EntityType row_type,
                           EntityType col_type,
                           EntitiesFieldData::EntData &row_data,
                           EntitiesFieldData::EntData &col_data) {
      MoFEMFunctionBegin;

      int nb_row = row_data.getIndices().size();
      int nb_col = col_data.getIndices().size();
      int *row_indices_ptr = &row_data.getIndices()[0];
      int *col_indices_ptr = &col_data.getIndices()[0];

      if (!dAta.forcesOnlyOnEntitiesRow.empty()) {
        rowIndices.resize(nb_row, false);
        noalias(rowIndices) = row_data.getIndices();
        if (!smootherData.sTabilised) {
          row_indices_ptr = &rowIndices[0];
        }
        rowFrontIndices.resize(nb_row, false);
        noalias(rowFrontIndices) = row_data.getIndices();
        VectorDofs &dofs = row_data.getFieldDofs();
        VectorDofs::iterator dit = dofs.begin();
        for (int ii = 0; dit != dofs.end(); dit++, ii++) {
          if (dAta.forcesOnlyOnEntitiesRow.find((*dit)->getEnt()) !=
              dAta.forcesOnlyOnEntitiesRow.end()) {
            rowIndices[ii] = -1;
          } else {
            rowFrontIndices[ii] = -1;
          }
        }
      }

      CHKERR MatSetValues(getFEMethod()->snes_B, nb_row, row_indices_ptr,
                          nb_col, col_indices_ptr, &k(0, 0), ADD_VALUES);

      if (smootherData.tangentFrontF) {

        const auto bit_number_for_crack_area_tangent_constrain =
            getFEMethod()->getFieldBitNumber(fieldCrackAreaTangentConstrains);
        const auto bit_number_for_mesh_position =
            getFEMethod()->getFieldBitNumber("MESH_NODE_POSITIONS");

        // get tangent vector array
        double *f_tangent_front_mesh_array;
        CHKERR VecGetArray(smootherData.tangentFrontF,
                           &f_tangent_front_mesh_array);

        auto row_dofs = getFEMethod()->getRowDofsPtr();

        // iterate nodes on tet
        for (int nn = 0; nn < 4; nn++) {

          // get indices with Lagrange multiplier at node nn
          auto dit = row_dofs->get<Unique_mi_tag>().lower_bound(
              DofEntity::getLoFieldEntityUId(
                  bit_number_for_crack_area_tangent_constrain, getConn()[nn]));
          auto hi_dit = row_dofs->get<Unique_mi_tag>().upper_bound(
              DofEntity::getHiFieldEntityUId(
                  bit_number_for_crack_area_tangent_constrain, getConn()[nn]));

          // continue if Lagrange are on element
          if (std::distance(dit, hi_dit) > 0) {

            // get mesh node positions at node nn
            auto diit = row_dofs->get<Unique_mi_tag>().lower_bound(
                DofEntity::getLoFieldEntityUId(bit_number_for_mesh_position,
                                               getConn()[nn]));

            auto hi_diit = row_dofs->get<Unique_mi_tag>().upper_bound(
                DofEntity::getHiFieldEntityUId(bit_number_for_mesh_position,
                                               getConn()[nn]));

            // iterate over dofs on node nn
            for (; diit != hi_diit; diit++) {
              // iterate overt dofs in element column
              for (int ddd = 0; ddd < nb_col; ddd++) {
                // check consistency, node has to be at crack front
                if (rowFrontIndices[3 * nn + diit->get()->getDofCoeffIdx()] !=
                    diit->get()->getPetscGlobalDofIdx()) {
                  SETERRQ2(
                      PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                      "data inconsistency %d != %d",
                      rowFrontIndices[3 * nn + diit->get()->getDofCoeffIdx()],
                      diit->get()->getPetscGlobalDofIdx());
                }
                // dof is not on this partition
                if (diit->get()->getPetscLocalDofIdx() == -1) {
                  SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
                          "data inconsistency");
                }
                double g =
                    f_tangent_front_mesh_array[diit->get()
                                                   ->getPetscLocalDofIdx()] *
                    k(3 * nn + diit->get()->getDofCoeffIdx(), ddd);
                int lambda_idx = dit->get()->getPetscGlobalDofIdx();
                CHKERR MatSetValues(getFEMethod()->snes_B, 1, &lambda_idx, 1,
                                    &col_indices_ptr[ddd], &g, ADD_VALUES);
              }
            }
          }
        }
        CHKERR VecRestoreArray(smootherData.tangentFrontF,
                               &f_tangent_front_mesh_array);
      }

      MoFEMFunctionReturn(0);
    }
  };
};

#endif //__SMOOTHER_HPP__
