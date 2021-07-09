/** \file damper_jacobian_test.cpp
  \brief Atom test testing calculation of element residual vectors and tangent
  matrices \ingroup damper
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

#include <BasicFiniteElements.hpp>
using namespace MoFEM;

namespace bio = boost::iostreams;
using bio::stream;
using bio::tee_device;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;

    {
      PetscBool flg = PETSC_TRUE;
      char mesh_file_name[255];
      CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                   mesh_file_name, 255, &flg);
      ;
      if (flg != PETSC_TRUE) {
        SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
      }
      const char *option;
      option = ""; //"PARALLEL=BCAST;";//;DEBUG_IO";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    }

    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;
    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);
    ;

    // Define fields and finite elements
    {

      // Set approximation fields
      {
        // Seed all mesh entities to MoFEM database, those entities can be
        // potentially used as finite elements or as entities which carry some
        // approximation field.

        bool check_if_spatial_field_exist =
            m_field.check_field("SPATIAL_POSITION");
        CHKERR m_field.add_field("SPATIAL_POSITION", H1,
                                 AINSWORTH_LEGENDRE_BASE, 3, MB_TAG_SPARSE,
                                 MF_ZERO);
        CHKERR m_field.add_field("SPATIAL_POSITION_DOT", H1,
                                 AINSWORTH_LEGENDRE_BASE, 3, MB_TAG_SPARSE,
                                 MF_ZERO);

        // meshset consisting all entities in mesh
        EntityHandle root_set = moab.get_root_set();
        // add entities to field (root_mesh, i.e. on all mesh entities fields
        // are approx.)
        CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET,
                                                 "SPATIAL_POSITION");
        CHKERR m_field.add_ents_to_field_by_type(root_set, MBTET,
                                                 "SPATIAL_POSITION_DOT");

        PetscBool flg;
        PetscInt order;
        CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &order,
                                  &flg);
        ;
        if (flg != PETSC_TRUE) {
          order = 2;
        }
        if (order < 2) {
          // SETERRQ()
        }

        CHKERR m_field.set_field_order(root_set, MBTET, "SPATIAL_POSITION",
                                       order);
        CHKERR m_field.set_field_order(root_set, MBTRI, "SPATIAL_POSITION",
                                       order);
        CHKERR m_field.set_field_order(root_set, MBEDGE, "SPATIAL_POSITION",
                                       order);
        CHKERR m_field.set_field_order(root_set, MBVERTEX, "SPATIAL_POSITION",
                                       1);

        CHKERR m_field.set_field_order(root_set, MBTET, "SPATIAL_POSITION_DOT",
                                       order);
        CHKERR m_field.set_field_order(root_set, MBTRI, "SPATIAL_POSITION_DOT",
                                       order);
        CHKERR m_field.set_field_order(root_set, MBEDGE, "SPATIAL_POSITION_DOT",
                                       order);
        CHKERR m_field.set_field_order(root_set, MBVERTEX,
                                       "SPATIAL_POSITION_DOT", 1);

        CHKERR m_field.build_fields();

        // Sett geometry approximation and initial spatial positions
        // 10 node tets
        if (!check_if_spatial_field_exist) {
          Projection10NodeCoordsOnField ent_method_spatial(m_field,
                                                           "SPATIAL_POSITION");
          CHKERR m_field.loop_dofs("SPATIAL_POSITION", ent_method_spatial);
        }
      }

      // Set finite elements
      {
        CHKERR m_field.add_finite_element("DAMPER_FE", MF_ZERO);
        CHKERR m_field.modify_finite_element_add_field_row("DAMPER_FE",
                                                           "SPATIAL_POSITION");
        CHKERR m_field.modify_finite_element_add_field_col("DAMPER_FE",
                                                           "SPATIAL_POSITION");

        CHKERR m_field.modify_finite_element_add_field_data("DAMPER_FE",
                                                            "SPATIAL_POSITION");
        CHKERR m_field.modify_finite_element_add_field_data(
            "DAMPER_FE", "SPATIAL_POSITION_DOT");
        EntityHandle root_set = moab.get_root_set();
        CHKERR m_field.add_ents_to_finite_element_by_type(root_set, MBTET,
                                                          "DAMPER_FE");

        // build finite elemnts
        CHKERR m_field.build_finite_elements();
        // build adjacencies
        CHKERR m_field.build_adjacencies(bit_level0);
      }
    }

    // Create damper instance
    KelvinVoigtDamper damper(m_field);
    {

      int id = 0;
      KelvinVoigtDamper::BlockMaterialData &material_data =
          damper.blockMaterialDataMap[id];
      damper.constitutiveEquationMap.insert(
          id,
          new KelvinVoigtDamper::ConstitutiveEquation<adouble>(material_data));

      // Set material parameters
      CHKERR moab.get_entities_by_type(0, MBTET, material_data.tEts);
      material_data.gBeta = 1;
      material_data.vBeta = 0.3;

      KelvinVoigtDamper::CommonData &common_data = damper.commonData;
      common_data.spatialPositionName = "SPATIAL_POSITION";
      common_data.spatialPositionNameDot = "SPATIAL_POSITION_DOT";

      KelvinVoigtDamper::DamperFE *fe_ptr[] = {&damper.feRhs, &damper.feLhs};
      for (int ss = 0; ss < 2; ss++) {
        fe_ptr[ss]->getOpPtrVector().push_back(
            new KelvinVoigtDamper::OpGetDataAtGaussPts(
                "SPATIAL_POSITION", common_data, false, true));
        fe_ptr[ss]->getOpPtrVector().push_back(
            new KelvinVoigtDamper::OpGetDataAtGaussPts(
                "SPATIAL_POSITION_DOT", common_data, false, true));
      }

      // attach tags for each recorder
      std::vector<int> tags;
      tags.push_back(1);

      KelvinVoigtDamper::ConstitutiveEquation<adouble> &ce =
          damper.constitutiveEquationMap.at(id);

      // Right hand side operators
      damper.feRhs.getOpPtrVector().push_back(new KelvinVoigtDamper::OpJacobian(
          "SPATIAL_POSITION", tags, ce, damper.commonData, true, false));
      damper.feRhs.getOpPtrVector().push_back(
          new KelvinVoigtDamper::OpRhsStress(damper.commonData));

      // Left hand side operators
      damper.feLhs.getOpPtrVector().push_back(new KelvinVoigtDamper::OpJacobian(
          "SPATIAL_POSITION", tags, ce, damper.commonData, false, true));
      damper.feLhs.getOpPtrVector().push_back(
          new KelvinVoigtDamper::OpLhsdxdx(damper.commonData));
    }

    // Create dm instance
    DM dm;
    DMType dm_name = "DMDAMPER";
    {
      CHKERR DMRegister_MoFEM(dm_name);
      CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
      CHKERR DMSetType(dm, dm_name);
      CHKERR DMMoFEMCreateMoFEM(dm, &m_field, dm_name, bit_level0);
      CHKERR DMSetFromOptions(dm);
      // add elements to dm
      CHKERR DMMoFEMAddElement(dm, "DAMPER_FE");
      CHKERR DMSetUp(dm);
    }

    // Make calculations
    Mat M;
    Vec F, U_t;
    {
      CHKERR DMCreateGlobalVector_MoFEM(dm, &F);
      CHKERR DMCreateGlobalVector_MoFEM(dm, &U_t);
      CHKERR DMCreateMatrix_MoFEM(dm, &M);
      CHKERR VecZeroEntries(F);
      CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecZeroEntries(U_t);
      CHKERR VecGhostUpdateBegin(U_t, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(U_t, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR MatZeroEntries(M);
      damper.feRhs.ts_F = F; // Set right hand side vector manually
      damper.feRhs.ts_u_t = U_t;
      damper.feRhs.ts_ctx = TSMethod::CTX_TSSETIFUNCTION;
      CHKERR DMoFEMLoopFiniteElements(dm, "DAMPER_FE", &damper.feRhs);
      damper.feLhs.ts_B = M;   // Set matrix M
      damper.feLhs.ts_a = 1.0; // Set time step parameter
      CHKERR DMoFEMLoopFiniteElements(dm, "DAMPER_FE", &damper.feLhs);
      CHKERR VecAssemblyBegin(F);
      CHKERR VecAssemblyEnd(F);
      CHKERR MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
      CHKERR MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    }

    // See results
    {

      PetscViewer viewer;
      CHKERR PetscViewerASCIIOpen(PETSC_COMM_WORLD, "damper_jacobian_test.txt",
                                  &viewer);
      CHKERR PetscViewerDestroy(&viewer);
      // MatView(M,PETSC_VIEWER_DRAW_WORLD);
      // std::string wait;
      // std::cin >> wait;
    }

    // Clean and destroy
    {
      CHKERR VecDestroy(&F);
      CHKERR VecDestroy(&U_t);
      CHKERR MatDestroy(&M);
      CHKERR DMDestroy(&dm);
    }
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
