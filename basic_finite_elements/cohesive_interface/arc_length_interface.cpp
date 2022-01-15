/** \file arc_length_interface.cpp
  * \brief Example of arc-length with interface element

  \todo Mak it work with multi-grid and distributed mesh

  \todo Make more clever step adaptation

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

#include <CohesiveInterfaceElement.hpp>
#include <Hooke.hpp>
#include <InterfaceGapArcLengthControl.hpp>

using namespace boost::numeric;

static char help[] = "\
 -my_file mesh file name\n\
 -my_sr reduction of step size\n\
 -my_its_d desired number of steps\n\
 -my_ms maximal number of steps\n\n";

#define DATAFILENAME "load_disp.txt"

namespace CohesiveElement {

struct ArcLengthElement : public ArcLengthIntElemFEMethod {
  MoFEM::Interface &mField;
  Range postProcNodes;
  ArcLengthElement(MoFEM::Interface &m_field,
                   boost::shared_ptr<ArcLengthCtx> &arc_ptr)
      : ArcLengthIntElemFEMethod(m_field.get_moab(), arc_ptr), mField(m_field) {

    for (_IT_CUBITMESHSETS_BY_NAME_FOR_LOOP_(mField, "LoadPath", cit)) {
      EntityHandle meshset = cit->getMeshset();
      Range nodes;
      rval = mOab.get_entities_by_type(meshset, MBVERTEX, nodes, true);
      MOAB_THROW(rval);
      postProcNodes.merge(nodes);
    }

    PetscPrintf(PETSC_COMM_WORLD, "Nb. PostProcNodes %lu\n",
                postProcNodes.size());
  };

  MoFEMErrorCode postProcessLoadPath() {
    MoFEMFunctionBegin;
    FILE *datafile;
    PetscFOpen(PETSC_COMM_SELF, DATAFILENAME, "a+", &datafile);
    const auto bit_number_lambda = mField.get_field_bit_number("LAMBDA");
    
    boost::shared_ptr<NumeredDofEntity_multiIndex> numered_dofs_rows =
        problemPtr->getNumeredRowDofsPtr();
    auto lit = numered_dofs_rows->lower_bound(
        FieldEntity::getLoBitNumberUId(bit_number_lambda));
    auto hi_lit = numered_dofs_rows->upper_bound(
        FieldEntity::getHiBitNumberUId(bit_number_lambda));

    if (lit == numered_dofs_rows->end()) {
      fclose(datafile);
      MoFEMFunctionReturnHot(0);
    } else if (std::distance(lit, hi_lit) != 1) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "Only one DOF is expected");
    }

    Range::iterator nit = postProcNodes.begin();
    for (; nit != postProcNodes.end(); nit++) {
      NumeredDofEntityByEnt::iterator dit, hi_dit;
      dit = numered_dofs_rows->get<Ent_mi_tag>().lower_bound(*nit);
      hi_dit = numered_dofs_rows->get<Ent_mi_tag>().upper_bound(*nit);
      double coords[3];
      CHKERR mOab.get_coords(&*nit, 1, coords);
      for (; dit != hi_dit; dit++) {
        PetscPrintf(PETSC_COMM_WORLD, "%s [ %d ] %6.4e -> ",
                    lit->get()->getName().c_str(), lit->get()->getDofCoeffIdx(),
                    lit->get()->getFieldData());
        PetscPrintf(PETSC_COMM_WORLD, "%s [ %d ] %6.4e ",
                    dit->get()->getName().c_str(), dit->get()->getDofCoeffIdx(),
                    dit->get()->getFieldData());
        PetscPrintf(PETSC_COMM_WORLD, "-> %3.4f %3.4f %3.4f\n", coords[0],
                    coords[1], coords[2]);
        PetscFPrintf(PETSC_COMM_WORLD, datafile, "%6.4e %6.4e ",
                     dit->get()->getFieldData(), lit->get()->getFieldData());
      }
    }
    PetscFPrintf(PETSC_COMM_WORLD, datafile, "\n");
    fclose(datafile);
    MoFEMFunctionReturn(0);
  }
};

struct AssembleRhsVectors : public FEMethod {

  MoFEM::Interface &mField;
  Vec &bodyForce;
  boost::shared_ptr<ArcLengthCtx> arcPtr;

  AssembleRhsVectors(MoFEM::Interface &m_field, Vec &body_force,
                     boost::shared_ptr<ArcLengthCtx> &arc_ptr)
      : mField(m_field), bodyForce(body_force), arcPtr(arc_ptr) {}

  MoFEMErrorCode preProcess() {
    MoFEMFunctionBegin;

    switch (snes_ctx) {
    case CTX_SNESNONE: {
    } break;
    case CTX_SNESSETFUNCTION: {
      CHKERR VecZeroEntries(snes_f);
      CHKERR VecGhostUpdateBegin(snes_f, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(snes_f, INSERT_VALUES, SCATTER_FORWARD);
    } break;
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "not implemented");
    }

    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode postProcess() {
    MoFEMFunctionBegin;
    switch (snes_ctx) {
    case CTX_SNESNONE: {
    } break;
    case CTX_SNESSETFUNCTION: {
      CHKERR VecGhostUpdateBegin(snes_f, ADD_VALUES, SCATTER_REVERSE);
      CHKERR VecGhostUpdateEnd(snes_f, ADD_VALUES, SCATTER_REVERSE);
      CHKERR VecAssemblyBegin(snes_f);
      CHKERR VecAssemblyEnd(snes_f);
      // add F_lambda
      CHKERR VecAXPY(snes_f, arcPtr->getFieldData(), arcPtr->F_lambda);
      CHKERR VecAXPY(snes_f, -1., bodyForce);
      PetscPrintf(PETSC_COMM_WORLD, "\tlambda = %6.4e\n",
                  arcPtr->getFieldData());
      // snes_f norm
      double fnorm;
      CHKERR VecNorm(snes_f, NORM_2, &fnorm);
      PetscPrintf(PETSC_COMM_WORLD, "\tfnorm = %6.4e\n", fnorm);
    } break;
    default:
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED, "not implemented");
    }

    MoFEMFunctionReturn(0);
  }
};

} // namespace CohesiveElement

using namespace CohesiveElement;

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps\n"
                                 "-mat_mumps_icntl_20 0\n" 
                                 "-ksp_monitor \n"
                                 "-ksp_atol 1e-10 \n"
                                 "-ksp_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type l2 \n"
                                 "-snes_linesearch_monitor \n"
                                 "-snes_max_it 16 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-snes_converged_reason \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }
  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Reade parameters from line command
    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                 mesh_file_name, 255, &flg);
    if (flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_INVALID_DATA,
              "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    PetscScalar step_size_reduction;
    CHKERR PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-my_sr",
                               &step_size_reduction, &flg);
    if (flg != PETSC_TRUE) {
      step_size_reduction = 1.;
    }

    PetscInt max_steps;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_ms", &max_steps,
                              &flg);
    if (flg != PETSC_TRUE) {
      max_steps = 5;
    }

    int its_d;
    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-my_its_d", &its_d, &flg);
    if (flg != PETSC_TRUE) {
      its_d = 6;
    }

    PetscInt order;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &order,
                              &flg);
    if (flg != PETSC_TRUE) {
      order = 2;
    }

    // Check if new start or restart. If new start, delete previous
    // load_disp.txt
    if (std::string(mesh_file_name).find("restart") == std::string::npos) {
      remove(DATAFILENAME);
    }

    // Read mesh to MOAB
    const char *option;
    option = ""; 
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Data stored on mesh for restart
    Tag th_step_size, th_step;
    double def_step_size = 1;
    rval = moab.tag_get_handle("_STEPSIZE", 1, MB_TYPE_DOUBLE, th_step_size,
                               MB_TAG_CREAT | MB_TAG_MESH, &def_step_size);
    if (rval == MB_ALREADY_ALLOCATED)
      rval = MB_SUCCESS;
    CHKERRG(rval);
    int def_step = 1;
    rval = moab.tag_get_handle("_STEP", 1, MB_TYPE_INTEGER, th_step,
                               MB_TAG_CREAT | MB_TAG_MESH, &def_step);
    if (rval == MB_ALREADY_ALLOCATED)
      rval = MB_SUCCESS;
    CHKERRG(rval);
    const void *tag_data_step_size[1];
    EntityHandle root = 0;
    CHKERR moab.tag_get_by_ptr(th_step_size, &root, 1, tag_data_step_size);
    double &step_size = *(double *)tag_data_step_size[0];
    const void *tag_data_step[1];
    CHKERR moab.tag_get_by_ptr(th_step, &root, 1, tag_data_step);
    int &step = *(int *)tag_data_step[0];
    // end of data stored for restart
    CHKERR PetscPrintf(PETSC_COMM_WORLD,
                       "Start step %D and step_size = %6.4e\n", step,
                       step_size);

    // Create MoFEM 2database
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    PrismInterface *interface_ptr;
    CHKERR m_field.getInterface(interface_ptr);

    Tag th_my_ref_level;
    BitRefLevel def_bit_level = 0;
    CHKERR m_field.get_moab().tag_get_handle(
        "_MY_REFINEMENT_LEVEL", sizeof(BitRefLevel), MB_TYPE_OPAQUE,
        th_my_ref_level, MB_TAG_CREAT | MB_TAG_SPARSE | MB_TAG_BYTES,
        &def_bit_level);
    const EntityHandle root_meshset = m_field.get_moab().get_root_set();
    BitRefLevel *ptr_bit_level0;
    CHKERR m_field.get_moab().tag_get_by_ptr(th_my_ref_level, &root_meshset, 1,
                                             (const void **)&ptr_bit_level0);
    BitRefLevel &bit_level0 = *ptr_bit_level0;
    BitRefLevel problem_bit_level = bit_level0;

    if (step == 1) {

      // ref meshset ref level 0
      CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
          0, 3, BitRefLevel().set(0));

      std::vector<BitRefLevel> bit_levels;
      bit_levels.push_back(BitRefLevel().set(0));

      int ll = 1;

      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
               m_field, SIDESET | INTERFACESET, cit)) {
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Insert Interface %d\n",
                           cit->getMeshsetId());
        EntityHandle cubit_meshset = cit->getMeshset();
        {
          // get tet entities form back bit_level
          EntityHandle ref_level_meshset = 0;
          CHKERR moab.create_meshset(MESHSET_SET, ref_level_meshset);
          CHKERR m_field.getInterface<BitRefManager>()
              ->getEntitiesByTypeAndRefLevel(bit_levels.back(),
                                             BitRefLevel().set(), MBTET,
                                             ref_level_meshset);
          CHKERR m_field.getInterface<BitRefManager>()
              ->getEntitiesByTypeAndRefLevel(bit_levels.back(),
                                             BitRefLevel().set(), MBPRISM,
                                             ref_level_meshset);
          Range ref_level_tets;
          CHKERR moab.get_entities_by_handle(ref_level_meshset, ref_level_tets,
                                             true);
          // get faces and test to split
          CHKERR interface_ptr->getSides(cubit_meshset, bit_levels.back(), true,
                                         0);
          // set new bit level
          bit_levels.push_back(BitRefLevel().set(ll++));
          // split faces and
          CHKERR interface_ptr->splitSides(ref_level_meshset, bit_levels.back(),
                                           cubit_meshset, true, true, 0);
          // clean meshsets
          CHKERR moab.delete_entities(&ref_level_meshset, 1);
        }
        // Update cubit meshsets
        for (_IT_CUBITMESHSETS_FOR_LOOP_(m_field, ciit)) {
          EntityHandle cubit_meshset = ciit->meshset;
          CHKERR m_field.getInterface<BitRefManager>()
              ->updateMeshsetByEntitiesChildren(cubit_meshset,
                                                bit_levels.back(),
                                                cubit_meshset, MBVERTEX, true);
          CHKERR m_field.getInterface<BitRefManager>()
              ->updateMeshsetByEntitiesChildren(cubit_meshset,
                                                bit_levels.back(),
                                                cubit_meshset, MBEDGE, true);
          CHKERR m_field.getInterface<BitRefManager>()
              ->updateMeshsetByEntitiesChildren(
                  cubit_meshset, bit_levels.back(), cubit_meshset, MBTRI, true);
          CHKERR m_field.getInterface<BitRefManager>()
              ->updateMeshsetByEntitiesChildren(
                  cubit_meshset, bit_levels.back(), cubit_meshset, MBTET, true);
        }
      }

      bit_level0 = bit_levels.back();
      problem_bit_level = bit_level0;

      /***/
      // Define problem

      // Fields
      CHKERR m_field.add_field("DISPLACEMENT", H1, AINSWORTH_LEGENDRE_BASE, 3);
      CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1,
                               AINSWORTH_LEGENDRE_BASE, 3);

      CHKERR m_field.add_field("LAMBDA", NOFIELD, NOBASE, 1);

      // Field for ArcLength
      CHKERR
      m_field.add_field("X0_DISPLACEMENT", H1, AINSWORTH_LEGENDRE_BASE, 3);

      // FE
      CHKERR m_field.add_finite_element("ELASTIC");

      // Define rows/cols and element data
      CHKERR m_field.modify_finite_element_add_field_row("ELASTIC",
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_col("ELASTIC",
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data("ELASTIC",
                                                          "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data(
          "ELASTIC", "MESH_NODE_POSITIONS");
      CHKERR m_field.modify_finite_element_add_field_row("ELASTIC", "LAMBDA");
      CHKERR m_field.modify_finite_element_add_field_col("ELASTIC", "LAMBDA");
      // this is for paremtis
      CHKERR m_field.modify_finite_element_add_field_data("ELASTIC", "LAMBDA");

      // FE Interface
      CHKERR m_field.add_finite_element("INTERFACE");
      CHKERR m_field.modify_finite_element_add_field_row("INTERFACE",
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_col("INTERFACE",
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data("INTERFACE",
                                                          "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data(
          "INTERFACE", "MESH_NODE_POSITIONS");

      // FE ArcLength
      CHKERR m_field.add_finite_element("ARC_LENGTH");

      // Define rows/cols and element data
      CHKERR m_field.modify_finite_element_add_field_row("ARC_LENGTH",
                                                         "LAMBDA");
      CHKERR m_field.modify_finite_element_add_field_col("ARC_LENGTH",
                                                         "LAMBDA");

      // elem data
      CHKERR
      m_field.modify_finite_element_add_field_data("ARC_LENGTH", "LAMBDA");

      // define problems
      CHKERR m_field.add_problem("ELASTIC_MECHANICS");

      // set finite elements for problem
      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "ELASTIC");
      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "INTERFACE");
      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "ARC_LENGTH");
      // set refinement level for problem
      CHKERR m_field.modify_problem_ref_level_add_bit("ELASTIC_MECHANICS",
                                                      problem_bit_level);

      /***/
      // Declare problem

      // add entities (by tets) to the field
      CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "DISPLACEMENT");
      CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "X0_DISPLACEMENT");
      CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");

      // add finite elements entities
      CHKERR m_field.add_ents_to_finite_element_by_bit_ref(
          problem_bit_level, BitRefLevel().set(), "ELASTIC", MBTET);
      CHKERR m_field.add_ents_to_finite_element_by_bit_ref(
          problem_bit_level, BitRefLevel().set(), "INTERFACE", MBPRISM);

      // Setting up LAMBDA field and ARC_LENGTH interface
      {
        // Add dummy no-field vertex
        EntityHandle no_field_vertex;
        {
          const double coords[] = {0, 0, 0};
          CHKERR m_field.get_moab().create_vertex(coords, no_field_vertex);
          Range range_no_field_vertex;
          range_no_field_vertex.insert(no_field_vertex);
          CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevel(
              range_no_field_vertex, BitRefLevel().set());

          EntityHandle lambda_meshset = m_field.get_field_meshset("LAMBDA");
          CHKERR m_field.get_moab().add_entities(lambda_meshset,
                                                 range_no_field_vertex);
        }
        // this entity will carray data for this finite element
        EntityHandle meshset_fe_arc_length;
        {
          CHKERR moab.create_meshset(MESHSET_SET, meshset_fe_arc_length);
          CHKERR moab.add_entities(meshset_fe_arc_length, &no_field_vertex, 1);
          CHKERR m_field.getInterface<BitRefManager>()->setBitLevelToMeshset(
              meshset_fe_arc_length, BitRefLevel().set());
        }
        // finally add created meshset to the ARC_LENGTH finite element
        CHKERR m_field.add_ents_to_finite_element_by_MESHSET(
            meshset_fe_arc_length, "ARC_LENGTH", false);
      }

      // set app. order
      // see Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes
      // (Mark Ainsworth & Joe Coyle)
      CHKERR m_field.set_field_order(0, MBTET, "DISPLACEMENT", order);
      CHKERR m_field.set_field_order(0, MBTRI, "DISPLACEMENT", order);
      CHKERR m_field.set_field_order(0, MBEDGE, "DISPLACEMENT", order);
      CHKERR m_field.set_field_order(0, MBVERTEX, "DISPLACEMENT", 1);

      CHKERR m_field.set_field_order(0, MBTET, "X0_DISPLACEMENT", order);
      CHKERR m_field.set_field_order(0, MBTRI, "X0_DISPLACEMENT", order);
      CHKERR m_field.set_field_order(0, MBEDGE, "X0_DISPLACEMENT", order);
      CHKERR m_field.set_field_order(0, MBVERTEX, "X0_DISPLACEMENT", 1);

      CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

      // Elements with boundary conditions
      CHKERR MetaNeumannForces::addNeumannBCElements(m_field, "DISPLACEMENT");
      CHKERR MetaNodalForces::addElement(m_field, "DISPLACEMENT");

      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "FORCE_FE");
      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "PRESSURE_FE");
      CHKERR m_field.modify_finite_element_add_field_row("FORCE_FE", "LAMBDA");
      CHKERR m_field.modify_finite_element_add_field_col("FORCE_FE", "LAMBDA");
      CHKERR m_field.modify_finite_element_add_field_data("FORCE_FE", "LAMBDA");

      CHKERR m_field.modify_finite_element_add_field_row("PRESSURE_FE",
                                                         "LAMBDA");
      CHKERR m_field.modify_finite_element_add_field_col("PRESSURE_FE",
                                                         "LAMBDA");
      CHKERR m_field.modify_finite_element_add_field_data("PRESSURE_FE",
                                                          "LAMBDA");
      CHKERR m_field.add_finite_element("BODY_FORCE");
      CHKERR m_field.modify_finite_element_add_field_row("BODY_FORCE",
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_col("BODY_FORCE",
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data("BODY_FORCE",
                                                          "DISPLACEMENT");
      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "BODY_FORCE");

      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
               m_field, BLOCKSET | BODYFORCESSET, it)) {
        Range tets;
        CHKERR m_field.get_moab().get_entities_by_type(it->meshset, MBTET, tets,
                                                       true);
        CHKERR m_field.add_ents_to_finite_element_by_type(tets, MBTET,
                                                          "BODY_FORCE");
      }
    }

    /****/
    // build database

    // build field
    CHKERR m_field.build_fields();
    Projection10NodeCoordsOnField ent_method_material(m_field,
                                                      "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(problem_bit_level);

    /****/
    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    // build problem
    CHKERR prb_mng_ptr->buildProblem("ELASTIC_MECHANICS", true);
    // partition
    CHKERR prb_mng_ptr->partitionProblem("ELASTIC_MECHANICS");
    CHKERR prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS", false, 0,
                                                m_field.get_comm_size());
    // what are ghost nodes, see Petsc Manual
    CHKERR prb_mng_ptr->partitionGhostDofs("ELASTIC_MECHANICS");

    // print bcs
    MeshsetsManager *mmanager_ptr;
    CHKERR m_field.getInterface(mmanager_ptr);
    CHKERR mmanager_ptr->printDisplacementSet();
    CHKERR mmanager_ptr->printForceSet();
    // print block sets with materials
    CHKERR mmanager_ptr->printMaterialsSet();

    // create matrices
    Vec F, F_body_force, D;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost(
        "ELASTIC_MECHANICS", COL, &F);
    CHKERR VecDuplicate(F, &D);
    CHKERR VecDuplicate(F, &F_body_force);
    Mat Aij;
    CHKERR m_field.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("ELASTIC_MECHANICS",
                                                        &Aij);

    // Assemble F and Aij
    double young_modulus = 1;
    double poisson_ratio = 0.0;

    boost::ptr_vector<CohesiveInterfaceElement::PhysicalEquation>
        interface_materials;

    // FIXME this in fact allow only for one type of interface,
    // problem is Young Modulus in interface mayoung_modulusterial
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      cout << std::endl << *it << std::endl;

      // Get block name
      string name = it->getName();

      if (name.compare(0, 11, "MAT_ELASTIC") == 0) {
        Mat_Elastic mydata;
        CHKERR it->getAttributeDataStructure(mydata);
        cout << mydata;
        young_modulus = mydata.data.Young;
        poisson_ratio = mydata.data.Poisson;
      } else if (name.compare(0, 10, "MAT_INTERF") == 0) {
        Mat_Interf mydata;
        CHKERR it->getAttributeDataStructure(mydata);
        cout << mydata;

        interface_materials.push_back(
            new CohesiveInterfaceElement::PhysicalEquation(m_field));
        interface_materials.back().h = 1;
        interface_materials.back().youngModulus = mydata.data.alpha;
        interface_materials.back().beta = mydata.data.beta;
        interface_materials.back().ft = mydata.data.ft;
        interface_materials.back().Gf = mydata.data.Gf;

        EntityHandle meshset = it->getMeshset();
        Range tris;
        rval = moab.get_entities_by_type(meshset, MBTRI, tris, true);
        CHKERRG(rval);
        Range ents3d;
        rval = moab.get_adjacencies(tris, 3, false, ents3d,
                                    moab::Interface::UNION);
        CHKERRG(rval);
        interface_materials.back().pRisms = ents3d.subset_by_type(MBPRISM);
      }
    }

    { // FIXME
      boost::ptr_vector<CohesiveInterfaceElement::PhysicalEquation>::iterator
          pit = interface_materials.begin();
      for (; pit != interface_materials.end(); pit++) {
        pit->youngModulus = young_modulus;
      }
    }

    boost::shared_ptr<ArcLengthCtx> arc_ctx = boost::shared_ptr<ArcLengthCtx>(
        new ArcLengthCtx(m_field, "ELASTIC_MECHANICS"));
    boost::scoped_ptr<ArcLengthElement> my_arc_method_ptr(
        new ArcLengthElement(m_field, arc_ctx));
    ArcLengthSnesCtx snes_ctx(m_field, "ELASTIC_MECHANICS", arc_ctx);
    AssembleRhsVectors pre_post_proc_fe(m_field, F_body_force, arc_ctx);

    DirichletDisplacementBc my_dirichlet_bc(m_field, "DISPLACEMENT", Aij, D, F);
    CHKERR m_field.get_problem("ELASTIC_MECHANICS",
                               &my_dirichlet_bc.problemPtr);
    CHKERR my_dirichlet_bc.iNitalize();
    boost::shared_ptr<Hooke<adouble>> hooke_adouble_ptr(new Hooke<adouble>);
    boost::shared_ptr<Hooke<double>> hooke_double_ptr(new Hooke<double>);
    NonlinearElasticElement elastic(m_field, 2);
    {
      int id = 0;
      elastic.setOfBlocks[id].iD = id;
      elastic.setOfBlocks[id].E = young_modulus;
      elastic.setOfBlocks[id].PoissonRatio = poisson_ratio;
      elastic.setOfBlocks[id].materialDoublePtr = hooke_double_ptr;
      elastic.setOfBlocks[id].materialAdoublePtr = hooke_adouble_ptr;
      CHKERR m_field.get_moab().get_entities_by_type(
          m_field.get_finite_element_meshset("ELASTIC"), MBTET,
          elastic.setOfBlocks[id].tEts, true);
      CHKERR addHOOpsVol("MESH_NODE_POSITIONS", elastic.getLoopFeRhs(), true,
                      false, false, false);
      CHKERR addHOOpsVol("MESH_NODE_POSITIONS", elastic.getLoopFeRhs(), true,
                      false, false, false);
      CHKERR addHOOpsVol("MESH_NODE_POSITIONS", elastic.getLoopFeEnergy(), true,
                      false, false, false);
      CHKERR elastic.setOperators("DISPLACEMENT", "MESH_NODE_POSITIONS", false,
                                  true);
    }
    CohesiveInterfaceElement cohesive_elements(m_field);
    CHKERR cohesive_elements.addOps("DISPLACEMENT", interface_materials);

    PetscInt M, N;
    CHKERR MatGetSize(Aij, &M, &N);
    PetscInt m, n;
    MatGetLocalSize(Aij, &m, &n);
    boost::scoped_ptr<ArcLengthMatShell> mat_ctx(
        new ArcLengthMatShell(Aij, arc_ctx, "ELASTIC_MECHANICS"));
    Mat ShellAij;
    CHKERR MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, (void *)mat_ctx.get(),
                          &ShellAij);
    CHKERR MatShellSetOperation(ShellAij, MATOP_MULT,
                                (void (*)(void))ArcLengthMatMultShellOp);

    // body forces
    BodyForceConstantField body_forces_methods(m_field);
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, BLOCKSET | BODYFORCESSET, it)) {
      CHKERR body_forces_methods.addBlock("DISPLACEMENT", F_body_force,
                                          it->getMeshsetId());
    }
    CHKERR VecZeroEntries(F_body_force);
    CHKERR VecGhostUpdateBegin(F_body_force, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F_body_force, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "BODY_FORCE",
                                        body_forces_methods.getLoopFe());
    CHKERR VecGhostUpdateBegin(F_body_force, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F_body_force, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F_body_force);
    CHKERR VecAssemblyEnd(F_body_force);

    // surface forces
    boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
    string fe_name_str = "FORCE_FE";
    neumann_forces.insert(fe_name_str, new NeumannForcesSurface(m_field));
    CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS",
                          neumann_forces.at(fe_name_str).getLoopFe(), false,
                          false);
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      CHKERR neumann_forces.at(fe_name_str)
          .addForce("DISPLACEMENT", arc_ctx->F_lambda, it->getMeshsetId());
    }
    fe_name_str = "PRESSURE_FE";
    neumann_forces.insert(fe_name_str, new NeumannForcesSurface(m_field));
    CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS",
                          neumann_forces.at(fe_name_str).getLoopFe(), false,
                          false);
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      CHKERR neumann_forces.at(fe_name_str)
          .addPressure("DISPLACEMENT", arc_ctx->F_lambda, it->getMeshsetId());
    }
    // add nodal forces
    boost::ptr_map<std::string, NodalForce> nodal_forces;
    fe_name_str = "FORCE_FE";
    nodal_forces.insert(fe_name_str, new NodalForce(m_field));
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      CHKERR nodal_forces.at(fe_name_str)
          .addForce("DISPLACEMENT", arc_ctx->F_lambda, it->getMeshsetId());
    }

    SNES snes;
    CHKERR SNESCreate(PETSC_COMM_WORLD, &snes);
    CHKERR SNESSetApplicationContext(snes, &snes_ctx);
    CHKERR SNESSetFunction(snes, F, SnesRhs, &snes_ctx);
    CHKERR SNESSetJacobian(snes, ShellAij, Aij, SnesMat, &snes_ctx);
    CHKERR SNESSetFromOptions(snes);

    KSP ksp;
    CHKERR SNESGetKSP(snes, &ksp);
    PC pc;
    CHKERR KSPGetPC(ksp, &pc);
    boost::scoped_ptr<PCArcLengthCtx> pc_ctx(
        new PCArcLengthCtx(ShellAij, Aij, arc_ctx));
    CHKERR PCSetType(pc, PCSHELL);
    CHKERR PCShellSetContext(pc, pc_ctx.get());
    CHKERR PCShellSetApply(pc, PCApplyArcLength);
    CHKERR PCShellSetSetUp(pc, PCSetupArcLength);

    // Rhs
    SnesCtx::FEMethodsSequence &loops_to_do_Rhs =
        snes_ctx.get_loops_to_do_Rhs();
    snes_ctx.get_preProcess_to_do_Rhs().push_back(&my_dirichlet_bc);
    snes_ctx.get_preProcess_to_do_Rhs().push_back(&pre_post_proc_fe);
    loops_to_do_Rhs.push_back(SnesCtx::PairNameFEMethodPtr(
        "INTERFACE", &cohesive_elements.getFeRhs()));
    loops_to_do_Rhs.push_back(
        SnesCtx::PairNameFEMethodPtr("ELASTIC", &elastic.getLoopFeRhs()));
    loops_to_do_Rhs.push_back(
        SnesCtx::PairNameFEMethodPtr("ARC_LENGTH", my_arc_method_ptr.get()));
    snes_ctx.get_postProcess_to_do_Rhs().push_back(&pre_post_proc_fe);
    snes_ctx.get_postProcess_to_do_Rhs().push_back(&my_dirichlet_bc);

    // Mat
    SnesCtx::FEMethodsSequence &loops_to_do_Mat =
        snes_ctx.get_loops_to_do_Mat();
    snes_ctx.get_preProcess_to_do_Mat().push_back(&my_dirichlet_bc);
    loops_to_do_Mat.push_back(SnesCtx::PairNameFEMethodPtr(
        "INTERFACE", &cohesive_elements.getFeLhs()));
    loops_to_do_Mat.push_back(
        SnesCtx::PairNameFEMethodPtr("ELASTIC", &elastic.getLoopFeLhs()));
    loops_to_do_Mat.push_back(
        SnesCtx::PairNameFEMethodPtr("ARC_LENGTH", my_arc_method_ptr.get()));
    snes_ctx.get_postProcess_to_do_Mat().push_back(&my_dirichlet_bc);

    double gamma = 0.5, reduction = 1;
    // step = 1;
    if (step == 1) {
      step_size = step_size_reduction;
    } else {
      reduction = step_size_reduction;
      step++;
    }

    boost::ptr_map<std::string, NeumannForcesSurface>::iterator mit =
        neumann_forces.begin();
    CHKERR VecZeroEntries(arc_ctx->F_lambda);
    CHKERR VecGhostUpdateBegin(arc_ctx->F_lambda, INSERT_VALUES,
                               SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(arc_ctx->F_lambda, INSERT_VALUES, SCATTER_FORWARD);
    for (; mit != neumann_forces.end(); mit++) {
      CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", mit->first,
                                          mit->second->getLoopFe());
    }
    CHKERR VecGhostUpdateBegin(arc_ctx->F_lambda, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(arc_ctx->F_lambda, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(arc_ctx->F_lambda);
    CHKERR VecAssemblyEnd(arc_ctx->F_lambda);
    for (std::vector<int>::iterator vit = my_dirichlet_bc.dofsIndices.begin();
         vit != my_dirichlet_bc.dofsIndices.end(); vit++) {
      CHKERR VecSetValue(arc_ctx->F_lambda, *vit, 0, INSERT_VALUES);
    }
    CHKERR VecAssemblyBegin(arc_ctx->F_lambda);
    CHKERR VecAssemblyEnd(arc_ctx->F_lambda);
    // F_lambda2
    CHKERR VecDot(arc_ctx->F_lambda, arc_ctx->F_lambda, &arc_ctx->F_lambda2);
    PetscPrintf(PETSC_COMM_WORLD, "\tFlambda2 = %6.4e\n", arc_ctx->F_lambda2);

    if (step > 1) {
      CHKERR m_field.getInterface<VecManager>()->setLocalGhostVector(
          "ELASTIC_MECHANICS", COL, D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR m_field.getInterface<VecManager>()->setOtherGlobalGhostVector(
          "ELASTIC_MECHANICS", "DISPLACEMENT", "X0_DISPLACEMENT", COL,
          arc_ctx->x0, INSERT_VALUES, SCATTER_FORWARD);
      double x0_nrm;
      CHKERR VecNorm(arc_ctx->x0, NORM_2, &x0_nrm);
      CHKERR PetscPrintf(PETSC_COMM_WORLD,
                         "\tRead x0_nrm = %6.4e dlambda = %6.4e\n", x0_nrm,
                         arc_ctx->dLambda);
      CHKERR arc_ctx->setAlphaBeta(1, 0);
    } else {
      CHKERR arc_ctx->setS(0);
      CHKERR arc_ctx->setAlphaBeta(0, 1);
    }
    CHKERR SnesRhs(snes, D, F, &snes_ctx);

    PostProcVolumeOnRefinedMesh post_proc(m_field);
    CHKERR post_proc.generateReferenceElementMesh();
    CHKERR post_proc.addFieldValuesPostProc("DISPLACEMENT");
    CHKERR post_proc.addFieldValuesGradientPostProc("DISPLACEMENT");
    // add postpocessing for sresses
    post_proc.getOpPtrVector().push_back(new PostProcHookStress(
        m_field, post_proc.postProcMesh, post_proc.mapGaussPts, "DISPLACEMENT",
        post_proc.commonData));

    bool converged_state = false;
    for (; step < max_steps; step++) {

      if (step == 1) {
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Load Step %D step_size = %6.4e\n",
                           step, step_size);
        CHKERR arc_ctx->setS(step_size);
        CHKERR arc_ctx->setAlphaBeta(0, 1);
        CHKERR VecCopy(D, arc_ctx->x0);
        double dlambda;
        CHKERR my_arc_method_ptr->calculate_init_dlambda(&dlambda);
        CHKERR my_arc_method_ptr->set_dlambda_to_x(D, dlambda);
      } else if (step == 2) {
        CHKERR arc_ctx->setAlphaBeta(1, 0);
        CHKERR my_arc_method_ptr->calculate_dx_and_dlambda(D);
        CHKERR my_arc_method_ptr->calculate_lambda_int(step_size);
        CHKERR arc_ctx->setS(step_size);
        double dlambda = arc_ctx->dLambda;
        double dx_nrm;
        CHKERR VecNorm(arc_ctx->dx, NORM_2, &dx_nrm);
        CHKERR PetscPrintf(PETSC_COMM_WORLD,
                           "Load Step %D step_size = %6.4e dlambda0 = %6.4e "
                           "dx_nrm = %6.4e dx2 = %6.4e\n",
                           step, step_size, dlambda, dx_nrm, arc_ctx->dx2);
        CHKERR VecCopy(D, arc_ctx->x0);
        CHKERR VecAXPY(D, 1., arc_ctx->dx);
        CHKERR my_arc_method_ptr->set_dlambda_to_x(D, dlambda);
      } else {
        CHKERR my_arc_method_ptr->calculate_dx_and_dlambda(D);
        CHKERR my_arc_method_ptr->calculate_lambda_int(step_size);
        // step_size0_1/step_size0 = step_stize1/step_size
        // step_size0_1 = step_size0*(step_stize1/step_size)
        step_size *= reduction;
        CHKERR arc_ctx->setS(step_size);
        double dlambda = reduction * arc_ctx->dLambda;
        CHKERR VecScale(arc_ctx->dx, reduction);
        double dx_nrm;
        CHKERR VecNorm(arc_ctx->dx, NORM_2, &dx_nrm);
        CHKERR PetscPrintf(PETSC_COMM_WORLD,
                           "Load Step %D step_size = %6.4e dlambda0 = %6.4e "
                           "dx_nrm = %6.4e dx2 = %6.4e\n",
                           step, step_size, dlambda, dx_nrm, arc_ctx->dx2);
        CHKERR VecCopy(D, arc_ctx->x0);
        CHKERR VecAXPY(D, 1., arc_ctx->dx);
        CHKERR my_arc_method_ptr->set_dlambda_to_x(D, dlambda);
      }

      CHKERR SNESSolve(snes, PETSC_NULL, D);

      // Distribute displacements on all processors
      CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
          "ELASTIC_MECHANICS", COL, D, INSERT_VALUES, SCATTER_REVERSE);
      CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "INTERFACE",
                                          cohesive_elements.getFeHistory(), 0,
                                          m_field.get_comm_size());
      // Remove nodes of damaged prisms
      CHKERR my_arc_method_ptr->remove_damaged_prisms_nodes();

      int its;
      CHKERR SNESGetIterationNumber(snes, &its);
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "number of Newton iterations = %D\n",
                         its);

      SNESConvergedReason reason;
      CHKERR SNESGetConvergedReason(snes, &reason);

      if (reason < 0) {
        CHKERR arc_ctx->setAlphaBeta(1, 0);
        reduction = 0.1;
        converged_state = false;
        continue;
      } else {
        if (step > 1 && converged_state) {
          reduction = pow((double)its_d / (double)(its + 1), gamma);
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "reduction step_size = %6.4e\n",
                             reduction);
        }

        // Save data on mesh
        CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
            "ELASTIC_MECHANICS", COL, D, INSERT_VALUES, SCATTER_REVERSE);
        CHKERR m_field.getInterface<VecManager>()->setOtherGlobalGhostVector(
            "ELASTIC_MECHANICS", "DISPLACEMENT", "X0_DISPLACEMENT", COL,
            arc_ctx->x0, INSERT_VALUES, SCATTER_REVERSE);
        converged_state = true;
      }
      //
      if (reason > 0) {
        FILE *datafile;
        PetscFOpen(PETSC_COMM_SELF, DATAFILENAME, "a+", &datafile);
        PetscFPrintf(PETSC_COMM_WORLD, datafile, "%d %d ", reason, its);
        fclose(datafile);
        CHKERR my_arc_method_ptr->postProcessLoadPath();
      }

      if (step % 1 == 0) {

        CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "ELASTIC",
                                            post_proc);
        std::ostringstream ss;
        ss << "out_values_" << step << ".h5m";
        CHKERR post_proc.writeFile(ss.str().c_str());
      }
    }

    // Save data on mesh
    CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
        "ELASTIC_MECHANICS", COL, D, INSERT_VALUES, SCATTER_REVERSE);

    // detroy matrices
    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&D);
    CHKERR VecDestroy(&F_body_force);
    CHKERR MatDestroy(&Aij);
    CHKERR SNESDestroy(&snes);
    CHKERR MatDestroy(&ShellAij);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
