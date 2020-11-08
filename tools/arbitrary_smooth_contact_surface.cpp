/** \file simple_contact.cpp
 * \example simple_contact.cpp
 *
 * Implementation of mortar contact between surfaces with matching meshes
 *
 **/

/* MoFEM is free software: you can redistribute it and/or modify it under
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
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>.
 */

#include <Mortar.hpp>
#include <Hooke.hpp>
#include <NeoHookean.hpp>

static char help[] = "\n";

double MortarContactProblem::LoadScale::lAmbda = 1;
int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_divergence_tolerance 0 \n"
                                 "-snes_max_it 50 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-ksp_monitor \n"
                                 "-snes_converged_reason \n"
                                 "-my_order 1 \n"

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  enum arbitrary_smoothing_tests {
    EIGHT_CUBE = 1,
    LAST_TEST
  };

  // Initialize MoFEM
  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);

  // Create mesh database
  moab::Core mb_instance;              // create database
  moab::Interface &moab = mb_instance; // create interface to database

  try {
    PetscBool flg_file;

    char mesh_file_name[255];
    PetscInt order = 1;
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool is_newton_cotes = PETSC_FALSE;
    PetscInt test_num = 0;
    PetscBool print_contact_state = PETSC_FALSE;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order",
                           "approximation order of spatial positions", "", 1,
                           &order, PETSC_NULL);
    
    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsInt("-my_test_num", "test number", "", 0, &test_num,
                           PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    // Check if mesh file was provided
    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    if (is_partitioned == PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, MOFEM_NOT_IMPLEMENTED,
              "Partitioned mesh is not supported");
    }

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    MeshsetsManager *mmanager_ptr;
    CHKERR m_field.getInterface(mmanager_ptr);
    CHKERR mmanager_ptr->printDisplacementSet();
    CHKERR mmanager_ptr->printForceSet();
    // print block sets with materials
    CHKERR mmanager_ptr->printMaterialsSet();

    Range contact_prisms, master_tris, slave_tris;
    std::vector<BitRefLevel> bit_levels;

    bit_levels.push_back(BitRefLevel().set(0));
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_levels.back());

    Range meshset_level0;
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
        bit_levels.back(), BitRefLevel().set(), meshset_level0);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD, "meshset_level0 %d\n",
                            meshset_level0.size());
    PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT);

    int help_maybe_unwanted = 1;
    CHKERR MortarContactStructures::masterSlaveTrianglesCreation(
        m_field, master_tris, slave_tris, help_maybe_unwanted);

    ContactSearchKdTree contact_search_kd_tree(m_field);

    boost::shared_ptr<ContactSearchKdTree::ContactCommonData_multiIndex>
        contact_commondata_multi_index;

    contact_commondata_multi_index =
        boost::shared_ptr<ContactSearchKdTree::ContactCommonData_multiIndex>(
            new ContactSearchKdTree::ContactCommonData_multiIndex());

    CHKERR contact_search_kd_tree.buildTree(master_tris);

    CHKERR contact_search_kd_tree.contactSearchAlgorithm(
        master_tris, slave_tris, contact_commondata_multi_index,
        contact_prisms);

    EntityHandle meshset_slave_master_prisms;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_slave_master_prisms);

    CHKERR
    moab.add_entities(meshset_slave_master_prisms, contact_prisms);

    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        meshset_slave_master_prisms, 3, bit_levels.back());

    CHKERR moab.write_mesh("slave_master_prisms.vtk",
                           &meshset_slave_master_prisms, 1);

    Range tris_from_prism;
    // find actual masters and slave used
    CHKERR m_field.get_moab().get_adjacencies(
        contact_prisms, 2, true, tris_from_prism, moab::Interface::UNION);

    tris_from_prism = tris_from_prism.subset_by_type(MBTRI);
    slave_tris = intersect(tris_from_prism, slave_tris);

    EntityHandle meshset_surf_slave;
    CHKERR m_field.get_moab().create_meshset(MESHSET_SET, meshset_surf_slave);
    CHKERR m_field.get_moab().add_entities(meshset_surf_slave, slave_tris);

    CHKERR m_field.get_moab().write_mesh("surf_slave.vtk", &meshset_surf_slave,
                                         1);
    EntityHandle meshset_tri_slave;
    CHKERR m_field.get_moab().create_meshset(MESHSET_SET, meshset_tri_slave);

    CHKERR m_field.add_field("CONTACT_MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);

    // Declare problem add entities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET,
                                             "CONTACT_MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "CONTACT_MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "CONTACT_MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "CONTACT_MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBVERTEX, "CONTACT_MESH_NODE_POSITIONS", 1);

    if (!slave_tris.empty()) {
      CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI,
                                               "CONTACT_MESH_NODE_POSITIONS");
      CHKERR m_field.set_field_order(0, MBTRI, "CONTACT_MESH_NODE_POSITIONS",
                                     order);
      CHKERR m_field.set_field_order(0, MBEDGE, "CONTACT_MESH_NODE_POSITIONS",
                                     order);
      CHKERR m_field.set_field_order(0, MBVERTEX, "CONTACT_MESH_NODE_POSITIONS",
                                     1);
    }
    if (!slave_tris.empty()) {
      CHKERR m_field.add_ents_to_field_by_type(master_tris, MBTRI,
                                               "CONTACT_MESH_NODE_POSITIONS");
      CHKERR m_field.set_field_order(0, MBTRI, "CONTACT_MESH_NODE_POSITIONS",
                                     order);
      CHKERR m_field.set_field_order(0, MBEDGE, "CONTACT_MESH_NODE_POSITIONS",
                                     order);
      CHKERR m_field.set_field_order(0, MBVERTEX, "CONTACT_MESH_NODE_POSITIONS",
                                     1);
    }
    // build field
    CHKERR m_field.build_fields();

    // Projection on "X" field
    {
      Projection10NodeCoordsOnField ent_method(m_field, "CONTACT_MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("CONTACT_MESH_NODE_POSITIONS", ent_method);
    }

    auto make_contact_element = [&]() {
      return boost::make_shared<MortarContactProblem::MortarContactElement>(
          m_field, contact_commondata_multi_index, "SPATIAL_POSITION",
          "MESH_NODE_POSITIONS");
    };

    auto make_contact_common_data = [&]() {
      return boost::make_shared<MortarContactProblem::CommonDataMortarContact>(
          m_field);
    };

    auto get_contact_rhs = [&](auto contact_problem, auto make_element,
                               bool is_alm = false) {
      auto fe_rhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      if (print_contact_state) {
        fe_rhs_simple_contact->contactStateVec =
            common_data_simple_contact->gaussPtsStateVec;
      }
      contact_problem->setContactOperatorsRhs(
          fe_rhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", is_alm);
      return fe_rhs_simple_contact;
    };


    auto get_contact_help_lhs = [&](auto contact_problem, auto make_element,
                                    bool is_alm = false) {
      auto fe_lhs_simple_contact = make_element();
      auto common_data_simple_contact = make_contact_common_data();
      contact_problem->setContactOperatorsLhs(
          fe_lhs_simple_contact, common_data_simple_contact, "SPATIAL_POSITION",
          "LAGMULT", is_alm);
      return fe_lhs_simple_contact;
    };

    auto cn_value_ptr = boost::make_shared<double>(cn_value);
    auto contact_problem = boost::make_shared<MortarContactProblem>(
        m_field, contact_commondata_multi_index, cn_value_ptr, is_newton_cotes);

    // add fields to the global matrix by adding the element

    contact_problem->addMortarContactElement("CONTACT_ELEM", "SPATIAL_POSITION",
                                             "LAGMULT", contact_prisms);

    contact_problem->addPostProcContactElement(
        "CONTACT_POST_PROC", "SPATIAL_POSITION", "LAGMULT",
        "MESH_NODE_POSITIONS", slave_tris);

    CHKERR MetaNeumannForces::addNeumannBCElements(m_field, "SPATIAL_POSITION");

    // Add spring boundary condition applied on surfaces.
    CHKERR MetaSpringBC::addSpringElements(m_field, "SPATIAL_POSITION",
                                           "MESH_NODE_POSITIONS");

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_levels.back());

    // define problems
    CHKERR m_field.add_problem("CONTACT_PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("CONTACT_PROB",
                                                    bit_levels.back());

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    SmartPetscObj<DM> dm;
    dm = createSmartDM(m_field.get_comm(), dm_name);

    // create dm instance
    CHKERR DMSetType(dm, dm_name);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "CONTACT_PROB", bit_levels.back());
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "SURFACE_SMOOTHER");
    CHKERR DMMoFEMAddElement(dm, "VOLUME_CALCULATOR");

    CHKERR DMSetUp(dm);

    // Vector of DOFs and the RHS
    auto D = smartCreateDMVector(dm);
    auto F = smartVectorDuplicate(D);

    // Stiffness matrix
    auto Aij = smartCreateDMMatrix(dm);

    CHKERR VecZeroEntries(D);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);
    CHKERR MatZeroEntries(Aij);

    // Dirichlet BC
    boost::shared_ptr<FEMethod> dirichlet_bc_ptr =
        boost::shared_ptr<FEMethod>(new DirichletSpatialPositionsBc(
            m_field, "SPATIAL_POSITION", Aij, D, F));

    dirichlet_bc_ptr->snes_ctx = SnesMethod::CTX_SNESNONE;
    dirichlet_bc_ptr->snes_x = D;

    CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL,
                                  dirichlet_bc_ptr.get(), NULL);

    if (convect_pts == PETSC_TRUE) {
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_contact_rhs(contact_problem, make_convective_master_element),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_master_traction_rhs(contact_problem,
                                  make_convective_slave_element),
          PETSC_NULL, PETSC_NULL);
    } else {
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_contact_rhs(contact_problem, make_contact_element, alm_flag),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetFunction(
          dm, "CONTACT_ELEM",
          get_master_traction_rhs(contact_problem, make_contact_element,
                                  alm_flag),
          PETSC_NULL, PETSC_NULL);
    }

    CHKERR DMMoFEMSNESSetFunction(dm, "ELASTIC", &elastic.getLoopFeRhs(),
                                  PETSC_NULL, PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, "SPRING", fe_spring_rhs_ptr, PETSC_NULL,
                                  PETSC_NULL);
    CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL, NULL,
                                  dirichlet_bc_ptr.get());

    boost::shared_ptr<FEMethod> fe_null;
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, dirichlet_bc_ptr,
                                  fe_null);

    if (convect_pts == PETSC_TRUE) {
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_contact_help_lhs(contact_problem, make_convective_master_element),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_help_traction_lhs(contact_problem,
                                       make_convective_slave_element),
          PETSC_NULL, PETSC_NULL);
    } else {
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_contact_lhs(contact_problem, make_contact_element, alm_flag),
          PETSC_NULL, PETSC_NULL);
      CHKERR DMMoFEMSNESSetJacobian(
          dm, "CONTACT_ELEM",
          get_master_traction_lhs(contact_problem, make_contact_element,
                                  alm_flag),
          PETSC_NULL, PETSC_NULL);
    }

    CHKERR DMMoFEMSNESSetJacobian(dm, "SPRING", fe_spring_lhs_ptr, NULL, NULL);
    CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, fe_null,
                                  dirichlet_bc_ptr);

    if (test_num) {
      char testing_options[] = "-ksp_type fgmres "
                               "-pc_type lu "
                               "-pc_factor_mat_solver_type mumps "
                               "-snes_type newtonls "
                               "-snes_linesearch_type basic "
                               "-snes_max_it 20 "
                               "-snes_atol 1e-8 "
                               "-snes_rtol 1e-8 ";
      CHKERR PetscOptionsInsertString(NULL, testing_options);
    }

    auto snes = MoFEM::createSNES(m_field.get_comm());
    CHKERR SNESSetDM(snes, dm);
    SNESConvergedReason snes_reason;
    SnesCtx *snes_ctx;
    // create snes nonlinear solver
    {
      CHKERR SNESSetDM(snes, dm);
      CHKERR DMMoFEMGetSnesCtx(dm, &snes_ctx);
      CHKERR SNESSetFunction(snes, F, SnesRhs, snes_ctx);
      CHKERR SNESSetJacobian(snes, Aij, Aij, SnesMat, snes_ctx);
      CHKERR SNESSetFromOptions(snes);
    }

    PostProcVolumeOnRefinedMesh post_proc(m_field);
    // Add operators to the elements, starting with some generic
    CHKERR post_proc.generateReferenceElementMesh();
    CHKERR post_proc.addFieldValuesPostProc("SPATIAL_POSITION");
    CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
    CHKERR post_proc.addFieldValuesGradientPostProc("SPATIAL_POSITION");

    std::map<int, NonlinearElasticElement::BlockData>::iterator sit =
        elastic.setOfBlocks.begin();
    for (; sit != elastic.setOfBlocks.end(); sit++) {
      post_proc.getOpPtrVector().push_back(new PostProcStress(
          post_proc.postProcMesh, post_proc.mapGaussPts, "SPATIAL_POSITION",
          sit->second, post_proc.commonData));
    }

    for (int ss = 0; ss != nb_sub_steps; ++ss) {

      MortarContactProblem::LoadScale::lAmbda = (ss + 1.0) / nb_sub_steps;
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "lAmbda %6.4e\n",
                         MortarContactProblem::LoadScale::lAmbda);

      CHKERR SNESSolve(snes, PETSC_NULL, D);

      CHKERR SNESGetConvergedReason(snes, &snes_reason);

      int its;
      CHKERR SNESGetIterationNumber(snes, &its);
      CHKERR PetscPrintf(PETSC_COMM_WORLD,
                         "number of Newton iterations = %D\n\n", its);

      // save on mesh
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

      PetscPrintf(PETSC_COMM_WORLD, "Loop post proc\n");
      CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);

      elastic.getLoopFeEnergy().snes_ctx = SnesMethod::CTX_SNESNONE;
      elastic.getLoopFeEnergy().eNergy = 0;
      PetscPrintf(PETSC_COMM_WORLD, "Loop energy\n");
      CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC",
                                      &elastic.getLoopFeEnergy());
      // Print elastic energy
      PetscPrintf(PETSC_COMM_WORLD, "Elastic energy %6.4e\n",
                  elastic.getLoopFeEnergy().eNergy);

      string out_file_name;
      std::ostringstream stm;
      stm << "out_" << ss << ".h5m";
      out_file_name = stm.str();
      CHKERR
      PetscPrintf(PETSC_COMM_WORLD, "out file %s\n", out_file_name.c_str());

      CHKERR post_proc.postProcMesh.write_file(out_file_name.c_str(), "MOAB",
                                               "PARALLEL=WRITE_PART");

      // moab_instance
      moab::Core mb_post;                   // create database
      moab::Interface &moab_proc = mb_post; // create interface to database

      auto common_data_simple_contact = make_contact_common_data();

      boost::shared_ptr<MortarContactProblem::MortarContactElement>
          fe_post_proc_simple_contact;
      if (convect_pts == PETSC_TRUE) {
        fe_post_proc_simple_contact = make_convective_master_element();
      } else {
        fe_post_proc_simple_contact = make_contact_element();
      }

      contact_problem->setContactOperatorsForPostProc(
          fe_post_proc_simple_contact, common_data_simple_contact, m_field,
          "SPATIAL_POSITION", "LAGMULT", mb_post, alm_flag);

      mb_post.delete_mesh();
      CHKERR VecZeroEntries(common_data_simple_contact->gaussPtsStateVec);
      CHKERR VecZeroEntries(common_data_simple_contact->contactAreaVec);

      std::ofstream ofs((std ::string("test_simple_contact") + ".txt").c_str());

      boost::shared_ptr<SimpleContactProblem::CommonDataSimpleContact>
          common_data_simple_mortar = boost::static_pointer_cast<
              SimpleContactProblem::CommonDataSimpleContact>(
              common_data_simple_contact);

      fe_post_proc_simple_contact->getOpPtrVector().push_back(
          new MortarContactProblem::OpMakeTestTextFile(
              m_field, "SPATIAL_POSITION", common_data_simple_mortar, ofs));

      CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_ELEM",
                                      fe_post_proc_simple_contact);

      if (m_field.get_comm_rank() == 0) {
        PetscPrintf(PETSC_COMM_SELF, "Active gauss pts: %d out of %d\n",
                    (int)nb_gauss_pts[0], (int)nb_gauss_pts[1]);

        PetscPrintf(PETSC_COMM_SELF,
                    "Active contact area: %9.9f out of %9.9f\n",
                    contact_area[0], contact_area[1]);
      }

      ofs << "Elastic energy: " << elastic.getLoopFeEnergy().eNergy << endl;
      ofs.flush();
      ofs.close();

      if (test_num) {
        double expected_energy, expected_contact_area;
        int expected_nb_gauss_pts;
        constexpr double eps = 1e-8;
        switch (test_num) {
        case EIGHT_CUBE:
          expected_energy = 3.0e-04;
          expected_contact_area = 2.999995061;
          expected_nb_gauss_pts = 5718;
          break;
        case T_INTERFACE:
          expected_energy = 3.0e-04;
          expected_contact_area = 1.749997733;
          expected_nb_gauss_pts = 2046;
          break;
        case PUNCH_TOP_AND_MID:
          expected_energy = 7.8125e-05;
          expected_contact_area = 0.062499557;
          expected_nb_gauss_pts = 3408;
          break;
        case PUNCH_TOP_ONLY:
          expected_energy = 2.4110e-05;
          expected_contact_area = 0.062499557;
          expected_nb_gauss_pts = 3408;
          break;
        case PUNCH_TOP_ONLY_ALM:
          expected_energy = 2.4104e-05;
          expected_contact_area = 0.062499557;
          expected_nb_gauss_pts = 3408;
          break;
        case SMILING_FACE:
          expected_energy = 7.5944e-04;
          expected_contact_area = 2.727269391;
          expected_nb_gauss_pts = 1020;
          break;
        case SMILING_FACE_CONVECT:
          expected_energy = 7.2982e-04;
          expected_contact_area = 2.230868859;
          expected_nb_gauss_pts = 794;
          break;
        case WAVE_2D:
          expected_energy = 0.008537863;
          expected_contact_area = 0.125;
          expected_nb_gauss_pts = 384;
          break;
        case WAVE_2D_ALM:
          expected_energy = 0.008538894;
          expected_contact_area = 0.125;
          expected_nb_gauss_pts = 384;
          break;
        case NIGHT_HERTZ_2D:
          CHKERR get_night_hertz_2D_results(
              expected_energy, expected_contact_area, expected_nb_gauss_pts);
          break;
        case NIGHT_WAVE_2D:
          CHKERR get_night_wave_2D_results(
              expected_energy, expected_contact_area, expected_nb_gauss_pts);
          break;
        case NIGHT_HERTZ_3D:
          CHKERR get_night_hertz_3D_results(
              expected_energy, expected_contact_area, expected_nb_gauss_pts);
          break;
        default:
          SETERRQ1(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                   "Unknown test number %d", test_num);
        }
        if (std::abs(elastic.getLoopFeEnergy().eNergy - expected_energy) >
            eps) {
          SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                   "Wrong energy %6.4e != %6.4e", expected_energy,
                   elastic.getLoopFeEnergy().eNergy);
        }
        if (m_field.get_comm_rank() == 0) {
          if ((int)nb_gauss_pts[0] != expected_nb_gauss_pts) {
            SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                     "Wrong number of active gauss pts %d != %d",
                     expected_nb_gauss_pts, (int)nb_gauss_pts[0]);
          }
          if (std::abs(contact_area[0] - expected_contact_area) > eps) {
            SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                     "Wrong active contact area %6.4e != %6.4e",
                     expected_contact_area, contact_area[0]);
          }
        }
      }

      {
        std::ostringstream ostrm;

        ostrm << "out_contact_integ_pts_" << ss << ".h5m";

        out_file_name = ostrm.str();
        CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
                           out_file_name.c_str());
        CHKERR mb_post.write_file(out_file_name.c_str(), "MOAB",
                                  "PARALLEL=WRITE_PART");
      }
      // lagrange
      boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_contact_ptr(
          new PostProcFaceOnRefinedMesh(m_field));

      CHKERR post_proc_contact_ptr->generateReferenceElementMesh();

      CHKERR post_proc_contact_ptr->addFieldValuesPostProc("LAGMULT");
      CHKERR post_proc_contact_ptr->addFieldValuesPostProc("SPATIAL_POSITION");
      CHKERR post_proc_contact_ptr->addFieldValuesPostProc(
          "MESH_NODE_POSITIONS");

      CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_POST_PROC",
                                      post_proc_contact_ptr);
      std::ostringstream stm_2;
      stm_2 << "out_contact_" << ss << ".h5m";
      out_file_name = stm_2.str();
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
                         out_file_name.c_str());
      CHKERR post_proc_contact_ptr->postProcMesh.write_file(
          out_file_name.c_str(), "MOAB", "PARALLEL=WRITE_PART");
    }
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}