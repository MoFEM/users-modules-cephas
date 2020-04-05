/** \file reynolds.cpp
 * \example reynolds.cpp
 *
 * Simulation of a thin fluid flow governed by the Reynolds equation
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

#include <BasicFiniteElements.hpp>

using namespace std;
using namespace MoFEM;

static char help[] = "\n";
int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_package mumps \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_max_it 10 \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-my_order 1 \n"
                                 "-my_is_test 0 \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

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
    PetscBool is_test = PETSC_FALSE;

    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");

    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order", "default approximation order", "", 1,
                           &order, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_test", "set if run as test", "",
                            PETSC_FALSE, &is_test, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);

    // Check if mesh file was provided
    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    if (is_partitioned == PETSC_TRUE) {
      // Read mesh to MOAB
      const char *option;
      option = "PARALLEL=BCAST_DELETE;"
               "PARALLEL_RESOLVE_SHARED_ENTS;"
               "PARTITION=PARALLEL_PARTITION;";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    } else {
      const char *option;
      option = "";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    }

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



    Range slave_tris, tris;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, cit)) {
      if (cit->getName().compare(0, 8, "REYNOLDS") == 0) {
        tris.clear();
        CHKERR mmanager_ptr->getEntitiesByDimension(cit->getMeshsetId(),
                                                     BLOCKSET, 2, tris, true);
        slave_tris.merge(tris);
      }
    }

    BitRefLevel bit0 = BitRefLevel().set(0);

    Range flow_prisms;
    EntityHandle meshset_prisms;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_prisms);
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByTypeAndRefLevel(
        bit0, BitRefLevel().set(), MBPRISM, meshset_prisms);
    CHKERR moab.get_entities_by_handle(meshset_prisms, flow_prisms);
    CHKERR moab.delete_entities(&meshset_prisms, 1);

    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(0, 3,
                                                                      bit0);

    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_field("PRESSURE", H1, AINSWORTH_LEGENDRE_BASE, 1,
                             MB_TAG_SPARSE, MF_ZERO);

    CHKERR m_field.add_ents_to_field_by_type(0, MBPRISM, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBPRISM, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBQUAD, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 1);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    CHKERR m_field.add_ents_to_field_by_type(slave_tris, MBTRI, "PRESSURE");
    CHKERR m_field.set_field_order(0, MBTRI, "PRESSURE", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "PRESSURE", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "PRESSURE", 1);

    // build field
    CHKERR m_field.build_fields();

    // Projection on "X" field
    {
      Projection10NodeCoordsOnField ent_method(m_field, "MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method);
    }

    auto make_element = [&]() {
      return boost::make_shared<ThinFluidFlowProblem::ThinFluidFlowElement>(
          m_field);
    };

    auto make_common_data = [&]() {
      return boost::make_shared<ThinFluidFlowProblem::CommonData>(m_field);
    };

    auto get_rhs = [&](auto thin_fluid_flow_problem, auto make_element) {
      auto fe_rhs = make_element();
      auto common_data = make_common_data();
      thin_fluid_flow_problem->setThinFluidFlowOperatorsRhs(
          fe_rhs, common_data, "MESH_NODE_POSITIONS", "PRESSURE");
      return fe_rhs;
    };

    // auto get_master_contact_lhs = [&](auto thin_fluid_flow_problem,
    //                                   auto make_element) {
    //   auto fe_lhs = make_element();
    //   auto common_data = make_common_data();
    //   thin_fluid_flow_problem->setThinFluidFlowOperatorsLhs(
    //       fe_lhs, common_data, "MESH_NODE_POSITIONS", "PRESSURE");
    //   return fe_lhs;
    // };

    auto thin_fluid_flow_problem =
        boost::make_shared<ThinFluidFlowProblem>(m_field);

    // add fields to the global matrix by adding the element
    thin_fluid_flow_problem->addThinFluidFlowElement(
        "THIN_FLUID_FLOW_ELEM", "MESH_NODE_POSITIONS", "PRESSURE", flow_prisms);

    // build finite elemnts
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(bit0);

    // define problems
    CHKERR m_field.add_problem("THIN_FLUID_FLOW_PROB");

    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("THIN_FLUID_FLOW_PROB",
                                                    bit0);

    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    SmartPetscObj<DM> dm;
    dm = createSmartDM(m_field.get_comm(), dm_name);

    // create dm instance
    CHKERR DMSetType(dm, dm_name);

    // set dm datastruture which created mofem datastructures
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "THIN_FLUID_FLOW_PROB", bit0);
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // add elements to dm
    CHKERR DMMoFEMAddElement(dm, "THIN_FLUID_FLOW_ELEM");

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
    // boost::shared_ptr<FEMethod> dirichlet_bc_ptr =
    //     boost::shared_ptr<FEMethod>(new DirichletSpatialPositionsBc(
    //         m_field, "SPATIAL_POSITION", Aij, D, F));

    // dirichlet_bc_ptr->snes_ctx = SnesMethod::CTX_SNESNONE;
    // dirichlet_bc_ptr->snes_x = D;

    // CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
    // CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    // CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
    // CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL,
    //                               dirichlet_bc_ptr.get(), NULL);

    CHKERR DMMoFEMSNESSetFunction(
        dm, "THIN_FLUID_FLOW_ELEM",
        get_rhs(thin_fluid_flow_problem, make_element), PETSC_NULL, PETSC_NULL);

    // CHKERR DMMoFEMSNESSetFunction(dm, DM_NO_ELEMENT, NULL, NULL,
    //                               dirichlet_bc_ptr.get());

    boost::shared_ptr<FEMethod> fe_null;
    // CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, dirichlet_bc_ptr,
    //                               fe_null);


      // CHKERR DMMoFEMSNESSetJacobian(
      //     dm, "THIN_FLUID_FLOW_ELEM",
      //     get_master_contact_lhs(thin_fluid_flow_problem, make_element), NULL,
      //     NULL);
  
    // CHKERR DMMoFEMSNESSetJacobian(dm, DM_NO_ELEMENT, fe_null, fe_null,
    //                               dirichlet_bc_ptr);

    if (is_test == PETSC_TRUE) {
      char testing_options[] = "-ksp_type fgmres "
                               "-pc_type lu "
                               "-pc_factor_mat_solver_package mumps "
                               "-snes_type newtonls "
                               "-snes_linesearch_type basic "
                               "-snes_max_it 10 "
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

    // PostProcVolumeOnRefinedMesh post_proc(m_field);
    // // Add operators to the elements, starting with some generic
    // CHKERR post_proc.generateReferenceElementMesh();
    // CHKERR post_proc.addFieldValuesPostProc("SPATIAL_POSITION");
    // CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
    // CHKERR post_proc.addFieldValuesGradientPostProc("SPATIAL_POSITION");


    CHKERR SNESSolve(snes, PETSC_NULL, D);

    CHKERR SNESGetConvergedReason(snes, &snes_reason);

    int its;
    CHKERR SNESGetIterationNumber(snes, &its);
    CHKERR PetscPrintf(PETSC_COMM_WORLD, "number of Newton iterations = %D\n\n",
                       its);

    // save on mesh
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToGlobalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

    // PetscPrintf(PETSC_COMM_WORLD, "Loop post proc\n");
    // CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);

    // elastic.getLoopFeEnergy().snes_ctx = SnesMethod::CTX_SNESNONE;
    // elastic.getLoopFeEnergy().eNergy = 0;
    // PetscPrintf(PETSC_COMM_WORLD, "Loop energy\n");
    // CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &elastic.getLoopFeEnergy());
    // // Print elastic energy
    // PetscPrintf(PETSC_COMM_WORLD, "Elastic energy %6.4e\n",
    //             elastic.getLoopFeEnergy().eNergy);

    // string out_file_name;
    // std::ostringstream stm;
    // stm << "out"
    //     << ".h5m";
    // out_file_name = stm.str();
    // CHKERR
    // PetscPrintf(PETSC_COMM_WORLD, "out file %s\n", out_file_name.c_str());

    // CHKERR post_proc.postProcMesh.write_file(out_file_name.c_str(), "MOAB",
    //                                          "PARALLEL=WRITE_PART");

    // moab_instance
    // moab::Core mb_post;                   // create database
    // moab::Interface &moab_proc = mb_post; // create interface to database

    // auto common_data = make_common_data();

    // boost::shared_ptr<ThinFluidFlowProblem::ThinFluidFlowElement>
    //     fe_post_proc_thin_fluid_flow;
    // if (convect_pts == PETSC_TRUE) {
    //   fe_post_proc_thin_fluid_flow = make_convective_master_element();
    // } else {
    //   fe_post_proc_thin_fluid_flow = make_element();
    // }

    // thin_fluid_flow_problem->setContactOperatorsForPostProc(
    //     fe_post_proc_thin_fluid_flow, common_data, m_field, "SPATIAL_POSITION",
    //     "PRESSURE", mb_post);

    // mb_post.delete_mesh();

    // if (is_test == PETSC_TRUE) {
    //   std::ofstream ofs(
    //       (std ::string("test_thin_fluid_flow") + ".txt").c_str());

    //   fe_post_proc_thin_fluid_flow->getOpPtrVector().push_back(
    //       new ThinFluidFlowProblem::OpMakeTestTextFile(
    //           m_field, "SPATIAL_POSITION", common_data, ofs));

    //   CHKERR DMoFEMLoopFiniteElements(dm, "THIN_FLUID_FLOW_ELEM",
    //                                   fe_post_proc_thin_fluid_flow);

    //   ofs << "Elastic energy: " << elastic.getLoopFeEnergy().eNergy << endl;
    //   ofs.flush();
    //   ofs.close();
    // } else {
    //   CHKERR DMoFEMLoopFiniteElements(dm, "THIN_FLUID_FLOW_ELEM",
    //                                   fe_post_proc_thin_fluid_flow);
    // }

    // std::ostringstream ostrm;

    // ostrm << "out_contact"
    //       << ".h5m";

    // out_file_name = ostrm.str();
    // CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
    //                    out_file_name.c_str());
    // CHKERR mb_post.write_file(out_file_name.c_str(), "MOAB",
    //                           "PARALLEL=WRITE_PART");

    // if (true) {

    //   boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc_contact_ptr(
    //       new PostProcFaceOnRefinedMesh(m_field));

    //   CHKERR post_proc_contact_ptr->generateReferenceElementMesh();

    //   auto common_post_proc_data_thin_fluid_flow = make_common_data();

    //   CHKERR thin_fluid_flow_problem->setPostProcContactOperators(
    //       post_proc_contact_ptr, "SPATIAL_POSITION", "PRESSURE",
    //       common_post_proc_data_thin_fluid_flow);

    //   CHKERR DMoFEMLoopFiniteElements(dm, "CONTACT_VTK", post_proc_contact_ptr);
    //   std::ostringstream stm;
    //   std::string file_name_for_lagrange = "out_lagrange_for_vtk_";
    //   stm << "file_name_for_lagrange"
    //       << ".h5m";

    //   out_file_name = stm.str();
    //   CHKERR PetscPrintf(PETSC_COMM_WORLD, "out file %s\n",
    //                      out_file_name.c_str());
    //   CHKERR post_proc_contact_ptr->postProcMesh.write_file(
    //       out_file_name.c_str(), "MOAB", "PARALLEL=WRITE_PART");
    // }

    // EntityHandle out_meshset_slave_tris;
    // EntityHandle out_meshset_master_tris;

    // CHKERR moab.create_meshset(MESHSET_SET, out_meshset_slave_tris);
    // CHKERR moab.create_meshset(MESHSET_SET, out_meshset_master_tris);

    // CHKERR moab.add_entities(out_meshset_slave_tris, slave_tris);
    // CHKERR moab.add_entities(out_meshset_master_tris, master_tris);

    // CHKERR moab.write_file("out_slave_tris.vtk", "VTK", "",
    //                        &out_meshset_slave_tris, 1);
    // CHKERR moab.write_file("out_master_tris.vtk", "VTK", "",
    //                        &out_meshset_master_tris, 1);

    // CHKERR moab.delete_entities(&out_meshset_slave_tris, 1);
    // CHKERR moab.delete_entities(&out_meshset_master_tris, 1);
  }
  CATCH_ERRORS;

  // finish work cleaning memory, getting statistics, etc
  MoFEM::Core::Finalize();

  return 0;
}