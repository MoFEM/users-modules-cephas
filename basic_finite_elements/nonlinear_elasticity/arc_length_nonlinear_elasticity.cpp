/** \file arc_length_nonlinear_elasticity.cpp
 * \ingroup nonlinear_elastic_elem
 * \brief nonlinear elasticity (arc-length control)
 *
 * Solves nonlinear elastic problem. Using arc length control.
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

static char help[] = "\
 -my_file mesh file name\n\
 -my_sr reduction of step size\n\
 -my_ms maximal number of steps\n\n";

#include <BasicFiniteElements.hpp>
using namespace MoFEM;

#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;
#include <ElasticMaterials.hpp>
#include <NeoHookean.hpp>

#include <SurfacePressureComplexForLazy.hpp>

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type fgmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_package mumps \n"
                                 "-ksp_atol 1e-10 \n"
                                 "-ksp_rtol 1e-10 \n"
                                 "-snes_monitor \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_max_it 100 \n"
                                 "-snes_atol 1e-7 \n"
                                 "-snes_rtol 1e-7 \n"
                                 "-ts_monitor \n"
                                 "-ts_type alpha \n";

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
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    CHKERR PetscOptionsGetString(PETSC_NULL, PETSC_NULL, "-my_file",
                                 mesh_file_name, 255, &flg);
    if (flg != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    PetscInt order;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &order,
                              &flg);
    if (flg != PETSC_TRUE) {
      order = 2;
    }

    // use this if your mesh is partitioned and you run code on parts,
    // you can solve very big problems
    PetscBool is_partitioned = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-my_is_partitioned",
                               &is_partitioned, &flg);

    if (is_partitioned == PETSC_TRUE) {
      // Read mesh to MOAB
      const char *option;
      option = "PARALLEL=BCAST_DELETE;PARALLEL_RESOLVE_SHARED_ENTS;PARTITION="
               "PARALLEL_PARTITION;";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    } else {
      const char *option;
      option = "";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    }

    // data stored on mesh for restart
    Tag th_step_size, th_step;
    double def_step_size = 1;
    CHKERR moab.tag_get_handle("_STEPSIZE", 1, MB_TYPE_DOUBLE, th_step_size,
                               MB_TAG_CREAT | MB_TAG_MESH, &def_step_size);
    if (rval == MB_ALREADY_ALLOCATED)
      CHKERR MB_SUCCESS;

    int def_step = 1;
    CHKERR moab.tag_get_handle("_STEP", 1, MB_TYPE_INTEGER, th_step,
                               MB_TAG_CREAT | MB_TAG_MESH, &def_step);
    if (rval == MB_ALREADY_ALLOCATED)
      CHKERR MB_SUCCESS;

    const void *tag_data_step_size[1];
    EntityHandle root = moab.get_root_set();
    CHKERR moab.tag_get_by_ptr(th_step_size, &root, 1, tag_data_step_size);
    double &step_size = *(double *)tag_data_step_size[0];
    const void *tag_data_step[1];
    CHKERR moab.tag_get_by_ptr(th_step, &root, 1, tag_data_step);
    int &step = *(int *)tag_data_step[0];
    // end of data stored for restart
    CHKERR PetscPrintf(PETSC_COMM_WORLD,
                       "Start step %D and step_size = %6.4e\n", step,
                       step_size);

    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // ref meshset ref level 0
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, BitRefLevel().set(0));
    std::vector<BitRefLevel> bit_levels;
    bit_levels.push_back(BitRefLevel().set(0));
    BitRefLevel problem_bit_level;

    if (step == 1) {

      problem_bit_level = bit_levels.back();

      // Fields
      CHKERR m_field.add_field("SPATIAL_POSITION", H1, AINSWORTH_LEGENDRE_BASE,
                               3);
      CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1,
                               AINSWORTH_LEGENDRE_BASE, 3);

      CHKERR m_field.add_field("LAMBDA", NOFIELD, NOBASE, 1);

      // Field for ArcLength
      CHKERR m_field.add_field("X0_SPATIAL_POSITION", H1,
                               AINSWORTH_LEGENDRE_BASE, 3);

      // FE
      CHKERR m_field.add_finite_element("ELASTIC");
      CHKERR m_field.add_finite_element("ARC_LENGTH");

      // Add spring boundary condition applied on surfaces.
      // This is only declaration not implementation.
      CHKERR MetaSpringBC::addSpringElements(m_field, "SPATIAL_POSITION",
                                             "MESH_NODE_POSITIONS");

      // Define rows/cols and element data
      CHKERR m_field.modify_finite_element_add_field_row("ELASTIC",
                                                         "SPATIAL_POSITION");
      CHKERR m_field.modify_finite_element_add_field_row("ELASTIC", "LAMBDA");
      CHKERR m_field.modify_finite_element_add_field_col("ELASTIC",
                                                         "SPATIAL_POSITION");
      CHKERR m_field.modify_finite_element_add_field_col(
          "ELASTIC", "LAMBDA"); // this is for parmetis
      CHKERR m_field.modify_finite_element_add_field_data("ELASTIC",
                                                          "SPATIAL_POSITION");
      CHKERR m_field.modify_finite_element_add_field_data(
          "ELASTIC", "MESH_NODE_POSITIONS");
      CHKERR m_field.modify_finite_element_add_field_data("ELASTIC", "LAMBDA");

      // Define rows/cols and element data
      CHKERR m_field.modify_finite_element_add_field_row("ARC_LENGTH",
                                                         "LAMBDA");
      CHKERR m_field.modify_finite_element_add_field_col("ARC_LENGTH",
                                                         "LAMBDA");
      // elem data
      CHKERR m_field.modify_finite_element_add_field_data("ARC_LENGTH",
                                                          "LAMBDA");

      // define problems
      CHKERR m_field.add_problem("ELASTIC_MECHANICS");

      // set finite elements for problems
      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "ELASTIC");
      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "ARC_LENGTH");

      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "SPRING");

      // set refinement level for problem
      CHKERR m_field.modify_problem_ref_level_add_bit("ELASTIC_MECHANICS",
                                                      problem_bit_level);

      // add entities (by tets) to the field
      CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");
      CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "X0_SPATIAL_POSITION");
      CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");

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
        // this entity will carry data for this finite element
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
      CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_POSITION", order);
      CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_POSITION", order);
      CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_POSITION", order);
      CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_POSITION", 1);

      CHKERR m_field.set_field_order(0, MBTET, "X0_SPATIAL_POSITION", order);
      CHKERR m_field.set_field_order(0, MBTRI, "X0_SPATIAL_POSITION", order);
      CHKERR m_field.set_field_order(0, MBEDGE, "X0_SPATIAL_POSITION", order);
      CHKERR m_field.set_field_order(0, MBVERTEX, "X0_SPATIAL_POSITION", 1);

      CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

      // add Neumman finite elements to add static boundary conditions
      CHKERR m_field.add_finite_element("NEUMANN_FE");
      CHKERR m_field.modify_finite_element_add_field_row("NEUMANN_FE",
                                                         "SPATIAL_POSITION");
      CHKERR m_field.modify_finite_element_add_field_col("NEUMANN_FE",
                                                         "SPATIAL_POSITION");
      CHKERR m_field.modify_finite_element_add_field_data("NEUMANN_FE",
                                                          "SPATIAL_POSITION");
      CHKERR m_field.modify_finite_element_add_field_data(
          "NEUMANN_FE", "MESH_NODE_POSITIONS");
      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "NEUMANN_FE");
      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,
                                                      NODESET | FORCESET, it)) {
        Range tris;
        CHKERR moab.get_entities_by_type(it->meshset, MBTRI, tris, true);
        CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                          "NEUMANN_FE");
      }
      for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
               m_field, SIDESET | PRESSURESET, it)) {
        Range tris;
        CHKERR moab.get_entities_by_type(it->meshset, MBTRI, tris, true);
        CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                          "NEUMANN_FE");
      }
      // add nodal force element
      CHKERR MetaNodalForces::addElement(m_field, "SPATIAL_POSITION");
      CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                       "FORCE_FE");
    }

    // Implementation of spring element
    // Create new instances of face elements for springs
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr(
        new FaceElementForcesAndSourcesCore(m_field));

    CHKERR MetaSpringBC::setSpringOperators(
        m_field, fe_spring_lhs_ptr, fe_spring_rhs_ptr, "SPATIAL_POSITION",
        "MESH_NODE_POSITIONS");

    PetscBool linear;
    CHKERR PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-is_linear", &linear,
                               &linear);

    NonlinearElasticElement elastic(m_field, 2);
    ElasticMaterials elastic_materials(m_field);
    CHKERR elastic_materials.setBlocks(elastic.setOfBlocks);
    CHKERR elastic.addElement("ELASTIC", "SPATIAL_POSITION");
    CHKERR elastic.setOperators("SPATIAL_POSITION");

    // post_processing
    PostProcVolumeOnRefinedMesh post_proc(m_field);
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

    // build field
    CHKERR m_field.build_fields();
    if (step == 1) {
      // 10 node tets
      Projection10NodeCoordsOnField ent_method_material(m_field,
                                                        "MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material, 0);
      CHKERR m_field.getInterface<FieldBlas>()->setField(0, MBVERTEX,
                                                         "SPATIAL_POSITION");
      CHKERR m_field.getInterface<FieldBlas>()->setField(0, MBEDGE,
                                                         "SPATIAL_POSITION");
      CHKERR m_field.getInterface<FieldBlas>()->fieldAxpy(
          1., "MESH_NODE_POSITIONS", "SPATIAL_POSITION");
      CHKERR m_field.getInterface<FieldBlas>()->setField(0, MBTRI,
                                                         "SPATIAL_POSITION");
      CHKERR m_field.getInterface<FieldBlas>()->setField(0, MBTET,
                                                         "SPATIAL_POSITION");
    }

    // build finite elements
    CHKERR m_field.build_finite_elements();

    // build adjacencies
    CHKERR m_field.build_adjacencies(problem_bit_level);

    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    // build database
    if (is_partitioned) {
      SETERRQ(PETSC_COMM_SELF, 1,
              "Not implemented, problem with arc-length force multiplayer");
    } else {
      CHKERR prb_mng_ptr->buildProblem("ELASTIC_MECHANICS", true);
      CHKERR prb_mng_ptr->partitionProblem("ELASTIC_MECHANICS");
      CHKERR prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS");
    }
    CHKERR prb_mng_ptr->partitionGhostDofs("ELASTIC_MECHANICS");

    // print bcs
    MeshsetsManager *mmanager_ptr;
    CHKERR m_field.getInterface(mmanager_ptr);
    CHKERR mmanager_ptr->printDisplacementSet();
    CHKERR mmanager_ptr->printForceSet();
    // print block sets with materials
    CHKERR mmanager_ptr->printMaterialsSet();

    // create matrices
    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost(
        "ELASTIC_MECHANICS", COL, &F);
    Vec D;
    CHKERR VecDuplicate(F, &D);
    Mat Aij;
    CHKERR m_field.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("ELASTIC_MECHANICS",
                                                        &Aij);

    boost::shared_ptr<ArcLengthCtx> arc_ctx = boost::shared_ptr<ArcLengthCtx>(
        new ArcLengthCtx(m_field, "ELASTIC_MECHANICS"));

    PetscInt M, N;
    CHKERR MatGetSize(Aij, &M, &N);
    PetscInt m, n;
    CHKERR MatGetLocalSize(Aij, &m, &n);
    boost::scoped_ptr<ArcLengthMatShell> mat_ctx(
        new ArcLengthMatShell(Aij, arc_ctx, "ELASTIC_MECHANICS"));

    Mat ShellAij;
    CHKERR MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, mat_ctx.get(),
                          &ShellAij);
    CHKERR MatShellSetOperation(ShellAij, MATOP_MULT,
                                (void (*)(void))ArcLengthMatMultShellOp);

    ArcLengthSnesCtx snes_ctx(m_field, "ELASTIC_MECHANICS", arc_ctx);
    ///< do not very if element of given name exist when do loop over elements
    snes_ctx.bH = MF_ZERO;

    Range node_set;
    for (_IT_CUBITMESHSETS_BY_NAME_FOR_LOOP_(m_field, "LoadPath", cit)) {
      EntityHandle meshset = cit->getMeshset();
      Range nodes;
      CHKERR moab.get_entities_by_type(meshset, MBVERTEX, nodes, true);
      MOAB_THROW(rval);
      node_set.merge(nodes);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Nb. nodes in load path: %u\n",
                node_set.size());

    SphericalArcLengthControl arc_method(arc_ctx);

    double scaled_reference_load = 1;
    double *scale_lhs = &(arc_ctx->getFieldData());
    double *scale_rhs = &(scaled_reference_load);
    NeummanForcesSurfaceComplexForLazy neumann_forces(
        m_field, Aij, arc_ctx->F_lambda, scale_lhs, scale_rhs);
    NeummanForcesSurfaceComplexForLazy::MyTriangleSpatialFE &fe_neumann =
        neumann_forces.getLoopSpatialFe();
    if (linear) {
      fe_neumann.typeOfForces = NeummanForcesSurfaceComplexForLazy::
          MyTriangleSpatialFE::NONCONSERVATIVE;
    }
    fe_neumann.uSeF = true;
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      CHKERR fe_neumann.addForce(it->getMeshsetId());
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      CHKERR fe_neumann.addPressure(it->getMeshsetId());
    }

    boost::shared_ptr<FEMethod> my_dirichlet_bc =
        boost::shared_ptr<FEMethod>(new DirichletSpatialPositionsBc(
            m_field, "SPATIAL_POSITION", Aij, D, F));
    CHKERR m_field.get_problem("ELASTIC_MECHANICS",
                               &(my_dirichlet_bc->problemPtr));
    CHKERR dynamic_cast<DirichletSpatialPositionsBc *>(my_dirichlet_bc.get())
        ->iNitalize();

    struct AssembleRhsVectors : public FEMethod {

      boost::shared_ptr<ArcLengthCtx> arcPtr;
      Range &nodeSet;

      AssembleRhsVectors(boost::shared_ptr<ArcLengthCtx> &arc_ptr,
                         Range &node_set)
          : arcPtr(arc_ptr), nodeSet(node_set) {}

      MoFEMErrorCode preProcess() {
        MoFEMFunctionBeginHot;

        // PetscAttachDebugger();
        switch (snes_ctx) {
        case CTX_SNESSETFUNCTION: {
          CHKERR VecZeroEntries(snes_f);
          CHKERR VecGhostUpdateBegin(snes_f, INSERT_VALUES, SCATTER_FORWARD);
          CHKERR VecGhostUpdateEnd(snes_f, INSERT_VALUES, SCATTER_FORWARD);
          CHKERR VecZeroEntries(arcPtr->F_lambda);
          CHKERR VecGhostUpdateBegin(arcPtr->F_lambda, INSERT_VALUES,
                                     SCATTER_FORWARD);
          CHKERR VecGhostUpdateEnd(arcPtr->F_lambda, INSERT_VALUES,
                                   SCATTER_FORWARD);
        } break;
        default:
          SETERRQ(PETSC_COMM_SELF, 1, "not implemented");
        }

        MoFEMFunctionReturnHot(0);
      }

      MoFEMErrorCode postProcess() {
        MoFEMFunctionBeginHot;
        switch (snes_ctx) {
        case CTX_SNESSETFUNCTION: {
          // snes_f
          CHKERR VecGhostUpdateBegin(snes_f, ADD_VALUES, SCATTER_REVERSE);
          CHKERR VecGhostUpdateEnd(snes_f, ADD_VALUES, SCATTER_REVERSE);
          CHKERR VecAssemblyBegin(snes_f);
          CHKERR VecAssemblyEnd(snes_f);
        } break;
        default:
          SETERRQ(PETSC_COMM_SELF, 1, "not implemented");
        }
        MoFEMFunctionReturnHot(0);
      }

      MoFEMErrorCode potsProcessLoadPath() {
        MoFEMFunctionBeginHot;
        boost::shared_ptr<NumeredDofEntity_multiIndex> numered_dofs_rows =
            problemPtr->getNumeredDofsRows();
        Range::iterator nit = nodeSet.begin();
        for (; nit != nodeSet.end(); nit++) {
          NumeredDofEntityByEnt::iterator dit, hi_dit;
          dit = numered_dofs_rows->get<Ent_mi_tag>().lower_bound(*nit);
          hi_dit = numered_dofs_rows->get<Ent_mi_tag>().upper_bound(*nit);
          for (; dit != hi_dit; dit++) {
            PetscPrintf(PETSC_COMM_WORLD, "%s [ %d ] %6.4e -> ", "LAMBDA", 0,
                        arcPtr->getFieldData());
            PetscPrintf(PETSC_COMM_WORLD, "%s [ %d ] %6.4e\n",
                        dit->get()->getName().c_str(),
                        dit->get()->getDofCoeffIdx(),
                        dit->get()->getFieldData());
          }
        }
        MoFEMFunctionReturnHot(0);
      }
    };

    struct AddLambdaVectorToFInternal : public FEMethod {

      boost::shared_ptr<ArcLengthCtx> arcPtr;
      boost::shared_ptr<DirichletSpatialPositionsBc> bC;

      AddLambdaVectorToFInternal(boost::shared_ptr<ArcLengthCtx> &arc_ptr,
                                 boost::shared_ptr<FEMethod> &bc)
          : arcPtr(arc_ptr),
            bC(boost::shared_ptr<DirichletSpatialPositionsBc>(
                bc, dynamic_cast<DirichletSpatialPositionsBc *>(bc.get()))) {}

      MoFEMErrorCode preProcess() {
        MoFEMFunctionBeginHot;
        MoFEMFunctionReturnHot(0);
      }
      MoFEMErrorCode operator()() {
        MoFEMFunctionBeginHot;
        MoFEMFunctionReturnHot(0);
      }
      MoFEMErrorCode postProcess() {
        MoFEMFunctionBeginHot;
        switch (snes_ctx) {
        case CTX_SNESSETFUNCTION: {
          // F_lambda
          CHKERR VecGhostUpdateBegin(arcPtr->F_lambda, ADD_VALUES,
                                     SCATTER_REVERSE);
          CHKERR VecGhostUpdateEnd(arcPtr->F_lambda, ADD_VALUES,
                                   SCATTER_REVERSE);
          CHKERR VecAssemblyBegin(arcPtr->F_lambda);
          CHKERR VecAssemblyEnd(arcPtr->F_lambda);
          for (std::vector<int>::iterator vit = bC->dofsIndices.begin();
               vit != bC->dofsIndices.end(); vit++) {
            CHKERR VecSetValue(arcPtr->F_lambda, *vit, 0, INSERT_VALUES);
          }
          CHKERR VecAssemblyBegin(arcPtr->F_lambda);
          CHKERR VecAssemblyEnd(arcPtr->F_lambda);
          CHKERR VecDot(arcPtr->F_lambda, arcPtr->F_lambda, &arcPtr->F_lambda2);
          PetscPrintf(PETSC_COMM_WORLD, "\tFlambda2 = %6.4e\n",
                      arcPtr->F_lambda2);
          // add F_lambda
          CHKERR VecAXPY(snes_f, arcPtr->getFieldData(), arcPtr->F_lambda);
          PetscPrintf(PETSC_COMM_WORLD, "\tlambda = %6.4e\n",
                      arcPtr->getFieldData());
          double fnorm;
          CHKERR VecNorm(snes_f, NORM_2, &fnorm);
          PetscPrintf(PETSC_COMM_WORLD, "\tfnorm = %6.4e\n", fnorm);
        } break;
        default:
          SETERRQ(PETSC_COMM_SELF, 1, "not implemented");
        }
        MoFEMFunctionReturnHot(0);
      }
    };

    AssembleRhsVectors pre_post_method(arc_ctx, node_set);
    AddLambdaVectorToFInternal assemble_F_lambda(arc_ctx, my_dirichlet_bc);

    SNES snes;
    CHKERR SNESCreate(PETSC_COMM_WORLD, &snes);
    CHKERR SNESSetApplicationContext(snes, &snes_ctx);
    CHKERR SNESSetFunction(snes, F, SnesRhs, &snes_ctx);
    CHKERR SNESSetJacobian(snes, ShellAij, Aij, SnesMat, &snes_ctx);
    CHKERR SNESSetFromOptions(snes);

    PetscReal my_tol;
    CHKERR PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-my_tol", &my_tol,
                               &flg);
    if (flg == PETSC_TRUE) {
      PetscReal atol, rtol, stol;
      PetscInt maxit, maxf;
      CHKERR SNESGetTolerances(snes, &atol, &rtol, &stol, &maxit, &maxf);
      atol = my_tol;
      rtol = atol * 1e2;
      CHKERR SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf);
    }

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

    if (flg == PETSC_TRUE) {
      PetscReal rtol, atol, dtol;
      PetscInt maxits;
      CHKERR KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxits);
      atol = my_tol * 1e-2;
      rtol = atol * 1e-2;
      CHKERR KSPSetTolerances(ksp, rtol, atol, dtol, maxits);
    }

    SnesCtx::FEMethodsSequence &loops_to_do_Rhs =
        snes_ctx.get_loops_to_do_Rhs();
    snes_ctx.get_preProcess_to_do_Rhs().push_back(my_dirichlet_bc);
    snes_ctx.get_preProcess_to_do_Rhs().push_back(&pre_post_method);
    loops_to_do_Rhs.push_back(
        SnesCtx::PairNameFEMethodPtr("ELASTIC", &elastic.getLoopFeRhs()));

    loops_to_do_Rhs.push_back(
        SnesCtx::PairNameFEMethodPtr("SPRING", fe_spring_rhs_ptr.get()));

    // surface forces and pressures
    loops_to_do_Rhs.push_back(
        SnesCtx::PairNameFEMethodPtr("NEUMANN_FE", &fe_neumann));

    // edge forces
    boost::ptr_map<std::string, EdgeForce> edge_forces;
    string fe_name_str = "FORCE_FE";
    edge_forces.insert(fe_name_str, new EdgeForce(m_field));
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      CHKERR edge_forces.at(fe_name_str)
          .addForce("SPATIAL_POSITION", arc_ctx->F_lambda, it->getMeshsetId());
    }
    for (boost::ptr_map<std::string, EdgeForce>::iterator eit =
             edge_forces.begin();
         eit != edge_forces.end(); eit++) {
      loops_to_do_Rhs.push_back(
          SnesCtx::PairNameFEMethodPtr(eit->first, &eit->second->getLoopFe()));
    }

    // nodal forces
    boost::ptr_map<std::string, NodalForce> nodal_forces;
    // string fe_name_str ="FORCE_FE";
    nodal_forces.insert(fe_name_str, new NodalForce(m_field));
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      CHKERR nodal_forces.at(fe_name_str)
          .addForce("SPATIAL_POSITION", arc_ctx->F_lambda, it->getMeshsetId());
    }
    for (boost::ptr_map<std::string, NodalForce>::iterator fit =
             nodal_forces.begin();
         fit != nodal_forces.end(); fit++) {
      loops_to_do_Rhs.push_back(
          SnesCtx::PairNameFEMethodPtr(fit->first, &fit->second->getLoopFe()));
    }

    // arc length
    loops_to_do_Rhs.push_back(
        SnesCtx::PairNameFEMethodPtr("NONE", &assemble_F_lambda));
    loops_to_do_Rhs.push_back(
        SnesCtx::PairNameFEMethodPtr("ARC_LENGTH", &arc_method));
    snes_ctx.get_postProcess_to_do_Rhs().push_back(&pre_post_method);
    snes_ctx.get_postProcess_to_do_Rhs().push_back(my_dirichlet_bc);

    SnesCtx::FEMethodsSequence &loops_to_do_Mat =
        snes_ctx.get_loops_to_do_Mat();
    snes_ctx.get_preProcess_to_do_Mat().push_back(my_dirichlet_bc);
    loops_to_do_Mat.push_back(
        SnesCtx::PairNameFEMethodPtr("ELASTIC", &elastic.getLoopFeLhs()));

    loops_to_do_Mat.push_back(
        SnesCtx::PairNameFEMethodPtr("SPRING", fe_spring_lhs_ptr.get()));

    loops_to_do_Mat.push_back(
        SnesCtx::PairNameFEMethodPtr("NEUMANN_FE", &fe_neumann));
    loops_to_do_Mat.push_back(
        SnesCtx::PairNameFEMethodPtr("ARC_LENGTH", &arc_method));
    snes_ctx.get_postProcess_to_do_Mat().push_back(my_dirichlet_bc);

    CHKERR m_field.getInterface<VecManager>()->setLocalGhostVector(
        "ELASTIC_MECHANICS", COL, D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

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
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_its_d", &its_d,
                              &flg);
    if (flg != PETSC_TRUE) {
      its_d = 4;
    }
    PetscScalar max_reduction = 10, min_reduction = 0.1;
    CHKERR PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-my_max_step_reduction",
                               &max_reduction, &flg);
    CHKERR PetscOptionsGetReal(PETSC_NULL, PETSC_NULL, "-my_min_step_reduction",
                               &min_reduction, &flg);

    double gamma = 0.5, reduction = 1;
    // step = 1;
    if (step == 1) {
      step_size = step_size_reduction;
    } else {
      reduction = step_size_reduction;
      step++;
    }
    double step_size0 = step_size;

    if (step > 1) {
      CHKERR m_field.getInterface<VecManager>()->setOtherGlobalGhostVector(
          "ELASTIC_MECHANICS", "SPATIAL_POSITION", "X0_SPATIAL_POSITION", COL,
          arc_ctx->x0, INSERT_VALUES, SCATTER_FORWARD);
      double x0_nrm;
      CHKERR VecNorm(arc_ctx->x0, NORM_2, &x0_nrm);
      CHKERR PetscPrintf(PETSC_COMM_WORLD,
                         "\tRead x0_nrm = %6.4e dlambda = %6.4e\n", x0_nrm,
                         arc_ctx->dLambda);
      CHKERR arc_ctx->setAlphaBeta(1, 0);
    } else {
      CHKERR arc_ctx->setS(step_size);
      CHKERR arc_ctx->setAlphaBeta(0, 1);
    }

    CHKERR SnesRhs(snes, D, F, &snes_ctx);

    Vec D0, x00;
    CHKERR VecDuplicate(D, &D0);
    CHKERR VecDuplicate(arc_ctx->x0, &x00);
    bool converged_state = false;

    for (int jj = 0; step < max_steps; step++, jj++) {

      CHKERR VecCopy(D, D0);
      CHKERR VecCopy(arc_ctx->x0, x00);

      if (step == 1) {

        CHKERR PetscPrintf(PETSC_COMM_WORLD, "Load Step %D step_size = %6.4e\n",
                           step, step_size);
        CHKERR arc_ctx->setS(step_size);
        CHKERR arc_ctx->setAlphaBeta(0, 1);
        CHKERR VecCopy(D, arc_ctx->x0);
        double dlambda;
        CHKERR arc_method.calculateInitDlambda(&dlambda);
        CHKERR arc_method.setDlambdaToX(D, dlambda);

      } else if (step == 2) {

        CHKERR arc_ctx->setAlphaBeta(1, 0);
        CHKERR arc_method.calculateDxAndDlambda(D);
        step_size = sqrt(arc_method.calculateLambdaInt());
        step_size0 = step_size;
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
        CHKERR arc_method.setDlambdaToX(D, dlambda);

      } else {

        if (jj == 0) {
          step_size0 = step_size;
        }

        CHKERR arc_method.calculateDxAndDlambda(D);
        step_size *= reduction;
        if (step_size > max_reduction * step_size0) {
          step_size = max_reduction * step_size0;
        } else if (step_size < min_reduction * step_size0) {
          step_size = min_reduction * step_size0;
        }
        CHKERR arc_ctx->setS(step_size);
        double dlambda = reduction * arc_ctx->dLambda;
        double dx_nrm;
        CHKERR VecScale(arc_ctx->dx, reduction);
        CHKERR VecNorm(arc_ctx->dx, NORM_2, &dx_nrm);
        CHKERR PetscPrintf(PETSC_COMM_WORLD,
                           "Load Step %D step_size = %6.4e dlambda0 = %6.4e "
                           "dx_nrm = %6.4e dx2 = %6.4e\n",
                           step, step_size, dlambda, dx_nrm, arc_ctx->dx2);
        CHKERR VecCopy(D, arc_ctx->x0);
        CHKERR VecAXPY(D, 1., arc_ctx->dx);
        CHKERR arc_method.setDlambdaToX(D, dlambda);
      }

      CHKERR SNESSolve(snes, PETSC_NULL, D);
      int its;
      CHKERR SNESGetIterationNumber(snes, &its);
      CHKERR PetscPrintf(PETSC_COMM_WORLD, "number of Newton iterations = %D\n",
                         its);

      SNESConvergedReason reason;
      CHKERR SNESGetConvergedReason(snes, &reason);
      if (reason < 0) {

        CHKERR VecCopy(D0, D);
        CHKERR VecCopy(x00, arc_ctx->x0);

        double x0_nrm;
        CHKERR VecNorm(arc_ctx->x0, NORM_2, &x0_nrm);
        CHKERR PetscPrintf(PETSC_COMM_WORLD,
                           "\tRead x0_nrm = %6.4e dlambda = %6.4e\n", x0_nrm,
                           arc_ctx->dLambda);
        CHKERR arc_ctx->setAlphaBeta(1, 0);

        reduction = 0.1;
        converged_state = false;

        continue;

      } else {

        if (step > 1 && converged_state) {

          reduction = pow((double)its_d / (double)(its + 1), gamma);
          if (step_size >= max_reduction * step_size0 && reduction > 1) {
            reduction = 1;
          } else if (step_size <= min_reduction * step_size0 && reduction < 1) {
            reduction = 1;
          }
          CHKERR PetscPrintf(PETSC_COMM_WORLD, "reduction step_size = %6.4e\n",
                             reduction);
        }

        // Save data on mesh
        CHKERR m_field.getInterface<VecManager>()->setGlobalGhostVector(
            "ELASTIC_MECHANICS", COL, D, INSERT_VALUES, SCATTER_REVERSE);
        CHKERR m_field.getInterface<VecManager>()->setOtherGlobalGhostVector(
            "ELASTIC_MECHANICS", "SPATIAL_POSITION", "X0_SPATIAL_POSITION", COL,
            arc_ctx->x0, INSERT_VALUES, SCATTER_REVERSE);
        converged_state = true;
      }

      if (step % 1 == 0) {
        // Save restart file
        // #ifdef MOAB_HDF5_PARALLEL
        //   std::ostringstream sss;
        //   sss << "restart_" << step << ".h5m";
        //   CHKERR
        //   moab.write_file(sss.str().c_str(),"MOAB","PARALLEL=WRITE_PART");
        // #else
        // #warning "No parallel HDF5, no writing restart file"
        // #endif
        // Save data on mesh
        CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "ELASTIC",
                                            post_proc);
        std::ostringstream o1;
        o1 << "out_" << step << ".h5m";
        CHKERR post_proc.writeFile(o1.str().c_str());
      }

      CHKERR pre_post_method.potsProcessLoadPath();
    }

    CHKERR VecDestroy(&D0);
    CHKERR VecDestroy(&x00);

    // detroy matrices
    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&D);
    CHKERR MatDestroy(&Aij);
    CHKERR MatDestroy(&ShellAij);
    CHKERR SNESDestroy(&snes);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
