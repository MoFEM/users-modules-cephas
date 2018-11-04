/** \file elasticity.cpp
 * \ingroup nonlinear_elastic_elem
 * \example elasticity.cpp

 The example shows how to solve the linear elastic problem. An example can read
 file with temperature field, then thermal stresses are included.

 What example can do:
 - take into account temperature field, i.e. calculate thermal stresses and deformation
 - stationary and time depend field is considered
 - take into account gravitational body forces
 - take in account fluid pressure
 - can work with higher order geometry definition
 - works on distributed meshes
 - multi-grid solver where each grid level is approximation order level
 - each mesh block can have different material parameters and approximation order

See example how code can be used \cite jordi:2017,
 \image html SquelaDamExampleByJordi.png "Example what you can do with this code. 
 Analysis of the arch dam of Susqueda, located in Catalonia (Spain)" width=800px

 This is an example of application code; it does not show how elements are
implemented. Example presents how to:
 - read mesh
 - set-up problem
 - run finite elements on the problem
 - assemble matrices and vectors
 - solve the linear system of equations
 - save results


 If you like to see how to implement finite elements, material, are other parts
of the code, look here;
 - Hooke material, see \ref Hooke
 - Thermal-stress assembly, see \ref  ThermalElement
 - Body forces element, see \ref BodyForceConstantField
 - Fluid pressure element, see \ref FluidPressure
 - The general implementation of an element for arbitrary Lagrangian-Eulerian
 formulation for a nonlinear elastic problem is here \ref
 NonlinearElasticElement. Here we limit ourselves to Hooke equation and fix
 mesh, so the problem becomes linear. Not that elastic element is implemented
 with automatic differentiation.

*/

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
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */

#include <BasicFiniteElements.hpp>
using namespace MoFEM;

#include <boost/program_options.hpp>
using namespace std;
namespace po = boost::program_options;

#include <Hooke.hpp>

using namespace boost::numeric;

static char help[] = "-my_block_config set block data\n"
                     "\n";

struct BlockOptionData {
  int oRder;
  double yOung;
  double pOisson;
  double initTemp;
  BlockOptionData() : oRder(-1), yOung(-1), pOisson(-2), initTemp(0) {}
};

int main(int argc, char *argv[]) {

  // Initialize PETSCc
  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    PetscBool flg_block_config, flg_file;
    char mesh_file_name[255];
    char block_config_file[255];
    PetscInt order = 2;
    PetscBool is_partitioned = PETSC_FALSE;

    // Read options from command line
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");
    CHKERR(ierr);
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order", "default approximation order", "", 2,
                           &order, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only part of the mes",
                            "", PETSC_FALSE, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsString("-my_block_config", "elastic configure file name",
                              "", "block_conf.in", block_config_file, 255,
                              &flg_block_config);

    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    // Throw error if file with mesh is not give
    if (flg_file != PETSC_TRUE) {
      SETERRQ(PETSC_COMM_SELF, 1, "*** ERROR -my_file (MESH FILE NEEDED)");
    }

    // Create mesh database
    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;

    // Create moab communicator
    // Create separate MOAB communicator, it is duplicate of PETSc communicator.
    // NOTE That this should eliminate potential communication problems between
    // MOAB and PETSC functions.
    MPI_Comm moab_comm_world;
    MPI_Comm_dup(PETSC_COMM_WORLD, &moab_comm_world);
    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, moab_comm_world);

    // Read whole mesh or part of is if partitioned
    if (is_partitioned == PETSC_TRUE) {
      // This is a case of distributed mesh and algebra. In that case each
      // processor
      // keep only part of the problem.
      const char *option;
      option = "PARALLEL=READ_PART;"
               "PARALLEL_RESOLVE_SHARED_ENTS;"
               "PARTITION=PARALLEL_PARTITION;";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    } else {
      // If that case we have distributed algebra, i.e. assembly of vectors and
      // matrices is in parallel, but whole mesh is stored on all processors.
      // Solver and matrix scales well, however problem set-up of problem is
      // not fully parallel.
      const char *option;
      option = "";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    }

    // Create MoFEM database and link it to MoAB
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    // Print boundary conditions and material parameters
    MeshsetsManager *meshsets_mng_ptr;
    CHKERR m_field.getInterface(meshsets_mng_ptr);
    CHKERR meshsets_mng_ptr->printDisplacementSet();
    CHKERR meshsets_mng_ptr->printForceSet();
    CHKERR meshsets_mng_ptr->printMaterialsSet();

    // Set bit refinement level to all entities (we do not refine mesh in this
    // example
    // so all entities have the same bit refinement level)
    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    // Declare approximation fields
    CHKERR m_field.add_field("DISPLACEMENT", H1, AINSWORTH_LOBATTO_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    // We can use higher oder geometry to define body
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);

    // Declare problem

    // Add entities (by tets) to the field ( all entities in the mesh, root_set
    // = 0 )
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "DISPLACEMENT");
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");

    // Set apportion order.
    // See Hierarchic Finite Element Bases on Unstructured Tetrahedral Meshes.
    CHKERR m_field.set_field_order(0, MBTET, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBTRI, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "DISPLACEMENT", 1);

    // Set order of approximation of geometry.
    // Apply 2nd order only on skin (or in in whole body)
    auto setting_second_order_geometry = [&m_field]() {
      MoFEMFunctionBegin;
      // Setting geometry order everywhere
      Range tets, edges;
      CHKERR m_field.get_moab().get_entities_by_type(0, MBTET, tets);
      CHKERR m_field.get_moab().get_adjacencies(tets, 1, false, edges,
                                                moab::Interface::UNION);

      // Setting 2nd geometry order only on skin
      // Range tets, faces, edges;
      // Skinner skin(&m_field.get_moab());
      // CHKERR skin.find_skin(0,tets,false,faces);
      // CHKERR m_field.get_moab().get_adjacencies(
      //   faces,1,false,edges,moab::Interface::UNION
      // );
      // CHKERR m_field.synchronise_entities(edges);

      CHKERR m_field.set_field_order(edges, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);
      MoFEMFunctionReturn(0);
    };
    CHKERR setting_second_order_geometry();

    // Configure blocks by parsing config file. It allows setting approximation
    // order for each block independently.
    auto setting_blocks_data_and_order_from_config_file = [&m_field,
                                                           flg_block_config]() {
      MoFEMFunctionBegin;
      std::map<int, BlockOptionData> block_data;
      if (flg_block_config) {
        ifstream ini_file(block_config_file);
        // std::cerr << block_config_file << std::endl;
        po::variables_map vm;
        po::options_description config_file_options;
        for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
          std::ostringstream str_order;
          str_order << "block_" << it->getMeshsetId() << ".displacement_order";
          config_file_options.add_options()(
              str_order.str().c_str(),
              po::value<int>(&block_data[it->getMeshsetId()].oRder)
                  ->default_value(order));

          std::ostringstream str_cond;
          str_cond << "block_" << it->getMeshsetId() << ".young_modulus";
          config_file_options.add_options()(
              str_cond.str().c_str(),
              po::value<double>(&block_data[it->getMeshsetId()].yOung)
                  ->default_value(-1));

          std::ostringstream str_capa;
          str_capa << "block_" << it->getMeshsetId() << ".poisson_ratio";
          config_file_options.add_options()(
              str_capa.str().c_str(),
              po::value<double>(&block_data[it->getMeshsetId()].pOisson)
                  ->default_value(-2));

          std::ostringstream str_init_temp;
          str_init_temp << "block_" << it->getMeshsetId()
                        << ".initial_temperature";
          config_file_options.add_options()(
              str_init_temp.str().c_str(),
              po::value<double>(&block_data[it->getMeshsetId()].initTemp)
                  ->default_value(0));
        }
        po::parsed_options parsed =
            parse_config_file(ini_file, config_file_options, true);
        store(parsed, vm);
        po::notify(vm);
        for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
          if (block_data[it->getMeshsetId()].oRder == -1)
            continue;
          if (block_data[it->getMeshsetId()].oRder == order)
            continue;
          PetscPrintf(PETSC_COMM_WORLD, "Set block %d order to %d\n",
                      it->getMeshsetId(), block_data[it->getMeshsetId()].oRder);
          Range block_ents;
          CHKERR moab.get_entities_by_handle(it->getMeshset(), block_ents,
                                             true);
          Range ents_to_set_order;
          CHKERR moab.get_adjacencies(block_ents, 3, false, ents_to_set_order,
                                      moab::Interface::UNION);
          ents_to_set_order = ents_to_set_order.subset_by_type(MBTET);
          CHKERR moab.get_adjacencies(block_ents, 2, false, ents_to_set_order,
                                      moab::Interface::UNION);
          CHKERR moab.get_adjacencies(block_ents, 1, false, ents_to_set_order,
                                      moab::Interface::UNION);
          CHKERR m_field.synchronise_entities(ents_to_set_order);

          CHKERR m_field.set_field_order(ents_to_set_order, "DISPLACEMENT",
                                         block_data[it->getMeshsetId()].oRder);
        }
        std::vector<std::string> additional_parameters;
        additional_parameters =
            collect_unrecognized(parsed.options, po::include_positional);
        for (std::vector<std::string>::iterator vit =
                 additional_parameters.begin();
             vit != additional_parameters.end(); vit++) {
          CHKERR PetscPrintf(PETSC_COMM_WORLD,
                             "** WARNING Unrecognized option %s\n",
                             vit->c_str());
        }
      }
      MoFEMFunctionReturn(0);
    };
    CHKERR setting_blocks_data_and_order_from_config_file();

    // Add elastic element
    boost::shared_ptr<Hooke<adouble>> hooke_adouble_ptr(new Hooke<adouble>());
    boost::shared_ptr<Hooke<double>> hooke_double_ptr(new Hooke<double>());
    NonlinearElasticElement elastic(m_field, 2);
    CHKERR elastic.setBlocks(hooke_double_ptr, hooke_adouble_ptr);

    CHKERR elastic.addElement("ELASTIC", "DISPLACEMENT");
    CHKERR elastic.setOperators("DISPLACEMENT", "MESH_NODE_POSITIONS", false,
                                true);

    // Update material parameters. Set material parameters block by block.
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
      int id = it->getMeshsetId();
      if (block_data[id].yOung > 0) {
        elastic.setOfBlocks[id].E = block_data[id].yOung;
        CHKERR PetscPrintf(PETSC_COMM_WORLD,
                           "Block %d set Young modulus %3.4g\n", id,
                           elastic.setOfBlocks[id].E);
      }
      if (block_data[id].pOisson >= -1) {
        elastic.setOfBlocks[id].PoissonRatio = block_data[id].pOisson;
        CHKERR PetscPrintf(PETSC_COMM_WORLD,
                           "Block %d set Poisson ratio %3.4g\n", id,
                           elastic.setOfBlocks[id].PoissonRatio);
      }
    }

    // Add body force element. This is only declaration of element. not its
    // implementation.
    CHKERR m_field.add_finite_element("BODY_FORCE");
    CHKERR m_field.modify_finite_element_add_field_row("BODY_FORCE",
                                                       "DISPLACEMENT");
    CHKERR m_field.modify_finite_element_add_field_col("BODY_FORCE",
                                                       "DISPLACEMENT");
    CHKERR m_field.modify_finite_element_add_field_data("BODY_FORCE",
                                                        "DISPLACEMENT");
    CHKERR m_field.modify_finite_element_add_field_data("BODY_FORCE",
                                                        "MESH_NODE_POSITIONS");

    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, BLOCKSET | BODYFORCESSET, it)) {
      Range tets;
      m_field.get_moab().get_entities_by_type(it->meshset, MBTET, tets, true);
      CHKERR
      m_field.add_ents_to_finite_element_by_type(tets, MBTET, "BODY_FORCE");
    }

    // Add Neumann forces, i.e. pressure or traction forces applied on body
    // surface. This is only declaration not implementation.
    CHKERR MetaNeummanForces::addNeumannBCElements(m_field, "DISPLACEMENT");
    CHKERR MetaNodalForces::addElement(m_field, "DISPLACEMENT");
    CHKERR MetaEdgeForces::addElement(m_field, "DISPLACEMENT");

    // Add fluid pressure finite elements. This is special pressure on the
    // surface from fluid, i.e. pressure which linearly change with the depth.
    FluidPressure fluid_pressure_fe(m_field);
    // This function only declare element. Element is implemented by operators
    // in class FluidPressure.
    fluid_pressure_fe.addNeumannFluidPressureBCElements("DISPLACEMENT");

    // Add elements for thermo elasticity if temperature field is defined.
    ThermalStressElement thermal_stress_elem(m_field);
    // Check if TEMP field exist, and then add element.
    if (!m_field.check_field("TEMP")) {
      bool add_temp_field = false;
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
        if (block_data[it->getMeshsetId()].initTemp != 0) {
          add_temp_field = true;
          break;
        }
      }
      if (add_temp_field) {
        CHKERR m_field.add_field("TEMP", H1, AINSWORTH_LEGENDRE_BASE, 1,
                                 MB_TAG_SPARSE, MF_ZERO);

        CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "TEMP");

        CHKERR m_field.set_field_order(0, MBVERTEX, "TEMP", 1);
      }
    }
    if (m_field.check_field("TEMP")) {
      CHKERR thermal_stress_elem.addThermalStressElement(
          "ELASTIC", "DISPLACEMENT", "TEMP");
    }

    // All is declared, at that point build fields first,
    CHKERR m_field.build_fields();
    // If 10-node test are on the mesh, use mid nodes to set HO-geometry. Class
    // Projection10NodeCoordsOnField
    // do the trick.
    Projection10NodeCoordsOnField ent_method_material(m_field,
                                                      "MESH_NODE_POSITIONS");
    CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);
    if (m_field.check_field("TEMP")) {
      for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
        if (block_data[it->getMeshsetId()].initTemp != 0) {
          PetscPrintf(PETSC_COMM_WORLD, "Set block %d temperature to %3.2g\n",
                      it->getMeshsetId(),
                      block_data[it->getMeshsetId()].initTemp);
          Range block_ents;
          CHKERR moab.get_entities_by_handle(it->meshset, block_ents, true);
          Range vertices;
          CHKERR moab.get_connectivity(block_ents, vertices, true);
          CHKERR m_field.getInterface<FieldBlas>()->setField(
              block_data[it->getMeshsetId()].initTemp, MBVERTEX, vertices,
              "TEMP");
        }
      }
    }

    // Build database for elements. Actual implementation of element is not need
    // here, only elements has to be declared.
    CHKERR m_field.build_finite_elements();
    // Build adjacencies between elements and field entities
    CHKERR m_field.build_adjacencies(bit_level0);

    // Register MOFEM DM implementation in PETSc
    CHKERR DMRegister_MGViaApproxOrders("MOFEM");

    // Create DM manager
    DM dm;
    CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
    CHKERR DMSetType(dm, "MOFEM");
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "ELASTIC_PROB", bit_level0);
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // Add elements to DM manager
    CHKERR DMMoFEMAddElement(dm, "ELASTIC");
    CHKERR DMMoFEMAddElement(dm, "BODY_FORCE");
    CHKERR DMMoFEMAddElement(dm, "FLUID_PRESSURE_FE");
    CHKERR DMMoFEMAddElement(dm, "FORCE_FE");
    CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
    CHKERR DMSetUp(dm);

    // Create matrices & vectors. Note that native PETSc DM interface is used,
    // but under the PETSc interface MOFEM implementation is running.
    Vec F, D, D0;
    CHKERR DMCreateGlobalVector(dm, &F);
    CHKERR VecDuplicate(F, &D);
    CHKERR VecDuplicate(F, &D0);
    Mat Aij;
    CHKERR DMCreateMatrix(dm, &Aij);
    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);

    // Zero vectors and matrices
    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecZeroEntries(D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR MatZeroEntries(Aij);

    // This controls how kinematic constrains are set, by blockset or nodeset.
    // Cubit
    // sets kinetic boundary conditions by blockset.
    bool flag_cubit_disp = false;
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, NODESET | DISPLACEMENTSET, it)) {
      flag_cubit_disp = true;
    }

    // Below particular implementations of finite elements are used to assemble
    // problem matrixes and vectors.  Implementation of element does not change
    // how element is declared.

    // Assemble Aij and F. Define Dirichlet bc element, which stets constrains
    // to MatrixDouble and the right hand side vector.
    boost::shared_ptr<FEMethod> dirichlet_bc_ptr;

    // if normally defined boundary conditions are not found, try to use
    // DISPLACEMENT blockset. To implementations available here, depending how
    // BC is defined on mesh file.
    if (!flag_cubit_disp) {
      dirichlet_bc_ptr =
          boost::shared_ptr<FEMethod>(new DirichletSetFieldFromBlockWithFlags(
              m_field, "DISPLACEMENT", "DISPLACEMENT", Aij, D0, F));
    } else {
      dirichlet_bc_ptr = boost::shared_ptr<FEMethod>(
          new DirichletDisplacementBc(m_field, "DISPLACEMENT", Aij, D0, F));
    }
    // That sets Dirichlet bc objects that problem is linear, i.e. no newton
    // (SNES) solver is run for this problem.
    dirichlet_bc_ptr->snes_ctx = FEMethod::CTX_SNESNONE;
    dirichlet_bc_ptr->ts_ctx = FEMethod::CTX_TSNONE;

    // D0 vector will store initial displacements
    CHKERR VecZeroEntries(D0);
    CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D0, INSERT_VALUES, SCATTER_REVERSE);
    // Run dirichlet_bc, from that on the mesh set values in vector D0. Run
    // implementation
    // of particular dirichlet_bc.
    CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
    // Set values from D0 on the field (on the mesh)
    CHKERR DMoFEMMeshToLocalVector(dm, D0, INSERT_VALUES, SCATTER_REVERSE);

    // Calculate residual forces as result of applied kinematic constrains. Run
    // implementation
    // of particular finite element implementation. Look how
    // NonlinearElasticElement is implemented,
    // in that case. We run NonlinearElasticElement with hook material.
    // Calculate right hand side vector
    elastic.getLoopFeRhs().snes_f = F;
    PetscPrintf(PETSC_COMM_WORLD, "Assemble external force vector  ...");
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &elastic.getLoopFeRhs());
    PetscPrintf(PETSC_COMM_WORLD, " done\n");
    // Assemble matrix
    elastic.getLoopFeLhs().snes_B = Aij;
    PetscPrintf(PETSC_COMM_WORLD, "Calculate stiffness matrix  ...");
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &elastic.getLoopFeLhs());
    PetscPrintf(PETSC_COMM_WORLD, " done\n");

    // Assemble pressure and traction forces. Run particular implemented for do
    // this, see
    // MetaNeummanForces how this is implemented.
    boost::ptr_map<std::string, NeummanForcesSurface> neumann_forces;
    CHKERR MetaNeummanForces::setMomentumFluxOperators(m_field, neumann_forces,
                                                       F, "DISPLACEMENT");

    {
      boost::ptr_map<std::string, NeummanForcesSurface>::iterator mit =
          neumann_forces.begin();
      for (; mit != neumann_forces.end(); mit++) {
        CHKERR DMoFEMLoopFiniteElements(dm, mit->first.c_str(),
                                        &mit->second->getLoopFe());
      }
    }
    // Assemble forces applied to nodes, see implementation in NodalForce
    boost::ptr_map<std::string, NodalForce> nodal_forces;
    CHKERR
    MetaNodalForces::setOperators(m_field, nodal_forces, F, "DISPLACEMENT");

    {
      boost::ptr_map<std::string, NodalForce>::iterator fit =
          nodal_forces.begin();
      for (; fit != nodal_forces.end(); fit++) {
        CHKERR DMoFEMLoopFiniteElements(dm, fit->first.c_str(),
                                        &fit->second->getLoopFe());
      }
    }
    // Assemble edge forces
    boost::ptr_map<std::string, EdgeForce> edge_forces;
    CHKERR MetaEdgeForces::setOperators(m_field, edge_forces, F,
                                        "DISPLACEMENT");
    {
      boost::ptr_map<std::string, EdgeForce>::iterator fit =
          edge_forces.begin();
      for (; fit != edge_forces.end(); fit++) {
        CHKERR DMoFEMLoopFiniteElements(dm, fit->first.c_str(),
                                        &fit->second->getLoopFe());
      }
    }
    // Assemble body forces, implemented in BodyForceConstantField
    BodyForceConstantField body_forces_methods(m_field);
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, BLOCKSET | BODYFORCESSET, it)) {
      CHKERR body_forces_methods.addBlock("DISPLACEMENT", F,
                                          it->getMeshsetId());
    }
    CHKERR DMoFEMLoopFiniteElements(dm, "BODY_FORCE",
                                    &body_forces_methods.getLoopFe());
    // Assemble fluid pressure forces
    CHKERR fluid_pressure_fe.setNeumannFluidPressureFiniteElementOperators(
        "DISPLACEMENT", F, false, true);

    CHKERR DMoFEMLoopFiniteElements(dm, "FLUID_PRESSURE_FE",
                                    &fluid_pressure_fe.getLoopFe());
    // Apply kinematic constrain to right hand side vector and matrix
    CHKERR DMoFEMPostProcessFiniteElements(dm, dirichlet_bc_ptr.get());

    // Matrix View
    // MatView(Aij,PETSC_VIEWER_STDOUT_WORLD);
    // MatView(Aij,PETSC_VIEWER_DRAW_WORLD);//PETSC_VIEWER_STDOUT_WORLD);
    // std::string wait;
    // std::cin >> wait;

    // Set matrix positive defined and symmetric for Cholesky and icc
    // pre-conditioner
    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);
    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);
    CHKERR VecScale(F, -1);

    // Create solver
    KSP solver;
    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);
    CHKERR KSPSetDM(solver, dm);
    CHKERR KSPSetFromOptions(solver);
    CHKERR KSPSetOperators(solver, Aij, Aij);
    // Setup multi-grid pre-conditioner if set from command line
    {
      // from PETSc example ex42.c
      PetscBool same = PETSC_FALSE;
      PC pc;
      CHKERR KSPGetPC(solver, &pc);
      PetscObjectTypeCompare((PetscObject)pc, PCMG, &same);
      if (same) {
        PCMGSetUpViaApproxOrdersCtx pc_ctx(dm, Aij, true);
        CHKERR PCMGSetUpViaApproxOrders(pc, &pc_ctx);
        CHKERR PCSetFromOptions(pc);
      } else {
        // Operators are already set, do not use DM for doing that
        CHKERR KSPSetDMActive(solver, PETSC_FALSE);
      }
    }
    CHKERR KSPSetInitialGuessKnoll(solver, PETSC_FALSE);
    CHKERR KSPSetInitialGuessNonzero(solver, PETSC_TRUE);
    // Set up solver
    CHKERR KSPSetUp(solver);

    // Set up post-processor. It is some generic implementation of finite
    // element.
    PostProcVolumeOnRefinedMesh post_proc(m_field);
    // Add operators to the elements, starting with some generic operators
    CHKERR post_proc.generateReferenceElementMesh();
    CHKERR post_proc.addFieldValuesPostProc("DISPLACEMENT");
    CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
    CHKERR post_proc.addFieldValuesGradientPostProc("DISPLACEMENT");
    // Add problem specific operator on element to post-process stresses
    post_proc.getOpPtrVector().push_back(new PostProcHookStress(
        m_field, post_proc.postProcMesh, post_proc.mapGaussPts, "DISPLACEMENT",
        post_proc.commonData, &elastic.setOfBlocks));

    // Temperature field is defined on the mesh
    if (m_field.check_field("TEMP")) {

      // Create thermal vector
      Vec F_thermal;
      CHKERR VecDuplicate(F, &F_thermal);

      // Set up implementation for calculation of thermal stress vector. Look
      // how thermal stresses and vector is assembled in ThermalStressElement.
      CHKERR thermal_stress_elem.setThermalStressRhsOperators(
          "DISPLACEMENT", "TEMP", F_thermal);

      SeriesRecorder *recorder_ptr;
      CHKERR m_field.getInterface(recorder_ptr);

      // Read time series and do thermo-elastic analysis, this is when time
      // dependent
      // temperature problem was run before on the mesh. It means that before
      // non-stationary
      // problem was solved for temperature and filed "TEMP" is stored for
      // subsequent time
      // steps in the recorder.
      if (recorder_ptr->check_series("THEMP_SERIES")) {
        // This is time dependent case, so loop of data series stored by tape
        // recorder.
        // Loop over time steps
        for (_IT_SERIES_STEPS_BY_NAME_FOR_LOOP_(recorder_ptr, "THEMP_SERIES",
                                                sit)) {
          PetscPrintf(PETSC_COMM_WORLD, "Process step %d\n",
                      sit->get_step_number());
          // Load field data for this time step
          CHKERR recorder_ptr->load_series_data("THEMP_SERIES",
                                                sit->get_step_number());

          CHKERR VecZeroEntries(F_thermal);
          CHKERR VecGhostUpdateBegin(F_thermal, INSERT_VALUES, SCATTER_FORWARD);
          CHKERR VecGhostUpdateEnd(F_thermal, INSERT_VALUES, SCATTER_FORWARD);

          // Calculate the right-hand side vector as result of thermal stresses.
          // It MetaNodalForces
          // that on "ELASTIC" element data structure the element implementation
          // in thermal_stress_elem
          // is executed.
          CHKERR DMoFEMLoopFiniteElements(
              dm, "ELASTIC", &thermal_stress_elem.getLoopThermalStressRhs());

          // Assemble vector
          CHKERR VecAssemblyBegin(F_thermal);
          CHKERR VecAssemblyEnd(F_thermal);
          // Accumulate ghost dofs
          CHKERR VecGhostUpdateBegin(F_thermal, ADD_VALUES, SCATTER_REVERSE);
          CHKERR VecGhostUpdateEnd(F_thermal, ADD_VALUES, SCATTER_REVERSE);

          // Calculate norm of vector and print values
          PetscReal nrm_F;
          CHKERR VecNorm(F, NORM_2, &nrm_F);

          PetscPrintf(PETSC_COMM_WORLD, "norm2 F = %6.4e\n", nrm_F);
          PetscReal nrm_F_thermal;
          CHKERR VecNorm(F_thermal, NORM_2, &nrm_F_thermal);
          PetscPrintf(PETSC_COMM_WORLD, "norm2 F_thermal = %6.4e\n",
                      nrm_F_thermal);

          CHKERR VecScale(F_thermal, -1);
          // check this !!!
          CHKERR VecAXPY(F_thermal, 1, F);

          // Set dirichlet boundary to thermal stresses vector
          dirichlet_bc_ptr->snes_x = D;
          dirichlet_bc_ptr->snes_f = F_thermal;
          CHKERR DMoFEMPostProcessFiniteElements(dm, dirichlet_bc_ptr.get());

          // Solve problem
          CHKERR KSPSolve(solver, F_thermal, D);
          // Add boundary conditions vector
          CHKERR VecAXPY(D, 1., D0);
          CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
          CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

          // Save data on the mesh
          CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

          // Save data on mesh
          CHKERR DMoFEMPreProcessFiniteElements(dm, dirichlet_bc_ptr.get());
          // Post-process results
          CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);

          std::ostringstream o1;
          o1 << "out_" << sit->step_number << ".h5m";
          CHKERR post_proc.writeFile(o1.str().c_str());
        }
      } else {

        // This is a case when stationary problem for temperature was solved.
        CHKERR VecZeroEntries(F_thermal);
        CHKERR VecGhostUpdateBegin(F_thermal, INSERT_VALUES, SCATTER_FORWARD);
        CHKERR VecGhostUpdateEnd(F_thermal, INSERT_VALUES, SCATTER_FORWARD);

        // Calculate the right-hand side vector with thermal stresses
        CHKERR DMoFEMLoopFiniteElements(
            dm, "ELASTIC", &thermal_stress_elem.getLoopThermalStressRhs());

        // Assemble vector
        CHKERR VecAssemblyBegin(F_thermal);
        CHKERR VecAssemblyEnd(F_thermal);

        // Accumulate ghost dofs
        CHKERR VecGhostUpdateBegin(F_thermal, ADD_VALUES, SCATTER_REVERSE);
        CHKERR VecGhostUpdateEnd(F_thermal, ADD_VALUES, SCATTER_REVERSE);

        // Calculate norm
        PetscReal nrm_F;
        CHKERR VecNorm(F, NORM_2, &nrm_F);

        PetscPrintf(PETSC_COMM_WORLD, "norm2 F = %6.4e\n", nrm_F);
        PetscReal nrm_F_thermal;
        CHKERR VecNorm(F_thermal, NORM_2, &nrm_F_thermal);

        PetscPrintf(PETSC_COMM_WORLD, "norm2 F_thermal = %6.4e\n",
                    nrm_F_thermal);

        // Add thermal stress vector and other forces vector
        CHKERR VecScale(F_thermal, -1);
        CHKERR VecAXPY(F_thermal, 1, F);

        // Apply kinetic boundary conditions
        dirichlet_bc_ptr->snes_x = D;
        dirichlet_bc_ptr->snes_f = F_thermal;
        CHKERR DMoFEMPostProcessFiniteElements(dm, dirichlet_bc_ptr.get());

        // Solve problem
        CHKERR KSPSolve(solver, F_thermal, D);
        CHKERR VecAXPY(D, 1., D0);

        // Update ghost values for solution vector
        CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
        CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

        // Save data on mesh
        CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
        CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);
        // Save results to file
        PetscPrintf(PETSC_COMM_WORLD, "Write output file ...");
        CHKERR post_proc.writeFile("out.h5m");
        PetscPrintf(PETSC_COMM_WORLD, " done\n");
      }

      // Destroy vector, no needed any more
      CHKERR VecDestroy(&F_thermal);

    } else {
      // Elastic analysis no temperature field
      // Solve for vector D
      CHKERR KSPSolve(solver, F, D);
      // Add kinetic boundary conditions
      CHKERR VecAXPY(D, 1., D0);
      // Update ghost values
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      // Save data from vector on mesh
      CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
      // Post-process results
      CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);
      // Write mesh in parallel (using h5m MOAB format, writing is in parallel)
      PetscPrintf(PETSC_COMM_WORLD, "Write output file ..,");
      CHKERR post_proc.writeFile("out.h5m");
      PetscPrintf(PETSC_COMM_WORLD, " done\n");
    }

    // Calculate elastic energy
    elastic.getLoopFeEnergy().snes_ctx = SnesMethod::CTX_SNESNONE;
    elastic.getLoopFeEnergy().eNergy = 0;
    PetscPrintf(PETSC_COMM_WORLD, "Calculate elastic energy  ...");
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &elastic.getLoopFeEnergy());
    PetscPrintf(PETSC_COMM_WORLD, " done\n");

    // Print elastic energy
    PetscPrintf(PETSC_COMM_WORLD, "Elastic energy %6.4e\n",
                elastic.getLoopFeEnergy().eNergy);

    // Destroy matrices, vecors, solver and DM
    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&D);
    CHKERR VecDestroy(&D0);
    CHKERR MatDestroy(&Aij);
    CHKERR KSPDestroy(&solver);
    CHKERR DMDestroy(&dm);

    MPI_Comm_free(&moab_comm_world);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
