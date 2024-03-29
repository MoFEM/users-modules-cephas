/** \file elasticity.cpp
 * \ingroup nonlinear_elastic_elem
 * \example elasticity.cpp

 The example shows how to solve the linear elastic problem. An example can read
 file with temperature field, then thermal stresses are included.

 What example can do:
 - take into account temperature field, i.e. calculate thermal stresses and
deformation
 - stationary and time depend field is considered
 - take into account gravitational body forces
 - take in account fluid pressure
 - can work with higher order geometry definition
 - works on distributed meshes
 - multi-grid solver where each grid level is approximation order level
 - each mesh block can have different material parameters and approximation
order

See example how code can be used \cite jordi:2017,
 \image html SquelaDamExampleByJordi.png "Example what you can do with this
code. Analysis of the arch dam of Susqueda, located in Catalonia (Spain)"
width=800px

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



#include <BasicFiniteElements.hpp>
using namespace MoFEM;

#include <Hooke.hpp>
using namespace boost::numeric;

static char help[] = "-my_block_config set block data\n"
                     "-my_order approximation order\n"
                     "-my_is_partitioned set if mesh is partitioned\n"
                     "\n";

struct BlockOptionData {
  int oRder;
  int iD;
  double yOung;
  double pOisson;
  double initTemp;

  Range tRis;

  BlockOptionData() : oRder(-1), yOung(-1), pOisson(-2), initTemp(0) {}
};

using BlockData = NonlinearElasticElement::BlockData;
using MassBlockData = ConvectiveMassElement::BlockData;
using VolUserDataOperator = VolumeElementForcesAndSourcesCore::UserDataOperator;

/// Set integration rule
struct VolRule {
  int operator()(int, int, int order) const { return 2 * order; }
};

struct PrismFE : public FatPrismElementForcesAndSourcesCore {

  using FatPrismElementForcesAndSourcesCore::
      FatPrismElementForcesAndSourcesCore;
  int getRuleTrianglesOnly(int order);
  int getRuleThroughThickness(int order);
};

int PrismFE::getRuleTrianglesOnly(int order) { return 2 * order; };
int PrismFE::getRuleThroughThickness(int order) { return 2 * order; };

using EdgeEle = MoFEM::EdgeElementForcesAndSourcesCore;

int main(int argc, char *argv[]) {

  const string default_options = "-ksp_type gmres \n"
                                 "-pc_type lu \n"
                                 "-pc_factor_mat_solver_type mumps \n"
                                 "-mat_mumps_icntl_20 0 \n"
                                 "-ksp_monitor \n"
                                 "-snes_type newtonls \n"
                                 "-snes_linesearch_type basic \n"
                                 "-snes_atol 1e-8 \n"
                                 "-snes_rtol 1e-8 \n"
                                 "-snes_monitor \n"
                                 "-ts_monitor \n"
                                 "-ts_type beuler \n";

  string param_file = "param_file.petsc";
  if (!static_cast<bool>(ifstream(param_file))) {
    std::ofstream file(param_file.c_str(), std::ios::ate);
    if (file.is_open()) {
      file << default_options;
      file.close();
    }
  }

  MoFEM::Core::Initialize(&argc, &argv, param_file.c_str(), help);
  auto core_log = logging::core::get();
  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmWorld(), "ELASTIC"));
  LogManager::setLog("ELASTIC");
  MOFEM_LOG_TAG("ELASTIC", "elasticity")

  core_log->add_sink(
      LogManager::createSink(LogManager::getStrmSync(), "ELASTIC_SYNC"));
  LogManager::setLog("ELASTIC_SYNC");
  MOFEM_LOG_TAG("ELASTIC_SYNC", "elastic_sync");

  try {

    PetscBool flg_block_config, flg_file;
    char mesh_file_name[255];
    char block_config_file[255];
    PetscInt test_nb = 0;
    PetscInt order = 2;
    PetscBool is_partitioned = PETSC_FALSE;
    PetscBool is_calculating_frequency = PETSC_FALSE;
    PetscBool is_post_proc_volume = PETSC_TRUE;

    // Select base
    enum bases { LEGENDRE, LOBATTO, BERNSTEIN_BEZIER, JACOBI, LASBASETOP };
    const char *list_bases[] = {"legendre", "lobatto", "bernstein_bezier",
                                "jacobi"};
    PetscInt choice_base_value = LOBATTO;

    // Read options from command line
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Elastic Config", "none");
    CHKERR(ierr);
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "mesh.h5m",
                              mesh_file_name, 255, &flg_file);

    CHKERR PetscOptionsInt("-my_order", "default approximation order", "",
                           order, &order, PETSC_NULL);

    CHKERR PetscOptionsEList("-base", "approximation base", "", list_bases,
                             LASBASETOP, list_bases[choice_base_value],
                             &choice_base_value, PETSC_NULL);

    CHKERR PetscOptionsInt("-is_atom_test", "ctest number", "", test_nb,
                           &test_nb, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only one part of the mesh)",
                            "", is_partitioned, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsString("-my_block_config", "elastic configure file name",
                              "", "block_conf.in", block_config_file, 255,
                              &flg_block_config);

    CHKERR PetscOptionsBool(
        "-my_is_calculating_frequency", "set if frequency will be calculated",
        "", is_calculating_frequency, &is_calculating_frequency, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_post_proc_volume",
                            "if true post proc volume", "", is_post_proc_volume,
                            &is_post_proc_volume, PETSC_NULL);

    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    // Throw error if file with mesh is not provided
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

    // Read whole mesh or part of it if partitioned
    if (is_partitioned == PETSC_TRUE) {
      // This is a case of distributed mesh and algebra. In this case each
      // processor keeps only one part of the problem.
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

    bool mesh_has_tets = false;
    bool mesh_has_prisms = false;
    int nb_tets = 0;
    int nb_hexs = 0;
    int nb_prisms = 0;

    CHKERR moab.get_number_entities_by_type(0, MBTET, nb_tets, true);
    CHKERR moab.get_number_entities_by_type(0, MBHEX, nb_hexs, true);
    CHKERR moab.get_number_entities_by_type(0, MBPRISM, nb_prisms, true);

    mesh_has_tets = (nb_tets + nb_hexs) > 0;
    mesh_has_prisms = nb_prisms > 0;

    // Set bit refinement level to all entities (we do not refine mesh in
    // this example so all entities have the same bit refinement level)
    BitRefLevel bit_level0;
    bit_level0.set(0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);

    // CHECK IF EDGE BLOCKSET EXIST AND IF IT IS ADD ALL ENTITIES FROM IT
    // CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
    // MESHSET_OF_EDGE_BLOCKSET, 1, bit_level0);

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 3, "ROD") == 0) {
        CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
            0, 1, bit_level0);
      }
    }

    // Declare approximation fields
    FieldApproximationBase base = NOBASE;
    switch (choice_base_value) {
    case LEGENDRE:
      base = AINSWORTH_LEGENDRE_BASE;
      break;
    case LOBATTO:
      base = AINSWORTH_LOBATTO_BASE;
      break;
    case BERNSTEIN_BEZIER:
      base = AINSWORTH_BERNSTEIN_BEZIER_BASE;
      break;
    case JACOBI:
      base = DEMKOWICZ_JACOBI_BASE;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, MOFEM_NOT_IMPLEMENTED, "Base not implemented");
    };
    CHKERR m_field.add_field("DISPLACEMENT", H1, base, 3, MB_TAG_DENSE,
                             MF_ZERO);

    // We can use higher oder geometry to define body
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, base, 3, MB_TAG_DENSE,
                             MF_ZERO);

    // Declare problem

    // Add entities (by tets) to the field (all entities in the mesh, root_set
    // = 0 )
    CHKERR m_field.add_ents_to_field_by_dim(0, 3, "DISPLACEMENT");
    CHKERR m_field.add_ents_to_field_by_dim(0, 3, "MESH_NODE_POSITIONS");

    // Get all edges in the mesh
    Range all_edges;
    CHKERR m_field.get_moab().get_entities_by_type(0, MBEDGE, all_edges, true);

    // Get edges associated with simple rod
    Range edges_in_simple_rod;
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 3, "ROD") == 0) {
        Range edges;
        CHKERR m_field.get_moab().get_entities_by_type(bit->getMeshset(),
                                                       MBEDGE, edges, true);
        edges_in_simple_rod.merge(edges);
      }
    }

    CHKERR m_field.add_ents_to_field_by_type(edges_in_simple_rod, MBEDGE,
                                             "DISPLACEMENT");

    // Set order of edge in rod to be 1
    CHKERR m_field.set_field_order(edges_in_simple_rod, "DISPLACEMENT", 1);

    // Get remaining edges (not associated with simple rod) to set order
    Range edges_to_set_order;
    edges_to_set_order = subtract(all_edges, edges_in_simple_rod);

    // Set approximation order.
    // See Hierarchical Finite Element Bases on Unstructured Tetrahedral
    // Meshes.
    CHKERR m_field.set_field_order(0, MBPRISM, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBTET, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBHEX, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBTRI, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBQUAD, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(edges_to_set_order, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "DISPLACEMENT", 1);

    if (base == AINSWORTH_BERNSTEIN_BEZIER_BASE)
      CHKERR m_field.set_field_order(0, MBVERTEX, "DISPLACEMENT", order);
    else
      CHKERR m_field.set_field_order(0, MBVERTEX, "DISPLACEMENT", 1);

    // Set order of approximation of geometry.
    // Apply 2nd order only on skin (or in whole body)
    auto setting_second_order_geometry = [&m_field]() {
      MoFEMFunctionBegin;
      // Setting geometry order everywhere
      Range tets, edges;
      CHKERR m_field.get_moab().get_entities_by_dimension(0, 3, tets);
      CHKERR m_field.get_moab().get_adjacencies(tets, 1, false, edges,
                                                moab::Interface::UNION);

      // Setting 2nd geometry order only on skin
      // Range tets, faces, edges;
      // Skinner skin(&m_field.get_moab());
      // CHKERR skin.find_skin(0,tets,false,faces);
      // CHKERR m_field.get_moab().get_adjacencies(
      //   faces,1,false,edges,moab::Interface::UNION
      // );
      // CHKERR
      // m_field.getInterface<CommInterface>()->synchroniseEntities(edges);

      CHKERR m_field.set_field_order(edges, "MESH_NODE_POSITIONS", 2);
      CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

      MoFEMFunctionReturn(0);
    };
    CHKERR setting_second_order_geometry();

    // Configure blocks by parsing config file. It allows setting
    // approximation order for each block independently.
    std::map<int, BlockOptionData> block_data;
    auto setting_blocks_data_and_order_from_config_file =
        [&m_field, &moab, &block_data, flg_block_config, block_config_file,
         order](boost::shared_ptr<std::map<int, BlockData>> &block_sets_ptr) {
          MoFEMFunctionBegin;
          if (flg_block_config) {
            ifstream ini_file(block_config_file);
            po::variables_map vm;
            po::options_description config_file_options;
            for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET,
                                                         it)) {
              std::ostringstream str_order;
              str_order << "block_" << it->getMeshsetId()
                        << ".displacement_order";
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
            for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET,
                                                         it)) {
              if (block_data[it->getMeshsetId()].oRder == -1)
                continue;
              if (block_data[it->getMeshsetId()].oRder == order)
                continue;
              MOFEM_LOG_C("ELASTIC", Sev::inform, "Set block %d order to %d",
                          it->getMeshsetId(),
                          block_data[it->getMeshsetId()].oRder);
              Range block_ents;
              CHKERR moab.get_entities_by_handle(it->getMeshset(), block_ents,
                                                 true);
              Range ents_to_set_order;
              CHKERR moab.get_adjacencies(block_ents, 3, false,
                                          ents_to_set_order,
                                          moab::Interface::UNION);
              ents_to_set_order = ents_to_set_order.subset_by_dimension(3);
              CHKERR moab.get_adjacencies(block_ents, 2, false,
                                          ents_to_set_order,
                                          moab::Interface::UNION);
              CHKERR moab.get_adjacencies(block_ents, 1, false,
                                          ents_to_set_order,
                                          moab::Interface::UNION);
              CHKERR m_field.getInterface<CommInterface>()->synchroniseEntities(
                  ents_to_set_order);

              CHKERR m_field.set_field_order(
                  ents_to_set_order, "DISPLACEMENT",
                  block_data[it->getMeshsetId()].oRder);
            }
            std::vector<std::string> additional_parameters;
            additional_parameters =
                collect_unrecognized(parsed.options, po::include_positional);
            for (std::vector<std::string>::iterator vit =
                     additional_parameters.begin();
                 vit != additional_parameters.end(); vit++) {
              MOFEM_LOG_C("ELASTIC", Sev::warning, "Unrecognized option %s",
                          vit->c_str());
            }
          }

          // Update material parameters. Set material parameters block by
          // block.
          for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
                   m_field, BLOCKSET | MAT_ELASTICSET, it)) {
            const int id = it->getMeshsetId();
            auto &bd = (*block_sets_ptr)[id];
            if (block_data[id].yOung > 0)
              bd.E = block_data[id].yOung;
            if (block_data[id].pOisson >= -1)
              bd.PoissonRatio = block_data[id].pOisson;
            MOFEM_LOG_C("ELASTIC", Sev::inform, "Block %d", id);
            MOFEM_LOG_C("ELASTIC", Sev::inform, "\tYoung modulus %3.4g", bd.E);
            MOFEM_LOG_C("ELASTIC", Sev::inform, "\tPoisson ratio %3.4g",
                        bd.PoissonRatio);
          }

          MoFEMFunctionReturn(0);
        };

    // Add elastic element

    boost::shared_ptr<std::map<int, HookeElement::BlockData>> block_sets_ptr =
        boost::make_shared<std::map<int, HookeElement::BlockData>>();
    CHKERR HookeElement::setBlocks(m_field, block_sets_ptr);
    CHKERR setting_blocks_data_and_order_from_config_file(block_sets_ptr);

    boost::shared_ptr<std::map<int, MassBlockData>> mass_block_sets_ptr =
        boost::make_shared<std::map<int, MassBlockData>>();
    CHKERR ConvectiveMassElement::setBlocks(m_field, mass_block_sets_ptr);

    auto fe_lhs_ptr =
        boost::make_shared<VolumeElementForcesAndSourcesCore>(m_field);
    auto fe_rhs_ptr =
        boost::make_shared<VolumeElementForcesAndSourcesCore>(m_field);
    fe_lhs_ptr->getRuleHook = VolRule();
    fe_rhs_ptr->getRuleHook = VolRule();

    CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *fe_lhs_ptr, true, false, false,
                    false);
    CHKERR addHOOpsVol("MESH_NODE_POSITIONS", *fe_rhs_ptr, true, false, false,
                    false);

    boost::shared_ptr<ForcesAndSourcesCore> prism_fe_lhs_ptr(
        new PrismFE(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> prism_fe_rhs_ptr(
        new PrismFE(m_field));

    CHKERR HookeElement::addElasticElement(m_field, block_sets_ptr, "ELASTIC",
                                           "DISPLACEMENT",
                                           "MESH_NODE_POSITIONS", false);

    auto add_skin_element_for_post_processing = [&]() {
      MoFEMFunctionBegin;
      Range elastic_element_ents;
      CHKERR m_field.get_finite_element_entities_by_dimension(
          "ELASTIC", 3, elastic_element_ents);
      Skinner skin(&m_field.get_moab());
      Range skin_faces; // skin faces from 3d ents
      CHKERR skin.find_skin(0, elastic_element_ents, false, skin_faces);
      Range proc_skin;
      if (is_partitioned) {
        CHKERR pcomm->filter_pstatus(skin_faces,
                                     PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                     PSTATUS_NOT, -1, &proc_skin);
      } else {
        proc_skin = skin_faces;
      }
      CHKERR m_field.add_finite_element("POST_PROC_SKIN");
      CHKERR m_field.modify_finite_element_add_field_row("POST_PROC_SKIN",
                                                          "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_col("POST_PROC_SKIN",
                                                         "DISPLACEMENT");
      CHKERR m_field.modify_finite_element_add_field_data("POST_PROC_SKIN",
                                                          "DISPLACEMENT");
      if (m_field.check_field("TEMP")) {
        // CHKERR m_field.modify_finite_element_add_field_row("POST_PROC_SKIN",
        //                                                    "TEMP");
        // CHKERR m_field.modify_finite_element_add_field_col("POST_PROC_SKIN",
        //                                                    "TEMP");
        CHKERR m_field.modify_finite_element_add_field_data("POST_PROC_SKIN",
                                                            "TEMP");
      }                                                    
      CHKERR m_field.modify_finite_element_add_field_data(
          "POST_PROC_SKIN", "MESH_NODE_POSITIONS");
      CHKERR m_field.add_ents_to_finite_element_by_dim(proc_skin, 2,
                                                       "POST_PROC_SKIN");
      MoFEMFunctionReturn(0);
    };
    CHKERR add_skin_element_for_post_processing();

    auto data_at_pts = boost::make_shared<HookeElement::DataAtIntegrationPts>();
    if (mesh_has_tets) {
      CHKERR HookeElement::setOperators(fe_lhs_ptr, fe_rhs_ptr, block_sets_ptr,
                                        "DISPLACEMENT", "MESH_NODE_POSITIONS",
                                        false, true, MBTET, data_at_pts);
    }
    if (mesh_has_prisms) {
      CHKERR HookeElement::setOperators(
          prism_fe_lhs_ptr, prism_fe_rhs_ptr, block_sets_ptr, "DISPLACEMENT",
          "MESH_NODE_POSITIONS", false, true, MBPRISM, data_at_pts);
    }

    if (test_nb == 4) {

      auto thermal_strain =
          [](FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> &t_coords) {
            FTensor::Tensor2_symmetric<double, 3> t_thermal_strain;
            constexpr double alpha = 1;
            FTensor::Index<'i', 3> i;
            FTensor::Index<'k', 3> j;
            constexpr auto t_kd = FTensor::Kronecker_Delta_symmetric<int>();
            t_thermal_strain(i, j) = alpha * t_coords(2) * t_kd(i, j);
            return t_thermal_strain;
          };

      fe_rhs_ptr->getOpPtrVector().push_back(
          new HookeElement::OpAnalyticalInternalStrain_dx<0>(
              "DISPLACEMENT", data_at_pts, thermal_strain));
    }

    boost::shared_ptr<VolumeElementForcesAndSourcesCore> fe_mass_ptr(
        new VolumeElementForcesAndSourcesCore(m_field));

    for (auto &sit : *block_sets_ptr) {
      for (auto &mit : *mass_block_sets_ptr) {
        fe_mass_ptr->getOpPtrVector().push_back(
            new HookeElement::OpCalculateMassMatrix("DISPLACEMENT",
                                                    "DISPLACEMENT", sit.second,
                                                    mit.second, data_at_pts));
      }
    }

    // Add spring boundary condition applied on surfaces.
    // This is only declaration not implementation.
    CHKERR MetaSpringBC::addSpringElements(m_field, "DISPLACEMENT",
                                           "MESH_NODE_POSITIONS");

    // Implementation of spring element
    // Create new instances of face elements for springs
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_lhs_ptr(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<FaceElementForcesAndSourcesCore> fe_spring_rhs_ptr(
        new FaceElementForcesAndSourcesCore(m_field));

    CHKERR MetaSpringBC::setSpringOperators(m_field, fe_spring_lhs_ptr,
                                            fe_spring_rhs_ptr, "DISPLACEMENT",
                                            "MESH_NODE_POSITIONS");

    // Add Simple Rod elements
    // This is only declaration not implementation.
    CHKERR MetaSimpleRodElement::addSimpleRodElements(m_field, "DISPLACEMENT",
                                                      "MESH_NODE_POSITIONS");

    // CHKERR m_field.add_ents_to_finite_element_by_type(edges_in_simple_rod,
    //                                                   MBEDGE, "SIMPLE_ROD");

    // Implementation of Simple Rod element
    // Create new instances of edge elements for Simple Rod
    boost::shared_ptr<EdgeEle> fe_simple_rod_lhs_ptr(new EdgeEle(m_field));
    boost::shared_ptr<EdgeEle> fe_simple_rod_rhs_ptr(new EdgeEle(m_field));

    
    CHKERR MetaSimpleRodElement::setSimpleRodOperators(
        m_field, fe_simple_rod_lhs_ptr, fe_simple_rod_rhs_ptr, "DISPLACEMENT",
        "MESH_NODE_POSITIONS");

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
      CHKERR m_field.get_moab().get_entities_by_dimension(it->meshset, 3, tets,
                                                          true);
      CHKERR m_field.add_ents_to_finite_element_by_dim(tets, 3, "BODY_FORCE");
    }
    CHKERR m_field.build_finite_elements("BODY_FORCE");

    // Add Neumann forces, i.e. pressure or traction forces applied on body
    // surface. This is only declaration not implementation.
    CHKERR MetaNeumannForces::addNeumannBCElements(m_field, "DISPLACEMENT");
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

    // All is declared, at this point build fields first,
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
          MOFEM_LOG_C("ELASTIC", Sev::inform,
                      "Set block %d temperature to %3.2g\n", it->getMeshsetId(),
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
    auto dm = createDM(PETSC_COMM_WORLD, "MOFEM");
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "ELASTIC_PROB", bit_level0);
    CHKERR DMSetFromOptions(dm);
    CHKERR DMMoFEMSetIsPartitioned(dm, is_partitioned);
    // Add elements to DM manager
    CHKERR DMMoFEMAddElement(dm, "ELASTIC");
    CHKERR DMMoFEMAddElement(dm, "SPRING");
    CHKERR DMMoFEMAddElement(dm, "SIMPLE_ROD");
    CHKERR DMMoFEMAddElement(dm, "BODY_FORCE");
    CHKERR DMMoFEMAddElement(dm, "FLUID_PRESSURE_FE");
    CHKERR DMMoFEMAddElement(dm, "FORCE_FE");
    CHKERR DMMoFEMAddElement(dm, "PRESSURE_FE");
    CHKERR DMMoFEMAddElement(dm, "POST_PROC_SKIN");
    CHKERR DMSetUp(dm);

    // Create matrices & vectors. Note that native PETSc DM interface is used,
    // but under the PETSc interface MoFEM implementation is running.
    SmartPetscObj<Vec> F;
    CHKERR DMCreateGlobalVector_MoFEM(dm, F);
    auto D = vectorDuplicate(F);
    auto D0 = vectorDuplicate(F);
    SmartPetscObj<Mat> Aij;
    CHKERR DMCreateMatrix_MoFEM(dm, Aij);
    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);

    // Initialise mass matrix
    SmartPetscObj<Mat> Mij;
    if (is_calculating_frequency == PETSC_TRUE) {
      Mij = matDuplicate(Aij, MAT_DO_NOT_COPY_VALUES);
      CHKERR MatSetOption(Mij, MAT_SPD, PETSC_TRUE);
      // MatView(Mij, PETSC_VIEWER_STDOUT_SELF);
    }

    // Assign global matrix/vector contributed by springs
    fe_spring_lhs_ptr->ksp_B = Aij;
    fe_spring_rhs_ptr->ksp_f = F;

    // Assign global matrix/vector contributed by Simple Rod
    fe_simple_rod_lhs_ptr->ksp_B = Aij;
    fe_simple_rod_rhs_ptr->ksp_f = F;

    // Zero vectors and matrices
    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecZeroEntries(D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
    CHKERR MatZeroEntries(Aij);

    // Below particular implementations of finite elements are used to assemble
    // problem matrixes and vectors.  Implementation of element does not change
    // how element is declared.

    // Assemble Aij and F. Define Dirichlet bc element, which sets constrains
    // to MatrixDouble and the right hand side vector.

    // if normally defined boundary conditions are not found,
    // DirichletDisplacementBc will try to use DISPLACEMENT blockset. Two
    // implementations are available, depending how BC is defined on mesh file.
    auto dirichlet_bc_ptr = boost::make_shared<DirichletDisplacementBc>(
        m_field, "DISPLACEMENT", Aij, D0, F);

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

    CHKERR VecGhostUpdateBegin(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D0, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR DMoFEMMeshToLocalVector(dm, D0, INSERT_VALUES, SCATTER_REVERSE);

    // Calculate residual forces as result of applied kinematic constrains. Run
    // implementation
    // of particular finite element implementation. Look how
    // NonlinearElasticElement is implemented,
    // in that case. We run NonlinearElasticElement with hook material.
    // Calculate right hand side vector
    fe_rhs_ptr->snes_f = F;
    prism_fe_rhs_ptr->snes_f = F;
    MOFEM_LOG("ELASTIC", Sev::inform) << "Assemble external force vector  ...";
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", fe_rhs_ptr);
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", prism_fe_rhs_ptr);
    MOFEM_LOG("ELASTIC", Sev::inform) << "done";
    // Assemble matrix
    fe_lhs_ptr->snes_B = Aij;
    prism_fe_lhs_ptr->snes_B = Aij;
    MOFEM_LOG("ELASTIC", Sev::inform) << "Calculate stiffness matrix  ...";
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", fe_lhs_ptr);
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", prism_fe_lhs_ptr);
    MOFEM_LOG("ELASTIC", Sev::inform) << "done";

    // Assemble springs
    CHKERR DMoFEMLoopFiniteElements(dm, "SPRING", fe_spring_lhs_ptr);
    CHKERR DMoFEMLoopFiniteElements(dm, "SPRING", fe_spring_rhs_ptr);

    // Assemble Simple Rod
    CHKERR DMoFEMLoopFiniteElements(dm, "SIMPLE_ROD", fe_simple_rod_lhs_ptr);
    CHKERR DMoFEMLoopFiniteElements(dm, "SIMPLE_ROD", fe_simple_rod_rhs_ptr);

    if (is_calculating_frequency == PETSC_TRUE) {
      // Assemble mass matrix
      fe_mass_ptr->snes_B = Mij;
      MOFEM_LOG("ELASTIC", Sev::inform) << "Calculate mass matrix  ...";
      CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", fe_mass_ptr);
      MOFEM_LOG("ELASTIC", Sev::inform) << "done";
    }

    // MatView(Aij, PETSC_VIEWER_STDOUT_SELF);

    // Assemble pressure and traction forces. Run particular implemented for do
    // this, see
    // MetaNeumannForces how this is implemented.
    boost::ptr_map<std::string, NeumannForcesSurface> neumann_forces;
    CHKERR MetaNeumannForces::setMomentumFluxOperators(m_field, neumann_forces,
                                                       F, "DISPLACEMENT");

    {
      boost::ptr_map<std::string, NeumannForcesSurface>::iterator mit =
          neumann_forces.begin();
      for (; mit != neumann_forces.end(); mit++) {
        CHKERR DMoFEMLoopFiniteElements(dm, mit->first.c_str(),
                                        &mit->second->getLoopFe());
      }
    }
    // Assemble forces applied to nodes, see implementation in NodalForce
    boost::ptr_map<std::string, NodalForce> nodal_forces;
    CHKERR MetaNodalForces::setOperators(m_field, nodal_forces, F,
                                         "DISPLACEMENT");

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
      auto fit = edge_forces.begin();
      for (; fit != edge_forces.end(); fit++) {
        auto &fe = fit->second->getLoopFe();
        CHKERR DMoFEMLoopFiniteElements(dm, fit->first.c_str(), &fe);
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
    CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS", fluid_pressure_fe.getLoopFe(),
                          false, false);
    CHKERR fluid_pressure_fe.setNeumannFluidPressureFiniteElementOperators(
        "DISPLACEMENT", F, false, true);

    CHKERR DMoFEMLoopFiniteElements(dm, "FLUID_PRESSURE_FE",
                                    &fluid_pressure_fe.getLoopFe());
    // Apply kinematic constrain to right hand side vector and matrix
    CHKERR DMoFEMPostProcessFiniteElements(dm, dirichlet_bc_ptr.get());

    // Matrix View
    PetscViewerPushFormat(
        PETSC_VIEWER_STDOUT_SELF,
        PETSC_VIEWER_ASCII_MATLAB); /// PETSC_VIEWER_ASCII_DENSE,
                                    /// PETSC_VIEWER_ASCII_MATLAB
    // MatView(Aij, PETSC_VIEWER_STDOUT_SELF);
    // MatView(Aij,PETSC_VIEWER_DRAW_WORLD);//PETSC_VIEWER_STDOUT_WORLD);
    // std::string wait;
    // std::cin >> wait;

    if (is_calculating_frequency == PETSC_TRUE) {
      CHKERR MatAssemblyBegin(Mij, MAT_FINAL_ASSEMBLY);
      CHKERR MatAssemblyEnd(Mij, MAT_FINAL_ASSEMBLY);
    }

    // Set matrix positive defined and symmetric for Cholesky and icc
    // pre-conditioner

    CHKERR MatSetOption(Aij, MAT_SPD, PETSC_TRUE);
    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);
    CHKERR VecScale(F, -1);

    // Create solver
    auto solver = createKSP(PETSC_COMM_WORLD);
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

    auto set_post_proc_skin = [&](auto &post_proc_skin) {
      MoFEMFunctionBegin;
      CHKERR addHOOpsFace3D("MESH_NODE_POSITIONS", post_proc_skin, false,
                            false);
      CHKERR post_proc_skin.generateReferenceElementMesh();
      CHKERR post_proc_skin.addFieldValuesPostProc("DISPLACEMENT");
      CHKERR post_proc_skin.addFieldValuesPostProc("MESH_NODE_POSITIONS");
      CHKERR post_proc_skin.addFieldValuesGradientPostProcOnSkin(
          "DISPLACEMENT", "ELASTIC", data_at_pts->hMat, true);
      CHKERR post_proc_skin.addFieldValuesGradientPostProcOnSkin(
          "MESH_NODE_POSITIONS", "ELASTIC", data_at_pts->HMat, false);
      if (m_field.check_field("TEMP")) {
        CHKERR post_proc_skin.addFieldValuesPostProc("TEMP");
        CHKERR post_proc_skin.addFieldValuesGradientPostProc("TEMP");
      }
      post_proc_skin.getOpPtrVector().push_back(
          new HookeElement::OpPostProcHookeElement<
              FaceElementForcesAndSourcesCore>(
              "DISPLACEMENT", data_at_pts, *block_sets_ptr,
              post_proc_skin.postProcMesh, post_proc_skin.mapGaussPts, true,
              true));
      MoFEMFunctionReturn(0);
    };

    auto set_post_proc_tets = [&](auto &post_proc) {
      MoFEMFunctionBegin;
      // Add operators to the elements, starting with some generic operators
      CHKERR post_proc.generateReferenceElementMesh();
      CHKERR addHOOpsVol("MESH_NODE_POSITIONS", post_proc, true, false, false,
                      false);
      CHKERR post_proc.addFieldValuesPostProc("DISPLACEMENT");
      CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
      CHKERR post_proc.addFieldValuesGradientPostProc("DISPLACEMENT");
      if (m_field.check_field("TEMP")) {
        CHKERR post_proc.addFieldValuesPostProc("TEMP");
        CHKERR post_proc.addFieldValuesGradientPostProc("TEMP");
      }
      // Add problem specific operator on element to post-process stresses
      post_proc.getOpPtrVector().push_back(new PostProcHookStress(
          m_field, post_proc.postProcMesh, post_proc.mapGaussPts,
          "DISPLACEMENT", post_proc.commonData, block_sets_ptr.get()));
      MoFEMFunctionReturn(0);
    };

    auto set_post_proc_edge = [&](auto &post_proc_edge) {
      MoFEMFunctionBegin;
      CHKERR post_proc_edge.generateReferenceElementMesh();
      CHKERR post_proc_edge.addFieldValuesPostProc("DISPLACEMENT");
      MoFEMFunctionReturn(0);
    };

    auto set_post_proc_prisms = [&](auto &prism_post_proc) {
      MoFEMFunctionBegin;
      CHKERR prism_post_proc.generateReferenceElementMesh();
      boost::shared_ptr<MatrixDouble> inv_jac_ptr(new MatrixDouble);
      prism_post_proc.getOpPtrVector().push_back(
          new OpCalculateInvJacForFatPrism(inv_jac_ptr));
      prism_post_proc.getOpPtrVector().push_back(
          new OpSetInvJacH1ForFatPrism(inv_jac_ptr));
      CHKERR prism_post_proc.addFieldValuesPostProc("DISPLACEMENT");
      CHKERR prism_post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
      CHKERR prism_post_proc.addFieldValuesGradientPostProc("DISPLACEMENT");
      prism_post_proc.getOpPtrVector().push_back(new PostProcHookStress(
          m_field, prism_post_proc.postProcMesh, prism_post_proc.mapGaussPts,
          "DISPLACEMENT", prism_post_proc.commonData, block_sets_ptr.get()));
      MoFEMFunctionReturn(0);
    };

    PostProcFaceOnRefinedMesh post_proc_skin(m_field);
    PostProcFatPrismOnRefinedMesh prism_post_proc(m_field);
    PostProcEdgeOnRefinedMesh post_proc_edge(m_field);
    PostProcVolumeOnRefinedMesh post_proc(m_field);

    CHKERR set_post_proc_skin(post_proc_skin);
    CHKERR set_post_proc_tets(post_proc);
    CHKERR set_post_proc_prisms(prism_post_proc);
    CHKERR set_post_proc_edge(post_proc_edge);

    PetscBool field_eval_flag = PETSC_FALSE;
    std::array<double, 3> field_eval_coords;
    boost::shared_ptr<FieldEvaluatorInterface::SetPtsData> field_eval_data;
    PetscInt coords_dim = 3;
    CHKERR PetscOptionsGetRealArray(NULL, NULL, "-field_eval_coords",
                                    field_eval_coords.data(), &coords_dim,
                                    &field_eval_flag);

    auto scalar_field_ptr = boost::make_shared<VectorDouble>();
    auto vector_field_ptr = boost::make_shared<MatrixDouble>();
    auto tensor_field_ptr = boost::make_shared<MatrixDouble>();

    if (field_eval_flag) {
      field_eval_data = m_field.getInterface<FieldEvaluatorInterface>()
                            ->getData<VolumeElementForcesAndSourcesCore>();
      CHKERR m_field.getInterface<FieldEvaluatorInterface>()->buildTree3D(
          field_eval_data, "ELASTIC");

      field_eval_data->setEvalPoints(field_eval_coords.data(), 1);
      auto no_rule = [](int, int, int) { return -1; };

      auto field_eval_fe_ptr = field_eval_data->feMethodPtr.lock();
      field_eval_fe_ptr->getRuleHook = no_rule;

      if (m_field.check_field("TEMP")) {
        field_eval_fe_ptr->getOpPtrVector().push_back(
            new OpCalculateScalarFieldValues("TEMP", scalar_field_ptr));
      }
      field_eval_fe_ptr->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<3>("DISPLACEMENT", vector_field_ptr));
       field_eval_fe_ptr->getOpPtrVector().push_back(
         new OpCalculateVectorFieldGradient<3, 3>("DISPLACEMENT", tensor_field_ptr));
    }

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
          MOFEM_LOG_C("ELASTIC", Sev::inform, "Process step %d",
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

          MOFEM_LOG_C("ELASTIC", Sev::inform, "norm2 F = %6.4e", nrm_F);
          PetscReal nrm_F_thermal;
          CHKERR VecNorm(F_thermal, NORM_2, &nrm_F_thermal);
          MOFEM_LOG_C("ELASTIC", Sev::inform, "norm2 F_thermal = %6.4e",
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

          if (field_eval_flag) {
            CHKERR m_field.getInterface<FieldEvaluatorInterface>()
                ->evalFEAtThePoint3D(
                    field_eval_coords.data(), 1e-12, "ELASTIC_PROB", "ELASTIC",
                    field_eval_data, m_field.get_comm_rank(),
                    m_field.get_comm_rank(), nullptr, MF_EXIST, QUIET);
            if (scalar_field_ptr->size()) {
              auto t_temp = getFTensor0FromVec(*scalar_field_ptr);
              MOFEM_LOG("ELASTIC_SYNC", Sev::inform)
                  << "Eval point TEMP: " << t_temp;
            }
            if (vector_field_ptr->size1()) {
              FTensor::Index<'i', 3> i;
              auto t_disp = getFTensor1FromMat<3>(*vector_field_ptr);
              MOFEM_LOG("ELASTIC_SYNC", Sev::inform)
                  << "Eval point DISPLACEMENT magnitude: "
                  << sqrt(t_disp(i) * t_disp(i));
            }
            if (tensor_field_ptr->size1()) {
              FTensor::Index<'i', 3> i;
              auto t_disp_grad = getFTensor2FromMat<3, 3>(*tensor_field_ptr);
              MOFEM_LOG("ELASTIC_SYNC", Sev::inform)
                  << "Eval point DISPLACEMENT_GRAD trace: " << t_disp_grad(i, i);
            }

            MOFEM_LOG_SYNCHRONISE(m_field.get_comm());
          }

          // Post-process results
          if (is_post_proc_volume == PETSC_TRUE) {
            MOFEM_LOG("ELASTIC", Sev::inform) << "Write output file ...";
            CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);
            std::ostringstream o1;
            o1 << "out_" << sit->step_number << ".h5m";
            if (!test_nb)
              CHKERR post_proc.writeFile(o1.str().c_str());
            MOFEM_LOG("ELASTIC", Sev::inform) << "done ...";
          }

          MOFEM_LOG("ELASTIC", Sev::inform) << "Write output file skin ...";
          CHKERR DMoFEMLoopFiniteElements(dm, "POST_PROC_SKIN",
                                          &post_proc_skin);
          std::ostringstream o1_skin;
          o1_skin << "out_skin_" << sit->step_number << ".h5m";
          if (!test_nb)
            CHKERR post_proc_skin.writeFile(o1_skin.str().c_str());
          MOFEM_LOG("ELASTIC", Sev::inform) << "done ...";
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

        MOFEM_LOG_C("ELASTIC", Sev::inform, "norm2 F = %6.4e", nrm_F);
        PetscReal nrm_F_thermal;
        CHKERR VecNorm(F_thermal, NORM_2, &nrm_F_thermal);

        MOFEM_LOG_C("ELASTIC", Sev::inform, "norm2 F_thermal = %6.4e",
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
        CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);

        // Save data on mesh
        if (is_post_proc_volume == PETSC_TRUE) {
          MOFEM_LOG("ELASTIC", Sev::inform) << "Write output file ...";
          CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);
          // Save results to file
          if (!test_nb)
            CHKERR post_proc.writeFile("out.h5m");
          MOFEM_LOG("ELASTIC", Sev::inform) << "done";
        }

        MOFEM_LOG("ELASTIC", Sev::inform) << "Write output file skin ...";
        CHKERR DMoFEMLoopFiniteElements(dm, "POST_PROC_SKIN", &post_proc_skin);
        if (!test_nb)
          CHKERR post_proc_skin.writeFile("out_skin.h5m");
        MOFEM_LOG("ELASTIC", Sev::inform) << "done";
      }

      // Destroy vector, no needed any more
      CHKERR VecDestroy(&F_thermal);
    } else {
      // Elastic analysis no temperature field
      // VecView(F, PETSC_VIEWER_STDOUT_WORLD);
      // Solve for vector D
      CHKERR KSPSolve(solver, F, D);

      // VecView(D, PETSC_VIEWER_STDOUT_WORLD);
      // cerr << F;

      // Add kinetic boundary conditions
      CHKERR VecAXPY(D, 1., D0);
      // Update ghost values
      CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
      CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
      // Save data from vector on mesh
      CHKERR DMoFEMMeshToLocalVector(dm, D, INSERT_VALUES, SCATTER_REVERSE);
      // Post-process results
      MOFEM_LOG("ELASTIC", Sev::inform) << "Post-process start ...";
      if (is_post_proc_volume == PETSC_TRUE) {
        CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &post_proc);
      }
      CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", &prism_post_proc);
      CHKERR DMoFEMLoopFiniteElements(dm, "SIMPLE_ROD", &post_proc_edge);
      CHKERR DMoFEMLoopFiniteElements(dm, "POST_PROC_SKIN", &post_proc_skin);
      MOFEM_LOG("ELASTIC", Sev::inform) << "done";

      // Write mesh in parallel (using h5m MOAB format, writing is in parallel)
      MOFEM_LOG("ELASTIC", Sev::inform) << "Write output file ...";
      if (mesh_has_tets) {
        if (is_post_proc_volume == PETSC_TRUE) {
          if (!test_nb)
            CHKERR post_proc.writeFile("out.h5m");
        }
        if (!test_nb)
          CHKERR post_proc_skin.writeFile("out_skin.h5m");
      }
      if (mesh_has_prisms) {
        if (!test_nb)
          CHKERR prism_post_proc.writeFile("prism_out.h5m");
      }
      if (!edges_in_simple_rod.empty())
        if (!test_nb)
          CHKERR post_proc_edge.writeFile("out_edge.h5m");
      MOFEM_LOG("ELASTIC", Sev::inform) << "done";
    }

    if (is_calculating_frequency == PETSC_TRUE) {
      // Calculate mode mass, m = u^T * M * u
      Vec u1;
      VecDuplicate(D, &u1);
      CHKERR MatMult(Mij, D, u1);
      double mode_mass;
      CHKERR VecDot(u1, D, &mode_mass);
      MOFEM_LOG_C("ELASTIC", Sev::inform, "Mode mass  %6.4e\n", mode_mass);

      Vec v1;
      VecDuplicate(D, &v1);
      CHKERR MatMult(Aij, D, v1);

      double mode_stiffness;
      CHKERR VecDot(v1, D, &mode_stiffness);
      MOFEM_LOG_C("ELASTIC", Sev::inform, "Mode stiffness  %6.4e\n",
                  mode_stiffness);

      double frequency;
      double pi = 3.14159265359;
      frequency = std::sqrt(mode_stiffness / mode_mass) / (2 * pi);
      MOFEM_LOG_C("ELASTIC", Sev::inform, "Frequency  %6.4e", frequency);
    }

    // Calculate elastic energy
    auto calculate_strain_energy = [&]() {
      MoFEMFunctionBegin;

      SmartPetscObj<Vec> v_energy;
      CHKERR HookeElement::calculateEnergy(dm, block_sets_ptr, "DISPLACEMENT",
                                           "MESH_NODE_POSITIONS", false, true,
                                           v_energy);

      // Print elastic energy
      double energy;
      CHKERR VecSum(v_energy, &energy);
      MOFEM_LOG_C("ELASTIC", Sev::inform, "Elastic energy %6.4e", energy);

      switch (test_nb) {
      case 1:
        if (fabs(energy - 17.129) > 1e-3)
          SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                  "atom test diverged!");
        break;
      case 2:
        if (fabs(energy - 5.6475e-03) > 1e-4)
          SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                  "atom test diverged!");
        break;
      case 3:
        if (fabs(energy - 7.4679e-03) > 1e-4)
          SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                  "atom test diverged!");
        break;
      case 4:
        if (fabs(energy - 2.4992e+00) > 1e-3)
          SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                  "atom test diverged!");
        break;
      // FIXME: Here are missing regersion tests
      case 8: {
        double min;
        CHKERR VecMin(D, PETSC_NULL, &min);
        constexpr double expected_val = 0.10001;
        if (fabs(min + expected_val) > 1e-10)
          SETERRQ2(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                   "atom test diverged! %3.4e != %3.4e", min, expected_val);
      } break;
      case 9: {
        if (fabs(energy - 4.7416e-04) > 1e-8)
          SETERRQ(PETSC_COMM_SELF, MOFEM_ATOM_TEST_INVALID,
                  "atom test diverged!");
      }
      default:
        break;
      }

      MoFEMFunctionReturn(0);
    };
    CHKERR calculate_strain_energy();

    MPI_Comm_free(&moab_comm_world);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
