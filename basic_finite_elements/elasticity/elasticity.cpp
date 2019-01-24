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
  int iD;
  double yOung;
  double pOisson;
  double initTemp;

  double springStiffness0; // Spring stiffness
  double springStiffness1;
  double springStiffness2;

  Range tEts;
  Range tRis;

  BlockOptionData()
      : oRder(-1), yOung(-1), pOisson(-2), springStiffness0(-1),
        springStiffness1(-1), springStiffness2(-1), initTemp(0) {}
};

struct DataAtIntegrationPtsSprings {

  boost::shared_ptr<MatrixDouble> gradDispPtr;
  boost::shared_ptr<VectorDouble> pPtr;
  FTensor::Ddg<double, 3, 3> tD;
  boost::shared_ptr<MatrixDouble> xAtPts =
      boost::shared_ptr<MatrixDouble>(new MatrixDouble());

  double springStiffness0; // Spring stiffness
  double springStiffness1;
  double springStiffness2;

  // std::map<int, BlockOptionData> mapElastic;
  std::map<int, BlockOptionData> mapSpring;

  DataAtIntegrationPtsSprings(MoFEM::Interface &m_field) : mField(m_field) {

    // Setting default values for coeffcients
    gradDispPtr = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    xAtPts = boost::shared_ptr<MatrixDouble>(new MatrixDouble());
    pPtr = boost::shared_ptr<VectorDouble>(new VectorDouble());

    ierr = setBlocks();
    CHKERRABORT(PETSC_COMM_WORLD, ierr);
  }

  MoFEMErrorCode getParameters() {
    MoFEMFunctionBegin; // They will be overwriten by BlockData
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Problem", "none");

    ierr = PetscOptionsEnd();
    CHKERRQ(ierr);
    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode getBlockData(BlockOptionData &data) {
    MoFEMFunctionBegin;

    springStiffness0 = data.springStiffness0;
    springStiffness1 = data.springStiffness1;
    springStiffness2 = data.springStiffness2;


    MoFEMFunctionReturn(0);
  }

  MoFEMErrorCode setBlocks() {
    MoFEMFunctionBegin;

    // for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
    //          mField, BLOCKSET | MAT_ELASTICSET, it)) {
    //   Mat_Elastic mydata;
    //   CHKERR it->getAttributeDataStructure(mydata);
    //   int id = it->getMeshsetId();
    //   EntityHandle meshset = it->getMeshset();
    //   CHKERR mField.get_moab().get_entities_by_type(meshset, MBTET,
    //                                                 mapElastic[id].tEts, true);
    //   mapElastic[id].iD = id;
    //   mapElastic[id].yOung = mydata.data.Young;
    //   mapElastic[id].pOisson = mydata.data.Poisson;
    // }

    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(mField, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {

        const int id = bit->getMeshsetId();
        CHKERR mField.get_moab().get_entities_by_type(bit->getMeshset(), MBTRI,
                                                      mapSpring[id].tRis, true);

        // EntityHandle out_meshset;
        // CHKERR mField.get_moab().create_meshset(MESHSET_SET, out_meshset);
        // CHKERR mField.get_moab().add_entities(out_meshset,
        // mapSpring[id].tRis); CHKERR mField.get_moab().write_file("error.vtk",
        // "VTK", "",
        //                                     &out_meshset, 1);

        std::vector<double> attributes;
        bit->getAttributes(attributes);
        if (attributes.size() != 3) {
          SETERRQ1(PETSC_COMM_WORLD, MOFEM_ATOM_TEST_INVALID,
                   "should be 3 attributes but is %d", attributes.size());
        }
        mapSpring[id].iD = id;
        mapSpring[id].springStiffness0 = attributes[0];
        mapSpring[id].springStiffness1 = attributes[1];
        mapSpring[id].springStiffness2 = attributes[2];
      }
    }

    MoFEMFunctionReturn(0);
  }

private:
  MoFEM::Interface &mField;
};

/** * @brief Assemble contribution of spring to RHS *
 * \f[
 * {K^s} = \int\limits_\Omega ^{} {{\psi ^T}{k_s}\psi d\Omega }
 * \f]
 *
 */
struct OpSpringKs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  DataAtIntegrationPtsSprings &commonData;
  MatrixDouble locKs;
  MatrixDouble transLocKs;
  BlockOptionData &dAta;

  OpSpringKs(DataAtIntegrationPtsSprings &common_data, BlockOptionData &data)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator(
            "DISPLACEMENT", "DISPLACEMENT", OPROWCOL),
        commonData(common_data), dAta(data) {
    sYmm = true;
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

    // std::cout << dAta.tRis << endl;
    if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tRis.end()) {
      MoFEMFunctionReturnHot(0);
    }

    CHKERR commonData.getBlockData(dAta);
    // size associated to the entity
    locKs.resize(row_nb_dofs, col_nb_dofs, false);
    locKs.clear();

    // get number of Gauss points
    const int row_nb_gauss_pts = row_data.getN().size1();
    if (!row_nb_gauss_pts) // check if number of Gauss point <> 0
      MoFEMFunctionReturnHot(0);

    const int row_nb_base_functions = row_data.getN().size2();
    auto row_base_functions = row_data.getFTensor0N();

    // vector<double> spring_stiffness; // spring_stiffness[0]
    // spring_stiffness.push_back(commonData.springStiffness0);
    // spring_stiffness.push_back(commonData.springStiffness1);
    // spring_stiffness.push_back(commonData.springStiffness2);

    FTensor::Tensor1<double, 3> spring_stiffness(commonData.springStiffness0,
                                                 commonData.springStiffness1,
                                                 commonData.springStiffness2);

    // FTensor::Index<'i', 3> i;

    // loop over all Gauss point of the volume
    for (int gg = 0; gg != row_nb_gauss_pts; gg++) {
      // get area and integration weight
      double w = getArea() * getGaussPts()(2, gg);

      for (int row_index = 0; row_index != row_nb_dofs / 3; row_index++) {
        auto col_base_functions = col_data.getFTensor0N(gg, 0);
        for (int col_index = 0; col_index != col_nb_dofs / 3; col_index++) {
          locKs(row_index, col_index) += w * row_base_functions *
                                         spring_stiffness(col_index % 3) *
                                         col_base_functions;
          ++col_base_functions;
        }
        ++row_base_functions;
      }
    }

    // Add computed values of spring stiffness to the global LHS matrix
    CHKERR MatSetValues(
        getFEMethod()->ksp_B, row_nb_dofs, &*row_data.getIndices().begin(),
        col_nb_dofs, &*col_data.getIndices().begin(), &locKs(0, 0), ADD_VALUES);

    // is symmetric
    if (row_side != col_side || row_type != col_type) {
      transLocKs.resize(col_nb_dofs, row_nb_dofs, false);
      noalias(transLocKs) = trans(locKs);
      CHKERR MatSetValues(getFEMethod()->ksp_B, col_nb_dofs,
                          &*col_data.getIndices().begin(), row_nb_dofs,
                          &*row_data.getIndices().begin(), &transLocKs(0, 0),
                          ADD_VALUES);
    }

    MoFEMFunctionReturn(0);
  }
};

/** * @brief Assemble contribution of springs to LHS *
 * \f[
 * f_s =  \int\limits_{\partial \Omega }^{} {{\psi ^T}{F^s}\left( u
 * \right)d\partial \Omega }  = \int\limits_{\partial \Omega }^{} {{\psi
 * ^T}{k_s}ud\partial \Omega }
 * \f]
 *
 */
struct OpSpringFs : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator {

  DataAtIntegrationPtsSprings &commonData;
  BlockOptionData &dAta;

  OpSpringFs(DataAtIntegrationPtsSprings &common_data, BlockOptionData &data)
      : MoFEM::FaceElementForcesAndSourcesCore::UserDataOperator("DISPLACEMENT",
                                                                 OPROW),
        commonData(common_data), dAta(data) {}

  // vector used to store force vector for each degree of freedom
  VectorDouble nF;

  FTensor::Index<'i', 3> i;

  MoFEMErrorCode doWork(int side, EntityType type,
                        DataForcesAndSourcesCore::EntData &data) {

    MoFEMFunctionBegin;
    // check that the faces have associated degrees of freedom
    const int nb_dofs = data.getIndices().size();
    if (nb_dofs == 0)
      MoFEMFunctionReturnHot(0);

    // std::cout << dAta.tRis << endl;
    if (dAta.tRis.find(getNumeredEntFiniteElementPtr()->getEnt()) ==
        dAta.tRis.end()) {
      MoFEMFunctionReturnHot(0);
    }

    // size of force vector associated to the entity
    // set equal to the number of degrees of freedom of associated with the
    // entity
    nF.resize(nb_dofs, false);
    nF.clear();

    // get number of gauss points
    const int nb_gauss_pts = data.getN().size1();

    // get intergration weights
    auto t_w = getFTensor0IntegrationWeight();

    // vector of base functions
    auto base_functions = data.getFTensor0N();

    FTensor::Tensor1<double, 3> spring_stiffness(commonData.springStiffness0,
                                                 commonData.springStiffness1,
                                                 commonData.springStiffness2);

    auto disp_at_gauss_point = getFTensor1FromMat<3>(*commonData.xAtPts);

    // loop over all gauss points of the face
    for (int gg = 0; gg != nb_gauss_pts; ++gg) {
      // weight of gg gauss point
      double w = 0.5 * t_w;

      // create a vector t_nf whose pointer points an array of 3 pointers
      // pointing to nF  memory location of components
      FTensor::Tensor1<FTensor::PackPtr<double *, 3>, 3> t_nf(&nF[0], &nF[1],
                                                              &nF[2]);
      for (int col_index = 0; col_index != nb_dofs / 3;
           ++col_index) { // loop over the nodes
        for (int ii = 0; ii != 3; ++ii) {
          t_nf(ii) += (w * spring_stiffness(ii) * base_functions) *
                      disp_at_gauss_point(ii);
        }
        // move to next base function
        ++base_functions;
        // move the pointer to next element of t_nf
        ++t_nf;
      }
      // move to next integration weight
      ++t_w;
      //
      ++disp_at_gauss_point;
    }
    // add computed values of pressure in the global right hand side vector
    CHKERR VecSetValues(getFEMethod()->ksp_f, nb_dofs, &data.getIndices()[0],
                        &nF[0], ADD_VALUES);
    MoFEMFunctionReturn(0);
  }
};

using BlockData = NonlinearElasticElement::BlockData;
using VolUserDataOperator = VolumeElementForcesAndSourcesCore::UserDataOperator;

/// Set integration rule
struct VolRule {
  int operator()(int, int, int order) const { return 2 * (order - 1); }
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

    CHKERR PetscOptionsInt("-my_order", "default approximation order", "",
                           order, &order, PETSC_NULL);

    CHKERR PetscOptionsBool("-my_is_partitioned",
                            "set if mesh is partitioned (this result that each "
                            "process keeps only one part of the mesh)",
                            "", is_partitioned, &is_partitioned, PETSC_NULL);

    CHKERR PetscOptionsString("-my_block_config", "elastic configure file name",
                              "", "block_conf.in", block_config_file, 255,
                              &flg_block_config);

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

    // Set bit refinement level to all entities (we do not refine mesh in this
    // example so all entities have the same bit refinement level)
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

    // Add entities (by tets) to the field (all entities in the mesh, root_set
    // = 0 )
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "DISPLACEMENT");
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");

    // Set approximation order.
    // See Hierarchical Finite Element Bases on Unstructured Tetrahedral Meshes.
    CHKERR m_field.set_field_order(0, MBTET, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBTRI, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBEDGE, "DISPLACEMENT", order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "DISPLACEMENT", 1);

    // Set order of approximation of geometry.
    // Apply 2nd order only on skin (or in whole body)
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
              PetscPrintf(PETSC_COMM_WORLD, "Set block %d order to %d\n",
                          it->getMeshsetId(),
                          block_data[it->getMeshsetId()].oRder);
              Range block_ents;
              CHKERR moab.get_entities_by_handle(it->getMeshset(), block_ents,
                                                 true);
              Range ents_to_set_order;
              CHKERR moab.get_adjacencies(block_ents, 3, false,
                                          ents_to_set_order,
                                          moab::Interface::UNION);
              ents_to_set_order = ents_to_set_order.subset_by_type(MBTET);
              CHKERR moab.get_adjacencies(block_ents, 2, false,
                                          ents_to_set_order,
                                          moab::Interface::UNION);
              CHKERR moab.get_adjacencies(block_ents, 1, false,
                                          ents_to_set_order,
                                          moab::Interface::UNION);
              CHKERR m_field.synchronise_entities(ents_to_set_order);

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
              CHKERR PetscPrintf(PETSC_COMM_WORLD,
                                 "** WARNING Unrecognized option %s\n",
                                 vit->c_str());
            }
          }

          // Update material parameters. Set material parameters block by block.
          for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
                   m_field, BLOCKSET | MAT_ELASTICSET, it)) {
            const int id = it->getMeshsetId();
            auto &bd = (*block_sets_ptr)[id];
            if (block_data[id].yOung > 0)
              bd.E = block_data[id].yOung;
            if (block_data[id].pOisson >= -1)
              bd.PoissonRatio = block_data[id].pOisson;
            CHKERR PetscPrintf(PETSC_COMM_WORLD, "Block %d\n", id);
            CHKERR PetscPrintf(PETSC_COMM_WORLD, "\tYoung modulus %3.4g\n",
                               bd.E);
            CHKERR PetscPrintf(PETSC_COMM_WORLD, "\tPoisson ratio %3.4g\n",
                               bd.PoissonRatio);
          }

          MoFEMFunctionReturn(0);
        };

    // Add elastic element

    boost::shared_ptr<std::map<int, BlockData>> block_sets_ptr =
        boost::make_shared<std::map<int, BlockData>>();
    CHKERR HookeElement::setBlocks(m_field, block_sets_ptr);
    CHKERR setting_blocks_data_and_order_from_config_file(block_sets_ptr);

    boost::shared_ptr<ForcesAndSourcesCore> fe_lhs_ptr(
        new VolumeElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<ForcesAndSourcesCore> fe_rhs_ptr(
        new VolumeElementForcesAndSourcesCore(m_field));
    fe_lhs_ptr->getRuleHook = VolRule();
    fe_rhs_ptr->getRuleHook = VolRule();

    // Create new instances of face elements for springs
    boost::shared_ptr<FaceElementForcesAndSourcesCore> feSpringLhs(
        new FaceElementForcesAndSourcesCore(m_field));
    boost::shared_ptr<FaceElementForcesAndSourcesCore> feSpringRhs(
        new FaceElementForcesAndSourcesCore(m_field));

    // Push operators to instances for springs
    // loop over blocks
    DataAtIntegrationPtsSprings commonData(m_field);
    CHKERR commonData.getParameters();

    for (auto &sitSpring : commonData.mapSpring) {
      feSpringLhs->getOpPtrVector().push_back(
          new OpSpringKs(commonData, sitSpring.second));

      feSpringRhs->getOpPtrVector().push_back(
          new OpCalculateVectorFieldValues<3>("DISPLACEMENT",
                                              commonData.xAtPts));
      feSpringRhs->getOpPtrVector().push_back(
          new OpSpringFs(commonData, sitSpring.second));
    }

    CHKERR HookeElement::addElasticElement(m_field, block_sets_ptr, "ELASTIC",
                                           "DISPLACEMENT",
                                           "MESH_NODE_POSITIONS", false);
    CHKERR HookeElement::setOperators(fe_lhs_ptr, fe_rhs_ptr, block_sets_ptr,
                                      "DISPLACEMENT", "MESH_NODE_POSITIONS",
                                      false, true);

    // Add spring element, just depends on "DISPLACEMENT"
    CHKERR m_field.add_finite_element("SPRING");
    CHKERR m_field.modify_finite_element_add_field_row("SPRING",
                                                       "DISPLACEMENT");
    CHKERR m_field.modify_finite_element_add_field_col("SPRING",
                                                       "DISPLACEMENT");
    CHKERR m_field.modify_finite_element_add_field_data("SPRING",
                                                        "DISPLACEMENT");
    // Add entities to spring elements
    for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, bit)) {
      if (bit->getName().compare(0, 9, "SPRING_BC") == 0) {
        CHKERR m_field.add_ents_to_finite_element_by_type(bit->getMeshset(),
                                                          MBTRI, "SPRING");
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
    CHKERR DMMoFEMAddElement(dm, "SPRING");
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

    // Assign global matrix/vector contributed by springs
    feSpringLhs->ksp_B = Aij;
    feSpringRhs->ksp_f = F;

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
    // Cubit sets kinetic boundary conditions by blockset.
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
    fe_rhs_ptr->snes_f = F;
    PetscPrintf(PETSC_COMM_WORLD, "Assemble external force vector  ...");
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", fe_rhs_ptr);
    PetscPrintf(PETSC_COMM_WORLD, " done\n");
    // Assemble matrix
    fe_lhs_ptr->snes_B = Aij;
    PetscPrintf(PETSC_COMM_WORLD, "Calculate stiffness matrix  ...");
    CHKERR DMoFEMLoopFiniteElements(dm, "ELASTIC", fe_lhs_ptr);
    PetscPrintf(PETSC_COMM_WORLD, " done\n");

    // Assemble springs
    CHKERR DMoFEMLoopFiniteElements(dm, "SPRING", feSpringLhs);
    CHKERR DMoFEMLoopFiniteElements(dm, "SPRING", feSpringRhs);

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
        post_proc.commonData, block_sets_ptr.get()));

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
      // CHKERR DMoFEMLoopFiniteElements(dm, "SPRING", &post_proc);
      // Write mesh in parallel (using h5m MOAB format, writing is in parallel)
      PetscPrintf(PETSC_COMM_WORLD, "Write output file ..,");
      CHKERR post_proc.writeFile("out.h5m");
      PetscPrintf(PETSC_COMM_WORLD, " done\n");
    }

    // Calculate elastic energy
    auto calculate_strain_energy = [dm, &block_sets_ptr]() {
      MoFEMFunctionBegin;

      Vec v_energy;
      CHKERR HookeElement::calculateEnergy(dm, block_sets_ptr, "DISPLACEMENT",
                                           "MESH_NODE_POSITIONS", false, true,
                                           &v_energy);

      // Print elastic energy
      const double *eng_ptr;
      CHKERR VecGetArrayRead(v_energy, &eng_ptr);
      PetscPrintf(PETSC_COMM_WORLD, "Elastic energy %6.4e\n", *eng_ptr);
      CHKERR VecRestoreArrayRead(v_energy, &eng_ptr);

      CHKERR VecDestroy(&v_energy);

      MoFEMFunctionReturn(0);
    };
    CHKERR calculate_strain_energy();

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
