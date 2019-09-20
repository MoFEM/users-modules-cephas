/** \file stability.cpp
 * \ingroup nonlinear_elastic_elem
 *
 * Solves stability problem. Currently uses 3d tetrahedral elements.
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

#include <BasicFiniteElements.hpp>
using namespace MoFEM;
#include <Hooke.hpp>

#undef EPS
#include <slepceps.h>

#include <SurfacePressureComplexForLazy.hpp>

static char help[] = "...\n\n";

template <typename TYPE>
struct MyMat_double
    : public NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<
          TYPE> {

  bool doAotherwiseB;
  MyMat_double() : doAotherwiseB(true){};

  MatrixDouble D_lambda, D_mu, D;
  ublas::vector<TYPE, ublas::bounded_array<TYPE, 6>> sTrain, sTrain0, sTress;
  ublas::matrix<adouble, ublas::row_major, ublas::bounded_array<adouble, 9>>
      invF, CauchyStress;

  virtual MoFEMErrorCode calculateP_PiolaKirchhoffI(
      const NonlinearElasticElement::BlockData block_data,
      boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr) {
    MoFEMFunctionBegin;
    double lambda = LAMBDA(block_data.E, block_data.PoissonRatio);
    double mu = MU(block_data.E, block_data.PoissonRatio);
    if (D_lambda.size1() == 0) {
      D_lambda.resize(6, 6);
      D_lambda.clear();
      for (int rr = 0; rr < 3; rr++) {
        for (int cc = 0; cc < 3; cc++) {
          D_lambda(rr, cc) = 1;
        }
      }
    }
    if (D_mu.size1() == 0) {
      D_mu.resize(6, 6);
      D_mu.clear();
      for (int rr = 0; rr < 6; rr++) {
        D_mu(rr, rr) = rr < 3 ? 2 : 1;
      }
    }
    D.resize(6, 6);
    noalias(D) = lambda * D_lambda + mu * D_mu;

    if (doAotherwiseB) {
      sTrain.resize(6);
      sTrain[0] = this->F(0, 0) - 1;
      sTrain[1] = this->F(1, 1) - 1;
      sTrain[2] = this->F(2, 2) - 1;
      sTrain[3] = this->F(0, 1) + this->F(1, 0);
      sTrain[4] = this->F(1, 2) + this->F(2, 1);
      sTrain[5] = this->F(0, 2) + this->F(2, 0);
      sTress.resize(6);
      noalias(sTress) = prod(D, sTrain);
      this->P.resize(3, 3);
      this->P(0, 0) = sTress[0];
      this->P(1, 1) = sTress[1];
      this->P(2, 2) = sTress[2];
      this->P(0, 1) = this->P(1, 0) = sTress[3];
      this->P(1, 2) = this->P(2, 1) = sTress[4];
      this->P(0, 2) = this->P(2, 0) = sTress[5];
      // std::cerr << this->P << std::endl;
    } else {
      adouble J;
      CHKERR this->dEterminant(this->F, J);
      invF.resize(3, 3);
      CHKERR this->iNvert(J, this->F, invF);
      sTrain0.resize(6, 0);
      noalias(sTress) = prod(D, sTrain0);
      CauchyStress.resize(3, 3);
      CauchyStress(0, 0) = sTress[0];
      CauchyStress(1, 1) = sTress[1];
      CauchyStress(2, 2) = sTress[2];
      CauchyStress(0, 1) = CauchyStress(1, 0) = sTress[3];
      CauchyStress(1, 2) = CauchyStress(2, 1) = sTress[4];
      CauchyStress(0, 2) = CauchyStress(2, 0) = sTress[5];
      noalias(this->P) = J * prod(CauchyStress, trans(invF));
    }

    MoFEMFunctionReturn(0);
  }
};

template <typename TYPE> struct MyMat : public MyMat_double<TYPE> {

  int nbActiveVariables0;

  virtual MoFEMErrorCode setUserActiveVariables(int &nb_active_variables) {
    MoFEMFunctionBeginHot;

    try {

      this->sTrain0.resize(6);
      MatrixDouble &G0 = (this->commonDataPtr->gradAtGaussPts["D0"][this->gG]);
      this->sTrain0[0] <<= G0(0, 0);
      this->sTrain0[1] <<= G0(1, 1);
      this->sTrain0[2] <<= G0(2, 2);
      this->sTrain0[3] <<= (G0(1, 0) + G0(0, 1));
      this->sTrain0[4] <<= (G0(2, 1) + G0(1, 2));
      this->sTrain0[5] <<= (G0(2, 0) + G0(0, 2));
      nbActiveVariables0 = nb_active_variables;
      nb_active_variables += 6;

    } catch (const std::exception &ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF, 1, ss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }

  virtual MoFEMErrorCode setUserActiveVariables(VectorDouble &active_variables) {
    MoFEMFunctionBeginHot;

    try {

      int shift = nbActiveVariables0; // is a number of elements in F
      MatrixDouble &G0 = (this->commonDataPtr->gradAtGaussPts["D0"][this->gG]);
      active_variables[shift + 0] = G0(0, 0);
      active_variables[shift + 1] = G0(1, 1);
      active_variables[shift + 2] = G0(2, 2);
      active_variables[shift + 3] = G0(0, 1) + G0(1, 0);
      active_variables[shift + 4] = G0(1, 2) + G0(2, 1);
      active_variables[shift + 5] = G0(0, 2) + G0(2, 0);

    } catch (const std::exception &ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF, 1, ss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }
};

int main(int argc, char *argv[]) {

  // PetscInitialize(&argc,&argv,(char *)0,help);
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

  SlepcInitialize(&argc, &argv, param_file.c_str(), help);

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

    // use this if your mesh is partotioned and you run code on parts,
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
      CHKERR pcomm->resolve_shared_ents(0, 3, 0);
      CHKERR pcomm->resolve_shared_ents(0, 3, 1);
      CHKERR pcomm->resolve_shared_ents(0, 3, 2);
    } else {
      const char *option;
      option = "";
      CHKERR moab.load_file(mesh_file_name, 0, option);
    }

    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    Range CubitSIDESETs_meshsets;
    CHKERR m_field.getInterface<MeshsetsManager>()->getMeshsetsByType(
        SIDESET, CubitSIDESETs_meshsets);

    // ref meshset ref level 0
    BitRefLevel bit_level0;
    bit_level0.set(0);
    EntityHandle meshset_level0;
    CHKERR moab.create_meshset(MESHSET_SET, meshset_level0);
    CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(
        0, 3, bit_level0);
    CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
        bit_level0, BitRefLevel().set(), meshset_level0);

    // Fields
    CHKERR m_field.add_field("MESH_NODE_POSITIONS", H1, AINSWORTH_LEGENDRE_BASE,
                             3, MB_TAG_SPARSE, MF_ZERO);
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "MESH_NODE_POSITIONS");
    CHKERR m_field.set_field_order(0, MBTET, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBTRI, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBEDGE, "MESH_NODE_POSITIONS", 2);
    CHKERR m_field.set_field_order(0, MBVERTEX, "MESH_NODE_POSITIONS", 1);

    bool check_if_spatial_field_exist = m_field.check_field("SPATIAL_POSITION");
    CHKERR m_field.add_field("SPATIAL_POSITION", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);
    CHKERR m_field.add_field("EIGEN_VECTOR", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);
    CHKERR m_field.add_field("D0", H1, AINSWORTH_LEGENDRE_BASE, 3,
                             MB_TAG_SPARSE, MF_ZERO);

    // add entitities (by tets) to the field
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "SPATIAL_POSITION");
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "EIGEN_VECTOR");
    CHKERR m_field.add_ents_to_field_by_type(0, MBTET, "D0");

    boost::shared_ptr<Hooke<double>> mat_double =
        boost::make_shared<Hooke<double>>();
    boost::shared_ptr<MyMat<adouble>> mat_adouble =
        boost::make_shared<MyMat<adouble>>();

    NonlinearElasticElement elastic(m_field, 2);
    CHKERR elastic.setBlocks(mat_double, mat_adouble);
    CHKERR elastic.addElement("ELASTIC", "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_data("ELASTIC",
                                                        "EIGEN_VECTOR");
    CHKERR m_field.modify_finite_element_add_field_data("ELASTIC", "D0");

    elastic.feRhs.getOpPtrVector().push_back(
        new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
            "D0", elastic.commonData));
    elastic.feLhs.getOpPtrVector().push_back(
        new NonlinearElasticElement::OpGetCommonDataAtGaussPts(
            "D0", elastic.commonData));
    CHKERR elastic.setOperators("SPATIAL_POSITION");

    // define problems
    CHKERR m_field.add_problem("ELASTIC_MECHANICS", MF_ZERO);
    // set finite elements for problems
    CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                     "ELASTIC");
    // set refinement level for problem
    CHKERR m_field.modify_problem_ref_level_add_bit("ELASTIC_MECHANICS",
                                                    bit_level0);

    // set app. order

    PetscInt disp_order;
    CHKERR PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-my_order", &disp_order,
                              &flg);
    if (flg != PETSC_TRUE) {
      disp_order = 1;
    }

    CHKERR m_field.set_field_order(0, MBTET, "SPATIAL_POSITION", disp_order);
    CHKERR m_field.set_field_order(0, MBTRI, "SPATIAL_POSITION", disp_order);
    CHKERR m_field.set_field_order(0, MBEDGE, "SPATIAL_POSITION", disp_order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "SPATIAL_POSITION", 1);
    CHKERR m_field.set_field_order(0, MBTET, "EIGEN_VECTOR", disp_order);
    CHKERR m_field.set_field_order(0, MBTRI, "EIGEN_VECTOR", disp_order);
    CHKERR m_field.set_field_order(0, MBEDGE, "EIGEN_VECTOR", disp_order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "EIGEN_VECTOR", 1);
    CHKERR m_field.set_field_order(0, MBTET, "D0", disp_order);
    CHKERR m_field.set_field_order(0, MBTRI, "D0", disp_order);
    CHKERR m_field.set_field_order(0, MBEDGE, "D0", disp_order);
    CHKERR m_field.set_field_order(0, MBVERTEX, "D0", 1);

    CHKERR m_field.add_finite_element("NEUAMNN_FE", MF_ZERO);
    CHKERR m_field.modify_finite_element_add_field_row("NEUAMNN_FE",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_col("NEUAMNN_FE",
                                                       "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_data("NEUAMNN_FE",
                                                        "SPATIAL_POSITION");
    CHKERR m_field.modify_finite_element_add_field_data("NEUAMNN_FE",
                                                        "MESH_NODE_POSITIONS");
    CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                     "NEUAMNN_FE");
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      Range tris;
      CHKERR moab.get_entities_by_type(it->meshset, MBTRI, tris, true);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        "NEUAMNN_FE");
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      Range tris;
      CHKERR moab.get_entities_by_type(it->meshset, MBTRI, tris, true);
      CHKERR m_field.add_ents_to_finite_element_by_type(tris, MBTRI,
                                                        "NEUAMNN_FE");
    }
    // add nodal force element
    CHKERR MetaNodalForces::addElement(m_field, "SPATIAL_POSITION");
    CHKERR m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS",
                                                     "FORCE_FE");

    // build field
    CHKERR m_field.build_fields();
    // 10 node tets
    if (!check_if_spatial_field_exist) {
      Projection10NodeCoordsOnField ent_method_material(m_field,
                                                        "MESH_NODE_POSITIONS");
      CHKERR m_field.loop_dofs("MESH_NODE_POSITIONS", ent_method_material);
      Projection10NodeCoordsOnField ent_method_spatial(m_field,
                                                       "SPATIAL_POSITION");
      CHKERR m_field.loop_dofs("SPATIAL_POSITION", ent_method_spatial);
      // CHKERR m_field.set_field(0,MBTRI,"SPATIAL_POSITION");
      // CHKERR m_field.set_field(0,MBTET,"SPATIAL_POSITION");
      // CHKERR m_field.field_axpy(1,"SPATIAL_POSITION","D0",true);
    }

    // build finite elemnts
    CHKERR m_field.build_finite_elements();
    // build adjacencies
    CHKERR m_field.build_adjacencies(bit_level0);

    // build database
    ProblemsManager *prb_mng_ptr;
    CHKERR m_field.getInterface(prb_mng_ptr);
    if (is_partitioned) {
      CHKERR prb_mng_ptr->buildProblemOnDistributedMesh("ELASTIC_MECHANICS",
                                                        true);
      CHKERR prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS", true, 0,
                                                  pcomm->size(), 1);
    } else {
      CHKERR prb_mng_ptr->buildProblem("ELASTIC_MECHANICS", true);
      CHKERR prb_mng_ptr->partitionProblem("ELASTIC_MECHANICS");
      CHKERR prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS");
    }
    CHKERR prb_mng_ptr->partitionGhostDofs("ELASTIC_MECHANICS");

    // create matrices
    Vec F;
    CHKERR m_field.getInterface<VecManager>()->vecCreateGhost(
        "ELASTIC_MECHANICS", ROW, &F);
    Vec D;
    CHKERR VecDuplicate(F, &D);
    Mat Aij;
    CHKERR m_field.getInterface<MatrixManager>()
        ->createMPIAIJWithArrays<PetscGlobalIdx_mi_tag>("ELASTIC_MECHANICS",
                                                        &Aij);

    // surface forces
    NeummanForcesSurfaceComplexForLazy neumann_forces(m_field, Aij, F);
    NeummanForcesSurfaceComplexForLazy::MyTriangleSpatialFE &neumann =
        neumann_forces.getLoopSpatialFe();
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field, NODESET | FORCESET,
                                                    it)) {
      CHKERR neumann.addForce(it->getMeshsetId());
    }
    for (_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(
             m_field, SIDESET | PRESSURESET, it)) {
      CHKERR neumann.addPressure(it->getMeshsetId());
    }
    DirichletSpatialPositionsBc my_Dirichlet_bc(m_field, "SPATIAL_POSITION",
                                                Aij, D, F);

    CHKERR VecZeroEntries(F);
    CHKERR VecGhostUpdateBegin(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(F, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR MatZeroEntries(Aij);

    CHKERR m_field.getInterface<VecManager>()->setLocalGhostVector(
        "ELASTIC_MECHANICS", COL, D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    // F Vector
    // preproc
    my_Dirichlet_bc.snes_ctx = SnesMethod::CTX_SNESSETFUNCTION;
    my_Dirichlet_bc.snes_x = D;
    my_Dirichlet_bc.snes_f = F;
    CHKERR m_field.problem_basic_method_preProcess("ELASTIC_MECHANICS",
                                                   my_Dirichlet_bc);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR m_field.getInterface<VecManager>()->setLocalGhostVector(
        "ELASTIC_MECHANICS", COL, D, INSERT_VALUES, SCATTER_REVERSE);
    // elem loops
    // noadl forces
    boost::ptr_map<std::string, NodalForce> nodal_forces;
    string fe_name_str = "FORCE_FE";
    nodal_forces.insert(fe_name_str, new NodalForce(m_field));
    CHKERR MetaNodalForces::setOperators(m_field, nodal_forces, F,
                                         "SPATIAL_POSITION");
    boost::ptr_map<std::string, NodalForce>::iterator fit =
        nodal_forces.begin();
    for (; fit != nodal_forces.end(); fit++) {
      CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", fit->first,
                                          fit->second->getLoopFe());
    }
    // surface forces
    neumann.snes_ctx = SnesMethod::CTX_SNESSETFUNCTION;
    neumann.snes_x = D;
    neumann.snes_f = F;
    m_field.loop_finite_elements("ELASTIC_MECHANICS", "NEUAMNN_FE", neumann);
    // stiffnes
    elastic.getLoopFeRhs().snes_ctx = SnesMethod::CTX_SNESSETFUNCTION;
    elastic.getLoopFeRhs().snes_x = D;
    elastic.getLoopFeRhs().snes_f = F;
    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "ELASTIC",
                                        elastic.getLoopFeRhs());
    // postproc
    CHKERR m_field.problem_basic_method_postProcess("ELASTIC_MECHANICS",
                                                    my_Dirichlet_bc);

    // Aij Matrix
    // preproc
    my_Dirichlet_bc.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
    my_Dirichlet_bc.snes_B = Aij;
    CHKERR m_field.problem_basic_method_preProcess("ELASTIC_MECHANICS",
                                                   my_Dirichlet_bc);
    // surface forces
    // neumann.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
    // neumann.snes_B = Aij;
    // CHKERR
    // m_field.loop_finite_elements("ELASTIC_MECHANICS","NEUAMNN_FE",neumann);
    // stiffnes
    elastic.getLoopFeLhs().snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
    elastic.getLoopFeLhs().snes_B = Aij;
    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "ELASTIC",
                                        elastic.getLoopFeLhs());
    // postproc
    CHKERR m_field.problem_basic_method_postProcess("ELASTIC_MECHANICS",
                                                    my_Dirichlet_bc);

    CHKERR VecAssemblyBegin(F);
    CHKERR VecAssemblyEnd(F);
    CHKERR VecGhostUpdateBegin(F, ADD_VALUES, SCATTER_REVERSE);
    CHKERR VecGhostUpdateEnd(F, ADD_VALUES, SCATTER_REVERSE);

    CHKERR MatAssemblyBegin(Aij, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(Aij, MAT_FINAL_ASSEMBLY);

    // Matrix View
    // MatView(Aij,PETSC_VIEWER_STDOUT_WORLD);
    // MatView(Aij,PETSC_VIEWER_DRAW_WORLD);//PETSC_VIEWER_STDOUT_WORLD);
    // std::string wait;
    // std::cin >> wait;

    // Solver
    KSP solver;
    CHKERR KSPCreate(PETSC_COMM_WORLD, &solver);
    CHKERR KSPSetOperators(solver, Aij, Aij);
    CHKERR KSPSetFromOptions(solver);

    CHKERR KSPSetUp(solver);

    CHKERR VecZeroEntries(D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR KSPSolve(solver, F, D);
    CHKERR VecGhostUpdateBegin(D, INSERT_VALUES, SCATTER_FORWARD);
    CHKERR VecGhostUpdateEnd(D, INSERT_VALUES, SCATTER_FORWARD);

    CHKERR m_field.getInterface<VecManager>()->setOtherGlobalGhostVector(
        "ELASTIC_MECHANICS", "SPATIAL_POSITION", "D0", COL, D, INSERT_VALUES,
        SCATTER_REVERSE);

    Mat Bij;
    CHKERR MatDuplicate(Aij, MAT_SHARE_NONZERO_PATTERN, &Bij);
    // CHKERR MatZeroEntries(Aij);
    CHKERR MatZeroEntries(Bij);

    /*//Aij Matrix
    //preproc
    my_Dirichlet_bc.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
    my_Dirichlet_bc.snes_B = Aij;
    CHKERR
    m_field.problem_basic_method_preProcess("ELASTIC_MECHANICS",my_Dirichlet_bc);
    //stiffnes
    elastic.getLoopFeLhs().snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
    elastic.getLoopFeLhs().snes_B = Aij;
    CHKERR
    m_field.loop_finite_elements("ELASTIC_MECHANICS","ELASTIC",elastic.getLoopFeLhs());
    //postproc
    CHKERR
    m_field.problem_basic_method_postProcess("ELASTIC_MECHANICS",my_Dirichlet_bc);
  */

    // Bij Matrix
    mat_adouble->doAotherwiseB = false;
    // preproc
    my_Dirichlet_bc.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
    my_Dirichlet_bc.snes_B = Bij;
    CHKERR m_field.problem_basic_method_preProcess("ELASTIC_MECHANICS",
                                                   my_Dirichlet_bc);
    // surface forces
    neumann.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
    neumann.snes_B = Bij;
    PetscBool is_conservative = PETSC_FALSE;
    CHKERR PetscOptionsGetBool(PETSC_NULL, PETSC_NULL, "-my_is_conservative",
                               &is_conservative, &flg);
    if (is_conservative) {
      CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "NEUAMNN_FE",
                                          neumann);
    }
    // stiffnes
    elastic.getLoopFeLhs().snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
    elastic.getLoopFeLhs().snes_B = Bij;
    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "ELASTIC",
                                        elastic.getLoopFeLhs());
    // postproc
    CHKERR m_field.problem_basic_method_postProcess("ELASTIC_MECHANICS",
                                                    my_Dirichlet_bc);

    CHKERR MatSetOption(Bij, MAT_SPD, PETSC_TRUE);
    CHKERR MatAssemblyBegin(Bij, MAT_FINAL_ASSEMBLY);
    CHKERR MatAssemblyEnd(Bij, MAT_FINAL_ASSEMBLY);

    // Matrix View
    // MatView(Bij,PETSC_VIEWER_STDOUT_WORLD);
    // MatView(Bij,PETSC_VIEWER_DRAW_WORLD);//PETSC_VIEWER_STDOUT_WORLD);
    // std::string wait;
    // std::cin >> wait;

    EPS eps;
    ST st;
    EPSType type;
    PetscReal tol;
    PetscInt nev, maxit, its;

    /*
      Create eigensolver context
    */
    CHKERR EPSCreate(PETSC_COMM_WORLD, &eps);
    /*
      Set operators. In this case, it is a generalized eigenvalue problem
    */
    CHKERR EPSSetOperators(eps, Bij, Aij);
    /*
      Set solver parameters at runtime
    */
    CHKERR EPSSetFromOptions(eps);
    /*
      Optional: Get some information from the solver and display it
    */
    CHKERR EPSSolve(eps);

    /*
      Optional: Get some information from the solver and display it
    */
    CHKERR EPSGetIterationNumber(eps, &its);
    PetscPrintf(PETSC_COMM_WORLD, " Number of iterations of the method: %D\n",
                its);
    CHKERR EPSGetST(eps, &st);
    // CHKERR STGetOperationCounters(st,NULL,&lits);
    // PetscPrintf(PETSC_COMM_WORLD," Number of linear iterations of the method:
    // %D\n",lits);
    CHKERR EPSGetType(eps, &type);
    PetscPrintf(PETSC_COMM_WORLD, " Solution method: %s\n", type);
    CHKERR EPSGetDimensions(eps, &nev, NULL, NULL);
    PetscPrintf(PETSC_COMM_WORLD, " Number of requested eigenvalues: %D\n",
                nev);
    CHKERR EPSGetTolerances(eps, &tol, &maxit);
    PetscPrintf(PETSC_COMM_WORLD, " Stopping condition: tol=%.4g, maxit=%D\n",
                (double)tol, maxit);

    // get solutions
    PostProcVolumeOnRefinedMesh post_proc(m_field);
    CHKERR post_proc.generateReferenceElementMesh();
    CHKERR post_proc.addFieldValuesGradientPostProc("SPATIAL_POSITION");
    CHKERR post_proc.addFieldValuesPostProc("SPATIAL_POSITION");
    CHKERR post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS");
    CHKERR post_proc.addFieldValuesPostProc("EIGEN_VECTOR");
    CHKERR post_proc.addFieldValuesGradientPostProc("EIGEN_VECTOR");
    CHKERR post_proc.addFieldValuesPostProc("D0");
    CHKERR post_proc.addFieldValuesGradientPostProc("D0");
    CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "ELASTIC",
                                        post_proc);
    CHKERR post_proc.writeFile("out.h5m");

    PetscScalar eigr, eigi, nrm2r;
    for (int nn = 0; nn < nev; nn++) {
      CHKERR EPSGetEigenpair(eps, nn, &eigr, &eigi, D, PETSC_NULL);
      CHKERR VecNorm(D, NORM_2, &nrm2r);
      PetscPrintf(
          PETSC_COMM_WORLD,
          " ncov = %D eigr = %.4g eigi = %.4g (inv eigr = %.4g) nrm2r = %.4g\n",
          nn, eigr, eigi, 1. / eigr, nrm2r);
      std::ostringstream o1;
      o1 << "eig_" << nn << ".h5m";
      CHKERR m_field.getInterface<VecManager>()->setOtherGlobalGhostVector(
          "ELASTIC_MECHANICS", "SPATIAL_POSITION", "EIGEN_VECTOR", COL, D,
          INSERT_VALUES, SCATTER_REVERSE);
      CHKERR m_field.loop_finite_elements("ELASTIC_MECHANICS", "ELASTIC",
                                          post_proc);
      CHKERR post_proc.writeFile(o1.str().c_str());
    }

    CHKERR KSPDestroy(&solver);
    CHKERR VecDestroy(&F);
    CHKERR VecDestroy(&D);
    CHKERR MatDestroy(&Aij);
    CHKERR MatDestroy(&Bij);
    CHKERR EPSDestroy(&eps);
  }
  CATCH_ERRORS;

  SlepcFinalize();
  // PetscFinalize();

  return 0;
}
