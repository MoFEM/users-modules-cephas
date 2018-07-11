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

template<typename TYPE>
struct MyMat_double: public NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<TYPE> {

  bool doAotherwiseB;
  MyMat_double(): doAotherwiseB(true) {};

  MatrixDouble D_lambda,D_mu,D;
  ublas::vector<TYPE,ublas::bounded_array<TYPE,6> > sTrain,sTrain0,sTress;
  ublas::matrix<adouble,ublas::row_major,ublas::bounded_array<adouble,9> > invF,CauchyStress;

  virtual MoFEMErrorCode calculateP_PiolaKirchhoffI(
    const NonlinearElasticElement::BlockData block_data,
    boost::shared_ptr<const NumeredEntFiniteElement> fe_ptr
  ) {
    MoFEMFunctionBeginHot;

    try {

      double lambda = LAMBDA(block_data.E,block_data.PoissonRatio);
      double mu = MU(block_data.E,block_data.PoissonRatio);
      if(D_lambda.size1()==0) {
        D_lambda.resize(6,6);
        D_lambda.clear();
        for(int rr = 0;rr<3;rr++) {
          for(int cc = 0;cc<3;cc++) {
            D_lambda(rr,cc) = 1;
          }
        }
      }
      if(D_mu.size1()==0) {
        D_mu.resize(6,6);
        D_mu.clear();
        for(int rr = 0;rr<6;rr++) {
          D_mu(rr,rr) = rr<3 ? 2 : 1;
        }
      }
      D.resize(6,6);
      noalias(D) = lambda*D_lambda + mu*D_mu;

      if(doAotherwiseB) {
        sTrain.resize(6);
        sTrain[0] = this->F(0,0)-1;
        sTrain[1] = this->F(1,1)-1;
        sTrain[2] = this->F(2,2)-1;
        sTrain[3] = this->F(0,1)+this->F(1,0);
        sTrain[4] = this->F(1,2)+this->F(2,1);
        sTrain[5] = this->F(0,2)+this->F(2,0);
        sTress.resize(6);
        noalias(sTress) = prod(D,sTrain);
        this->P.resize(3,3);
        this->P(0,0) = sTress[0];
        this->P(1,1) = sTress[1];
        this->P(2,2) = sTress[2];
        this->P(0,1) = this->P(1,0) = sTress[3];
        this->P(1,2) = this->P(2,1) = sTress[4];
        this->P(0,2) = this->P(2,0) = sTress[5];
        //std::cerr << this->P << std::endl;
      } else {
        adouble J;
        ierr = this->dEterminant(this->F,J); CHKERRG(ierr);
        invF.resize(3,3);
        ierr = this->iNvert(J,this->F,invF); CHKERRG(ierr);
        sTrain0.resize(6,0);
        noalias(sTress) = prod(D,sTrain0);
        CauchyStress.resize(3,3);
        CauchyStress(0,0) = sTress[0];
        CauchyStress(1,1) = sTress[1];
        CauchyStress(2,2) = sTress[2];
        CauchyStress(0,1) = CauchyStress(1,0) = sTress[3];
        CauchyStress(1,2) = CauchyStress(2,1) = sTress[4];
        CauchyStress(0,2) = CauchyStress(2,0) = sTress[5];
        //std::cerr << D << std::endl;
        //std::cerr << CauchyStress << std::endl;
        noalias(this->P) = J*prod(CauchyStress,trans(invF));
      }

    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }

};

template<typename TYPE>
struct MyMat: public MyMat_double<TYPE> {

  int nbActiveVariables0;

  virtual MoFEMErrorCode setUserActiveVariables(
    int &nb_active_variables) {
    MoFEMFunctionBeginHot;

    try {

      this->sTrain0.resize(6);
      MatrixDouble &G0 = (this->commonDataPtr->gradAtGaussPts["D0"][this->gG]);
      this->sTrain0[0] <<= G0(0,0);
      this->sTrain0[1] <<= G0(1,1);
      this->sTrain0[2] <<= G0(2,2);
      this->sTrain0[3] <<= (G0(1,0) + G0(0,1));
      this->sTrain0[4] <<= (G0(2,1) + G0(1,2));
      this->sTrain0[5] <<= (G0(2,0) + G0(0,2));
      nbActiveVariables0 = nb_active_variables;
      nb_active_variables += 6;

    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }

  virtual MoFEMErrorCode setUserActiveVariables(
    VectorDouble &active_varibles) {
    MoFEMFunctionBeginHot;

    try {

      int shift = nbActiveVariables0; // is a number of elements in F
      MatrixDouble &G0 = (this->commonDataPtr->gradAtGaussPts["D0"][this->gG]);
      active_varibles[shift+0] = G0(0,0);
      active_varibles[shift+1] = G0(1,1);
      active_varibles[shift+2] = G0(2,2);
      active_varibles[shift+3] = G0(0,1)+G0(1,0);
      active_varibles[shift+4] = G0(1,2)+G0(2,1);
      active_varibles[shift+5] = G0(0,2)+G0(2,0);

    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "throw in method: " << ex.what() << std::endl;
      SETERRQ(PETSC_COMM_SELF,1,ss.str().c_str());
    }

    MoFEMFunctionReturnHot(0);
  }


};

int main(int argc, char *argv[]) {

  //PetscInitialize(&argc,&argv,(char *)0,help);
  SlepcInitialize(&argc,&argv,(char*)0,help);

  try {

  moab::Core mb_instance;
  moab::Interface& moab = mb_instance;
  ParallelComm* pcomm = ParallelComm::get_pcomm(&moab,MYPCOMM_INDEX);
  if(pcomm == NULL) pcomm =  new ParallelComm(&moab,PETSC_COMM_WORLD);


  PetscBool flg = PETSC_TRUE;
  char mesh_file_name[255];
  ierr = PetscOptionsGetString(PETSC_NULL,PETSC_NULL,"-my_file",mesh_file_name,255,&flg); CHKERRG(ierr);
  if(flg != PETSC_TRUE) {
    SETERRQ(PETSC_COMM_SELF,1,"*** ERROR -my_file (MESH FILE NEEDED)");
  }

  // use this if your mesh is partotioned and you run code on parts,
  // you can solve very big problems
  PetscBool is_partitioned = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,PETSC_NULL,"-my_is_partitioned",&is_partitioned,&flg); CHKERRG(ierr);

  if(is_partitioned == PETSC_TRUE) {
    //Read mesh to MOAB
    const char *option;
    option = "PARALLEL=BCAST_DELETE;PARALLEL_RESOLVE_SHARED_ENTS;PARTITION=PARALLEL_PARTITION;";
    rval = moab.load_file(mesh_file_name, 0, option); CHKERRG(rval);
    rval = pcomm->resolve_shared_ents(0,3,0); CHKERRG(rval);
    rval = pcomm->resolve_shared_ents(0,3,1); CHKERRG(rval);
    rval = pcomm->resolve_shared_ents(0,3,2); CHKERRG(rval);
  } else {
    const char *option;
    option = "";
    rval = moab.load_file(mesh_file_name, 0, option); CHKERRG(rval);
  }

  MoFEM::Core core(moab);
  MoFEM::Interface& m_field = core;

  Range CubitSIDESETs_meshsets;
  CHKERR m_field.getInterface<MeshsetsManager>()->getMeshsetsByType(
      SIDESET, CubitSIDESETs_meshsets);

  //ref meshset ref level 0
  BitRefLevel bit_level0;
  bit_level0.set(0);
  EntityHandle meshset_level0;
  CHKERR moab.create_meshset(MESHSET_SET,meshset_level0); CHKERRG(rval);
  CHKERR m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(0, 3,
                                                                    bit_level0);
  CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByRefLevel(
      bit_level0, BitRefLevel().set(), meshset_level0);
  CHKERRG(ierr);

  //Fields
  ierr = m_field.add_field("MESH_NODE_POSITIONS",H1,AINSWORTH_LEGENDRE_BASE,3,MB_TAG_SPARSE,MF_ZERO); CHKERRG(ierr);
  ierr = m_field.add_ents_to_field_by_type(0,MBTET,"MESH_NODE_POSITIONS"); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBTET,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBTRI,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBEDGE,"MESH_NODE_POSITIONS",2); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBVERTEX,"MESH_NODE_POSITIONS",1); CHKERRG(ierr);

  bool check_if_spatial_field_exist = m_field.check_field("SPATIAL_POSITION");
  ierr = m_field.add_field("SPATIAL_POSITION",H1,AINSWORTH_LEGENDRE_BASE,3,MB_TAG_SPARSE,MF_ZERO); CHKERRG(ierr);
  ierr = m_field.add_field("EIGEN_VECTOR",H1,AINSWORTH_LEGENDRE_BASE,3,MB_TAG_SPARSE,MF_ZERO); CHKERRG(ierr);
  ierr = m_field.add_field("D0",H1,AINSWORTH_LEGENDRE_BASE,3,MB_TAG_SPARSE,MF_ZERO); CHKERRG(ierr);

  //add entitities (by tets) to the field
  ierr = m_field.add_ents_to_field_by_type(0,MBTET,"SPATIAL_POSITION"); CHKERRG(ierr);
  ierr = m_field.add_ents_to_field_by_type(0,MBTET,"EIGEN_VECTOR"); CHKERRG(ierr);
  ierr = m_field.add_ents_to_field_by_type(0,MBTET,"D0"); CHKERRG(ierr);

  boost::shared_ptr<Hooke<double> > mat_double = boost::make_shared<Hooke<double> >();
  boost::shared_ptr<MyMat<adouble> > mat_adouble = boost::make_shared<MyMat<adouble> >();

  NonlinearElasticElement elastic(m_field,2);
  ierr = elastic.setBlocks(mat_double,mat_adouble); CHKERRG(ierr);
  ierr = elastic.addElement("ELASTIC","SPATIAL_POSITION"); CHKERRG(ierr);
  ierr = m_field.modify_finite_element_add_field_data("ELASTIC","EIGEN_VECTOR"); CHKERRG(ierr);
  ierr = m_field.modify_finite_element_add_field_data("ELASTIC","D0"); CHKERRG(ierr);

  elastic.feRhs.getOpPtrVector().push_back(
    new NonlinearElasticElement::OpGetCommonDataAtGaussPts("D0",elastic.commonData));
  elastic.feLhs.getOpPtrVector().push_back(
    new NonlinearElasticElement::OpGetCommonDataAtGaussPts("D0",elastic.commonData));
  ierr = elastic.setOperators("SPATIAL_POSITION"); CHKERRG(ierr);

  //define problems
  ierr = m_field.add_problem("ELASTIC_MECHANICS",MF_ZERO); CHKERRG(ierr);
  //set finite elements for problems
  ierr = m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS","ELASTIC"); CHKERRG(ierr);
  //set refinement level for problem
  ierr = m_field.modify_problem_ref_level_add_bit("ELASTIC_MECHANICS",bit_level0); CHKERRG(ierr);

  //set app. order

  PetscInt disp_order;
  ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-my_order",&disp_order,&flg); CHKERRG(ierr);
  if(flg!=PETSC_TRUE) {
    disp_order = 1;
  }

  ierr = m_field.set_field_order(0,MBTET,"SPATIAL_POSITION",disp_order); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBTRI,"SPATIAL_POSITION",disp_order); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBEDGE,"SPATIAL_POSITION",disp_order); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBVERTEX,"SPATIAL_POSITION",1); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBTET,"EIGEN_VECTOR",disp_order); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBTRI,"EIGEN_VECTOR",disp_order); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBEDGE,"EIGEN_VECTOR",disp_order); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBVERTEX,"EIGEN_VECTOR",1); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBTET,"D0",disp_order); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBTRI,"D0",disp_order); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBEDGE,"D0",disp_order); CHKERRG(ierr);
  ierr = m_field.set_field_order(0,MBVERTEX,"D0",1); CHKERRG(ierr);

  ierr = m_field.add_finite_element("NEUAMNN_FE",MF_ZERO); CHKERRG(ierr);
  ierr = m_field.modify_finite_element_add_field_row("NEUAMNN_FE","SPATIAL_POSITION"); CHKERRG(ierr);
  ierr = m_field.modify_finite_element_add_field_col("NEUAMNN_FE","SPATIAL_POSITION"); CHKERRG(ierr);
  ierr = m_field.modify_finite_element_add_field_data("NEUAMNN_FE","SPATIAL_POSITION"); CHKERRG(ierr);
  ierr = m_field.modify_finite_element_add_field_data("NEUAMNN_FE","MESH_NODE_POSITIONS"); CHKERRG(ierr);
  ierr = m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS","NEUAMNN_FE"); CHKERRG(ierr);
  for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,NODESET|FORCESET,it)) {
    Range tris;
    rval = moab.get_entities_by_type(it->meshset,MBTRI,tris,true); CHKERRG(rval);
    ierr = m_field.add_ents_to_finite_element_by_type(tris,MBTRI,"NEUAMNN_FE"); CHKERRG(ierr);
  }
  for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,SIDESET|PRESSURESET,it)) {
    Range tris;
    rval = moab.get_entities_by_type(it->meshset,MBTRI,tris,true); CHKERRG(rval);
    ierr = m_field.add_ents_to_finite_element_by_type(tris,MBTRI,"NEUAMNN_FE"); CHKERRG(ierr);
  }
  //add nodal force element
  ierr = MetaNodalForces::addElement(m_field,"SPATIAL_POSITION"); CHKERRG(ierr);
  ierr = m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS","FORCE_FE"); CHKERRG(ierr);

  //build field
  ierr = m_field.build_fields(); CHKERRG(ierr);
  //10 node tets
  if(!check_if_spatial_field_exist) {
    Projection10NodeCoordsOnField ent_method_material(m_field,"MESH_NODE_POSITIONS");
    ierr = m_field.loop_dofs("MESH_NODE_POSITIONS",ent_method_material); CHKERRG(ierr);
    Projection10NodeCoordsOnField ent_method_spatial(m_field,"SPATIAL_POSITION");
    ierr = m_field.loop_dofs("SPATIAL_POSITION",ent_method_spatial); CHKERRG(ierr);
    //ierr = m_field.set_field(0,MBTRI,"SPATIAL_POSITION"); CHKERRG(ierr);
    //ierr = m_field.set_field(0,MBTET,"SPATIAL_POSITION"); CHKERRG(ierr);
    //ierr = m_field.field_axpy(1,"SPATIAL_POSITION","D0",true); CHKERRG(ierr);
  }

  //build finite elemnts
  ierr = m_field.build_finite_elements(); CHKERRG(ierr);
  //build adjacencies
  ierr = m_field.build_adjacencies(bit_level0); CHKERRG(ierr);

  //build database
  ProblemsManager *prb_mng_ptr;
  ierr = m_field.getInterface(prb_mng_ptr); CHKERRG(ierr);
  if(is_partitioned) {
    ierr = prb_mng_ptr->buildProblemOnDistributedMesh("ELASTIC_MECHANICS",true); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS",true,0,pcomm->size(),1); CHKERRG(ierr);
  } else {
    ierr = prb_mng_ptr->buildProblem("ELASTIC_MECHANICS",true); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionProblem("ELASTIC_MECHANICS"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS"); CHKERRG(ierr);
  }
  ierr = prb_mng_ptr->partitionGhostDofs("ELASTIC_MECHANICS"); CHKERRG(ierr);

  //create matrices
  Vec F;
  ierr = m_field.getInterface<VecManager>()->vecCreateGhost("ELASTIC_MECHANICS",ROW,&F); CHKERRG(ierr);
  Vec D;
  ierr = VecDuplicate(F,&D); CHKERRG(ierr);
  Mat Aij;
  ierr = m_field.MatCreateMPIAIJWithArrays("ELASTIC_MECHANICS",&Aij); CHKERRG(ierr);

  //surface forces
  NeummanForcesSurfaceComplexForLazy neumann_forces(m_field,Aij,F);
  NeummanForcesSurfaceComplexForLazy::MyTriangleSpatialFE &neumann = neumann_forces.getLoopSpatialFe();
  for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,NODESET|FORCESET,it)) {
    ierr = neumann.addForce(it->getMeshsetId()); CHKERRG(ierr);
  }
  for(_IT_CUBITMESHSETS_BY_BCDATA_TYPE_FOR_LOOP_(m_field,SIDESET|PRESSURESET,it)) {
    ierr = neumann.addPressure(it->getMeshsetId()); CHKERRG(ierr);
  }
  DirichletSpatialPositionsBc my_Dirichlet_bc(m_field,"SPATIAL_POSITION",Aij,D,F);

  ierr = VecZeroEntries(F); CHKERRG(ierr);
  ierr = VecGhostUpdateBegin(F,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
  ierr = VecGhostUpdateEnd(F,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
  ierr = MatZeroEntries(Aij); CHKERRG(ierr);

  ierr = m_field.getInterface<VecManager>()->setLocalGhostVector("ELASTIC_MECHANICS",COL,D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
  ierr = VecGhostUpdateBegin(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
  ierr = VecGhostUpdateEnd(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

  //F Vector
  //preproc
  my_Dirichlet_bc.snes_ctx = SnesMethod::CTX_SNESSETFUNCTION;
  my_Dirichlet_bc.snes_x = D;
  my_Dirichlet_bc.snes_f = F;
  ierr = m_field.problem_basic_method_preProcess("ELASTIC_MECHANICS",my_Dirichlet_bc); CHKERRG(ierr);
  ierr = VecGhostUpdateBegin(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
  ierr = VecGhostUpdateEnd(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
  ierr = m_field.getInterface<VecManager>()->setLocalGhostVector("ELASTIC_MECHANICS",COL,D,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
  //elem loops
  //noadl forces
  boost::ptr_map<std::string,NodalForce> nodal_forces;
  string fe_name_str ="FORCE_FE";
  nodal_forces.insert(fe_name_str,new NodalForce(m_field));
  ierr = MetaNodalForces::setOperators(m_field,nodal_forces,F,"SPATIAL_POSITION"); CHKERRG(ierr);
  boost::ptr_map<std::string,NodalForce>::iterator fit = nodal_forces.begin();
  for(;fit!=nodal_forces.end();fit++) {
    ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS",fit->first,fit->second->getLoopFe()); CHKERRG(ierr);
  }
  //surface forces
  neumann.snes_ctx = SnesMethod::CTX_SNESSETFUNCTION;
  neumann.snes_x = D;
  neumann.snes_f = F;
  m_field.loop_finite_elements("ELASTIC_MECHANICS","NEUAMNN_FE",neumann);
  //stiffnes
  elastic.getLoopFeRhs().snes_ctx = SnesMethod::CTX_SNESSETFUNCTION;
  elastic.getLoopFeRhs().snes_x = D;
  elastic.getLoopFeRhs().snes_f = F;
  ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","ELASTIC",elastic.getLoopFeRhs()); CHKERRG(ierr);
  //postproc
  ierr = m_field.problem_basic_method_postProcess("ELASTIC_MECHANICS",my_Dirichlet_bc); CHKERRG(ierr);

  //Aij Matrix
  //preproc
  my_Dirichlet_bc.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
  my_Dirichlet_bc.snes_B = Aij;
  ierr = m_field.problem_basic_method_preProcess("ELASTIC_MECHANICS",my_Dirichlet_bc); CHKERRG(ierr);
  //surface forces
  //neumann.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
  //neumann.snes_B = Aij;
  //ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","NEUAMNN_FE",neumann); CHKERRG(ierr);
  //stiffnes
  elastic.getLoopFeLhs().snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
  elastic.getLoopFeLhs().snes_B = Aij;
  ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","ELASTIC",elastic.getLoopFeLhs()); CHKERRG(ierr);
  //postproc
  ierr = m_field.problem_basic_method_postProcess("ELASTIC_MECHANICS",my_Dirichlet_bc); CHKERRG(ierr);

  ierr = VecAssemblyBegin(F); CHKERRG(ierr);
  ierr = VecAssemblyEnd(F); CHKERRG(ierr);
  ierr = VecGhostUpdateBegin(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
  ierr = VecGhostUpdateEnd(F,ADD_VALUES,SCATTER_REVERSE); CHKERRG(ierr);

  ierr = MatAssemblyBegin(Aij,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);
  ierr = MatAssemblyEnd(Aij,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);

  //Matrix View
  //MatView(Aij,PETSC_VIEWER_STDOUT_WORLD);
  //MatView(Aij,PETSC_VIEWER_DRAW_WORLD);//PETSC_VIEWER_STDOUT_WORLD);
  //std::string wait;
  //std::cin >> wait;

  //Solver
  KSP solver;
  ierr = KSPCreate(PETSC_COMM_WORLD,&solver); CHKERRG(ierr);
  ierr = KSPSetOperators(solver,Aij,Aij); CHKERRG(ierr);
  ierr = KSPSetFromOptions(solver); CHKERRG(ierr);

  ierr = KSPSetUp(solver); CHKERRG(ierr);

  ierr = VecZeroEntries(D); CHKERRG(ierr);
  ierr = VecGhostUpdateBegin(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
  ierr = VecGhostUpdateEnd(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

  ierr = KSPSolve(solver,F,D); CHKERRG(ierr);
  ierr = VecGhostUpdateBegin(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);
  ierr = VecGhostUpdateEnd(D,INSERT_VALUES,SCATTER_FORWARD); CHKERRG(ierr);

  ierr = m_field.getInterface<VecManager>()->setOtherGlobalGhostVector(
    "ELASTIC_MECHANICS","SPATIAL_POSITION","D0",COL,D,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);

  Mat Bij;
  ierr = MatDuplicate(Aij,MAT_SHARE_NONZERO_PATTERN,&Bij); CHKERRG(ierr);
  //ierr = MatZeroEntries(Aij); CHKERRG(ierr);
  ierr = MatZeroEntries(Bij); CHKERRG(ierr);

  /*//Aij Matrix
  //preproc
  my_Dirichlet_bc.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
  my_Dirichlet_bc.snes_B = Aij;
  ierr = m_field.problem_basic_method_preProcess("ELASTIC_MECHANICS",my_Dirichlet_bc); CHKERRG(ierr);
  //stiffnes
  elastic.getLoopFeLhs().snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
  elastic.getLoopFeLhs().snes_B = Aij;
  ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","ELASTIC",elastic.getLoopFeLhs()); CHKERRG(ierr);
  //postproc
  ierr = m_field.problem_basic_method_postProcess("ELASTIC_MECHANICS",my_Dirichlet_bc); CHKERRG(ierr);*/

  //Bij Matrix
  mat_adouble->doAotherwiseB = false;
  //preproc
  my_Dirichlet_bc.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
  my_Dirichlet_bc.snes_B = Bij;
  ierr = m_field.problem_basic_method_preProcess("ELASTIC_MECHANICS",my_Dirichlet_bc); CHKERRG(ierr);
  //surface forces
  neumann.snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
  neumann.snes_B = Bij;
  PetscBool is_conservative = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,PETSC_NULL,"-my_is_conservative",&is_conservative,&flg); CHKERRG(ierr);
  if(is_conservative) {
    ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","NEUAMNN_FE",neumann); CHKERRG(ierr);
  }
  //stiffnes
  elastic.getLoopFeLhs().snes_ctx = SnesMethod::CTX_SNESSETJACOBIAN;
  elastic.getLoopFeLhs().snes_B = Bij;
  ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","ELASTIC",elastic.getLoopFeLhs()); CHKERRG(ierr);
  //postproc
  ierr = m_field.problem_basic_method_postProcess("ELASTIC_MECHANICS",my_Dirichlet_bc); CHKERRG(ierr);

  ierr = MatSetOption(Bij,MAT_SPD,PETSC_TRUE); CHKERRG(ierr);
  ierr = MatAssemblyBegin(Bij,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);
  ierr = MatAssemblyEnd(Bij,MAT_FINAL_ASSEMBLY); CHKERRG(ierr);

  //Matrix View
  //MatView(Bij,PETSC_VIEWER_STDOUT_WORLD);
  //MatView(Bij,PETSC_VIEWER_DRAW_WORLD);//PETSC_VIEWER_STDOUT_WORLD);
  //std::string wait;
  //std::cin >> wait;

  EPS eps;
  ST st;
  EPSType type;
  PetscReal tol;
  PetscInt nev,maxit,its;

  /*
    Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps); CHKERRG(ierr);
  /*
    Set operators. In this case, it is a generalized eigenvalue problem
  */
  ierr = EPSSetOperators(eps,Bij,Aij); CHKERRG(ierr);
  /*
    Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps); CHKERRG(ierr);
  /*
    Optional: Get some information from the solver and display it
  */
  ierr = EPSSolve(eps); CHKERRG(ierr);

  /*
    Optional: Get some information from the solver and display it
  */
  ierr = EPSGetIterationNumber(eps,&its); CHKERRG(ierr);
  PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);
  ierr = EPSGetST(eps,&st); CHKERRG(ierr);
  //ierr = STGetOperationCounters(st,NULL,&lits); CHKERRG(ierr);
  //PetscPrintf(PETSC_COMM_WORLD," Number of linear iterations of the method: %D\n",lits);
  ierr = EPSGetType(eps,&type); CHKERRG(ierr);
  PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type);
  ierr = EPSGetDimensions(eps,&nev,NULL,NULL); CHKERRG(ierr);
  PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);
  ierr = EPSGetTolerances(eps,&tol,&maxit); CHKERRG(ierr);
  PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);

  //get solutions
  PostProcVolumeOnRefinedMesh post_proc(m_field);
  ierr = post_proc.generateReferenceElementMesh(); CHKERRG(ierr);
  ierr = post_proc.addFieldValuesGradientPostProc("SPATIAL_POSITION"); CHKERRG(ierr);
  ierr = post_proc.addFieldValuesPostProc("SPATIAL_POSITION"); CHKERRG(ierr);
  ierr = post_proc.addFieldValuesPostProc("MESH_NODE_POSITIONS"); CHKERRG(ierr);
  ierr = post_proc.addFieldValuesPostProc("EIGEN_VECTOR"); CHKERRG(ierr);
  ierr = post_proc.addFieldValuesGradientPostProc("EIGEN_VECTOR"); CHKERRG(ierr);
  ierr = post_proc.addFieldValuesPostProc("D0"); CHKERRG(ierr);
  ierr = post_proc.addFieldValuesGradientPostProc("D0"); CHKERRG(ierr);
  ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","ELASTIC",post_proc); CHKERRG(ierr);
  ierr = post_proc.writeFile("out.h5m"); CHKERRG(ierr);

  PetscScalar eigr,eigi,nrm2r;
  for(int nn = 0;nn<nev;nn++) {
    ierr = EPSGetEigenpair(eps,nn,&eigr,&eigi,D,PETSC_NULL); CHKERRG(ierr);
    ierr = VecNorm(D,NORM_2,&nrm2r); CHKERRG(ierr);
    PetscPrintf(PETSC_COMM_WORLD," ncov = %D eigr = %.4g eigi = %.4g (inv eigr = %.4g) nrm2r = %.4g\n",nn,eigr,eigi,1./eigr,nrm2r);
    std::ostringstream o1;
    o1 << "eig_" << nn << ".h5m";
    ierr = m_field.getInterface<VecManager>()->setOtherGlobalGhostVector(
      "ELASTIC_MECHANICS","SPATIAL_POSITION","EIGEN_VECTOR",COL,D,INSERT_VALUES,SCATTER_REVERSE); CHKERRG(ierr);
    ierr = m_field.loop_finite_elements("ELASTIC_MECHANICS","ELASTIC",post_proc); CHKERRG(ierr);
    ierr = post_proc.writeFile(o1.str().c_str()); CHKERRG(ierr);
  }

  ierr = KSPDestroy(&solver); CHKERRG(ierr);
  ierr = VecDestroy(&F); CHKERRG(ierr);
  ierr = VecDestroy(&D); CHKERRG(ierr);
  ierr = MatDestroy(&Aij); CHKERRG(ierr);
  ierr = MatDestroy(&Bij); CHKERRG(ierr);
  ierr = EPSDestroy(&eps); CHKERRG(ierr);

  } CATCH_ERRORS;

  SlepcFinalize();
  //PetscFinalize();

  return 0;
}
