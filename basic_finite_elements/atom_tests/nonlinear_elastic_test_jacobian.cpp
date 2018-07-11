#include <BasicFiniteElements.hpp>
using namespace MoFEM;

#include <Hooke.hpp>
#include <NeoHookean.hpp>

namespace bio = boost::iostreams;
using bio::tee_device;
using bio::stream;

static char help[] = "...\n\n";

struct OpCheck: public MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator {
  NonlinearElasticElement::CommonData &commonData;
  OpCheck(
    NonlinearElasticElement::CommonData &common_data
  ):
  MoFEM::VolumeElementForcesAndSourcesCore::UserDataOperator("SPATIAL_POSITION",UserDataOperator::OPROW),
  commonData(common_data) {
  }

  FTensor::Index<'i',3> i;
  FTensor::Index<'j',3> j;

  MoFEMErrorCode doWork(
    int side,EntityType type,DataForcesAndSourcesCore::EntData &data
  ) {
    MoFEMFunctionBeginHot;
    try {
      const int nb_dofs = data.getFieldData().size();
      //const int nb_base_functions = data.getN().size2();
      if(nb_dofs == 0) {
        MoFEMFunctionReturnHot(0);
      }
      const double eps = 1e-12;
      const int nb_gauss_pts = data.getN().size1();
      for(int gg = 0;gg!=nb_gauss_pts;gg++) {
        MatrixDouble3by3 &str0 = commonData.sTress[gg];
        FTensor::Tensor2<double,3,3> t_stress_0(
          str0(0,0),str0(0,1),str0(0,2),
          str0(1,0),str0(1,1),str0(1,2),
          str0(2,0),str0(2,1),str0(2,2)
        );
        VectorDouble &str1 = commonData.jacEnergy[gg];
        FTensor::Tensor2<double,3,3> t_stress_1(
          str1[0],str1[1],str1[2],
          str1[3],str1[4],str1[5],
          str1[6],str1[7],str1[8]
        );
        t_stress_0(i,j) -= t_stress_1(i,j);
        double nrm2 = t_stress_0(i,j)*t_stress_0(i,j);
        // PetscPrintf(PETSC_COMM_WORLD,"nrm2 = %3.2e\n",nrm2); CHKERRG(ierr);
        if(nrm2>eps) {
          SETERRQ1(
            PETSC_COMM_SELF,
            MOFEM_ATOM_TEST_INVALID,
            "direvative of enregy and stress inconsistency nrm2 = %6.4e",nrm2
          );
        }
      }
    } catch (MoFEMException const &e) {
      SETERRQ(PETSC_COMM_SELF,e.errorCode,e.errorMessage);
    }
    MoFEMFunctionReturnHot(0);
  }
};

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc,&argv,(char *)0,help);

  try {

    enum materials {
      HOOKE,
      NEOHOOKEAN,
      LASTOP
    };

    const char *materials_list[] = {
      "HOOKE",
      "NEOHOOKEAN"
    };

    PetscBool flg_test_mat;
    PetscInt choise_value = HOOKE;
    ierr = PetscOptionsGetEList(
      PETSC_NULL,NULL,"-mat",materials_list,LASTOP,&choise_value,&flg_test_mat
    ); CHKERRG(ierr);

    moab::Core mb_instance;
    moab::Interface& moab = mb_instance;
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    PetscBool flg = PETSC_TRUE;
    char mesh_file_name[255];
    ierr = PetscOptionsGetString(PETSC_NULL,PETSC_NULL,"-my_file",mesh_file_name,255,&flg); CHKERRG(ierr);

    const char *option;
    option = "";//"PARALLEL=BCAST;";//;DEBUG_IO";
    rval = moab.load_file(mesh_file_name, 0, option); CHKERRG(rval);
    ParallelComm* pcomm = ParallelComm::get_pcomm(&moab,MYPCOMM_INDEX);
    if(pcomm == NULL) pcomm =  new ParallelComm(&moab,PETSC_COMM_WORLD);

    MoFEM::Core core(moab);
    MoFEM::Interface& m_field = core;

    //ref meshset ref level 0
    BitRefLevel bit_level0;
    bit_level0.set(0);
    ierr = m_field.getInterface<BitRefManager>()->setBitRefLevelByDim(0,3, bit_level0); CHKERRG(ierr);

    //Fields
    ierr = m_field.add_field("SPATIAL_POSITION",H1,AINSWORTH_LEGENDRE_BASE,3); CHKERRG(ierr);
    //add entitities (by tets) to the field
    ierr = m_field.add_ents_to_field_by_type(0,MBTET,"SPATIAL_POSITION"); CHKERRG(ierr);
    //set app. order
    PetscInt order;
    ierr = PetscOptionsGetInt(PETSC_NULL,PETSC_NULL,"-my_order",&order,&flg); CHKERRG(ierr);
    if(flg != PETSC_TRUE) {
      order = 1;
    }
    ierr = m_field.set_field_order(0,MBTET,"SPATIAL_POSITION",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBTRI,"SPATIAL_POSITION",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBEDGE,"SPATIAL_POSITION",order); CHKERRG(ierr);
    ierr = m_field.set_field_order(0,MBVERTEX,"SPATIAL_POSITION",1); CHKERRG(ierr);

    //build field
    ierr = m_field.build_fields(); CHKERRG(ierr);

    //use this to apply some strain field to the body (testing only)
    double scale_positions = 2;
    {
      EntityHandle node = 0;
      double coords[3];
      for(_IT_GET_DOFS_FIELD_BY_NAME_FOR_LOOP_(m_field,"SPATIAL_POSITION",dof_ptr)) {
        if(dof_ptr->get()->getEntType()!=MBVERTEX) continue;
        EntityHandle ent = dof_ptr->get()->getEnt();
        int dof_rank = dof_ptr->get()->getDofCoeffIdx();
        double &fval = dof_ptr->get()->getFieldData();
        if(node!=ent) {
          rval = moab.get_coords(&ent,1,coords); CHKERRG(rval);
          node = ent;
        }
        fval = scale_positions*coords[dof_rank];
      }
    }

    NonlinearElasticElement elastic(m_field,1);

    ierr = elastic.setBlocks(
      boost::make_shared<NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<double> >(),
      boost::make_shared<NonlinearElasticElement::FunctionsToCalculatePiolaKirchhoffI<adouble> >()
    ); CHKERRG(ierr);
    elastic.commonData.spatialPositions = "SPATIAL_POSITION";
    elastic.commonData.meshPositions = "MESH_NODE_POSITIONS";

    ierr = elastic.addElement("ELASTIC","SPATIAL_POSITION"); CHKERRG(ierr);

    //build finite elemnts
    ierr = m_field.build_finite_elements(); CHKERRG(ierr);
    //build adjacencies
    ierr = m_field.build_adjacencies(bit_level0); CHKERRG(ierr);

    //define problems
    ierr = m_field.add_problem("ELASTIC_MECHANICS"); CHKERRG(ierr);
    //set refinement level for problem
    ierr = m_field.modify_problem_ref_level_add_bit("ELASTIC_MECHANICS",bit_level0); CHKERRG(ierr);
    //set finite elements for problems
    ierr = m_field.modify_problem_add_finite_element("ELASTIC_MECHANICS","ELASTIC"); CHKERRG(ierr);

    ProblemsManager *prb_mng_ptr;
    ierr = m_field.getInterface(prb_mng_ptr); CHKERRG(ierr);
    //build problem
    ierr = prb_mng_ptr->buildProblem("ELASTIC_MECHANICS",true); CHKERRG(ierr);
    //partition
    ierr = prb_mng_ptr->partitionProblem("ELASTIC_MECHANICS"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionFiniteElements("ELASTIC_MECHANICS"); CHKERRG(ierr);
    ierr = prb_mng_ptr->partitionGhostDofs("ELASTIC_MECHANICS"); CHKERRG(ierr);

    // test nonlinear element
    {
      elastic.getLoopFeRhs().getOpPtrVector().push_back(
        new NonlinearElasticElement::OpGetCommonDataAtGaussPts("SPATIAL_POSITION",elastic.commonData)
      );
      const int tag0 = 1;
      const int tag1 = 2;
      std::map<int,NonlinearElasticElement::BlockData>::iterator sit = elastic.setOfBlocks.begin();
      for(;sit!=elastic.setOfBlocks.end();sit++) {
        elastic.getLoopFeRhs().getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianPiolaKirchhoffStress(
            "SPATIAL_POSITION",sit->second,elastic.commonData,tag0,false,false,false
          )
        );
        elastic.getLoopFeRhs().getOpPtrVector().push_back(
          new NonlinearElasticElement::OpJacobianEnergy(
            "SPATIAL_POSITION",sit->second,elastic.commonData,tag1,true,false,false,false
          )
        );
      }
      // Run opperator to check consistency
      elastic.getLoopFeRhs().getOpPtrVector().push_back(new OpCheck(elastic.commonData));

      ierr = m_field.loop_finite_elements(
        "ELASTIC_MECHANICS","ELASTIC",elastic.getLoopFeRhs()
      ); CHKERRG(ierr);
    }

    // test materials
    if(flg_test_mat==PETSC_TRUE) {
      PetscPrintf(PETSC_COMM_WORLD,"Testing %s\n",materials_list[choise_value]);
      std::map<int,NonlinearElasticElement::BlockData>::iterator sit = elastic.setOfBlocks.begin();
      for(;sit!=elastic.setOfBlocks.end();sit++) {
        switch(choise_value) {
          case HOOKE:
          sit->second.materialDoublePtr = boost::make_shared<Hooke<double> >();
          sit->second.materialAdoublePtr = boost::make_shared<Hooke<adouble> >();
          break;
          case NEOHOOKEAN:
          sit->second.materialDoublePtr = boost::make_shared<NeoHookean<double> >();
          sit->second.materialAdoublePtr = boost::make_shared<NeoHookean<adouble> >();
        }
      }

    }

  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
