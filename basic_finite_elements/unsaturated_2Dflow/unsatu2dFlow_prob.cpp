#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <Unsaturated2DFlowUDOP.hpp>

using namespace MoFEM;
using namespace UFOperators2D;

static char help[] = "...\n\n";
template <int dim>
struct UFProblem {
public: 
  UFProblem(moab::Core &mb_instance, MoFEM::Core &core, const int order, const double satu_cond, const int n_species)
      : moab( mb_instance)
      , m_field(core)
      , order(order)
      , K_s(satu_cond)
      , nb_species(n_species)
      , cOmm(m_field.get_comm())
      , rAnk(m_field.get_comm_rank()) {
    vol_ele_stiff_rhs = boost::shared_ptr<VolEle>(new VolEle(m_field));
    vol_ele_stiff_lhs = boost::shared_ptr<VolEle>(new VolEle(m_field));

    boundary_ele_rhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));


    post_proc = boost::shared_ptr<PostProc>(
        new PostProc(m_field));

    data.resize(nb_species);
    values_ptr.resize(nb_species);
    grads_ptr.resize(nb_species);
    dots_ptr.resize(nb_species);
    inner_surface.resize(nb_species);

    for(int i = 0; i < nb_species; ++i){
      data[i] = boost::shared_ptr<PreviousData>(new PreviousData());
      grads_ptr[i] = boost::shared_ptr<MatrixDouble>(data[i], &data[i]->grads);
      values_ptr[i] = boost::shared_ptr<VectorDouble>(data[i], &data[i]->values);
      dots_ptr[i] = boost::shared_ptr<VectorDouble>(data[i], &data[i]->dot_values);
    }
  
  }

  // UFProblem(const int order) : order(order){}
  MoFEMErrorCode  run_analysis();

private:
  MoFEMErrorCode setup_system();
  MoFEMErrorCode add_fe(std::string field_name);
  MoFEMErrorCode set_blockData(std::map<int, BlockData> &block_data_map);

  MoFEMErrorCode set_initial_values(std::string field_name, int block_id,
                                    Range &surface, double &init_val);



  MoFEMErrorCode push_mass_ele(std::string field_name);



  

  MoFEMErrorCode update_vol_fe(boost::shared_ptr<VolEle> &vol_ele,
                    boost::shared_ptr<PreviousData> &data);
  MoFEMErrorCode update_stiff_rhs(std::string field_name,
                                  boost::shared_ptr<VectorDouble> &values_ptr,
                                  boost::shared_ptr<MatrixDouble> &grads_ptr,
                                  boost::shared_ptr<VectorDouble> &dots_ptr);

  MoFEMErrorCode push_stiff_rhs(std::string field_name, 
                                boost::shared_ptr<PreviousData> &data,
                                std::map<int, BlockData> &block_map);
  
  MoFEMErrorCode update_stiff_lhs(std::string field_name,
                                  boost::shared_ptr<VectorDouble> &values_ptr,
                                  boost::shared_ptr<MatrixDouble> &grads_ptr);
  MoFEMErrorCode push_stiff_lhs(std::string field_name,
                                boost::shared_ptr<PreviousData> &data,
                                std::map<int, BlockData> &block_map);

  MoFEMErrorCode set_integration_rule();


  MoFEMErrorCode set_fe_in_loop();
  MoFEMErrorCode post_proc_fields(std::string field_name);
  MoFEMErrorCode output_result();
  MoFEMErrorCode solve();

  MoFEM::Interface &m_field;
  Simple *simple_interface;

  moab::Interface &moab;

  SmartPetscObj<DM> dm;
  SmartPetscObj<TS> ts;
 

  Range natural_bdry_ents;

  std::vector<Range> inner_surface;

  double global_error;

  MPI_Comm cOmm;
  const int rAnk;

  int order;
  double K_s;
  int nb_species;

  std::map<int, BlockData> material_blocks;

  boost::shared_ptr<VolEle> vol_ele_stiff_rhs;
  boost::shared_ptr<VolEle> vol_ele_stiff_lhs;
  boost::shared_ptr<FaceEle> boundary_ele_rhs;

  boost::shared_ptr<VolEle> vol_mass_ele;

  boost::shared_ptr<PostProc> post_proc;
  boost::shared_ptr<Monitor> monitor_ptr;


  std::vector<boost::shared_ptr<PreviousData>> data;

  std::vector<boost::shared_ptr<MatrixDouble>> grads_ptr;
  std::vector<boost::shared_ptr<VectorDouble>> values_ptr;
  std::vector<boost::shared_ptr<VectorDouble>> dots_ptr;

  boost::shared_ptr<ForcesAndSourcesCore> null;
};
template <int dim>
MoFEMErrorCode UFProblem<dim>::setup_system() {
  MoFEMFunctionBegin; 
  CHKERR m_field.getInterface(simple_interface);
  CHKERR simple_interface->getOptions();
  CHKERR simple_interface->loadFile();
  MoFEMFunctionReturn(0);
}

template <int dim>
MoFEMErrorCode UFProblem<dim>::add_fe(std::string field_name) {
  MoFEMFunctionBegin; 
  CHKERR simple_interface->addDomainField(field_name, H1, AINSWORTH_LEGENDRE_BASE, 1);
  
  CHKERR simple_interface->addBoundaryField(field_name, H1,
                                            AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple_interface->setFieldOrder(field_name, order);
  

  MoFEMFunctionReturn(0);
}

template <int dim>
MoFEMErrorCode UFProblem<dim>::set_blockData(std::map<int, BlockData> &block_map) {
  MoFEMFunctionBegin; 
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    string name = it->getName();
    const int id = it->getMeshsetId();
    if (name.compare(0, 14, "REGION1") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, dim, block_map[id].block_ents, true);
      // cout << "Name: " << name << endl;

      // block_map[id].block_ents.print();
      // cout << "size: ";
      // cout << block_map[id].block_ents.size() << endl;
     

      block_map[id].block_id = id;
      block_map[id].K_s = 1.000;
      block_map[id].h_s = 0.000;
      block_map[id].theta_s = 0.43000;
      block_map[id].theta_m = 0.43000;
      block_map[id].theta_r = 0.04500;
      block_map[id].alpha = 1.812500;
      block_map[id].nn = 5.3800;

    } else if (name.compare(0, 14, "REGION2") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, dim, block_map[id].block_ents, true);
    
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION3") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, dim, block_map[id].block_ents, true);
  
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION4") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, dim, block_map[id].block_ents, true);

      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION5") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, dim, block_map[id].block_ents, true);

      block_map[id].block_id = id;
    }
  }
  MoFEMFunctionReturn(0);
}

template <int dim>
MoFEMErrorCode UFProblem<dim>::set_initial_values(std::string field_name,
                                              int block_id, Range &surface, double &init_val) {
  MoFEMFunctionBegin;
  if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id,
                                                            BLOCKSET)) {
    CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        block_id, BLOCKSET, dim, surface, true);
  }
  if (!surface.empty()) {
    Range surface_verts;

    CHKERR moab.get_connectivity(surface, surface_verts, false);
    CHKERR m_field.getInterface<FieldBlas>()->setField(
        init_val, MBVERTEX, surface_verts, field_name);
  }

    MoFEMFunctionReturn(0);
}



template <int dim>
MoFEMErrorCode UFProblem<dim>::update_vol_fe(boost::shared_ptr<VolEle> &vol_ele,
                                         boost::shared_ptr<PreviousData> &data) {
  MoFEMFunctionBegin;
 
  vol_ele->getOpPtrVector().push_back(
      new OpCalculateInvJacForFace(data->invJac));
  vol_ele->getOpPtrVector().push_back(
      new OpSetInvJacH1ForFace(data->invJac));

  MoFEMFunctionReturn(0);
}

template <int dim>
MoFEMErrorCode UFProblem<dim>::update_stiff_rhs(std::string field_name,
                             boost::shared_ptr<VectorDouble> &values_ptr,
                            boost::shared_ptr<MatrixDouble> &grads_ptr,
                            boost::shared_ptr<VectorDouble> &dots_ptr) {

  MoFEMFunctionBegin;

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(field_name, values_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarValuesDot(field_name, dots_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldGradient<dim>(field_name, grads_ptr));
  MoFEMFunctionReturn(0);
}
template <int dim>
MoFEMErrorCode UFProblem<dim>::push_stiff_rhs(std::string field_name,
                                          boost::shared_ptr<PreviousData> &data,
                                         std::map<int, BlockData> &block_map) {
  MoFEMFunctionBegin;

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhs(field_name, data, K_s, block_map));
  // boundary_ele_rhs->getOpPtrVector().push_back(
  //     new OpAssembleNaturalBCRhs(field_name,natural_bdry_ents));
  MoFEMFunctionReturn(0);
}

template <int dim>
MoFEMErrorCode UFProblem<dim>::update_stiff_lhs(std::string field_name, 
                             boost::shared_ptr<VectorDouble> &values_ptr,
                            boost::shared_ptr<MatrixDouble> &grads_ptr) {
  MoFEMFunctionBegin;
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(field_name, values_ptr));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldGradient<dim>(field_name, grads_ptr));
  MoFEMFunctionReturn(0);
}
template <int dim>
MoFEMErrorCode UFProblem<dim>::push_stiff_lhs(std::string field_name,
                                          boost::shared_ptr<PreviousData> &data,
                                         std::map<int, BlockData> &block_map) {
  MoFEMFunctionBegin;

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleStiffLhs(field_name, data, K_s, block_map));

  MoFEMFunctionReturn(0);
}
template <int dim>
MoFEMErrorCode UFProblem<dim>::set_integration_rule() {
  MoFEMFunctionBegin; 
  auto vol_rule = [](int, int, int p) -> int { return 2 * p; };


  vol_ele_stiff_rhs->getRuleHook = vol_rule;

  vol_ele_stiff_lhs->getRuleHook = vol_rule;
  boundary_ele_rhs->getRuleHook = vol_rule;
  MoFEMFunctionReturn(0);
}


template <int dim>
MoFEMErrorCode UFProblem<dim>::set_fe_in_loop() {
  MoFEMFunctionBegin; 
  CHKERR TSSetType(ts, TSARKIMEX);
  CHKERR TSARKIMEXSetType(ts, TSARKIMEXA2);

  CHKERR DMMoFEMTSSetIJacobian(dm, simple_interface->getDomainFEName(),
                               vol_ele_stiff_lhs, null, null);

  CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getDomainFEName(),
                               vol_ele_stiff_rhs, null, null);

  CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getBoundaryFEName(),
                               boundary_ele_rhs, null, null);

  MoFEMFunctionReturn(0);
}
template <int dim>
MoFEMErrorCode UFProblem<dim>::post_proc_fields(std::string field_name) {
  MoFEMFunctionBegin; 
  post_proc->addFieldValuesPostProc(field_name);
  MoFEMFunctionReturn(0);
}

template <int dim>
MoFEMErrorCode UFProblem<dim>::output_result() {
  MoFEMFunctionBegin; 
  CHKERR DMMoFEMTSSetMonitor(dm, ts, simple_interface->getDomainFEName(),
                             monitor_ptr, null, null);
  MoFEMFunctionReturn(0);
}
template <int dim>
MoFEMErrorCode UFProblem<dim>::solve() {
  MoFEMFunctionBegin; 
  // Create solution vector
  SmartPetscObj<Vec> X;
  CHKERR DMCreateGlobalVector_MoFEM(dm, X);
  CHKERR DMoFEMMeshToLocalVector(dm, X, INSERT_VALUES, SCATTER_FORWARD);
  // Solve problem
  double ftime = 1;
  CHKERR TSSetDM(ts, dm);
  CHKERR TSSetDuration(ts, PETSC_DEFAULT, ftime);
  CHKERR TSSetSolution(ts, X);
  CHKERR TSSetFromOptions(ts);
  CHKERR TSSolve(ts, X);
  MoFEMFunctionReturn(0);
}
template <int dim>
MoFEMErrorCode UFProblem<dim>::run_analysis() {
  MoFEMFunctionBegin; 
  global_error = 0;
  std::vector<std::string> mass_names(nb_species);

  for(int i = 0; i < nb_species; ++i){
    mass_names[i] = "h" + boost::lexical_cast<std::string>(i+1);
  }
  CHKERR setup_system();
  for (int i = 0; i < nb_species; ++i) {
    add_fe(mass_names[i]);
  }

   

  CHKERR simple_interface->setUp();

  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    string name = it->getName();
    if (name.compare(0, 9, "ESSENTIAL") == 0) {
      CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), dim-1,
                                                  natural_bdry_ents, true);
    }
  }  
  // Range face_edges;

  // CHKERR moab.get_adjacencies(natural_bdry_ents, dim-1, false, face_edges, moab::Interface::UNION);

  Range edges_verts;
  CHKERR moab.get_connectivity(natural_bdry_ents, edges_verts, false);

  Range bdry_ents;
  bdry_ents = unite(natural_bdry_ents, edges_verts);
  // bdry_ents = unite(bdry_ents, face_edges_verts);
  
  // bdry_ents.print();
  // cout << "size: " <<endl;
  // cout << bdry_ents.size() <<endl;
  // cout << bdry_ents <<endl;

  CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
      "SimpleProblem", mass_names[0], bdry_ents);

  CHKERR set_blockData(material_blocks);

  cout << "Material Block Size: " << material_blocks.size() << endl;

  material_blocks[3].block_ents.print();
  cout << "size: ";
  cout << material_blocks[3].block_ents.size() << endl;

  VectorDouble initVals;
  initVals.resize(3, false);
  initVals.clear();

  initVals[0] = -0.8;
  initVals[1] = 0.0;
  initVals[2] = 0.0;

  for (int i = 0; i < nb_species; ++i) {
    CHKERR set_initial_values(mass_names[i], i + 2, inner_surface[i], initVals[i]);

    }

  if (!natural_bdry_ents.empty()) {
  Range surface_verts;
  
  CHKERR moab.get_connectivity(natural_bdry_ents, surface_verts, false);
  CHKERR m_field.getInterface<FieldBlas>()->setField(
      0.0, MBVERTEX, surface_verts, mass_names[0]);
  }



 



    dm = simple_interface->getDM();

      CHKERR update_vol_fe(vol_ele_stiff_rhs, data[0]);

      for (int i = 0; i < nb_species; ++i) {
        CHKERR update_stiff_rhs(mass_names[i], values_ptr[i], grads_ptr[i],
                                dots_ptr[i]);
        CHKERR push_stiff_rhs(mass_names[i], data[i], material_blocks);
    }
         

    CHKERR update_vol_fe(vol_ele_stiff_lhs, data[0]);

    for (int i = 0; i < nb_species; ++i) {
      CHKERR update_stiff_lhs(mass_names[i], values_ptr[i], grads_ptr[i]);
      CHKERR push_stiff_lhs(mass_names[i], data[i],
                            material_blocks); // nb_species times
    }
  
      
    CHKERR set_integration_rule();

        ts = createTS(m_field.get_comm());

        CHKERR set_fe_in_loop();

        post_proc->generateReferenceElementMesh(); // only once

        for (int i = 0; i < nb_species; ++i) {
          CHKERR post_proc_fields(mass_names[i]);
        }
      
      monitor_ptr = boost::shared_ptr<Monitor>(
          new Monitor(cOmm, rAnk, dm, post_proc, global_error)); // nb_species times
      CHKERR output_result();          // only once
      
      CHKERR solve();                  // only once

      MoFEMFunctionReturn(0);
}

int main(int argc, char *argv[]) {
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);
  try {
    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    MoFEM::Core core(moab);
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

  
    int order = 1;
    double satu_cond = 1.00;
    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
    
    CHKERR PetscOptionsGetReal(PETSC_NULL, "", "-saturated_conductivity",
                              &satu_cond, PETSC_NULL);
    int nb_species = 1;
    UFProblem<2> unsatu_flow_problem(mb_instance, core, order, satu_cond, nb_species);
    CHKERR unsatu_flow_problem.run_analysis();
  }
  CATCH_ERRORS;
  MoFEM::Core::Finalize();
  return 0;
  }