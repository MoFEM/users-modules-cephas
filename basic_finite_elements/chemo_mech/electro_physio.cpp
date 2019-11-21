#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <EP_Operators.hpp>

using namespace MoFEM;
using namespace ElecPhys;

static char help[] = "...\n\n";

const double init_u = 0;
const double init_v = 0;
const double init_w = 1;
const double init_s = 0.02155;

struct ElectroPhysioProblem {
public:
  ElectroPhysioProblem(MoFEM::Core &core, const int order)
      : m_field(core), order(order)
      , cOmm(m_field.get_comm())
      , rAnk(m_field.get_comm_rank()) 
      {
          vol_ele_slow_rhs = boost::shared_ptr<VolEle>(new VolEle(m_field));
          natural_bdry_ele_slow_rhs =
              boost::shared_ptr<FaceEle>(new FaceEle(m_field));
          vol_ele_stiff_rhs = boost::shared_ptr<VolEle>(new VolEle(m_field));
          vol_ele_stiff_lhs = boost::shared_ptr<VolEle>(new VolEle(m_field));
          post_proc = boost::shared_ptr<PostProcVolumeOnRefinedMesh>(
              new PostProcVolumeOnRefinedMesh(m_field));

          data_u = boost::shared_ptr<PreviousData>(new PreviousData());
          data_v = boost::shared_ptr<PreviousData>(new PreviousData());
          data_w = boost::shared_ptr<PreviousData>(new PreviousData());
          data_s = boost::shared_ptr<PreviousData>(new PreviousData());

          flux_u_ptr =
              boost::shared_ptr<MatrixDouble>(data_u, &data_u->flux_values);
          divs_u_ptr =
              boost::shared_ptr<VectorDouble>(data_u, &data_u->flux_divs);
          values_u_ptr =
              boost::shared_ptr<VectorDouble>(data_u, &data_u->mass_values);
          dots_u_ptr =
              boost::shared_ptr<VectorDouble>(data_u, &data_u->mass_dots);

          flux_v_ptr =
              boost::shared_ptr<MatrixDouble>(data_v, &data_v->flux_values);
          divs_v_ptr =
              boost::shared_ptr<VectorDouble>(data_v, &data_v->flux_divs);
          values_v_ptr =
              boost::shared_ptr<VectorDouble>(data_v, &data_v->mass_values);
          dots_v_ptr =
              boost::shared_ptr<VectorDouble>(data_v, &data_v->mass_dots);


          flux_w_ptr =
              boost::shared_ptr<MatrixDouble>(data_w, &data_w->flux_values);
          divs_w_ptr =
              boost::shared_ptr<VectorDouble>(data_w, &data_w->flux_divs);
          values_w_ptr =
              boost::shared_ptr<VectorDouble>(data_w, &data_w->mass_values);
          dots_w_ptr =
              boost::shared_ptr<VectorDouble>(data_w, &data_w->mass_dots);

          flux_s_ptr =
              boost::shared_ptr<MatrixDouble>(data_s, &data_s->flux_values);
          divs_s_ptr =
              boost::shared_ptr<VectorDouble>(data_s, &data_s->flux_divs);
          values_s_ptr =
              boost::shared_ptr<VectorDouble>(data_s, &data_s->mass_values);
          dots_s_ptr =
              boost::shared_ptr<VectorDouble>(data_s, &data_s->mass_dots);
        }
        MoFEMErrorCode run_analysis();

private:
  MoFEMErrorCode setup_system();
  MoFEMErrorCode add_fe();
  MoFEMErrorCode extract_bd_ents(std::string ESSENTIAL, std::string NATURAL);

  MoFEMErrorCode extract_initial_ents(int block_id, Range &surface);

  MoFEMErrorCode update_slow_rhs(std::string mass_field,
                                 boost::shared_ptr<VectorDouble> &value_ptr);
  MoFEMErrorCode push_slow_rhs(boost::shared_ptr<PreviousData> &data_u,
                               boost::shared_ptr<PreviousData> &data_v,
                               boost::shared_ptr<PreviousData> &data_w,
                               boost::shared_ptr<PreviousData> &data_s);

  MoFEMErrorCode update_stiff_rhs(std::string mass_field,
                                  boost::shared_ptr<VectorDouble> &mass_ptr,
                                  boost::shared_ptr<VectorDouble> &mass_dot_ptr);
  MoFEMErrorCode push_stiff_rhs(boost::shared_ptr<PreviousData> &data_u,
                                boost::shared_ptr<PreviousData> &data_v,
                                boost::shared_ptr<PreviousData> &data_w,
                                boost::shared_ptr<PreviousData> &data_s);
  MoFEMErrorCode update_stiff_lhs(std::string mass_fiedl,
                                  boost::shared_ptr<VectorDouble> &mass_ptr);
  MoFEMErrorCode push_stiff_lhs();

  MoFEMErrorCode set_integration_rule();
  MoFEMErrorCode apply_IC(std::string mass_field, Range &surface,
                          boost::shared_ptr<VolEle> &initial_ele,
                          double init_val);
  MoFEMErrorCode apply_BC();
  MoFEMErrorCode loop_fe();
  MoFEMErrorCode post_proc_fields(std::string mass_field);
  MoFEMErrorCode output_result();
  MoFEMErrorCode solve();

  MoFEM::Interface &m_field;
  Simple *simple_interface;
  SmartPetscObj<DM> dm;
  SmartPetscObj<TS> ts;

  Range essential_bdry_ents;
  Range natural_bdry_ents;

  Range inner_surface_u; // nb_species times
  Range inner_surface_v;
  Range inner_surface_w;
  Range inner_surface_s;

  MPI_Comm cOmm;
  const int rAnk;

  int order;

  boost::shared_ptr<VolEle> vol_ele_slow_rhs;
  boost::shared_ptr<VolEle> vol_ele_stiff_rhs;
  boost::shared_ptr<VolEle> vol_ele_stiff_lhs;
  boost::shared_ptr<FaceEle> natural_bdry_ele_slow_rhs;
  boost::shared_ptr<PostProcVolumeOnRefinedMesh> post_proc;
  boost::shared_ptr<Monitor> monitor_ptr;

  boost::shared_ptr<PreviousData> data_u; // nb_species times
  boost::shared_ptr<PreviousData> data_v;
  boost::shared_ptr<PreviousData> data_w;
  boost::shared_ptr<PreviousData> data_s;

  boost::shared_ptr<MatrixDouble> flux_u_ptr; // nb_species times
  boost::shared_ptr<MatrixDouble> flux_v_ptr;
  boost::shared_ptr<MatrixDouble> flux_w_ptr;
  boost::shared_ptr<MatrixDouble> flux_s_ptr;

  boost::shared_ptr<VectorDouble> divs_u_ptr; // nb_species times
  boost::shared_ptr<VectorDouble> divs_v_ptr;
  boost::shared_ptr<VectorDouble> divs_w_ptr;
  boost::shared_ptr<VectorDouble> divs_s_ptr;

  boost::shared_ptr<VectorDouble> values_u_ptr; // nb_species times
  boost::shared_ptr<VectorDouble> values_v_ptr;
  boost::shared_ptr<VectorDouble> values_w_ptr;
  boost::shared_ptr<VectorDouble> values_s_ptr;

  boost::shared_ptr<VectorDouble> dots_u_ptr;
  boost::shared_ptr<VectorDouble> dots_v_ptr;
  boost::shared_ptr<VectorDouble> dots_w_ptr;
  boost::shared_ptr<VectorDouble> dots_s_ptr; 

  boost::shared_ptr<ForcesAndSourcesCore> null;
};

MoFEMErrorCode ElectroPhysioProblem::setup_system() {
  MoFEMFunctionBegin;
  CHKERR m_field.getInterface(simple_interface);
  CHKERR simple_interface->getOptions();
  CHKERR simple_interface->loadFile();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ElectroPhysioProblem::add_fe() {
  MoFEMFunctionBegin;
  CHKERR simple_interface->addDomainField("u", L2,
                                          AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple_interface->addDomainField("f", HDIV, DEMKOWICZ_JACOBI_BASE,
                                          1);

  CHKERR simple_interface->addBoundaryField("f", HDIV, DEMKOWICZ_JACOBI_BASE,
                                             1);

  CHKERR simple_interface->addDataField("U", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple_interface->addDataField("V", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple_interface->addDataField("W", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple_interface->addDataField("S", L2, AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple_interface->setFieldOrder("u", order - 1);
  CHKERR simple_interface->setFieldOrder("f", order);

  CHKERR simple_interface->setFieldOrder("U", order - 1);
  CHKERR simple_interface->setFieldOrder("V", order - 1);
  CHKERR simple_interface->setFieldOrder("W", order - 1);
  CHKERR simple_interface->setFieldOrder("S", order - 1);
  
  MoFEMFunctionReturn(0);
}
MoFEMErrorCode ElectroPhysioProblem::extract_bd_ents(std::string essential,
                                                     std::string natural) {
  MoFEMFunctionBegin;
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    string name = it->getName();
    if (name.compare(0, 14, natural) == 0) {

      CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), 2,
                                                 natural_bdry_ents, true);
    } else if (name.compare(0, 14, essential) == 0) {
      CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), 2,
                                                 essential_bdry_ents, true);
    }
  }
  MoFEMFunctionReturn(0);
}
MoFEMErrorCode ElectroPhysioProblem::extract_initial_ents(int block_id,
                                                          Range &surface) {
  MoFEMFunctionBegin;
  if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id,
                                                            BLOCKSET)) {
    CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        block_id, BLOCKSET, 3, surface, true);
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ElectroPhysioProblem::update_slow_rhs(
    std::string mass_field, boost::shared_ptr<VectorDouble> &mass_ptr) {
  MoFEMFunctionBegin;
  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, mass_ptr));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ElectroPhysioProblem::push_slow_rhs(boost::shared_ptr<PreviousData> &data_u,
                                    boost::shared_ptr<PreviousData> &data_v,
                                    boost::shared_ptr<PreviousData> &data_w,
                                    boost::shared_ptr<PreviousData> &data_s) {
  MoFEMFunctionBegin;

  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpAssembleSlowRhsV(data_u, data_v, data_w, data_s));

  // natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
  //     new OpAssembleNaturalBCRhsTau(natural_bdry_ents));

  natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
      new OpEssentialBC(essential_bdry_ents));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ElectroPhysioProblem::update_stiff_rhs(
    std::string mass_field,
    boost::shared_ptr<VectorDouble> &mass_ptr,
    boost::shared_ptr<VectorDouble> &mass_dot_ptr) {

  MoFEMFunctionBegin;

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, mass_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarValuesDot(mass_field, mass_dot_ptr));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ElectroPhysioProblem::push_stiff_rhs(boost::shared_ptr<PreviousData> &data_u,
                                     boost::shared_ptr<PreviousData> &data_v,
                                     boost::shared_ptr<PreviousData> &data_w,
                                     boost::shared_ptr<PreviousData> &data_s) {
  MoFEMFunctionBegin;
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsTau<3>(data_u, data_v, data_w, data_s));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsV<3>(data_u, data_v, data_w, data_s));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ElectroPhysioProblem::update_stiff_lhs(
    std::string mass_field, 
    boost::shared_ptr<VectorDouble> &mass_ptr) {
  MoFEMFunctionBegin;
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, mass_ptr));

  
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ElectroPhysioProblem::push_stiff_lhs() {
  MoFEMFunctionBegin;
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauTau<3>());

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsVV());

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauV<3>());

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsVTau());

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpDamp_dofs_to_field_data("U"));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpDamp_dofs_to_field_data("V"));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpDamp_dofs_to_field_data("W"));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpDamp_dofs_to_field_data("S"));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ElectroPhysioProblem::set_integration_rule() {
  MoFEMFunctionBegin;
  auto vol_rule = [](int, int, int p) -> int { return 2 * p + 1; };
  vol_ele_slow_rhs->getRuleHook = vol_rule;
  natural_bdry_ele_slow_rhs->getRuleHook = vol_rule;

  vol_ele_stiff_rhs->getRuleHook = vol_rule;

  vol_ele_stiff_lhs->getRuleHook = vol_rule;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
ElectroPhysioProblem::apply_IC(std::string mass_field, Range &surface,
                               boost::shared_ptr<VolEle> &initial_ele, double init_val) {
  MoFEMFunctionBegin;
  initial_ele->getOpPtrVector().push_back(
      new OpInitialMass(mass_field, surface, init_val));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ElectroPhysioProblem::apply_BC() {
  MoFEMFunctionBegin;
  CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
      "SimpleProblem", "f", essential_bdry_ents);

  MoFEMFunctionReturn(0);
}
MoFEMErrorCode ElectroPhysioProblem::loop_fe() {
  MoFEMFunctionBegin;
  CHKERR TSSetType(ts, TSARKIMEX);
  CHKERR TSARKIMEXSetType(ts, TSARKIMEXA2);

  CHKERR DMMoFEMTSSetIJacobian(dm, simple_interface->getDomainFEName(),
                               vol_ele_stiff_lhs, null, null);

  CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getDomainFEName(),
                               vol_ele_stiff_rhs, null, null);

  CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getDomainFEName(),
                                 vol_ele_slow_rhs, null, null);
  CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getBoundaryFEName(),
                                 natural_bdry_ele_slow_rhs, null, null);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ElectroPhysioProblem::post_proc_fields(std::string mass_field) {
  MoFEMFunctionBegin;
  post_proc->addFieldValuesPostProc(mass_field);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ElectroPhysioProblem::output_result() {
  MoFEMFunctionBegin;
  CHKERR DMMoFEMTSSetMonitor(dm, ts, simple_interface->getDomainFEName(),
                             monitor_ptr, null, null);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode ElectroPhysioProblem::solve() {
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

MoFEMErrorCode ElectroPhysioProblem::run_analysis() {
  MoFEMFunctionBegin;

  CHKERR setup_system(); // only once
  CHKERR add_fe(); // nb_species times
    

  CHKERR simple_interface->setUp();


  

  CHKERR update_slow_rhs("U", values_u_ptr);
  CHKERR update_slow_rhs("V", values_v_ptr);
  CHKERR update_slow_rhs("W", values_w_ptr);
  CHKERR update_slow_rhs("S", values_s_ptr);

  CHKERR push_slow_rhs(data_u, data_v, data_w, data_s);

  // natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
  //     new OpSetContrariantPiolaTransformOnEdge());

  CHKERR extract_bd_ents("ESSENTIAL", "NATURAL"); // nb_species times

  

   

  
  CHKERR update_stiff_rhs("U", values_u_ptr, dots_u_ptr);
  CHKERR update_stiff_rhs("V", values_v_ptr, dots_v_ptr);
  CHKERR update_stiff_rhs("W", values_w_ptr, dots_w_ptr);
  CHKERR update_stiff_rhs("S", values_s_ptr, dots_s_ptr);

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("f", flux_u_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("f", flux_v_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("f", flux_w_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("f", flux_s_ptr));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorDivergence<3, 3>("f", divs_u_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorDivergence<3, 3>("f", divs_v_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorDivergence<3, 3>("f", divs_w_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorDivergence<3, 3>("f", divs_s_ptr));

  CHKERR push_stiff_rhs(data_u, data_v, data_w, data_s);



  

  CHKERR update_stiff_lhs("U", values_u_ptr);
  CHKERR update_stiff_lhs("V", values_v_ptr);
  CHKERR update_stiff_lhs("W", values_w_ptr);
  CHKERR update_stiff_lhs("S", values_s_ptr);

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("f", flux_u_ptr));
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("f", flux_v_ptr));
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("f", flux_w_ptr));
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("f", flux_s_ptr));

  

  CHKERR push_stiff_lhs();
 


  
  CHKERR set_integration_rule();
  dm = simple_interface->getDM();
  ts = createTS(m_field.get_comm());

  boost::shared_ptr<VolEle> initial_mass_ele(new VolEle(m_field));

  CHKERR apply_IC("u", inner_surface_u, initial_mass_ele, init_u);
  CHKERR apply_IC("U", inner_surface_u, initial_mass_ele, init_u); // nb_species times
  CHKERR apply_IC("V", inner_surface_u, initial_mass_ele, init_v);
  CHKERR apply_IC("W", inner_surface_u, initial_mass_ele, init_w);
  CHKERR apply_IC("S", inner_surface_u, initial_mass_ele, init_s);

  

  CHKERR DMoFEMLoopFiniteElements(
                  dm, simple_interface->getDomainFEName(), initial_mass_ele);

  CHKERR apply_BC(); 
    

  CHKERR loop_fe();                          // only once
  post_proc->generateReferenceElementMesh(); // only once

  CHKERR post_proc_fields("U");
  CHKERR post_proc_fields("V");
  CHKERR post_proc_fields("W");
  CHKERR post_proc_fields("S");

  monitor_ptr = boost::shared_ptr<Monitor>(
      new Monitor(cOmm, rAnk, dm, post_proc)); 
  CHKERR output_result();                                   
  CHKERR solve();                                            
  MoFEMFunctionReturn(0);
 }

int main(int argc, char *argv[]) {
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);
  // for(int nsteps = 0; nsteps < 20; ++nsteps){
  //   if(((nsteps + 0) % 4) == 0){
  //     cout << "U" << nsteps << endl;
  //   } else if (((nsteps + 3) % 4) == 0) {
  //     cout << "V" << nsteps << endl;
  //   } else if (((nsteps + 2) % 4) == 0) {
  //     cout << "W" << nsteps << endl;
  //   } else if (((nsteps + 1) % 4) == 0) {
  //     cout << "S" << nsteps << endl;
  //   }
  // }
  try {
    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    MoFEM::Core core(moab);
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    int order = 1;
    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
    ElectroPhysioProblem electro_physio_problem(core, order + 1);
    CHKERR electro_physio_problem.run_analysis();
  }
  CATCH_ERRORS;
  MoFEM::Core::Finalize();
  return 0;
}