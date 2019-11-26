#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <ElecPhysOperators.hpp>

using namespace MoFEM;
using namespace ElectroPhysiology;

static char help[] = "...\n\n";

// #define M_PI 3.14159265358979323846 /* pi */

double init_val_u = 0.4;
double init_val_v = 0.0;

// double alpha = -0.08; 
// double gma = 3; 
// double ep = 0.005; 

  

struct Istimulus{
  double operator()(const double x, const double y, const double z, const double t) const{
    if(y<= -1.6000 && t<= 0.50000){
      return 80;
    } else{
      return 0;
    }
  }
};

struct RhsU {
double operator() (const double u, const double v) const {
    return c * u * (u - alpha) * (1.0 - u) - u * v;
  }
};

struct RhsV {
double operator() (const double u, const double v) const {
    return (gma + mu1*v/(mu2 + u)) * (-v - c * u * (u - b -1.0));
  }
};

struct RK4{
  RhsV rhs_v;
  double operator()(const double u, const double v, const double dt) const {
    double k1 = dt * rhs_v(u, v);
    double k2 = dt * rhs_v(u, v + 0.5 * k1);
    double k3 = dt * rhs_v(u, v + 0.5 * k2);
    double k4 = dt * rhs_v(u, v + k3);
    return v + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
  }
};


struct RDProblem {
public:
  RDProblem(MoFEM::Core &core, const int order)
      : m_field(core), order(order), cOmm(m_field.get_comm()),
        rAnk(m_field.get_comm_rank()) {
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

    flux_values_ptr_u =
        boost::shared_ptr<MatrixDouble>(data_u, &data_u->flux_values);

    flux_divs_ptr_u = boost::shared_ptr<VectorDouble>(data_u, &data_u->flux_divs);

    mass_values_ptr_u =
        boost::shared_ptr<VectorDouble>(data_u, &data_u->mass_values);

    mass_dots_ptr_u = boost::shared_ptr<VectorDouble>(data_u, &data_u->mass_dots);

   

    mass_values_ptr_v =
        boost::shared_ptr<VectorDouble>(data_v, &data_v->mass_values);
    mass_dots_ptr_v = boost::shared_ptr<VectorDouble>(data_v, &data_v->mass_dots);


    mass_values_ptr_w =
        boost::shared_ptr<VectorDouble>(data_w, &data_w->mass_values);

    mass_dots_ptr_w = boost::shared_ptr<VectorDouble>(data_w, &data_w->mass_dots);

    mass_values_ptr_s =
        boost::shared_ptr<VectorDouble>(data_s, &data_s->mass_values);

    mass_dots_ptr_s =
        boost::shared_ptr<VectorDouble>(data_s, &data_s->mass_dots);
  }

  // RDProblem(const int order) : order(order){}
  MoFEMErrorCode run_analysis();

  double global_error;

private:
  MoFEMErrorCode setup_system();
  MoFEMErrorCode add_fe();
  MoFEMErrorCode extract_bd_ents(std::string ESSENTIAL, std::string NATURAL);
  MoFEMErrorCode extract_initial_ents(int block_id, Range &surface);
  MoFEMErrorCode update_slow_rhs(std::string mass_field,
                                 boost::shared_ptr<VectorDouble> &mass_ptr);
  MoFEMErrorCode push_slow_rhs();
  MoFEMErrorCode
  update_stiff_rhs();
  MoFEMErrorCode push_stiff_rhs();
  MoFEMErrorCode update_stiff_lhs();
  MoFEMErrorCode push_stiff_lhs();

  MoFEMErrorCode set_integration_rule();
  MoFEMErrorCode apply_IC(std::string mass_field, Range &surface,
                          boost::shared_ptr<VolEle> &initial_ele, double &init);
  MoFEMErrorCode apply_BC(std::string flux_field);
  MoFEMErrorCode loop_fe();
  MoFEMErrorCode post_proc_fields();
  MoFEMErrorCode output_result();
  MoFEMErrorCode solve();

  MoFEM::Interface &m_field;
  Simple *simple_interface;
  SmartPetscObj<DM> dm;
  SmartPetscObj<TS> ts;

  Range essential_bdry_ents;
  Range natural_bdry_ents;

  Range inner_surface1; // nb_species times
  Range inner_surface2;
  Range inner_surface3;

  MPI_Comm cOmm;
  const int rAnk;

  int order;
  int nb_species;

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

  boost::shared_ptr<MatrixDouble> flux_values_ptr_u; // nb_species times
 

  boost::shared_ptr<VectorDouble> flux_divs_ptr_u; // nb_species times


  boost::shared_ptr<VectorDouble> mass_values_ptr_u; // nb_species times
  boost::shared_ptr<VectorDouble> mass_values_ptr_v;
  boost::shared_ptr<VectorDouble> mass_values_ptr_w;
  boost::shared_ptr<VectorDouble> mass_values_ptr_s;

  boost::shared_ptr<VectorDouble> mass_dots_ptr_u; // nb_species times
  boost::shared_ptr<VectorDouble> mass_dots_ptr_v;
  boost::shared_ptr<VectorDouble> mass_dots_ptr_w;
  boost::shared_ptr<VectorDouble> mass_dots_ptr_s;

  boost::shared_ptr<ForcesAndSourcesCore> null;
};


MoFEMErrorCode RDProblem::setup_system() {
  MoFEMFunctionBegin;
  CHKERR m_field.getInterface(simple_interface);
  CHKERR simple_interface->getOptions();
  CHKERR simple_interface->loadFile();
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::add_fe() {
  MoFEMFunctionBegin;
  CHKERR simple_interface->addDomainField("F", HDIV, DEMKOWICZ_JACOBI_BASE, 1);

  CHKERR simple_interface->addBoundaryField("F", HDIV, DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simple_interface->addDomainField("U", L2, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple_interface->addDataField("V", L2, AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple_interface->setFieldOrder("F", order);
  CHKERR simple_interface->setFieldOrder("U", order - 1);
  CHKERR simple_interface->setFieldOrder("V", order - 1);

  MoFEMFunctionReturn(0);
}


MoFEMErrorCode RDProblem::extract_bd_ents(std::string essential,
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

MoFEMErrorCode RDProblem::extract_initial_ents(int block_id, Range &surface) {
  MoFEMFunctionBegin;
  if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id,
                                                            BLOCKSET)) {
    CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        block_id, BLOCKSET, 3, surface, true);
  }
  MoFEMFunctionReturn(0);
}
MoFEMErrorCode
RDProblem::update_slow_rhs(std::string mass_field,
                           boost::shared_ptr<VectorDouble> &mass_ptr) {
  MoFEMFunctionBegin;
  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, mass_ptr));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::push_slow_rhs() {
  MoFEMFunctionBegin;

  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpAssembleSlowRhsV("U", data_u, data_v,RhsU()));

  // natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
  //     new OpAssembleNaturalBCRhsTau("F", natural_bdry_ents));

  natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
      new OpEssentialBC("F", essential_bdry_ents));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::update_stiff_rhs() {

  MoFEMFunctionBegin;

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("U", mass_values_ptr_u));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("V", mass_values_ptr_v));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpSolveRecovery("V", data_u, data_v, RK4()));

  

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarValuesDot("U", mass_dots_ptr_u));
 

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("F", flux_values_ptr_u));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorDivergence<3, 3>("F", flux_divs_ptr_u));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::push_stiff_rhs() {
  MoFEMFunctionBegin;
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsTau<3>("F", data_u));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsV<3>("U", data_u, inner_surface2));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::update_stiff_lhs() {
  MoFEMFunctionBegin;
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("U", mass_values_ptr_u));
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues("V", mass_values_ptr_v));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>("F", flux_values_ptr_u));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::push_stiff_lhs() {
  MoFEMFunctionBegin;
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauTau<3>("F"));
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauV<3>("F", "U"));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsVV("U"));



  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsVTau("U", "F"));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::set_integration_rule() {
  MoFEMFunctionBegin;
  auto vol_rule = [](int, int, int p) -> int { return 2 * p; };
  vol_ele_slow_rhs->getRuleHook = vol_rule;
  natural_bdry_ele_slow_rhs->getRuleHook = vol_rule;

  vol_ele_stiff_rhs->getRuleHook = vol_rule;

  vol_ele_stiff_lhs->getRuleHook = vol_rule;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::apply_IC(std::string mass_field, Range &surface,
                                   boost::shared_ptr<VolEle> &initial_ele, double &init_val) {
  MoFEMFunctionBegin;
  initial_ele->getOpPtrVector().push_back(
      new OpInitialMass(mass_field, surface, init_val));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::apply_BC(std::string flux_field) {
  MoFEMFunctionBegin;
  CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
      "SimpleProblem", flux_field, essential_bdry_ents);

  MoFEMFunctionReturn(0);
}
MoFEMErrorCode RDProblem::loop_fe() {
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

MoFEMErrorCode RDProblem::post_proc_fields() {
  MoFEMFunctionBegin;
  post_proc->addFieldValuesPostProc("U");
  // post_proc->addFieldValuesPostProc("F");
  post_proc->addFieldValuesPostProc("V");


  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::output_result() {
  MoFEMFunctionBegin;
  CHKERR DMMoFEMTSSetMonitor(dm, ts, simple_interface->getDomainFEName(),
                             monitor_ptr, null, null);
  MoFEMFunctionReturn(0);
}
MoFEMErrorCode RDProblem::solve() {
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

MoFEMErrorCode RDProblem::run_analysis() {
  MoFEMFunctionBegin;
  global_error = 0;

  CHKERR setup_system(); // only once


  CHKERR add_fe(); // nb_species times
   

  CHKERR simple_interface->setUp();

  // CHKERR set_blockData(material_blocks);

  CHKERR extract_bd_ents("ESSENTIAL", "NATURAL"); // nb_species times

  CHKERR extract_initial_ents(2, inner_surface1);
  CHKERR extract_initial_ents(3, inner_surface2);

  CHKERR update_slow_rhs("U", mass_values_ptr_u);
  CHKERR update_slow_rhs("V", mass_values_ptr_v);


  // natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
  //     new OpSetContrariantPiolaTransformOnEdge());

  CHKERR push_slow_rhs(); // nb_species times


  CHKERR update_stiff_rhs();
  CHKERR push_stiff_rhs(); // nb_species times



  CHKERR update_stiff_lhs();
  CHKERR push_stiff_lhs(); // nb_species times
   
  CHKERR set_integration_rule();
  dm = simple_interface->getDM();
  ts = createTS(m_field.get_comm());
  boost::shared_ptr<VolEle> initial_mass_ele(new VolEle(m_field));


  CHKERR apply_IC("U", inner_surface1,
                    initial_mass_ele, init_val_u); // nb_species times
  CHKERR apply_IC("V", inner_surface1,
                  initial_mass_ele, init_val_v); // nb_species times


  CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                  initial_mass_ele);

  
  CHKERR apply_BC("F"); // nb_species times
   

  CHKERR loop_fe();                          // only once
  post_proc->generateReferenceElementMesh(); // only once


  CHKERR post_proc_fields();
    

  monitor_ptr = boost::shared_ptr<Monitor>(
      new Monitor(cOmm, rAnk, dm, post_proc)); // nb_species times
  CHKERR output_result();                                    // only once
  CHKERR solve();                                            // only once
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
    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
    RDProblem reac_diff_problem(core, order + 1);
    CHKERR reac_diff_problem.run_analysis();
  }
  CATCH_ERRORS;
  MoFEM::Core::Finalize();
  return 0;
}