#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <RDOperators.hpp>

using namespace MoFEM;
using namespace ReactionDiffusion;




static char help[] = "...\n\n";

struct RDProblem {
public:
  RDProblem(MoFEM::Core &core, const int order)
  : m_field(core), 
    order(order)
    {
      vol_ele_slow_rhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));
      natural_bdry_ele_slow_rhs =
          boost::shared_ptr<BoundaryEle>(new BoundaryEle(m_field));
      vol_ele_stiff_rhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));
      vol_ele_stiff_lhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));
      post_proc = boost::shared_ptr<PostProcFaceOnRefinedMesh>(
          new PostProcFaceOnRefinedMesh(m_field));

      data1 = boost::shared_ptr<PreviousData>(new PreviousData());
      data2 = boost::shared_ptr<PreviousData>(new PreviousData());
      data3 = boost::shared_ptr<PreviousData>(new PreviousData());

      flux_values_ptr1 =
          boost::shared_ptr<MatrixDouble>(data1, &data1->flux_values);
      flux_divs_ptr1 = boost::shared_ptr<VectorDouble>(data1,
      &data1->flux_divs); mass_values_ptr1 =
          boost::shared_ptr<VectorDouble>(data1, &data1->mass_values);
      mass_dots_ptr1 = boost::shared_ptr<VectorDouble>(data1,
      &data1->mass_dots);

      flux_values_ptr2 =
          boost::shared_ptr<MatrixDouble>(data2, &data2->flux_values);
      flux_divs_ptr2 = boost::shared_ptr<VectorDouble>(data2,
      &data2->flux_divs); mass_values_ptr2 =
          boost::shared_ptr<VectorDouble>(data2, &data2->mass_values);
      mass_dots_ptr2 = boost::shared_ptr<VectorDouble>(data2,
      &data2->mass_dots);

      flux_values_ptr3 =
          boost::shared_ptr<MatrixDouble>(data3, &data3->flux_values);
      flux_divs_ptr3 = boost::shared_ptr<VectorDouble>(data3,
      &data3->flux_divs); mass_values_ptr3 =
          boost::shared_ptr<VectorDouble>(data3, &data3->mass_values);
      mass_dots_ptr3 = boost::shared_ptr<VectorDouble>(data3,
      &data3->mass_dots);
    }
    
  //RDProblem(const int order) : order(order){}
  void run_analysis();

private:
  void setup_system();
  void add_fe(std::string mass_field, std::string flux_field);
  void extract_bd_ents(std::string ESSENTIAL, std::string NATURAL);
  void extract_initial_ents(int block_id, Range &surface);
  void update_slow_rhs(std::string mass_fiedl,
                  boost::shared_ptr<VectorDouble> &mass_ptr);
  void push_slow_rhs(std::string mass_field, std::string flux_field,
                     boost::shared_ptr<PreviousData> &data);
  void update_vol_fe(boost::shared_ptr<FaceEle> &vol_ele, boost::shared_ptr<PreviousData> &data);
  void update_stiff_rhs(std::string mass_field, std::string flux_field,
    boost::shared_ptr<VectorDouble> &mass_ptr,
    boost::shared_ptr<MatrixDouble> &flux_ptr,
    boost::shared_ptr<VectorDouble> &mass_dot_ptr,
    boost::shared_ptr<VectorDouble> &flux_div_ptr);
  void push_stiff_rhs(std::string mass_field, std::string flux_field,
                      boost::shared_ptr<PreviousData> &data);
  void update_stiff_lhs(std::string mass_fiedl, std::string flux_field,
                        boost::shared_ptr<VectorDouble> &mass_ptr,
                        boost::shared_ptr<MatrixDouble> &flux_ptr);
  void push_stiff_lhs(std::string mass_field, std::string flux_field,
                      boost::shared_ptr<PreviousData> &data);

  void set_integration_rule();
  void apply_IC(std::string mass_field, Range &surface,
                boost::shared_ptr<FaceEle> &initial_ele);
  void apply_BC(std::string flux_field);
  void loop_fe();
  void post_proc_fields(std::string mass_field, std::string flux_field);
  void output_result();
  void solve();


  MoFEM::Interface &m_field;
  Simple *simple_interface;
  SmartPetscObj<DM> dm;
  SmartPetscObj<TS> ts;

  Range essential_bdry_ents;
  Range natural_bdry_ents;

  Range inner_surface1; // nb_species times
  Range inner_surface2;
  Range inner_surface3;

  int order;


  boost::shared_ptr<FaceEle>                   vol_ele_slow_rhs;
  boost::shared_ptr<FaceEle>                   vol_ele_stiff_rhs;
  boost::shared_ptr<FaceEle>                   vol_ele_stiff_lhs;
  boost::shared_ptr<BoundaryEle>               natural_bdry_ele_slow_rhs;
  boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc;
  boost::shared_ptr<Monitor>                   monitor_ptr;

  boost::shared_ptr<PreviousData> data1; // nb_species times
  boost::shared_ptr<PreviousData> data2;
  boost::shared_ptr<PreviousData> data3;

  boost::shared_ptr<MatrixDouble> flux_values_ptr1; // nb_species times
  boost::shared_ptr<MatrixDouble> flux_values_ptr2;
  boost::shared_ptr<MatrixDouble> flux_values_ptr3;

  boost::shared_ptr<VectorDouble> flux_divs_ptr1;   // nb_species times
  boost::shared_ptr<VectorDouble> flux_divs_ptr2;
  boost::shared_ptr<VectorDouble> flux_divs_ptr3;

  boost::shared_ptr<VectorDouble> mass_values_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> mass_values_ptr2;
  boost::shared_ptr<VectorDouble> mass_values_ptr3;

  boost::shared_ptr<VectorDouble> mass_dots_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> mass_dots_ptr2;
  boost::shared_ptr<VectorDouble> mass_dots_ptr3;

  boost::shared_ptr<ForcesAndSourcesCore>      null;
};

void RDProblem::setup_system()
{
  CHKERR m_field.getInterface(simple_interface);
  CHKERR simple_interface->getOptions();
  CHKERR simple_interface->loadFile();
}

void RDProblem::add_fe(std::string mass_field, std::string flux_field) {
  CHKERR simple_interface->addDomainField(mass_field, L2, AINSWORTH_LEGENDRE_BASE,
                                          1);

  CHKERR simple_interface->addDomainField(flux_field, HCURL,
                                          AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple_interface->addBoundaryField(flux_field, HCURL,
                                            DEMKOWICZ_JACOBI_BASE, 1);

  CHKERR simple_interface->setFieldOrder(mass_field, order - 1);
  CHKERR simple_interface->setFieldOrder(flux_field, order);
}
void RDProblem::extract_bd_ents(std::string essential, std::string natural) {
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    string name = it->getName();
    if (name.compare(0, 14, natural) == 0) {

      CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), 1,
                                                 natural_bdry_ents, true);
    } else if (name.compare(0, 14, essential) == 0) {
      CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), 1,
                                                 essential_bdry_ents, true);
    }
  }
}

void RDProblem::extract_initial_ents(int block_id, Range &surface)
{
  if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id,
                                                            BLOCKSET)) {
    CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        block_id, BLOCKSET, 2, surface, true);
  }
}
void RDProblem::update_slow_rhs(std::string mass_field,
                                boost::shared_ptr<VectorDouble> &mass_ptr)
{
  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, mass_ptr));
}

void RDProblem::push_slow_rhs(std::string mass_field, std::string flux_field,
                              boost::shared_ptr<PreviousData> &data) {

  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpAssembleSlowRhsV(mass_field, data));



  natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
      new OpAssembleNaturalBCRhsTau(flux_field, natural_bdry_ents));

  natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
      new OpEssentialBC(flux_field, essential_bdry_ents));
}

void RDProblem::update_vol_fe(boost::shared_ptr<FaceEle> &vol_ele,
                   boost::shared_ptr<PreviousData> &data)
{
  vol_ele->getOpPtrVector().push_back(
      new OpCalculateJacForFace(data->jac));
  vol_ele->getOpPtrVector().push_back(
      new OpCalculateInvJacForFace(data->inv_jac));
  vol_ele->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());

  vol_ele->getOpPtrVector().push_back(
      new OpSetContravariantPiolaTransformFace(data->jac));

  vol_ele->getOpPtrVector().push_back(
      new OpSetInvJacHcurlFace(data->inv_jac));
}
void RDProblem::update_stiff_rhs(
    std::string mass_field, std::string flux_field,
    boost::shared_ptr<VectorDouble> &mass_ptr,
    boost::shared_ptr<MatrixDouble> &flux_ptr,
    boost::shared_ptr<VectorDouble> &mass_dot_ptr,
    boost::shared_ptr<VectorDouble> &flux_div_ptr) {

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, mass_ptr));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>(flux_field, flux_ptr));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarValuesDot(mass_field, mass_dot_ptr));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorDivergence<3, 2>(flux_field, flux_div_ptr));
}

void RDProblem::push_stiff_rhs(std::string mass_field, std::string flux_field,
                               boost::shared_ptr<PreviousData> &data) {
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsTau<3>(flux_field, data));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsV<3>(mass_field, data));
}

void RDProblem::update_stiff_lhs(std::string mass_field, std::string flux_field,
                                 boost::shared_ptr<VectorDouble> &mass_ptr,
                                 boost::shared_ptr<MatrixDouble> &flux_ptr) {
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, mass_ptr));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>(flux_field, flux_ptr));
}

void RDProblem::push_stiff_lhs(std::string mass_field, std::string flux_field,
                               boost::shared_ptr<PreviousData> &data) {

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauTau<3>(flux_field, data));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(new OpAssembleLhsVV(mass_field));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauV<3>(flux_field, mass_field, data));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsVTau(mass_field, flux_field));
}

void RDProblem::set_integration_rule()
{
  auto vol_rule = [](int, int, int p) -> int { return 2 * p; };
  vol_ele_slow_rhs->getRuleHook = vol_rule;
  natural_bdry_ele_slow_rhs->getRuleHook = vol_rule;

  vol_ele_stiff_rhs->getRuleHook = vol_rule;

  vol_ele_stiff_lhs->getRuleHook = vol_rule;
}

void RDProblem::apply_IC(std::string mass_field, Range &surface,
                         boost::shared_ptr<FaceEle> &initial_ele) {

  initial_ele->getOpPtrVector().push_back(
      new OpInitialMass(mass_field, surface));

}

void RDProblem::apply_BC(std::string flux_field)
{

  CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
      "SimpleProblem", flux_field, essential_bdry_ents);
}
void RDProblem::loop_fe()
{
 
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
}

void RDProblem::post_proc_fields(std::string mass_field, std::string flux_field)
{
  post_proc->addFieldValuesPostProc(mass_field);
  post_proc->addFieldValuesPostProc(flux_field);
}

void RDProblem::output_result()
{
  
  CHKERR DMMoFEMTSSetMonitor(dm, ts, simple_interface->getDomainFEName(),
                             monitor_ptr, null, null);
}
void RDProblem::solve()
{
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
}

void RDProblem::run_analysis() {
  setup_system(); // only once
  add_fe("MASS1", "FLUX1"); // nb_species times
  // add_fe("MASS2", "FLUX2");
  // add_fe("MASS3", "FLUX3");

  CHKERR simple_interface->setUp();

  extract_bd_ents("ESSENTIAL", "NATURAL"); // nb_species times

  extract_initial_ents(2, inner_surface1);
  // extract_initial_ents(3, inner_surface2);
  // extract_initial_ents(4, inner_surface3);

  update_slow_rhs("MASS1", mass_values_ptr1);
  // update_slow_rhs("MASS2", mass_values_ptr2);
  // update_slow_rhs("MASS3", mass_values_ptr3);

  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpComputeSlowValue("MASS1", data1, data1, data1));
  // vol_ele_slow_rhs->getOpPtrVector().push_back(
  //     new OpComputeSlowValue("MASS2", data1, data2, data3));
  // vol_ele_slow_rhs->getOpPtrVector().push_back(
  //     new OpComputeSlowValue("MASS3", data1, data2, data3));

  natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
      new OpSetContrariantPiolaTransformOnEdge());

  push_slow_rhs("MASS1", "FLUX1", data1); // nb_species times
  // push_slow_rhs("MASS2", "FLUX2", data2);
  // push_slow_rhs("MASS3", "FLUX3", data3);

  update_vol_fe(vol_ele_stiff_rhs, data1);
  // update_vol_fe(vol_ele_stiff_rhs, data2);
  // update_vol_fe(vol_ele_stiff_rhs, data3);

  update_stiff_rhs("MASS1", "FLUX1", mass_values_ptr1, flux_values_ptr1,
                   mass_dots_ptr1, flux_divs_ptr1);
  // update_stiff_rhs("MASS2", "FLUX2", mass_values_ptr2, flux_values_ptr2,
  //                  mass_dots_ptr2, flux_divs_ptr2);
  // update_stiff_rhs("MASS3", "FLUX3", mass_values_ptr3, flux_values_ptr3,
  //                  mass_dots_ptr3, flux_divs_ptr3);

  push_stiff_rhs("MASS1", "FLUX1", data1); // nb_species times
  // push_stiff_rhs("MASS2", "FLUX2", data2);
  // push_stiff_rhs("MASS3", "FLUX3", data3);

  update_vol_fe(vol_ele_stiff_lhs, data1);
  // update_vol_fe(vol_ele_stiff_lhs, data2);
  // update_vol_fe(vol_ele_stiff_lhs, data3);

  update_stiff_lhs("MASS1", "FLUX1", mass_values_ptr1, flux_values_ptr1);
  // update_stiff_lhs("MASS2", "FLUX2", mass_values_ptr2, flux_values_ptr2);
  // update_stiff_lhs("MASS3", "FLUX3", mass_values_ptr3, flux_values_ptr3);

  push_stiff_lhs("MASS1", "FLUX1", data1); // nb_species times
  // push_stiff_lhs("MASS2", "FLUX2", data2);
  // push_stiff_lhs("MASS3", "FLUX3", data3);

  set_integration_rule();
  dm = simple_interface->getDM();
  ts = createTS(m_field.get_comm());
  boost::shared_ptr<FaceEle> initial_mass_ele(new FaceEle(m_field));

  apply_IC("MASS1", inner_surface1, initial_mass_ele); // nb_species times
  // apply_IC("MASS2", inner_surface2, initial_mass_ele);
  // apply_IC("MASS3", inner_surface3, initial_mass_ele);

  CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                  initial_mass_ele);

  apply_BC("FLUX1"); // nb_species times 
  // apply_BC("FLUX2");
  // apply_BC("FLUX3");

  loop_fe(); // only once
  post_proc->generateReferenceElementMesh(); // only once

  post_proc_fields("MASS1", "FLUX1");
  // post_proc_fields("MASS2", "FLUX2");
  // post_proc_fields("MASS3", "FLUX3");

  //auto dm = simple_interface->getDM();
  monitor_ptr = boost::shared_ptr<Monitor>(
      new Monitor(dm, post_proc)); // nb_species times
  output_result(); // only once
  solve(); // only once
}

int main(int argc, char *argv[]) {
  const char param_file[] = "param_file.petsc";
  MoFEM::Core::Initialize(&argc, &argv, param_file, help);
  try{
    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;
    MoFEM::Core core(moab);
    DMType dm_name = "DMMOFEM";
    CHKERR DMRegister_MoFEM(dm_name);

    int order = 3;
    RDProblem reac_diff_problem(core, order);
    reac_diff_problem.run_analysis();
  }
  CATCH_ERRORS;
  MoFEM::Core::Finalize();
  return 0;
}