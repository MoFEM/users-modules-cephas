#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <ElecPhysOperators2D.hpp>

using namespace MoFEM;
using namespace ElectroPhysiology;

static char help[] = "...\n\n";

// #define M_PI 3.14159265358979323846 /* pi */

const int dim = 2;

double init_val_u = 0.4;
double init_val_v = 0.0;

// double alpha = -0.08;
// double gma = 3;
// double ep = 0.005;



struct Istimulus {
  double operator()(const double x, const double y, const double z,
                    const double t) const {
    if (y <= -1.6000 && t <= 0.50000) {
      return 80;
    } else {
      return 0;
    }
  }
};

struct RhsU {
  double operator()(const double u, const double v) const {
    return factor * (1.0 * (c * u * (u - alpha) * (1.0 - u) - u * v));
  }
};

struct DRhsU_u {
  double operator()(const double u, const double v) const {
    return factor * (c * ((u - alpha) * (1.0 - u) + u * (1.0 - u) - u * (u - alpha)) - v);
  }
};

struct RhsV {
  double operator()(const double u, const double v) const {
    return factor * ((gma + mu1 * v / (mu2 + u)) * (-v - c * u * (u - b - 1.0)));
  }
};

struct DRhsV_v {
  double operator()(const double u, const double v) const {
    return factor * (((gma + mu1 / (mu2 + u)) * (-v - c * u * (u - b - 1.0)) -
            - (gma + mu1 * v / (mu2 + u))));
  }
};

const double Tol = 1e-9;

struct RK4 {
  RhsV rhs_v;
  DRhsV_v Drhs_v;
  double operator()(const double u, const double v, const double dt) const {
    

    double dv;

    double vk = v;

    double err = 1;

    while (err > Tol){
      double rhsv = rhs_v(u, vk);
      double Drhsv = Drhs_v(u, vk);
      dv = - 1.0 / (1 - dt * Drhsv) * (vk - v - dt * rhsv);
      vk += dv;
      err = abs(dv);
    }

    return vk;

    // double k1 = dt * rhs_v(u, v);
    // // double k2 = dt * rhs_v(u, v + 0.5 * k1);
    // // double k3 = dt * rhs_v(u, v + 0.5 * k2);
    // double k4 = dt * rhs_v(u, v + k1);
    // return v + 0.5 * (k1 + k4);
  }
};

struct RDProblem {
public:
  RDProblem(MoFEM::Core &core, const int order)
      : m_field(core), order(order), cOmm(m_field.get_comm()),
        rAnk(m_field.get_comm_rank()) {
    vol_ele_slow_rhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));
    natural_bdry_ele_slow_rhs =
        boost::shared_ptr<BoundaryEle>(new BoundaryEle(m_field));
    vol_ele_stiff_rhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));
    vol_ele_stiff_lhs = boost::shared_ptr<FaceEle>(new FaceEle(m_field));
    post_proc = boost::shared_ptr<PostProcFaceOnRefinedMesh>(
        new PostProcFaceOnRefinedMesh(m_field));

    data1 = boost::shared_ptr<PreviousData>(new PreviousData());
    data2 = boost::shared_ptr<PreviousData>(new PreviousData());
  

    flux_values_ptr1 =
        boost::shared_ptr<MatrixDouble>(data1, &data1->flux_values);

    flux_divs_ptr1 = boost::shared_ptr<VectorDouble>(data1, &data1->flux_divs);

    mass_values_ptr1 =
        boost::shared_ptr<VectorDouble>(data1, &data1->mass_values);

    mass_dots_ptr1 = boost::shared_ptr<VectorDouble>(data1, &data1->mass_dots);

    flux_values_ptr2 =
        boost::shared_ptr<MatrixDouble>(data2, &data2->flux_values);
    flux_divs_ptr2 = boost::shared_ptr<VectorDouble>(data2, &data2->flux_divs);

    mass_values_ptr2 =
        boost::shared_ptr<VectorDouble>(data2, &data2->mass_values);
    mass_dots_ptr2 = boost::shared_ptr<VectorDouble>(data2, &data2->mass_dots);


  }

  // RDProblem(const int order) : order(order){}
  MoFEMErrorCode run_analysis();

  double global_error;

private:
  MoFEMErrorCode setup_system();
  MoFEMErrorCode add_fe();
  MoFEMErrorCode set_blockData(std::map<int, BlockData> &block_data_map);
  MoFEMErrorCode extract_bd_ents(std::string ESSENTIAL, std::string NATURAL);
  MoFEMErrorCode extract_initial_ents(int block_id, Range &surface);
  MoFEMErrorCode update_slow_rhs(std::string mass_fiedl,
                                 boost::shared_ptr<VectorDouble> &mass_ptr);
  MoFEMErrorCode push_slow_rhs(std::string mass_field, std::string flux_field,
                               boost::shared_ptr<PreviousData> &data1,
                               boost::shared_ptr<PreviousData> &data2);
  MoFEMErrorCode update_vol_fe(boost::shared_ptr<FaceEle> &vol_ele,
                               boost::shared_ptr<PreviousData> &data);
  MoFEMErrorCode
  update_stiff_rhs(std::string mass_field, std::string flux_field,
                   boost::shared_ptr<VectorDouble> &mass_ptr,
                   boost::shared_ptr<MatrixDouble> &flux_ptr,
                   boost::shared_ptr<VectorDouble> &mass_dot_ptr,
                   boost::shared_ptr<VectorDouble> &flux_div_ptr);
  MoFEMErrorCode push_stiff_rhs(std::string mass_field, std::string flux_field,
                                boost::shared_ptr<PreviousData> &data1,
                                boost::shared_ptr<PreviousData> &data2,
                                std::map<int, BlockData> &block_map,
                                Range &surface);
  MoFEMErrorCode update_stiff_lhs(std::string mass_fiedl,
                                  std::string flux_field,
                                  boost::shared_ptr<VectorDouble> &mass_ptr,
                                  boost::shared_ptr<MatrixDouble> &flux_ptr);
  MoFEMErrorCode push_stiff_lhs(std::string mass_field, std::string flux_field,
                                boost::shared_ptr<PreviousData> &datau,
                                boost::shared_ptr<PreviousData> &datav,
                                std::map<int, BlockData> &block_map);

  MoFEMErrorCode set_integration_rule();
  MoFEMErrorCode apply_IC(std::string mass_field, Range &surface,
                          boost::shared_ptr<FaceEle> &initial_ele,
                          double & init_val);
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


  MPI_Comm cOmm;
  const int rAnk;

  int order;
  int nb_species;

  std::map<int, BlockData> material_blocks;

  boost::shared_ptr<FaceEle> vol_ele_slow_rhs;
  boost::shared_ptr<FaceEle> vol_ele_stiff_rhs;
  boost::shared_ptr<FaceEle> vol_ele_stiff_lhs;
  boost::shared_ptr<BoundaryEle> natural_bdry_ele_slow_rhs;

  boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc;
  boost::shared_ptr<Monitor> monitor_ptr;

  boost::shared_ptr<PreviousData> data1; // nb_species times
  boost::shared_ptr<PreviousData> data2;


  boost::shared_ptr<MatrixDouble> flux_values_ptr1; // nb_species times
  boost::shared_ptr<MatrixDouble> flux_values_ptr2;


  boost::shared_ptr<VectorDouble> flux_divs_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> flux_divs_ptr2;


  boost::shared_ptr<VectorDouble> mass_values_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> mass_values_ptr2;


  boost::shared_ptr<VectorDouble> mass_dots_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> mass_dots_ptr2;


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
  CHKERR simple_interface->addDomainField("U", L2,
                                          AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple_interface->addDataField("V", L2, 
                                          AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple_interface->addDomainField("F", HCURL,
                                          AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple_interface->addBoundaryField("F", HCURL,
                                            DEMKOWICZ_JACOBI_BASE, 1);

  

  CHKERR simple_interface->setFieldOrder("U", order - 1);
  CHKERR simple_interface->setFieldOrder("V", order - 1);
  CHKERR simple_interface->setFieldOrder("F", order);


  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::set_blockData(std::map<int, BlockData> &block_map) {
  MoFEMFunctionBegin;
  for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
    string name = it->getName();
    const int id = it->getMeshsetId();
    if (name.compare(0, 14, "REGION1") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 1e-3;
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION2") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 1e-1;
      block_map[id].block_id = id;
    } 
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::extract_bd_ents(std::string essential,
                                          std::string natural) {
  MoFEMFunctionBegin;
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
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::extract_initial_ents(int block_id, Range &surface) {
  MoFEMFunctionBegin;
  if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id,
                                                            BLOCKSET)) {
    CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        block_id, BLOCKSET, 2, surface, true);
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
RDProblem::push_slow_rhs(std::string mass_field, std::string flux_field,
                         boost::shared_ptr<PreviousData> &data_1,
                         boost::shared_ptr<PreviousData> &data_2) {
  MoFEMFunctionBegin;

  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpAssembleSlowRhsV(mass_field, data_1, data_2, RhsU()));

  // natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
  //     new OpAssembleNaturalBCRhsTau(flux_field, natural_bdry_ents));

  natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
      new OpEssentialBC(flux_field, essential_bdry_ents));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::update_vol_fe(boost::shared_ptr<FaceEle> &vol_ele,
                                        boost::shared_ptr<PreviousData> &data) {
  MoFEMFunctionBegin;
  vol_ele->getOpPtrVector().push_back(new OpCalculateJacForFace(data->jac));
  vol_ele->getOpPtrVector().push_back(
      new OpCalculateInvJacForFace(data->inv_jac));
  vol_ele->getOpPtrVector().push_back(new OpMakeHdivFromHcurl());

  vol_ele->getOpPtrVector().push_back(
      new OpSetContravariantPiolaTransformFace(data->jac));

  vol_ele->getOpPtrVector().push_back(new OpSetInvJacHcurlFace(data->inv_jac));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::update_stiff_rhs(std::string mass_field, std::string flux_field,
                            boost::shared_ptr<VectorDouble> &mass_ptr,
                            boost::shared_ptr<MatrixDouble> &flux_ptr,
                            boost::shared_ptr<VectorDouble> &mass_dot_ptr,
                            boost::shared_ptr<VectorDouble> &flux_div_ptr) {

  MoFEMFunctionBegin;

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, mass_ptr));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>(flux_field, flux_ptr));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarValuesDot(mass_field, mass_dot_ptr));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorDivergence<3, 2>(flux_field, flux_div_ptr));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::push_stiff_rhs(std::string mass_field,
                                         std::string flux_field,
                                         boost::shared_ptr<PreviousData> &data1,
                                         boost::shared_ptr<PreviousData> &data2,
                                         std::map<int, BlockData> &block_map,
                                         Range &surface) {
  MoFEMFunctionBegin;
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsTau<3>(flux_field, data1, block_map));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsV<3>(mass_field, data1, data2, RhsU(), block_map, surface));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::update_stiff_lhs(std::string mass_field, std::string flux_field,
                            boost::shared_ptr<VectorDouble> &mass_ptr,
                            boost::shared_ptr<MatrixDouble> &flux_ptr) {
  MoFEMFunctionBegin;
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, mass_ptr));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateHdivVectorField<3>(flux_field, flux_ptr));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::push_stiff_lhs(std::string mass_field,
                                         std::string flux_field,
                                         boost::shared_ptr<PreviousData> &data1,
                                         boost::shared_ptr<PreviousData> &data2,
                                         std::map<int, BlockData> &block_map) {
  MoFEMFunctionBegin;
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauTau<3>(flux_field, data1, block_map));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsVV(mass_field, data1, data2, DRhsU_u()));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauV<3>(flux_field, mass_field, data1, block_map));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsVTau(mass_field, flux_field));
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
                                   boost::shared_ptr<FaceEle> &initial_ele,
                                   double & init_val) {
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
  post_proc->addFieldValuesPostProc("F");
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

    CHKERR set_blockData(material_blocks);

    CHKERR extract_bd_ents("ESSENTIAL", "NATURAL"); // nb_species times

    CHKERR extract_initial_ents(2, inner_surface1);
    CHKERR extract_initial_ents(3, inner_surface2);

    CHKERR update_slow_rhs("U", mass_values_ptr1);
    CHKERR update_slow_rhs("V", mass_values_ptr2);

    natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
        new OpSetContrariantPiolaTransformOnEdge());

    CHKERR push_slow_rhs("U", "F", data1, data2); // nb_species times

    CHKERR update_vol_fe(vol_ele_stiff_rhs, data1);

    CHKERR update_stiff_rhs("U", "F", mass_values_ptr1,
                            flux_values_ptr1, mass_dots_ptr1, flux_divs_ptr1);
    vol_ele_stiff_rhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("V", mass_values_ptr2));
    CHKERR push_stiff_rhs("U", "F", data1, data2,
                          material_blocks, inner_surface2); // nb_species times

    CHKERR update_vol_fe(vol_ele_stiff_lhs, data1);

    CHKERR update_stiff_lhs("U", "F", mass_values_ptr1,
                            flux_values_ptr1);

    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpCalculateScalarFieldValues("V", mass_values_ptr2));
    CHKERR push_stiff_lhs("U", "F", data1, data2,
                              material_blocks); // nb_species times
    vol_ele_stiff_lhs->getOpPtrVector().push_back(
        new OpSolveRecovery("V", data1, data2, RK4()));

    CHKERR set_integration_rule();
    dm = simple_interface->getDM();
    ts = createTS(m_field.get_comm());
    boost::shared_ptr<FaceEle> initial_mass_ele(new FaceEle(m_field));

    CHKERR apply_IC("U", inner_surface1, initial_mass_ele,
                    init_val_u); // nb_species times
    CHKERR apply_IC("V", inner_surface1, initial_mass_ele,
                    init_val_v); // nb_species times

    CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                    initial_mass_ele);

    CHKERR apply_BC("F"); // nb_species times

    CHKERR loop_fe();                          // only once
    post_proc->generateReferenceElementMesh(); // only once

    CHKERR post_proc_fields();

    monitor_ptr = boost::shared_ptr<Monitor>(
        new Monitor(cOmm, rAnk, dm, post_proc)); // nb_species times
    CHKERR output_result();                      // only once
    CHKERR solve();                              // only once
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