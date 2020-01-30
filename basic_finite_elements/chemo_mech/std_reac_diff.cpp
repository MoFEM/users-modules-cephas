#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <rd_stdOperators.hpp>

using namespace MoFEM;
using namespace StdRDOperators;

static char help[] = "...\n\n";

struct RhsU {
  double operator()(const double u, const double v) const {
    return c * u * (u - alpha) * (1.0 - u) - u * v;
  }
};

struct RhsV {
  double operator()(const double u, const double v) const {
    return (gma + mu1 * v / (mu2 + u)) * (-v - c * u * (u - b - 1.0));
  }
};

struct RDProblem {
public:
  RDProblem(moab::Core &mb_instance, MoFEM::Core &core, const int order, const int n_species)
      : moab(mb_instance)
      , m_field(core)
      , order(order)
      , nb_species(n_species)
      , cOmm(m_field.get_comm())
      , rAnk(m_field.get_comm_rank()) {
    vol_ele_slow_rhs = boost::shared_ptr<Ele>(new Ele(m_field));
    vol_ele_stiff_rhs = boost::shared_ptr<Ele>(new Ele(m_field));
    vol_ele_stiff_lhs = boost::shared_ptr<Ele>(new Ele(m_field));

    boundary_ele_rhs = boost::shared_ptr<BoundaryEle>(new BoundaryEle(m_field));

    vol_mass_ele = boost::shared_ptr<Ele>(new Ele(m_field));

    post_proc = boost::shared_ptr<PostProcFaceOnRefinedMesh>(
        new PostProcFaceOnRefinedMesh(m_field));

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

  MoFEMErrorCode run_analysis();

private:
  MoFEMErrorCode setup_system();
  MoFEMErrorCode add_fe(std::string field_name);
  MoFEMErrorCode set_blockData(std::map<int, BlockData> &block_data_map);

  MoFEMErrorCode set_initial_values(std::string field_name, int block_id,
                                    Range &surface, double &init_val);

  MoFEMErrorCode update_slow_rhs(std::string mass_fiedl,
                                 boost::shared_ptr<VectorDouble> &mass_ptr);

  MoFEMErrorCode push_slow_rhs(std::string field_name,
                               boost::shared_ptr<PreviousData> &dataU,
                               boost::shared_ptr<PreviousData> &dataV);

  MoFEMErrorCode push_mass_ele(std::string field_name);

  MoFEMErrorCode resolve_slow_rhs();

  

  MoFEMErrorCode update_vol_fe(boost::shared_ptr<Ele> &vol_ele,
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
  SmartPetscObj<Mat> mass_matrix;
  SmartPetscObj<KSP> mass_Ksp;

  Range bdry_ents;

  std::vector<Range> inner_surface;
  Range stimulation_surface;



  double global_error;

  MPI_Comm cOmm;
  const int rAnk;

  int order;
  int nb_species;

  std::map<int, BlockData> material_blocks;

  boost::shared_ptr<Ele> vol_ele_slow_rhs;
  boost::shared_ptr<Ele> vol_ele_stiff_rhs;
  boost::shared_ptr<Ele> vol_ele_stiff_lhs;
  boost::shared_ptr<BoundaryEle> boundary_ele_rhs;

  boost::shared_ptr<Ele> vol_mass_ele;

  boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc;
  boost::shared_ptr<Monitor> monitor_ptr;


  std::vector<boost::shared_ptr<PreviousData>> data;

  std::vector<boost::shared_ptr<MatrixDouble>> grads_ptr;
  std::vector<boost::shared_ptr<VectorDouble>> values_ptr;
  std::vector<boost::shared_ptr<VectorDouble>> dots_ptr;

  boost::shared_ptr<ForcesAndSourcesCore> null;
};

const double ramp_t = 1.0;
const double sml = 0.0;
const double T = 2.0 * M_PI;

struct ExactFunction {
  double operator()(const double x, const double y, const double t) const {
    double g = sin(T * x) * sin(T * y);
    double val = 0;
    if (x > -sml) {
      val = 1.0 * g;
    } else {
      val = g;
    }
    if (t <= ramp_t) {
      return val * t;
    } else {
      return val * ramp_t;
    }
  }
};

struct ExactFunctionGrad {
  FTensor::Tensor1<double, 3> operator()(const double x, const double y,
                                         const double t) const {
    FTensor::Tensor1<double, 3> grad;
    double mx = -T * cos(T * x) * sin(T * y);
    double my = -T * sin(T * x) * cos(T * y);
    double hx, hy;
    if (x > -sml) {
      hx = 1.0 * mx;
      hy = 1.0 * my;
    } else {
      hx = mx;
      hy = my;
    }
    if (t <= ramp_t) {
      grad(0) = hx * t;
      grad(1) = hy * t;
    } else {
      grad(0) = hx * ramp_t;
      grad(1) = hy * ramp_t;
    }
    grad(2) = 0.0;
    return grad;
  }
};

struct ExactFunctionLap {
  double operator()(const double x, const double y, const double t) const {
    double glap = -2.0 * pow(T, 2) * sin(T * x) * sin(T * y);
    double lap;
    if (x > -sml) {
      lap = 1.0 * glap;
    } else {
      lap = glap;
    }
    if (t <= ramp_t) {
      return lap * t;
    } else {
      return lap * ramp_t;
    }
  }
};

struct ExactFunctionDot {
  double operator()(const double x, const double y, const double t) const {
    double gdot = sin(T * x) * sin(T * y);
    double dot;
    if (x > -sml) {
      dot = 1.0 * gdot;
    } else {
      dot = gdot;
    }
    if (t <= ramp_t) {
      return dot;
    } else {
      return 0;
    }
  }
};

// struct ReactionTerms {
//   void operator()(const double u1, const double u2, const double u3, BlockData &coeffs, VectorDouble &vec){
//     vec.resize(3, false);
//     vec.clear();
    
//     vec[0] = coeffs.rate[0] * u1 * (1.0 - coeffs.coef(0, 0) * u1 
//                                         - coeffs.coef(0, 1) * u2 
//                                         - coeffs.coef(0, 2) * u3);

//     vec[1] = coeffs.rate[1] * u1 * (1.0 - coeffs.coef(1, 0) * u1 
//                                         - coeffs.coef(1, 1) * u2 
//                                         - coeffs.coef(1, 2) * u3);
//     vec[2] = coeffs.rate[2] * (u1 * u2 + u3) * (1.0 - coeffs.coef(0, 0) * u1 
//                                         - coeffs.coef(0, 1) * u2 
//                                         - coeffs.coef(0, 2) * u3);                                  
//   }

// };
MoFEMErrorCode RDProblem::setup_system() {
  MoFEMFunctionBegin;
  CHKERR m_field.getInterface(simple_interface);
  CHKERR simple_interface->getOptions();
  CHKERR simple_interface->loadFile();
  MoFEMFunctionReturn(0);
}
MoFEMErrorCode RDProblem::add_fe(std::string field_name) {
  MoFEMFunctionBegin;
  CHKERR simple_interface->addDomainField(field_name, H1, AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple_interface->addDataField("ERROR", L2, AINSWORTH_LEGENDRE_BASE,
                                        1);
  // CHKERR simple_interface->addBoundaryField(field_name, H1,
  //                                           AINSWORTH_LEGENDRE_BASE, 1);
  CHKERR simple_interface->setFieldOrder(field_name, order);
  CHKERR simple_interface->setFieldOrder("ERROR", 0);

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
      block_map[id].B0 = 1e-2;
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION2") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 1e-1;
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION3") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 1e-3;
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION4") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 1e-1;
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION5") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 1e-1;
      block_map[id].block_id = id;
    }
  }
  MoFEMFunctionReturn(0);
}


MoFEMErrorCode RDProblem::set_initial_values(std::string field_name,
                                             int block_id, Range &surface, double &init_val) {
  MoFEMFunctionBegin;
  if (m_field.getInterface<MeshsetsManager>()->checkMeshset(block_id,
                                                            BLOCKSET)) {
    CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        block_id, BLOCKSET, 2, surface, true);
  }
  if (!surface.empty()) {
    Range surface_verts;
    CHKERR moab.get_connectivity(surface, surface_verts, false);
    CHKERR m_field.getInterface<FieldBlas>()->setField(
        init_val, MBVERTEX, surface_verts, field_name);
  }

    MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::update_slow_rhs(std::string mass_field,
                           boost::shared_ptr<VectorDouble> &values_ptr) {
  MoFEMFunctionBegin;
  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(mass_field, values_ptr));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::push_slow_rhs(std::string field_name,
                         boost::shared_ptr<PreviousData> &dataU,
                         boost::shared_ptr<PreviousData> &dataV ) {
  MoFEMFunctionBegin;

  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpAssembleSlowRhs(field_name, dataU, dataV, RhsU(),
                            RhsV(), stimulation_surface));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::update_vol_fe(boost::shared_ptr<Ele> &vol_ele,
                                        boost::shared_ptr<PreviousData> &data) {
  MoFEMFunctionBegin;
 
  vol_ele->getOpPtrVector().push_back(
      new OpCalculateInvJacForFace(data->invJac));
  vol_ele->getOpPtrVector().push_back(
      new OpSetInvJacH1ForFace(data->invJac));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::update_stiff_rhs(std::string field_name,
                            boost::shared_ptr<VectorDouble> &values_ptr,
                            boost::shared_ptr<MatrixDouble> &grads_ptr,
                            boost::shared_ptr<VectorDouble> &dots_ptr) {

  MoFEMFunctionBegin;

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(field_name, values_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarValuesDot(field_name, dots_ptr));
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldGradient<2>(field_name, grads_ptr));
  MoFEMFunctionReturn(0);
}
MoFEMErrorCode RDProblem::push_stiff_rhs(std::string field_name,
                                         boost::shared_ptr<PreviousData> &data,
                                         std::map<int, BlockData> &block_map) {
  MoFEMFunctionBegin;

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhs<2>(field_name, data, block_map, ExactFunction(),
                                ExactFunctionDot(), ExactFunctionLap()));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode
RDProblem::update_stiff_lhs(std::string field_name, 
                            boost::shared_ptr<VectorDouble> &values_ptr,
                            boost::shared_ptr<MatrixDouble> &grads_ptr) {
  MoFEMFunctionBegin;
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldValues(field_name, values_ptr));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpCalculateScalarFieldGradient<2>(field_name, grads_ptr));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::push_stiff_lhs(std::string field_name,
                                         boost::shared_ptr<PreviousData> &data,
                                         std::map<int, BlockData> &block_map) {
  MoFEMFunctionBegin;

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleStiffLhs<2>(field_name, data, block_map));

  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::set_integration_rule() {
  MoFEMFunctionBegin;
  auto vol_rule = [](int, int, int p) -> int { return 2 * p; };
  vol_ele_slow_rhs->getRuleHook = vol_rule;

  vol_ele_stiff_rhs->getRuleHook = vol_rule;

  vol_ele_stiff_lhs->getRuleHook = vol_rule;
  boundary_ele_rhs->getRuleHook = vol_rule;
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::push_mass_ele(std::string field_name){
  MoFEMFunctionBegin;
  vol_mass_ele->getOpPtrVector().push_back(
      new OpAssembleMass(field_name, mass_matrix));
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::resolve_slow_rhs() {
  MoFEMFunctionBegin;
  CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                  vol_mass_ele);
  CHKERR MatAssemblyBegin(mass_matrix, MAT_FINAL_ASSEMBLY);
  CHKERR MatAssemblyEnd(mass_matrix, MAT_FINAL_ASSEMBLY);
  // Create and septup KSP (linear solver), we need this to calculate g(t,u) =
  // M^-1G(t,u)
  mass_Ksp = createKSP(m_field.get_comm());
  CHKERR KSPSetOperators(mass_Ksp, mass_matrix, mass_matrix);
  CHKERR KSPSetFromOptions(mass_Ksp);
  CHKERR KSPSetUp(mass_Ksp);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::set_fe_in_loop() {
  MoFEMFunctionBegin;
  CHKERR TSSetType(ts, TSARKIMEX);
  CHKERR TSARKIMEXSetType(ts, TSARKIMEXA2);

  CHKERR DMMoFEMTSSetIJacobian(dm, simple_interface->getDomainFEName(),
                               vol_ele_stiff_lhs, null, null);

  CHKERR DMMoFEMTSSetIFunction(dm, simple_interface->getDomainFEName(),
                               vol_ele_stiff_rhs, null, null);

  CHKERR DMMoFEMTSSetRHSFunction(dm, simple_interface->getDomainFEName(),
                                 vol_ele_slow_rhs, null, null);
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::post_proc_fields(std::string field_name) {
  MoFEMFunctionBegin;
  post_proc->addFieldValuesPostProc(field_name);
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
  std::vector<std::string> mass_names(nb_species);

  for(int i = 0; i < nb_species; ++i){
    mass_names[i] = "MASS" + boost::lexical_cast<std::string>(i+1);
  }
  CHKERR setup_system();
  for (int i = 0; i < nb_species; ++i) {
    add_fe(mass_names[i]);
  }

   

  CHKERR simple_interface->setUp();

  // for (_IT_CUBITMESHSETS_BY_SET_TYPE_FOR_LOOP_(m_field, BLOCKSET, it)) {
  //   string name = it->getName();
  //   if (name.compare(0, 14, "ESSENTIAL") == 0) {
  //     CHKERR it->getMeshsetIdEntitiesByDimension(m_field.get_moab(), 1,
  //                                                 bdry_ents, true);
  //   }
  // }
  Range surface;
  CHKERR moab.get_entities_by_type(0, MBTRI, surface, false);
  Skinner skin(&m_field.get_moab());
  Range edges;
  CHKERR skin.find_skin(0, surface, false, edges);
  Range edges_part;
  ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
  CHKERR pcomm->filter_pstatus(edges, PSTATUS_SHARED | PSTATUS_MULTISHARED,
                               PSTATUS_NOT, -1, &edges_part);
  Range edges_verts;
  CHKERR moab.get_connectivity(edges_part, edges_verts, false);

  CHKERR set_blockData(material_blocks);

  VectorDouble initVals;
  initVals.resize(3, false);
  initVals.clear();

  initVals[0] = 0.4;
  initVals[1] = 3;
  initVals[2] = 0.0;

  for (int i = 0; i < 1; ++i) {
    CHKERR set_initial_values(mass_names[i], i + 2, inner_surface[i], initVals[i]);
    CHKERR update_slow_rhs(mass_names[i], values_ptr[i]);
    }

  if (m_field.getInterface<MeshsetsManager>()->checkMeshset(3,
                                                            BLOCKSET)) {
    CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
        3, BLOCKSET, 2, stimulation_surface, true);
  }

    // if (nb_species == 1) {
    //   vol_ele_slow_rhs->getOpPtrVector().push_back(new OpComputeSlowValue(
    //       mass_names[0], data[0], data[0], data[0], material_blocks));
    // } else if(nb_species == 2){
    //   vol_ele_slow_rhs->getOpPtrVector().push_back(new OpComputeSlowValue(
    //       mass_names[0], data[0], data[1], data[1], material_blocks));
    // } else if(nb_species == 3){
    //   vol_ele_slow_rhs->getOpPtrVector().push_back(new OpComputeSlowValue(
    //       mass_names[0], data[0], data[1], data[2], material_blocks));
    // }
     

    for (int i = 0; i < nb_species; ++i) {
      CHKERR push_slow_rhs(mass_names[i], data[0], data[1]);
      CHKERR push_slow_rhs(mass_names[i], data[0], data[1]);
      // boundary_ele_rhs->getOpPtrVector().push_back(
      //     new OpAssembleNaturalBCRhs(mass_names[i], bdry_ents));
    }

    dm = simple_interface->getDM();
    

    
    // cout << "essential : " << essential_bdry_ents.empty() << endl;
    

    auto solve_for_g = [&]() {
      MoFEMFunctionBegin;
      if (vol_ele_slow_rhs->vecAssembleSwitch) {
        CHKERR VecGhostUpdateBegin(vol_ele_slow_rhs->ts_F, ADD_VALUES,
                                   SCATTER_REVERSE);
        CHKERR VecGhostUpdateEnd(vol_ele_slow_rhs->ts_F, ADD_VALUES,
                                 SCATTER_REVERSE);
        CHKERR VecAssemblyBegin(vol_ele_slow_rhs->ts_F);
        CHKERR VecAssemblyEnd(vol_ele_slow_rhs->ts_F);
        *vol_ele_slow_rhs->vecAssembleSwitch = false;
      }
      CHKERR KSPSolve(mass_Ksp, vol_ele_slow_rhs->ts_F, vol_ele_slow_rhs->ts_F);
      MoFEMFunctionReturn(0);
    };
    // Add hook to the element to calculate g.
    bdry_ents = unite(edges_verts, edges_part);
    
    // CHKERR m_field.getInterface<ProblemsManager>()->removeDofsOnEntities(
    //     simple_interface->getProblemName(), mass_names[0], bdry_ents);

    CHKERR DMCreateMatrix_MoFEM(dm, mass_matrix);
    CHKERR MatZeroEntries(mass_matrix);

    for (int i = 0; i < nb_species; ++i) {
      CHKERR push_mass_ele(mass_names[i]);
      }
      
      CHKERR resolve_slow_rhs();

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
    // vol_ele_stiff_lhs->getOpPtrVector().push_back(
    //   new OpError(ExactFunction(), ExactFunctionLap(), ExactFunctionGrad(), data[0], material_blocks, global_error));
      
    CHKERR set_integration_rule();

        ts = createTS(m_field.get_comm());

        vol_ele_slow_rhs->postProcessHook = solve_for_g;

        CHKERR set_fe_in_loop();

        post_proc->generateReferenceElementMesh(); // only once

        for (int i = 0; i < nb_species; ++i) {
          CHKERR post_proc_fields(mass_names[i]);
        }
        post_proc->addFieldValuesPostProc("ERROR");
      
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
    CHKERR PetscOptionsGetInt(PETSC_NULL, "", "-order", &order, PETSC_NULL);
    int nb_species = 2;
    RDProblem reac_diff_problem(mb_instance, core, order, nb_species);
    CHKERR reac_diff_problem.run_analysis();
  }
  CATCH_ERRORS;
  MoFEM::Core::Finalize();
  return 0;
  }