#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <rd_stdOperators.hpp>

using namespace MoFEM;
using namespace StdRDOperators;

static char help[] = "...\n\n";

struct RDProblem {
public:
  RDProblem(moab::Core &mb_instance, MoFEM::Core &core, const int order)
      : moab(mb_instance), m_field(core), order(order) {
    vol_ele_slow_rhs = boost::shared_ptr<Ele>(new Ele(m_field));
    vol_ele_stiff_rhs = boost::shared_ptr<Ele>(new Ele(m_field));
    vol_ele_stiff_lhs = boost::shared_ptr<Ele>(new Ele(m_field));

    vol_mass_ele = boost::shared_ptr<Ele>(new Ele(m_field));

    post_proc = boost::shared_ptr<PostProcFaceOnRefinedMesh>(
        new PostProcFaceOnRefinedMesh(m_field));

    data1 = boost::shared_ptr<PreviousData>(new PreviousData());
    data2 = boost::shared_ptr<PreviousData>(new PreviousData());
    data3 = boost::shared_ptr<PreviousData>(new PreviousData());

    grads_ptr1 = boost::shared_ptr<MatrixDouble>(data1, &data1->grads);
    values_ptr1 = boost::shared_ptr<VectorDouble>(data1, &data1->values);
    dots_ptr1 = boost::shared_ptr<VectorDouble>(data1, &data1->dot_values);

    grads_ptr2 = boost::shared_ptr<MatrixDouble>(data2, &data2->grads);
    values_ptr2 = boost::shared_ptr<VectorDouble>(data2, &data2->values);
    dots_ptr2 = boost::shared_ptr<VectorDouble>(data2, &data2->dot_values);

    grads_ptr3 = boost::shared_ptr<MatrixDouble>(data3, &data3->grads);
    values_ptr3 = boost::shared_ptr<VectorDouble>(data3, &data3->values);
    dots_ptr3 = boost::shared_ptr<VectorDouble>(data3, &data3->dot_values);
  }

  // RDProblem(const int order) : order(order){}
  MoFEMErrorCode run_analysis(int nb_sp);

private:
  MoFEMErrorCode setup_system();
  MoFEMErrorCode add_fe(std::string field_name);
  MoFEMErrorCode set_blockData(std::map<int, BlockData> &block_data_map);

  MoFEMErrorCode set_initial_values(std::string field_name, int block_id,
                                    Range &surface);

  MoFEMErrorCode update_slow_rhs(std::string mass_fiedl,
                                 boost::shared_ptr<VectorDouble> &mass_ptr);

  MoFEMErrorCode push_slow_rhs(std::string field_name,
                               boost::shared_ptr<PreviousData> &data);

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

  Range inner_surface1; // nb_species times
  Range inner_surface2;
  Range inner_surface3;

  int order;
  int nb_species;

  std::map<int, BlockData> material_blocks;

  boost::shared_ptr<Ele> vol_ele_slow_rhs;
  boost::shared_ptr<Ele> vol_ele_stiff_rhs;
  boost::shared_ptr<Ele> vol_ele_stiff_lhs;

  boost::shared_ptr<Ele> vol_mass_ele;

      boost::shared_ptr<PostProcFaceOnRefinedMesh> post_proc;
  boost::shared_ptr<Monitor> monitor_ptr;

  boost::shared_ptr<PreviousData> data1; // nb_species times
  boost::shared_ptr<PreviousData> data2;
  boost::shared_ptr<PreviousData> data3;

  boost::shared_ptr<MatrixDouble> grads_ptr1; // nb_species times
  boost::shared_ptr<MatrixDouble> grads_ptr2;
  boost::shared_ptr<MatrixDouble> grads_ptr3;

  boost::shared_ptr<VectorDouble> values_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> values_ptr2;
  boost::shared_ptr<VectorDouble> values_ptr3;


  boost::shared_ptr<VectorDouble> dots_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> dots_ptr2;
  boost::shared_ptr<VectorDouble> dots_ptr3;

  boost::shared_ptr<ForcesAndSourcesCore> null;
};

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
  CHKERR simple_interface->setFieldOrder(field_name, order);
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
      block_map[id].B0 = 1e-3;
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION3") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 1e-3;
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION4") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 5e-3;
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION5") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 1e-2;
      block_map[id].block_id = id;
    }
  }
  MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::set_initial_values(std::string field_name,
                                             int block_id, Range &surface) {
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
        0.5, MBVERTEX, surface_verts, field_name);
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

MoFEMErrorCode RDProblem::push_slow_rhs(std::string field_name,
                                        boost::shared_ptr<PreviousData> &data) {
  MoFEMFunctionBegin;

  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpAssembleSlowRhs(field_name, data));

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
      new OpAssembleStiffRhs<2>(field_name, data, block_map));
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

MoFEMErrorCode RDProblem::run_analysis(int nb_sp) {
  MoFEMFunctionBegin;
  // set nb_species
  CHKERR setup_system(); // only once
  nb_species = nb_sp;
  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR add_fe("MASS1"); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR add_fe("MASS2");
      if (nb_species == 3) {
        CHKERR add_fe("MASS3");
      }
    }
  }

  CHKERR simple_interface->setUp();

  CHKERR set_blockData(material_blocks);

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR set_initial_values("MASS1", 2, inner_surface1);
    CHKERR update_slow_rhs("MASS1", values_ptr1);
    if (nb_species == 1) {
      vol_ele_slow_rhs->getOpPtrVector().push_back(new OpComputeSlowValue(
          "MASS1", data1, data1, data1, material_blocks));
    } else if (nb_species == 2 || nb_species == 3) {
      CHKERR set_initial_values("MASS2", 3, inner_surface2);
      CHKERR update_slow_rhs("MASS2", values_ptr2);
      if (nb_species == 2) {
        vol_ele_slow_rhs->getOpPtrVector().push_back(new OpComputeSlowValue(
            "MASS1", data1, data2, data2, material_blocks));
      } else if (nb_species == 3) {
        CHKERR set_initial_values("MASS3", 4, inner_surface3);
        CHKERR update_slow_rhs("MASS3", values_ptr3);
        vol_ele_slow_rhs->getOpPtrVector().push_back(new OpComputeSlowValue(
            "MASS1", data1, data2, data3, material_blocks));
      }
    }
  }

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR push_slow_rhs("MASS1", data1); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR push_slow_rhs("MASS2", data2);
      if (nb_species == 3) {
        CHKERR push_slow_rhs("MASS3", data3);
      }
    }
  }

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
  vol_ele_slow_rhs->postProcessHook = solve_for_g;

  dm = simple_interface->getDM();
  ts = createTS(m_field.get_comm());

  CHKERR DMCreateMatrix_MoFEM(dm, mass_matrix);
  CHKERR MatZeroEntries(mass_matrix);

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR push_mass_ele("MASS1"); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR push_mass_ele("MASS2");
      if (nb_species == 3) {
        CHKERR push_mass_ele("MASS3");
      }
    }
  }

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR resolve_slow_rhs(); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR resolve_slow_rhs();
      if (nb_species == 3) {
        CHKERR resolve_slow_rhs();
      }
    }
  }

  CHKERR update_vol_fe(vol_ele_stiff_rhs, data1);

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR update_stiff_rhs("MASS1", values_ptr1,
                            grads_ptr1, dots_ptr1);
    CHKERR push_stiff_rhs("MASS1", data1,
                          material_blocks); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR update_stiff_rhs("MASS2", values_ptr2,
                              grads_ptr2, dots_ptr2);
      CHKERR push_stiff_rhs("MASS2", data2, material_blocks);
      if (nb_species == 3) {
        CHKERR update_stiff_rhs("MASS3", values_ptr3,
                                grads_ptr3, dots_ptr3);
        CHKERR push_stiff_rhs("MASS3", data3, material_blocks);
      }
    }
  }

  CHKERR update_vol_fe(vol_ele_stiff_lhs, data1);

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR update_stiff_lhs("MASS1", values_ptr1,
                            grads_ptr1);
    CHKERR push_stiff_lhs("MASS1", data1,
                          material_blocks); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR update_stiff_lhs("MASS2", values_ptr2,
                              grads_ptr2);
      CHKERR push_stiff_lhs("MASS2", data2, material_blocks);
      if (nb_species == 3) {
        CHKERR update_stiff_lhs("MASS3", values_ptr3,
                                grads_ptr3);
        CHKERR push_stiff_lhs("MASS3", data3, material_blocks);
      }
    }
  }
  CHKERR set_integration_rule();
  


  CHKERR set_fe_in_loop();                          // only once
  post_proc->generateReferenceElementMesh(); // only once

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR post_proc_fields("MASS1");
    if (nb_species == 2 || nb_species == 3) {
      CHKERR post_proc_fields("MASS2");
      if (nb_species == 3) {
        CHKERR post_proc_fields("MASS3");
      }
    }
  }
  monitor_ptr = boost::shared_ptr<Monitor>(
      new Monitor(dm, post_proc)); // nb_species times
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

    int order = 3;
    int nb_species = 1;
    RDProblem reac_diff_problem(mb_instance, core, order);
    CHKERR reac_diff_problem.run_analysis(nb_species);
  }
  CATCH_ERRORS;
  MoFEM::Core::Finalize();
  return 0;
  }