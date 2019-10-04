#include <stdlib.h>
#include <BasicFiniteElements.hpp>
#include <RDOperators.hpp>

using namespace MoFEM;
using namespace ReactionDiffusion;

static char help[] = "...\n\n";

// #define M_PI 3.14159265358979323846 /* pi */

struct RDProblem {
public:
  RDProblem(MoFEM::Core &core, const int order) : m_field(core), order(order) {
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

    flux_values_ptr3 =
        boost::shared_ptr<MatrixDouble>(data3, &data3->flux_values);
    flux_divs_ptr3 = boost::shared_ptr<VectorDouble>(data3, &data3->flux_divs);

    mass_values_ptr3 =
        boost::shared_ptr<VectorDouble>(data3, &data3->mass_values);

    mass_dots_ptr3 = boost::shared_ptr<VectorDouble>(data3, &data3->mass_dots);
  }

  // RDProblem(const int order) : order(order){}
  MoFEMErrorCode run_analysis(int nb_sp);

private:
  MoFEMErrorCode setup_system();
  MoFEMErrorCode add_fe(std::string mass_field, std::string flux_field);
  MoFEMErrorCode set_blockData(std::map<int, BlockData> &block_data_map);
  MoFEMErrorCode extract_bd_ents(std::string ESSENTIAL, std::string NATURAL);
  MoFEMErrorCode extract_initial_ents(int block_id, Range &surface);
  MoFEMErrorCode update_slow_rhs(std::string mass_fiedl,
                                 boost::shared_ptr<VectorDouble> &mass_ptr);
  MoFEMErrorCode push_slow_rhs(std::string mass_field, std::string flux_field,
                               boost::shared_ptr<PreviousData> &data);
  MoFEMErrorCode update_vol_fe(boost::shared_ptr<FaceEle> &vol_ele,
                               boost::shared_ptr<PreviousData> &data);
  MoFEMErrorCode
  update_stiff_rhs(std::string mass_field, std::string flux_field,
                   boost::shared_ptr<VectorDouble> &mass_ptr,
                   boost::shared_ptr<MatrixDouble> &flux_ptr,
                   boost::shared_ptr<VectorDouble> &mass_dot_ptr,
                   boost::shared_ptr<VectorDouble> &flux_div_ptr);
  MoFEMErrorCode push_stiff_rhs(std::string mass_field, std::string flux_field,
                                boost::shared_ptr<PreviousData> &data,
                                std::map<int, BlockData> &block_map);
  MoFEMErrorCode update_stiff_lhs(std::string mass_fiedl,
                                  std::string flux_field,
                                  boost::shared_ptr<VectorDouble> &mass_ptr,
                                  boost::shared_ptr<MatrixDouble> &flux_ptr);
  MoFEMErrorCode push_stiff_lhs(std::string mass_field, std::string flux_field,
                                boost::shared_ptr<PreviousData> &data,
                                std::map<int, BlockData> &block_map);

  MoFEMErrorCode set_integration_rule();
  MoFEMErrorCode apply_IC(std::string mass_field, Range &surface,
                          boost::shared_ptr<FaceEle> &initial_ele);
  MoFEMErrorCode apply_BC(std::string flux_field);
  MoFEMErrorCode loop_fe();
  MoFEMErrorCode post_proc_fields(std::string mass_field,
                                  std::string flux_field);
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
  boost::shared_ptr<PreviousData> data3;

  boost::shared_ptr<MatrixDouble> flux_values_ptr1; // nb_species times
  boost::shared_ptr<MatrixDouble> flux_values_ptr2;
  boost::shared_ptr<MatrixDouble> flux_values_ptr3;

  boost::shared_ptr<VectorDouble> flux_divs_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> flux_divs_ptr2;
  boost::shared_ptr<VectorDouble> flux_divs_ptr3;

  boost::shared_ptr<VectorDouble> mass_values_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> mass_values_ptr2;
  boost::shared_ptr<VectorDouble> mass_values_ptr3;

  boost::shared_ptr<VectorDouble> mass_dots_ptr1; // nb_species times
  boost::shared_ptr<VectorDouble> mass_dots_ptr2;
  boost::shared_ptr<VectorDouble> mass_dots_ptr3;

  boost::shared_ptr<ForcesAndSourcesCore> null;
};

struct ExactFunction {
double operator()(const double x, const double y, const double t) const {
  return sin(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * t);
}
};

struct ExactFunctionGrad{
FTensor::Tensor1<double, 3> operator()(const double x, const double y, const double t) const {
FTensor::Tensor1<double, 3> grad;
grad(0) = 2 * M_PI * cos(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * t);
grad(1) = 2 * M_PI * sin(2 * M_PI * x) * cos(2 * M_PI * y) * sin(2 * M_PI * t);
grad(2) = 0.0;
return grad;
}
};

struct ExactFunctionLap{
  double operator()(const double x, const double y, const double t) const {
    return -8 * pow(M_PI, 2) * sin(2 * M_PI * x) * sin(2 * M_PI * y) *
           sin(2 * M_PI * t);
  }
};

struct ExactFunctionDot{
  double operator()(const double x, const double y, const double t) const {
    return 2 * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y) * cos(2 * M_PI * t);
  }
};

  MoFEMErrorCode RDProblem::setup_system() {
    MoFEMFunctionBegin;
    CHKERR m_field.getInterface(simple_interface);
    CHKERR simple_interface->getOptions();
    CHKERR simple_interface->loadFile();
    MoFEMFunctionReturn(0);
}

MoFEMErrorCode RDProblem::add_fe(std::string mass_field,
                                 std::string flux_field) {
  MoFEMFunctionBegin;
  CHKERR simple_interface->addDomainField(mass_field, L2,
                                          AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple_interface->addDomainField(flux_field, HCURL,
                                          AINSWORTH_LEGENDRE_BASE, 1);

  CHKERR simple_interface->addBoundaryField(flux_field, HCURL,
                                            DEMKOWICZ_JACOBI_BASE, 1);
  CHKERR simple_interface->addDataField("ERROR", L2, AINSWORTH_LEGENDRE_BASE,
                                        1);

  CHKERR simple_interface->setFieldOrder(mass_field, order - 1);
  CHKERR simple_interface->setFieldOrder(flux_field, order);
  CHKERR simple_interface->setFieldOrder("ERROR", 0); // approximation order for error

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
      block_map[id].B0 = 1e-0;
      block_map[id].block_id = id;
    } else if (name.compare(0, 14, "REGION2") == 0) {
      CHKERR m_field.getInterface<MeshsetsManager>()->getEntitiesByDimension(
          id, BLOCKSET, 2, block_map[id].block_ents, true);
      block_map[id].B0 = 5e-4;
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

MoFEMErrorCode RDProblem::push_slow_rhs(std::string mass_field,
                                        std::string flux_field,
                                        boost::shared_ptr<PreviousData> &data) {
  MoFEMFunctionBegin;

  vol_ele_slow_rhs->getOpPtrVector().push_back(
      new OpAssembleSlowRhsV(mass_field, data, ExactFunction(), ExactFunctionDot(), ExactFunctionLap()));

  natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
      new OpAssembleNaturalBCRhsTau(flux_field, natural_bdry_ents));

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
                                         boost::shared_ptr<PreviousData> &data,
                                         std::map<int, BlockData> &block_map) {
  MoFEMFunctionBegin;
  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsTau<3>(flux_field, data, block_map));

  vol_ele_stiff_rhs->getOpPtrVector().push_back(
      new OpAssembleStiffRhsV<3>(mass_field, data, ExactFunction(),
                                 ExactFunctionDot(), ExactFunctionLap()));
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
                                         boost::shared_ptr<PreviousData> &data,
                                         std::map<int, BlockData> &block_map) {
  MoFEMFunctionBegin;
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauTau<3>(flux_field, data, block_map));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsVV(mass_field));

  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpAssembleLhsTauV<3>(flux_field, mass_field, data, block_map));

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
                                   boost::shared_ptr<FaceEle> &initial_ele) {
  MoFEMFunctionBegin;
  initial_ele->getOpPtrVector().push_back(
      new OpInitialMass(mass_field, surface));
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

MoFEMErrorCode RDProblem::post_proc_fields(std::string mass_field,
                                           std::string flux_field) {
  MoFEMFunctionBegin;
  post_proc->addFieldValuesPostProc(mass_field);
  post_proc->addFieldValuesPostProc(flux_field);
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
    CHKERR add_fe("MASS1", "FLUX1"); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR add_fe("MASS2", "FLUX2");
      if (nb_species == 3) {
        CHKERR add_fe("MASS3", "FLUX3");
      }
    }
  }

  CHKERR simple_interface->setUp();

  CHKERR set_blockData(material_blocks);

  CHKERR extract_bd_ents("ESSENTIAL", "NATURAL"); // nb_species times

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR extract_initial_ents(2, inner_surface1);
    CHKERR update_slow_rhs("MASS1", mass_values_ptr1);
    if (nb_species == 1) {
      vol_ele_slow_rhs->getOpPtrVector().push_back(new OpComputeSlowValue(
          "MASS1", data1, data1, data1, material_blocks));
    } else if (nb_species == 2 || nb_species == 3) {
      CHKERR extract_initial_ents(3, inner_surface2);
      CHKERR update_slow_rhs("MASS2", mass_values_ptr2);
      if (nb_species == 2) {
        vol_ele_slow_rhs->getOpPtrVector().push_back(new OpComputeSlowValue(
            "MASS1", data1, data2, data2, material_blocks));
      } else if (nb_species == 3) {
        CHKERR extract_initial_ents(4, inner_surface3);
        CHKERR update_slow_rhs("MASS3", mass_values_ptr3);
        vol_ele_slow_rhs->getOpPtrVector().push_back(new OpComputeSlowValue(
            "MASS1", data1, data2, data3, material_blocks));
      }
    }
  }
  natural_bdry_ele_slow_rhs->getOpPtrVector().push_back(
      new OpSetContrariantPiolaTransformOnEdge());

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR push_slow_rhs("MASS1", "FLUX1", data1); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR push_slow_rhs("MASS2", "FLUX2", data2);
      if (nb_species == 3) {
        CHKERR push_slow_rhs("MASS3", "FLUX3", data3);
      }
    }
  }

  CHKERR update_vol_fe(vol_ele_stiff_rhs, data1);

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR update_stiff_rhs("MASS1", "FLUX1", mass_values_ptr1,
                            flux_values_ptr1, mass_dots_ptr1, flux_divs_ptr1);
    CHKERR push_stiff_rhs("MASS1", "FLUX1", data1,
                          material_blocks); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR update_stiff_rhs("MASS2", "FLUX2", mass_values_ptr2,
                              flux_values_ptr2, mass_dots_ptr2, flux_divs_ptr2);
      CHKERR push_stiff_rhs("MASS2", "FLUX2", data2, material_blocks);
      if (nb_species == 3) {
        CHKERR update_stiff_rhs("MASS3", "FLUX3", mass_values_ptr3,
                                flux_values_ptr3, mass_dots_ptr3,
                                flux_divs_ptr3);
        CHKERR push_stiff_rhs("MASS3", "FLUX3", data3, material_blocks);
      }
    }
  }

  CHKERR update_vol_fe(vol_ele_stiff_lhs, data1);

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR update_stiff_lhs("MASS1", "FLUX1", mass_values_ptr1,
                            flux_values_ptr1);
    CHKERR push_stiff_lhs("MASS1", "FLUX1", data1,
                          material_blocks); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR update_stiff_lhs("MASS2", "FLUX2", mass_values_ptr2,
                              flux_values_ptr2);
      CHKERR push_stiff_lhs("MASS2", "FLUX2", data2, material_blocks);
      if (nb_species == 3) {
        CHKERR update_stiff_lhs("MASS3", "FLUX3", mass_values_ptr3,
                                flux_values_ptr3);
        CHKERR push_stiff_lhs("MASS3", "FLUX3", data3, material_blocks);
      }
    }
  }
  vol_ele_stiff_lhs->getOpPtrVector().push_back(
      new OpError(ExactFunction(), ExactFunctionLap(), data1));
  CHKERR set_integration_rule();
  dm = simple_interface->getDM();
  ts = createTS(m_field.get_comm());
  boost::shared_ptr<FaceEle> initial_mass_ele(new FaceEle(m_field));

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR apply_IC("MASS1", inner_surface1,
                    initial_mass_ele); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR apply_IC("MASS2", inner_surface2, initial_mass_ele);
      if (nb_species == 3) {
        CHKERR apply_IC("MASS3", inner_surface3, initial_mass_ele);
      }
    }
  }
  CHKERR DMoFEMLoopFiniteElements(dm, simple_interface->getDomainFEName(),
                                  initial_mass_ele);

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR apply_BC("MASS1"); // nb_species times
    if (nb_species == 2 || nb_species == 3) {
      CHKERR apply_BC("MASS2");
      if (nb_species == 3) {
        CHKERR apply_BC("MASS3");
      }
    }
  }

  CHKERR loop_fe();                          // only once
  post_proc->generateReferenceElementMesh(); // only once

  if (nb_species == 1 || nb_species == 2 || nb_species == 3) {
    CHKERR post_proc_fields("MASS1", "FLUX1");
    post_proc->addFieldValuesPostProc("ERROR");
    if (nb_species == 2 || nb_species == 3) {
      CHKERR post_proc_fields("MASS2", "FLUX2");
      if (nb_species == 3) {
        CHKERR post_proc_fields("MASS3", "FLUX3");
      }
    }
  }
  // int step = 0;
  // auto increment = [&]() -> int {
  //   MoFEMFunctionBegin;
  //   ++step;
  //   cout << "step : " << step << endl;
  //   MoFEMFunctionReturn(0);
  // };

  // vol_ele_slow_rhs->postProcessHook = increment;

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
    RDProblem reac_diff_problem(core, order);
    CHKERR reac_diff_problem.run_analysis(nb_species);
  }
  CATCH_ERRORS;
  MoFEM::Core::Finalize();
  return 0;
}