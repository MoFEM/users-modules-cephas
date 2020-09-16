/**
 * \file lorentz_force.cpp
 * \example lorentz_force.cpp
 * \brief Calculate Lorentz fore from magnetic field.
 *
 * It is not attemt to have accurate or realistic model of moving particles in
 * magnetic field. This example was create to show how field evaluator works
 * with precalculated magnetic field.
 *
 * \todo use physical quantities
 * \todo take in account higher order geometry
 * \todo code panellisation, works serial only
 *
 * Exaltation
 * \f[
 *  v_i = (p_{i+1} - p_{i-1}) / (2 \delta t) \\
 *  (p_{i-1} - 2 p_i + p_{i+1}) / \delta t^2 = \frac{q}{m} v_i \times B_i \\
 *  (p_{i-1} - 2 p_i + p_{i+1}) / \delta t^2 = \frac{q}{m} (p_{i+1} - p_{i-1}) /
 * (2 \delta t)  \times B_i \\
 *  (p_{i-1} - 2 p_i + p_{i+1}) / \delta t = \frac{q}{m} (p_{i+1} - p_{i-1})
 * \times B_i / 2  \\
 *  p_{i+1} / \delta t - p_{i+1} \times B_i / 2= (2 p_i - p_{i-1}) / \delta t -
 * p_{i-1} \times B_i / 2 \\ p_{i+1} (\mathbf{1} / \delta t - \frac{q}{m}
 * \mathbf{1} \times B_i / 2 )= (2 p_i - p_{i-1}) / \delta t - \frac{q}{m}
 * p_{i-1} \times B_i / 2 \f]
 *
 * \ingroup maxwell_element
 */

/* This file is part of MoFEM.
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
#include <MagneticElement.hpp>
using namespace MoFEM;

static char help[] = "...\n\n";

int main(int argc, char *argv[]) {

  MoFEM::Core::Initialize(&argc, &argv, (char *)0, help);

  try {

    moab::Core mb_instance;
    moab::Interface &moab = mb_instance;

    ParallelComm *pcomm = ParallelComm::get_pcomm(&moab, MYPCOMM_INDEX);
    if (pcomm == NULL)
      pcomm = new ParallelComm(&moab, PETSC_COMM_WORLD);

    // Read parameters from line command
    PetscBool flg_file;
    char mesh_file_name[255];
    CHKERR PetscOptionsBegin(PETSC_COMM_WORLD, "", "Lorenz force configure",
                             "none");
    CHKERR PetscOptionsString("-my_file", "mesh file name", "", "solution.h5m",
                              mesh_file_name, 255, &flg_file);
    ierr = PetscOptionsEnd();
    CHKERRG(ierr);

    const char *option;
    option = "";
    CHKERR moab.load_file(mesh_file_name, 0, option);

    // Create mofem interface
    MoFEM::Core core(moab);
    MoFEM::Interface &m_field = core;

    CHKERR m_field.build_fields();
    CHKERR m_field.build_finite_elements();
    CHKERR m_field.build_adjacencies(BitRefLevel().set(0));

    // set up DM
    DM dm;
    CHKERR DMRegister_MoFEM("DMMOFEM");
    CHKERR DMCreate(PETSC_COMM_WORLD, &dm);
    CHKERR DMSetType(dm, "DMMOFEM");
    CHKERR DMMoFEMCreateMoFEM(dm, &m_field, "MAGNETIC_PROBLEM",
                              BitRefLevel().set(0));
    CHKERR DMSetFromOptions(dm);
    // add elements to blockData.dM
    CHKERR DMMoFEMAddElement(dm, "MAGNETIC");
    CHKERR DMSetUp(dm);

    using VolEle = VolumeElementForcesAndSourcesCore;
    using VolOp = VolumeElementForcesAndSourcesCore::UserDataOperator;
    // using SetPtsData = FieldEvaluatorInterface::SetPtsData;
    // using SetPts = FieldEvaluatorInterface::SetPts;

    /**
     * @brief Only for debuging
     */
    struct MyOpDebug : public VolOp {

      boost::shared_ptr<MatrixDouble> B;
      MyOpDebug(decltype(B) b) : VolOp("MAGNETIC_POTENTIAL", OPROW), B(b) {}

      MoFEMErrorCode doWork(int side, EntityType type,
                            DataForcesAndSourcesCore::EntData &data) {
        MoFEMFunctionBegin;
        if (type == MBEDGE && side == 0) {
          std::cout << "found " << (*B) << endl;
          std::cout << "data " << data.getFieldData() << endl;
        }

        MoFEMFunctionReturn(0);
      }
    };

    const double dist = 1e-12;             // Distance for tree search
    const int nb_random_points = 100;      // number of points
    const int nb_steps = 10000;            // number of time steps
    const int mod_step = 10;               // save every step
    const double dt = 1e-5;                // Time step size
    const double velocity_scale = 0.1;     // scale velocity vector
    const double magnetic_field_scale = 1; // scale magnetic field vector
    const double scale_box = 0.5; // scale box where partices are placed

    FieldEvaluatorInterface *field_eval_ptr;
    CHKERR m_field.getInterface(field_eval_ptr);

    // Get access to data
    auto data_at_pts = field_eval_ptr->getData<VolEle>();
    auto vol_ele = data_at_pts->feMethodPtr.lock();
    if (!vol_ele)
      SETERRQ(PETSC_COMM_SELF, MOFEM_DATA_INCONSISTENCY,
              "Pointer to element does not exists");
    auto get_rule = [&](int order_row, int order_col, int order_data) {
      return -1;
    };
    vol_ele->getRuleHook = get_rule;

    boost::shared_ptr<MatrixDouble> B(new MatrixDouble());
    vol_ele->getOpPtrVector().push_back(
        new OpCalculateHcurlVectorCurl<3>("MAGNETIC_POTENTIAL", B));
    // vol_ele->getOpPtrVector().push_back(new MyOpDebug(B));

    const MoFEM::Problem *prb_ptr;
    CHKERR DMMoFEMGetProblemPtr(dm, &prb_ptr);

    CHKERR field_eval_ptr->buildTree3D(data_at_pts, "MAGNETIC");
    BoundBox box;
    CHKERR data_at_pts->treePtr->get_bounding_box(box);

    const double bMin = box.bMin[0];
    const double bMax = box.bMax[0];

    auto create_vertices = [nb_random_points, bMin, bMax,
                            scale_box](moab::Interface &moab, auto &verts,
                                       auto &arrays_coord) {
      MoFEMFunctionBegin;
      ReadUtilIface *iface;
      EntityHandle startv;
      CHKERR moab.query_interface(iface);
      CHKERR iface->get_node_coords(3, nb_random_points, 0, startv,
                                    arrays_coord);
      FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3> t_coords = {
          arrays_coord[0], arrays_coord[1], arrays_coord[2]};
      FTensor::Index<'i', 3> i;
      verts = Range(startv, startv + nb_random_points - 1);
      for (int n = 0; n != nb_random_points; ++n) {
        t_coords(0) = 0;
        for (auto ii : {1, 2}) {
          t_coords(ii) = scale_box * (bMax - bMin) *
                             (std::rand() / static_cast<double>(RAND_MAX)) -
                         bMax * scale_box;
        }
        ++t_coords;
      }
      MoFEMFunctionReturn(0);
    };

    auto set_positions = [nb_random_points, dt, velocity_scale](
                             moab::Interface &moab, auto &arrays_coord) {
      MatrixDouble init_pos(3, nb_random_points);
      FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3> t_coords = {
          arrays_coord[0], arrays_coord[1], arrays_coord[2]};
      FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3> t_init_coords = {
          &init_pos(0, 0), &init_pos(1, 0), &init_pos(2, 0)};
      FTensor::Index<'i', 3> i;
      for (int n = 0; n != nb_random_points; ++n) {

        FTensor::Tensor1<double, 3> t_velocity;
        for (auto ii : {0, 1, 2})
          t_velocity(ii) = (rand() / static_cast<double>(RAND_MAX) - 0.5);
        t_velocity(i) /= sqrt(t_velocity(i) * t_velocity(i));
        t_init_coords(i) = t_coords(i) + dt * velocity_scale * t_velocity(i);

        ++t_coords;
        ++t_init_coords;
      }
      return init_pos;
    };

    moab::Core mb_charged_partices;
    moab::Interface &moab_charged_partices = mb_charged_partices;
    vector<double *> arrays_coord;
    Range verts;
    CHKERR create_vertices(moab_charged_partices, verts, arrays_coord);
    auto init_positions = set_positions(moab, arrays_coord);

    auto get_t_coords = [&]() {
      return FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3>(
          arrays_coord[0], arrays_coord[1], arrays_coord[2]);
    };

    auto get_t_init_coords = [&]() {
      return FTensor::Tensor1<FTensor::PackPtr<double *, 1>, 3>(
          &init_positions(0, 0), &init_positions(1, 0), &init_positions(2, 0));
    };

    auto calc_rhs = [&](auto &t_p, auto &t_init_p, auto &t_B) {
      FTensor::Tensor1<double, 3> t_rhs;
      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      t_rhs(k) = (2 * t_p(k) - t_init_p(k)) / dt -
                 levi_civita(j, i, k) * t_init_p(i) * t_B(j);
      return t_rhs;
    };

    auto calc_lhs = [&](auto &t_B) {
      FTensor::Tensor2<double, 3, 3> t_lhs;
      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;
      FTensor::Index<'k', 3> k;
      t_lhs(i, k) = levi_civita(j, i, k) * (-t_B(j));
      for (auto ii : {0, 1, 2})
        t_lhs(ii, ii) += 1 / dt;
      return t_lhs;
    };

    // auto set_periodicity = [&](auto &t_p, auto &t_init_p) {
    //   for (int i : {0, 1, 2})
    //     if (t_p(i) > bMax) {
    //       t_p(i) -= 2 * bMax;
    //       t_init_p(i) -= 2 * bMax;
    //     } else if (t_p(i) < bMin) {
    //       t_p(i) -= 2 * bMin;
    //       t_init_p(i) -= 2 * bMin;
    //     }
    // };

    auto is_out = [&](auto &t_p) {
      for (int i : {0, 1, 2})
        if (t_p(i) > bMax) {
          return true;
        } else if (t_p(i) < bMin) {
          return true;
        }
      return false;
    };

    auto calc_position = [&]() {
      auto t_p = get_t_coords();
      auto t_init_p = get_t_init_coords();
      FTensor::Index<'i', 3> i;
      FTensor::Index<'j', 3> j;

      for (int n = 0; n != nb_random_points; ++n) {

        if (is_out(t_p)) {
          ++t_p;
          ++t_init_p;
          continue;
        }

        std::array<double, 3> point = {t_p(0), t_p(1), t_p(2)};
        data_at_pts->setEvalPoints(point.data(), 1);

        CHKERR field_eval_ptr->evalFEAtThePoint3D(
            point.data(), dist, prb_ptr->getName(), "MAGNETIC", data_at_pts,
            m_field.get_comm_rank(), m_field.get_comm_rank(), MF_EXIST, QUIET);

        FTensor::Tensor1<double, 3> t_B;

        if (B->size2())
          for (int ii : {0, 1, 2})
            t_B(ii) = (*B)(ii, 0);
        else
          t_B(i) = 0;

        t_B(i) *= magnetic_field_scale * 0.5;

        auto t_rhs = calc_rhs(t_p, t_init_p, t_B);
        auto t_lhs = calc_lhs(t_B);

        double det;
        CHKERR determinantTensor3by3(t_lhs, det);
        FTensor::Tensor2<double, 3, 3> t_inv_lhs;
        CHKERR invertTensor3by3(t_lhs, det, t_inv_lhs);

        t_init_p(i) = t_p(i);
        t_p(i) = t_inv_lhs(i, j) * t_rhs(j);

        // set_periodicity(t_p, t_init_p);

        ++t_p;
        ++t_init_p;
      }
    };

    for (int t = 0; t != nb_steps; ++t) {

      std::string print_step =
          "Step : " + boost::lexical_cast<std::string>(t) + "\r";
      std::cout << print_step << std::flush;

      calc_position();

      if ((t % mod_step) == 0) {
        // write points
        CHKERR moab_charged_partices.write_file(
            ("step_" + boost::lexical_cast<std::string>(t / mod_step) + ".vtk")
                .c_str(),
            "VTK", "");
      }
    }

    std::cout << endl;

    CHKERR DMDestroy(&dm);
  }
  CATCH_ERRORS;

  MoFEM::Core::Finalize();

  return 0;
}
