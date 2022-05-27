/**
 * \file poisson_2d_homogeneous.hpp
 * \example poisson_2d_homogeneous.hpp
 *
 * Solution of poisson equation. Direct implementation of User Data Operators
 * for teaching proposes.
 *
 * \note In practical application we suggest use form integrators to generalise
 * and simplify code. However, here we like to expose user to ways how to
 * implement data operator from scratch.
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


// Define name if it has not been defined yet
#ifndef __RANDOM_MESH_REFINMENT_HPP__
#define __RANDOM_MESH_REFINMENT_HPP__

namespace Poisson2DHomogeneousOperators {

using DomainParentEle = FaceElementForcesAndSourcesCoreOnChildParent;

/**
 * @brief set bit
 *
 */
auto bit = [](auto l) { return BitRefLevel().set(l); };

/**
 * @brief set bit to marker
 *
 * Marker is used to mark field entities on skin on which we have hanging nodes
 */
auto marker = [](auto l) {
  return BitRefLevel().set(BITREFLEVEL_SIZE - 1 - l);
};

/**
 * @brief lambda function used to select elements on which finite element
 * pipelines are executed.
 *
 * @note childs elements on pipeline, retrive data from parents using operators
 * pushed by \ref set_parent_dofs
 *
 */
auto test_bit_child = [](FEMethod *fe_ptr) {
  return fe_ptr->numeredEntFiniteElementPtr->getBitRefLevel().test(
      nb_ref_levels);
};

/**
 * @brief set levels of projection operators, which project field data from
 * parent entities, to child, up to to level, i.e. last mesh refinement.
 *
 */
auto set_parent_dofs = [](auto &m_field, auto &fe_top, auto op, auto verbosity,
                          auto sev) {
  auto jac_ptr = boost::make_shared<MatrixDouble>();
  auto inv_jac_ptr = boost::make_shared<MatrixDouble>();
  auto det_ptr = boost::make_shared<VectorDouble>();

  BitRefLevel bit_marker;
  for (auto l = 1; l <= nb_ref_levels; ++l)
    bit_marker |= marker(l);

  boost::function<void(boost::shared_ptr<ForcesAndSourcesCore>, int)>
      add_parent_level =
          [&](boost::shared_ptr<ForcesAndSourcesCore> parent_fe_pt, int level) {
            if (level > 0) {

              auto fe_ptr_current = boost::shared_ptr<ForcesAndSourcesCore>(
                  new DomainParentEle(m_field));
              if (op == OpFaceEle::OPSPACE) {
                fe_ptr_current->getOpPtrVector().push_back(
                    new OpCalculateHOJacForFace(jac_ptr));
                fe_ptr_current->getOpPtrVector().push_back(
                    new OpInvertMatrix<2>(jac_ptr, det_ptr, inv_jac_ptr));
                fe_ptr_current->getOpPtrVector().push_back(
                    new OpSetInvJacH1ForFace(inv_jac_ptr));
              }

              add_parent_level(
                  boost::dynamic_pointer_cast<ForcesAndSourcesCore>(
                      fe_ptr_current),
                  level - 1);

              if (op == OpFaceEle::OPSPACE) {

                parent_fe_pt->getOpPtrVector().push_back(

                    new OpAddParentEntData(

                        H1, op, fe_ptr_current,

                        BitRefLevel().set(), bit(0).flip(),

                        bit_marker, BitRefLevel().set(),

                        verbosity, sev));

              } else {

                parent_fe_pt->getOpPtrVector().push_back(

                    new OpAddParentEntData(

                        field_name, op, fe_ptr_current,

                        BitRefLevel().set(), bit(0).flip(),

                        bit_marker, BitRefLevel().set(),

                        verbosity, sev));
              }
            }
          };

  add_parent_level(boost::dynamic_pointer_cast<ForcesAndSourcesCore>(fe_top),
                   nb_ref_levels);
};

auto random_mesh_refine = [](MoFEM::Interface &m_field) {
  MoFEMFunctionBegin;
  constexpr int SPACE_DIM = 2;

  auto &moab = m_field.get_moab();
  ParallelComm *pcomm =
      ParallelComm::get_pcomm(&m_field.get_moab(), MYPCOMM_INDEX);
  Skinner skin(&moab);

  auto bit_mng = m_field.getInterface<BitRefManager>();

  Range level0_ents;
  CHKERR m_field.getInterface<BitRefManager>()->getEntitiesByDimAndRefLevel(
      bit(0), BitRefLevel().set(), SPACE_DIM, level0_ents);
  Range level0_skin;
  CHKERR skin.find_skin(0, level0_ents, false, level0_skin);
  CHKERR pcomm->filter_pstatus(level0_skin,
                               PSTATUS_SHARED | PSTATUS_MULTISHARED,
                               PSTATUS_NOT, -1, nullptr);

  auto refine_mesh = [&](auto l) {
    MoFEMFunctionBegin;

    auto refine = m_field.getInterface<MeshRefinement>();

    auto meshset_level0_ptr = get_temp_meshset_ptr(moab);
    CHKERR bit_mng->getEntitiesByDimAndRefLevel(bit(l - 1), BitRefLevel().set(),
                                                SPACE_DIM, *meshset_level0_ptr);

    // random mesh refinement
    auto meshset_ref_edges_ptr = get_temp_meshset_ptr(moab);

    Range els;
    CHKERR moab.get_entities_by_dimension(*meshset_level0_ptr, SPACE_DIM, els);
    CHKERR bit_mng->filterEntitiesByRefLevel(bit(l - 1), bit(l - 1), els);

    Range ele_to_refine;

    if (l == 1) {
      int ii = 0;
      for (auto t : els) {
        if ((ii % 2) == 0) {
          ele_to_refine.insert(t);
          std::vector<EntityHandle> adj_edges;
          CHKERR m_field.get_moab().get_adjacencies(&t, 1, SPACE_DIM - 1, false,
                                                    adj_edges);
          CHKERR moab.add_entities(*meshset_ref_edges_ptr, &*adj_edges.begin(),
                                   adj_edges.size());
        }
        ++ii;
      }
    } else {
      Range level_skin;
      CHKERR skin.find_skin(0, els, false, level_skin);
      CHKERR pcomm->filter_pstatus(level_skin,
                                   PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                   PSTATUS_NOT, -1, nullptr);
      level_skin = subtract(level_skin, level0_skin);
      Range adj;
      CHKERR m_field.get_moab().get_adjacencies(level_skin, SPACE_DIM, false,
                                                adj, moab::Interface::UNION);
      els = subtract(els, adj);
      ele_to_refine.merge(els);
      Range adj_edges;
      CHKERR m_field.get_moab().get_adjacencies(
          els, SPACE_DIM - 1, false, adj_edges, moab::Interface::UNION);
      CHKERR moab.add_entities(*meshset_ref_edges_ptr, adj_edges);
    }

    CHKERR refine->addVerticesInTheMiddleOfEdges(*meshset_ref_edges_ptr, bit(l),
                                                 false, VERBOSE);
    CHKERR refine->refineTrisHangingNodes(*meshset_level0_ptr, bit(l), VERBOSE);
    CHKERR bit_mng->updateRangeByChildren(level0_skin, level0_skin);
    CHKERR m_field.getInterface<MeshsetsManager>()
        ->updateAllMeshsetsByEntitiesChildren(bit(l));

    CHKERR bit_mng->writeBitLevelByDim(
        bit(l), BitRefLevel().set(), SPACE_DIM,
        (boost::lexical_cast<std::string>(l) + "_ref_mesh.vtk").c_str(), "VTK",
        "");
    CHKERR bit_mng->writeBitLevelByDim(
        bit(l), bit(l), MBTRI,
        (boost::lexical_cast<std::string>(l) + "_only_ref_mesh.vtk").c_str(),
        "VTK", "");

    MoFEMFunctionReturn(0);
  };

  auto mark_skins = [&](auto l, auto m) {
    MoFEMFunctionBegin;
    Range ents;
    CHKERR bit_mng->getEntitiesByDimAndRefLevel(bit(l), bit(l), SPACE_DIM,
                                                ents);
    Range level_skin;
    CHKERR skin.find_skin(0, ents, false, level_skin);
    CHKERR pcomm->filter_pstatus(level_skin,
                                 PSTATUS_SHARED | PSTATUS_MULTISHARED,
                                 PSTATUS_NOT, -1, nullptr);
    level_skin = subtract(level_skin, level0_skin);
    CHKERR m_field.get_moab().get_adjacencies(level_skin, 0, false, level_skin,
                                              moab::Interface::UNION);
    CHKERR bit_mng->addBitRefLevel(level_skin, marker(m));
    MoFEMFunctionReturn(0);
  };

  BitRefLevel bit_sum;
  for (auto l = 0; l != nb_ref_levels; ++l) {
    CHKERR refine_mesh(l + 1);
    CHKERR mark_skins(l, l + 1);
    CHKERR mark_skins(l + 1, l + 1);
    bit_sum |= bit(l);
  }
  bit_sum |= bit(nb_ref_levels);

  auto simple = m_field.getInterface<Simple>();
  simple->getBitRefLevel() = bit_sum;
  simple->getBitRefLevelMask() = BitRefLevel().set();

  // Simple interface will resolve adjacency to DOFs of parent of the element.
  // Using that information MAtrixManager  allocate appropriately size of
  // matrix.
  simple->getParentAdjacencies() = true;
  BitRefLevel bit_marker;
  for (auto l = 1; l <= nb_ref_levels; ++l)
    bit_marker |= marker(l);
  simple->getBitAdjEnt() = bit_marker;

  MoFEMFunctionReturn(0);
};

auto remove_hanging_dofs = [](MoFEM::Interface &m_field) {
  MoFEMFunctionBegin;

  auto simple = m_field.getInterface<Simple>();
  auto prb_mng = m_field.getInterface<ProblemsManager>();

  // remove obsolete DOFs from problem
  for (int l = 0; l != nb_ref_levels; ++l) {
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), field_name,
                                         bit(l), bit(l));
    CHKERR prb_mng->removeDofsOnEntities(simple->getProblemName(), field_name,
                                         marker(l + 1), bit(l).flip());
  }

  MoFEMFunctionReturn(0);
};

}

#endif