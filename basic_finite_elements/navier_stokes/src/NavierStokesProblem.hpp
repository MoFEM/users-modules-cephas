/**
 * \file NavierStokesProblem.hpp
 * \example NavierStokesProblem.hpp
 *
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

#ifndef __NAVIER_STOKES_PROBLEM_HPP__
#define __NAVIER_STOKES_PROBLEM_HPP__

//namespace NavierStokesProblem {

struct FatPrism : public MoFEM::FatPrismElementForcesAndSourcesCore {
  FatPrism(MoFEM::Interface &m_field)
      : MoFEM::FatPrismElementForcesAndSourcesCore(m_field) {}
  int getRuleTrianglesOnly(int order) { return 2 * (order + 0); }
  int getRuleThroughThickness(int order) { return 2 * (order + 1); }
};

//} // namespace NavierStokesProblem

#endif //__NAVIER_STOKES_PROBLEM_HPP__
