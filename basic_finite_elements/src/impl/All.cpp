/** \file All.cpp
  \ingroup Header file for basic finite elements implementation
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

#include <ArcLengthTools.cpp>
#include <ConstrainMatrixCtx.cpp>
#include <DirichletBC.cpp>
#include <EdgeForce.cpp>
#include <NodeForce.cpp>
#include <PCMGSetUpViaApproxOrders.cpp>
#include <PostProcOnRefMesh.cpp>
#include <SurfacePressure.cpp>
#include <ThermalElement.cpp>
#include <FluidPressure.cpp>
#include <SurfacePressureComplexForLazy.cpp>
#include <AnalyticalDirichlet.cpp>
#include <HookeElement.cpp>
#include <SpringElement.cpp>
#include <SimpleRodElement.cpp>


#ifdef WITH_ADOL_C
  #include <NonLinearElasticElement.cpp>
  #include <ConvectiveMassElement.cpp>
#endif // WITH_ADOL_C
