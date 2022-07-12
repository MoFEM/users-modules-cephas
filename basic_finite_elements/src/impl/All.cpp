/** \file All.cpp
  \ingroup Header file for basic finite elements implementation
*/

/* MIT License
 *
 * Copyright (c) 2022
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

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
#include <NavierStokesElement.cpp>
#include <SimpleRodElement.cpp>


#ifdef WITH_ADOL_C
  #include <NonLinearElasticElement.cpp>
  #include <ConvectiveMassElement.cpp>
#endif // WITH_ADOL_C
