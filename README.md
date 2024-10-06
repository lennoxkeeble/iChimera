# GRSuite
Julia code which computes gravitational waveforms of extreme-mass-ratio inspirals (EMRIs) based on the "Chimera": a local kludge scheme introduced in <a href="https://arxiv.org/abs/1109.0572">Sopuerta & Yunes, 2011</a> (hereafter Ref. [2]). See the example notebook <em>InspiralExamples.ipynb</em> for code which computes EMRIs based on the Chimera, and <em>GeodesicExamples.ipynb</em> for code which computes timelike Kerr geodesics and their associated constants of motion. Below we provide more detail about the contents of this package.

## Components ##

* **Geodesics:** example code which numerically solves the first-order Hamilton-Jacobi equations governing timelike geodesic motion in Kerr can be found in the <em>GeodesicExamples.ipynb</em> notebook. Conversions between orbital parameterizations $(E, L_{z}, Q)$ and $(p, e, θ_\text{min})$ based on the methods in Refs. [1-4] can also be found in this example notebook.

* **Gravitational Waveforms:** the main goal of this package is to efficiently implement the Chimera kludge scheme introduced in Ref. [2] for generating gravitational waveforms of EMRIs based on a local, non-adiabatic approximation of the gravitational self-force. The Chimera employs the method of osculating orbits to construct the overall EMRI trajectory from several piecewise geodesics. At the end of each geodesic, the self-force is computed using expressions in Ref. [2], after which the constants of motion are updated and the next piecewise geodesic is evolved. We provide example code in <em>InspiralExamples.ipynb</em> which computes the constants of motions and gravitational waveform of an EMRI evolved using the Chimera. See below for futher remarks on our implementation, as well as a brief summary of the approach taken to derive the equations behind the Chimera.

## Important Remarks ##
* **Efficiency:** we have not focused on fully optimizing the efficiency of our code, and, as such, there is room to improve the typical runtime of the Chimera. Nonetheless, Julia's multi-threading capabilities has allowed for a significant increase in the speed of the Chimera compared to the original implementation by the authors in Ref. [2]. In particular, we expect that a year long inspiral can be evolved in ~10 hours using this code. The local nature of the Chimera makes it markedly more expensive than, for example, EMRIs evolved using adiabatic fluxes---<b>we don't expect a fully optimized version of our code to be anywhere close in speed to such models</b>. The computational expense of the Chimera essentially boils down to the combination of the following two things: (1) the self-force expressions contain several high-order time derivatives which we approximate using computationally expensive fitting techniques, making a single computation of fluxes much more expensive than evaluating closed-form expressions analytically; (2) the local nature of the Chimera requires one to compute these fluxes several times per orbit.
 
* **Executive summary of the Chimera:** the equations used to approximate the self-force in the Chimera are schematically derived as follows. First, the metric in the far-field region is expanded into a sum of time-symmetric and time-asymmetric potentials using a multipolar post-Minkowskian expansion. The expressions used to evaluate these radiation reaction (RR) potentials are obtained via post-Newtonian expansions in the near-zone of the source. The expanded metric is then resummed in terms of the Kerr metric and metric perturbations thereof, with the latter being identified with the radiaction reaction potentials. The metric perturbations, expressed in terms of the RR potentials, are then substituted into the MiSaTaQuWa equation from black hole perturbation theory to obtain local, non-adiabatic expressions for the components of the self-force. In its current formulation, the Chimera does not take into account the conservative part of the self-force. We refer the reader to Ref. [2] and the references therein for further detail.

* **Numerical Implementation:** see the example notebook <em>InspiralExamples.ipynb</em> for details on our numerical implementation.

## Dependencies ##

All the dependencies are located in the <em>dependencies.jl</em> file. Simply run <code>include("dependencies.jl")</code> to install all the necessary packages.

## Limitations and known possible performance issues ##

* This code is in its early stages. In particular, we are yet to carry out convergence tests on the generated waveforms and constants of motion.
* This code has only been tested on a 2 GHz Quad-Core Intel Core i5 Macbook Pro.
  
## Authors ##

- Lennox Keeble
- [Alejandro Cardenas-Avendano](https://alejandroc137.bitbucket.io)

Last updated: 10.06.2024

## References ##
[1] Schmidt, W. Celestial mechanics in Kerr spacetime. [arXiv:gr-qc/0202090](https://arxiv.org/abs/gr-qc/0202090)

[2] Sopuerta, C., & Yunes, N. New Kludge Scheme for the Construction of Approximate Waveforms for Extreme-Mass-Ratio Inspirals. [arXiv:1109.0572](https://arxiv.org/abs/1109.0572)

[3] Fujita, R., & Hikida, W. Analytical solutions of bound timelike geodesic orbits in Kerr spacetime. [arXiv:0906.1420v2](https://arxiv.org/abs/0906.1420)

[4] Hughes, S. A. Parameterizing black hole orbits for adiabatic inspiral. [arXiv:2401.09577v2](https://arxiv.org/abs/2401.09577)

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this 
software and associated documentation files (the "Software"), to deal in the Software 
without restriction, including without limitation the rights to use, copy, modify, merge, 
publish, distribute, sublicense, and/or sell copies of the Software, and to permit 
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.
