module CoordinateDerivs

include("derivs.jl");    # ~ 3 minutes
a=0.8;
x = [131.69564626674685, 4.3344401621897894, 0.8549256185290545,5.787064500738192];
dx = [1.6738006089625808, -0.004481795575031199, -0.1266999794332496, 0.1753912531441377];

# compute auxillary quantities to reduce re-computation
SinTheta=sin(x[3]); Sin2Theta=sin(2.0*x[3]); Sin3Theta=sin(3.0*x[3]); Sin4Theta=sin(4.0 * x[3]); CosTheta=cos(x[3]); Cos2Theta=cos(2.0*x[3]); Cos3Theta=cos(3.0*x[3]); Cos4Theta=cos(4.0*x[3]); TanTheta=tan(x[3]); CscTheta=csc(x[3]); SecTheta=sec(x[3]); CotTheta=cot(x[3]);
dx2 = d2x(dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)
dx3 = d3x(dx2, dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)
dx4 = d4x(dx3, dx2, dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)
dx5 = d5x(dx4, dx3, dx2, dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)
# dx6 = d6x(dx5, dx4, dx3, dx2, dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)
# dx7 = d7x(dx6, dx5, dx4, dx3, dx2, dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)
# dx8 = d8x(dx7, dx6, dx5, dx4, dx3, dx2, dx, x, SinTheta, Sin2Theta, Sin3Theta, Sin4Theta, CosTheta, Cos2Theta, Cos3Theta, Cos4Theta, TanTheta, CscTheta, SecTheta, CotTheta, a)

end