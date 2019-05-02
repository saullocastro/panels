#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif

#ifndef BARDELL_FUNCTIONS_UV_H
#define BARDELL_FUNCTIONS_UV_H

IMPORTIT void vec_fuv(double *f, double xi,
        double xi1t, double xi2t);

IMPORTIT void vec_fuv_x(double *fxi, double xi,
        double xi1t, double xi2t);

IMPORTIT void vec_fuv_xx(double *fxixi, double xi,
        double xi1t, double xi2t);

IMPORTIT double fuv(int i, double xi,
        double xi1t, double xi2t);

IMPORTIT double fuv_x(int i, double xi,
        double xi1t, double xi2t);

IMPORTIT double fuv_xx(int i, double xi,
        double xi1t, double xi2t);

#endif /** BARDELL_FUNCTIONS_UV_H */