#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif

#ifndef BARDELL_FUNCTIONS_W_H
#define BARDELL_FUNCTIONS_W_H

IMPORTIT void vec_fw(double *f, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

IMPORTIT void vec_fw_x(double *fxi, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

IMPORTIT void vec_fw_xx(double *fxixi, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

IMPORTIT double fw(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

IMPORTIT double fw_x(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

IMPORTIT double fw_xx(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

#endif /** BARDELL_FUNCTIONS_W_H */