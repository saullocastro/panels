#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif

#ifndef BARDELL_FUNCTIONS_H
#define BARDELL_FUNCTIONS_H

IMPORTIT void vec_f(double *f, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

IMPORTIT void vec_fp(double *fp, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

IMPORTIT void vec_fpp(double *fpp, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

IMPORTIT double f(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

IMPORTIT double fp(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

IMPORTIT double fpp(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

#endif /** BARDELL_FUNCTIONS_H */
