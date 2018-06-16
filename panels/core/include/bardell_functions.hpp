#ifndef BARDELL_FUNCTIONS_H
#define BARDELL_FUNCTIONS_H

void calc_vec_f(double *f, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

void calc_vec_fx(double *fxi, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

void calc_vec_fxx(double *fxixi, double xi,
        double xi1t, double xi1r,double xi2t, double xi2r);

double calc_f(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

double calc_fx(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

double calc_fxx(int i, double xi,
        double xi1t, double xi1r, double xi2t, double xi2r);

#endif /** BARDELL_FUNCTIONS_H */
