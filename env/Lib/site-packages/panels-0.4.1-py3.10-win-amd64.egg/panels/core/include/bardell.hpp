
#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif

#ifndef BARDELL_FF_H
#define BARDELL_FF_H
IMPORTIT double integral_ff(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FF_H */


#ifndef BARDELL_FFXI_H
#define BARDELL_FFXI_H
IMPORTIT double integral_ffp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FFXI_H */


#ifndef BARDELL_FFXIXI_H
#define BARDELL_FFXIXI_H
IMPORTIT double integral_ffpp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FFXIXI_H */


#ifndef BARDELL_FXIFXI_H
#define BARDELL_FXIFXI_H
IMPORTIT double integral_fpfp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FXIFXI_H */


#ifndef BARDELL_FXIFXIXI_H
#define BARDELL_FXIFXIXI_H
IMPORTIT double integral_fpfpp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FXIFXIXI_H */


#ifndef BARDELL_FXIXIFXIXI_H
#define BARDELL_FXIXIFXIXI_H
IMPORTIT double integral_fppfpp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r);
#endif /** BARDELL_FXIXIFXIXI_H */

