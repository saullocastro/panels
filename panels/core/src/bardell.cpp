
#include <stdlib.h>
#include <math.h>
#if defined(_WIN32) || defined(__WIN32__)
  #define EXPORTIT __declspec(dllexport)
#else
  #define EXPORTIT
#endif
EXPORTIT double integral_ff(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r) {
    switch(i) {
    case 0:
        switch(j) {
        case 0:
            return 0.742857142857143*x1t*y1t;
        case 1:
            return 0.104761904761905*x1t*y1r;
        case 2:
            return 0.257142857142857*x1t*y2t;
        case 3:
            return -0.0619047619047619*x1t*y2r;
        case 4:
            return 0.0666666666666667*x1t;
        case 5:
            return -0.0126984126984127*x1t;
        case 7:
            return 0.000288600288600289*x1t;
        default:
            return 0.;
        }
    case 1:
        switch(j) {
        case 0:
            return 0.104761904761905*x1r*y1t;
        case 1:
            return 0.019047619047619*x1r*y1r;
        case 2:
            return 0.0619047619047619*x1r*y2t;
        case 3:
            return -0.0142857142857143*x1r*y2r;
        case 4:
            return 0.0142857142857143*x1r;
        case 5:
            return -0.00158730158730159*x1r;
        case 6:
            return -0.000529100529100529*x1r;
        case 7:
            return 0.000144300144300144*x1r;
        default:
            return 0.;
        }
    case 2:
        switch(j) {
        case 0:
            return 0.257142857142857*x2t*y1t;
        case 1:
            return 0.0619047619047619*x2t*y1r;
        case 2:
            return 0.742857142857143*x2t*y2t;
        case 3:
            return -0.104761904761905*x2t*y2r;
        case 4:
            return 0.0666666666666667*x2t;
        case 5:
            return 0.0126984126984127*x2t;
        case 7:
            return -0.000288600288600289*x2t;
        default:
            return 0.;
        }
    case 3:
        switch(j) {
        case 0:
            return -0.0619047619047619*x2r*y1t;
        case 1:
            return -0.0142857142857143*x2r*y1r;
        case 2:
            return -0.104761904761905*x2r*y2t;
        case 3:
            return 0.019047619047619*x2r*y2r;
        case 4:
            return -0.0142857142857143*x2r;
        case 5:
            return -0.00158730158730159*x2r;
        case 6:
            return 0.000529100529100529*x2r;
        case 7:
            return 0.000144300144300144*x2r;
        default:
            return 0.;
        }
    case 4:
        switch(j) {
        case 0:
            return 0.0666666666666667*y1t;
        case 1:
            return 0.0142857142857143*y1r;
        case 2:
            return 0.0666666666666667*y2t;
        case 3:
            return -0.0142857142857143*y2r;
        case 4:
            return 0.0126984126984127;
        case 6:
            return -0.000769600769600770;
        case 8:
            return 4.44000444000444e-5;
        default:
            return 0.;
        }
    case 5:
        switch(j) {
        case 0:
            return -0.0126984126984127*y1t;
        case 1:
            return -0.00158730158730159*y1r;
        case 2:
            return 0.0126984126984127*y2t;
        case 3:
            return -0.00158730158730159*y2r;
        case 5:
            return 0.00115440115440115;
        case 7:
            return -0.000177600177600178;
        case 9:
            return 1.48000148000148e-5;
        default:
            return 0.;
        }
    case 6:
        switch(j) {
        case 1:
            return -0.000529100529100529*y1r;
        case 3:
            return 0.000529100529100529*y2r;
        case 4:
            return -0.000769600769600770;
        case 6:
            return 0.000266400266400266;
        case 8:
            return -5.92000592000592e-5;
        case 10:
            return 6.09412374118256e-6;
        default:
            return 0.;
        }
    case 7:
        switch(j) {
        case 0:
            return 0.000288600288600289*y1t;
        case 1:
            return 0.000144300144300144*y1r;
        case 2:
            return -0.000288600288600289*y2t;
        case 3:
            return 0.000144300144300144*y2r;
        case 5:
            return -0.000177600177600178;
        case 7:
            return 8.88000888000888e-5;
        case 9:
            return -2.43764949647303e-5;
        case 11:
            return 2.88669019319174e-6;
        default:
            return 0.;
        }
    case 8:
        switch(j) {
        case 4:
            return 4.44000444000444e-5;
        case 6:
            return -5.92000592000592e-5;
        case 8:
            return 3.65647424470954e-5;
        case 10:
            return -1.15467607727670e-5;
        case 12:
            return 1.51207581548139e-6;
        default:
            return 0.;
        }
    case 9:
        switch(j) {
        case 5:
            return 1.48000148000148e-5;
        case 7:
            return -2.43764949647303e-5;
        case 9:
            return 1.73201411591504e-5;
        case 11:
            return -6.04830326192555e-6;
        case 13:
            return 8.54651547880785e-7;
        default:
            return 0.;
        }
    case 10:
        switch(j) {
        case 6:
            return 6.09412374118256e-6;
        case 8:
            return -1.15467607727670e-5;
        case 10:
            return 9.07245489288833e-6;
        case 12:
            return -3.41860619152314e-6;
        case 14:
            return 5.12790928728471e-7;
        default:
            return 0.;
        }
    case 11:
        switch(j) {
        case 7:
            return 2.88669019319174e-6;
        case 9:
            return -6.04830326192555e-6;
        case 11:
            return 5.12790928728471e-6;
        case 13:
            return -2.05116371491388e-6;
        case 15:
            return 3.22868362532741e-7;
        default:
            return 0.;
        }
    case 12:
        switch(j) {
        case 8:
            return 1.51207581548139e-6;
        case 10:
            return -3.41860619152314e-6;
        case 12:
            return 3.07674557237082e-6;
        case 14:
            return -1.29147345013096e-6;
        case 16:
            return 2.11534444418003e-7;
        default:
            return 0.;
        }
    case 13:
        switch(j) {
        case 9:
            return 8.54651547880785e-7;
        case 11:
            return -2.05116371491388e-6;
        case 13:
            return 1.93721017519645e-6;
        case 15:
            return -8.46137777672011e-7;
        case 17:
            return 1.43297526863808e-7;
        default:
            return 0.;
        }
    case 14:
        switch(j) {
        case 10:
            return 5.12790928728471e-7;
        case 12:
            return -1.29147345013096e-6;
        case 14:
            return 1.26920666650802e-6;
        case 16:
            return -5.73190107455233e-7;
        case 18:
            return 9.98740338747754e-8;
        default:
            return 0.;
        }
    case 15:
        switch(j) {
        case 11:
            return 3.22868362532741e-7;
        case 13:
            return -8.46137777672011e-7;
        case 15:
            return 8.59785161182849e-7;
        case 17:
            return -3.99496135499102e-7;
        case 19:
            return 7.13385956248396e-8;
        default:
            return 0.;
        }
    case 16:
        switch(j) {
        case 12:
            return 2.11534444418003e-7;
        case 14:
            return -5.73190107455233e-7;
        case 16:
            return 5.99244203248653e-7;
        case 18:
            return -2.85354382499358e-7;
        case 20:
            return 5.20578941046127e-8;
        default:
            return 0.;
        }
    case 17:
        switch(j) {
        case 13:
            return 1.43297526863808e-7;
        case 15:
            return -3.99496135499102e-7;
        case 17:
            return 4.28031573749038e-7;
        case 19:
            return -2.08231576418451e-7;
        case 21:
            return 3.87097161290710e-8;
        default:
            return 0.;
        }
    case 18:
        switch(j) {
        case 14:
            return 9.98740338747754e-8;
        case 16:
            return -2.85354382499358e-7;
        case 18:
            return 3.12347364627676e-7;
        case 20:
            return -1.54838864516284e-7;
        case 22:
            return 2.92683219512488e-8;
        default:
            return 0.;
        }
    case 19:
        switch(j) {
        case 15:
            return 7.13385956248396e-8;
        case 17:
            return -2.08231576418451e-7;
        case 19:
            return 2.32258296774426e-7;
        case 21:
            return -1.17073287804995e-7;
        case 23:
            return 2.24617354509584e-8;
        default:
            return 0.;
        }
    case 20:
        switch(j) {
        case 16:
            return 5.20578941046127e-8;
        case 18:
            return -1.54838864516284e-7;
        case 20:
            return 1.75609931707493e-7;
        case 22:
            return -8.98469418038335e-8;
        case 24:
            return 1.74702386840787e-8;
        default:
            return 0.;
        }
    case 21:
        switch(j) {
        case 17:
            return 3.87097161290710e-8;
        case 19:
            return -1.17073287804995e-7;
        case 21:
            return 1.34770412705750e-7;
        case 23:
            return -6.98809547363149e-8;
        case 25:
            return 1.37531666236364e-8;
        default:
            return 0.;
        }
    case 22:
        switch(j) {
        case 18:
            return 2.92683219512488e-8;
        case 20:
            return -8.98469418038335e-8;
        case 22:
            return 1.04821432104472e-7;
        case 24:
            return -5.50126664945458e-8;
        case 26:
            return 1.09463979249351e-8;
        default:
            return 0.;
        }
    case 23:
        switch(j) {
        case 19:
            return 2.24617354509584e-8;
        case 21:
            return -6.98809547363149e-8;
        case 23:
            return 8.25189997418187e-8;
        case 25:
            return -4.37855916997405e-8;
        case 27:
            return 8.80004539063412e-9;
        default:
            return 0.;
        }
    case 24:
        switch(j) {
        case 20:
            return 1.74702386840787e-8;
        case 22:
            return -5.50126664945458e-8;
        case 24:
            return 6.56783875496108e-8;
        case 26:
            return -3.52001815625365e-8;
        case 28:
            return 7.13965946787297e-9;
        default:
            return 0.;
        }
    case 25:
        switch(j) {
        case 21:
            return 1.37531666236364e-8;
        case 23:
            return -4.37855916997405e-8;
        case 25:
            return 5.28002723438047e-8;
        case 27:
            return -2.85586378714919e-8;
        case 29:
            return 5.84153956462334e-9;
        default:
            return 0.;
        }
    case 26:
        switch(j) {
        case 22:
            return 1.09463979249351e-8;
        case 24:
            return -3.52001815625365e-8;
        case 26:
            return 4.28379568072378e-8;
        case 28:
            return -2.33661582584934e-8;
        default:
            return 0.;
        }
    case 27:
        switch(j) {
        case 23:
            return 8.80004539063412e-9;
        case 25:
            return -2.85586378714919e-8;
        case 27:
            return 3.50492373877400e-8;
        case 29:
            return -1.92668322482314e-8;
        default:
            return 0.;
        }
    case 28:
        switch(j) {
        case 24:
            return 7.13965946787297e-9;
        case 26:
            return -2.33661582584934e-8;
        case 28:
            return 2.89002483723470e-8;
        default:
            return 0.;
        }
    case 29:
        switch(j) {
        case 25:
            return 5.84153956462334e-9;
        case 27:
            return -1.92668322482314e-8;
        case 29:
            return 2.40019011905933e-8;
        default:
            return 0.;
        }
    default:
        return 0.;
    }
}
EXPORTIT double integral_ffp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r) {
    switch(i) {
    case 0:
        switch(j) {
        case 0:
            return -0.5*x1t*y1t;
        case 1:
            return 0.1*x1t*y1r;
        case 2:
            return 0.5*x1t*y2t;
        case 3:
            return -0.1*x1t*y2r;
        case 4:
            return 0.0857142857142857*x1t;
        case 6:
            return -0.00317460317460317*x1t;
        default:
            return 0.;
        }
    case 1:
        switch(j) {
        case 0:
            return -0.1*x1r*y1t;
        case 2:
            return 0.1*x1r*y2t;
        case 3:
            return -0.0166666666666667*x1r*y2r;
        case 4:
            return 0.00952380952380952*x1r;
        case 5:
            return 0.00476190476190476*x1r;
        case 6:
            return -0.00158730158730159*x1r;
        default:
            return 0.;
        }
    case 2:
        switch(j) {
        case 0:
            return -0.5*x2t*y1t;
        case 1:
            return -0.1*x2t*y1r;
        case 2:
            return 0.5*x2t*y2t;
        case 3:
            return 0.1*x2t*y2r;
        case 4:
            return -0.0857142857142857*x2t;
        case 6:
            return 0.00317460317460317*x2t;
        default:
            return 0.;
        }
    case 3:
        switch(j) {
        case 0:
            return 0.1*x2r*y1t;
        case 1:
            return 0.0166666666666667*x2r*y1r;
        case 2:
            return -0.1*x2r*y2t;
        case 4:
            return 0.00952380952380952*x2r;
        case 5:
            return -0.00476190476190476*x2r;
        case 6:
            return -0.00158730158730159*x2r;
        default:
            return 0.;
        }
    case 4:
        switch(j) {
        case 0:
            return -0.0857142857142857*y1t;
        case 1:
            return -0.00952380952380952*y1r;
        case 2:
            return 0.0857142857142857*y2t;
        case 3:
            return -0.00952380952380952*y2r;
        case 5:
            return 0.00634920634920635;
        case 7:
            return -0.000577200577200577;
        default:
            return 0.;
        }
    case 5:
        switch(j) {
        case 1:
            return -0.00476190476190476*y1r;
        case 3:
            return 0.00476190476190476*y2r;
        case 4:
            return -0.00634920634920635;
        case 6:
            return 0.00173160173160173;
        case 8:
            return -0.000222000222000222;
        default:
            return 0.;
        }
    case 6:
        switch(j) {
        case 0:
            return 0.00317460317460317*y1t;
        case 1:
            return 0.00158730158730159*y1r;
        case 2:
            return -0.00317460317460317*y2t;
        case 3:
            return 0.00158730158730159*y2r;
        case 5:
            return -0.00173160173160173;
        case 7:
            return 0.000666000666000666;
        case 9:
            return -0.000103600103600104;
        default:
            return 0.;
        }
    case 7:
        switch(j) {
        case 4:
            return 0.000577200577200577;
        case 6:
            return -0.000666000666000666;
        case 8:
            return 0.000310800310800311;
        case 10:
            return -5.48471136706431e-5;
        default:
            return 0.;
        }
    case 8:
        switch(j) {
        case 5:
            return 0.000222000222000222;
        case 7:
            return -0.000310800310800311;
        case 9:
            return 0.000164541341011929;
        case 11:
            return -3.17535921251092e-5;
        default:
            return 0.;
        }
    case 9:
        switch(j) {
        case 6:
            return 0.000103600103600104;
        case 8:
            return -0.000164541341011929;
        case 10:
            return 9.52607763753275e-5;
        case 12:
            return -1.96569856012580e-5;
        default:
            return 0.;
        }
    case 10:
        switch(j) {
        case 7:
            return 5.48471136706431e-5;
        case 9:
            return -9.52607763753275e-5;
        case 11:
            return 5.89709568037741e-5;
        case 13:
            return -1.28197732182118e-5;
        default:
            return 0.;
        }
    case 11:
        switch(j) {
        case 8:
            return 3.17535921251092e-5;
        case 10:
            return -5.89709568037741e-5;
        case 12:
            return 3.84593196546353e-5;
        case 14:
            return -8.71744578838400e-6;
        default:
            return 0.;
        }
    case 12:
        switch(j) {
        case 9:
            return 1.96569856012580e-5;
        case 11:
            return -3.84593196546353e-5;
        case 13:
            return 2.61523373651520e-5;
        case 15:
            return -6.13449888812208e-6;
        default:
            return 0.;
        }
    case 13:
        switch(j) {
        case 10:
            return 1.28197732182118e-5;
        case 12:
            return -2.61523373651520e-5;
        case 14:
            return 1.84034966643662e-5;
        case 16:
            return -4.44222333277806e-6;
        default:
            return 0.;
        }
    case 14:
        switch(j) {
        case 11:
            return 8.71744578838400e-6;
        case 13:
            return -1.84034966643662e-5;
        case 15:
            return 1.33266699983342e-5;
        case 17:
            return -3.29584311786759e-6;
        default:
            return 0.;
        }
    case 15:
        switch(j) {
        case 12:
            return 6.13449888812208e-6;
        case 14:
            return -1.33266699983342e-5;
        case 16:
            return 9.88752935360277e-6;
        case 18:
            return -2.49685084686939e-6;
        default:
            return 0.;
        }
    case 16:
        switch(j) {
        case 13:
            return 4.44222333277806e-6;
        case 15:
            return -9.88752935360277e-6;
        case 17:
            return 7.49055254060816e-6;
        case 19:
            return -1.92614208187067e-6;
        default:
            return 0.;
        }
    case 17:
        switch(j) {
        case 14:
            return 3.29584311786759e-6;
        case 16:
            return -7.49055254060816e-6;
        case 18:
            return 5.77842624561201e-6;
        case 20:
            return -1.50967892903377e-6;
        default:
            return 0.;
        }
    case 18:
        switch(j) {
        case 15:
            return 2.49685084686939e-6;
        case 17:
            return -5.77842624561201e-6;
        case 19:
            return 4.52903678710130e-6;
        case 21:
            return -1.20000120000120e-6;
        default:
            return 0.;
        }
    case 19:
        switch(j) {
        case 16:
            return 1.92614208187067e-6;
        case 18:
            return -4.52903678710130e-6;
        case 20:
            return 3.60000360000360e-6;
        case 22:
            return -9.65854624391210e-7;
        default:
            return 0.;
        }
    case 20:
        switch(j) {
        case 17:
            return 1.50967892903377e-6;
        case 19:
            return -3.60000360000360e-6;
        case 21:
            return 2.89756387317363e-6;
        case 23:
            return -7.86160740783543e-7;
        default:
            return 0.;
        }
    case 21:
        switch(j) {
        case 18:
            return 1.20000120000120e-6;
        case 20:
            return -2.89756387317363e-6;
        case 22:
            return 2.35848222235063e-6;
        case 24:
            return -6.46398831310913e-7;
        default:
            return 0.;
        }
    case 22:
        switch(j) {
        case 19:
            return 9.65854624391210e-7;
        case 21:
            return -2.35848222235063e-6;
        case 23:
            return 1.93919649393274e-6;
        case 25:
            return -5.36373498321821e-7;
        default:
            return 0.;
        }
    case 23:
        switch(j) {
        case 20:
            return 7.86160740783543e-7;
        case 22:
            return -1.93919649393274e-6;
        case 24:
            return 1.60912049496546e-6;
        case 26:
            return -4.48802314922340e-7;
        default:
            return 0.;
        }
    case 24:
        switch(j) {
        case 21:
            return 6.46398831310913e-7;
        case 23:
            return -1.60912049496546e-6;
        case 25:
            return 1.34640694476702e-6;
        case 27:
            return -3.78401951797267e-7;
        default:
            return 0.;
        }
    case 25:
        switch(j) {
        case 22:
            return 5.36373498321821e-7;
        case 24:
            return -1.34640694476702e-6;
        case 26:
            return 1.13520585539180e-6;
        case 28:
            return -3.21284676054284e-7;
        default:
            return 0.;
        }
    case 26:
        switch(j) {
        case 23:
            return 4.48802314922340e-7;
        case 25:
            return -1.13520585539180e-6;
        case 27:
            return 9.63854028162851e-7;
        case 29:
            return -2.74552359537297e-7;
        default:
            return 0.;
        }
    case 27:
        switch(j) {
        case 24:
            return 3.78401951797267e-7;
        case 26:
            return -9.63854028162851e-7;
        case 28:
            return 8.23657078611891e-7;
        default:
            return 0.;
        }
    case 28:
        switch(j) {
        case 25:
            return 3.21284676054284e-7;
        case 27:
            return -8.23657078611891e-7;
        case 29:
            return 7.08056085122503e-7;
        default:
            return 0.;
        }
    case 29:
        switch(j) {
        case 26:
            return 2.74552359537297e-7;
        case 28:
            return -7.08056085122503e-7;
        default:
            return 0.;
        }
    default:
        return 0.;
    }
}
EXPORTIT double integral_ffpp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r) {
    switch(i) {
    case 0:
        switch(j) {
        case 0:
            return -0.6*x1t*y1t;
        case 1:
            return -0.55*x1t*y1r;
        case 2:
            return 0.6*x1t*y2t;
        case 3:
            return -0.05*x1t*y2r;
        case 5:
            return 0.0285714285714286*x1t;
        default:
            return 0.;
        }
    case 1:
        switch(j) {
        case 0:
            return -0.05*x1r*y1t;
        case 1:
            return -0.0666666666666667*x1r*y1r;
        case 2:
            return 0.05*x1r*y2t;
        case 3:
            return 0.0166666666666667*x1r*y2r;
        case 4:
            return -0.0333333333333333*x1r;
        case 5:
            return 0.0142857142857143*x1r;
        default:
            return 0.;
        }
    case 2:
        switch(j) {
        case 0:
            return 0.6*x2t*y1t;
        case 1:
            return 0.05*x2t*y1r;
        case 2:
            return -0.6*x2t*y2t;
        case 3:
            return 0.55*x2t*y2r;
        case 5:
            return -0.0285714285714286*x2t;
        default:
            return 0.;
        }
    case 3:
        switch(j) {
        case 0:
            return -0.05*x2r*y1t;
        case 1:
            return 0.0166666666666667*x2r*y1r;
        case 2:
            return 0.05*x2r*y2t;
        case 3:
            return -0.0666666666666667*x2r*y2r;
        case 4:
            return 0.0333333333333333*x2r;
        case 5:
            return 0.0142857142857143*x2r;
        default:
            return 0.;
        }
    case 4:
        switch(j) {
        case 1:
            return -0.0333333333333333*y1r;
        case 3:
            return 0.0333333333333333*y2r;
        case 4:
            return -0.0380952380952381;
        case 6:
            return 0.00634920634920635;
        default:
            return 0.;
        }
    case 5:
        switch(j) {
        case 0:
            return 0.0285714285714286*y1t;
        case 1:
            return 0.0142857142857143*y1r;
        case 2:
            return -0.0285714285714286*y2t;
        case 3:
            return 0.0142857142857143*y2r;
        case 5:
            return -0.0126984126984127;
        case 7:
            return 0.00288600288600289;
        default:
            return 0.;
        }
    case 6:
        switch(j) {
        case 4:
            return 0.00634920634920635;
        case 6:
            return -0.00577200577200577;
        case 8:
            return 0.00155400155400155;
        default:
            return 0.;
        }
    case 7:
        switch(j) {
        case 5:
            return 0.00288600288600289;
        case 7:
            return -0.00310800310800311;
        case 9:
            return 0.000932400932400932;
        default:
            return 0.;
        }
    case 8:
        switch(j) {
        case 6:
            return 0.00155400155400155;
        case 8:
            return -0.00186480186480186;
        case 10:
            return 0.000603318250377074;
        default:
            return 0.;
        }
    case 9:
        switch(j) {
        case 7:
            return 0.000932400932400932;
        case 9:
            return -0.00120663650075415;
        case 11:
            return 0.000412796697626419;
        default:
            return 0.;
        }
    case 10:
        switch(j) {
        case 8:
            return 0.000603318250377074;
        case 10:
            return -0.000825593395252838;
        case 12:
            return 0.000294854784018871;
        default:
            return 0.;
        }
    case 11:
        switch(j) {
        case 9:
            return 0.000412796697626419;
        case 11:
            return -0.000589709568037741;
        case 13:
            return 0.000217936144709600;
        default:
            return 0.;
        }
    case 12:
        switch(j) {
        case 10:
            return 0.000294854784018871;
        case 12:
            return -0.000435872289419200;
        case 14:
            return 0.000165631469979296;
        default:
            return 0.;
        }
    case 13:
        switch(j) {
        case 11:
            return 0.000217936144709600;
        case 13:
            return -0.000331262939958592;
        case 15:
            return 0.000128824476650564;
        default:
            return 0.;
        }
    case 14:
        switch(j) {
        case 12:
            return 0.000165631469979296;
        case 14:
            return -0.000257648953301127;
        case 16:
            return 0.000102171136653895;
        default:
            return 0.;
        }
    case 15:
        switch(j) {
        case 13:
            return 0.000128824476650564;
        case 15:
            return -0.000204342273307791;
        case 17:
            return 8.23960779466897e-5;
        default:
            return 0.;
        }
    case 16:
        switch(j) {
        case 14:
            return 0.000102171136653895;
        case 16:
            return -0.000164792155893379;
        case 18:
            return 6.74149728654734e-5;
        default:
            return 0.;
        }
    case 17:
        switch(j) {
        case 15:
            return 8.23960779466897e-5;
        case 17:
            return -0.000134829945730947;
        case 19:
            return 5.58581203742494e-5;
        default:
            return 0.;
        }
    case 18:
        switch(j) {
        case 16:
            return 6.74149728654734e-5;
        case 18:
            return -0.000111716240748499;
        case 20:
            return 4.68000468000468e-5;
        default:
            return 0.;
        }
    case 19:
        switch(j) {
        case 17:
            return 5.58581203742494e-5;
        case 19:
            return -9.36000936000936e-5;
        case 21:
            return 3.96000396000396e-5;
        default:
            return 0.;
        }
    case 20:
        switch(j) {
        case 18:
            return 4.68000468000468e-5;
        case 20:
            return -7.92000792000792e-5;
        case 22:
            return 3.38049118536923e-5;
        default:
            return 0.;
        }
    case 21:
        switch(j) {
        case 19:
            return 3.96000396000396e-5;
        case 21:
            return -6.76098237073847e-5;
        case 23:
            return 2.90879474089911e-5;
        default:
            return 0.;
        }
    case 22:
        switch(j) {
        case 20:
            return 3.38049118536923e-5;
        case 22:
            return -5.81758948179822e-5;
        case 24:
            return 2.52095544211256e-5;
        default:
            return 0.;
        }
    case 23:
        switch(j) {
        case 21:
            return 2.90879474089911e-5;
        case 23:
            return -5.04191088422512e-5;
        case 25:
            return 2.19913134311947e-5;
        default:
            return 0.;
        }
    case 24:
        switch(j) {
        case 22:
            return 2.52095544211256e-5;
        case 24:
            return -4.39826268623894e-5;
        case 26:
            return 1.92984995416606e-5;
        default:
            return 0.;
        }
    case 25:
        switch(j) {
        case 23:
            return 2.19913134311947e-5;
        case 25:
            return -3.85969990833213e-5;
        case 27:
            return 1.70280878308770e-5;
        default:
            return 0.;
        }
    case 26:
        switch(j) {
        case 24:
            return 1.92984995416606e-5;
        case 26:
            return -3.40561756617541e-5;
        case 28:
            return 1.51003797745513e-5;
        default:
            return 0.;
        }
    case 27:
        switch(j) {
        case 25:
            return 1.70280878308770e-5;
        case 27:
            return -3.02007595491027e-5;
        case 29:
            return 1.34530656173275e-5;
        default:
            return 0.;
        }
    case 28:
        switch(j) {
        case 26:
            return 1.51003797745513e-5;
        case 28:
            return -2.69061312346551e-5;
        default:
            return 0.;
        }
    case 29:
        switch(j) {
        case 27:
            return 1.34530656173275e-5;
        case 29:
            return -2.40739068941651e-5;
        default:
            return 0.;
        }
    default:
        return 0.;
    }
}
EXPORTIT double integral_fpfp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r) {
    switch(i) {
    case 0:
        switch(j) {
        case 0:
            return 0.6*x1t*y1t;
        case 1:
            return 0.05*x1t*y1r;
        case 2:
            return -0.6*x1t*y2t;
        case 3:
            return 0.05*x1t*y2r;
        case 5:
            return -0.0285714285714286*x1t;
        default:
            return 0.;
        }
    case 1:
        switch(j) {
        case 0:
            return 0.05*x1r*y1t;
        case 1:
            return 0.0666666666666667*x1r*y1r;
        case 2:
            return -0.05*x1r*y2t;
        case 3:
            return -0.0166666666666667*x1r*y2r;
        case 4:
            return 0.0333333333333333*x1r;
        case 5:
            return -0.0142857142857143*x1r;
        default:
            return 0.;
        }
    case 2:
        switch(j) {
        case 0:
            return -0.6*x2t*y1t;
        case 1:
            return -0.05*x2t*y1r;
        case 2:
            return 0.6*x2t*y2t;
        case 3:
            return -0.05*x2t*y2r;
        case 5:
            return 0.0285714285714286*x2t;
        default:
            return 0.;
        }
    case 3:
        switch(j) {
        case 0:
            return 0.05*x2r*y1t;
        case 1:
            return -0.0166666666666667*x2r*y1r;
        case 2:
            return -0.05*x2r*y2t;
        case 3:
            return 0.0666666666666667*x2r*y2r;
        case 4:
            return -0.0333333333333333*x2r;
        case 5:
            return -0.0142857142857143*x2r;
        default:
            return 0.;
        }
    case 4:
        switch(j) {
        case 1:
            return 0.0333333333333333*y1r;
        case 3:
            return -0.0333333333333333*y2r;
        case 4:
            return 0.0380952380952381;
        case 6:
            return -0.00634920634920635;
        default:
            return 0.;
        }
    case 5:
        switch(j) {
        case 0:
            return -0.0285714285714286*y1t;
        case 1:
            return -0.0142857142857143*y1r;
        case 2:
            return 0.0285714285714286*y2t;
        case 3:
            return -0.0142857142857143*y2r;
        case 5:
            return 0.0126984126984127;
        case 7:
            return -0.00288600288600289;
        default:
            return 0.;
        }
    case 6:
        switch(j) {
        case 4:
            return -0.00634920634920635;
        case 6:
            return 0.00577200577200577;
        case 8:
            return -0.00155400155400155;
        default:
            return 0.;
        }
    case 7:
        switch(j) {
        case 5:
            return -0.00288600288600289;
        case 7:
            return 0.00310800310800311;
        case 9:
            return -0.000932400932400932;
        default:
            return 0.;
        }
    case 8:
        switch(j) {
        case 6:
            return -0.00155400155400155;
        case 8:
            return 0.00186480186480186;
        case 10:
            return -0.000603318250377074;
        default:
            return 0.;
        }
    case 9:
        switch(j) {
        case 7:
            return -0.000932400932400932;
        case 9:
            return 0.00120663650075415;
        case 11:
            return -0.000412796697626419;
        default:
            return 0.;
        }
    case 10:
        switch(j) {
        case 8:
            return -0.000603318250377074;
        case 10:
            return 0.000825593395252838;
        case 12:
            return -0.000294854784018871;
        default:
            return 0.;
        }
    case 11:
        switch(j) {
        case 9:
            return -0.000412796697626419;
        case 11:
            return 0.000589709568037741;
        case 13:
            return -0.000217936144709600;
        default:
            return 0.;
        }
    case 12:
        switch(j) {
        case 10:
            return -0.000294854784018871;
        case 12:
            return 0.000435872289419200;
        case 14:
            return -0.000165631469979296;
        default:
            return 0.;
        }
    case 13:
        switch(j) {
        case 11:
            return -0.000217936144709600;
        case 13:
            return 0.000331262939958592;
        case 15:
            return -0.000128824476650564;
        default:
            return 0.;
        }
    case 14:
        switch(j) {
        case 12:
            return -0.000165631469979296;
        case 14:
            return 0.000257648953301127;
        case 16:
            return -0.000102171136653895;
        default:
            return 0.;
        }
    case 15:
        switch(j) {
        case 13:
            return -0.000128824476650564;
        case 15:
            return 0.000204342273307791;
        case 17:
            return -8.23960779466897e-5;
        default:
            return 0.;
        }
    case 16:
        switch(j) {
        case 14:
            return -0.000102171136653895;
        case 16:
            return 0.000164792155893379;
        case 18:
            return -6.74149728654734e-5;
        default:
            return 0.;
        }
    case 17:
        switch(j) {
        case 15:
            return -8.23960779466897e-5;
        case 17:
            return 0.000134829945730947;
        case 19:
            return -5.58581203742494e-5;
        default:
            return 0.;
        }
    case 18:
        switch(j) {
        case 16:
            return -6.74149728654734e-5;
        case 18:
            return 0.000111716240748499;
        case 20:
            return -4.68000468000468e-5;
        default:
            return 0.;
        }
    case 19:
        switch(j) {
        case 17:
            return -5.58581203742494e-5;
        case 19:
            return 9.36000936000936e-5;
        case 21:
            return -3.96000396000396e-5;
        default:
            return 0.;
        }
    case 20:
        switch(j) {
        case 18:
            return -4.68000468000468e-5;
        case 20:
            return 7.92000792000792e-5;
        case 22:
            return -3.38049118536923e-5;
        default:
            return 0.;
        }
    case 21:
        switch(j) {
        case 19:
            return -3.96000396000396e-5;
        case 21:
            return 6.76098237073847e-5;
        case 23:
            return -2.90879474089911e-5;
        default:
            return 0.;
        }
    case 22:
        switch(j) {
        case 20:
            return -3.38049118536923e-5;
        case 22:
            return 5.81758948179822e-5;
        case 24:
            return -2.52095544211256e-5;
        default:
            return 0.;
        }
    case 23:
        switch(j) {
        case 21:
            return -2.90879474089911e-5;
        case 23:
            return 5.04191088422512e-5;
        case 25:
            return -2.19913134311947e-5;
        default:
            return 0.;
        }
    case 24:
        switch(j) {
        case 22:
            return -2.52095544211256e-5;
        case 24:
            return 4.39826268623894e-5;
        case 26:
            return -1.92984995416606e-5;
        default:
            return 0.;
        }
    case 25:
        switch(j) {
        case 23:
            return -2.19913134311947e-5;
        case 25:
            return 3.85969990833213e-5;
        case 27:
            return -1.70280878308770e-5;
        default:
            return 0.;
        }
    case 26:
        switch(j) {
        case 24:
            return -1.92984995416606e-5;
        case 26:
            return 3.40561756617541e-5;
        case 28:
            return -1.51003797745513e-5;
        default:
            return 0.;
        }
    case 27:
        switch(j) {
        case 25:
            return -1.70280878308770e-5;
        case 27:
            return 3.02007595491027e-5;
        case 29:
            return -1.34530656173275e-5;
        default:
            return 0.;
        }
    case 28:
        switch(j) {
        case 26:
            return -1.51003797745513e-5;
        case 28:
            return 2.69061312346551e-5;
        default:
            return 0.;
        }
    case 29:
        switch(j) {
        case 27:
            return -1.34530656173275e-5;
        case 29:
            return 2.40739068941651e-5;
        default:
            return 0.;
        }
    default:
        return 0.;
    }
}
EXPORTIT double integral_fpfpp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r) {
    switch(i) {
    case 0:
        switch(j) {
        case 1:
            return 0.25*x1t*y1r;
        case 3:
            return -0.25*x1t*y2r;
        case 4:
            return 0.2*x1t;
        default:
            return 0.;
        }
    case 1:
        switch(j) {
        case 0:
            return -0.25*x1r*y1t;
        case 1:
            return -0.125*x1r*y1r;
        case 2:
            return 0.25*x1r*y2t;
        case 3:
            return -0.125*x1r*y2r;
        case 4:
            return 0.1*x1r;
        default:
            return 0.;
        }
    case 2:
        switch(j) {
        case 1:
            return -0.25*x2t*y1r;
        case 3:
            return 0.25*x2t*y2r;
        case 4:
            return -0.2*x2t;
        default:
            return 0.;
        }
    case 3:
        switch(j) {
        case 0:
            return 0.25*x2r*y1t;
        case 1:
            return 0.125*x2r*y1r;
        case 2:
            return -0.25*x2r*y2t;
        case 3:
            return 0.125*x2r*y2r;
        case 4:
            return 0.1*x2r;
        default:
            return 0.;
        }
    case 4:
        switch(j) {
        case 0:
            return -0.2*y1t;
        case 1:
            return -0.1*y1r;
        case 2:
            return 0.2*y2t;
        case 3:
            return -0.1*y2r;
        case 5:
            return 0.0571428571428571;
        default:
            return 0.;
        }
    case 5:
        switch(j) {
        case 4:
            return -0.0571428571428571;
        case 6:
            return 0.0317460317460317;
        default:
            return 0.;
        }
    case 6:
        switch(j) {
        case 5:
            return -0.0317460317460317;
        case 7:
            return 0.0202020202020202;
        default:
            return 0.;
        }
    case 7:
        switch(j) {
        case 6:
            return -0.0202020202020202;
        case 8:
            return 0.0139860139860140;
        default:
            return 0.;
        }
    case 8:
        switch(j) {
        case 7:
            return -0.0139860139860140;
        case 9:
            return 0.0102564102564103;
        default:
            return 0.;
        }
    case 9:
        switch(j) {
        case 8:
            return -0.0102564102564103;
        case 10:
            return 0.00784313725490196;
        default:
            return 0.;
        }
    case 10:
        switch(j) {
        case 9:
            return -0.00784313725490196;
        case 11:
            return 0.00619195046439629;
        default:
            return 0.;
        }
    case 11:
        switch(j) {
        case 10:
            return -0.00619195046439629;
        case 12:
            return 0.00501253132832080;
        default:
            return 0.;
        }
    case 12:
        switch(j) {
        case 11:
            return -0.00501253132832080;
        case 13:
            return 0.00414078674948240;
        default:
            return 0.;
        }
    case 13:
        switch(j) {
        case 12:
            return -0.00414078674948240;
        case 14:
            return 0.00347826086956522;
        default:
            return 0.;
        }
    case 14:
        switch(j) {
        case 13:
            return -0.00347826086956522;
        case 15:
            return 0.00296296296296296;
        default:
            return 0.;
        }
    case 15:
        switch(j) {
        case 14:
            return -0.00296296296296296;
        case 16:
            return 0.00255427841634738;
        default:
            return 0.;
        }
    case 16:
        switch(j) {
        case 15:
            return -0.00255427841634738;
        case 17:
            return 0.00222469410456062;
        default:
            return 0.;
        }
    case 17:
        switch(j) {
        case 16:
            return -0.00222469410456062;
        case 18:
            return 0.00195503421309873;
        default:
            return 0.;
        }
    case 18:
        switch(j) {
        case 17:
            return -0.00195503421309873;
        case 19:
            return 0.00173160173160173;
        default:
            return 0.;
        }
    case 19:
        switch(j) {
        case 18:
            return -0.00173160173160173;
        case 20:
            return 0.00154440154440154;
        default:
            return 0.;
        }
    case 20:
        switch(j) {
        case 19:
            return -0.00154440154440154;
        case 21:
            return 0.00138600138600139;
        default:
            return 0.;
        }
    case 21:
        switch(j) {
        case 20:
            return -0.00138600138600139;
        case 22:
            return 0.00125078173858662;
        default:
            return 0.;
        }
    case 22:
        switch(j) {
        case 21:
            return -0.00125078173858662;
        case 23:
            return 0.00113442994895065;
        default:
            return 0.;
        }
    case 23:
        switch(j) {
        case 22:
            return -0.00113442994895065;
        case 24:
            return 0.00103359173126615;
        default:
            return 0.;
        }
    case 24:
        switch(j) {
        case 23:
            return -0.00103359173126615;
        case 25:
            return 0.000945626477541371;
        default:
            return 0.;
        }
    case 25:
        switch(j) {
        case 24:
            return -0.000945626477541371;
        case 26:
            return 0.000868432479374729;
        default:
            return 0.;
        }
    case 26:
        switch(j) {
        case 25:
            return -0.000868432479374729;
        case 27:
            return 0.000800320128051221;
        default:
            return 0.;
        }
    case 27:
        switch(j) {
        case 26:
            return -0.000800320128051221;
        case 28:
            return 0.000739918608953015;
        default:
            return 0.;
        }
    case 28:
        switch(j) {
        case 27:
            return -0.000739918608953015;
        case 29:
            return 0.000686106346483705;
        default:
            return 0.;
        }
    case 29:
        switch(j) {
        case 28:
            return -0.000686106346483705;
        default:
            return 0.;
        }
    default:
        return 0.;
    }
}
EXPORTIT double integral_fppfpp(int i, int j,
           double x1t, double x1r, double x2t, double x2r,
           double y1t, double y1r, double y2t, double y2r) {
    switch(i) {
    case 0:
        switch(j) {
        case 0:
            return 1.5*x1t*y1t;
        case 1:
            return 0.75*x1t*y1r;
        case 2:
            return -1.5*x1t*y2t;
        case 3:
            return 0.75*x1t*y2r;
        default:
            return 0.;
        }
    case 1:
        switch(j) {
        case 0:
            return 0.75*x1r*y1t;
        case 1:
            return 0.5*x1r*y1r;
        case 2:
            return -0.75*x1r*y2t;
        case 3:
            return 0.25*x1r*y2r;
        default:
            return 0.;
        }
    case 2:
        switch(j) {
        case 0:
            return -1.5*x2t*y1t;
        case 1:
            return -0.75*x2t*y1r;
        case 2:
            return 1.5*x2t*y2t;
        case 3:
            return -0.75*x2t*y2r;
        default:
            return 0.;
        }
    case 3:
        switch(j) {
        case 0:
            return 0.75*x2r*y1t;
        case 1:
            return 0.25*x2r*y1r;
        case 2:
            return -0.75*x2r*y2t;
        case 3:
            return 0.5*x2r*y2r;
        default:
            return 0.;
        }
    case 4:
        switch(j) {
        case 4:
            return 0.400000000000000;
        default:
            return 0.;
        }
    case 5:
        switch(j) {
        case 5:
            return 0.285714285714286;
        default:
            return 0.;
        }
    case 6:
        switch(j) {
        case 6:
            return 0.222222222222222;
        default:
            return 0.;
        }
    case 7:
        switch(j) {
        case 7:
            return 0.181818181818182;
        default:
            return 0.;
        }
    case 8:
        switch(j) {
        case 8:
            return 0.153846153846154;
        default:
            return 0.;
        }
    case 9:
        switch(j) {
        case 9:
            return 0.133333333333333;
        default:
            return 0.;
        }
    case 10:
        switch(j) {
        case 10:
            return 0.117647058823529;
        default:
            return 0.;
        }
    case 11:
        switch(j) {
        case 11:
            return 0.105263157894737;
        default:
            return 0.;
        }
    case 12:
        switch(j) {
        case 12:
            return 0.0952380952380952;
        default:
            return 0.;
        }
    case 13:
        switch(j) {
        case 13:
            return 0.0869565217391304;
        default:
            return 0.;
        }
    case 14:
        switch(j) {
        case 14:
            return 0.0800000000000000;
        default:
            return 0.;
        }
    case 15:
        switch(j) {
        case 15:
            return 0.0740740740740741;
        default:
            return 0.;
        }
    case 16:
        switch(j) {
        case 16:
            return 0.0689655172413793;
        default:
            return 0.;
        }
    case 17:
        switch(j) {
        case 17:
            return 0.0645161290322581;
        default:
            return 0.;
        }
    case 18:
        switch(j) {
        case 18:
            return 0.0606060606060606;
        default:
            return 0.;
        }
    case 19:
        switch(j) {
        case 19:
            return 0.0571428571428571;
        default:
            return 0.;
        }
    case 20:
        switch(j) {
        case 20:
            return 0.0540540540540541;
        default:
            return 0.;
        }
    case 21:
        switch(j) {
        case 21:
            return 0.0512820512820513;
        default:
            return 0.;
        }
    case 22:
        switch(j) {
        case 22:
            return 0.0487804878048781;
        default:
            return 0.;
        }
    case 23:
        switch(j) {
        case 23:
            return 0.0465116279069767;
        default:
            return 0.;
        }
    case 24:
        switch(j) {
        case 24:
            return 0.0444444444444444;
        default:
            return 0.;
        }
    case 25:
        switch(j) {
        case 25:
            return 0.0425531914893617;
        default:
            return 0.;
        }
    case 26:
        switch(j) {
        case 26:
            return 0.0408163265306122;
        default:
            return 0.;
        }
    case 27:
        switch(j) {
        case 27:
            return 0.0392156862745098;
        default:
            return 0.;
        }
    case 28:
        switch(j) {
        case 28:
            return 0.0377358490566038;
        default:
            return 0.;
        }
    case 29:
        switch(j) {
        case 29:
            return 0.0363636363636364;
        default:
            return 0.;
        }
    default:
        return 0.;
    }
}
