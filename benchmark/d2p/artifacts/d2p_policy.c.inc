/* Generated D2-P policy; no sklearn runtime dependency. */
static double D2PStage16SafeProbability(D2PFeatures f)
{
    if (f.d32_d1 <= 1.4477165341377258)
        if (f.d64_d1 <= 1.5011770129203796)
            if (f.gap32_16_d1 <= 0.079872917383909225)
                return 0.093862815884476536;
            else
                return 0.23756906077348067;
        else
            if (f.d4_d1 <= 1.1515732407569885)
                return 0.44705882352941179;
            else
                return 0.19047619047619047;
    else
        if (f.d32_d1 <= 1.6509846448898315)
            if (f.d2_d1 <= 1.0805763602256775)
                return 0.69999999999999996;
            else
                return 0.37623762376237624;
        else
            if (f.d32_d1 <= 1.8089315891265869)
                return 0.83606557377049184;
            else
                return 0.99248120300751874;
}

static double D2PStage32SafeProbability(D2PFeatures f)
{
    if (f.d64_d1 <= 1.5536543130874634)
        if (f.s16_candidates <= 39234.5)
            if (f.d64 <= 2.135347843170166)
                return 0.36363636363636365;
            else
                return 0.59027777777777779;
        else
            if (f.gap32_16_d1 <= 0.094625022262334824)
                return 0.58461538461538465;
            else
                return 0.90697674418604646;
    else
        if (f.d32_d1 <= 1.5726179480552673)
            if (f.candidate_growth_16_32 <= 1.0391929149627686)
                return 0.91366906474820142;
            else
                return 0.73493975903614461;
        else
            if (f.candidate_growth_16_32 <= 1.1196491718292236)
                return 0.9946236559139785;
            else
                return 0.92307692307692313;
}

#define D2P_STAGE16_THRESHOLD 0.92500000000000004
#define D2P_STAGE32_THRESHOLD 0.79365079365079361
