/* Generated D2-P policy; no sklearn runtime dependency. */
static double D2PStage16SafeProbability(D2PFeatures f)
{
    if (f.d32_d1 <= 1.5485947728157043)
        if (f.d32_d1 <= 1.294619083404541)
            if (f.d64_d1 <= 1.3067447543144226)
                return 0;
            else
                return 0.1164021164021164;
        else
            if (f.s16_pages <= 21184)
                return 0.29106628242074928;
            else
                return 0.47872340425531917;
    else
        if (f.d64_d1 <= 2.1293016672134399)
            if (f.s16_replacement_rate <= 0.0050152700860053301)
                return 0.88095238095238093;
            else
                return 0.63461538461538458;
        else
            if (f.d4_d1 <= 1.2957057356834412)
                return 0.90000000000000002;
            else
                return 1;
}

static double D2PStage32SafeProbability(D2PFeatures f)
{
    if (f.d64_d1 <= 1.5021027326583862)
        if (f.gap64_32_d1 <= 0.091004308313131332)
            if (f.s32_top10_std <= 0.063272316008806229)
                return 0.27500000000000002;
            else
                return 0.63636363636363635;
        else
            if (f.d16 <= 1.6238608360290527)
                return 0.50210970464135019;
            else
                return 0.75;
    else
        if (f.d64_d1 <= 1.7157695889472961)
            if (f.s32_replacement_rate <= 0.0035484861582517624)
                return 0.83941605839416056;
            else
                return 0.6696428571428571;
        else
            if (f.kth10_relative_change_16_32 <= 0.0027708339039236307)
                return 0.98290598290598286;
            else
                return 0.875;
}

#define D2P_STAGE16_THRESHOLD 0.94999999999999996
#define D2P_STAGE32_THRESHOLD 0.85046728971962615
