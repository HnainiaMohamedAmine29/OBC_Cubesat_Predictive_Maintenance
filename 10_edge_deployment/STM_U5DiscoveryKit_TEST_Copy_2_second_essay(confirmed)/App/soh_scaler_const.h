/* Generated from scaler_X.pkl / features_used.pkl by tools/gen_scaler_header.py - do not edit by hand. */
#ifndef SOH_SCALER_CONST_H_
#define SOH_SCALER_CONST_H_

#define SOH_N_FEAT 20

/* feature order: V_mean_V, V_spread, V_mean_rolling, V_mean_lag_1, V_min_fade, Tavg_C, thermal_range, delta_T_ambient, cold_cycle_count, Tavg_rolling, eclipse_flag, QD_Ah, coulombic_eff, discharge_C_rate, capacity_retention, QD_rolling, cycle, cumul_Ah, SOH_lag_1, QD_diff */
static const double SOH_MEAN[SOH_N_FEAT] = {
    7.7147059180328315, /* V_mean_V */
    0.9504956634534449, /* V_spread */
    7.715014580061444, /* V_mean_rolling */
    7.714771322686469, /* V_mean_lag_1 */
    -0.27624283407889577, /* V_min_fade */
    16.51970244221568, /* Tavg_C */
    36.710472528371014, /* thermal_range */
    0.0014692378328741965, /* delta_T_ambient */
    0.0, /* cold_cycle_count */
    16.517370265823722, /* Tavg_rolling */
    0.49990817263544535, /* eclipse_flag */
    3.538717238908437, /* QD_Ah */
    0.9800000000000001, /* coulombic_eff */
    0.5323669918245945, /* discharge_C_rate */
    0.877049181406778, /* capacity_retention */
    3.539520840905154, /* QD_rolling */
    2724.0, /* cycle */
    10086.892333515905, /* cumul_Ah */
    0.8770498027981078, /* SOH_lag_1 */
    -0.00017871538192863737, /* QD_diff */
};
static const double SOH_SCALE[SOH_N_FEAT] = {
    0.13439567550610326, /* V_mean_V */
    0.10403208653301363, /* V_spread */
    0.13145605075972322, /* V_mean_rolling */
    0.13441952578050287, /* V_mean_lag_1 */
    0.14276185117040263, /* V_min_fade */
    1.4983986854718976, /* Tavg_C */
    3.5891571089516745, /* thermal_range */
    7.999999865083762, /* delta_T_ambient */
    1.0, /* cold_cycle_count */
    0.10966091527511301, /* Tavg_rolling */
    0.4999999915677351, /* eclipse_flag */
    0.2844869898088013, /* QD_Ah */
    1.0, /* coulombic_eff */
    0.04288426032775218, /* discharge_C_rate */
    0.07050834092911358, /* capacity_retention */
    0.28451235892276544, /* QD_rolling */
    1571.836081360479, /* cycle */
    5560.585027111583, /* cumul_Ah */
    0.07050629984944926, /* SOH_lag_1 */
    3.7819250232888465e-05, /* QD_diff */
};

#endif
