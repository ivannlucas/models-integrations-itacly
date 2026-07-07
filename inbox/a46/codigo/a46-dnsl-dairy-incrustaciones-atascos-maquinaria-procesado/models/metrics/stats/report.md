# CU07 synthetic dataset EDA

## Resumen ejecutivo

- **telemetry_rows**: 198615
- **telemetry_columns**: 89
- **assets**: 10
- **cycles**: 180
- **time_min**: 2026-01-01 00:26:00+00:00
- **time_max**: 2026-01-26 07:27:00+00:00
- **time_span_h**: 607.0166666666667
- **median_dt_min**: 1.0
- **informative_row_pct**: 36.43632152657151
- **low_info_production_row_pct**: 63.563678473428496
- **production_row_pct**: 95.53910832515167
- **informative_within_production_pct**: 33.468419804484725
- **low_info_within_production_pct**: 66.53158019551528
- **severity_col**: Rf_m2K_W
- **severity_min**: 9.7764e-06
- **severity_p50**: 0.0003166589
- **severity_p99**: 0.002319189255999999
- **severity_max**: 0.00306
- **maintenance_rows**: 244
- **planned_maintenance_events**: 171
- **unplanned_maintenance_events**: 73
- **maintenance_duration_min_total**: 8860.0
- **maintenance_duration_min_median**: 38.0

## Checks de calidad

- **sequences_total**: 180
- **sequence_issue_rows**: 0
- **stage_threshold_mismatch_pct**: 0.0
- **max_target_mismatch_pct**: 0.0
- **naive_cross_cycle_window_pct_if_group_by_asset**: 10.240567232210687
- **sequence_aware_windows_total**: 35508
- **bad_windows_crossing_cycle_boundary**: 0
- **sequences_shorter_than_seq_len**: 0
- **median_inter_cycle_gap_h**: 12.825
- **median_negative_rf_steps_in_production_per_cycle**: 0.0
- **median_negative_mass_steps_in_production_per_cycle**: 0.0

## Top ciclos por severidad

| asset_id   | sequence_id     | cycle_id        |   cycle_index |   rows | start_time                | end_time                  |   duration_h |   observed_h_from_rows |   median_dt_min |   production_rows |   production_row_pct |   cip_rows |   maintenance_rows |   informative_row_pct |   low_info_prod_pct |   informative_within_production_pct |   low_info_within_production_pct |   max_stage | stage_mode   |   start_severity |   end_severity |   max_severity |   severity_increase |   fouling_onset_events |   clog_onset_events |   clog_rows |   has_unplanned_horizon | phase_set                                    | last_maintenance_type_mode   |   maintenance_events |   planned_maint_events |   unplanned_maint_events |   maintenance_duration_min | maintenance_types    | fault_types         |
|:-----------|:----------------|:----------------|--------------:|-------:|:--------------------------|:--------------------------|-------------:|-----------------------:|----------------:|------------------:|---------------------:|-----------:|-------------------:|----------------------:|--------------------:|------------------------------------:|---------------------------------:|------------:|:-------------|-----------------:|---------------:|---------------:|--------------------:|-----------------------:|--------------------:|------------:|------------------------:|:---------------------------------------------|:-----------------------------|---------------------:|-----------------------:|-------------------------:|---------------------------:|:---------------------|:--------------------|
| asset_06   | asset_06_C00013 | asset_06_C00013 |            13 |   1449 | 2026-01-16 08:56:00+00:00 | 2026-01-17 09:04:00+00:00 |      24.1333 |               24.15    |               1 |              1405 |              96.9634 |          0 |                 44 |               24.9137 |             75.0863 |                             22.5623 |                          77.4377 |           2 | stable       |      3.60959e-05 |    0.000102273 |     0.00306    |         6.61768e-05 |                      1 |                   0 |           0 |                       1 | maintenance,production                       | scheduled_CIP                |                    1 |                      0 |                        1 |                         44 | CIP_extra            | fouling             |
| asset_05   | asset_05_C00004 | asset_05_C00004 |             4 |    953 | 2026-01-05 03:25:00+00:00 | 2026-01-05 19:17:00+00:00 |      15.8667 |               15.8833  |               1 |               916 |              96.1175 |         37 |                  0 |               39.0346 |             60.9654 |                             36.5721 |                          63.4279 |           2 | stable       |      7.42658e-05 |    0.000239753 |     0.00304948 |         0.000165487 |                      1 |                   0 |           0 |                       0 | CIP_acid,CIP_alkaline,production             | scheduled_CIP                |                    1 |                      1 |                        0 |                         37 | scheduled_CIP        | preventive          |
| asset_01   | asset_01_C00018 | asset_01_C00018 |            18 |    699 | 2026-01-25 05:53:00+00:00 | 2026-01-25 17:31:00+00:00 |      11.6333 |               11.65    |               1 |               667 |              95.422  |          0 |                 32 |               50.2146 |             49.7854 |                             47.8261 |                          52.1739 |           2 | stable       |      0.000211643 |    0.000339328 |     0.00291289 |         0.000127685 |                      1 |                   0 |           0 |                       1 | maintenance,production                       | scheduled_CIP                |                    1 |                      0 |                        1 |                         32 | CIP_extra            | fouling             |
| asset_01   | asset_01_C00002 | asset_01_C00002 |             2 |   1483 | 2026-01-02 07:18:00+00:00 | 2026-01-03 08:00:00+00:00 |      24.7    |               24.7167  |               1 |              1441 |              97.1679 |          0 |                 42 |               50.1011 |             49.8989 |                             48.6468 |                          51.3532 |           2 | stable       |      0.000268837 |    9.43167e-05 |     0.00286315 |        -0.00017452  |                      1 |                   0 |           0 |                       1 | maintenance,production                       | scheduled_CIP                |                    1 |                      0 |                        1 |                         42 | CIP_extra            | fouling             |
| asset_06   | asset_06_C00008 | asset_06_C00008 |             8 |   1281 | 2026-01-09 19:36:00+00:00 | 2026-01-10 16:56:00+00:00 |      21.3333 |               21.35    |               1 |              1238 |              96.6432 |         43 |                  0 |               38.7198 |             61.2802 |                             36.5913 |                          63.4087 |           2 | stable       |      4.20692e-05 |    0.000104458 |     0.0028196  |         6.23892e-05 |                      1 |                   0 |           0 |                       0 | CIP_acid,CIP_alkaline,production             | scheduled_CIP                |                    1 |                      1 |                        0 |                         43 | scheduled_CIP        | preventive          |
| asset_05   | asset_05_C00011 | asset_05_C00011 |            11 |   1434 | 2026-01-16 09:42:00+00:00 | 2026-01-17 09:35:00+00:00 |      23.8833 |               23.9     |               1 |              1394 |              97.2106 |         40 |                  0 |               34.3096 |             65.6904 |                             32.4247 |                          67.5753 |           2 | stable       |      0.000157778 |    0.000110952 |     0.0027947  |        -4.68251e-05 |                      1 |                   0 |           0 |                       0 | CIP_acid,CIP_alkaline,production             | CIP_extra                    |                    1 |                      1 |                        0 |                         40 | scheduled_CIP        | preventive          |
| asset_06   | asset_06_C00002 | asset_06_C00002 |             2 |    786 | 2026-01-02 00:25:00+00:00 | 2026-01-02 13:30:00+00:00 |      13.0833 |               13.1     |               1 |               740 |              94.1476 |         46 |                  0 |               61.7048 |             38.2952 |                             59.3243 |                          40.6757 |           2 | stable       |      0.000193202 |    7.21316e-05 |     0.00277072 |        -0.000121071 |                      1 |                   0 |           0 |                       0 | CIP_acid,CIP_alkaline,production             | scheduled_CIP                |                    1 |                      1 |                        0 |                         46 | scheduled_CIP        | preventive          |
| asset_02   | asset_02_C00006 | asset_02_C00006 |             6 |    930 | 2026-01-08 03:06:00+00:00 | 2026-01-08 18:35:00+00:00 |      15.4833 |               15.5     |               1 |               847 |              91.0753 |         45 |                 38 |               30.2151 |             69.7849 |                             23.3766 |                          76.6234 |           2 | stable       |      0.000179547 |    0.000119964 |     0.0027407  |        -5.95829e-05 |                      1 |                   1 |          11 |                       1 | CIP_acid,CIP_alkaline,maintenance,production | scheduled_CIP                |                    2 |                      1 |                        1 |                         83 | scheduled_CIP,unclog | clogging,preventive |
| asset_05   | asset_05_C00006 | asset_05_C00006 |             6 |   1513 | 2026-01-08 02:54:00+00:00 | 2026-01-09 04:06:00+00:00 |      25.2    |               25.2167  |               1 |              1472 |              97.2902 |         27 |                 14 |               38.7971 |             61.2029 |                             37.0924 |                          62.9076 |           2 | stable       |      0.000190868 |    0.000359739 |     0.0027242  |         0.000168872 |                      1 |                   1 |          30 |                       1 | CIP_acid,CIP_alkaline,maintenance,production | scheduled_CIP                |                    2 |                      1 |                        1 |                         41 | scheduled_CIP,unclog | clogging,preventive |
| asset_08   | asset_08_C00001 | asset_08_C00001 |             1 |   1261 | 2026-01-01 03:01:00+00:00 | 2026-01-02 00:01:00+00:00 |      21      |               21.0167  |               1 |              1204 |              95.4798 |         33 |                 24 |               22.3632 |             77.6368 |                             18.6877 |                          81.3123 |           2 | stable       |      5.58936e-05 |    0.000126696 |     0.00271117 |         7.08025e-05 |                      1 |                   1 |          18 |                       1 | CIP_acid,CIP_alkaline,maintenance,production | none                         |                    2 |                      1 |                        1 |                         57 | scheduled_CIP,unclog | clogging,preventive |
| asset_01   | asset_01_C00006 | asset_01_C00006 |             6 |   1398 | 2026-01-08 08:27:00+00:00 | 2026-01-09 07:44:00+00:00 |      23.2833 |               23.3     |               1 |              1356 |              96.9957 |          0 |                 42 |               29.2561 |             70.7439 |                             27.0649 |                          72.9351 |           2 | stable       |      0.000163838 |    0.0001411   |     0.00264632 |        -2.27383e-05 |                      1 |                   0 |           0 |                       1 | maintenance,production                       | scheduled_CIP                |                    1 |                      0 |                        1 |                         42 | CIP_extra            | fouling             |
| asset_06   | asset_06_C00007 | asset_06_C00007 |             7 |    839 | 2026-01-08 13:46:00+00:00 | 2026-01-09 03:44:00+00:00 |      13.9667 |               13.9833  |               1 |               767 |              91.4184 |         45 |                 27 |               40.5244 |             59.4756 |                             34.9413 |                          65.0587 |           2 | stable       |      9.09726e-05 |    4.2067e-05  |     0.00262099 |        -4.89056e-05 |                      1 |                   1 |           8 |                       1 | CIP_acid,CIP_alkaline,maintenance,production | scheduled_CIP                |                    2 |                      1 |                        1 |                         72 | scheduled_CIP,unclog | clogging,preventive |
| asset_04   | asset_04_C00015 | asset_04_C00015 |            15 |    883 | 2026-01-17 15:28:00+00:00 | 2026-01-18 06:10:00+00:00 |      14.7    |               14.7167  |               1 |               848 |              96.0362 |         35 |                  0 |               65.1189 |             34.8811 |                             63.6792 |                          36.3208 |           2 | stable       |      6.58206e-05 |    0.000327095 |     0.00261748 |         0.000261275 |                      1 |                   0 |           0 |                       0 | CIP_acid,CIP_alkaline,production             | scheduled_CIP                |                    1 |                      1 |                        0 |                         35 | scheduled_CIP        | preventive          |
| asset_02   | asset_02_C00013 | asset_02_C00013 |            13 |    481 | 2026-01-16 05:29:00+00:00 | 2026-01-16 13:29:00+00:00 |       8      |                8.01667 |               1 |               442 |              91.8919 |         39 |                  0 |               72.7651 |             27.2349 |                             70.362  |                          29.638  |           2 | advanced     |      0.000134104 |    0.000345345 |     0.00261543 |         0.000211241 |                      1 |                   0 |           0 |                       0 | CIP_acid,CIP_alkaline,production             | scheduled_CIP                |                    1 |                      1 |                        0 |                         39 | scheduled_CIP        | preventive          |
| asset_01   | asset_01_C00010 | asset_01_C00010 |            10 |   1224 | 2026-01-14 03:59:00+00:00 | 2026-01-15 00:22:00+00:00 |      20.3833 |               20.4     |               1 |              1169 |              95.5065 |         37 |                 18 |               29.7386 |             70.2614 |                             26.4328 |                          73.5672 |           2 | stable       |      0.000139384 |    0.000160542 |     0.00257375 |         2.1158e-05  |                      1 |                   1 |          28 |                       1 | CIP_acid,CIP_alkaline,maintenance,production | scheduled_CIP                |                    2 |                      1 |                        1 |                         55 | scheduled_CIP,unclog | clogging,preventive |
| asset_01   | asset_01_C00005 | asset_01_C00005 |             5 |   1486 | 2026-01-06 23:26:00+00:00 | 2026-01-08 00:11:00+00:00 |      24.75   |               24.7667  |               1 |              1431 |              96.2988 |         38 |                 17 |               31.3594 |             68.6406 |                             28.7212 |                          71.2788 |           2 | stable       |      9.31514e-05 |    0.000163838 |     0.00253278 |         7.06862e-05 |                      1 |                   1 |          18 |                       1 | CIP_acid,CIP_alkaline,maintenance,production | scheduled_CIP                |                    2 |                      1 |                        1 |                         55 | scheduled_CIP,unclog | clogging,preventive |
| asset_02   | asset_02_C00015 | asset_02_C00015 |            15 |    654 | 2026-01-18 14:29:00+00:00 | 2026-01-19 01:22:00+00:00 |      10.8833 |               10.9     |               1 |               614 |              93.8838 |         40 |                  0 |               60.2446 |             39.7554 |                             57.6547 |                          42.3453 |           2 | stable       |      0.000108662 |    0.000194075 |     0.00253075 |         8.54136e-05 |                      1 |                   0 |           0 |                       0 | CIP_acid,CIP_alkaline,production             | scheduled_CIP                |                    1 |                      1 |                        0 |                         40 | scheduled_CIP        | preventive          |
| asset_07   | asset_07_C00013 | asset_07_C00013 |            13 |    778 | 2026-01-15 11:31:00+00:00 | 2026-01-16 00:28:00+00:00 |      12.95   |               12.9667  |               1 |               747 |              96.0154 |         31 |                  0 |               52.8278 |             47.1722 |                             50.8701 |                          49.1299 |           2 | stable       |      0.000173202 |    0.000354834 |     0.00252213 |         0.000181632 |                      1 |                   0 |           0 |                       0 | CIP_acid,CIP_alkaline,production             | scheduled_CIP                |                    1 |                      1 |                        0 |                         31 | scheduled_CIP        | preventive          |
| asset_06   | asset_06_C00004 | asset_06_C00004 |             4 |   1226 | 2026-01-04 17:05:00+00:00 | 2026-01-05 13:30:00+00:00 |      20.4167 |               20.4333  |               1 |              1174 |              95.7586 |          0 |                 52 |               22.5122 |             77.4878 |                             19.0801 |                          80.9199 |           2 | stable       |      5.18247e-05 |    9.7764e-06  |     0.00251641 |        -4.20483e-05 |                      1 |                   0 |           0 |                       1 | maintenance,production                       | scheduled_CIP                |                    1 |                      0 |                        1 |                         52 | CIP_extra            | fouling             |
| asset_09   | asset_09_C00007 | asset_09_C00007 |             7 |    954 | 2026-01-08 06:57:00+00:00 | 2026-01-08 22:50:00+00:00 |      15.8833 |               15.9     |               1 |               872 |              91.4046 |         38 |                 44 |               52.6205 |             47.3795 |                             48.1651 |                          51.8349 |           2 | stable       |      8.44868e-05 |    0.000125922 |     0.00251264 |         4.14354e-05 |                      1 |                   1 |          18 |                       1 | CIP_acid,CIP_alkaline,maintenance,production | scheduled_CIP                |                    2 |                      1 |                        1 |                         82 | scheduled_CIP,unclog | clogging,preventive |

## Consistencia de targets

| check_name                                                 |   rows_evaluated |   mismatch_count |   mismatch_pct | detail                                                                                    |   mae_min |
|:-----------------------------------------------------------|-----------------:|-----------------:|---------------:|:------------------------------------------------------------------------------------------|----------:|
| clog_onset_within_15min_vs_time_to_clog_onset_min          |           198615 |                0 |              0 | binary target should match 0 <= time_to_clog_onset_min <= 15                              |       nan |
| fouling_onset_within_30min_vs_time_to_fouling_onset_min    |           198615 |                0 |              0 | binary target should match 0 <= time_to_fouling_onset_min <= 30                           |       nan |
| fouling_stage_vs_physical_thresholds                       |           189755 |                0 |              0 | reported stage should match stage derived from physical thresholds (production rows only) |       nan |
| time_to_clog_onset_min_countdown_step                      |            66147 |                0 |              0 | next time-to-event should roughly equal current minus dt                                  |         0 |
| time_to_fouling_onset_min_countdown_step                   |           131647 |                0 |              0 | next time-to-event should roughly equal current minus dt                                  |         0 |
| ttm_to_planned_cip_min_countdown_step                      |           181833 |                0 |              0 | next time-to-event should roughly equal current minus dt                                  |         0 |
| ttm_to_unplanned_event_min_countdown_step                  |            77936 |                0 |              0 | next time-to-event should roughly equal current minus dt                                  |         0 |
| unplanned_event_within_60min_vs_ttm_to_unplanned_event_min |           198615 |                0 |              0 | binary target should match 0 <= ttm_to_unplanned_event_min <= 60                          |       nan |

## Gráficos clave

### phase_counts
![](plots/phase_counts.png)

### stage_counts
![](plots/stage_counts.png)

### maintenance_type_counts
![](plots/maintenance_type_counts.png)

### maintenance_fault_type_counts
![](plots/maintenance_fault_type_counts.png)

### cycle_duration_hist
![](plots/cycle_duration_hist.png)

### rows_per_cycle_hist
![](plots/rows_per_cycle_hist.png)

### informative_row_pct_by_cycle_hist
![](plots/informative_row_pct_by_cycle_hist.png)

### inter_cycle_gap_hist
![](plots/inter_cycle_gap_hist.png)

### Rf_m2K_W_hist
![](plots/Rf_m2K_W_hist.png)

### m_total_kg_m2_hist
![](plots/m_total_kg_m2_hist.png)

### flow_kg_s_hist
![](plots/flow_kg_s_hist.png)

### dP_kPa_hist
![](plots/dP_kPa_hist.png)

### vibration_mm_s_hist
![](plots/vibration_mm_s_hist.png)

### hot_drop_C_hist
![](plots/hot_drop_C_hist.png)

### cold_lift_C_hist
![](plots/cold_lift_C_hist.png)

### thermal_eff_proxy_hist
![](plots/thermal_eff_proxy_hist.png)

### heat_proxy_hist
![](plots/heat_proxy_hist.png)

### ttm_to_planned_cip_min_hist
![](plots/ttm_to_planned_cip_min_hist.png)

### ttm_to_unplanned_event_min_hist
![](plots/ttm_to_unplanned_event_min_hist.png)

### time_to_fouling_onset_min_hist
![](plots/time_to_fouling_onset_min_hist.png)

## Muestras de ciclos

### asset_02_C00006
![](samples/asset_02_C00006.png)

### asset_05_C00006
![](samples/asset_05_C00006.png)

### asset_08_C00001
![](samples/asset_08_C00001.png)

### asset_06_C00007
![](samples/asset_06_C00007.png)

### asset_01_C00010
![](samples/asset_01_C00010.png)

### asset_01_C00005
![](samples/asset_01_C00005.png)

### asset_09_C00007
![](samples/asset_09_C00007.png)

### asset_02_C00010
![](samples/asset_02_C00010.png)

### asset_06_C00006
![](samples/asset_06_C00006.png)

### asset_06_C00001
![](samples/asset_06_C00001.png)
