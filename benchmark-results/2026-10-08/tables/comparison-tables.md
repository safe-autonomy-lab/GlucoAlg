# Complete shield comparisons — snapshot 2026-10-08T12:08:31.720550+00:00

Unseen patients #002–#010; requested seven-day horizon. TIR is the fraction of the **requested** horizon with recorded CGM 70–180 mg/dL; early endings reduce coverage and remain in the denominator.

Complete six-algorithm outcomes: 3,240/3,240 predictive batches accepted in 33 original windows. None/static controls are complete. Later live results are excluded from this frozen snapshot.

Each numeric predictive row contains the complete design: three policy seeds × three predictor seeds × nine unseen patients × three resets, or 243 episodes. None/static controls use 81 physical episodes and are matched to the predictive arm. Means weight resets, patients, predictor seeds within policy, then the three policy seeds equally. ± is sample SD across three policy means. P is the number of policy seeds; B is policy/predictor batches out of nine. Exact IDs are in the CSV. Deterministic and stochastic policies are reported separately.

Selected penalties were chosen from absolute magnitudes 1/5/10/15/20 on seen #001 one-day validation only. no_eligible_penalty means no candidate met validation eligibility; model_unavailable means required training support was unavailable. These are scientific NA outcomes, not unfinished jobs. Selected10 aliases fixed10. Unsafe and early-ended valid outcomes remain included.

These are authenticated accepted-proof summaries. No simulation or scientific checker was rerun. All experiment stages and original window audits are complete. This table export uses accepted proof summaries; the separate archive-wide native-metric supplement is not claimed by this export.

## Six additional algorithms: deterministic

| Algorithm | Type | Cohort | Arm / penalty | State | P | B/9 | Matched none TIR % | Static TIR % | Predictive TIR % | Δ vs none pp | Δ vs static pp |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CUP | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 92.06 ± 2.08 | 88.51 ± 2.86 | 88.49 ± 2.90 | -3.57 | -0.02 |
| CUP | t1d | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CUP | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 49.74 ± 5.94 | 46.17 ± 6.21 | 46.86 ± 5.68 | -2.88 | +0.69 |
| CUP | t1d | adult | selected / 1 | complete | 3 | 9 | 49.74 ± 5.94 | 46.17 ± 6.21 | 46.62 ± 5.90 | -3.12 | +0.45 |
| CUP | t1d | child | fixed10 / 10 | complete | 3 | 9 | 72.61 ± 5.16 | 83.29 ± 4.17 | 84.92 ± 3.32 | +12.31 | +1.63 |
| CUP | t1d | child | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CUP | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 72.70 ± 25.18 | 82.02 ± 12.03 | 82.78 ± 11.34 | +10.08 | +0.76 |
| CUP | t2d | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CUP | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 82.04 ± 5.62 | 82.93 ± 5.32 | 83.06 ± 5.21 | +1.03 | +0.13 |
| CUP | t2d | adult | selected / 1 | complete | 3 | 9 | 82.04 ± 5.62 | 82.93 ± 5.32 | 82.73 ± 5.40 | +0.69 | -0.21 |
| CUP | t2d | child | fixed10 / 10 | complete | 3 | 9 | 89.16 ± 1.85 | 90.08 ± 1.18 | 90.52 ± 0.81 | +1.36 | +0.44 |
| CUP | t2d | child | selected / 1 | complete | 3 | 9 | 89.16 ± 1.85 | 90.08 ± 1.18 | 90.41 ± 1.03 | +1.25 | +0.34 |
| CUP | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 95.39 ± 1.18 | 96.17 ± 0.79 | 96.17 ± 0.82 | +0.78 | -0.00 |
| CUP | t2d_no_pump | adolescent | selected / 1 | complete | 3 | 9 | 95.39 ± 1.18 | 96.17 ± 0.79 | 96.17 ± 0.80 | +0.78 | -0.00 |
| CUP | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 93.26 ± 3.73 | 93.74 ± 3.86 | 93.74 ± 3.86 | +0.49 | +0.00 |
| CUP | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 93.26 ± 3.73 | 93.74 ± 3.86 | 93.74 ± 3.86 | +0.49 | +0.00 |
| CUP | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 77.03 ± 14.55 | 79.36 ± 10.21 | 85.81 ± 5.91 | +8.78 | +6.45 |
| CUP | t2d_no_pump | child | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| FOCOPS | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 88.77 ± 4.67 | 87.04 ± 4.51 | 87.12 ± 4.43 | -1.65 | +0.08 |
| FOCOPS | t1d | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| FOCOPS | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 48.40 ± 10.32 | 44.05 ± 9.01 | 44.29 ± 8.96 | -4.10 | +0.25 |
| FOCOPS | t1d | adult | selected / 1 | complete | 3 | 9 | 48.40 ± 10.32 | 44.05 ± 9.01 | 44.19 ± 9.08 | -4.21 | +0.14 |
| FOCOPS | t1d | child | fixed10 / 10 | complete | 3 | 9 | 81.15 ± 9.41 | 88.27 ± 4.87 | 90.17 ± 3.04 | +9.03 | +1.90 |
| FOCOPS | t1d | child | selected / 10 | complete | 3 | 9 | 81.15 ± 9.41 | 88.27 ± 4.87 | 90.17 ± 3.04 | +9.03 | +1.90 |
| FOCOPS | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 97.09 ± 0.62 | 97.12 ± 0.65 | 97.10 ± 0.67 | +0.02 | -0.01 |
| FOCOPS | t2d | adolescent | selected / 1 | complete | 3 | 9 | 97.09 ± 0.62 | 97.12 ± 0.65 | 97.11 ± 0.65 | +0.03 | -0.00 |
| FOCOPS | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 69.42 ± 1.51 | 68.86 ± 2.18 | 69.23 ± 1.66 | -0.19 | +0.37 |
| FOCOPS | t2d | adult | selected / 1 | complete | 3 | 9 | 69.42 ± 1.51 | 68.86 ± 2.18 | 68.83 ± 1.83 | -0.59 | -0.03 |
| FOCOPS | t2d | child | fixed10 / 10 | complete | 3 | 9 | 81.12 ± 13.90 | 89.00 ± 4.48 | 90.89 ± 4.45 | +9.77 | +1.89 |
| FOCOPS | t2d | child | selected / 10 | complete | 3 | 9 | 81.12 ± 13.90 | 89.00 ± 4.48 | 90.89 ± 4.45 | +9.77 | +1.89 |
| FOCOPS | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 78.98 ± 15.59 | 88.06 ± 8.58 | 89.34 ± 7.45 | +10.36 | +1.28 |
| FOCOPS | t2d_no_pump | adolescent | selected / 1 | complete | 3 | 9 | 78.98 ± 15.59 | 88.06 ± 8.58 | 89.11 ± 7.62 | +10.13 | +1.05 |
| FOCOPS | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 91.03 ± 0.59 | 91.72 ± 0.46 | 91.72 ± 0.46 | +0.70 | +0.00 |
| FOCOPS | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 91.03 ± 0.59 | 91.72 ± 0.46 | 91.72 ± 0.46 | +0.70 | +0.00 |
| FOCOPS | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 87.66 ± 1.42 | 91.22 ± 1.64 | 92.81 ± 0.61 | +5.16 | +1.59 |
| FOCOPS | t2d_no_pump | child | selected / 10 | complete | 3 | 9 | 87.66 ± 1.42 | 91.22 ± 1.64 | 92.81 ± 0.61 | +5.16 | +1.59 |
| CRPO | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 85.91 ± 3.49 | 85.72 ± 4.01 | 85.54 ± 4.12 | -0.37 | -0.18 |
| CRPO | t1d | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CRPO | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 73.21 ± 10.72 | 69.58 ± 10.43 | 69.48 ± 10.51 | -3.73 | -0.10 |
| CRPO | t1d | adult | selected / 1 | complete | 3 | 9 | 73.21 ± 10.72 | 69.58 ± 10.43 | 69.56 ± 10.42 | -3.66 | -0.03 |
| CRPO | t1d | child | fixed10 / 10 | complete | 3 | 9 | 73.86 ± 9.91 | 79.72 ± 8.73 | 80.50 ± 7.29 | +6.64 | +0.78 |
| CRPO | t1d | child | selected / 1 | complete | 3 | 9 | 73.86 ± 9.91 | 79.72 ± 8.73 | 79.73 ± 9.12 | +5.87 | +0.01 |
| CRPO | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 39.44 ± 39.25 | 48.79 ± 36.08 | 69.86 ± 20.19 | +30.41 | +21.06 |
| CRPO | t2d | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CRPO | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 85.05 ± 5.39 | 85.65 ± 4.65 | 85.56 ± 4.91 | +0.50 | -0.09 |
| CRPO | t2d | adult | selected / 1 | complete | 3 | 9 | 85.05 ± 5.39 | 85.65 ± 4.65 | 85.55 ± 4.78 | +0.50 | -0.09 |
| CRPO | t2d | child | fixed10 / 10 | complete | 3 | 9 | 61.30 ± 21.33 | 70.07 ± 17.77 | 72.00 ± 18.20 | +10.70 | +1.93 |
| CRPO | t2d | child | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CRPO | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 79.57 ± 14.19 | 83.81 ± 12.39 | 86.05 ± 9.22 | +6.48 | +2.24 |
| CRPO | t2d_no_pump | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CRPO | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 92.44 ± 1.76 | 92.93 ± 1.04 | 92.96 ± 0.99 | +0.52 | +0.03 |
| CRPO | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 92.44 ± 1.76 | 92.93 ± 1.04 | 92.93 ± 1.04 | +0.49 | +0.00 |
| CRPO | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 63.60 ± 15.07 | 71.51 ± 14.32 | 73.65 ± 12.67 | +10.06 | +2.15 |
| CRPO | t2d_no_pump | child | selected / 15 | complete | 3 | 9 | 63.60 ± 15.07 | 71.51 ± 14.32 | 75.17 ± 10.24 | +11.58 | +3.67 |
| RCPO | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 86.88 ± 4.60 | 86.84 ± 4.50 | 86.85 ± 4.52 | -0.03 | +0.01 |
| RCPO | t1d | adolescent | selected / 1 | complete | 3 | 9 | 86.88 ± 4.60 | 86.84 ± 4.50 | 86.85 ± 4.52 | -0.03 | +0.01 |
| RCPO | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 78.23 ± 7.53 | 76.35 ± 7.97 | 76.47 ± 7.76 | -1.76 | +0.12 |
| RCPO | t1d | adult | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| RCPO | t1d | child | fixed10 / 10 | complete | 3 | 9 | 69.43 ± 3.81 | 71.10 ± 2.91 | 72.45 ± 0.76 | +3.01 | +1.34 |
| RCPO | t1d | child | selected / 5 | complete | 3 | 9 | 69.43 ± 3.81 | 71.10 ± 2.91 | 70.08 ± 3.44 | +0.65 | -1.02 |
| RCPO | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 91.62 ± 1.20 | 91.45 ± 2.73 | 91.43 ± 2.70 | -0.19 | -0.02 |
| RCPO | t2d | adolescent | selected / 1 | complete | 3 | 9 | 91.62 ± 1.20 | 91.45 ± 2.73 | 91.45 ± 2.73 | -0.17 | +0.00 |
| RCPO | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 84.32 ± 2.92 | 85.87 ± 3.45 | 85.85 ± 3.36 | +1.53 | -0.02 |
| RCPO | t2d | adult | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| RCPO | t2d | child | fixed10 / 10 | complete | 3 | 9 | 64.27 ± 8.36 | 63.74 ± 8.15 | 63.87 ± 8.27 | -0.40 | +0.13 |
| RCPO | t2d | child | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| RCPO | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 92.87 ± 3.20 | 94.02 ± 2.04 | 94.03 ± 2.06 | +1.16 | +0.01 |
| RCPO | t2d_no_pump | adolescent | selected / 5 | complete | 3 | 9 | 92.87 ± 3.20 | 94.02 ± 2.04 | 94.03 ± 2.06 | +1.16 | +0.01 |
| RCPO | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 92.29 ± 1.74 | 92.43 ± 1.77 | 92.47 ± 1.75 | +0.17 | +0.04 |
| RCPO | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 92.29 ± 1.74 | 92.43 ± 1.77 | 92.43 ± 1.75 | +0.14 | +0.00 |
| RCPO | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 67.25 ± 15.59 | 71.61 ± 17.26 | 73.22 ± 17.47 | +5.97 | +1.61 |
| RCPO | t2d_no_pump | child | selected / 15 | complete | 3 | 9 | 67.25 ± 15.59 | 71.61 ± 17.26 | 72.45 ± 18.13 | +5.20 | +0.84 |
| TRPOLag | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 87.93 ± 2.57 | 85.55 ± 5.18 | 85.59 ± 5.20 | -2.34 | +0.04 |
| TRPOLag | t1d | adolescent | selected / 1 | complete | 3 | 9 | 87.93 ± 2.57 | 85.55 ± 5.18 | 85.55 ± 5.18 | -2.37 | +0.00 |
| TRPOLag | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 87.26 ± 5.33 | 86.10 ± 5.04 | 86.26 ± 5.34 | -1.00 | +0.16 |
| TRPOLag | t1d | adult | selected / 1 | complete | 3 | 9 | 87.26 ± 5.33 | 86.10 ± 5.04 | 86.10 ± 5.04 | -1.16 | +0.00 |
| TRPOLag | t1d | child | fixed10 / 10 | complete | 3 | 9 | 68.29 ± 2.71 | 75.72 ± 3.82 | 75.63 ± 3.52 | +7.35 | -0.08 |
| TRPOLag | t1d | child | selected / 10 | complete | 3 | 9 | 68.29 ± 2.71 | 75.72 ± 3.82 | 75.63 ± 3.52 | +7.35 | -0.08 |
| TRPOLag | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 91.49 ± 6.15 | 93.62 ± 3.62 | 93.62 ± 3.62 | +2.13 | -0.00 |
| TRPOLag | t2d | adolescent | selected / 1 | complete | 3 | 9 | 91.49 ± 6.15 | 93.62 ± 3.62 | 93.62 ± 3.62 | +2.13 | +0.00 |
| TRPOLag | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 87.29 ± 4.54 | 88.74 ± 5.12 | 88.88 ± 5.21 | +1.59 | +0.14 |
| TRPOLag | t2d | adult | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| TRPOLag | t2d | child | fixed10 / 10 | complete | 3 | 9 | 59.05 ± 1.53 | 67.32 ± 2.95 | 67.82 ± 2.80 | +8.77 | +0.50 |
| TRPOLag | t2d | child | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| TRPOLag | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 85.63 ± 5.99 | 86.84 ± 5.29 | 86.86 ± 5.45 | +1.23 | +0.02 |
| TRPOLag | t2d_no_pump | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| TRPOLag | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 94.56 ± 1.59 | 94.83 ± 1.39 | 94.84 ± 1.37 | +0.28 | +0.02 |
| TRPOLag | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 94.56 ± 1.59 | 94.83 ± 1.39 | 94.83 ± 1.39 | +0.27 | +0.00 |
| TRPOLag | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 67.96 ± 16.24 | 71.22 ± 16.47 | 74.26 ± 18.19 | +6.30 | +3.04 |
| TRPOLag | t2d_no_pump | child | selected / 15 | complete | 3 | 9 | 67.96 ± 16.24 | 71.22 ± 16.47 | 73.25 ± 18.71 | +5.29 | +2.03 |
| PCPO | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 87.99 ± 6.06 | 87.37 ± 5.90 | 87.42 ± 5.88 | -0.57 | +0.05 |
| PCPO | t1d | adolescent | selected / 1 | complete | 3 | 9 | 87.99 ± 6.06 | 87.37 ± 5.90 | 87.37 ± 5.90 | -0.62 | +0.00 |
| PCPO | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 80.43 ± 9.80 | 77.85 ± 10.76 | 77.92 ± 10.79 | -2.52 | +0.06 |
| PCPO | t1d | adult | selected / 1 | complete | 3 | 9 | 80.43 ± 9.80 | 77.85 ± 10.76 | 77.91 ± 10.78 | -2.52 | +0.06 |
| PCPO | t1d | child | fixed10 / 10 | complete | 3 | 9 | 75.87 ± 6.18 | 84.22 ± 6.14 | 85.76 ± 3.80 | +9.89 | +1.54 |
| PCPO | t1d | child | selected / 1 | complete | 3 | 9 | 75.87 ± 6.18 | 84.22 ± 6.14 | 85.10 ± 5.28 | +9.23 | +0.88 |
| PCPO | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 94.34 ± 1.34 | 94.27 ± 1.25 | 94.27 ± 1.25 | -0.07 | +0.00 |
| PCPO | t2d | adolescent | selected / 1 | complete | 3 | 9 | 94.34 ± 1.34 | 94.27 ± 1.25 | 94.27 ± 1.25 | -0.07 | +0.00 |
| PCPO | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 81.91 ± 13.41 | 81.87 ± 14.77 | 81.78 ± 14.96 | -0.13 | -0.09 |
| PCPO | t2d | adult | selected / 1 | complete | 3 | 9 | 81.91 ± 13.41 | 81.87 ± 14.77 | 81.87 ± 14.79 | -0.04 | -0.00 |
| PCPO | t2d | child | fixed10 / 10 | complete | 3 | 9 | 70.95 ± 22.74 | 78.77 ± 11.20 | 80.22 ± 9.38 | +9.27 | +1.46 |
| PCPO | t2d | child | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| PCPO | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 77.29 ± 7.27 | 77.53 ± 6.97 | 82.26 ± 3.56 | +4.98 | +4.73 |
| PCPO | t2d_no_pump | adolescent | selected / 10 | complete | 3 | 9 | 77.29 ± 7.27 | 77.53 ± 6.97 | 82.26 ± 3.56 | +4.98 | +4.73 |
| PCPO | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 93.44 ± 2.17 | 93.58 ± 2.24 | 93.62 ± 2.20 | +0.18 | +0.04 |
| PCPO | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 93.44 ± 2.17 | 93.58 ± 2.24 | 93.60 ± 2.21 | +0.16 | +0.02 |
| PCPO | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 54.07 ± 17.14 | 59.30 ± 17.87 | 63.43 ± 15.87 | +9.36 | +4.13 |
| PCPO | t2d_no_pump | child | selected / 1 | complete | 3 | 9 | 54.07 ± 17.14 | 59.30 ± 17.87 | 60.22 ± 17.77 | +6.15 | +0.92 |

### Safety and coverage: deterministic

| Algorithm | Type | Cohort | Arm | State | B/9 | Observed TIR % none → shield | Below54 % none → shield | Early ending % none → shield | Physiological ending % none → shield | Coverage % none → shield |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| CUP | t1d | adolescent | fixed10 | complete | 9 | 92.21 → 90.31 | 0.10 → 0.00 | 1.23 → 3.70 | 1.23 → 3.70 | 99.81 → 97.95 |
| CUP | t1d | adult | fixed10 | complete | 9 | 55.05 → 52.17 | 0.67 → 0.04 | 17.28 → 18.52 | 17.28 → 18.52 | 88.73 → 88.59 |
| CUP | t1d | adult | selected | complete | 9 | 55.05 → 51.93 | 0.67 → 0.05 | 17.28 → 18.52 | 17.28 → 18.52 | 88.73 → 88.59 |
| CUP | t1d | child | fixed10 | complete | 9 | 77.82 → 86.63 | 10.32 → 3.42 | 12.35 → 4.12 | 12.35 → 4.12 | 92.87 → 97.88 |
| CUP | t2d | adolescent | fixed10 | complete | 9 | 80.16 → 82.78 | 3.62 → 0.44 | 19.75 → 0.00 | 20.99 → 0.00 | 89.36 → 100.00 |
| CUP | t2d | adult | fixed10 | complete | 9 | 82.04 → 83.06 | 0.72 → 0.05 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d | adult | selected | complete | 9 | 82.04 → 82.73 | 0.72 → 0.07 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d | child | fixed10 | complete | 9 | 89.16 → 90.52 | 1.67 → 1.24 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d | child | selected | complete | 9 | 89.16 → 90.41 | 1.67 → 1.33 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d_no_pump | adolescent | fixed10 | complete | 9 | 95.39 → 96.17 | 0.36 → 0.10 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d_no_pump | adolescent | selected | complete | 9 | 95.39 → 96.17 | 0.36 → 0.11 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d_no_pump | adult | fixed10 | complete | 9 | 93.26 → 93.74 | 0.07 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d_no_pump | adult | selected | complete | 9 | 93.26 → 93.74 | 0.07 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d_no_pump | child | fixed10 | complete | 9 | 77.98 → 87.28 | 9.67 → 2.51 | 2.47 → 3.70 | 2.47 → 3.70 | 98.64 → 98.11 |
| FOCOPS | t1d | adolescent | fixed10 | complete | 9 | 88.77 → 87.12 | 0.15 → 0.01 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t1d | adult | fixed10 | complete | 9 | 48.40 → 44.29 | 0.66 → 0.02 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t1d | adult | selected | complete | 9 | 48.40 → 44.19 | 0.66 → 0.03 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t1d | child | fixed10 | complete | 9 | 83.31 → 90.66 | 6.51 → 0.28 | 3.70 → 3.29 | 3.70 → 3.29 | 97.31 → 99.42 |
| FOCOPS | t1d | child | selected | complete | 9 | 83.31 → 90.66 | 6.51 → 0.28 | 3.70 → 3.29 | 3.70 → 3.29 | 97.31 → 99.42 |
| FOCOPS | t2d | adolescent | fixed10 | complete | 9 | 97.09 → 97.10 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t2d | adolescent | selected | complete | 9 | 97.09 → 97.11 | 0.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t2d | adult | fixed10 | complete | 9 | 69.42 → 69.23 | 0.20 → 0.01 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t2d | adult | selected | complete | 9 | 69.42 → 68.83 | 0.20 → 0.01 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t2d | child | fixed10 | complete | 9 | 83.26 → 91.05 | 7.02 → 0.44 | 4.94 → 0.41 | 4.94 → 0.41 | 96.92 → 99.80 |
| FOCOPS | t2d | child | selected | complete | 9 | 83.26 → 91.05 | 7.02 → 0.44 | 4.94 → 0.41 | 4.94 → 0.41 | 96.92 → 99.80 |
| FOCOPS | t2d_no_pump | adolescent | fixed10 | complete | 9 | 84.05 → 89.34 | 4.92 → 0.32 | 12.35 → 0.00 | 12.35 → 0.00 | 93.28 → 100.00 |
| FOCOPS | t2d_no_pump | adolescent | selected | complete | 9 | 84.05 → 89.11 | 4.92 → 0.33 | 12.35 → 0.00 | 12.35 → 0.00 | 93.28 → 100.00 |
| FOCOPS | t2d_no_pump | adult | fixed10 | complete | 9 | 91.03 → 91.72 | 0.10 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t2d_no_pump | adult | selected | complete | 9 | 91.03 → 91.72 | 0.10 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t2d_no_pump | child | fixed10 | complete | 9 | 87.66 → 92.99 | 2.62 → 0.17 | 0.00 → 0.41 | 0.00 → 0.41 | 100.00 → 99.80 |
| FOCOPS | t2d_no_pump | child | selected | complete | 9 | 87.66 → 92.99 | 2.62 → 0.17 | 0.00 → 0.41 | 0.00 → 0.41 | 100.00 → 99.80 |
| CRPO | t1d | adolescent | fixed10 | complete | 9 | 85.91 → 85.54 | 0.24 → 0.06 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t1d | adult | fixed10 | complete | 9 | 74.22 → 71.14 | 1.18 → 0.71 | 2.47 → 3.70 | 2.47 → 3.70 | 98.16 → 97.41 |
| CRPO | t1d | adult | selected | complete | 9 | 74.22 → 71.22 | 1.18 → 0.71 | 2.47 → 3.70 | 2.47 → 3.70 | 98.16 → 97.41 |
| CRPO | t1d | child | fixed10 | complete | 9 | 79.06 → 82.93 | 7.57 → 4.83 | 22.22 → 11.11 | 22.22 → 11.11 | 91.77 → 96.44 |
| CRPO | t1d | child | selected | complete | 9 | 79.06 → 82.09 | 7.57 → 5.19 | 22.22 → 11.52 | 22.22 → 11.52 | 91.77 → 96.08 |
| CRPO | t2d | adolescent | fixed10 | complete | 9 | 84.93 → 86.93 | 7.98 → 3.55 | 70.37 → 41.15 | 70.37 → 41.15 | 44.20 → 78.84 |
| CRPO | t2d | adult | fixed10 | complete | 9 | 85.05 → 85.56 | 2.34 → 1.20 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t2d | adult | selected | complete | 9 | 85.05 → 85.55 | 2.34 → 1.17 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t2d | child | fixed10 | complete | 9 | 69.51 → 78.20 | 16.64 → 9.83 | 19.75 → 10.70 | 19.75 → 10.70 | 85.89 → 91.50 |
| CRPO | t2d_no_pump | adolescent | fixed10 | complete | 9 | 87.99 → 89.43 | 3.30 → 2.28 | 22.22 → 9.88 | 22.22 → 9.88 | 88.99 → 95.55 |
| CRPO | t2d_no_pump | adult | fixed10 | complete | 9 | 92.44 → 92.96 | 0.34 → 0.16 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t2d_no_pump | adult | selected | complete | 9 | 92.44 → 92.93 | 0.34 → 0.17 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t2d_no_pump | child | fixed10 | complete | 9 | 75.29 → 80.88 | 11.94 → 5.68 | 34.57 → 22.63 | 34.57 → 22.63 | 84.80 → 90.27 |
| CRPO | t2d_no_pump | child | selected | complete | 9 | 75.29 → 81.23 | 11.94 → 5.49 | 34.57 → 21.81 | 34.57 → 21.81 | 84.80 → 92.19 |
| RCPO | t1d | adolescent | fixed10 | complete | 9 | 87.69 → 87.66 | 0.02 → 0.00 | 3.70 → 3.70 | 3.70 → 3.70 | 98.51 → 98.51 |
| RCPO | t1d | adolescent | selected | complete | 9 | 87.69 → 87.66 | 0.02 → 0.00 | 3.70 → 3.70 | 3.70 → 3.70 | 98.51 → 98.51 |
| RCPO | t1d | adult | fixed10 | complete | 9 | 79.39 → 77.19 | 2.13 → 1.55 | 6.17 → 3.70 | 6.17 → 3.70 | 97.94 → 98.33 |
| RCPO | t1d | child | fixed10 | complete | 9 | 79.53 → 82.22 | 8.68 → 5.52 | 19.75 → 21.81 | 19.75 → 21.81 | 86.35 → 87.28 |
| RCPO | t1d | child | selected | complete | 9 | 79.53 → 81.78 | 8.68 → 5.59 | 19.75 → 25.51 | 19.75 → 25.51 | 86.35 → 84.62 |
| RCPO | t2d | adolescent | fixed10 | complete | 9 | 92.01 → 92.44 | 0.74 → 0.39 | 1.23 → 2.47 | 1.23 → 2.47 | 99.53 → 98.79 |
| RCPO | t2d | adolescent | selected | complete | 9 | 92.01 → 92.46 | 0.74 → 0.41 | 1.23 → 2.47 | 1.23 → 2.47 | 99.53 → 98.79 |
| RCPO | t2d | adult | fixed10 | complete | 9 | 84.32 → 85.85 | 2.04 → 0.82 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| RCPO | t2d | child | fixed10 | complete | 9 | 70.81 → 73.65 | 15.85 → 13.66 | 20.99 → 24.28 | 20.99 → 24.28 | 90.73 → 86.65 |
| RCPO | t2d_no_pump | adolescent | fixed10 | complete | 9 | 94.59 → 94.89 | 0.35 → 0.19 | 2.47 → 1.23 | 2.47 → 1.23 | 98.19 → 99.09 |
| RCPO | t2d_no_pump | adolescent | selected | complete | 9 | 94.59 → 94.89 | 0.35 → 0.19 | 2.47 → 1.23 | 2.47 → 1.23 | 98.19 → 99.09 |
| RCPO | t2d_no_pump | adult | fixed10 | complete | 9 | 92.29 → 92.47 | 0.08 → 0.04 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| RCPO | t2d_no_pump | adult | selected | complete | 9 | 92.29 → 92.43 | 0.08 → 0.04 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| RCPO | t2d_no_pump | child | fixed10 | complete | 9 | 72.01 → 76.05 | 13.82 → 9.40 | 16.05 → 11.52 | 16.05 → 11.52 | 92.73 → 95.26 |
| RCPO | t2d_no_pump | child | selected | complete | 9 | 72.01 → 76.08 | 13.82 → 9.93 | 16.05 → 15.23 | 16.05 → 15.23 | 92.73 → 93.75 |
| TRPOLag | t1d | adolescent | fixed10 | complete | 9 | 89.61 → 88.59 | 0.00 → 0.00 | 3.70 → 6.17 | 3.70 → 6.17 | 97.91 → 96.06 |
| TRPOLag | t1d | adolescent | selected | complete | 9 | 89.61 → 88.56 | 0.00 → 0.00 | 3.70 → 6.17 | 3.70 → 6.17 | 97.91 → 96.06 |
| TRPOLag | t1d | adult | fixed10 | complete | 9 | 87.54 → 86.26 | 1.97 → 1.14 | 1.23 → 0.00 | 1.23 → 0.00 | 99.62 → 100.00 |
| TRPOLag | t1d | adult | selected | complete | 9 | 87.54 → 86.10 | 1.97 → 1.14 | 1.23 → 0.00 | 1.23 → 0.00 | 99.62 → 100.00 |
| TRPOLag | t1d | child | fixed10 | complete | 9 | 79.44 → 82.08 | 8.97 → 6.81 | 23.46 → 16.87 | 23.46 → 16.87 | 85.02 → 91.13 |
| TRPOLag | t1d | child | selected | complete | 9 | 79.44 → 82.08 | 8.97 → 6.81 | 23.46 → 16.87 | 23.46 → 16.87 | 85.02 → 91.13 |
| TRPOLag | t2d | adolescent | fixed10 | complete | 9 | 93.50 → 93.77 | 0.50 → 0.24 | 4.94 → 1.23 | 4.94 → 1.23 | 97.68 → 99.81 |
| TRPOLag | t2d | adolescent | selected | complete | 9 | 93.50 → 93.78 | 0.50 → 0.24 | 4.94 → 1.23 | 4.94 → 1.23 | 97.68 → 99.81 |
| TRPOLag | t2d | adult | fixed10 | complete | 9 | 87.29 → 88.88 | 0.51 → 0.07 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| TRPOLag | t2d | child | fixed10 | complete | 9 | 68.05 → 76.84 | 18.65 → 11.73 | 18.52 → 18.11 | 18.52 → 18.11 | 87.00 → 88.08 |
| TRPOLag | t2d_no_pump | adolescent | fixed10 | complete | 9 | 89.58 → 90.34 | 2.51 → 1.97 | 11.11 → 7.41 | 11.11 → 7.82 | 95.38 → 95.98 |
| TRPOLag | t2d_no_pump | adult | fixed10 | complete | 9 | 94.56 → 94.84 | 0.12 → 0.07 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| TRPOLag | t2d_no_pump | adult | selected | complete | 9 | 94.56 → 94.83 | 0.12 → 0.07 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| TRPOLag | t2d_no_pump | child | fixed10 | complete | 9 | 72.20 → 76.26 | 13.38 → 9.37 | 11.11 → 7.82 | 11.11 → 7.82 | 93.58 → 96.24 |
| TRPOLag | t2d_no_pump | child | selected | complete | 9 | 72.20 → 76.54 | 13.38 → 9.81 | 11.11 → 12.76 | 11.11 → 12.76 | 93.58 → 94.14 |
| PCPO | t1d | adolescent | fixed10 | complete | 9 | 88.12 → 87.55 | 0.03 → 0.02 | 1.23 → 1.23 | 1.23 → 1.23 | 99.81 → 99.81 |
| PCPO | t1d | adolescent | selected | complete | 9 | 88.12 → 87.50 | 0.03 → 0.02 | 1.23 → 1.23 | 1.23 → 1.23 | 99.81 → 99.81 |
| PCPO | t1d | adult | fixed10 | complete | 9 | 80.43 → 77.94 | 1.02 → 0.64 | 0.00 → 1.23 | 0.00 → 1.23 | 100.00 → 99.97 |
| PCPO | t1d | adult | selected | complete | 9 | 80.43 → 77.94 | 1.02 → 0.64 | 0.00 → 1.23 | 0.00 → 1.23 | 100.00 → 99.97 |
| PCPO | t1d | child | fixed10 | complete | 9 | 85.28 → 87.90 | 5.75 → 2.31 | 19.75 → 6.58 | 19.75 → 6.58 | 87.55 → 97.32 |
| PCPO | t1d | child | selected | complete | 9 | 85.28 → 87.90 | 5.75 → 2.19 | 19.75 → 9.47 | 19.75 → 9.47 | 87.55 → 96.53 |
| PCPO | t2d | adolescent | fixed10 | complete | 9 | 94.34 → 94.27 | 0.24 → 0.22 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d | adolescent | selected | complete | 9 | 94.34 → 94.27 | 0.24 → 0.22 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d | adult | fixed10 | complete | 9 | 81.91 → 81.78 | 0.43 → 0.24 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d | adult | selected | complete | 9 | 81.91 → 81.87 | 0.43 → 0.25 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d | child | fixed10 | complete | 9 | 74.64 → 80.81 | 12.82 → 6.89 | 11.11 → 3.29 | 11.11 → 3.29 | 91.99 → 99.20 |
| PCPO | t2d_no_pump | adolescent | fixed10 | complete | 9 | 80.16 → 83.66 | 5.37 → 2.78 | 7.41 → 3.70 | 7.41 → 3.70 | 96.22 → 98.33 |
| PCPO | t2d_no_pump | adolescent | selected | complete | 9 | 80.16 → 83.66 | 5.37 → 2.78 | 7.41 → 3.70 | 7.41 → 3.70 | 96.22 → 98.33 |
| PCPO | t2d_no_pump | adult | fixed10 | complete | 9 | 93.44 → 93.62 | 0.09 → 0.09 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d_no_pump | adult | selected | complete | 9 | 93.44 → 93.60 | 0.09 → 0.07 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d_no_pump | child | fixed10 | complete | 9 | 61.09 → 69.27 | 21.99 → 14.29 | 23.46 → 18.11 | 23.46 → 18.11 | 87.74 → 90.15 |
| PCPO | t2d_no_pump | child | selected | complete | 9 | 61.09 → 67.12 | 21.99 → 14.64 | 23.46 → 19.75 | 23.46 → 19.75 | 87.74 → 88.26 |

## Six additional algorithms: stochastic

| Algorithm | Type | Cohort | Arm / penalty | State | P | B/9 | Matched none TIR % | Static TIR % | Predictive TIR % | Δ vs none pp | Δ vs static pp |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CUP | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 92.60 ± 2.52 | 90.18 ± 3.67 | 90.39 ± 3.78 | -2.20 | +0.21 |
| CUP | t1d | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CUP | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 49.96 ± 6.63 | 46.66 ± 6.17 | 47.43 ± 5.56 | -2.53 | +0.77 |
| CUP | t1d | adult | selected / 1 | complete | 3 | 9 | 49.96 ± 6.63 | 46.66 ± 6.17 | 47.06 ± 5.95 | -2.91 | +0.40 |
| CUP | t1d | child | fixed10 / 10 | complete | 3 | 9 | 73.03 ± 4.69 | 81.57 ± 6.22 | 82.74 ± 5.37 | +9.71 | +1.17 |
| CUP | t1d | child | selected / 5 | complete | 3 | 9 | 73.03 ± 4.69 | 81.57 ± 6.22 | 82.66 ± 5.46 | +9.63 | +1.09 |
| CUP | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 74.90 ± 11.33 | 79.43 ± 13.72 | 81.25 ± 12.10 | +6.36 | +1.82 |
| CUP | t2d | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CUP | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 78.50 ± 5.49 | 78.49 ± 6.40 | 78.66 ± 6.11 | +0.16 | +0.16 |
| CUP | t2d | adult | selected / 1 | complete | 3 | 9 | 78.50 ± 5.49 | 78.49 ± 6.40 | 78.51 ± 6.41 | +0.01 | +0.02 |
| CUP | t2d | child | fixed10 / 10 | complete | 3 | 9 | 82.70 ± 5.29 | 87.30 ± 1.94 | 88.68 ± 0.58 | +5.98 | +1.38 |
| CUP | t2d | child | selected / 5 | complete | 3 | 9 | 82.70 ± 5.29 | 87.30 ± 1.94 | 88.82 ± 0.32 | +6.12 | +1.52 |
| CUP | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 90.34 ± 5.33 | 93.68 ± 2.18 | 94.53 ± 1.08 | +4.19 | +0.85 |
| CUP | t2d_no_pump | adolescent | selected / 5 | complete | 3 | 9 | 90.34 ± 5.33 | 93.68 ± 2.18 | 94.51 ± 1.12 | +4.17 | +0.83 |
| CUP | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 93.25 ± 3.61 | 93.88 ± 3.50 | 93.90 ± 3.50 | +0.65 | +0.01 |
| CUP | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 93.25 ± 3.61 | 93.88 ± 3.50 | 93.89 ± 3.49 | +0.64 | +0.01 |
| CUP | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 66.56 ± 12.87 | 74.35 ± 11.94 | 80.72 ± 6.07 | +14.16 | +6.36 |
| CUP | t2d_no_pump | child | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| FOCOPS | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 85.45 ± 11.03 | 85.58 ± 9.88 | 85.96 ± 9.09 | +0.51 | +0.39 |
| FOCOPS | t1d | adolescent | selected / 1 | complete | 3 | 9 | 85.45 ± 11.03 | 85.58 ± 9.88 | 85.88 ± 9.45 | +0.43 | +0.31 |
| FOCOPS | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 47.91 ± 4.08 | 43.01 ± 3.02 | 43.56 ± 2.64 | -4.34 | +0.56 |
| FOCOPS | t1d | adult | selected / 1 | complete | 3 | 9 | 47.91 ± 4.08 | 43.01 ± 3.02 | 43.38 ± 2.79 | -4.53 | +0.37 |
| FOCOPS | t1d | child | fixed10 / 10 | complete | 3 | 9 | 76.35 ± 7.77 | 82.86 ± 9.88 | 84.88 ± 9.45 | +8.52 | +2.01 |
| FOCOPS | t1d | child | selected / 5 | complete | 3 | 9 | 76.35 ± 7.77 | 82.86 ± 9.88 | 85.08 ± 9.09 | +8.73 | +2.22 |
| FOCOPS | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 94.52 ± 2.48 | 96.93 ± 0.11 | 96.94 ± 0.13 | +2.41 | +0.01 |
| FOCOPS | t2d | adolescent | selected / 1 | complete | 3 | 9 | 94.52 ± 2.48 | 96.93 ± 0.11 | 96.94 ± 0.13 | +2.41 | +0.01 |
| FOCOPS | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 70.68 ± 2.18 | 69.54 ± 2.32 | 70.14 ± 1.68 | -0.54 | +0.60 |
| FOCOPS | t2d | adult | selected / 1 | complete | 3 | 9 | 70.68 ± 2.18 | 69.54 ± 2.32 | 69.90 ± 2.04 | -0.78 | +0.36 |
| FOCOPS | t2d | child | fixed10 / 10 | complete | 3 | 9 | 72.54 ± 13.07 | 84.59 ± 5.12 | 87.65 ± 4.82 | +15.11 | +3.07 |
| FOCOPS | t2d | child | selected / 15 | complete | 3 | 9 | 72.54 ± 13.07 | 84.59 ± 5.12 | 87.84 ± 4.51 | +15.30 | +3.25 |
| FOCOPS | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 58.10 ± 30.83 | 74.67 ± 19.00 | 82.58 ± 12.25 | +24.48 | +7.91 |
| FOCOPS | t2d_no_pump | adolescent | selected / 10 | complete | 3 | 9 | 58.10 ± 30.83 | 74.67 ± 19.00 | 82.58 ± 12.25 | +24.48 | +7.91 |
| FOCOPS | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 91.42 ± 0.58 | 92.08 ± 0.65 | 91.68 ± 0.49 | +0.25 | -0.40 |
| FOCOPS | t2d_no_pump | adult | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| FOCOPS | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 82.53 ± 3.57 | 85.59 ± 1.46 | 89.55 ± 0.82 | +7.02 | +3.96 |
| FOCOPS | t2d_no_pump | child | selected / 10 | complete | 3 | 9 | 82.53 ± 3.57 | 85.59 ± 1.46 | 89.55 ± 0.82 | +7.02 | +3.96 |
| CRPO | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 85.21 ± 3.29 | 85.15 ± 3.74 | 85.05 ± 3.80 | -0.16 | -0.10 |
| CRPO | t1d | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| CRPO | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 71.67 ± 8.40 | 67.17 ± 9.73 | 67.25 ± 9.65 | -4.42 | +0.08 |
| CRPO | t1d | adult | selected / 1 | complete | 3 | 9 | 71.67 ± 8.40 | 67.17 ± 9.73 | 67.17 ± 9.73 | -4.51 | +0.00 |
| CRPO | t1d | child | fixed10 / 10 | complete | 3 | 9 | 71.66 ± 12.00 | 76.11 ± 11.00 | 76.61 ± 9.62 | +4.95 | +0.50 |
| CRPO | t1d | child | selected / 10 | complete | 3 | 9 | 71.66 ± 12.00 | 76.11 ± 11.00 | 76.61 ± 9.62 | +4.95 | +0.50 |
| CRPO | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 40.60 ± 42.21 | 49.38 ± 36.82 | 60.96 ± 25.41 | +20.37 | +11.59 |
| CRPO | t2d | adolescent | selected / 20 | complete | 3 | 9 | 40.60 ± 42.21 | 49.38 ± 36.82 | 74.33 ± 23.88 | +33.73 | +24.95 |
| CRPO | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 85.11 ± 4.83 | 85.97 ± 5.30 | 85.84 ± 5.47 | +0.73 | -0.12 |
| CRPO | t2d | adult | selected / 1 | complete | 3 | 9 | 85.11 ± 4.83 | 85.97 ± 5.30 | 85.99 ± 5.60 | +0.87 | +0.02 |
| CRPO | t2d | child | fixed10 / 10 | complete | 3 | 9 | 61.76 ± 21.50 | 70.38 ± 17.30 | 72.84 ± 17.02 | +11.08 | +2.46 |
| CRPO | t2d | child | selected / 1 | complete | 3 | 9 | 61.76 ± 21.50 | 70.38 ± 17.30 | 70.67 ± 17.11 | +8.91 | +0.30 |
| CRPO | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 78.38 ± 15.25 | 80.95 ± 14.18 | 84.11 ± 10.09 | +5.72 | +3.15 |
| CRPO | t2d_no_pump | adolescent | selected / 5 | complete | 3 | 9 | 78.38 ± 15.25 | 80.95 ± 14.18 | 83.90 ± 10.22 | +5.52 | +2.95 |
| CRPO | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 92.50 ± 1.89 | 92.86 ± 1.35 | 92.91 ± 1.28 | +0.40 | +0.05 |
| CRPO | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 92.50 ± 1.89 | 92.86 ± 1.35 | 92.86 ± 1.35 | +0.36 | +0.00 |
| CRPO | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 59.52 ± 19.11 | 66.54 ± 17.28 | 68.38 ± 16.79 | +8.85 | +1.83 |
| CRPO | t2d_no_pump | child | selected / 15 | complete | 3 | 9 | 59.52 ± 19.11 | 66.54 ± 17.28 | 72.42 ± 13.71 | +12.90 | +5.87 |
| RCPO | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 84.32 ± 6.09 | 84.18 ± 6.08 | 84.18 ± 6.08 | -0.14 | +0.00 |
| RCPO | t1d | adolescent | selected / 1 | complete | 3 | 9 | 84.32 ± 6.09 | 84.18 ± 6.08 | 84.18 ± 6.08 | -0.14 | +0.00 |
| RCPO | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 78.18 ± 7.43 | 74.37 ± 6.74 | 74.42 ± 6.65 | -3.76 | +0.05 |
| RCPO | t1d | adult | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| RCPO | t1d | child | fixed10 / 10 | complete | 3 | 9 | 70.12 ± 3.35 | 67.89 ± 5.70 | 72.32 ± 2.48 | +2.20 | +4.43 |
| RCPO | t1d | child | selected / 5 | complete | 3 | 9 | 70.12 ± 3.35 | 67.89 ± 5.70 | 70.39 ± 4.35 | +0.27 | +2.50 |
| RCPO | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 90.21 ± 1.95 | 91.20 ± 2.11 | 91.19 ± 2.11 | +0.98 | -0.00 |
| RCPO | t2d | adolescent | selected / 1 | complete | 3 | 9 | 90.21 ± 1.95 | 91.20 ± 2.11 | 91.20 ± 2.11 | +0.98 | +0.00 |
| RCPO | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 84.16 ± 3.36 | 86.20 ± 3.37 | 86.26 ± 3.28 | +2.09 | +0.05 |
| RCPO | t2d | adult | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| RCPO | t2d | child | fixed10 / 10 | complete | 3 | 9 | 62.85 ± 10.14 | 62.73 ± 11.75 | 62.97 ± 10.27 | +0.13 | +0.24 |
| RCPO | t2d | child | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| RCPO | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 93.16 ± 2.69 | 93.43 ± 2.63 | 93.64 ± 2.58 | +0.48 | +0.22 |
| RCPO | t2d_no_pump | adolescent | selected / 10 | complete | 3 | 9 | 93.16 ± 2.69 | 93.43 ± 2.63 | 93.64 ± 2.58 | +0.48 | +0.22 |
| RCPO | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 92.01 ± 1.57 | 92.39 ± 1.73 | 92.43 ± 1.72 | +0.42 | +0.04 |
| RCPO | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 92.01 ± 1.57 | 92.39 ± 1.73 | 92.39 ± 1.74 | +0.38 | -0.00 |
| RCPO | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 64.65 ± 15.81 | 69.18 ± 16.80 | 71.69 ± 17.05 | +7.04 | +2.51 |
| RCPO | t2d_no_pump | child | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| TRPOLag | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 85.34 ± 6.73 | 83.96 ± 7.98 | 83.97 ± 7.99 | -1.36 | +0.01 |
| TRPOLag | t1d | adolescent | selected / 1 | complete | 3 | 9 | 85.34 ± 6.73 | 83.96 ± 7.98 | 83.97 ± 7.98 | -1.37 | +0.00 |
| TRPOLag | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 85.62 ± 5.78 | 84.01 ± 3.65 | 84.19 ± 3.89 | -1.42 | +0.18 |
| TRPOLag | t1d | adult | selected / 1 | complete | 3 | 9 | 85.62 ± 5.78 | 84.01 ± 3.65 | 84.02 ± 3.64 | -1.59 | +0.01 |
| TRPOLag | t1d | child | fixed10 / 10 | complete | 3 | 9 | 69.50 ± 4.16 | 75.37 ± 2.66 | 76.95 ± 2.25 | +7.45 | +1.58 |
| TRPOLag | t1d | child | selected / 1 | complete | 3 | 9 | 69.50 ± 4.16 | 75.37 ± 2.66 | 75.35 ± 2.75 | +5.85 | -0.02 |
| TRPOLag | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 91.29 ± 5.90 | 93.57 ± 4.28 | 93.57 ± 4.27 | +2.28 | +0.00 |
| TRPOLag | t2d | adolescent | selected / 1 | complete | 3 | 9 | 91.29 ± 5.90 | 93.57 ± 4.28 | 93.57 ± 4.28 | +2.27 | +0.00 |
| TRPOLag | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 88.00 ± 4.18 | 89.17 ± 5.28 | 89.22 ± 5.23 | +1.22 | +0.05 |
| TRPOLag | t2d | adult | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| TRPOLag | t2d | child | fixed10 / 10 | complete | 3 | 9 | 58.53 ± 2.11 | 68.35 ± 5.47 | 68.62 ± 7.33 | +10.09 | +0.27 |
| TRPOLag | t2d | child | selected / 5 | complete | 3 | 9 | 58.53 ± 2.11 | 68.35 ± 5.47 | 66.45 ± 5.48 | +7.92 | -1.89 |
| TRPOLag | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 84.37 ± 4.81 | 85.29 ± 4.32 | 85.50 ± 4.31 | +1.13 | +0.22 |
| TRPOLag | t2d_no_pump | adolescent | selected / NA | no_eligible_penalty | 0 | 0 | — | — | — | — | — |
| TRPOLag | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 94.54 ± 1.61 | 94.62 ± 1.58 | 94.68 ± 1.52 | +0.14 | +0.05 |
| TRPOLag | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 94.54 ± 1.61 | 94.62 ± 1.58 | 94.63 ± 1.58 | +0.09 | +0.01 |
| TRPOLag | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 68.60 ± 18.86 | 69.87 ± 16.06 | 73.42 ± 18.29 | +4.81 | +3.54 |
| TRPOLag | t2d_no_pump | child | selected / 5 | complete | 3 | 9 | 68.60 ± 18.86 | 69.87 ± 16.06 | 73.34 ± 15.80 | +4.74 | +3.47 |
| PCPO | t1d | adolescent | fixed10 / 10 | complete | 3 | 9 | 85.78 ± 7.78 | 85.11 ± 7.61 | 85.11 ± 7.62 | -0.67 | +0.00 |
| PCPO | t1d | adolescent | selected / 1 | complete | 3 | 9 | 85.78 ± 7.78 | 85.11 ± 7.61 | 85.11 ± 7.61 | -0.67 | +0.00 |
| PCPO | t1d | adult | fixed10 / 10 | complete | 3 | 9 | 80.21 ± 10.05 | 78.33 ± 10.31 | 78.34 ± 10.33 | -1.87 | +0.02 |
| PCPO | t1d | adult | selected / 1 | complete | 3 | 9 | 80.21 ± 10.05 | 78.33 ± 10.31 | 78.35 ± 10.30 | -1.87 | +0.02 |
| PCPO | t1d | child | fixed10 / 10 | complete | 3 | 9 | 77.66 ± 5.92 | 84.15 ± 4.37 | 83.84 ± 5.36 | +6.18 | -0.31 |
| PCPO | t1d | child | selected / 1 | complete | 3 | 9 | 77.66 ± 5.92 | 84.15 ± 4.37 | 83.73 ± 5.43 | +6.06 | -0.43 |
| PCPO | t2d | adolescent | fixed10 / 10 | complete | 3 | 9 | 93.40 ± 1.21 | 93.67 ± 1.51 | 93.65 ± 1.50 | +0.25 | -0.02 |
| PCPO | t2d | adolescent | selected / 1 | complete | 3 | 9 | 93.40 ± 1.21 | 93.67 ± 1.51 | 93.67 ± 1.51 | +0.27 | +0.00 |
| PCPO | t2d | adult | fixed10 / 10 | complete | 3 | 9 | 82.43 ± 13.09 | 81.29 ± 14.01 | 81.61 ± 13.61 | -0.82 | +0.32 |
| PCPO | t2d | adult | selected / 1 | complete | 3 | 9 | 82.43 ± 13.09 | 81.29 ± 14.01 | 81.32 ± 13.96 | -1.12 | +0.02 |
| PCPO | t2d | child | fixed10 / 10 | complete | 3 | 9 | 71.44 ± 22.29 | 78.04 ± 11.39 | 79.80 ± 9.86 | +8.36 | +1.76 |
| PCPO | t2d | child | selected / 1 | complete | 3 | 9 | 71.44 ± 22.29 | 78.04 ± 11.39 | 78.50 ± 11.02 | +7.06 | +0.46 |
| PCPO | t2d_no_pump | adolescent | fixed10 / 10 | complete | 3 | 9 | 72.76 ± 11.22 | 73.14 ± 11.12 | 81.64 ± 3.95 | +8.88 | +8.50 |
| PCPO | t2d_no_pump | adolescent | selected / 20 | complete | 3 | 9 | 72.76 ± 11.22 | 73.14 ± 11.12 | 81.00 ± 5.07 | +8.25 | +7.86 |
| PCPO | t2d_no_pump | adult | fixed10 / 10 | complete | 3 | 9 | 93.65 ± 2.20 | 93.77 ± 2.18 | 93.81 ± 2.19 | +0.16 | +0.04 |
| PCPO | t2d_no_pump | adult | selected / 1 | complete | 3 | 9 | 93.65 ± 2.20 | 93.77 ± 2.18 | 93.79 ± 2.19 | +0.15 | +0.02 |
| PCPO | t2d_no_pump | child | fixed10 / 10 | complete | 3 | 9 | 52.54 ± 18.31 | 58.94 ± 11.02 | 60.40 ± 14.60 | +7.85 | +1.45 |
| PCPO | t2d_no_pump | child | selected / 1 | complete | 3 | 9 | 52.54 ± 18.31 | 58.94 ± 11.02 | 59.45 ± 12.57 | +6.91 | +0.51 |

### Safety and coverage: stochastic

| Algorithm | Type | Cohort | Arm | State | B/9 | Observed TIR % none → shield | Below54 % none → shield | Early ending % none → shield | Physiological ending % none → shield | Coverage % none → shield |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| CUP | t1d | adolescent | fixed10 | complete | 9 | 93.61 → 92.94 | 0.89 → 0.09 | 2.47 → 3.70 | 2.47 → 3.70 | 98.91 → 97.25 |
| CUP | t1d | adult | fixed10 | complete | 9 | 54.54 → 52.24 | 0.54 → 0.08 | 20.99 → 22.22 | 20.99 → 22.22 | 87.29 → 85.05 |
| CUP | t1d | adult | selected | complete | 9 | 54.54 → 51.86 | 0.54 → 0.08 | 20.99 → 22.22 | 20.99 → 22.22 | 87.29 → 85.05 |
| CUP | t1d | child | fixed10 | complete | 9 | 76.71 → 85.39 | 11.16 → 3.97 | 7.41 → 7.41 | 7.41 → 7.41 | 95.18 → 96.78 |
| CUP | t1d | child | selected | complete | 9 | 76.71 → 85.18 | 11.16 → 4.08 | 7.41 → 6.17 | 7.41 → 6.17 | 95.18 → 96.94 |
| CUP | t2d | adolescent | fixed10 | complete | 9 | 77.29 → 81.25 | 4.20 → 0.39 | 6.17 → 0.00 | 6.17 → 0.00 | 96.90 → 100.00 |
| CUP | t2d | adult | fixed10 | complete | 9 | 78.85 → 78.66 | 1.33 → 0.13 | 2.47 → 0.00 | 2.47 → 0.00 | 99.55 → 100.00 |
| CUP | t2d | adult | selected | complete | 9 | 78.85 → 78.51 | 1.33 → 0.14 | 2.47 → 0.00 | 2.47 → 0.00 | 99.55 → 100.00 |
| CUP | t2d | child | fixed10 | complete | 9 | 86.77 → 89.15 | 3.21 → 1.97 | 7.41 → 0.82 | 7.41 → 0.82 | 95.31 → 99.35 |
| CUP | t2d | child | selected | complete | 9 | 86.77 → 89.14 | 3.21 → 2.04 | 7.41 → 0.82 | 7.41 → 0.82 | 95.31 → 99.58 |
| CUP | t2d_no_pump | adolescent | fixed10 | complete | 9 | 91.19 → 94.53 | 2.09 → 0.45 | 1.23 → 0.00 | 1.23 → 0.00 | 99.09 → 100.00 |
| CUP | t2d_no_pump | adolescent | selected | complete | 9 | 91.19 → 94.51 | 2.09 → 0.44 | 1.23 → 0.00 | 1.23 → 0.00 | 99.09 → 100.00 |
| CUP | t2d_no_pump | adult | fixed10 | complete | 9 | 93.25 → 93.90 | 0.17 → 0.01 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d_no_pump | adult | selected | complete | 9 | 93.25 → 93.89 | 0.17 → 0.01 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CUP | t2d_no_pump | child | fixed10 | complete | 9 | 75.95 → 84.62 | 11.05 → 3.16 | 27.16 → 9.47 | 27.16 → 9.47 | 87.94 → 95.25 |
| FOCOPS | t1d | adolescent | fixed10 | complete | 9 | 86.93 → 87.22 | 1.04 → 0.04 | 6.17 → 4.12 | 6.17 → 4.12 | 97.97 → 98.22 |
| FOCOPS | t1d | adolescent | selected | complete | 9 | 86.93 → 87.08 | 1.04 → 0.04 | 6.17 → 3.70 | 6.17 → 3.70 | 97.97 → 98.28 |
| FOCOPS | t1d | adult | fixed10 | complete | 9 | 48.16 → 43.68 | 0.59 → 0.05 | 2.47 → 1.23 | 2.47 → 1.23 | 97.84 → 98.92 |
| FOCOPS | t1d | adult | selected | complete | 9 | 48.16 → 43.50 | 0.59 → 0.05 | 2.47 → 1.23 | 2.47 → 1.23 | 97.84 → 98.92 |
| FOCOPS | t1d | child | fixed10 | complete | 9 | 81.30 → 89.37 | 8.01 → 0.59 | 13.58 → 10.70 | 13.58 → 10.70 | 93.60 → 94.12 |
| FOCOPS | t1d | child | selected | complete | 9 | 81.30 → 89.42 | 8.01 → 0.58 | 13.58 → 10.29 | 13.58 → 10.29 | 93.60 → 94.41 |
| FOCOPS | t2d | adolescent | fixed10 | complete | 9 | 95.52 → 96.94 | 1.04 → 0.03 | 2.47 → 0.00 | 2.47 → 0.00 | 98.92 → 100.00 |
| FOCOPS | t2d | adolescent | selected | complete | 9 | 95.52 → 96.94 | 1.04 → 0.02 | 2.47 → 0.00 | 2.47 → 0.00 | 98.92 → 100.00 |
| FOCOPS | t2d | adult | fixed10 | complete | 9 | 70.68 → 70.14 | 0.26 → 0.04 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t2d | adult | selected | complete | 9 | 70.68 → 69.90 | 0.26 → 0.03 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t2d | child | fixed10 | complete | 9 | 77.77 → 88.01 | 11.07 → 0.99 | 20.99 → 1.23 | 20.99 → 1.23 | 92.42 → 99.53 |
| FOCOPS | t2d | child | selected | complete | 9 | 77.77 → 88.19 | 11.07 → 0.95 | 20.99 → 0.82 | 20.99 → 0.82 | 92.42 → 99.54 |
| FOCOPS | t2d_no_pump | adolescent | fixed10 | complete | 9 | 65.72 → 83.77 | 16.33 → 0.96 | 23.46 → 2.88 | 23.46 → 2.88 | 84.93 → 98.35 |
| FOCOPS | t2d_no_pump | adolescent | selected | complete | 9 | 65.72 → 83.77 | 16.33 → 0.96 | 23.46 → 2.88 | 23.46 → 2.88 | 84.93 → 98.35 |
| FOCOPS | t2d_no_pump | adult | fixed10 | complete | 9 | 91.42 → 91.68 | 0.15 → 0.03 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| FOCOPS | t2d_no_pump | child | fixed10 | complete | 9 | 84.79 → 90.61 | 3.97 → 0.70 | 4.94 → 3.29 | 4.94 → 3.29 | 97.36 → 98.74 |
| FOCOPS | t2d_no_pump | child | selected | complete | 9 | 84.79 → 90.61 | 3.97 → 0.70 | 4.94 → 3.29 | 4.94 → 3.29 | 97.36 → 98.74 |
| CRPO | t1d | adolescent | fixed10 | complete | 9 | 85.21 → 85.05 | 0.34 → 0.10 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t1d | adult | fixed10 | complete | 9 | 73.97 → 68.80 | 1.24 → 0.72 | 7.41 → 6.17 | 7.41 → 6.17 | 95.99 → 96.03 |
| CRPO | t1d | adult | selected | complete | 9 | 73.97 → 68.71 | 1.24 → 0.72 | 7.41 → 6.17 | 7.41 → 6.17 | 95.99 → 96.03 |
| CRPO | t1d | child | fixed10 | complete | 9 | 76.85 → 81.28 | 8.83 → 4.95 | 19.75 → 13.17 | 19.75 → 13.17 | 90.56 → 92.87 |
| CRPO | t1d | child | selected | complete | 9 | 76.85 → 81.28 | 8.83 → 4.95 | 19.75 → 13.17 | 19.75 → 13.17 | 90.56 → 92.87 |
| CRPO | t2d | adolescent | fixed10 | complete | 9 | 84.45 → 85.87 | 8.67 → 4.86 | 67.90 → 49.79 | 67.90 → 49.79 | 45.59 → 69.40 |
| CRPO | t2d | adolescent | selected | complete | 9 | 84.45 → 86.72 | 8.67 → 4.05 | 67.90 → 34.98 | 67.90 → 34.98 | 45.59 → 83.39 |
| CRPO | t2d | adult | fixed10 | complete | 9 | 85.11 → 85.84 | 2.18 → 1.15 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t2d | adult | selected | complete | 9 | 85.11 → 85.99 | 2.18 → 1.15 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t2d | child | fixed10 | complete | 9 | 69.94 → 78.43 | 15.85 → 9.44 | 18.52 → 9.47 | 18.52 → 9.47 | 86.60 → 92.56 |
| CRPO | t2d | child | selected | complete | 9 | 69.94 → 76.50 | 15.85 → 9.75 | 18.52 → 11.11 | 18.52 → 11.11 | 86.60 → 91.91 |
| CRPO | t2d_no_pump | adolescent | fixed10 | complete | 9 | 86.60 → 87.12 | 4.10 → 2.89 | 20.99 → 9.47 | 20.99 → 9.47 | 89.08 → 95.19 |
| CRPO | t2d_no_pump | adolescent | selected | complete | 9 | 86.60 → 86.86 | 4.10 → 3.09 | 20.99 → 9.05 | 20.99 → 9.05 | 89.08 → 95.35 |
| CRPO | t2d_no_pump | adult | fixed10 | complete | 9 | 92.50 → 92.91 | 0.39 → 0.20 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t2d_no_pump | adult | selected | complete | 9 | 92.50 → 92.86 | 0.39 → 0.20 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| CRPO | t2d_no_pump | child | fixed10 | complete | 9 | 75.00 → 79.59 | 12.44 → 5.68 | 41.98 → 30.86 | 41.98 → 30.86 | 79.88 → 84.33 |
| CRPO | t2d_no_pump | child | selected | complete | 9 | 75.00 → 80.60 | 12.44 → 5.38 | 41.98 → 23.05 | 41.98 → 23.05 | 79.88 → 88.83 |
| RCPO | t1d | adolescent | fixed10 | complete | 9 | 86.94 → 86.81 | 0.00 → 0.00 | 12.35 → 12.35 | 12.35 → 12.35 | 95.61 → 95.61 |
| RCPO | t1d | adolescent | selected | complete | 9 | 86.94 → 86.81 | 0.00 → 0.00 | 12.35 → 12.35 | 12.35 → 12.35 | 95.61 → 95.61 |
| RCPO | t1d | adult | fixed10 | complete | 9 | 79.38 → 76.10 | 1.91 → 1.51 | 7.41 → 8.64 | 7.41 → 8.64 | 97.91 → 96.27 |
| RCPO | t1d | child | fixed10 | complete | 9 | 79.26 → 81.82 | 8.87 → 5.58 | 17.28 → 19.34 | 17.28 → 19.34 | 88.00 → 87.82 |
| RCPO | t1d | child | selected | complete | 9 | 79.26 → 81.22 | 8.87 → 5.74 | 17.28 → 20.16 | 17.28 → 20.16 | 88.00 → 85.92 |
| RCPO | t2d | adolescent | fixed10 | complete | 9 | 90.86 → 91.84 | 1.07 → 0.40 | 2.47 → 2.47 | 2.47 → 2.47 | 99.21 → 99.21 |
| RCPO | t2d | adolescent | selected | complete | 9 | 90.86 → 91.84 | 1.07 → 0.40 | 2.47 → 2.47 | 2.47 → 2.47 | 99.21 → 99.21 |
| RCPO | t2d | adult | fixed10 | complete | 9 | 84.16 → 86.26 | 1.85 → 0.82 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| RCPO | t2d | child | fixed10 | complete | 9 | 70.66 → 74.01 | 16.18 → 13.66 | 22.22 → 26.75 | 22.22 → 26.75 | 88.86 → 84.70 |
| RCPO | t2d_no_pump | adolescent | fixed10 | complete | 9 | 94.63 → 94.81 | 0.23 → 0.17 | 2.47 → 2.06 | 2.47 → 2.06 | 98.41 → 98.70 |
| RCPO | t2d_no_pump | adolescent | selected | complete | 9 | 94.63 → 94.81 | 0.23 → 0.17 | 2.47 → 2.06 | 2.47 → 2.06 | 98.41 → 98.70 |
| RCPO | t2d_no_pump | adult | fixed10 | complete | 9 | 92.01 → 92.43 | 0.08 → 0.04 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| RCPO | t2d_no_pump | adult | selected | complete | 9 | 92.01 → 92.39 | 0.08 → 0.03 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| RCPO | t2d_no_pump | child | fixed10 | complete | 9 | 72.55 → 76.34 | 13.89 → 9.35 | 18.52 → 12.76 | 18.52 → 12.76 | 88.53 → 92.95 |
| TRPOLag | t1d | adolescent | fixed10 | complete | 9 | 88.66 → 87.99 | 0.02 → 0.00 | 7.41 → 8.64 | 7.41 → 8.64 | 95.21 → 94.27 |
| TRPOLag | t1d | adolescent | selected | complete | 9 | 88.66 → 87.99 | 0.02 → 0.00 | 7.41 → 8.64 | 7.41 → 8.64 | 95.21 → 94.27 |
| TRPOLag | t1d | adult | fixed10 | complete | 9 | 86.20 → 84.22 | 1.81 → 1.16 | 2.47 → 1.23 | 2.47 → 1.23 | 98.90 → 99.82 |
| TRPOLag | t1d | adult | selected | complete | 9 | 86.20 → 84.05 | 1.81 → 1.21 | 2.47 → 1.23 | 2.47 → 1.23 | 98.90 → 99.82 |
| TRPOLag | t1d | child | fixed10 | complete | 9 | 80.12 → 82.58 | 8.75 → 6.43 | 23.46 → 13.99 | 23.46 → 13.99 | 86.29 → 92.81 |
| TRPOLag | t1d | child | selected | complete | 9 | 80.12 → 82.84 | 8.75 → 6.13 | 23.46 → 18.52 | 23.46 → 18.52 | 86.29 → 90.44 |
| TRPOLag | t2d | adolescent | fixed10 | complete | 9 | 93.11 → 94.02 | 0.67 → 0.25 | 6.17 → 3.70 | 6.17 → 3.70 | 97.82 → 99.44 |
| TRPOLag | t2d | adolescent | selected | complete | 9 | 93.11 → 94.02 | 0.67 → 0.25 | 6.17 → 3.70 | 6.17 → 3.70 | 97.82 → 99.44 |
| TRPOLag | t2d | adult | fixed10 | complete | 9 | 88.00 → 89.22 | 0.63 → 0.07 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| TRPOLag | t2d | child | fixed10 | complete | 9 | 67.80 → 75.72 | 18.53 → 12.03 | 20.99 → 16.46 | 20.99 → 16.46 | 86.47 → 90.07 |
| TRPOLag | t2d | child | selected | complete | 9 | 67.80 → 75.13 | 18.53 → 12.34 | 20.99 → 19.34 | 20.99 → 19.34 | 86.47 → 87.87 |
| TRPOLag | t2d_no_pump | adolescent | fixed10 | complete | 9 | 89.98 → 90.49 | 2.33 → 1.89 | 13.58 → 11.93 | 13.58 → 11.93 | 93.43 → 94.11 |
| TRPOLag | t2d_no_pump | adult | fixed10 | complete | 9 | 94.54 → 94.68 | 0.10 → 0.04 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| TRPOLag | t2d_no_pump | adult | selected | complete | 9 | 94.54 → 94.63 | 0.10 → 0.04 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| TRPOLag | t2d_no_pump | child | fixed10 | complete | 9 | 72.19 → 76.46 | 13.57 → 9.33 | 9.88 → 9.47 | 9.88 → 9.47 | 94.21 → 95.14 |
| TRPOLag | t2d_no_pump | child | selected | complete | 9 | 72.19 → 75.96 | 13.57 → 9.23 | 9.88 → 10.29 | 9.88 → 10.29 | 94.21 → 95.65 |
| PCPO | t1d | adolescent | fixed10 | complete | 9 | 86.89 → 86.22 | 0.00 → 0.01 | 3.70 → 3.70 | 3.70 → 3.70 | 98.18 → 98.18 |
| PCPO | t1d | adolescent | selected | complete | 9 | 86.89 → 86.22 | 0.00 → 0.01 | 3.70 → 3.70 | 3.70 → 3.70 | 98.18 → 98.18 |
| PCPO | t1d | adult | fixed10 | complete | 9 | 80.27 → 78.37 | 1.02 → 0.62 | 2.47 → 1.23 | 2.47 → 1.23 | 99.93 → 99.97 |
| PCPO | t1d | adult | selected | complete | 9 | 80.27 → 78.37 | 1.02 → 0.62 | 2.47 → 1.23 | 2.47 → 1.23 | 99.93 → 99.97 |
| PCPO | t1d | child | fixed10 | complete | 9 | 83.83 → 86.85 | 6.42 → 2.38 | 16.05 → 9.47 | 16.05 → 9.47 | 91.65 → 96.18 |
| PCPO | t1d | child | selected | complete | 9 | 83.83 → 86.37 | 6.42 → 2.56 | 16.05 → 8.64 | 16.05 → 8.64 | 91.65 → 96.56 |
| PCPO | t2d | adolescent | fixed10 | complete | 9 | 93.40 → 93.65 | 0.41 → 0.25 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d | adolescent | selected | complete | 9 | 93.40 → 93.67 | 0.41 → 0.25 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d | adult | fixed10 | complete | 9 | 82.43 → 81.61 | 0.40 → 0.23 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d | adult | selected | complete | 9 | 82.43 → 81.32 | 0.40 → 0.23 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d | child | fixed10 | complete | 9 | 75.09 → 80.59 | 12.30 → 6.92 | 11.11 → 3.70 | 11.11 → 3.70 | 92.06 → 98.94 |
| PCPO | t2d | child | selected | complete | 9 | 75.09 → 79.39 | 12.30 → 7.61 | 11.11 → 3.29 | 11.11 → 3.29 | 92.06 → 98.82 |
| PCPO | t2d_no_pump | adolescent | fixed10 | complete | 9 | 78.15 → 82.51 | 6.57 → 3.28 | 12.35 → 2.88 | 12.35 → 2.88 | 92.90 → 98.89 |
| PCPO | t2d_no_pump | adolescent | selected | complete | 9 | 78.15 → 82.33 | 6.57 → 3.40 | 12.35 → 3.29 | 12.35 → 3.29 | 92.90 → 98.24 |
| PCPO | t2d_no_pump | adult | fixed10 | complete | 9 | 93.65 → 93.81 | 0.13 → 0.09 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d_no_pump | adult | selected | complete | 9 | 93.65 → 93.79 | 0.13 → 0.08 | 0.00 → 0.00 | 0.00 → 0.00 | 100.00 → 100.00 |
| PCPO | t2d_no_pump | child | fixed10 | complete | 9 | 61.18 → 67.34 | 21.57 → 15.25 | 27.16 → 21.81 | 27.16 → 21.81 | 84.95 → 89.12 |
| PCPO | t2d_no_pump | child | selected | complete | 9 | 61.18 → 66.38 | 21.57 → 15.50 | 27.16 → 23.87 | 27.16 → 23.87 | 84.95 → 89.17 |

## Complete cells: patient and horizon generalization

Only full panels are numeric here. Loss is seen #001 one-day requested TIR minus unseen #002–#010 seven-day requested TIR; positive loss means degradation.

| Algorithm | Type | Cohort | Mode | Arm | Seen1 TIR % | Unseen7 TIR % | Joint loss pp |
|---|---|---|---|---|---:|---:|---:|
| CUP | t1d | adolescent | deterministic | none | 99.73 ± 0.24 | 92.06 ± 2.08 | +7.67 |
| CUP | t1d | adolescent | deterministic | static | 99.38 ± 0.37 | 88.51 ± 2.86 | +10.88 |
| CUP | t1d | adolescent | deterministic | fixed10 | 99.59 ± 0.24 | 88.49 ± 2.90 | +11.10 |
| CUP | t1d | adolescent | stochastic | none | 94.91 ± 4.18 | 92.60 ± 2.52 | +2.31 |
| CUP | t1d | adolescent | stochastic | static | 98.11 ± 1.78 | 90.18 ± 3.67 | +7.93 |
| CUP | t1d | adolescent | stochastic | fixed10 | 98.11 ± 1.78 | 90.39 ± 3.78 | +7.72 |
| CUP | t1d | adult | deterministic | none | 98.46 ± 1.34 | 49.74 ± 5.94 | +48.72 |
| CUP | t1d | adult | deterministic | static | 98.46 ± 1.34 | 46.17 ± 6.21 | +52.29 |
| CUP | t1d | adult | deterministic | fixed10 | 98.46 ± 1.34 | 46.86 ± 5.68 | +51.60 |
| CUP | t1d | adult | deterministic | selected | 98.46 ± 1.34 | 46.62 ± 5.90 | +51.84 |
| CUP | t1d | adult | stochastic | none | 93.36 ± 7.20 | 49.96 ± 6.63 | +43.40 |
| CUP | t1d | adult | stochastic | static | 93.06 ± 7.03 | 46.66 ± 6.17 | +46.40 |
| CUP | t1d | adult | stochastic | fixed10 | 93.06 ± 7.03 | 47.43 ± 5.56 | +45.63 |
| CUP | t1d | adult | stochastic | selected | 93.06 ± 7.03 | 47.06 ± 5.95 | +46.00 |
| CUP | t1d | child | deterministic | none | 86.61 ± 7.95 | 72.61 ± 5.16 | +14.00 |
| CUP | t1d | child | deterministic | static | 85.92 ± 2.62 | 83.29 ± 4.17 | +2.63 |
| CUP | t1d | child | deterministic | fixed10 | 87.59 ± 3.65 | 84.92 ± 3.32 | +2.67 |
| CUP | t1d | child | stochastic | none | 87.08 ± 7.15 | 73.03 ± 4.69 | +14.05 |
| CUP | t1d | child | stochastic | static | 87.65 ± 2.41 | 81.57 ± 6.22 | +6.09 |
| CUP | t1d | child | stochastic | fixed10 | 88.13 ± 2.40 | 82.74 ± 5.37 | +5.39 |
| CUP | t1d | child | stochastic | selected | 87.96 ± 2.37 | 82.66 ± 5.46 | +5.30 |
| CUP | t2d | adolescent | deterministic | none | 82.91 ± 8.40 | 72.70 ± 25.18 | +10.21 |
| CUP | t2d | adolescent | deterministic | static | 86.38 ± 7.24 | 82.02 ± 12.03 | +4.36 |
| CUP | t2d | adolescent | deterministic | fixed10 | 88.22 ± 6.86 | 82.78 ± 11.34 | +5.44 |
| CUP | t2d | adolescent | stochastic | none | 75.12 ± 12.54 | 74.90 ± 11.33 | +0.22 |
| CUP | t2d | adolescent | stochastic | static | 81.40 ± 2.20 | 79.43 ± 13.72 | +1.97 |
| CUP | t2d | adolescent | stochastic | fixed10 | 87.28 ± 3.61 | 81.25 ± 12.10 | +6.03 |
| CUP | t2d | adult | deterministic | none | 88.70 ± 10.25 | 82.04 ± 5.62 | +6.66 |
| CUP | t2d | adult | deterministic | static | 88.27 ± 9.56 | 82.93 ± 5.32 | +5.34 |
| CUP | t2d | adult | deterministic | fixed10 | 88.27 ± 9.56 | 83.06 ± 5.21 | +5.21 |
| CUP | t2d | adult | deterministic | selected | 88.27 ± 9.56 | 82.73 ± 5.40 | +5.55 |
| CUP | t2d | adult | stochastic | none | 90.01 ± 6.78 | 78.50 ± 5.49 | +11.51 |
| CUP | t2d | adult | stochastic | static | 90.70 ± 7.81 | 78.49 ± 6.40 | +12.21 |
| CUP | t2d | adult | stochastic | fixed10 | 90.70 ± 7.81 | 78.66 ± 6.11 | +12.04 |
| CUP | t2d | adult | stochastic | selected | 90.70 ± 7.81 | 78.51 ± 6.41 | +12.19 |
| CUP | t2d | child | deterministic | none | 96.06 ± 1.01 | 89.16 ± 1.85 | +6.91 |
| CUP | t2d | child | deterministic | static | 96.06 ± 1.01 | 90.08 ± 1.18 | +5.99 |
| CUP | t2d | child | deterministic | fixed10 | 96.06 ± 1.01 | 90.52 ± 0.81 | +5.55 |
| CUP | t2d | child | deterministic | selected | 96.06 ± 1.01 | 90.41 ± 1.03 | +5.65 |
| CUP | t2d | child | stochastic | none | 93.40 ± 2.73 | 82.70 ± 5.29 | +10.70 |
| CUP | t2d | child | stochastic | static | 94.10 ± 1.62 | 87.30 ± 1.94 | +6.80 |
| CUP | t2d | child | stochastic | fixed10 | 94.01 ± 1.76 | 88.68 ± 0.58 | +5.33 |
| CUP | t2d | child | stochastic | selected | 94.01 ± 1.76 | 88.82 ± 0.32 | +5.18 |
| CUP | t2d_no_pump | adolescent | deterministic | none | 98.92 ± 1.37 | 95.39 ± 1.18 | +3.53 |
| CUP | t2d_no_pump | adolescent | deterministic | static | 98.92 ± 1.37 | 96.17 ± 0.79 | +2.75 |
| CUP | t2d_no_pump | adolescent | deterministic | fixed10 | 98.92 ± 1.37 | 96.17 ± 0.82 | +2.75 |
| CUP | t2d_no_pump | adolescent | deterministic | selected | 98.92 ± 1.37 | 96.17 ± 0.80 | +2.75 |
| CUP | t2d_no_pump | adolescent | stochastic | none | 94.64 ± 7.61 | 90.34 ± 5.33 | +4.29 |
| CUP | t2d_no_pump | adolescent | stochastic | static | 94.64 ± 7.61 | 93.68 ± 2.18 | +0.96 |
| CUP | t2d_no_pump | adolescent | stochastic | fixed10 | 97.74 ± 2.31 | 94.53 ± 1.08 | +3.21 |
| CUP | t2d_no_pump | adolescent | stochastic | selected | 97.74 ± 2.31 | 94.51 ± 1.12 | +3.23 |
| CUP | t2d_no_pump | adult | deterministic | none | 97.99 ± 1.74 | 93.26 ± 3.73 | +4.74 |
| CUP | t2d_no_pump | adult | deterministic | static | 97.99 ± 1.74 | 93.74 ± 3.86 | +4.25 |
| CUP | t2d_no_pump | adult | deterministic | fixed10 | 97.99 ± 1.74 | 93.74 ± 3.86 | +4.25 |
| CUP | t2d_no_pump | adult | deterministic | selected | 97.99 ± 1.74 | 93.74 ± 3.86 | +4.25 |
| CUP | t2d_no_pump | adult | stochastic | none | 98.03 ± 1.70 | 93.25 ± 3.61 | +4.78 |
| CUP | t2d_no_pump | adult | stochastic | static | 98.03 ± 1.70 | 93.88 ± 3.50 | +4.15 |
| CUP | t2d_no_pump | adult | stochastic | fixed10 | 98.03 ± 1.70 | 93.90 ± 3.50 | +4.14 |
| CUP | t2d_no_pump | adult | stochastic | selected | 98.03 ± 1.70 | 93.89 ± 3.49 | +4.14 |
| CUP | t2d_no_pump | child | deterministic | none | 84.84 ± 13.53 | 77.03 ± 14.55 | +7.80 |
| CUP | t2d_no_pump | child | deterministic | static | 88.97 ± 6.38 | 79.36 ± 10.21 | +9.60 |
| CUP | t2d_no_pump | child | deterministic | fixed10 | 90.47 ± 3.51 | 85.81 ± 5.91 | +4.66 |
| CUP | t2d_no_pump | child | stochastic | none | 82.29 ± 12.87 | 66.56 ± 12.87 | +15.73 |
| CUP | t2d_no_pump | child | stochastic | static | 87.04 ± 11.38 | 74.35 ± 11.94 | +12.68 |
| CUP | t2d_no_pump | child | stochastic | fixed10 | 92.90 ± 1.61 | 80.72 ± 6.07 | +12.18 |
| FOCOPS | t1d | adolescent | deterministic | none | 99.19 ± 1.30 | 88.77 ± 4.67 | +10.42 |
| FOCOPS | t1d | adolescent | deterministic | static | 99.88 ± 0.20 | 87.04 ± 4.51 | +12.85 |
| FOCOPS | t1d | adolescent | deterministic | fixed10 | 99.90 ± 0.18 | 87.12 ± 4.43 | +12.78 |
| FOCOPS | t1d | adolescent | stochastic | none | 95.06 ± 4.05 | 85.45 ± 11.03 | +9.61 |
| FOCOPS | t1d | adolescent | stochastic | static | 95.99 ± 4.38 | 85.58 ± 9.88 | +10.41 |
| FOCOPS | t1d | adolescent | stochastic | fixed10 | 96.99 ± 2.67 | 85.96 ± 9.09 | +11.03 |
| FOCOPS | t1d | adolescent | stochastic | selected | 95.99 ± 4.38 | 85.88 ± 9.45 | +10.10 |
| FOCOPS | t1d | adult | deterministic | none | 94.37 ± 6.00 | 48.40 ± 10.32 | +45.97 |
| FOCOPS | t1d | adult | deterministic | static | 92.52 ± 5.63 | 44.05 ± 9.01 | +48.47 |
| FOCOPS | t1d | adult | deterministic | fixed10 | 92.52 ± 5.63 | 44.29 ± 8.96 | +48.22 |
| FOCOPS | t1d | adult | deterministic | selected | 92.52 ± 5.63 | 44.19 ± 9.08 | +48.33 |
| FOCOPS | t1d | adult | stochastic | none | 90.16 ± 7.82 | 47.91 ± 4.08 | +42.25 |
| FOCOPS | t1d | adult | stochastic | static | 89.66 ± 8.31 | 43.01 ± 3.02 | +46.66 |
| FOCOPS | t1d | adult | stochastic | fixed10 | 89.66 ± 8.31 | 43.56 ± 2.64 | +46.10 |
| FOCOPS | t1d | adult | stochastic | selected | 89.66 ± 8.31 | 43.38 ± 2.79 | +46.28 |
| FOCOPS | t1d | child | deterministic | none | 90.35 ± 5.95 | 81.15 ± 9.41 | +9.21 |
| FOCOPS | t1d | child | deterministic | static | 88.04 ± 2.93 | 88.27 ± 4.87 | -0.23 |
| FOCOPS | t1d | child | deterministic | fixed10 | 89.17 ± 1.04 | 90.17 ± 3.04 | -1.00 |
| FOCOPS | t1d | child | deterministic | selected | 89.17 ± 1.04 | 90.17 ± 3.04 | -1.00 |
| FOCOPS | t1d | child | stochastic | none | 90.05 ± 5.22 | 76.35 ± 7.77 | +13.69 |
| FOCOPS | t1d | child | stochastic | static | 90.43 ± 1.83 | 82.86 ± 9.88 | +7.57 |
| FOCOPS | t1d | child | stochastic | fixed10 | 90.74 ± 0.32 | 84.88 ± 9.45 | +5.86 |
| FOCOPS | t1d | child | stochastic | selected | 90.59 ± 0.45 | 85.08 ± 9.09 | +5.51 |
| FOCOPS | t2d | adolescent | deterministic | none | 100.00 ± 0.00 | 97.09 ± 0.62 | +2.91 |
| FOCOPS | t2d | adolescent | deterministic | static | 100.00 ± 0.00 | 97.12 ± 0.65 | +2.88 |
| FOCOPS | t2d | adolescent | deterministic | fixed10 | 100.00 ± 0.00 | 97.10 ± 0.67 | +2.90 |
| FOCOPS | t2d | adolescent | deterministic | selected | 100.00 ± 0.00 | 97.11 ± 0.65 | +2.89 |
| FOCOPS | t2d | adolescent | stochastic | none | 87.81 ± 14.12 | 94.52 ± 2.48 | -6.72 |
| FOCOPS | t2d | adolescent | stochastic | static | 89.62 ± 15.07 | 96.93 ± 0.11 | -7.30 |
| FOCOPS | t2d | adolescent | stochastic | fixed10 | 89.62 ± 15.07 | 96.94 ± 0.13 | -7.32 |
| FOCOPS | t2d | adolescent | stochastic | selected | 89.62 ± 15.07 | 96.94 ± 0.13 | -7.32 |
| FOCOPS | t2d | adult | deterministic | none | 80.32 ± 12.73 | 69.42 ± 1.51 | +10.91 |
| FOCOPS | t2d | adult | deterministic | static | 80.32 ± 12.73 | 68.86 ± 2.18 | +11.47 |
| FOCOPS | t2d | adult | deterministic | fixed10 | 80.32 ± 12.73 | 69.23 ± 1.66 | +11.10 |
| FOCOPS | t2d | adult | deterministic | selected | 80.32 ± 12.73 | 68.83 ± 1.83 | +11.49 |
| FOCOPS | t2d | adult | stochastic | none | 84.22 ± 4.82 | 70.68 ± 2.18 | +13.54 |
| FOCOPS | t2d | adult | stochastic | static | 84.22 ± 4.82 | 69.54 ± 2.32 | +14.68 |
| FOCOPS | t2d | adult | stochastic | fixed10 | 84.22 ± 4.82 | 70.14 ± 1.68 | +14.08 |
| FOCOPS | t2d | adult | stochastic | selected | 84.22 ± 4.82 | 69.90 ± 2.04 | +14.32 |
| FOCOPS | t2d | child | deterministic | none | 89.31 ± 10.90 | 81.12 ± 13.90 | +8.19 |
| FOCOPS | t2d | child | deterministic | static | 93.71 ± 3.29 | 89.00 ± 4.48 | +4.71 |
| FOCOPS | t2d | child | deterministic | fixed10 | 94.52 ± 1.90 | 90.89 ± 4.45 | +3.63 |
| FOCOPS | t2d | child | deterministic | selected | 94.52 ± 1.90 | 90.89 ± 4.45 | +3.63 |
| FOCOPS | t2d | child | stochastic | none | 86.42 ± 10.08 | 72.54 ± 13.07 | +13.88 |
| FOCOPS | t2d | child | stochastic | static | 92.67 ± 3.98 | 84.59 ± 5.12 | +8.08 |
| FOCOPS | t2d | child | stochastic | fixed10 | 93.75 ± 3.72 | 87.65 ± 4.82 | +6.10 |
| FOCOPS | t2d | child | stochastic | selected | 93.79 ± 3.65 | 87.84 ± 4.51 | +5.95 |
| FOCOPS | t2d_no_pump | adolescent | deterministic | none | 90.24 ± 9.45 | 78.98 ± 15.59 | +11.26 |
| FOCOPS | t2d_no_pump | adolescent | deterministic | static | 90.28 ± 8.67 | 88.06 ± 8.58 | +2.22 |
| FOCOPS | t2d_no_pump | adolescent | deterministic | fixed10 | 93.20 ± 6.16 | 89.34 ± 7.45 | +3.86 |
| FOCOPS | t2d_no_pump | adolescent | deterministic | selected | 92.44 ± 6.66 | 89.11 ± 7.62 | +3.33 |
| FOCOPS | t2d_no_pump | adolescent | stochastic | none | 63.04 ± 32.09 | 58.10 ± 30.83 | +4.94 |
| FOCOPS | t2d_no_pump | adolescent | stochastic | static | 75.69 ± 21.06 | 74.67 ± 19.00 | +1.03 |
| FOCOPS | t2d_no_pump | adolescent | stochastic | fixed10 | 81.25 ± 16.28 | 82.58 ± 12.25 | -1.33 |
| FOCOPS | t2d_no_pump | adolescent | stochastic | selected | 81.25 ± 16.28 | 82.58 ± 12.25 | -1.33 |
| FOCOPS | t2d_no_pump | adult | deterministic | none | 96.99 ± 0.00 | 91.03 ± 0.59 | +5.96 |
| FOCOPS | t2d_no_pump | adult | deterministic | static | 96.99 ± 0.00 | 91.72 ± 0.46 | +5.27 |
| FOCOPS | t2d_no_pump | adult | deterministic | fixed10 | 96.99 ± 0.00 | 91.72 ± 0.46 | +5.27 |
| FOCOPS | t2d_no_pump | adult | deterministic | selected | 96.99 ± 0.00 | 91.72 ± 0.46 | +5.27 |
| FOCOPS | t2d_no_pump | adult | stochastic | none | 96.80 ± 0.53 | 91.42 ± 0.58 | +5.37 |
| FOCOPS | t2d_no_pump | adult | stochastic | static | 96.80 ± 0.53 | 92.08 ± 0.65 | +4.72 |
| FOCOPS | t2d_no_pump | adult | stochastic | fixed10 | 96.80 ± 0.53 | 91.68 ± 0.49 | +5.12 |
| FOCOPS | t2d_no_pump | child | deterministic | none | 94.25 ± 0.27 | 87.66 ± 1.42 | +6.60 |
| FOCOPS | t2d_no_pump | child | deterministic | static | 95.56 ± 1.94 | 91.22 ± 1.64 | +4.34 |
| FOCOPS | t2d_no_pump | child | deterministic | fixed10 | 95.50 ± 0.89 | 92.81 ± 0.61 | +2.69 |
| FOCOPS | t2d_no_pump | child | deterministic | selected | 95.50 ± 0.89 | 92.81 ± 0.61 | +2.69 |
| FOCOPS | t2d_no_pump | child | stochastic | none | 94.71 ± 2.05 | 82.53 ± 3.57 | +12.18 |
| FOCOPS | t2d_no_pump | child | stochastic | static | 96.33 ± 0.24 | 85.59 ± 1.46 | +10.74 |
| FOCOPS | t2d_no_pump | child | stochastic | fixed10 | 95.86 ± 0.16 | 89.55 ± 0.82 | +6.31 |
| FOCOPS | t2d_no_pump | child | stochastic | selected | 95.86 ± 0.16 | 89.55 ± 0.82 | +6.31 |
| OnCRPO | t1d | adolescent | deterministic | none | 100.00 ± 0.00 | 85.91 ± 3.49 | +14.09 |
| OnCRPO | t1d | adolescent | deterministic | static | 100.00 ± 0.00 | 85.72 ± 4.01 | +14.28 |
| OnCRPO | t1d | adolescent | deterministic | fixed10 | 100.00 ± 0.00 | 85.54 ± 4.12 | +14.46 |
| OnCRPO | t1d | adolescent | stochastic | none | 100.00 ± 0.00 | 85.21 ± 3.29 | +14.79 |
| OnCRPO | t1d | adolescent | stochastic | static | 100.00 ± 0.00 | 85.15 ± 3.74 | +14.85 |
| OnCRPO | t1d | adolescent | stochastic | fixed10 | 100.00 ± 0.00 | 85.05 ± 3.80 | +14.95 |
| OnCRPO | t1d | adult | deterministic | none | 95.37 ± 4.21 | 73.21 ± 10.72 | +22.16 |
| OnCRPO | t1d | adult | deterministic | static | 92.17 ± 2.26 | 69.58 ± 10.43 | +22.58 |
| OnCRPO | t1d | adult | deterministic | fixed10 | 92.17 ± 2.26 | 69.48 ± 10.51 | +22.69 |
| OnCRPO | t1d | adult | deterministic | selected | 92.17 ± 2.26 | 69.56 ± 10.42 | +22.61 |
| OnCRPO | t1d | adult | stochastic | none | 96.91 ± 4.65 | 71.67 ± 8.40 | +25.24 |
| OnCRPO | t1d | adult | stochastic | static | 96.95 ± 4.69 | 67.17 ± 9.73 | +29.79 |
| OnCRPO | t1d | adult | stochastic | fixed10 | 96.95 ± 4.69 | 67.25 ± 9.65 | +29.70 |
| OnCRPO | t1d | adult | stochastic | selected | 96.95 ± 4.69 | 67.17 ± 9.73 | +29.79 |
| OnCRPO | t1d | child | deterministic | none | 88.08 ± 9.06 | 73.86 ± 9.91 | +14.22 |
| OnCRPO | t1d | child | deterministic | static | 90.47 ± 6.60 | 79.72 ± 8.73 | +10.75 |
| OnCRPO | t1d | child | deterministic | fixed10 | 89.65 ± 6.91 | 80.50 ± 7.29 | +9.15 |
| OnCRPO | t1d | child | deterministic | selected | 90.47 ± 6.60 | 79.73 ± 9.12 | +10.74 |
| OnCRPO | t1d | child | stochastic | none | 87.50 ± 10.14 | 71.66 ± 12.00 | +15.84 |
| OnCRPO | t1d | child | stochastic | static | 88.46 ± 4.42 | 76.11 ± 11.00 | +12.35 |
| OnCRPO | t1d | child | stochastic | fixed10 | 88.46 ± 4.42 | 76.61 ± 9.62 | +11.86 |
| OnCRPO | t1d | child | stochastic | selected | 88.46 ± 4.42 | 76.61 ± 9.62 | +11.86 |
| OnCRPO | t2d | adolescent | deterministic | none | 92.09 ± 10.79 | 39.44 ± 39.25 | +52.65 |
| OnCRPO | t2d | adolescent | deterministic | static | 82.95 ± 26.63 | 48.79 ± 36.08 | +34.16 |
| OnCRPO | t2d | adolescent | deterministic | fixed10 | 91.69 ± 12.91 | 69.86 ± 20.19 | +21.84 |
| OnCRPO | t2d | adolescent | stochastic | none | 92.90 ± 11.60 | 40.60 ± 42.21 | +52.31 |
| OnCRPO | t2d | adolescent | stochastic | static | 88.81 ± 18.68 | 49.38 ± 36.82 | +39.44 |
| OnCRPO | t2d | adolescent | stochastic | fixed10 | 91.89 ± 13.56 | 60.96 ± 25.41 | +30.92 |
| OnCRPO | t2d | adolescent | stochastic | selected | 91.90 ± 13.53 | 74.33 ± 23.88 | +17.57 |
| OnCRPO | t2d | adult | deterministic | none | 97.57 ± 4.01 | 85.05 ± 5.39 | +12.52 |
| OnCRPO | t2d | adult | deterministic | static | 99.92 ± 0.13 | 85.65 ± 4.65 | +14.28 |
| OnCRPO | t2d | adult | deterministic | fixed10 | 99.92 ± 0.13 | 85.56 ± 4.91 | +14.37 |
| OnCRPO | t2d | adult | deterministic | selected | 99.92 ± 0.13 | 85.55 ± 4.78 | +14.37 |
| OnCRPO | t2d | adult | stochastic | none | 95.37 ± 4.17 | 85.11 ± 4.83 | +10.26 |
| OnCRPO | t2d | adult | stochastic | static | 95.37 ± 4.17 | 85.97 ± 5.30 | +9.40 |
| OnCRPO | t2d | adult | stochastic | fixed10 | 95.36 ± 4.18 | 85.84 ± 5.47 | +9.51 |
| OnCRPO | t2d | adult | stochastic | selected | 95.36 ± 4.18 | 85.99 ± 5.60 | +9.37 |
| OnCRPO | t2d | child | deterministic | none | 75.46 ± 17.63 | 61.30 ± 21.33 | +14.16 |
| OnCRPO | t2d | child | deterministic | static | 82.06 ± 12.83 | 70.07 ± 17.77 | +11.99 |
| OnCRPO | t2d | child | deterministic | fixed10 | 80.47 ± 13.37 | 72.00 ± 18.20 | +8.46 |
| OnCRPO | t2d | child | stochastic | none | 74.73 ± 18.40 | 61.76 ± 21.50 | +12.97 |
| OnCRPO | t2d | child | stochastic | static | 80.25 ± 13.57 | 70.38 ± 17.30 | +9.87 |
| OnCRPO | t2d | child | stochastic | fixed10 | 81.01 ± 13.03 | 72.84 ± 17.02 | +8.16 |
| OnCRPO | t2d | child | stochastic | selected | 80.25 ± 13.57 | 70.67 ± 17.11 | +9.57 |
| OnCRPO | t2d_no_pump | adolescent | deterministic | none | 95.79 ± 7.28 | 79.57 ± 14.19 | +16.22 |
| OnCRPO | t2d_no_pump | adolescent | deterministic | static | 94.83 ± 8.95 | 83.81 ± 12.39 | +11.02 |
| OnCRPO | t2d_no_pump | adolescent | deterministic | fixed10 | 96.64 ± 5.81 | 86.05 ± 9.22 | +10.59 |
| OnCRPO | t2d_no_pump | adolescent | stochastic | none | 93.44 ± 10.56 | 78.38 ± 15.25 | +15.06 |
| OnCRPO | t2d_no_pump | adolescent | stochastic | static | 95.95 ± 6.22 | 80.95 ± 14.18 | +15.00 |
| OnCRPO | t2d_no_pump | adolescent | stochastic | fixed10 | 96.41 ± 5.41 | 84.11 ± 10.09 | +12.31 |
| OnCRPO | t2d_no_pump | adolescent | stochastic | selected | 96.27 ± 5.66 | 83.90 ± 10.22 | +12.37 |
| OnCRPO | t2d_no_pump | adult | deterministic | none | 98.50 ± 1.20 | 92.44 ± 1.76 | +6.06 |
| OnCRPO | t2d_no_pump | adult | deterministic | static | 98.50 ± 1.20 | 92.93 ± 1.04 | +5.57 |
| OnCRPO | t2d_no_pump | adult | deterministic | fixed10 | 98.50 ± 1.20 | 92.96 ± 0.99 | +5.54 |
| OnCRPO | t2d_no_pump | adult | deterministic | selected | 98.50 ± 1.20 | 92.93 ± 1.04 | +5.57 |
| OnCRPO | t2d_no_pump | adult | stochastic | none | 98.77 ± 1.49 | 92.50 ± 1.89 | +6.26 |
| OnCRPO | t2d_no_pump | adult | stochastic | static | 98.77 ± 1.49 | 92.86 ± 1.35 | +5.90 |
| OnCRPO | t2d_no_pump | adult | stochastic | fixed10 | 98.77 ± 1.49 | 92.91 ± 1.28 | +5.86 |
| OnCRPO | t2d_no_pump | adult | stochastic | selected | 98.77 ± 1.49 | 92.86 ± 1.35 | +5.90 |
| OnCRPO | t2d_no_pump | child | deterministic | none | 86.34 ± 12.34 | 63.60 ± 15.07 | +22.75 |
| OnCRPO | t2d_no_pump | child | deterministic | static | 88.81 ± 5.90 | 71.51 ± 14.32 | +17.31 |
| OnCRPO | t2d_no_pump | child | deterministic | fixed10 | 89.57 ± 4.99 | 73.65 ± 12.67 | +15.92 |
| OnCRPO | t2d_no_pump | child | deterministic | selected | 89.22 ± 5.18 | 75.17 ± 10.24 | +14.05 |
| OnCRPO | t2d_no_pump | child | stochastic | none | 85.65 ± 11.77 | 59.52 ± 19.11 | +26.13 |
| OnCRPO | t2d_no_pump | child | stochastic | static | 88.54 ± 8.59 | 66.54 ± 17.28 | +22.00 |
| OnCRPO | t2d_no_pump | child | stochastic | fixed10 | 90.17 ± 5.80 | 68.38 ± 16.79 | +21.80 |
| OnCRPO | t2d_no_pump | child | stochastic | selected | 91.53 ± 3.53 | 72.42 ± 13.71 | +19.11 |
| RCPO | t1d | adolescent | deterministic | none | 99.65 ± 0.60 | 86.88 ± 4.60 | +12.77 |
| RCPO | t1d | adolescent | deterministic | static | 99.65 ± 0.60 | 86.84 ± 4.50 | +12.81 |
| RCPO | t1d | adolescent | deterministic | fixed10 | 99.65 ± 0.60 | 86.85 ± 4.52 | +12.80 |
| RCPO | t1d | adolescent | deterministic | selected | 99.65 ± 0.60 | 86.85 ± 4.52 | +12.80 |
| RCPO | t1d | adolescent | stochastic | none | 99.92 ± 0.13 | 84.32 ± 6.09 | +15.60 |
| RCPO | t1d | adolescent | stochastic | static | 99.85 ± 0.27 | 84.18 ± 6.08 | +15.66 |
| RCPO | t1d | adolescent | stochastic | fixed10 | 99.85 ± 0.27 | 84.18 ± 6.08 | +15.66 |
| RCPO | t1d | adolescent | stochastic | selected | 99.85 ± 0.27 | 84.18 ± 6.08 | +15.66 |
| RCPO | t1d | adult | deterministic | none | 100.00 ± 0.00 | 78.23 ± 7.53 | +21.77 |
| RCPO | t1d | adult | deterministic | static | 100.00 ± 0.00 | 76.35 ± 7.97 | +23.65 |
| RCPO | t1d | adult | deterministic | fixed10 | 100.00 ± 0.00 | 76.47 ± 7.76 | +23.53 |
| RCPO | t1d | adult | stochastic | none | 97.99 ± 3.47 | 78.18 ± 7.43 | +19.81 |
| RCPO | t1d | adult | stochastic | static | 97.99 ± 3.47 | 74.37 ± 6.74 | +23.62 |
| RCPO | t1d | adult | stochastic | fixed10 | 97.99 ± 3.47 | 74.42 ± 6.65 | +23.57 |
| RCPO | t1d | child | deterministic | none | 85.19 ± 1.45 | 69.43 ± 3.81 | +15.75 |
| RCPO | t1d | child | deterministic | static | 88.58 ± 4.49 | 71.10 ± 2.91 | +17.48 |
| RCPO | t1d | child | deterministic | fixed10 | 88.58 ± 4.49 | 72.45 ± 0.76 | +16.13 |
| RCPO | t1d | child | deterministic | selected | 88.58 ± 4.49 | 70.08 ± 3.44 | +18.50 |
| RCPO | t1d | child | stochastic | none | 86.96 ± 2.91 | 70.12 ± 3.35 | +16.84 |
| RCPO | t1d | child | stochastic | static | 89.54 ± 4.10 | 67.89 ± 5.70 | +21.65 |
| RCPO | t1d | child | stochastic | fixed10 | 89.54 ± 4.10 | 72.32 ± 2.48 | +17.22 |
| RCPO | t1d | child | stochastic | selected | 89.54 ± 4.10 | 70.39 ± 4.35 | +19.15 |
| RCPO | t2d | adolescent | deterministic | none | 99.96 ± 0.07 | 91.62 ± 1.20 | +8.34 |
| RCPO | t2d | adolescent | deterministic | static | 99.96 ± 0.07 | 91.45 ± 2.73 | +8.51 |
| RCPO | t2d | adolescent | deterministic | fixed10 | 99.96 ± 0.07 | 91.43 ± 2.70 | +8.53 |
| RCPO | t2d | adolescent | deterministic | selected | 99.96 ± 0.07 | 91.45 ± 2.73 | +8.51 |
| RCPO | t2d | adolescent | stochastic | none | 98.26 ± 2.81 | 90.21 ± 1.95 | +8.05 |
| RCPO | t2d | adolescent | stochastic | static | 98.26 ± 2.81 | 91.20 ± 2.11 | +7.07 |
| RCPO | t2d | adolescent | stochastic | fixed10 | 98.26 ± 2.81 | 91.19 ± 2.11 | +7.07 |
| RCPO | t2d | adolescent | stochastic | selected | 98.26 ± 2.81 | 91.20 ± 2.11 | +7.07 |
| RCPO | t2d | adult | deterministic | none | 97.45 ± 3.73 | 84.32 ± 2.92 | +13.14 |
| RCPO | t2d | adult | deterministic | static | 97.45 ± 3.73 | 85.87 ± 3.45 | +11.58 |
| RCPO | t2d | adult | deterministic | fixed10 | 97.45 ± 3.73 | 85.85 ± 3.36 | +11.60 |
| RCPO | t2d | adult | stochastic | none | 97.45 ± 3.73 | 84.16 ± 3.36 | +13.29 |
| RCPO | t2d | adult | stochastic | static | 97.45 ± 3.73 | 86.20 ± 3.37 | +11.25 |
| RCPO | t2d | adult | stochastic | fixed10 | 97.45 ± 3.73 | 86.26 ± 3.28 | +11.20 |
| RCPO | t2d | child | deterministic | none | 78.01 ± 12.58 | 64.27 ± 8.36 | +13.74 |
| RCPO | t2d | child | deterministic | static | 78.12 ± 12.51 | 63.74 ± 8.15 | +14.39 |
| RCPO | t2d | child | deterministic | fixed10 | 78.12 ± 12.51 | 63.87 ± 8.27 | +14.26 |
| RCPO | t2d | child | stochastic | none | 74.04 ± 11.46 | 62.85 ± 10.14 | +11.19 |
| RCPO | t2d | child | stochastic | static | 78.16 ± 14.11 | 62.73 ± 11.75 | +15.44 |
| RCPO | t2d | child | stochastic | fixed10 | 78.16 ± 14.11 | 62.97 ± 10.27 | +15.19 |
| RCPO | t2d_no_pump | adolescent | deterministic | none | 99.61 ± 0.57 | 92.87 ± 3.20 | +6.75 |
| RCPO | t2d_no_pump | adolescent | deterministic | static | 99.61 ± 0.57 | 94.02 ± 2.04 | +5.60 |
| RCPO | t2d_no_pump | adolescent | deterministic | fixed10 | 99.61 ± 0.57 | 94.03 ± 2.06 | +5.58 |
| RCPO | t2d_no_pump | adolescent | deterministic | selected | 99.61 ± 0.57 | 94.03 ± 2.06 | +5.58 |
| RCPO | t2d_no_pump | adolescent | stochastic | none | 99.42 ± 0.53 | 93.16 ± 2.69 | +6.26 |
| RCPO | t2d_no_pump | adolescent | stochastic | static | 99.42 ± 0.53 | 93.43 ± 2.63 | +6.00 |
| RCPO | t2d_no_pump | adolescent | stochastic | fixed10 | 99.27 ± 0.64 | 93.64 ± 2.58 | +5.63 |
| RCPO | t2d_no_pump | adolescent | stochastic | selected | 99.27 ± 0.64 | 93.64 ± 2.58 | +5.63 |
| RCPO | t2d_no_pump | adult | deterministic | none | 97.84 ± 3.16 | 92.29 ± 1.74 | +5.55 |
| RCPO | t2d_no_pump | adult | deterministic | static | 99.31 ± 0.69 | 92.43 ± 1.77 | +6.88 |
| RCPO | t2d_no_pump | adult | deterministic | fixed10 | 99.31 ± 0.69 | 92.47 ± 1.75 | +6.84 |
| RCPO | t2d_no_pump | adult | deterministic | selected | 99.31 ± 0.69 | 92.43 ± 1.75 | +6.87 |
| RCPO | t2d_no_pump | adult | stochastic | none | 98.23 ± 1.87 | 92.01 ± 1.57 | +6.22 |
| RCPO | t2d_no_pump | adult | stochastic | static | 98.23 ± 1.87 | 92.39 ± 1.73 | +5.84 |
| RCPO | t2d_no_pump | adult | stochastic | fixed10 | 98.23 ± 1.87 | 92.43 ± 1.72 | +5.79 |
| RCPO | t2d_no_pump | adult | stochastic | selected | 98.23 ± 1.87 | 92.39 ± 1.74 | +5.84 |
| RCPO | t2d_no_pump | child | deterministic | none | 81.29 ± 14.23 | 67.25 ± 15.59 | +14.04 |
| RCPO | t2d_no_pump | child | deterministic | static | 81.33 ± 14.16 | 71.61 ± 17.26 | +9.72 |
| RCPO | t2d_no_pump | child | deterministic | fixed10 | 82.11 ± 14.58 | 73.22 ± 17.47 | +8.89 |
| RCPO | t2d_no_pump | child | deterministic | selected | 82.23 ± 14.17 | 72.45 ± 18.13 | +9.78 |
| RCPO | t2d_no_pump | child | stochastic | none | 81.25 ± 14.18 | 64.65 ± 15.81 | +16.60 |
| RCPO | t2d_no_pump | child | stochastic | static | 81.25 ± 14.18 | 69.18 ± 16.80 | +12.07 |
| RCPO | t2d_no_pump | child | stochastic | fixed10 | 82.29 ± 14.77 | 71.69 ± 17.05 | +10.60 |
| TRPOLag | t1d | adolescent | deterministic | none | 100.00 ± 0.00 | 87.93 ± 2.57 | +12.07 |
| TRPOLag | t1d | adolescent | deterministic | static | 100.00 ± 0.00 | 85.55 ± 5.18 | +14.45 |
| TRPOLag | t1d | adolescent | deterministic | fixed10 | 100.00 ± 0.00 | 85.59 ± 5.20 | +14.41 |
| TRPOLag | t1d | adolescent | deterministic | selected | 100.00 ± 0.00 | 85.55 ± 5.18 | +14.45 |
| TRPOLag | t1d | adolescent | stochastic | none | 100.00 ± 0.00 | 85.34 ± 6.73 | +14.66 |
| TRPOLag | t1d | adolescent | stochastic | static | 100.00 ± 0.00 | 83.96 ± 7.98 | +16.04 |
| TRPOLag | t1d | adolescent | stochastic | fixed10 | 100.00 ± 0.00 | 83.97 ± 7.99 | +16.03 |
| TRPOLag | t1d | adolescent | stochastic | selected | 100.00 ± 0.00 | 83.97 ± 7.98 | +16.03 |
| TRPOLag | t1d | adult | deterministic | none | 99.11 ± 1.44 | 87.26 ± 5.33 | +11.86 |
| TRPOLag | t1d | adult | deterministic | static | 99.11 ± 1.44 | 86.10 ± 5.04 | +13.02 |
| TRPOLag | t1d | adult | deterministic | fixed10 | 99.11 ± 1.44 | 86.26 ± 5.34 | +12.86 |
| TRPOLag | t1d | adult | deterministic | selected | 99.11 ± 1.44 | 86.10 ± 5.04 | +13.01 |
| TRPOLag | t1d | adult | stochastic | none | 99.96 ± 0.07 | 85.62 ± 5.78 | +14.35 |
| TRPOLag | t1d | adult | stochastic | static | 99.96 ± 0.07 | 84.01 ± 3.65 | +15.95 |
| TRPOLag | t1d | adult | stochastic | fixed10 | 99.96 ± 0.07 | 84.19 ± 3.89 | +15.77 |
| TRPOLag | t1d | adult | stochastic | selected | 99.96 ± 0.07 | 84.02 ± 3.64 | +15.94 |
| TRPOLag | t1d | child | deterministic | none | 90.08 ± 6.49 | 68.29 ± 2.71 | +21.80 |
| TRPOLag | t1d | child | deterministic | static | 90.97 ± 6.00 | 75.72 ± 3.82 | +15.26 |
| TRPOLag | t1d | child | deterministic | fixed10 | 90.48 ± 5.24 | 75.63 ± 3.52 | +14.85 |
| TRPOLag | t1d | child | deterministic | selected | 90.48 ± 5.24 | 75.63 ± 3.52 | +14.85 |
| TRPOLag | t1d | child | stochastic | none | 89.78 ± 4.81 | 69.50 ± 4.16 | +20.28 |
| TRPOLag | t1d | child | stochastic | static | 90.66 ± 3.60 | 75.37 ± 2.66 | +15.29 |
| TRPOLag | t1d | child | stochastic | fixed10 | 90.48 ± 3.86 | 76.95 ± 2.25 | +13.54 |
| TRPOLag | t1d | child | stochastic | selected | 90.79 ± 3.42 | 75.35 ± 2.75 | +15.44 |
| TRPOLag | t2d | adolescent | deterministic | none | 100.00 ± 0.00 | 91.49 ± 6.15 | +8.51 |
| TRPOLag | t2d | adolescent | deterministic | static | 100.00 ± 0.00 | 93.62 ± 3.62 | +6.38 |
| TRPOLag | t2d | adolescent | deterministic | fixed10 | 100.00 ± 0.00 | 93.62 ± 3.62 | +6.38 |
| TRPOLag | t2d | adolescent | deterministic | selected | 100.00 ± 0.00 | 93.62 ± 3.62 | +6.38 |
| TRPOLag | t2d | adolescent | stochastic | none | 100.00 ± 0.00 | 91.29 ± 5.90 | +8.71 |
| TRPOLag | t2d | adolescent | stochastic | static | 100.00 ± 0.00 | 93.57 ± 4.28 | +6.43 |
| TRPOLag | t2d | adolescent | stochastic | fixed10 | 100.00 ± 0.00 | 93.57 ± 4.27 | +6.43 |
| TRPOLag | t2d | adolescent | stochastic | selected | 100.00 ± 0.00 | 93.57 ± 4.28 | +6.43 |
| TRPOLag | t2d | adult | deterministic | none | 97.80 ± 3.04 | 87.29 ± 4.54 | +10.51 |
| TRPOLag | t2d | adult | deterministic | static | 99.00 ± 1.04 | 88.74 ± 5.12 | +10.26 |
| TRPOLag | t2d | adult | deterministic | fixed10 | 99.00 ± 1.04 | 88.88 ± 5.21 | +10.12 |
| TRPOLag | t2d | adult | stochastic | none | 99.61 ± 0.67 | 88.00 ± 4.18 | +11.62 |
| TRPOLag | t2d | adult | stochastic | static | 99.61 ± 0.67 | 89.17 ± 5.28 | +10.44 |
| TRPOLag | t2d | adult | stochastic | fixed10 | 99.61 ± 0.67 | 89.22 ± 5.23 | +10.40 |
| TRPOLag | t2d | child | deterministic | none | 73.77 ± 1.69 | 59.05 ± 1.53 | +14.72 |
| TRPOLag | t2d | child | deterministic | static | 80.40 ± 2.45 | 67.32 ± 2.95 | +13.08 |
| TRPOLag | t2d | child | deterministic | fixed10 | 78.77 ± 0.43 | 67.82 ± 2.80 | +10.95 |
| TRPOLag | t2d | child | stochastic | none | 75.23 ± 0.80 | 58.53 ± 2.11 | +16.70 |
| TRPOLag | t2d | child | stochastic | static | 81.48 ± 2.12 | 68.35 ± 5.47 | +13.13 |
| TRPOLag | t2d | child | stochastic | fixed10 | 80.03 ± 1.66 | 68.62 ± 7.33 | +11.41 |
| TRPOLag | t2d | child | stochastic | selected | 79.99 ± 1.70 | 66.45 ± 5.48 | +13.54 |
| TRPOLag | t2d_no_pump | adolescent | deterministic | none | 97.38 ± 2.60 | 85.63 ± 5.99 | +11.74 |
| TRPOLag | t2d_no_pump | adolescent | deterministic | static | 98.80 ± 1.35 | 86.84 ± 5.29 | +11.96 |
| TRPOLag | t2d_no_pump | adolescent | deterministic | fixed10 | 97.56 ± 2.34 | 86.86 ± 5.45 | +10.69 |
| TRPOLag | t2d_no_pump | adolescent | stochastic | none | 97.99 ± 1.87 | 84.37 ± 4.81 | +13.62 |
| TRPOLag | t2d_no_pump | adolescent | stochastic | static | 97.99 ± 1.87 | 85.29 ± 4.32 | +12.71 |
| TRPOLag | t2d_no_pump | adolescent | stochastic | fixed10 | 97.80 ± 1.71 | 85.50 ± 4.31 | +12.30 |
| TRPOLag | t2d_no_pump | adult | deterministic | none | 98.38 ± 2.03 | 94.56 ± 1.59 | +3.82 |
| TRPOLag | t2d_no_pump | adult | deterministic | static | 98.38 ± 2.03 | 94.83 ± 1.39 | +3.55 |
| TRPOLag | t2d_no_pump | adult | deterministic | fixed10 | 98.38 ± 2.03 | 94.84 ± 1.37 | +3.54 |
| TRPOLag | t2d_no_pump | adult | deterministic | selected | 98.38 ± 2.03 | 94.83 ± 1.39 | +3.55 |
| TRPOLag | t2d_no_pump | adult | stochastic | none | 99.96 ± 0.07 | 94.54 ± 1.61 | +5.42 |
| TRPOLag | t2d_no_pump | adult | stochastic | static | 99.96 ± 0.07 | 94.62 ± 1.58 | +5.34 |
| TRPOLag | t2d_no_pump | adult | stochastic | fixed10 | 99.96 ± 0.07 | 94.68 ± 1.52 | +5.29 |
| TRPOLag | t2d_no_pump | adult | stochastic | selected | 99.96 ± 0.07 | 94.63 ± 1.58 | +5.33 |
| TRPOLag | t2d_no_pump | child | deterministic | none | 83.95 ± 16.07 | 67.96 ± 16.24 | +15.99 |
| TRPOLag | t2d_no_pump | child | deterministic | static | 82.83 ± 15.07 | 71.22 ± 16.47 | +11.61 |
| TRPOLag | t2d_no_pump | child | deterministic | fixed10 | 83.58 ± 15.65 | 74.26 ± 18.19 | +9.32 |
| TRPOLag | t2d_no_pump | child | deterministic | selected | 83.78 ± 15.29 | 73.25 ± 18.71 | +10.53 |
| TRPOLag | t2d_no_pump | child | stochastic | none | 83.95 ± 16.07 | 68.60 ± 18.86 | +15.35 |
| TRPOLag | t2d_no_pump | child | stochastic | static | 82.79 ± 15.14 | 69.87 ± 16.06 | +12.92 |
| TRPOLag | t2d_no_pump | child | stochastic | fixed10 | 83.24 ± 15.48 | 73.42 ± 18.29 | +9.83 |
| TRPOLag | t2d_no_pump | child | stochastic | selected | 83.53 ± 15.71 | 73.34 ± 15.80 | +10.19 |
| PCPO | t1d | adolescent | deterministic | none | 100.00 ± 0.00 | 87.99 ± 6.06 | +12.01 |
| PCPO | t1d | adolescent | deterministic | static | 100.00 ± 0.00 | 87.37 ± 5.90 | +12.63 |
| PCPO | t1d | adolescent | deterministic | fixed10 | 100.00 ± 0.00 | 87.42 ± 5.88 | +12.58 |
| PCPO | t1d | adolescent | deterministic | selected | 100.00 ± 0.00 | 87.37 ± 5.90 | +12.63 |
| PCPO | t1d | adolescent | stochastic | none | 100.00 ± 0.00 | 85.78 ± 7.78 | +14.22 |
| PCPO | t1d | adolescent | stochastic | static | 100.00 ± 0.00 | 85.11 ± 7.61 | +14.89 |
| PCPO | t1d | adolescent | stochastic | fixed10 | 100.00 ± 0.00 | 85.11 ± 7.62 | +14.89 |
| PCPO | t1d | adolescent | stochastic | selected | 100.00 ± 0.00 | 85.11 ± 7.61 | +14.89 |
| PCPO | t1d | adult | deterministic | none | 100.00 ± 0.00 | 80.43 ± 9.80 | +19.57 |
| PCPO | t1d | adult | deterministic | static | 100.00 ± 0.00 | 77.85 ± 10.76 | +22.15 |
| PCPO | t1d | adult | deterministic | fixed10 | 100.00 ± 0.00 | 77.92 ± 10.79 | +22.08 |
| PCPO | t1d | adult | deterministic | selected | 100.00 ± 0.00 | 77.91 ± 10.78 | +22.09 |
| PCPO | t1d | adult | stochastic | none | 100.00 ± 0.00 | 80.21 ± 10.05 | +19.79 |
| PCPO | t1d | adult | stochastic | static | 100.00 ± 0.00 | 78.33 ± 10.31 | +21.67 |
| PCPO | t1d | adult | stochastic | fixed10 | 100.00 ± 0.00 | 78.34 ± 10.33 | +21.66 |
| PCPO | t1d | adult | stochastic | selected | 100.00 ± 0.00 | 78.35 ± 10.30 | +21.65 |
| PCPO | t1d | child | deterministic | none | 91.78 ± 7.42 | 75.87 ± 6.18 | +15.91 |
| PCPO | t1d | child | deterministic | static | 97.22 ± 1.57 | 84.22 ± 6.14 | +13.01 |
| PCPO | t1d | child | deterministic | fixed10 | 93.33 ± 8.18 | 85.76 ± 3.80 | +7.57 |
| PCPO | t1d | child | deterministic | selected | 97.22 ± 1.57 | 85.10 ± 5.28 | +12.12 |
| PCPO | t1d | child | stochastic | none | 95.56 ± 1.01 | 77.66 ± 5.92 | +17.90 |
| PCPO | t1d | child | stochastic | static | 96.45 ± 2.74 | 84.15 ± 4.37 | +12.30 |
| PCPO | t1d | child | stochastic | fixed10 | 96.45 ± 2.74 | 83.84 ± 5.36 | +12.61 |
| PCPO | t1d | child | stochastic | selected | 96.45 ± 2.74 | 83.73 ± 5.43 | +12.72 |
| PCPO | t2d | adolescent | deterministic | none | 100.00 ± 0.00 | 94.34 ± 1.34 | +5.66 |
| PCPO | t2d | adolescent | deterministic | static | 100.00 ± 0.00 | 94.27 ± 1.25 | +5.73 |
| PCPO | t2d | adolescent | deterministic | fixed10 | 100.00 ± 0.00 | 94.27 ± 1.25 | +5.73 |
| PCPO | t2d | adolescent | deterministic | selected | 100.00 ± 0.00 | 94.27 ± 1.25 | +5.73 |
| PCPO | t2d | adolescent | stochastic | none | 98.84 ± 2.00 | 93.40 ± 1.21 | +5.44 |
| PCPO | t2d | adolescent | stochastic | static | 97.18 ± 4.88 | 93.67 ± 1.51 | +3.52 |
| PCPO | t2d | adolescent | stochastic | fixed10 | 97.18 ± 4.88 | 93.65 ± 1.50 | +3.54 |
| PCPO | t2d | adolescent | stochastic | selected | 97.18 ± 4.88 | 93.67 ± 1.51 | +3.52 |
| PCPO | t2d | adult | deterministic | none | 94.83 ± 8.36 | 81.91 ± 13.41 | +12.92 |
| PCPO | t2d | adult | deterministic | static | 94.83 ± 8.36 | 81.87 ± 14.77 | +12.96 |
| PCPO | t2d | adult | deterministic | fixed10 | 94.83 ± 8.36 | 81.78 ± 14.96 | +13.05 |
| PCPO | t2d | adult | deterministic | selected | 94.83 ± 8.36 | 81.87 ± 14.79 | +12.96 |
| PCPO | t2d | adult | stochastic | none | 94.44 ± 8.63 | 82.43 ± 13.09 | +12.01 |
| PCPO | t2d | adult | stochastic | static | 94.48 ± 8.67 | 81.29 ± 14.01 | +13.19 |
| PCPO | t2d | adult | stochastic | fixed10 | 94.48 ± 8.67 | 81.61 ± 13.61 | +12.87 |
| PCPO | t2d | adult | stochastic | selected | 94.48 ± 8.67 | 81.32 ± 13.96 | +13.17 |
| PCPO | t2d | child | deterministic | none | 84.88 ± 14.72 | 70.95 ± 22.74 | +13.92 |
| PCPO | t2d | child | deterministic | static | 85.30 ± 13.36 | 78.77 ± 11.20 | +6.54 |
| PCPO | t2d | child | deterministic | fixed10 | 83.67 ± 16.15 | 80.22 ± 9.38 | +3.44 |
| PCPO | t2d | child | stochastic | none | 83.68 ± 16.58 | 71.44 ± 22.29 | +12.24 |
| PCPO | t2d | child | stochastic | static | 87.27 ± 12.46 | 78.04 ± 11.39 | +9.23 |
| PCPO | t2d | child | stochastic | fixed10 | 88.01 ± 11.18 | 79.80 ± 9.86 | +8.21 |
| PCPO | t2d | child | stochastic | selected | 87.27 ± 12.46 | 78.50 ± 11.02 | +8.77 |
| PCPO | t2d_no_pump | adolescent | deterministic | none | 83.91 ± 15.68 | 77.29 ± 7.27 | +6.62 |
| PCPO | t2d_no_pump | adolescent | deterministic | static | 80.40 ± 18.96 | 77.53 ± 6.97 | +2.87 |
| PCPO | t2d_no_pump | adolescent | deterministic | fixed10 | 91.56 ± 10.49 | 82.26 ± 3.56 | +9.30 |
| PCPO | t2d_no_pump | adolescent | deterministic | selected | 91.56 ± 10.49 | 82.26 ± 3.56 | +9.30 |
| PCPO | t2d_no_pump | adolescent | stochastic | none | 80.98 ± 16.30 | 72.76 ± 11.22 | +8.22 |
| PCPO | t2d_no_pump | adolescent | stochastic | static | 80.52 ± 18.95 | 73.14 ± 11.12 | +7.38 |
| PCPO | t2d_no_pump | adolescent | stochastic | fixed10 | 88.90 ± 10.59 | 81.64 ± 3.95 | +7.26 |
| PCPO | t2d_no_pump | adolescent | stochastic | selected | 90.38 ± 8.44 | 81.00 ± 5.07 | +9.38 |
| PCPO | t2d_no_pump | adult | deterministic | none | 99.19 ± 0.81 | 93.44 ± 2.17 | +5.75 |
| PCPO | t2d_no_pump | adult | deterministic | static | 99.19 ± 0.81 | 93.58 ± 2.24 | +5.61 |
| PCPO | t2d_no_pump | adult | deterministic | fixed10 | 99.19 ± 0.81 | 93.62 ± 2.20 | +5.57 |
| PCPO | t2d_no_pump | adult | deterministic | selected | 99.19 ± 0.81 | 93.60 ± 2.21 | +5.59 |
| PCPO | t2d_no_pump | adult | stochastic | none | 99.11 ± 0.81 | 93.65 ± 2.20 | +5.47 |
| PCPO | t2d_no_pump | adult | stochastic | static | 99.11 ± 0.81 | 93.77 ± 2.18 | +5.34 |
| PCPO | t2d_no_pump | adult | stochastic | fixed10 | 99.11 ± 0.81 | 93.81 ± 2.19 | +5.30 |
| PCPO | t2d_no_pump | adult | stochastic | selected | 99.11 ± 0.81 | 93.79 ± 2.19 | +5.32 |
| PCPO | t2d_no_pump | child | deterministic | none | 69.75 ± 19.72 | 54.07 ± 17.14 | +15.68 |
| PCPO | t2d_no_pump | child | deterministic | static | 68.90 ± 22.17 | 59.30 ± 17.87 | +9.61 |
| PCPO | t2d_no_pump | child | deterministic | fixed10 | 74.94 ± 14.37 | 63.43 ± 15.87 | +11.51 |
| PCPO | t2d_no_pump | child | deterministic | selected | 68.90 ± 22.17 | 60.22 ± 17.77 | +8.69 |
| PCPO | t2d_no_pump | child | stochastic | none | 73.38 ± 23.70 | 52.54 ± 18.31 | +20.84 |
| PCPO | t2d_no_pump | child | stochastic | static | 75.66 ± 21.65 | 58.94 ± 11.02 | +16.71 |
| PCPO | t2d_no_pump | child | stochastic | fixed10 | 75.17 ± 18.65 | 60.40 ± 14.60 | +14.77 |
| PCPO | t2d_no_pump | child | stochastic | selected | 76.22 ± 19.77 | 59.45 ± 12.57 | +16.77 |

## Historical completed CPO / tuned PPO-Lag

Frozen historical implementations; keep separate from the six-algorithm campaign. All numeric arms use three policy seeds; predictive arms also use three predictor seeds. Full original report retains safety and forecast-support limitations.

### deterministic

| Algorithm | Type | Cohort | None TIR % | Static TIR % | Fixed10 TIR % | Selected TIR % | Selected penalty |
|---|---|---|---:|---:|---:|---:|---:|
| CPO | t1d | adolescent | 87.55 ± 9.61 | 87.40 ± 9.86 | 87.40 ± 9.87 | 87.40 ± 9.86 | 1 |
| CPO | t1d | adult | 72.49 ± 17.44 | 72.74 ± 16.57 | 72.72 ± 16.61 | NA | NA |
| CPO | t1d | child | 77.79 ± 11.88 | 78.53 ± 11.56 | 78.59 ± 11.55 | NA | NA |
| CPO | t2d | adolescent | 80.77 ± 20.78 | 86.88 ± 10.71 | 88.58 ± 8.02 | 88.46 ± 8.20 | 5 |
| CPO | t2d | adult | 86.76 ± 9.66 | 87.70 ± 8.97 | 87.75 ± 8.90 | NA | NA |
| CPO | t2d | child | 67.07 ± 24.70 | 69.35 ± 24.20 | 69.40 ± 24.35 | NA | NA |
| CPO | t2d_no_pump | adolescent | 48.45 ± 5.73 | 69.15 ± 3.67 | 76.38 ± 4.64 | 78.96 ± 8.03 | 20 |
| CPO | t2d_no_pump | adult | 95.94 ± 0.68 | 96.06 ± 0.64 | 96.07 ± 0.65 | 96.06 ± 0.64 | 1 |
| CPO | t2d_no_pump | child | 72.60 ± 13.30 | 75.83 ± 7.37 | 81.66 ± 5.52 | 79.82 ± 5.75 | 5 |
| PPOLag | t1d | adolescent | 87.70 ± 1.80 | 87.74 ± 1.84 | 87.73 ± 1.82 | 87.73 ± 1.82 | 1 |
| PPOLag | t1d | adult | 55.21 ± 10.33 | 50.68 ± 8.84 | 50.69 ± 8.84 | NA | NA |
| PPOLag | t1d | child | 43.94 ± 34.83 | 62.02 ± 25.15 | 67.92 ± 21.07 | 67.92 ± 21.07 | 5 |
| PPOLag | t2d | adolescent | 87.31 ± 9.68 | 88.41 ± 9.41 | 88.47 ± 9.32 | NA | NA |
| PPOLag | t2d | adult | 86.19 ± 3.51 | 87.63 ± 6.01 | 87.66 ± 5.97 | 87.66 ± 5.97 | 1 |
| PPOLag | t2d | child | 43.40 ± 6.35 | 71.74 ± 16.15 | 80.69 ± 6.67 | 80.87 ± 6.21 | 5 |
| PPOLag | t2d_no_pump | adolescent | 77.43 ± 22.27 | 84.43 ± 16.03 | 87.81 ± 9.91 | 87.81 ± 9.91 | 5 |
| PPOLag | t2d_no_pump | adult | 93.78 ± 2.65 | 94.03 ± 3.31 | 94.14 ± 3.18 | 94.03 ± 3.31 | 1 |
| PPOLag | t2d_no_pump | child | 70.27 ± 14.06 | 76.03 ± 12.13 | 86.01 ± 6.71 | 86.01 ± 6.71 | 10 |

### stochastic

| Algorithm | Type | Cohort | None TIR % | Static TIR % | Fixed10 TIR % | Selected TIR % | Selected penalty |
|---|---|---|---:|---:|---:|---:|---:|
| CPO | t1d | adolescent | 87.30 ± 10.25 | 87.12 ± 10.55 | 87.12 ± 10.56 | 87.12 ± 10.55 | 1 |
| CPO | t1d | adult | 75.15 ± 15.12 | 72.42 ± 17.10 | 72.46 ± 17.05 | 72.46 ± 17.05 | 1 |
| CPO | t1d | child | 73.83 ± 11.12 | 76.49 ± 10.89 | 77.42 ± 10.77 | NA | NA |
| CPO | t2d | adolescent | 80.65 ± 21.12 | 86.59 ± 11.30 | 88.59 ± 8.08 | 88.59 ± 8.08 | 10 |
| CPO | t2d | adult | 87.12 ± 9.61 | 87.86 ± 9.03 | 88.19 ± 8.72 | 88.19 ± 8.72 | 10 |
| CPO | t2d | child | 66.41 ± 25.29 | 68.34 ± 24.65 | 68.20 ± 25.27 | 68.60 ± 24.48 | 5 |
| CPO | t2d_no_pump | adolescent | 44.62 ± 5.01 | 64.33 ± 7.14 | 70.62 ± 7.22 | 72.68 ± 9.68 | 20 |
| CPO | t2d_no_pump | adult | 95.80 ± 0.85 | 95.92 ± 0.80 | 95.89 ± 0.79 | 95.92 ± 0.80 | 1 |
| CPO | t2d_no_pump | child | 70.60 ± 12.88 | 73.13 ± 8.82 | 78.79 ± 5.24 | NA | NA |
| PPOLag | t1d | adolescent | 87.85 ± 1.30 | 87.77 ± 1.28 | 87.79 ± 1.24 | 87.79 ± 1.24 | 1 |
| PPOLag | t1d | adult | 53.72 ± 4.87 | 49.49 ± 7.20 | 49.68 ± 7.09 | 49.56 ± 7.11 | 1 |
| PPOLag | t1d | child | 39.60 ± 34.77 | 59.67 ± 25.22 | 66.72 ± 19.99 | 66.72 ± 19.99 | 10 |
| PPOLag | t2d | adolescent | 75.94 ± 22.88 | 77.45 ± 21.99 | 81.86 ± 17.17 | NA | NA |
| PPOLag | t2d | adult | 86.14 ± 5.44 | 87.75 ± 7.20 | 87.78 ± 7.14 | 87.78 ± 7.18 | 1 |
| PPOLag | t2d | child | 39.12 ± 15.14 | 59.74 ± 18.72 | 67.07 ± 13.72 | 66.66 ± 14.63 | 5 |
| PPOLag | t2d_no_pump | adolescent | 53.78 ± 23.72 | 75.64 ± 15.19 | 80.58 ± 8.68 | 80.58 ± 8.68 | 10 |
| PPOLag | t2d_no_pump | adult | 93.83 ± 1.98 | 94.38 ± 2.52 | 94.38 ± 2.52 | 94.36 ± 2.55 | 1 |
| PPOLag | t2d_no_pump | child | 62.66 ± 7.35 | 72.48 ± 11.34 | 81.43 ± 8.17 | 79.42 ± 8.83 | 5 |

Existing historical report: /p/safead/glucoalg-reports-20260922/evaluation-analysis-20260930/analysis-report.md
