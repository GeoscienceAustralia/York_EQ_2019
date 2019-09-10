GA_vul = {}
PGA = '0.0 0.01 0.06 0.11 0.16 0.21 0.26 0.31 0.36 0.41 0.46 0.51 0.56 0.61 0.66 0.71 0.76 0.81 0.86 0.91 0.96 1.01 1.06 1.11 1.16 1.21 1.26 1.31 1.36'
PGA = [float(x) for x in PGA.split(' ')]

# URML
#    <vulnerabilityFunction id="13_LBM_T_Pre1945" dist="LN">
tmp = '0.0 0.00145 0.06930 0.12854 0.18126 0.22912 0.27253 0.31190 0.34768 0.38029 0.41007 0.43737 0.46244 0.48554 0.50687 0.52663 0.54496 0.56203 0.57793 0.59279 0.60670 0.61975 0.63200 0.64354 0.65441 0.66468 0.67439 0.68358 0.69229'
GA_vul['URML_Pre1945'] = [float(x) for x in tmp.split(' ')]

#    <vulnerabilityFunction id="13_LBM_T_Post1945" dist="LN">
tmp = '0.0 0.00054 0.02593 0.06484 0.11341 0.16629 0.22046 0.27375 0.32491 0.37314 0.41805 0.45955 0.49767 0.53256 0.56442 0.59350 0.62003 0.64424 0.66634 0.68655 0.70505 0.72200 0.73755 0.75188 0.76506 0.77723 0.78847 0.79890 0.80853'
GA_vul['URML_Post1945'] = [float(x) for x in tmp.split(' ')]

# Timber
#    <vulnerabilityFunction id="W1BVMETAL_Pre1945" dist="LN">
tmp = '0.0 0.00019 0.00889 0.02670 0.05397 0.08845 0.12839 0.17197 0.21763 0.26397 0.30988 0.35457 0.39739 0.43801 0.47619 0.51183 0.54495 0.57562 0.60391 0.63000 0.65402 0.67610 0.69640 0.71511 0.73231 0.74816 0.76277 0.77627 0.78870'
GA_vul['Timber_Pre1945'] = [float(x) for x in tmp.split(' ')]

#    <vulnerabilityFunction id="W1BVMETAL_Post1945" dist="LN">
tmp = '0.0 0.00007 0.00318 0.01273 0.03192 0.06113 0.10008 0.14728 0.20076 0.25810 0.31697 0.37537 0.43165 0.48476 0.53402 0.57911 0.62000 0.65684 0.68979 0.71924 0.74549 0.76883 0.78961 0.80817 0.82469 0.83944 0.85264 0.86450 0.87508'
GA_vul['Timber_Post1945'] = [float(x) for x in tmp.split(' ')]

#C1L
#    <vulnerabilityFunction id="13_S_URM_Pre1996" dist="LN">
tmp = '0.0 0.00000 0.00058 0.04944 0.19038 0.35186 0.49356 0.60588 0.69161 0.75640 0.80548 0.84290 0.87180 0.89430 0.91201 0.92612 0.93748 0.94672 0.95426 0.96049 0.96568 0.97001 0.97367 0.97678 0.97943 0.98171 0.98367 0.98537 0.98684'
GA_vul['C1L_Pre1996'] = [float(x) for x in tmp.split(' ')]

#    <vulnerabilityFunction id="13_S_URM_Post1996" dist="LN">
tmp = '0.0 0.00000 0.00000 0.00156 0.02644 0.09780 0.19840 0.30373 0.40185 0.48863 0.56333 0.62660 0.67982 0.72437 0.76164 0.79292 0.81926 0.84155 0.86040 0.87649 0.89029 0.90213 0.91236 0.92128 0.92903 0.93583 0.94180 0.94709 0.95173'
GA_vul['C1L_Post1996'] = [float(x) for x in tmp.split(' ')]

#C2L
#    <vulnerabilityFunction id="13_S_O_Pre1996" dist="LN">
tmp = '0.0 0.00000 0.00000 0.00075 0.02321 0.11048 0.25343 0.40934 0.54828 0.65957 0.74399 0.80631 0.85213 0.88578 0.91066 0.92932 0.94349 0.95440 0.96282 0.96945 0.97473 0.97894 0.98234 0.98512 0.98739 0.98927 0.99082 0.99213 0.99321'
GA_vul['C2L_Pre1996'] = [float(x) for x in tmp.split(' ')]

#    <vulnerabilityFunction id="13_S_O_Post1996" dist="LN">
tmp = '0.0 0.00000 0.00000 0.00000 0.00048 0.00754 0.03660 0.09677 0.18286 0.28201 0.38243 0.47632 0.55980 0.63165 0.69217 0.74263 0.78438 0.81890 0.84723 0.87063 0.89000 0.90599 0.91928 0.93043 0.93974 0.94758 0.95420 0.95984 0.96460'
GA_vul['C2L_Post1996'] = [float(x) for x in tmp.split(' ')]

# S2L
#    <vulnerabilityFunction id="ISS_SS_S_Pre1996" dist="LN">
tmp = '0.0 0.00000 0.00001 0.00487 0.04540 0.13335 0.24482 0.35652 0.45716 0.54324 0.61511 0.67443 0.72337 0.76373 0.79712 0.82492 0.84818 0.86779 0.88431 0.89837 0.91039 0.92067 0.92954 0.93724 0.94391 0.94974 0.95483 0.95933 0.96325'
GA_vul['S2L_Pre1996'] = [float(x) for x in tmp.split(' ')]

#    <vulnerabilityFunction id="ISS_SS_S_Post1996" dist="LN">
tmp = '0.0 0.00000 0.00000 0.00028 0.00572 0.02751 0.07065 0.13057 0.19970 0.27128 0.34087 0.40603 0.46562 0.51939 0.56750 0.61036 0.64848 0.68237 0.71242 0.73917 0.76299 0.78419 0.80312 0.82008 0.83527 0.84890 0.86116 0.87224 0.88218'
GA_vul['S2L_Post1996'] = [float(x) for x in tmp.split(' ')]

# S5L
#    <vulnerabilityFunction id="ISS_URM_PS_Pre1996" dist="LN">
tmp = '0.0 0.00000 0.00113 0.05760 0.20889 0.37487 0.51341 0.61983 0.69982 0.76002 0.80575 0.84085 0.86822 0.88977 0.90693 0.92079 0.93208 0.94138 0.94906 0.95549 0.96091 0.96548 0.96939 0.97275 0.97563 0.97814 0.98032 0.98223 0.98389'
GA_vul['S5L_Pre1996'] = [float(x) for x in tmp.split(' ')]


# May PGA
# <imls imt="PGA">0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40</imls>

# <vulnerabilityFunction id="13_LBM_T_Pre1945" dist="LN">
#     <meanLRs>0.00000 0.01171 0.03654 0.06968 0.10823 0.15002 0.19333 0.23688 0.27975 0.41792 0.54327 0.65733 0.76068 0.82528 0.86512 0.90062 0.91098 0.92073 0.92993 0.93858 0.94680 0.95463 0.96213 0.96930 0.97609 0.98266 0.98899 0.99501 1.00000</meanLRs>
# <vulnerabilityFunction id="13_LBM_T_Post1945" dist="LN">
#     <meanLRs>0.00000 0.00340 0.01502 0.03533 0.06386 0.09953 0.14083 0.18612 0.23382 0.33132 0.41947 0.49967 0.57235 0.63792 0.69768 0.75154 0.77745 0.80182 0.82483 0.84646 0.86699 0.88656 0.90531 0.92325 0.94021 0.95665 0.97248 0.98752 1.00000</meanLRs>
# <vulnerabilityFunction id="W1BVMETAL_Pre1945" dist="LN">
#     <meanLRs>0.00000 0.00330 0.01048 0.02049 0.03278 0.04698 0.06275 0.07978 0.09785 0.13589 0.17028 0.20155 0.22940 0.25517 0.27907 0.30247 0.34391 0.38291 0.41972 0.45433 0.48718 0.51850 0.54850 0.57720 0.60434 0.63063 0.65598 0.68003 0.70000</meanLRs>
# <vulnerabilityFunction id="W1BVMETAL_Post1945" dist="LN">
#     <meanLRs>0.00000 0.00095 0.00424 0.01013 0.01869 0.02995 0.04378 0.06004 0.07853 0.11057 0.13947 0.16577 0.18960 0.21164 0.23209 0.25216 0.28842 0.32254 0.35476 0.38504 0.41378 0.44119 0.46744 0.49255 0.51630 0.53930 0.56148 0.58253 0.60000</meanLRs>