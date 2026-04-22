# STTS-Reentry — Catalog-Scale Historical Characterization (sample)
Sample size: **3424** objects (of 3,623 targeted; rows emitted by pipeline after feature-extraction validation).
Periapsis-at-decay < 300 km (plausible natural decay): **2349**.
Periapsis-at-decay ≥ 300 km or unknown: **1075**.

## max P(failure) distribution

```
plausible-only: n=2349  mean=0.8931  median=0.9046  p25=0.9032  p75=0.9059  p90=0.9070  p95=0.9078  min=0.0444  max=0.9110
all objects  : n=3424  mean=0.8461  median=0.9039  p25=0.9011  p75=0.9055  p90=0.9067  p95=0.9074  min=0.0189  max=0.9110
78-object set: n=78  mean=0.8876  median=0.9047  p25=0.9032  p75=0.9060  p90=0.9072  p95=0.9077  min=0.2372  max=0.9101
```

Objects with max P < 0.5:  **27** of 2349 plausible (below tactical threshold)
Objects with max P < 0.25: **20** of 2349 plausible (bimodal-outlier signature)

## Lead-time distribution (days before decay_date at which P_forward first crossed threshold)

```
lead @P≥0.10  : n=2332  mean=720.3453  median=473.5214  p25=104.4469  p75=1170.3501  p90=1823.8646  p95=2114.8146  min=0.0019  max=3007.9693
lead @P≥0.25  : n=2329  mean=315.3414  median=191.5478  p25=76.5644  p75=349.5878  p90=714.6418  p95=1376.5167  min=0.0019  max=2688.1095
lead @P≥0.50  : n=2322  mean=163.8065  median=70.2697  p25=43.4500  p75=119.5880  p90=337.7761  p95=648.4958  min=0.0019  max=2625.1051
lead @P≥0.75  : n=2311  mean=85.3428  median=36.5263  p25=23.8340  p75=70.4294  p90=165.7875  p95=297.0149  min=0.0019  max=2336.1137

78-object reference:
  lead @P≥0.10: n=78  mean=478.1027  median=446.5228  p25=259.9375  p75=577.7292  p90=1117.1000  p95=1164.4625  min=16.4167  max=1230.0833
  lead @P≥0.25: n=76  mean=383.0859  median=308.6205  p25=96.8078  p75=472.0462  p90=1037.2500  p95=1129.3533  min=9.4859  max=1196.1199
  lead @P≥0.50: n=76  mean=288.8538  median=209.4970  p25=74.1476  p75=433.4373  p90=596.8690  p95=1077.3125  min=9.4859  max=1196.1199
  lead @P≥0.75: n=76  mean=153.6963  median=90.9978  p25=32.9664  p75=243.4011  p90=368.3514  p95=432.4373  min=3.8110  max=1058.4167
```
