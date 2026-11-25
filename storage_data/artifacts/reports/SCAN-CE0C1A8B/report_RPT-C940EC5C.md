========================================================================
          ADVERSCAN SECURITY ASSESSMENT REPORT           
========================================================================
  Report ID   : RPT-C940EC5C
  Scan ID     : SCAN-FGSM-210347
  Timestamp   : 2026-11-23 14:49:13
  Risk Level  : LOW
  Vuln. Score : 0.28
========================================================================

1. EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Scan ID            : SCAN-FGSM-210347
  Risk Level         : LOW
  Vulnerability Score: 0.28
  Baseline Accuracy  : 80.00%
  Attacks Evaluated  : fgsm
  XAI Enabled        : True
  Hardening Applied  : True
  Re-Test Conducted  : True
  Pipeline Status    : SUCCESS

2. MODEL INFORMATION
------------------------------------------------------------------------
  - model_name: GTSRB Vision Transformer
  - model_id: custom-model
  - domain: image
  - framework: HUGGINGFACE
  - model_architecture: ViTForImageClassification
  - total_parameters: 0
  - trainable_parameters: 0
  - input_shape: [3, 224, 224]
  - num_classes: 10

3. DATASET / EVALUATION CONFIGURATION
------------------------------------------------------------------------
  - dataset_name: bazyl/GTSRB
  - dataset_type: Test Dataset
  - total_samples: 5
  - domain: image
  - batch_size: 16
  - data_split: test

4. BASELINE PERFORMANCE
------------------------------------------------------------------------
  - accuracy            : 80.00%
  - f1_score            : 80.00%
  - loss                : Not Measured
  - confidence          : 99.87%
  - throughput_samples_per_sec: 9.15

5. ADVERSARIAL ATTACK RESULTS
------------------------------------------------------------------------
| Attack Vector | Parameters | Exec Time | Baseline Acc | Adv Acc | Acc Drop | Attack Success Rate (ASR) |
|---|---|---|---|---|---|---|
| FGSM          | targeted=False | 0.55s     | 80.00%       | 80.00%  | 0.00%    | **0.00%** |

  ▶ How Attacks Were Performed:
    • FGSM: Configured with [targeted=False].

6. VULNERABILITY ASSESSMENT
------------------------------------------------------------------------
  ▶ Vector: FGSM
    - Attack Success Rate (ASR) : 0.00%
    - Vulnerability Score       : 0.28
    - Risk Level                : LOW
    - Clean vs Adversarial Acc  : 80.00% ➔ 80.00% (Drop: 0.00%)
    - Perturbation Magnitude    : linf_mean=5.00%, l2_mean=19.40, l0_mean=100.00%, default_epsilon=5.00%, evaluated_key=None, timestamp=None, metadata={'linf': 0.050000011920928955, 'linf_mean': 0.050000011920928955, 'linf_max': 0.050000011920928955, 'linf_min': 0.050000011920928955, 'l2': 19.39772605895996, 'l2_mean': 19.39772605895996, 'l2_max': 19.39772605895996, 'l2_min': 19.39772605895996, 'l0': 150528.0, 'l0_mean_count': 150528.0, 'l0_mean_pct': 100.0, 'l0_pct': 100.0, 'mean_abs_error': 0.05000000447034836, 'mae_mean': 0.05000000447034836}

7. VULNERABILITY SCORE & RISK LEVEL
------------------------------------------------------------------------
  Overall Vulnerability Score : 0.28
  Risk Level                  : LOW

8. MITRE ATLAS MAPPING
------------------------------------------------------------------------
  ▶ FGSM
    tactic      : AML.TA0000 — ML Attack Staging
    technique   : AML.T0043 — Craft Adversarial Data (FGSM)
    mitigation  : AML.M0003 — Adversarial Input Detection + AML.M0002 — Model Hardening

9. XAI FINDINGS
------------------------------------------------------------------------
  ▶ Technique: shap
    - Explainer Technique : Shap
    - Execution Status    : ✅ SUCCESS
    - Attack Examined     : FGSM
    - Model Modality      : Image
    - Target Class        : Class 16
    - Explanation Coverage: 1 of 1 samples successfully explained

    • Prediction Analysis:
      | Clean Pred | Clean Conf | Adv Pred | Adv Conf | Pred Changed | Attack Failure |
      |---|---|---|---|---|---|
      | 16         | 99.90%     | 16       | 98.17%   | No           | No             |

    • Attribution Analysis:
      - Attribution Drift (L1)    : 2.0000
      - Salience Distortion       : 1.0102
      - Top Salient Features      : pixel_(0,47) (0.8415), pixel_(0,46) (0.8415), pixel_(0,45) (0.8415), pixel_(0,44) (0.8415), pixel_(0,43) (0.8415)

    • Root Cause Explanation:
      Attack 'fgsm' failed to flip model prediction (remained '16'). The decision boundary margin was sufficient to absorb the perturbation.

    • Evidence-Based Defense Recommendations:
      - [Training] Adversarial Defense
        Finding   : 3 features hold 80% of attribution and carry 80% of the perturbation
        Confidence: High (80% Attribution Share)
        Evidence  : {'top3_attribution_share': 0.8, 'perturbation_alignment': 0.4971}
        Rationale : Top 3 spatial regions capture 80% of adversarial attribution, indicating high vulnerability concentration in localized receptive fields.
      - [Preprocessing] Adversarial Defense
        Finding   : Perturbation exhibits high-frequency spatial noise across pixel dimensions
        Confidence: Medium (Spatial Noise Characteristic)
        Evidence  : {'modality': 'image', 'attack': 'fgsm', 'spatial_frequency': 'high'}
        Rationale : Micro-perturbations target high-frequency pixel dimensions to evade human visual detection.

10. HARDENING
------------------------------------------------------------------------
  - selected_defense    : None (Already Robust)
  - status              : not_needed
  - improvement_sufficient: False
  - fallback_used       : False

11. RE-TEST RESULTS
------------------------------------------------------------------------
  - Hardened Model    : GTSRB Vision Transformer
  - Dataset / Samples : N/A (N/A samples)
  - Defense Mechanism : None (Already Robust)
  - Re-Test Status    : ❌ FAILED — Vulnerabilities Persist

12. BEFORE VS AFTER COMPARISON
------------------------------------------------------------------------
| Attack Vector | Before Defense (Adv Acc) | After Defense (Hardened Acc) | Robustness Gain | Before Risk | After Risk | Defense Status |
|---|---|---|---|---|---|---|
| FGSM          | 80.00%                   | 80.00%                       | +0.00%          | UNKNOWN     | UNKNOWN    | **❌ FAILED** |

  ▶ Defense Verification & Retest Summary:
    • Verdict: ⚠️ PARTIAL DEFENSE — Some attack vectors showed persistent vulnerability.

13. EXECUTION PERFORMANCE
------------------------------------------------------------------------
  Run Label    : AdverScan Interactive Assessment
  Started At   : 2026-11-23 14:49:13
  Total Time   : 1.95s
  Overall      : SUCCESS

  MODULE                         STATUS          TIME
  ········································································
  M1 MODEL INGESTION             ✅ SUCCESS      0.05s
                                    └ model_name: GTSRB Vision Transformer
                                    └ framework: HUGGINGFACE
  M2 BASELINE EVALUATION         ✅ SUCCESS      0.10s
                                    └ accuracy: 80.00%
  M3 ATTACK EXECUTION (FGSM)     ✅ SUCCESS      0.55s
                                    └ attack: fgsm
                                    └ samples: 5.00
  M5 VULNERABILITY SCORING       ✅ SUCCESS      0.05s
                                    └ vulnerability_score: 28.00%
                                    └ risk_level: LOW
  M6 EXPLAINABILITY (XAI)        ✅ SUCCESS      0.20s
                                    └ technique: shap
  M7 MODEL HARDENING             ❓ not_needed   0.50s
                                    └ selected_defense: None (Already Robust)
  M8 RETEST & VALIDATION         ❓ not_needed   0.50s
                                    └ defense_effectiveness: Not Needed (Model Already Robust)

14. RECOMMENDATIONS
------------------------------------------------------------------------
  [01] Model exhibits strong baseline robustness against tested attack suites. Maintain continuous monitoring and conduct periodic adversarial re-testing.
  [02] XAI attribution maps are available. Review highlighted input regions disproportionately targeted by adversarial perturbations to guide robustness patches.
  [03] Adversarial defense was applied. Validate post-hardening accuracy retention and conduct periodic re-tests to ensure defense durability.
  [04] Re-test indicates persistent vulnerability for vector 'fgsm'. Increase adversarial training epochs or broaden epsilon schedules.

15. FINAL SECURITY SUMMARY
------------------------------------------------------------------------
  - risk_level              : LOW
  - vulnerability_score     : 28.00%
  - baseline_accuracy       : 80.00%
  - mean_adversarial_accuracy: 80.00%
  - hardening_applied       : True
  - retest_conducted        : True
  - total_recommendations   : 4
  - primary_recommendation  : Model exhibits strong baseline robustness against tested attack suites. Maintain continuous monitoring and conduct periodic adversarial re-testing.

========================================================================
  Generated by AdverScan — 2026-11-23 14:49:13
========================================================================