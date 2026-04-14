# Project Blueprint: Data-Assimilation-Inspired Acceleration for Sparse-Conditioned Video Prediction

## Working Title

Data assimilation for efficient sparse-conditioned video prediction in latent space

## 1. Project Goal

This project studies whether the theoretical framework of data assimilation (DA) can improve the quality-compute tradeoff of sequential video modeling. Instead of building a full text-to-video system, we focus on a cleaner and more feasible setting:

- predict future video frames from a short observed prefix;
- optionally receive sparse observations during rollout, such as keyframes, masked pixels, or low-resolution frames;
- use DA updates in latent space to correct the prediction and reduce the need for expensive refinement.

The central question is:

Can latent-space DA provide a better sequential estimate of future video states so that a generator needs fewer expensive correction steps while maintaining prediction quality?

The revised operational framing is:

- `SimVP` is the recognized video-prediction reference baseline;
- a separate lightweight latent sequential predictor is the DA host model;
- DA acts as an adaptive correction layer on top of that host model.

## 2. Main Hypothesis

The project hypothesis is that DA can speed up sequential video generation through three mechanisms:

1. Warm start from the previous analysis state instead of re-generating from an uninformed prior.
2. Assimilate sparse observations to correct only the uncertain directions of the latent state.
3. Use uncertainty indicators to trigger expensive refinement only when needed.

In short, DA should reduce both prediction error and compute cost in sequential video prediction.

## 3. Mathematical Formulation

Let `x_t` denote the true video frame at time `t`, and let `z_t` be its latent representation.

- Encoder: `z_t = E(x_t)`
- Decoder: `x_t = D(z_t)`
- Forecast model: `z_t^f = M_theta(z_{t-1}^a) + eta_t`
- Observation model: `y_t = H(x_t) + epsilon_t`

Here:

- `z_t^f` is the forecast latent state;
- `z_t^a` is the analysis latent state after assimilation;
- `H` is a sparse observation operator, for example:
  - full frame at a few time steps;
  - masked pixels or patches;
  - downsampled frame;
  - noisy partial measurement.

The assimilation update will use an ensemble Kalman style rule in latent space:

`z_t^a = z_t^f + K_t (y_t - H(D(z_t^f)))`

with Kalman gain

`K_t = P_t^f H_t^T (H_t P_t^f H_t^T + R_t)^(-1)`

where:

- `P_t^f` is the forecast covariance estimated from an ensemble;
- `R_t` is the observation noise covariance.

This gives a natural DA interpretation:

- forecast step = cheap latent propagation;
- analysis step = observation-informed correction;
- optional smoother-style correction can be considered later if time allows.

## 4. Proposed Method

### 4.1 Revised Framework

The revised framework has three layers:

1. Reference layer
   A literature-aware predictor baseline, currently `SimVP`, gives a strong non-DA comparison point.

2. DA host layer
   A lightweight latent sequential predictor defines the forecast model that DA will update.

3. Decision layer
   A DA trigger uses innovation, spread, or entropy to decide whether to apply assimilation at a given step.

This separation matters because it keeps the paper comparison fair while preserving a DA-centered model that we can control.

### 4.2 State Representation

Use a low-dimensional latent state rather than pixel-space DA. This follows reduced-order DA logic and keeps the state dimension manageable.

### 4.3 Forecast Stage

Train a lightweight sequential predictor in latent space as the DA host model. Candidate options:

- simple convolutional autoencoder + GRU/ConvGRU predictor;
- small deterministic latent predictor;
- lightweight stochastic latent predictor if training remains stable.

The DA host model does not need to be the strongest published predictor. It needs to be sequential, latent-space, and cheap enough that DA-triggered correction is meaningful.

### 4.4 Analysis Stage

At each future time step:

1. Propagate an ensemble of latent states forward.
2. Decode the forecast into observation space.
3. Compare with sparse observations.
4. Apply ETKF or EnKF update in latent space.
5. Decode the updated latent state into the predicted frame.

### 4.5 Speed-Up Mechanism

The project will test a selective-compute strategy:

- always do cheap latent forecasting;
- only apply expensive refinement when innovation, forecast spread, or entropy exceeds a threshold.

This creates a DA-assisted acceleration mechanism:

- low uncertainty -> skip refinement;
- high uncertainty -> assimilate observations and refine.

### 4.6 Current Implementation Mapping

At the current project stage:

- `SimVP` is integrated and serves as the main recognized baseline;
- the open-loop latent sequential model is the natural DA host model;
- the next missing module is the latent ETKF/EnKF update plus the uncertainty-triggered decision rule.

## 5. Experimental Scope

### 5.1 Primary Dataset

Moving MNIST

Reasons:

- easy to train on local hardware;
- controllable motion dynamics;
- standard benchmark for video prediction;
- allows clear quantitative comparisons.

### 5.2 Optional Secondary Dataset

KTH human action dataset, only if the primary pipeline is stable early enough.

### 5.3 Observation Scenarios

We will compare several sparse-observation settings:

1. Prefix-only prediction: observe the first `T_obs` frames, then predict future frames.
2. Keyframe assimilation: observe one full frame every `k` steps.
3. Masked observation assimilation: observe only patches or partial pixels.
4. Low-resolution assimilation: observe downsampled frames.
5. Noisy observation assimilation: add Gaussian noise to observations.

## 6. Baselines

The baseline set should be simple, defensible, and clearly separated by role:

1. `SimVP` reference baseline
   A recognized literature-style baseline for plain video prediction.

2. Open-loop DA host predictor
   The same sequential latent host model, but without DA after the observed prefix.

3. Always-assimilate or always-refine host model
   Apply correction at every future step to estimate the upper cost of dense DA usage.

4. DA-assisted selective refinement
   Forecast cheaply and assimilate only when uncertainty is high.

5. Oracle dense-observation variant
   Optional upper bound when the full future observation is available in evaluation only.

## 7. Evaluation Metrics

To align with the course requirements, the report will include:

1. RMSE
   Frame-wise root mean square error between prediction and ground truth.

2. PCC
   Pattern correlation coefficient between predicted and true frames.

3. Entropy
   Uncertainty of the ensemble forecast or analysis distribution.

4. Relative entropy
   Divergence between forecast and analysis distributions, or between model variants.

5. Mutual information
   Information gained from observations about the latent state or predicted frame.

In addition, since this project is about acceleration, we will report:

- wall-clock runtime per rollout;
- number of expensive refinement calls;
- quality-compute tradeoff curves.

## 8. Success Criteria

The framework is successful if we can show at least one of the following convincingly:

1. DA-selective refinement matches or improves the DA host model at lower compute.
2. DA-selective refinement approaches the quality of always-assimilate correction with fewer refinement calls.
3. DA-selective refinement is more robust than open-loop prediction under sparse observations.

## 9. Expected Contributions

The project aims to contribute:

1. A clear DA formulation for sparse-conditioned video prediction in latent space.
2. An empirical demonstration that DA-style analysis updates can reduce compute without large quality loss.
3. A quantitative comparison between open-loop prediction, always-refine prediction, and uncertainty-triggered DA correction.
4. A discussion of when DA ideas help and when they fail in video generation settings.

## 10. Risks and Mitigation

### Risk 1: Training instability

Mitigation:

- begin with a deterministic latent predictor;
- keep architecture small;
- use Moving MNIST first.

### Risk 2: DA update is numerically weak in latent space

Mitigation:

- start with EnKF or ETKF in a low-dimensional latent space;
- use covariance inflation or localization if needed;
- compare several latent dimensions.

### Risk 3: No visible speedup

Mitigation:

- define speed using both wall-clock time and number of expensive correction calls;
- include selective refinement thresholds;
- compare accuracy under equal compute budgets.

### Risk 4: Course relevance becomes unclear

Mitigation:

- keep the Bayesian filtering and Kalman update central;
- report the required DA metrics;
- emphasize reduced-order DA motivation throughout the report.

## 11. Eight-Week Execution Plan

### Week 1

- finalize project statement;
- set up dataset and preprocessing;
- stabilize the recognized baseline path.

### Week 2

- replicate the `SimVP` baseline on Moving MNIST;
- establish RMSE, PCC, and runtime baseline.

### Week 3

- train or stabilize the DA host predictor;
- implement latent-space ensemble generation.

### Week 4

- implement EnKF or ETKF update with synthetic sparse observations;
- test keyframe and masked-observation assimilation.

### Week 5

- add uncertainty-triggered selective refinement;
- tune ensemble size, latent dimension, and observation noise.

### Week 6

- compute entropy, relative entropy, and mutual information;
- measure quality versus compute tradeoff.

### Week 7

- prepare figures and ablation results;
- optional extension to KTH or another more realistic dataset.

### Week 8

- write final analysis;
- stabilize report narrative and presentation slides;
- prepare presentation and final report revision.

## 12. Minimum Viable Project

If time becomes tight, the minimum viable deliverable is:

- one dataset: Moving MNIST;
- one recognized reference baseline: `SimVP`;
- one DA host predictor;
- one DA method: ETKF or EnKF;
- two observation settings: prefix-only and periodic keyframes;
- metrics: RMSE, PCC, entropy, relative entropy, mutual information, and runtime.

This is still a complete DA project with modeling, assimilation, evaluation, and analysis.

## 13. Stretch Goals

If the core pipeline works early, possible extensions are:

- compare EnKF vs ETKF;
- add adaptive thresholding based on entropy;
- test smoothing rather than filtering only;
- move from Moving MNIST to KTH;
- compare latent-space DA against pixel-space correction.

## 14. Initial Reading List

The following papers are the most relevant starting references:

1. Buizza et al., "Data Learning: Integrating Data Assimilation and Machine Learning," Journal of Computational Science, 2022.
2. Manshausen et al., "Generative Data Assimilation of Sparse Weather Station Observations at Kilometer Scales," arXiv:2406.16947.
3. Zheng et al., "Ensemble Kalman Diffusion Guidance: A Derivative-free Method for Inverse Problems," arXiv:2409.20175.
4. Chen et al., "FlowDAS: A Stochastic Interpolant-based Framework for Data Assimilation," arXiv:2501.16642.
5. Hodyss and Morzfeld, "Using Diffusion Models to do Data Assimilation," arXiv:2506.02249.
6. Scholz and Turner, "Warm Starts Accelerate Conditional Diffusion," arXiv:2507.09212.
7. Huang et al., "Accelerated Sequential Flow Matching: A Bayesian Filtering Perspective," arXiv:2602.05319.
8. Voleti et al., "MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation," arXiv:2205.09853.
9. An et al., "Latent-Shift: Latent Diffusion with Temporal Shift for Efficient Text-to-Video Generation," arXiv:2304.08477.

## 15. One-Sentence Project Summary

This project will test whether latent-space data assimilation can improve the quality-compute tradeoff of sparse-conditioned video prediction by combining cheap sequential forecasting with observation-driven correction and uncertainty-aware selective refinement.
