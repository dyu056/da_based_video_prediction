# Dual-Use Plan: Final Project + Paper

## Goal

Use one core project to satisfy both:

1. the data assimilation final project requirements;
2. a first paper-style manuscript or arXiv-ready draft.

The key principle is:

- optimize first for a complete and correct DA project;
- design the experiments so the same results are strong enough to support a paper draft.

## Core Claim

The shared claim for both deliverables is:

Latent-space data assimilation can improve the quality-compute tradeoff of sparse-conditioned sequential video prediction.

For the class project, this is a DA application in machine learning.

For the paper, this becomes a sharper framing:

DA as adaptive latent filtering for efficient sparse-conditioned video prediction.

The paper story should now be built around three roles:

1. a strong literature reference baseline (`SimVP`);
2. a lightweight latent sequential model that DA can actually update;
3. a DA-triggered correction policy that decides when refinement is worth paying for.

## What One Deliverable Can Share

The following components should be identical between the class report and the paper draft:

1. Problem setting and motivation
2. Literature review
3. Mathematical formulation
4. Method section
5. Dataset description
6. Experimental protocol
7. Main result tables
8. Qualitative figures
9. Discussion of strengths and limitations

This means the class report should already be written in a paper-like structure.

## Non-Negotiable Requirements For The Class Project

To safely satisfy the course rubric, the final report must include:

1. a complete DA workflow:
   - model
   - observation operator
   - forecast
   - analysis/update
   - evaluation
2. a clear mathematical formulation of the DA method
3. implementation details and reproducible code
4. visualizations and numerical experiments
5. comparison with baseline methods
6. all required DA metrics:
   - RMSE
   - PCC
   - entropy
   - relative entropy
   - mutual information

If any of these are missing, the paper may still look fine, but the class submission becomes risky.

## Extra Requirements To Make It Paper-Ready

To turn the same project into a paper draft, add:

1. at least one recognized baseline from the video prediction literature
2. one clear novelty statement
3. ablations
4. quality-versus-compute curves
5. stronger qualitative comparisons
6. a sharper positioning against recent DA and video-generation literature

## Recommended Experimental Stack

### Backbone Split

Use two different model roles rather than forcing one model to do everything.

1. Reference baseline
   Use a recognized predictor as the main comparison target whenever possible.

2. DA host model
   Use a lightweight latent sequential predictor as the actual process model inside the DA loop.

Recommended options:

1. PredRNN++
2. TAU
3. SimVP

Current recommendation:

1. keep `SimVP` as the literature-aware reference baseline;
2. use the open-loop latent sequential model as the DA host.

### DA Method

Start with one main DA method:

1. ETKF or EnKF in latent space

Optional later comparisons:

1. EnKF vs ETKF
2. latent-space DA vs naive fusion
3. latent-space DA vs always-refine correction

### Observation Regimes

Minimum paper-safe set:

1. prefix-only prediction
2. periodic keyframe assimilation
3. one harder sparse regime:
   - masked observations, or
   - low-resolution observations

### Datasets

Minimum:

1. Moving MNIST

Better:

1. Moving MNIST
2. KTH as an extension if the pipeline stabilizes early

## Minimal Publishable Contribution

The safest minimal contribution is not:

- "we built a video predictor."

It is:

- "we show that a latent-space DA update can reduce expensive correction frequency while preserving prediction quality under sparse observations."

That is narrow enough to finish, but still novel enough to justify a short paper or workshop submission if the experiments are solid.

## Result Package We Need

To make the project dual-use, we should aim for the following result package.

### Main table

Compare:

1. open-loop predictor
2. always-refine predictor
3. DA-assisted selective refinement

Report:

1. RMSE
2. PCC
3. entropy
4. relative entropy
5. mutual information
6. runtime
7. number of expensive refinement calls

### Ablation table

At minimum:

1. without DA
2. with DA every step
3. with DA selective triggering
4. different ensemble sizes
5. different observation sparsity levels

### Figures

We should prepare:

1. method diagram
2. sample rollout visualization
3. quality-versus-compute curve
4. uncertainty or innovation diagnostic plot

## Writing Strategy

The safest writing order is:

1. write the report as a journal-style DA paper
2. ensure every class requirement is explicitly satisfied
3. later trim or reposition it for a workshop or arXiv draft

Recommended section order:

1. Introduction
2. Related Work / Background
3. Problem Formulation
4. Method
5. Experimental Setup
6. Results
7. Analysis and Discussion
8. Conclusion

This already matches the class report structure closely.

## Revised NeurIPS-Style Framing

For a NeurIPS-style draft, avoid presenting the project as only “video prediction on Moving MNIST.”

The better framing is:

1. sequential video prediction as a latent filtering problem;
2. DA as adaptive correction under sparse observations;
3. efficient Bayesian decision-making through selective refinement.

That keeps the work closer to efficient sequential generation and world-model style reasoning, even though the experiments start on a classic video-prediction benchmark.

## Immediate Milestones

### Milestone 1

Recognized baseline integration is underway through `SimVP`; the next step is to turn that into a stable report-quality reference run.

### Milestone 2

Implement latent-space ETKF or EnKF on top of the DA host predictor.

### Milestone 3

Add all required DA metrics:

1. entropy
2. relative entropy
3. mutual information

### Milestone 4

Run the first full comparison in one sparse-observation regime with:

1. `SimVP` reference baseline
2. open-loop DA host
3. always-assimilate DA host
4. selective-assimilation DA host

### Milestone 5

Produce report-quality tables and figures.

## Time-Safest Scope

If we want the safest path that still supports both deliverables, the scope should be:

1. one main dataset: Moving MNIST
2. one recognized reference baseline: `SimVP`
3. one DA host predictor
4. one DA method: ETKF or EnKF
5. two observation settings
6. all five DA metrics
7. one runtime comparison
8. one ablation on sparsity or ensemble size

This is enough for a strong final project and a realistic first paper draft.

## Decision Rule

When choosing what to work on next, prioritize tasks in this order:

1. anything required by the class rubric
2. anything needed for a credible baseline comparison
3. anything that sharpens the publishable claim
4. optional extensions

## Bottom Line

Yes, one project can safely serve both purposes.

The right strategy is:

- build a complete DA-first class project;
- make the experiments baseline-aware and paper-structured;
- avoid extra side projects that do not strengthen the main claim.
