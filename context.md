# Project Context

Updated: 2026-03-02

## One-paragraph summary

This project aims to continue a research direction started with Timothy Christensen at NYU, but in a form that is legible to the ML community. The core idea is to build a minimax robustness library for policy learning, expose it through a Hugging Face-compatible interface for adoption, and demonstrate its value in an agricultural planning benchmark where tail failures matter more than average-case performance.

## Why this exists

- Long-term goal: develop AI methods that help decision-makers in agriculture and support broader work on using AI to help feed the world.
- Near-term goal: produce a research contribution, a usable implementation, and a benchmark that makes the contribution easy to understand.
- Personal context: this is intended as a serious continuation of prior research work, not just a software side project.

## Research lineage

Primary local source:

- `C:\Users\adawg\Downloads\Minimax optimization.pdf`

What that PDF actually is:

- A minimax estimation sketch for regression with selective or strategic nonresponse.
- Nature chooses a response rule `q(X, Y)` from an uncertainty set `Q`.
- The learner chooses an estimator to minimize worst-case mean-squared prediction error.

Important implication:

- Christensen's note is not generic slice reweighting.
- The adversary in the note is a structured uncertainty set over observation or corruption mechanisms.
- Any ML extension should preserve that spirit if it is going to be described as a continuation of the research rather than merely inspired by it.

## Current drift and how to resolve it

Current drift:

- The original note is about robust estimation under selective observation.
- The earlier project draft drifted toward group-DRO-style slice weighting plus an agriculture simulation.

Resolution:

- Keep the core research object as an uncertainty set `Q`.
- Use slices only if they are a computational approximation to `Q`, not as the main research idea.
- Make the agriculture benchmark include both:
  - a true outcome process
  - an observation process that can be selective, missing, delayed, or corrupted
- Evaluate whether robust training improves downstream survival and tail outcomes when training data are biased or incomplete.

Clean research framing:

Minimax policy learning under selective observation for agricultural planning.

That is much closer to Christensen's note than "generic worst-slice training for farms."

## Product vs research contribution

Decision:

- Proceed with a Hugging Face implementation.

But:

- Hugging Face is the interface and distribution layer, not the core scientific contribution.

Correct framing:

- Research contribution: a minimax or robust objective tied to a defensible uncertainty set and downstream decision risk.
- Product layer: a Hugging Face-compatible adapter that makes the method easy to use.
- Benchmark: an agriculture simulation that shows why the method matters.

## Working thesis

Build a framework-agnostic minimax robustness library, expose it through Hugging Face for accessibility, and demonstrate its value in an agricultural survival benchmark where tail failures matter more than average-case performance.

## Preferred architecture

Repository structure should separate method from interface:

- `minimax_core/`
  - adversary definitions
  - uncertainty set definitions
  - robust objectives
  - metrics
- `minimax_hf/`
  - Hugging Face `Trainer` integration
  - configuration
  - examples
- `ag_sim/`
  - environment and scenario generation
  - crop and price processes
  - finance and bankruptcy logic
  - evaluation

This avoids making the whole project look like "just a Trainer wrapper."

## Agriculture benchmark role

The agriculture simulation is a benchmark for downstream robustness, not a claim of fully realistic agricultural forecasting.

The benchmark should answer:

"If two policies operate in the same uncertain agricultural environment, which one fails less often when conditions turn bad?"

It should not claim to answer:

"What will real farmers earn next year?"

## Benchmark design principles

- Compare Policy A vs Policy B under identical plausible futures.
- Use paired evaluation with the same random seeds and scenario draws.
- Prevent information leakage: no future weather or price information during decision time.
- Constrain actions by acres, budget, and credit limits.
- Keep the benchmark legible to non-ML audiences through survival metrics.

## Benchmark state, actions, and outputs

Suggested farm state:

- `cash`
- `debt`
- `credit_limit`
- `acres`
- `year`
- `alive`

Suggested policy inputs:

- soil features
- historical yields
- price history
- climate normals
- financial state

Suggested action space:

- crop choice
- input level

Examples:

- `corn_low_input`
- `soy_medium_input`

## Observation-process alignment with Christensen

If the benchmark is meant to align with the minimax note, the simulation should include selective observation.

Examples:

- yield observations missing more often in bad years
- market prices observed with delays or noise in stressed regions
- farmer-reported outcomes missing when losses are severe
- some actions or costs only partially observed in distressed cases

In that setup:

- the simulator generates full latent outcomes
- the learner trains on selectively observed or corrupted data
- evaluation happens on the full simulator

This creates a real bridge between the original minimax idea and the ML benchmark.

## Agriculture assumptions to keep explicit

Biology:

- Crop model answers "what grows," not "what sells."
- v0 should use DSSAT rather than a synthetic yield model.
- APSIM can be considered later as an optional alternative or comparison model if needed.

Markets:

- Farmers can initially be assumed able to sell output in liquid commodity markets.
- Market stress should be represented through price shocks, basis penalties, transport costs, storage costs, or quality discounts rather than unsold inventory.
- Demand shocks can initially be represented as price shocks.

Finance:

- Farms operate with working capital plus credit.
- Operating costs must be funded annually.
- Lenders impose solvency constraints.

Important simplifications to state explicitly:

- fixed acreage
- fixed debt structure
- constant interest rate
- no land appreciation in v0

## Price-model guidance

Price modeling needs to be more careful than the earlier rough note.

Preferred v0 options:

- empirical bootstrap from historical USDA or CME data
- scenario-based prices
- positive-valued stochastic process

Avoid:

- plain Gaussian prices if they can become negative

## Survival and failure rules

Primary metric:

- survival time, also called time-to-exit or time-to-default

Reason:

- easy to explain
- directly aligned with the "one bad year can end you" story
- hard to game if paired with average-profit sanity checks

Suggested bankruptcy rules:

- Liquidity failure:
  - `cash + remaining_credit < next_year_min_operating_cost`
- Debt-service failure:
  - `DSCR < 1` for two consecutive years

Where:

- `DSCR = net_income / debt_payment`

When the farm exits:

- `alive = False`
- stop the simulation

## Benchmark metrics

Primary:

- mean survival time
- median survival time
- 5th percentile survival time

Secondary:

- bankruptcy rate before horizon
- worst 5 percent terminal wealth or profit
- average profit as a sanity check

Optional output:

- Kaplan-Meier-style survival curve

## Method expectations

Expected result pattern:

- average profit should be similar between baseline and robust models
- robust model should improve survival metrics
- robust model should reduce bankruptcy rate
- robust model should improve tail outcomes

If robustness only improves by becoming uselessly conservative, the benchmark should expose that through lower average profit or low action diversity.

## Current strongest alignment path

Best current method direction:

1. Define a Christensen-consistent uncertainty set `Q1` over selective observation.
2. Implement a minimax objective in `minimax_core`.
3. Provide a thin Hugging Face adapter in `minimax_hf`.
4. Build an agriculture simulator that generates both full latent outcomes and selectively observed training data.
5. Compare robust vs baseline policies on full-information downstream outcomes.

This keeps the project grounded in Christensen's minimax framing while still making it usable for ML practitioners.

## Minimal research brief

### Problem statement

We want a training objective for ML models that is robust to biased or incomplete observations, especially when those biases are worst in the states that matter most for downstream decisions.

The Christensen-consistent version of that statement is:

- the world generates latent full outcomes
- the learner only sees selectively observed or corrupted outcomes
- the adversary ranges over a structured uncertainty set `Q`
- the learner minimizes worst-case predictive or decision loss over `Q`

### v0 learning problem

For consistency with Christensen's note, the first version of the package should solve a supervised learning problem under selective observation before expanding to more general policy learning.

Recommended v0 task:

- input `x`: observable pre-decision state
- target `y`: latent outcome or action-value label
- observed target `y_tilde`: selectively observed, masked, delayed, or corrupted version of `y`
- objective: learn a predictor from `(x, y_tilde)` that is robust to worst-case observation rules in `Q`

Then:

- the Hugging Face package exposes this as a robust training method
- the agriculture simulator provides the latent full data and the observed corrupted data
- downstream policy evaluation uses the latent simulator, not the corrupted labels

### Downstream agriculture interpretation

For agriculture, `x` can contain:

- soil features
- climate normals
- historical yield and price summaries
- financial state

The label can be one of two forms:

- predicted outcome for a fixed action
- predicted value of an action among a discrete action set

The second form is closer to policy learning, but the first form is simpler for v0 and closer to Christensen's original estimation setup.

Recommended sequencing:

1. Start with outcome prediction under selective observation.
2. Use predicted outcomes to rank discrete actions.
3. Only then generalize to direct policy optimization if needed.

### Core claim

The claim for v0 should be narrow and defensible:

Training with a minimax objective over plausible selective-observation mechanisms improves worst-case predictive quality and leads to better downstream survival outcomes than standard empirical risk minimization when the training data are selectively incomplete or corrupted.

This is a much safer claim than:

"Our package solves robust agriculture."

## v0 uncertainty set `Q1`

This is the only in-scope uncertainty set for `v0`, and it is the closest match to Christensen's note.

Form:

- nature chooses an observation probability `q(x, y)` or `q(x, a, y)`
- low-probability observations are more likely in difficult, low-performing, or distressed cases

Examples:

- bad-yield years are less likely to be fully observed
- financially distressed farms report less complete outcomes
- losses are more likely to be delayed or censored

Why this is Christensen-consistent:

- it preserves the original idea of adversarial response or observation mechanisms
- it makes the learner robust to non-ignorable missingness

v0 implementation approximation:

- discretize `q(x, y)` into groupwise observation probabilities `q_g`
- compute group priors from all examples, observed and unobserved
- estimate group losses from observed labels only
- optimize against the worst feasible `q_g` subject to:
  - `q_min <= q_g <= q_max`
  - weighted average observation rate matches the empirical observation rate

This keeps the package tied to selective observation rather than generic worst-group training.

How slices or groups fit:

- slices can approximate classes of `q`
- for example: `drought`, `price_crash`, `low_cash`, `high_debt`
- the slices are not the theory; they are a discretization of the uncertainty set

Future extensions such as stress-regime weighting or bounded corruption can be considered later, but they are explicitly out of scope for `v0`.

## Exact v0 minimax objective

For groups `g = 1, ..., G`, let:

- `pi_g` be the empirical group prior based on all examples
- `L_g` be the mean observed loss in group `g`
- `q_g` be the adversarial groupwise observation probability

The `v0` robust objective is:

- minimize over model parameters
- maximize over feasible `q_g`
- objective value `sum_g pi_g * L_g / q_g`

subject to:

- `q_min <= q_g <= q_max` for each group
- `sum_g pi_g * q_g = q_bar`, where `q_bar` is the empirical overall observation rate

Interpretation:

- high-loss groups are assigned lower feasible observation probabilities by the adversary
- that forces the learner to be robust to plausible selective under-observation of bad states
- if all groups have equal loss, the adversary has no reason to tilt probabilities

## Relation between training objective and downstream metric

The link to downstream survival should be stated carefully.

Not valid:

- "lower training loss automatically means better farm survival"

Valid:

- if selective observation disproportionately hides bad states, standard ERM can learn systematically optimistic or miscalibrated predictions
- those errors matter most in distressed states, which are exactly the states that drive bankruptcy and tail outcomes
- a minimax objective that is robust to those observation mechanisms should improve predictions in the tail
- better tail predictions should improve downstream action choice and therefore improve survival-related metrics

This is a hypothesis to test, not something to assume without evidence.

## Package design implication

The package API should not be built around slices as the main abstraction.

The package API should be built around:

- uncertainty model
- adversary
- objective

Slices can exist as one implementation mode inside the uncertainty model, but for `v0` they should be interpreted as groups approximating selective-observation regimes.

Suggested abstractions:

- `UncertaintySet`
- `Adversary`
- `RobustObjective`
- `PerExampleLossAdapter`

Then `minimax_hf` becomes a thin adapter that feeds per-example losses and metadata into the core engine.

## Legitimacy testing plan

The legitimacy of the package should be tested in four steps:

1. Mathematical correctness
   - confirm the implemented objective matches the written minimax problem
   - use toy cases where the solution can be brute-forced or solved analytically
2. Implementation correctness
   - unit-test adversary updates, loss aggregation, reduction to ERM, and numerical stability
3. Synthetic empirical validation
   - generate data with known selective-observation rules
   - compare ERM vs minimax on worst-case prediction error
4. Downstream benchmark validation
   - use the agriculture simulator to test whether better robustness to selective observation improves survival and tail outcomes

This order matters. DSSAT is part of the downstream benchmark, not the proof that the optimizer itself is legitimate.

## Major open questions

These are the main unresolved issues:

- What exact label form should v0 use: direct outcome prediction or action-value prediction?
- What are the labels in the first benchmark: simulated latent outcomes, simulated optimal actions, or something else?
- Are the current `Q1` constraints (`q_min`, `q_max`, and empirical mean observation rate) sufficient, or do they need refinement?
- How does the training objective connect mathematically to downstream survival or solvency?
- Which parts of the benchmark are synthetic and which are calibrated to real data?
- What group definitions best approximate selective-observation regimes without collapsing into arbitrary slices?
- What is the simplest benchmark that still makes the research point clearly?

## Existing Christensen software

As of January 2026, Timothy Christensen is Professor of Economics at Yale University. His public software footprint appears to include:

- `ValidMLInference` (Python)
- `MLBC` (R)
- `npiv` (R)

Public GitHub repositories also include paper-specific replication code such as:

- `MC`
- `NPIV`
- `NPSDFD`

Important negative result:

- no public Christensen package was found for minimax training
- no public Christensen package was found for Hugging Face integration
- no public Christensen package was found for agricultural decision simulation

So this project is additive rather than duplicative.

## External references already checked

- Timothy Christensen homepage: `https://tmchristensen.com/`
- Timothy Christensen CV: `https://tmchristensen.com/cv.pdf`
- `ValidMLInference`: `https://pypi.org/project/ValidMLInference/`
- `MLBC`: `https://cran.r-project.org/web/packages/MLBC/index.html`
- `npiv`: `https://cran.r-project.org/package=npiv`
- "Externally Valid Policy Choice": `https://arxiv.org/abs/2205.05561`
- "Optimal Decision Rules when Payoffs are Partially Identified": `https://arxiv.org/abs/2204.11748`
- "Counterfactual Sensitivity and Robustness": `https://doi.org/10.3982/ECTA17232`

## Recommended next step for the next model session

The compact research brief and the exact `v0` objective are now written above.

Do not start with the Hugging Face wrapper.

Start with:

- `minimax_core`
- the `Q1` selective-observation objective
- the first synthetic validation benchmark

After that:

- add `minimax_hf`
- build the smallest benchmark that demonstrates the point

## One-sentence mission

Build a minimax robustness library for ML policy learning, make it usable through Hugging Face, and show through an agricultural survival benchmark that robustness to biased or adverse data matters for real downstream outcomes.
