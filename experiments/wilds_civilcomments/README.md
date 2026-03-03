# WILDS CivilComments Experiments

This directory contains the CivilComments-specific experiment layer that sits on top of the core selective-observation package.

Current scope:

- overlapping identity groups are represented as multi-membership `group_id` values
- explicit MNAR masking utilities are available for the `train` split
- configuration files are tracked alongside the experiment

The intended experiment tracks are:

- `vanilla`: original WILDS supervision
- `explicit_mnar`: shared synthetic MNAR masking on the training split
- `latent_mnar`: unchanged training data with an internal latent-missingness adversary

Install dependencies with:

```bash
pip install ".[wilds]"
```
