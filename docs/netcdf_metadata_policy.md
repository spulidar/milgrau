# NetCDF metadata policy

MILGRAU NetCDF products should be readable, self-describing, and technically unambiguous without requiring a reader to inspect the source code first. This document defines that project policy. It does not claim conformance, alignment, or certification against any external metadata convention.

## Goals

- Give coordinates unambiguous physical meaning, including units, direction, reference level, and time representation where applicable.
- Give scientific variables clear names, `long_name` descriptions, and physical units when those units are meaningful.
- Do not invent absolute SI units for source-dependent or instrument-native quantities; document their unit status explicitly instead.
- Use machine-readable flag values and meanings for categorical diagnostics and quality states.
- Preserve missing/unsupported values honestly; do not fill data merely to make a product look complete.
- Record scientific-method identity, uncertainty scope, processing provenance, source identity, and configuration context needed to interpret the product.
- Use externally standardized names or attributes only when their semantics are verified to match the MILGRAU quantity exactly. Do not invent or approximate a standardized name for convenience.
- Keep instrument- and method-specific metadata explicit when generic metadata cannot express the required scientific meaning.

## External conventions and tools

External metadata conventions, controlled vocabularies, and validation tools may be consulted as references or diagnostics when they improve clarity or interoperability. Their use does not by itself make a MILGRAU product conformant with that convention, and MILGRAU does not currently publish a formal conformance claim.

A global `Conventions` declaration should only be introduced in the future if the project deliberately chooses a specific convention/version and accepts the corresponding maintenance and validation obligations.

Automated metadata checkers may be used during development as linting aids. Their warnings must be interpreted in the context of atmospheric lidar semantics; passing such a checker is neither required by this policy nor evidence that retrieval physics is scientifically correct.

## Scope

This policy concerns the representation and description of scientific products. It does not validate Klett-Fernald-Sasano physics, lidar-ratio assumptions, overlap characterization, Rayleigh-reference quality, detector corrections, cloud screening, or uncertainty-model correctness. Those remain separate scientific validation responsibilities.
