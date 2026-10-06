# SemRepair External Evidence Ledger

This ledger separates self-authored public evidence from independent external validation.

| Evidence class | Count |
| --- | ---: |
| independent third-party SemRepair reproductions | **0** |
| substantive upstream maintainer reviews of the SemRepair mechanism | **0** |
| upstream-retained SemRepair-derived regression tests | **0** |
| upstream-retained SemRepair-derived production fixes | **0** |
| external citations / technical references to the SemRepair mechanism | **0** |
| independent real-robot validations | **0** |
| externally corroborated underlying defect families | **1** |

## Counting rules

**Public != independent.** A public workflow or fork PR authored by this account does not count as external validation.

**Submitted != adopted.** An open upstream PR does not count as retained adoption.

**Reviewed != retained.** Technical feedback and merged code are separate evidence classes.

**Underlying defect corroboration != SemRepair validation.** The LeRobot issue and community PR show that the underlying mapping problem is real; they do not validate SemRepair's identifiability framework.

## Current public self-authored evidence

- SemRepair 0.3.0rc1 public prerelease;
- paired ManiSkill semantic-interaction evidence on two frozen corpora;
- ManiSkill execution-domain negative / variable replay evidence;
- LeRobot source-bound evidence-collision witness;
- executable identifiability certificate;
- claim-boundary and fail-closed promotion rules.

## Promotion events that would change the counters

A counter changes only when evidence originates outside this account and remains attributable, for example:

1. a third party posts an exact reproduction or falsification;
2. an upstream maintainer substantively reviews the SemRepair mechanism rather than only the underlying bug;
3. an external repository retains a regression or production change explicitly derived from the SemRepair evidence;
4. an external paper/report/benchmark cites this mechanism.

Until then, external-adoption claims remain zero.
