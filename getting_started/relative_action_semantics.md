# Relative action semantics: training representation vs inference output

GR00T supports learning selected action groups in a relative representation while
still returning controller-facing **absolute** actions at inference.

This distinction has caused repeated confusion (see issue #490), so it is useful
to treat relative actions as a two-stage contract rather than as a promise about
the final output format.

## Training

Relative conversion is enabled only when both conditions hold:

1. the global processor/model flag `use_relative_action=True`;
2. the action group's `ActionConfig.rep` is `RELATIVE`.

For NON_EEF joint actions, the processor uses the last state timestep as the
reference and learns

```text
relative_action = absolute_target - reference_state
```

before normalization.

## Inference

The model predicts in the learned/normalized representation. The processor then:

1. denormalizes the predicted relative action;
2. uses the current supplied reference state;
3. converts the chunk back to absolute joint targets.

Therefore the public policy output is intentionally absolute even when the model
was trained on relative actions.

```text
model-space relative prediction
        ↓
denormalize
        ↓
relative + reference_state
        ↓
absolute controller-facing action
```

The regression test
`tests/gr00t/data/state_action/test_relative_action_contract.py` freezes this
behavior and also verifies that an identical model-space relative chunk maps to
different absolute actions when the execution reference state changes.

## Deployment implication

The reference state is part of the executable action semantics. Logging only the
final absolute action is insufficient to reconstruct the model-space relative
prediction; conversely, replaying a relative prediction against a different
reference state intentionally changes the absolute command.

For debugging, record at least:

- whether relative processing is enabled;
- the per-group action representation;
- the reference state used for inverse conversion;
- the post-conversion absolute action.

This does not change the API or controller behavior; it documents and tests the
existing contract.
