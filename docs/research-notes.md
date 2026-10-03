# Research provenance and reproducibility

The primary artifact was `ENGDATA301_Project.ipynb`, now `notebooks/deep_q_learning.ipynb`. No PDF, model checkpoint, external dataset, or populated Kaggle scripts were present. The empty `Code_1.ipynb` and empty `kaggle-scripts/` directory were removed. Two lecture notebooks remain under `reference/` with clear attribution as supplied teaching material.

## What the stored evidence supports

- CartPole DQN: 527 logged episodes, final running reward 223.60; first 200 crossing at episode 521 (202.32).
- CartPole DDQN: 460 logged episodes, final running reward 66.06, followed by a saved `KeyboardInterrupt`. It must not be described as solved.
- LunarLander generalized DDQN: 590 logged episodes, final running reward 201.60 and a solved message.

Full original stdout is preserved in `results/*_original_log.txt`; parsed episode/reward/epsilon values are in the adjacent CSVs. Hundreds of repetitive training lines were replaced by concise notebook output summaries. Original plotted PNGs were extracted from the notebook without generating replacement research figures. Logs round rewards to two decimal places; per-episode raw returns and trained weights cannot be recovered from them.

## Source/output differences and methodological caveats

- The LunarLander saved source prints “Partly solved” only when the running reward is between -75 and -60, but its stored log prints that message at positive rewards as high as 201.60. The recorded execution therefore did not use exactly that saved logging condition. Reward values and original log text are retained; the consolidated runner reports only the main stopping threshold.
- The original target-network initialization copy is commented out. Target and online networks begin with independent weights until the first scheduled synchronization. This choice can affect early learning and is preserved.
- Replay updates start only when memory length is strictly greater than batch size. The smoke check collects one extra transition, preserving this behavior.
- The replay terminal flag is Gymnasium `terminated`; `truncated` stops the rollout but does not zero the bootstrapped value. This distinction is preserved.
- “General” refers to flat observation vectors and discrete action spaces, not arbitrary Gymnasium environments.
- The stopping condition can fire with fewer than 100 completed episodes because the running mean uses available history. The recorded solved runs are longer than 100 episodes. Thresholds are the project's criteria, not a claim about external benchmark certification.
- Network initialization, environment resets, and agent random generators were not seeded in the historical experiment. Smoke mode seeds its environment/replay generator for a bounded check; it does not reconstruct the historical random state.
- The two reference examples differ in replay-memory settings and per-step `clear_session()` behavior. They are retained as distinct teaching references, not deduplicated as if they were identical.

## Engineering changes

Moved agent code into an importable module and shared identical replay/action/decay methods through inheritance. Original network depths, widths, optimizer settings, loss functions, target equations, and training batch sizes remain intact. The generalized constructor still obtains dimensions from the environment. Each `fit` now spells out the original implicit `epochs=1`.

The action count is converted to a Python `int`: Gymnasium can return a NumPy integer, which the tested Keras Dense constructor rejects. This compatibility fix preserves the action count and architecture.

The notebook uses a bounded runner instead of duplicated unbounded training loops. Default smoke checks collect one batch plus one transition, perform one optimizer update, check inference, and copy target weights. Full runs retain the original per-step update and target-sync schedule, with an explicit episode cap and separate output files. No full training runs were performed during cleanup.

Dependencies define a compatible CPU environment, not the original Kaggle environment. Box2D/pygame are optional for the CartPole-only command and required for LunarLander. GPU execution and long-run convergence are separate from a successful smoke check.

## Validation

Validated all notebook schemas and Python syntax. On Python 3.12 with TensorFlow 2.19.0, Keras 3.15.1, Gymnasium 1.1.1, NumPy 2.1.3, Box2D 2.3.10, and pygame 2.6.1, both CartPole agents and the generalized LunarLander agent passed the CPU smoke check. Each performed exactly one optimizer update with finite loss, valid greedy inference, and successful target synchronization. The LunarLander check executed the main notebook's setup/import/experiment cells. Its first attempt exposed the integer-type compatibility issue described above, before any optimizer update. No full training, GPU validation, or historical-score reproduction was attempted.
