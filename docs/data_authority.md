# Data authority and unresolved evidence

The active source-only generation `fresh_20261026_01` freshly scanned **661,132 RPC frames** and **324,850,904 points** in the immutable source LMDB files. No previous research metadata was an input. All payloads satisfy the bounded float32 `[N,4]` serializer; byte order was detected from payload and validated during the full scan. Channel order is XYZ plus **intensity, never Doppler**. No nonfinite rows or zero rows were found; point counts range from 85 to 1,435, median 488. All RPC keys have matching params, calibration and indicator keys. New SHA-256 hashes and before/after file stats are stored in `derived_data/m4human_relational64/fresh_20261026_01/meta/source_inventory.json`.

There are **993 distinct subject/action pairs**, compared with **999 sequences described in the paper**. The seven absent pairs relative to the Cartesian product of 20 subjects and 50 actions are `(1,1), (1,15), (1,17), (1,18), (1,21), (13,2), (13,8)`. The cause of the discrepancy is unknown. This audit does not rename 993 groups as 999 recordings or invent missing recordings. Existing integer sequences contain 139 gaps representing 341 absent indices. Missing indices cause segment boundaries.

The SHA-256 split is train `[20,4,7,1,8,6,9,10,15,2,12,18]`, validation `[16,19,17,5]`, test `[3,11,14,13]`. The maximum allowed gap is 0.125 seconds. Earlier checkpoint exposure is not erased by a split; new models must start from scratch and an exposure ledger must identify prior use.

## Numerical coordinate evidence

Thirty-two examples from all 16 train/validation subjects were decoded independently from params/calib. No test target labels were read. The candidate Vicon-to-camera-to-radar transformation follows the published root transformation: Vicon translation multiplied by 0.001, radar translation unscaled. Rotations pass the declared orthogonality and determinant tolerances. The maximum transpose round-trip error is **1.358e-9 meter**. No matrix was orthogonalized or silently repaired. The stored JSON includes original joints, all calibration values, mapped joints, nearest point distances, translation/pelvis differences and per-example tests. `coordinate_overlay.png` shows the candidate mapping in XY, XZ and YZ projections, including train and validation examples.

These checks establish arithmetic consistency and spatial plausibility. They do **not** establish direct-joint equivalence to a particular SMPL-X mesh regressor, nor do they independently prove source axes, handedness or preprocessing. Published utilities also contain conflicting scale conventions between root and mesh paths. No constant Z offset is applied.

![Six train and validation RPC examples with candidate transformed skeletons in XY, XZ and YZ](figures/fresh_20261026_01/coordinate_overlay.png)

*Figure 1. Point colors follow intensity; red lines show the candidate transformed
direct joints. The three columns show XY, XZ and YZ. This is a byte-identical
publication copy of the local audit image, not a prediction accuracy result.
The source keys and original image hash are in the
[visual check](report_training/fresh_20261026_01/coordinate_visual_check.json).
See the [visual evidence gallery](visual_evidence.md) and
[image manifest](figures/fresh_20261026_01/manifest.json) for provenance.*

## G0 remains blocked

The following artifacts are required to resolve G0; a previous `audited: true` flag or prose assertion is insufficient:

1. **Target reference equivalence:** licensed matching SMPL-X model assets, their SHA-256 hashes, gender mapping, body/expression dimensions and declared hand/face defaults; the exact numeric `[22,V]` regressor with hash, ordered joint names and source-row mapping; and a numerical comparison of direct joints against independently generated mesh-regressed joints on identified train/validation source keys. Store sample keys, per-joint absolute errors, tolerance, expected unit/frame, regressor row sums and vertex count. Alternatively provide an authoritative export certificate with equivalent independently checkable content. Translation is not substituted for pelvis.
2. **RPC coordinate and preprocessing provenance:** the producer of `RawPoints`, including axis directions, handedness, meter conversion, static 4×4 transform, source offsets and whether body GT, camera information or bounding boxes affected preprocessing. Cache packing merely saves the existing `RawPoints`; it does not document their generation. Do not assume uncropped deployment input.
3. **Time and recording provenance:** raw filename-to-cache-key mapping or timestamp/synchronization table, source recording/take identifier and measured/nominal distinction. The official producer enumerates the union of radar/RGB filenames and then retains common files. Contiguous cache indices cannot rule out gaps already absent from that union. A nominal hardware rate of 12 Hz does not prove an actual timing grid for this package. Each frame must map to its source sequence without interpolating missing annotation.
4. **Package completeness:** explain the 993 versus 999 sequence discrepancy and supply a source inventory or an explicit package completeness limitation. Do not silently shrink or change the subject split.

The accompanying `data_authority.template.json` records unset fields as null. It is a collection template, not an approved runtime inventory. G0 must only change after the evidence is reviewed and hashes verified. Source datasets and the old repository remain read-only.

## Diagnostic preparation

The provisional manifest uses the specified 12/4/4 split, 64-frame windows, 150-point policy and train/validation/test strides 8/16/16. It contains 1,132 segments and 44,517/7,541/7,583 windows. Window membership depends only on sensor quality, recording identity and provisional time, not on annotation validity. Nominal timestamps are explicitly provisional; these manifests cannot authorize scientific physical training. No canonical target/QA export or scientifically accepted training result is presented as passed while G0 is unresolved. The separate `relational64.diagnostic_pilot` runner can train on actual RPC with explicitly provisional first22 direct joints and nominal time. Its panel-only normalizer, QA, metrics and diagnostic checkpoints cannot satisfy G0 or enter canonical artifact readers. See the step-by-step runbook and diagnostic run reports for this engineering execution.

Sources: [pinned cache producer](https://raw.githubusercontent.com/FanJunqiao/M4Human/e01b8fed6132306c3553ed6680be912c21594dce/dataset/m4human_utils.py), [pinned serializer](https://raw.githubusercontent.com/FanJunqiao/M4Human/e01b8fed6132306c3553ed6680be912c21594dce/dataset/lmdb_utils.py), [pinned coordinate utilities](https://raw.githubusercontent.com/FanJunqiao/M4Human/e01b8fed6132306c3553ed6680be912c21594dce/dataset/data_loader_camera_calibration.py), [M4Human paper](https://arxiv.org/html/2512.12378v3).

## Deeper source search following user direction

The old repository was searched including hidden and ignored files, excluding dependency and Git directories. No SMPL-X model/regressor files or raw MAT files were found; the NPZ matches were earlier diagnostic outputs. The dataset directory contains seven LMDB databases and their lock files. The official recursive Git tree contains 218 entries and is not truncated; 61 source/configuration text files were saved under `docs/reference/upstream/pinned` with hashes. The current official main commit matches the audited pinned commit. The tree contains no NPZ, PKL, MAT or MATLAB model/data assets. Its README explicitly requests separate SMPL-X model downloads. The exact search and hashes are saved in `provenance_search.json`.

The official loader establishes that its final joint supervision is produced from SMPL-X vertices with `J_regressor[:22]`. It does not simply return the stored `params.joints`. Those stored joints enter through external JSON whose generation is absent from this repository. This makes the missing model/regressor assets a concrete target-equivalence blocker. The official path first calibrates body parameters, generates vertices without a second calibration, then regresses joints. An alternative unused mesh conversion utility has ambiguous translation scaling; it is not used as an authority for changing the verified root formula. The optional minus1.5-meter Z normalization is applied after reading the cache, so it must not be silently applied to stored RPC.

`reference_targets.py` now supplies an optional independent reference route once licensed, hashed local assets are available. It applies audited parameter units, explicit gender and hand/face conventions, `use_pca=False`, transforms vertices once and applies the checked 22-row regressor. The synthetic model-stub tests establish software behavior only. `compare_direct_reference` writes per-joint numerical errors for train/validation samples without declaring G0 or global equivalence passed.
