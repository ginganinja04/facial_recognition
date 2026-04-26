# Figure Notes for Paper

Research question:

> Should the general public be concerned with public online cameras?

Project claim:

> Public video feeds can support the creation of persistent pseudonymous cyber
> identity profiles through repeated observable patterns, even without direct
> facial identification.

## Key Metrics From This Run

- Total person detections: 17,117
- Public camera views analyzed: 4
- Frames with detections: 2,400
- Per-camera tracks: 358
- Pseudonymous global profiles: 351
- Profiles linked across more than one camera: 7
- Detections with explicit cross-camera match scores: 7

These numbers should be framed as a proof-of-concept, not as ground-truth
identification accuracy.

## Suggested Figures and Captions

### Figure 1: `01_detection_volume_by_camera.png`

Suggested caption:

> Person detections, per-camera tracks, and pseudonymous global profiles
> extracted from four public video feeds. The volume of structured observations
> demonstrates that ordinary public streams can be converted into analyzable
> identity-adjacent records at scale.

Use this figure to show that public feeds are not just passive footage; they can
be transformed into a dataset of repeated observations.

### Figure 2: `02_global_ids_by_camera_span.png`

Suggested caption:

> Distribution of pseudonymous global IDs by the number of camera views in which
> they appear. Several profiles were linked across more than one public feed,
> showing the feasibility of cross-camera identity persistence without facial
> recognition.

Use this figure as the clearest support for the main claim.

### Figure 3: `03_top_persistent_profiles.png`

Suggested caption:

> The twenty most persistent pseudonymous profiles by detection count. Red bars
> indicate profiles observed across multiple camera views. Persistence alone can
> reveal repeated presence and behavioral patterns, even when names and faces are
> unknown.

Use this figure to explain why pseudonymity still matters: a profile can be
meaningful even without a real name.

### Figure 4: `04_profile_observation_timeline.png`

Suggested caption:

> Observation timeline for frequently detected global IDs. Each point represents
> a detection of a pseudonymous profile in a frame, colored by camera view. The
> timeline shows how repeated detections can form a behavioral trace over time.

Use this figure to connect the technical output to privacy risk: repeated
observations become a pattern.

### Figure 5: `05_confidence_and_cross_camera_matches.png`

Suggested caption:

> Detector confidence and cross-camera linkage rate by camera view. This figure
> separates basic detection reliability from the smaller subset of observations
> that were linked across feeds.

Use this figure to acknowledge limitations. Cross-camera linkage exists, but it
is not universal and should not be presented as perfect identification.

### Figure 6: `06_video_contact_sheet.png`

Suggested caption:

> Representative frames from annotated output videos. Bounding boxes and IDs
> illustrate how raw public video can be converted into machine-readable
> observation records.

Use this figure to make the pipeline concrete for readers who have not seen the
videos.

## Suggested Paper Argument

The project does not prove that anonymous public-camera viewers can identify
specific named individuals. Instead, it shows a lower but still important privacy
risk: public video can be converted into persistent pseudonymous profiles. A
profile such as `G23` can accumulate observations across frames and, in some
cases, across camera views. Over time, that profile can encode presence,
movement, clothing appearance, location, and recurrence patterns. This supports
the concern that public online cameras can enable identity-adjacent tracking
even when direct facial identification is not used.

## Suggested Limitations Paragraph

This pipeline uses YOLO person detection and simple HSV appearance histograms,
so the global IDs should be interpreted as pseudonymous profile candidates
rather than verified identities. Similar clothing, lighting changes, occlusions,
and camera angle differences can produce false matches or missed matches.
Because of these limitations, the results are best understood as a
proof-of-concept demonstration of feasibility and privacy risk, not as a
production-grade re-identification system.
