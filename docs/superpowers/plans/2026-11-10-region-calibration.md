# Region Calibration Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans in this session. User has explicitly requested implementation of the discussed local preview feature.

**Goal:** Mouse-based per-camera ROI, finite line and direction calibration in the existing preview.
**Architecture:** Standard-library JSON store/HTTP mixin, native Canvas editor, small integration into the current MJPEG handler and encoder.
**Tech Stack:** Python standard library, existing OpenCV, HTML/JavaScript Canvas.
**Spec:** docs/superpowers/specs/2026-11-10-region-calibration-design.md

## Global Constraints

- Preserve the existing dirty workspace and preview invocation; no new dependencies.
- Save normalized coordinates per source and crop, atomically; no flow counting in this change.
- Do not stop an existing service without identifying it.

## Review Focus

- Malformed/self-intersecting polygons and non-finite coordinates must not overwrite saved data.
- Camera reordering and crop changes must not reuse another coordinate space.
- Browser scaling must map clicks to the frozen image correctly.
- Request errors and disk failures must preserve the last saved configuration.
- Missing frames and changed aspect ratios must prevent misleading calibration.

## Tasks

- [x] Write and run failing unittest coverage for geometry validation, stable keys, atomic persistence, and real HTTP round trips.
- [x] Implement `region_calibration.py`: `camera_key`, `validate_calibration`, `calibration_matches_frame`, `CalibrationStore`, and `CalibrationHandlerMixin`.
- [x] Add `web/calibration.html` and `web/calibration.js`; integrate entry links, raw snapshots and overlays into `realtime_web.py`.
- [x] Run unit/HTTP tests and syntax checks; review changes and document launch/usage instructions. Browser interaction was attempted but blocked by the CUA tool's inability to verify the current Windows browser URL.

## Progress

Ruling: Edit the current local working copy because the requested preview and pre-existing edits live here; preserve user changes and do not create commits.

Verification: 18 Python tests and 6 Node editor event/request tests pass; Python compilation, JavaScript syntax and git whitespace checks pass.

Final review: independent reviewer found a cross-process whole-file overwrite risk. Reproduced with a failing two-store regression, then fixed with native OS file locking, reload-before-save and cache refresh. A synchronized two-process save test also passes.

Live rollout: verified the port-8000 process, workspace and keepalive parent before restarting only that preview process. The restarted service exposes 10 calibration links, and editor/API/raw JPEG endpoints return HTTP 200. Production calibration data was not written during tests; the separate temporary test server was stopped.
