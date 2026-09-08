# FallWatch — Browser-Based Fall Detection Prototype

A real-time fall detection demo built for hospital hallway–style monitoring.
Runs entirely in the browser using pose estimation on a live camera feed —
no video data leaves the device, and no server or install is required.

**[Live demo →](#)** *(add your GitHub Pages link here once deployed)*

## What it does

- Detects **every person** in the camera frame at once using **MoveNet
  MultiPose** (via TensorFlow.js), tracking each individually across frames
- Each tracked person gets their own status card: `Person walking`,
  `Falling…`, or `PERSON DOWN`, plus their own fall count
- Displays a live label above each person's head with their name and state
- Fires a **voice alarm** ("Fall detected. Person 2. Please respond.") plus
  an audible beep when a fall is confirmed, repeating every 6 seconds until
  the person recovers or the alarm is muted
- A visible on-screen banner lists everyone currently down
- Height calibration is available for the first tracked person ("Person 1")

## Why

Falls are one of the most common preventable safety incidents in hospitals,
especially for elderly or post-surgical patients. This is a proof-of-concept
for a camera-based monitoring system that could sit in a hallway and alert
staff automatically — built as a learning project to explore computer
vision, pose estimation, and real-time browser ML.

**Note:** this is a prototype using a personal webcam, not a
clinically validated medical device. Real deployment would require privacy
review, HIPAA-compliant infrastructure, and validation against real
clinical fall data.

## Tech stack

- Vanilla JavaScript, HTML, CSS — no framework, no build step
- [TensorFlow.js](https://www.tensorflow.org/js)
- [MoveNet](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection) (`SINGLEPOSE_LIGHTNING`) for pose keypoints

## How the detection works

1. Every frame, MoveNet returns body keypoints (shoulders, hips, nose, etc.)
2. The midpoint of the shoulders and hips gives a torso angle relative to
   vertical (~0° standing, ~90° lying flat)
3. A short rolling buffer of the hip's y-position estimates vertical drop
   speed
4. If the torso goes horizontal **and** dropped quickly, state moves to
   `falling`
5. If that horizontal position holds for ~20 consecutive frames, state
   escalates to `down` and the alert fires — this delay avoids triggering
   on bending down, sitting, or a single bad frame

This is a heuristic, not a trained classifier — the thresholds
(`angle > 55°`, `dropSpeed > 40px`) are tuned loosely and would need
proper calibration (and ideally a small trained model on real fall
datasets like [Le2i](http://le2i.cnrs.fr/Fall-detection-Dataset) or the
[UR Fall Detection Dataset](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html))
for production use.

## Running locally

1. Clone this repo
2. Open the folder in VS Code
3. Install the **Live Server** extension
4. Right-click `index.html` → **Open with Live Server**
5. Click **Start camera** and allow webcam access

No `npm install` needed — the TensorFlow.js/MoveNet libraries load from a CDN.

## Roadmap / stretch goals

- [ ] Multi-camera / multi-zone support
- [ ] Replace heuristic thresholds with a small trained classifier
- [ ] Alert delivery via Twilio (SMS) or email API
- [ ] Dashboard with historical alert log and timestamps
- [ ] Support for multiple people in frame simultaneously

## Disclaimer

Built as a personal learning / portfolio project. Not affiliated with,
tested in, or endorsed by any hospital or healthcare provider.