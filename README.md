FallWatch

Real-time fall detection for hospital corridors, running entirely in the browser.

A camera watches a corridor. When someone falls, an alarm goes off. No server, no app to install, and no video ever leaves the machine.

Why I built it

Falls are one of the most common serious incidents in hospitals and care homes, and the damage is not only the injury. People who fall once often stop trusting themselves to move around alone afterwards.

Most systems that detect this are expensive, need hardware installed, or send video to a company's servers. I wanted to know whether the problem could be solved with a webcam and a browser tab.

What it does

Watches several people at once. It tracks every person in frame independently rather than assuming there is only one.

Measures height and posture. It learns each person's standing height from how they normally move, so it can tell the difference between someone who is short and someone who has gone down.

Combines several signals before deciding. Torso angle, how fast the body dropped, and height against that person's own learned baseline. No single reading can trigger an alarm on its own.

Warns before the fall completes. There is a separate "loss of balance" state that fires when someone is unsteady but has not gone down yet.

Raises a real alarm. A spoken alert and a sound, not just a label on screen.

The part worth reading the code for

The hard problem was never detecting a fall. It was not detecting one when nothing had happened.

An early version fired every time a person bent down, or when a single noisy frame put the torso angle at a strange value. A system that cries wolf gets switched off by the staff it was bought for, so false alarms are not a cosmetic problem, they are the whole problem.

What fixed it was refusing to act on one frame. The angle is smoothed over several frames, the drop has to be sustained, and the person's height has to stay low rather than dipping for an instant. Ankle visibility is also checked properly, because if the feet are not actually in shot the height reading is meaningless and the app says so instead of guessing.

There is a second thing worth knowing. When I added multi-person tracking, fall detection stopped working entirely, and nothing I had changed looked broken. The cause was that the pose model issues an ID per person, and during a fall the body shape changes so much that the model decides it is looking at somebody new and assigns a fresh ID. The movement history was being wiped at the exact moment it was needed. The fix was to keep my own stable identity for each person rather than trusting the model's.

Running it

Open index.html with Live Server in VS Code. It has to be served over http rather than opened by double clicking, because browsers block camera access on file://.

Stand far enough back that your whole body including your feet is in frame. The app will tell you if it cannot see your ankles.

Limitations

Tested against my own movements rather than a labelled dataset, so there are no accuracy figures yet. Measuring how often it misses a real fall, and how often it fires when nothing happened, is the next thing worth doing.

Needs the full body in frame. It will not work from a chest-up webcam angle.

Not a medical device. It is a prototype built to explore the problem.

Built with

JavaScript, TensorFlow.js and MoveNet pose estimation. No build step, no backend. All processing happens on the device.
