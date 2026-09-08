// FallWatch — multi-person fall detection prototype
// Tracks every person MoveNet finds in frame. Each physical person gets
// a stable internal id that survives even if the pose model's own
// tracker id changes mid-motion (which happens during a fall, since the
// bounding box shape changes drastically in a couple frames).

let video = document.getElementById("video");
let canvas = document.getElementById("overlay");
let ctx = canvas.getContext("2d");
let startBtn = document.getElementById("startBtn");
let liveDot = document.getElementById("liveDot");
let peopleCount = document.getElementById("peopleCount");
let monitorTimeEl = document.getElementById("monitorTime");
let fallCountEl = document.getElementById("fallCount");
let muteBtn = document.getElementById("muteBtn");
let personList = document.getElementById("personList");
let latestEvent = document.getElementById("latestEvent");
let alertBanner = document.getElementById("alertBanner");
let alertBannerText = document.getElementById("alertBannerText");

let detector = null;
let running = false;
let monitorStartTime = null;
let totalFallCount = 0;
let alarmMuted = false;

// people, keyed by a STABLE internal id (e.g. "p1", "p2"), independent
// of MoveNet's own per-frame tracker id
let people = {};
let nextPersonNumber = 1;

// maps the pose model's current tracker id -> our stable internal id
let trackerToPerson = {};

// if a tracker id disappears (often because the tracker briefly lost
// someone during fast motion like a fall), we hold onto their history
// for a short grace period — if a new tracker id shows up nearby, we
// treat it as the same person instead of starting their history over
let lostBuffer = [];
let lostGraceFrames = 20;
let reclaimDistance = 130;

let minConfidence = 0.3;
let historyLen = 22;
let movementHistoryLen = 10;
let downConfirmFrames = 10;
let walkMovementThreshold = 2.8;
let fallCandidateRequired = 1;
let angleFallThreshold = 45;
let aspectFallThreshold = 1.1;      // bounding box wider than tall = lying down signal
let dropSpeedThreshold = 20;        // raw px/window fallback, used only before scale is learned
let dropSpeedRatioThreshold = 0.55; // hips fell more than ~55% of a torso-length within the window
let severeCollapseRatio = 0.55;     // current torso length vs learned "standing" torso length
let moderateCollapseRatio = 0.78;

async function setupCamera() {
    let stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
    });
    video.srcObject = stream;
    return new Promise(function(resolve) {
        video.onloadedmetadata = function() {
            resolve(video);
        };
    });
}

async function loadModel() {
    await tf.setBackend("webgl");
    await tf.ready();
    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
            modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
            enableTracking: true,
            trackerType: poseDetection.TrackerType.Keypoint
        }
    );
}

startBtn.addEventListener("click", async function() {
    startBtn.disabled = true;
    startBtn.textContent = "Loading model…";
    try {
        await setupCamera();
        video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        await loadModel();
        running = true;
        startBtn.textContent = "Running";
        liveDot.classList.add("on");
        monitorStartTime = Date.now();
        setInterval(updateMonitorTime, 1000);
        logEvent("Monitoring started");
        detectLoop();
    } catch (err) {
        console.error(err);
        startBtn.disabled = false;
        startBtn.textContent = "Start camera";
        logEvent("Camera error — check permissions", true);
    }
});

muteBtn.addEventListener("click", function() {
    alarmMuted = !alarmMuted;
    muteBtn.textContent = alarmMuted ? "🔇 Alarm muted" : "🔊 Alarm on";
    muteBtn.classList.toggle("muted", alarmMuted);
});

function updateMonitorTime() {
    if (!monitorStartTime) return;
    let elapsed = Math.floor((Date.now() - monitorStartTime) / 1000);
    let mins = Math.floor(elapsed / 60);
    let secs = elapsed % 60;
    let minsStr = mins < 10 ? "0" + mins : "" + mins;
    let secsStr = secs < 10 ? "0" + secs : "" + secs;
    monitorTimeEl.textContent = minsStr + ":" + secsStr;
}

// ---------- Person identity ----------

function roughHipPosition(keypoints) {
    let byName = {};
    keypoints.forEach(function(k) { byName[k.name] = k; });
    let lHip = byName["left_hip"];
    let rHip = byName["right_hip"];
    if (lHip && rHip && lHip.score > 0.3 && rHip.score > 0.3) {
        return { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
    }
    return null;
}

function createNewPerson() {
    let internalId = "p" + nextPersonNumber;
    let label = "Person " + nextPersonNumber;
    nextPersonNumber++;

    let person = {
        id: internalId,
        label: label,
        hipYHistory: [],
        hipXWindow: [],
        hipYWindow: [],
        ankleXWindow: [],
        angleHistory: [],
        aspectHistory: [],
        downFrameCount: 0,
        fallCandidateFrames: 0,
        unstableFrames: 0,
        currentState: "standing",
        alertFired: false,
        lastHipPos: null,
        prevHipMid: null,
        prevRawAngle: null,
        torsoScale: null,
        fallCount: 0,
        alarmIntervalId: null
    };

    people[internalId] = person;
    createPersonCard(person);
    return person;
}

function resolvePersonForTracker(trackerId, keypoints) {
    if (trackerToPerson[trackerId] !== undefined) {
        let existing = people[trackerToPerson[trackerId]];
        if (existing) return existing;
    }

    // this tracker id is new to us — check if it might be someone we
    // just lost track of a moment ago, based on position
    let hipPos = roughHipPosition(keypoints);
    if (hipPos) {
        for (let i = 0; i < lostBuffer.length; i++) {
            let entry = lostBuffer[i];
            if (!entry.lastHipPos) continue;
            let d = Math.hypot(entry.lastHipPos.x - hipPos.x, entry.lastHipPos.y - hipPos.y);
            if (d < reclaimDistance) {
                lostBuffer.splice(i, 1);
                trackerToPerson[trackerId] = entry.internalId;
                return people[entry.internalId];
            }
        }
    }

    let person = createNewPerson();
    trackerToPerson[trackerId] = person.id;
    return person;
}

function createPersonCard(person) {
    let card = document.createElement("div");
    card.className = "person-card";
    card.id = "person-card-" + person.id;
    card.innerHTML =
        '<div class="person-card-top">' +
            '<span class="person-name">' + person.label + '</span>' +
            '<span class="person-falls" data-role="falls">0 falls</span>' +
        '</div>' +
        '<div class="status-value status-standing" data-role="status">Person standing</div>' +
        '<div class="status-sub" data-role="sub">Steady, no movement</div>';
    personList.appendChild(card);
}

function removePersonCard(id) {
    let card = document.getElementById("person-card-" + id);
    if (card) card.remove();
}

function updatePersonCard(person) {
    let card = document.getElementById("person-card-" + person.id);
    if (!card) return;

    let labels = {
        standing: "Person standing",
        walking: "Person walking",
        unstable: "Losing balance",
        falling: "Possible fall",
        down: "PERSON DOWN"
    };
    let subLabels = {
        standing: "Steady, no movement",
        walking: "Moving normally",
        unstable: "Sudden loss of stability detected",
        falling: "Checking — could be a fall",
        down: "Immediate attention needed"
    };

    let statusEl = card.querySelector('[data-role="status"]');
    let subEl = card.querySelector('[data-role="sub"]');
    let fallsEl = card.querySelector('[data-role="falls"]');

    statusEl.textContent = labels[person.currentState] || person.currentState;
    statusEl.className = "status-value status-" + person.currentState;
    subEl.textContent = subLabels[person.currentState] || "";
    fallsEl.textContent = person.fallCount + (person.fallCount === 1 ? " fall" : " falls");
    card.classList.toggle("danger", person.currentState === "down");
}

// ---------- Main loop ----------

async function detectLoop() {
    if (!running) return;

    let poses = await detector.estimatePoses(video);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let seenTrackerIds = [];
    let seenInternalIds = [];

    poses.forEach(function(pose) {
        let trackerId = pose.id !== undefined ? pose.id : 0;
        seenTrackerIds.push(trackerId);

        let person = resolvePersonForTracker(trackerId, pose.keypoints);
        seenInternalIds.push(person.id);

        let hipPos = roughHipPosition(pose.keypoints);
        if (hipPos) person.lastHipPos = hipPos;

        drawSkeleton(pose.keypoints, person);
        evaluatePose(pose.keypoints, person);
    });

    // any tracker id we knew about last frame but didn't see this frame
    // goes into the lost buffer instead of being deleted immediately
    Object.keys(trackerToPerson).forEach(function(key) {
        let trackerId = Number(key);
        if (seenTrackerIds.indexOf(trackerId) === -1) {
            let internalId = trackerToPerson[trackerId];
            delete trackerToPerson[trackerId];
            if (people[internalId] && seenInternalIds.indexOf(internalId) === -1) {
                lostBuffer.push({
                    internalId: internalId,
                    lastHipPos: people[internalId].lastHipPos,
                    framesSinceLost: 0
                });
            }
        }
    });

    // age out anyone who's been gone too long to still be "the same person"
    lostBuffer = lostBuffer.filter(function(entry) {
        entry.framesSinceLost++;
        if (entry.framesSinceLost > lostGraceFrames) {
            let person = people[entry.internalId];
            if (person && person.alarmIntervalId) clearInterval(person.alarmIntervalId);
            removePersonCard(entry.internalId);
            delete people[entry.internalId];
            return false;
        }
        return true;
    });

    peopleCount.textContent = seenInternalIds.length;
    updateGlobalAlertBanner();

    requestAnimationFrame(detectLoop);
}

function updateGlobalAlertBanner() {
    let downPeople = Object.values(people).filter(function(p) { return p.currentState === "down"; });
    if (downPeople.length > 0) {
        alertBanner.classList.remove("hidden");
        let names = downPeople.map(function(p) { return p.label; }).join(", ");
        alertBannerText.textContent = names + " — FALL DETECTED";
    } else {
        alertBanner.classList.add("hidden");
    }
}

// ---------- Drawing ----------

function drawSkeleton(keypoints, person) {
    let byName = {};
    keypoints.forEach(function(k) { byName[k.name] = k; });

    let color = person.currentState === "down" ? "#e0483f"
        : person.currentState === "falling" ? "#e0a742"
        : person.currentState === "unstable" ? "#f4b860"
        : "#35c9d1";

    keypoints.forEach(function(k) {
        if (k.score < minConfidence) return;
        ctx.beginPath();
        ctx.arc(k.x, k.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
    });

    let pairs = [
        ["left_shoulder", "right_shoulder"],
        ["left_hip", "right_hip"],
        ["left_shoulder", "left_hip"],
        ["right_shoulder", "right_hip"],
        ["left_hip", "left_ankle"],
        ["right_hip", "right_ankle"]
    ];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    pairs.forEach(function(pair) {
        let p1 = byName[pair[0]];
        let p2 = byName[pair[1]];
        if (p1 && p2 && p1.score > minConfidence && p2.score > minConfidence) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }
    });

    let nose = byName["nose"];
    if (nose && nose.score > minConfidence) {
        ctx.save();
        ctx.translate(nose.x, nose.y - 40);
        ctx.scale(-1, 1);
        ctx.font = "600 14px 'IBM Plex Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillStyle = color;
        ctx.fillText(person.label + " · " + labelForState(person.currentState), 0, 0);
        ctx.restore();
    }
}

function labelForState(state) {
    if (state === "standing") return "standing";
    if (state === "walking") return "walking";
    if (state === "unstable") return "losing balance";
    if (state === "falling") return "falling…";
    if (state === "down") return "DOWN";
    return state;
}

// ---------- Fall detection logic ----------

// Wider-than-tall bounding box is a strong, calibration-free "lying down"
// signal — independent of the torso-angle math below, so it catches cases
// (e.g. limbs splayed, camera angle) that angle alone might miss.
function getBoundingBoxAspect(keypoints) {
    let xs = [];
    let ys = [];
    keypoints.forEach(function(k) {
        if (k.score >= minConfidence) {
            xs.push(k.x);
            ys.push(k.y);
        }
    });
    if (xs.length < 4) return null;

    let width = Math.max.apply(null, xs) - Math.min.apply(null, xs);
    let height = Math.max.apply(null, ys) - Math.min.apply(null, ys);
    if (height <= 0) return null;
    return width / height;
}

function evaluatePose(keypoints, person) {
    let byName = {};
    keypoints.forEach(function(k) { byName[k.name] = k; });

    let lHip = byName["left_hip"];
    let rHip = byName["right_hip"];
    let lSh = byName["left_shoulder"];
    let rSh = byName["right_shoulder"];
    let lAnkle = byName["left_ankle"];
    let rAnkle = byName["right_ankle"];

    if (!lHip || !rHip || !lSh || !rSh) return;
    if (lHip.score < minConfidence || rHip.score < minConfidence) return;
    if (lSh.score < minConfidence || rSh.score < minConfidence) return;

    let hipMid = { x: (lHip.x + rHip.x) / 2, y: (lHip.y + rHip.y) / 2 };
    let shMid = { x: (lSh.x + rSh.x) / 2, y: (lSh.y + rSh.y) / 2 };

    let dx = shMid.x - hipMid.x;
    let dy = shMid.y - hipMid.y;
    let rawAngle = Math.abs(90 - (Math.atan2(Math.abs(dy), Math.abs(dx)) * 180) / Math.PI);

    // single-frame jumps, used for the pre-fall "losing balance" check —
    // separate from the smoothed values used for the actual fall check
    let angleJump = person.prevRawAngle !== null && person.prevRawAngle !== undefined
        ? Math.abs(rawAngle - person.prevRawAngle) : 0;
    let hipJerk = person.prevHipMid
        ? Math.hypot(hipMid.x - person.prevHipMid.x, hipMid.y - person.prevHipMid.y) : 0;
    person.prevRawAngle = rawAngle;
    person.prevHipMid = hipMid;

    person.angleHistory.push(rawAngle);
    if (person.angleHistory.length > 3) person.angleHistory.shift();
    let angleFromVertical = person.angleHistory.reduce(function(sum, a) { return sum + a; }, 0) / person.angleHistory.length;

    person.hipYHistory.push(hipMid.y);
    if (person.hipYHistory.length > historyLen) person.hipYHistory.shift();
    let dropSpeed = person.hipYHistory.length >= 2
        ? person.hipYHistory[person.hipYHistory.length - 1] - person.hipYHistory[0]
        : 0;

    person.hipXWindow.push(hipMid.x);
    person.hipYWindow.push(hipMid.y);
    if (person.hipXWindow.length > movementHistoryLen) person.hipXWindow.shift();
    if (person.hipYWindow.length > movementHistoryLen) person.hipYWindow.shift();

    if (lAnkle && lAnkle.score > minConfidence) {
        person.ankleXWindow.push(lAnkle.x);
    } else if (rAnkle && rAnkle.score > minConfidence) {
        person.ankleXWindow.push(rAnkle.x);
    }
    if (person.ankleXWindow.length > movementHistoryLen) person.ankleXWindow.shift();

    let hipMovement = 0;
    for (let i = 1; i < person.hipXWindow.length; i++) {
        let stepX = person.hipXWindow[i] - person.hipXWindow[i - 1];
        let stepY = person.hipYWindow[i] - person.hipYWindow[i - 1];
        hipMovement += Math.sqrt(stepX * stepX + stepY * stepY);
    }
    hipMovement = person.hipXWindow.length > 1 ? hipMovement / (person.hipXWindow.length - 1) : 0;

    let ankleMovement = 0;
    for (let i = 1; i < person.ankleXWindow.length; i++) {
        ankleMovement += Math.abs(person.ankleXWindow[i] - person.ankleXWindow[i - 1]);
    }
    ankleMovement = person.ankleXWindow.length > 1 ? ankleMovement / (person.ankleXWindow.length - 1) : 0;

    let avgMovement = (hipMovement + ankleMovement) / 2;
    let isMoving = avgMovement > walkMovementThreshold;

    // Torso length (shoulder-mid to hip-mid) instead of full-body pixel
    // height: doesn't need ankles/feet visible at all, and shrinks the
    // same way full-body height used to when someone goes horizontal
    // toward the camera (2D foreshortening), so it's a drop-in scale
    // reference without ever asking the user for a real-world height.
    let torsoLen = Math.hypot(shMid.x - hipMid.x, shMid.y - hipMid.y);

    if ((person.currentState === "standing" || person.currentState === "walking") && !isMoving) {
        person.torsoScale = person.torsoScale
            ? person.torsoScale * 0.95 + torsoLen * 0.05
            : torsoLen;
    } else if (person.currentState === "standing" || person.currentState === "walking") {
        person.torsoScale = person.torsoScale
            ? person.torsoScale * 0.98 + torsoLen * 0.02
            : torsoLen;
    }

    let collapseRatio = person.torsoScale ? torsoLen / person.torsoScale : null;

    let aspect = getBoundingBoxAspect(keypoints);
    person.aspectHistory.push(aspect !== null ? aspect : 0);
    if (person.aspectHistory.length > 4) person.aspectHistory.shift();
    let avgAspect = person.aspectHistory.reduce(function(sum, a) { return sum + a; }, 0) / person.aspectHistory.length;

    // scale-normalized drop speed once we have a learned reference, so
    // the same real-world drop registers the same whether someone is
    // near or far from the camera; falls back to a raw-pixel threshold
    // only during the brief window before a scale has been learned
    let droppedFast = person.torsoScale
        ? (dropSpeed / person.torsoScale) > dropSpeedRatioThreshold
        : dropSpeed > dropSpeedThreshold;

    let isHorizontal = angleFromVertical > angleFallThreshold || avgAspect > aspectFallThreshold;

    // severe: near-total collapse (classic flat-on-the-floor fall)
    // moderate: a real but partial drop — knees buckling, slumping,
    // sitting down hard — doesn't need to be fully horizontal to count
    let severeCollapse = collapseRatio !== null && collapseRatio < severeCollapseRatio;
    let moderateCollapse = collapseRatio !== null && collapseRatio < moderateCollapseRatio;

    // a sudden sideways lurch or rapid tilt, without (yet) a confirmed
    // drop — this is the "about to fall" signal, not a fall itself
    let suddenTilt = angleJump > 20;
    let suddenJerk = hipJerk > Math.max(walkMovementThreshold * 4, 12);
    let instabilitySignal = (suddenTilt || suddenJerk) && !isMoving;

    if (person.currentState === "down") {
        if (!isHorizontal && !severeCollapse && !moderateCollapse) {
            person.currentState = "standing";
            person.downFrameCount = 0;
            person.fallCandidateFrames = 0;
            person.unstableFrames = 0;
            person.alertFired = false;
            if (person.alarmIntervalId) {
                clearInterval(person.alarmIntervalId);
                person.alarmIntervalId = null;
            }
            updatePersonCard(person);
            logEvent(person.label + " recovered");
        }
        return;
    }

    let strongFallSignal = (isHorizontal || severeCollapse) && droppedFast;
    let partialFallSignal = moderateCollapse && (droppedFast || suddenTilt);
    let fallSignal = strongFallSignal || partialFallSignal;

    if (person.currentState === "falling") {
        if (isHorizontal || severeCollapse || moderateCollapse) {
            person.downFrameCount++;
        } else {
            person.currentState = isMoving ? "walking" : "standing";
            person.downFrameCount = 0;
            person.fallCandidateFrames = 0;
            updatePersonCard(person);
            return;
        }
        if (person.downFrameCount >= downConfirmFrames) {
            person.currentState = "down";
            updatePersonCard(person);
            if (!person.alertFired) {
                person.alertFired = true;
                person.fallCount++;
                totalFallCount++;
                fallCountEl.textContent = totalFallCount;
                logEvent(person.label + " — FALL CONFIRMED", true);
                triggerAlarm(person.label);
                person.alarmIntervalId = setInterval(function() {
                    triggerAlarm(person.label);
                }, 6000);
            }
        }
    } else if (person.currentState === "unstable") {
        if (fallSignal) {
            person.fallCandidateFrames++;
            if (person.fallCandidateFrames >= fallCandidateRequired) {
                person.currentState = "falling";
                updatePersonCard(person);
                logEvent(person.label + " — possible fall, monitoring…");
            }
        } else if (instabilitySignal) {
            person.unstableFrames++;
        } else {
            person.unstableFrames = 0;
            person.currentState = isMoving ? "walking" : "standing";
            updatePersonCard(person);
        }
    } else {
        person.downFrameCount = 0;
        if (fallSignal) {
            person.fallCandidateFrames++;
            if (person.fallCandidateFrames >= fallCandidateRequired) {
                person.currentState = "falling";
                updatePersonCard(person);
                logEvent(person.label + " — possible fall, monitoring…");
            }
        } else if (instabilitySignal) {
            person.unstableFrames = (person.unstableFrames || 0) + 1;
            if (person.unstableFrames >= 2) {
                person.currentState = "unstable";
                updatePersonCard(person);
                logEvent(person.label + " — losing balance");
                beep();
            }
        } else {
            person.fallCandidateFrames = 0;
            person.unstableFrames = 0;
            let nextState = isMoving ? "walking" : "standing";
            if (person.currentState !== nextState) {
                person.currentState = nextState;
                updatePersonCard(person);
            }
        }
    }
}

// ---------- Alarm ----------

function beep() {
    if (alarmMuted) return;
    try {
        let audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        let osc = audioCtx.createOscillator();
        let gain = audioCtx.createGain();
        osc.frequency.value = 880;
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.3);
    } catch (e) {
        console.error("beep failed", e);
    }
}

function speak(text) {
    if (alarmMuted) return;
    if (!window.speechSynthesis) return;
    let utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
}

function triggerAlarm(personLabel) {
    beep();
    speak("Fall detected. " + personLabel + ". Please respond.");
}

// ---------- Event log ----------

function logEvent(message, isAlert) {
    let time = new Date().toLocaleTimeString();
    latestEvent.textContent = time + " — " + message;
    latestEvent.style.color = isAlert ? "#e0483f" : "#dbe4ea";
}