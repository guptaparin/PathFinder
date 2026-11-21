# PathFinder: demo video script (target 2:45, hard limit 3:00)

Record with OBS or any screen recorder at 1080p. Use the deck in full-screen for slides and a terminal with a
large font (≥ 18 pt) for the live run. Speak at a calm pace; each block below is timed.

---

**0:00–0:20 · Title slide**
> "Hi, I'm Parin Gupta, Team DEADLOCK, from Thapar Institute, Track 3, Path Finder. My system, PathFinder, reconstructs a full
> 6-DoF trajectory from nothing but a jammed IMU file, including the 5 seconds after the sensor dies."

**0:20–0:45 · Problem slide**
> "The jammer adds noise, bias and scale drift that get worse over time, spikes, stuck and duplicated samples,
> dropped rows, and finally 5 seconds of dead frames. If you just integrate twice, the error explodes:
> 99 metres on average, and 371 metres when the gyro is hit too. And we have to do it in under 8 MB and
> 20 ms per second of data."

**0:45–1:20 · Solution + Architecture slides**
> "We found the real enemy is tilt. One degree of tilt error leaks gravity into the accelerometer as fake
> motion. So step one is self-calibration. A wrong gyro bias makes velocity grow without bound, while a real
> drone's velocity stays bounded. A single least-squares solve over the whole flight recovers the biases and
> initial tilt.
> Step two, two independent velocity witnesses: an ensemble of three tiny neural networks that predicts velocity and its own uncertainty, and multirotor physics. Rotor drag pushes sideways in proportion to airspeed, so the accelerometer itself measures velocity while the rotors spin.
> Step three, a Kalman/RTS smoother fuses everything with physical priors, and the same model dead-reckons the
> blind 5 seconds. It's all numpy, single-threaded."

**1:20–2:05 · LIVE terminal run + viewer**
Show:
```bash
python src/main.py data/MH_01/imu0/data.csv -o estimated_trajectory.txt --meta metadata.json --profile
head -3 estimated_trajectory.txt
wc -l estimated_trajectory.txt data/MH_01/imu0/data.csv     # one pose per input row (+ header lines)
python tools/evaluate.py estimated_trajectory.txt data/MH_01/gt.csv data/MH_01/imu0/data.csv
```
> "Here it is on a degraded EuRoC flight: 35 thousand poses in about a second, about 7 milliseconds per second
> of IMU, 3.6 MB of heap. That's under half the memory budget and about a third of the time budget. One pose per
> IMU frame in TUM format, and it parses directly in evo."
Then open `results/viewers/MH_03_heldout.html` and let it play for ~10 s:
> "This is a flight the model never saw. Grey is the hidden ground truth, teal is PathFinder from the IMU alone.
> Red marks are dropouts, and at the end the IMU dies: the amber segment is our 5-second dead reckoning."

**2:05–2:35 · Results + Differentiation slides**
> "On all 11 EuRoC flights, held out with cross-validation so no flight is scored by a model that saw it, we get
> about 2 metres ATE against 99 to 371 for naive integration and 3.45 for standing still, and every single flight
> beats standing still. The ablation shows each piece adding up: physics alone 3.2, plus the network 2.5, plus
> rotor drag 2.03 metres."

**2:35–2:50 · Brownie slide**
> "One thing we noticed: the degradation script assumes accelerometer-first columns, while EuRoC is
> gyro-first. We validated on both readings, so PathFinder works either way."

**2:50–3:00 · Thank you**
> "Code, Docker and full results are in our repo. Thank you!"

---

Checklist before upload: video ≤ 3:00 · public on YouTube (unlisted is fine) or Drive with "anyone with the
link" · paste the link into the deck (Demo & Local Setup slide) and the Google Form.
