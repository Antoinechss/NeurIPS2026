# Our model, explained simply

This page explains `solution/vemg2pose_plus.py`: what the model is, what its settings mean, and how we train it. No deep-learning background needed beyond "a neural network learns by adjusting numbers to reduce an error".

---

## 1. The problem

A wristband has **16 electrodes** arranged in a ring around the forearm. Each one measures the tiny electrical signals the forearm muscles make when they contract (this is called **EMG**). It records **2,000 values per second** per electrode.

We get **5 seconds** of this signal (16 × 10,000 numbers) and must predict, for every instant of those 5 seconds, the **20 angles of the hand joints** (how bent each finger joint is), in radians.

The score is the **average error in degrees** over all joints and all instants. Lower is better.

**The hard part:** the final ranking uses **57 people the model has never seen**. Every arm is different (muscle size, fat, skin) and everyone wears the band slightly differently. A model that memorises the training people fails on new ones. Almost every choice below is about **generalising to new people**.

---

## 2. The model in one picture

```
 EMG, 5 s, 16 electrodes, 2,000 Hz
            │
   (0) add 0.9 s of "fake past" at the start
            │
   (1) ENCODER: convolutions that summarise the signal
       2,000 values/s  ──►  25 summaries/s  (then stretched to 50/s)
            │
   (2) DECODER: an LSTM that "draws" the hand pose step by step, 50 times per second,
       each step looking at the summary AND at the pose it drew just before
            │
   (3) OUTPUT: 20 joint angles, stretched back to 2,000 values/s
```

The architecture is **VEMG2Pose**, the model published by Meta with the emg2pose dataset (Salter et al., 2024). We take it as is from the `braindecode` library and change **how we train and use it**.

### (0) The fake past ("left context")
To predict the pose at a given instant, the model needs to see about **0.9 s of signal before it** (1,790 samples). At the very start of a window there is no past, so we create one by **mirroring** the first 0.9 s of the window. It's a simple trick that lets the model predict the whole 5 s.

### (1) The encoder: from raw signal to summaries
Raw EMG looks like noise: it oscillates very fast around zero. What tells us about the hand is **how strongly each muscle is active**, which shows up as the "loudness" of the signal over tens of milliseconds.

The encoder is a stack of **convolutions**: small filters that slide along time and learn to detect useful patterns. Step by step it shrinks the 2,000 values per second into **25 summaries per second**, each a list of numbers describing "what the muscles are doing right now". It uses a type of block called **TDS** (time-depth separable), which is just an efficient way to build such filters.

- Size: about **2.6 million** learned numbers (parameters).

### (2) The decoder: drawing the hand step by step
The decoder is an **LSTM**, a neural network with a memory. It runs **50 times per second** (250 steps for 5 s). At each step it receives:
- the encoder's summary for that moment, and
- **the pose it predicted at the previous step**.

So it builds the movement like an animation, frame after frame, and the result is smooth and consistent over time. That's the main difference from a model that would predict each instant independently.

- Size: 2 layers, 512 memory units each, about **3.3 million** parameters.

### (3) "Hybrid" output: position first, then speed
At each step the decoder outputs two things for every joint: an **angle** (position) and a **speed** (how fast the angle changes).
- During the **first 0.25 s**, we use the angle directly. The model has to guess the starting pose from the muscles alone.
- **After that**, we use the speed: new angle = previous angle + speed × time. Fingers move smoothly, and predicting small changes is easier than predicting absolute angles from scratch.

**Total model size: about 6 million parameters** (the weights file is about 24 MB).

---

## 3. How we train it

### The data
- **158 training people**, about **104,000 windows** of 5 s (≈ 144 hours of recordings).
- About half the windows of the raw corpus are dropped by the organisers' pipeline because their hand-tracking labels are incomplete.
- The evaluation people are **never** used for training.

### The basic loop
One **training step** = take **256 random windows** (a "batch"), predict, measure the error, nudge the 6 million numbers to reduce it. We do **25,000 steps**, so each window is seen about 60 times.

- **Error measured (loss): L1**, the average absolute difference between predicted and true angles. It's the same thing as the competition metric, so we optimise exactly what we're scored on. Instants with a missing label are skipped.

### Augmentation: teaching the model to ignore what changes between people
Before each step we **randomly distort** the batch, in ways that mimic the differences between people. The true hand pose stays the same, so the model learns that these differences don't matter.

| Distortion | What we do | What it imitates |
|---|---|---|
| **Band rotation** | Half the time, turn the ring of electrodes by one position (left, right or none) | The band worn slightly rotated |
| **Gain per electrode** | Multiply each electrode by a random factor, typically ±15% | Better or worse skin contact on one electrode |
| **Global gain** | Multiply everything by a random factor, typically ±25% | Thicker skin or fat, stronger or weaker muscles |
| **Noise** | Add random noise, 5% of the signal's size | Electrical interference, sensor noise |

### The optimiser and its schedule
- **AdamW**: the algorithm that decides how to nudge the numbers. "W" adds a mild pull of every number towards zero (**weight decay**), which discourages over-complicated solutions that memorise the training people.
- **Learning rate** = the size of each nudge. It is **not constant**:
  1. **Warm-up** (first 1,000 steps): starts tiny and grows to 0.001. Big steps at the start, when the model knows nothing, can make training unstable.
  2. **Cosine decay** (the rest): slowly shrinks to zero, like braking smoothly. Big steps early to learn fast, tiny steps at the end to fine-tune.
- **Gradient clipping**: if one batch asks for a huge nudge (a weird batch), we cap it. It's a safety belt.
- **Dropout (10%)**: during training, randomly switch off 10% of some internal signals, so the model can't rely on any single detail. Another guard against memorising.

### EMA: averaging the model over time
Besides the model being trained, we keep a **slowly moving average of its weights** (EMA, "exponential moving average"), roughly an average of the last ~1,000 steps. This averaged model is smoother and usually generalises better than the last snapshot. **The averaged model is the one we save and submit.**

### Speed trick: mixed precision on the encoder only
The encoder computes in **half precision** (16-bit numbers instead of 32-bit), about twice as fast on modern GPUs. The decoder stays in full precision: since it adds up speeds over 250 steps, small rounding errors would pile up and make the hand "drift".

### Reproducibility
Everything is driven by a **random seed** (0). Run the same training with the same seed and you get the same model. This matters because the organisers **re-train the top submissions themselves**: on one V100 GPU in at most 12 hours, and the result must match.

---

## 4. At prediction time: rotation averaging (TTA)

When predicting, we feed the model **three versions** of each window: as is, with the band turned one electrode left, and one electrode right. Then we **average the three predictions**. If the person wears the band a little rotated, one of the three versions matches what the model saw in training better. All three run in a single pass, so it costs little extra time.

We measured it: **TTA improves the score by about 0.45°**.

---

## 5. All the settings

These are the defaults in `parameters` of `vemg2pose_plus.py`. The organisers re-train with exactly these values.

| Setting | Value | In plain words |
|---|---|---|
| `max_steps` | 25,000 | Number of training steps. About 3.3 h on an RTX 6000 Ada, chosen to stay under 12 h on the organisers' V100 |
| `batch_size` | 256 | Windows per training step |
| `lr` | 0.001 | Largest step size (learning rate), reached after warm-up |
| `warmup_steps` | 1,000 | Steps to ramp the learning rate up from almost zero |
| `weight_decay` | 0.01 | Strength of the pull towards small weights (AdamW) |
| `ema_decay` | 0.999 | How slowly the averaged model moves (≈ average of the last 1,000 steps) |
| `drop_prob` | 0.1 | Dropout: 10% of internal signals switched off during training |
| `max_shift` | 1 | Band rotation augmentation: up to ±1 electrode |
| `tta` | True | Average predictions over 3 band rotations at test time |
| `n_models` | 1 | Number of models trained with different seeds and averaged (an ensemble). Each one trains for the full `max_steps` |
| `max_hours` | 11 | Emergency stop: training ends after 11 h even if not finished |
| `seed` | 0 | Starting point of all the randomness, for reproducibility |

Fixed inside the code (not in `parameters`): gain jitter ±15% per electrode and ±25% global, noise 5%, rotation applied to 50% of windows, gradient clipping at 1.0.

---

## 6. What the experiments have shown so far

Public test set: 20 new people. Score = average error in degrees, lower is better.

| Model | Error | Lesson |
|---|---|---|
| Always predict the average pose | 19.27° | The floor: any useful model must beat this |
| Linear regression on raw EMG | 19.27° | Raw EMG, instant by instant, carries no linear pose information |
| Our model, **8 people**, 1,000 steps | 19.96° | **Worse than the floor**: it memorised 8 hands and failed on new ones |
| Same, without rotation averaging | 20.41° | Rotation averaging (TTA) helps by about 0.45° |
| Our model, **158 people**, 3,000 steps | **17.11°** | With many people it beats the floor clearly, even with short training |

The jump from 8 to 158 people is the clearest lesson: **for new users, the number of training people matters more than anything else.** The full run (25,000 steps on 158 people) is the next result.

---

## 7. Mini glossary

- **EMG**: electrical activity of muscles, measured on the skin.
- **Parameter / weight**: one of the numbers the network learns.
- **Batch**: the group of examples used for one training step.
- **Step / epoch**: a step is one update; an epoch is one full pass over all training windows.
- **Loss**: the error the training tries to reduce.
- **Learning rate**: how big each update is.
- **Convolution**: a small learned filter that slides along the signal.
- **LSTM**: a neural network with memory, good for sequences.
- **Overfitting**: memorising the training examples instead of learning the general rule.
- **Augmentation**: random but realistic distortions of the training data.
- **EMA**: a running average of the model's weights over the last steps.
- **TTA (test-time augmentation)**: predicting on several distorted copies of the input and averaging.
