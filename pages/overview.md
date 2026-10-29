# EMG2Pose: hand pose from wrist EMG

<p align="center">
  <a href="https://neural-interfaces26.github.io/" target="_blank" rel="noopener noreferrer" title="NeurIPS 2026 neural interfaces competition"><img src="https://neural-interfaces26.github.io/exports/competition-trophy.png" alt="NeurIPS 2026 neural interfaces competition" height="120"></a>
</p>

A wristband records the electrical activity of the forearm muscles. From 16
channels of surface EMG, predict the 20 joint angles of the hand over time:
a building block of neural interfaces for AR/VR and assistive devices.

<p align="center">
  <img src="https://neural-interfaces26.github.io/exports/emg-to-pose.gif" alt="Wrist surface EMG decoded into continuous hand-pose trajectories" width="640" style="display:block;max-width:100%;height:auto;margin:1.25rem auto;">
</p>

The **Participation** page describes the data, the metric and how to submit,
and the **Timeline** page gives the dates.

## Going further: the NeurIPS 2026 competition

This challenge is adapted from **Track 04 (EMG-to-Pose)** of the
[NeurIPS 2026 neural interfaces competition](https://www.codabench.org/competitions/17984/).
Its sealed final phase, from **5 to 21 November 2026**, ranks models on a
private cohort of new users recorded by Meta Reality Labs. If you want to see
how your model does there, you are welcome to submit it: follow the
**Registration Guide** and **Submission Guide** of
[Track 04](https://www.codabench.org/competitions/17984/). Its submission
contract is close to this one, but check its guide before uploading.

## Partners

<a href="https://about.meta.com/reality-labs/" target="_blank" rel="noopener noreferrer" title="Meta Reality Labs"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/meta_reality.png" alt="Meta Reality Labs" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.imperial.ac.uk/" target="_blank" rel="noopener noreferrer" title="Imperial College London"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/imperial.png" alt="Imperial College London" height="34" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.yneuro.com/" target="_blank" rel="noopener noreferrer" title="Yneuro"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/yneuro.png" alt="Yneuro" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://www.inria.fr/" target="_blank" rel="noopener noreferrer" title="Inria"><img src="https://neural-interfaces26.github.io/assets/img/logos/trim/inria.png" alt="Inria" height="28" style="vertical-align:middle;"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://sccn.ucsd.edu/" target="_blank" rel="noopener noreferrer" title="UC San Diego"><img src="https://neural-interfaces26.github.io/assets/img/logos/ucsd.svg?v=2" alt="UC San Diego" height="28" style="vertical-align:middle;"></a>

**Meta Reality Labs** sponsors the NeurIPS track and co-leads its scientific
development, with **Imperial College London**. **Yneuro**, **Inria** and
**UC San Diego** organize the NeurIPS competition.

## Acknowledgments

Many thanks to the organizers of the
[NeurIPS 2026 neural interfaces competition](https://neural-interfaces26.github.io/),
and to the authors of the [emg2pose](https://github.com/facebookresearch/emg2pose)
dataset (Salter et al., 2024), which this challenge uses. Thanks also to
**Lisheng Sun** (Université Paris-Saclay) for preparing and testing the
challenges.
