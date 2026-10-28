# Terms and Conditions

This competition is run for educational purposes as part of a data-science
course. By participating you agree to the following.

1. **Submissions must be your own work.** You may use public libraries and
   pretrained models that were not trained on EMG data, but the training and
   modelling code you submit must be authored by your team.
2. **No attempt to access the hidden data.** The private test set is used
   only for the final ranking. Any attempt to read, exfiltrate or
   reverse-engineer it disqualifies the submission.
3. **Fair use of the platform.** Respect the per-day submission limit and the
   execution-time budget. Do not attempt to interfere with other participants'
   runs or the evaluation infrastructure.
4. **Reproducibility.** Your `submission.py` must retrain your submitted
   weights through its `fit`, with `benchopt run` on the `EMG2Pose` dataset,
   in at most 12 hours on one 32 GB V100 GPU. After the competition, the
   organizers retrain the top-ranked submissions this way. A submission
   whose retrained score is clearly worse than its leaderboard score, beyond
   run-to-run noise, is removed from the ranking.
5. **Data usage.** Train only on the training users of the `EMG2Pose`
   dataset. Using its test users, which feed the public column, or any other
   EMG data, including the rest of the public emg2pose corpus, is not
   allowed.

The organizers may adjust these rules and will announce any change on the
competition pages.
