"""Floor baseline: always predict the train-mean pose.

Any real model should beat it. It is also the smallest example of the
submission contract.
"""

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.baselines import ConstantPose


class Solver(CompetSolver):

    name = "MeanPose"

    def get_model(self, meta):
        return ConstantPose(meta["n_joints"])
