"""Beta sizing belongs to the retained pair method, not independent lists."""
import numpy as np
from backtest import ExecutionModel as CostModel


class ExecutionModel(CostModel):
    @staticmethod
    def size_pair(open_y, open_x, beta, leg_cap, direction, unit_steps=(1, 1)):
        """Apply one common cap, then round down to whole shares/FX lots."""
        step_y, step_x = unit_steps
        qy = int(np.floor(min(leg_cap / open_y, leg_cap / (beta * open_x)) / step_y)) * step_y
        qx = int(np.floor(beta * qy / step_x)) * step_x
        return direction * qy, -direction * qx
