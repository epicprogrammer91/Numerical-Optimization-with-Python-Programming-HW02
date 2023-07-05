import unittest
import tests.examples as examples
import src.utils as utils
import numpy as np
from src.constrained_min import ConstrainedMin

class ConstrainedMinTest(unittest.TestCase):

    def test_qp(self):
        print('-------------test_qp--------------')
        x0 = np.array([0.1, 0.2, 0.7])
        all_ineq_constraints = [examples.qp__ineq_constraint_1, examples.qp__ineq_constraint_2,
                                examples.qp__ineq_constraint_3]
        constrained_min = ConstrainedMin()
        x, f_at_x = constrained_min.interior_pt(examples.func_qp, all_ineq_constraints,
                                                examples.eq_constraints_mat_qp(), examples.eq_constraints_rhs_qp(), x0)

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x)

        utils.generate_constraint_opt_graph_3D([constrained_min.results])

    def test_lp(self):
        print('-------------test_lp--------------')
        x0 = np.array([0.5, 0.75])
        all_ineq_constraints = [examples.lp__ineq_constraint_1, examples.lp__ineq_constraint_2,
                                examples.lp__ineq_constraint_3, examples.lp__ineq_constraint_4]
        constrained_min = ConstrainedMin()
        x, f_at_x = constrained_min.interior_pt(examples.func_lp, all_ineq_constraints,
                                                examples.eq_constraints_mat_lp(), examples.eq_constraints_rhs_lp(), x0)

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x)

        utils.generate_constraint_opt_graph_2D("func_lp", [constrained_min.results], all_ineq_constraints,
                             [[-1, 3], [-1, 3]])


if __name__ == '__main__':
    unittest.main()