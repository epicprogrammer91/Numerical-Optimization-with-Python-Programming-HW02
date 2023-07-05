from src.unconstrained_min import LSM
import src.utils as utils
import numpy as np
import tests.examples as examples
import unittest

#all_examples = ['func_3d_i', 'func_3d_ii', 'func_3d_iii', 'func_3e', 'func_3f', 'func_3g']


class UnconstrainedMinTest(unittest.TestCase):

    def test_func_3g(self):
        print('-------------func_3g--------------')
        i = 0
        all_examples = ['func_3g']
        f = getattr(examples, all_examples[i])
        x0 = np.array([1., 1.])
        # max_iter = 100
        print('-------gradient_descent--------')
        if all_examples[i] == 'func_3e':
            x0 = np.array([-1., 2.])

            lsm = LSM(f, x0=x0, max_iter=10000)
            x, f_at_x, success = lsm.gradient_descent()

        else:
            lsm = LSM(f, x0=x0, max_iter=100)
            x, f_at_x, success = lsm.gradient_descent()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------newton--------')
        lsm2 = LSM(f, x0=x0)
        x, f_at_x, success = lsm2.newton()
        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------BFGS--------')
        lsm3 = LSM(f, x0=x0)
        x, f_at_x, success = lsm3.BFGS()
        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------SR1--------')
        lsm4 = LSM(f, x0=x0)
        x, f_at_x, success = lsm4.SR1()
        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        utils.generate_graph(all_examples[i], [lsm.results, lsm2.results, lsm3.results, lsm4.results],[[-4, 4], [-4, 4]])

    def test_func_3f(self):
        print('-------------func_3f--------------')
        i = 0
        all_examples = ['func_3f']
        f = getattr(examples, all_examples[i])
        x0 = np.array([1., 1.])
        # max_iter = 100
        print('-------gradient_descent--------')
        if all_examples[i] == 'func_3e':
            x0 = np.array([-1., 2.])

            lsm = LSM(f, x0=x0, max_iter=10000)
            x, f_at_x, success = lsm.gradient_descent()

        else:
            lsm = LSM(f, x0=x0, max_iter=100)
            x, f_at_x, success = lsm.gradient_descent()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------newton--------')
        lsm2 = LSM(f, x0=x0)
        x, f_at_x, success = lsm2.newton()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------BFGS--------')
        lsm3 = LSM(f, x0=x0)
        x, f_at_x, success = lsm3.BFGS()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------SR1--------')
        lsm4 = LSM(f, x0=x0)
        x, f_at_x, success = lsm4.SR1()
        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        utils.generate_graph(all_examples[i], [lsm.results, lsm2.results, lsm3.results, lsm4.results],[[-100, 10], [-100, 10]])

    def test_func_3e(self):
        print('-------------func_3e--------------')
        i = 0
        all_examples = ['func_3e']
        f = getattr(examples, all_examples[i])
        x0 = np.array([1., 1.])
        # max_iter = 100
        print('-------gradient_descent--------')
        if all_examples[i] == 'func_3e':
            x0 = np.array([-1., 2.])

            lsm = LSM(f, x0=x0, max_iter=10000)
            x, f_at_x, success = lsm.gradient_descent()

        else:
            lsm = LSM(f, x0=x0, max_iter=100)
            x, f_at_x, success = lsm.gradient_descent()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------newton--------')
        lsm2 = LSM(f, x0=x0)
        x, f_at_x, success = lsm2.newton()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------BFGS--------')
        lsm3 = LSM(f, x0=x0)
        x, f_at_x, success = lsm3.BFGS()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------SR1--------')
        lsm4 = LSM(f, x0=x0)
        x, f_at_x, success = lsm4.SR1()
        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        utils.generate_graph(all_examples[i], [lsm.results, lsm2.results, lsm3.results, lsm4.results],[[-8, 8], [-8, 8]])

    def test_func_3d_iii(self):
        print('-------------func_3d_iii--------------')
        i = 0
        all_examples = ['func_3d_iii']
        f = getattr(examples, all_examples[i])
        x0 = np.array([1., 1.])
        # max_iter = 100
        print('-------gradient_descent--------')
        if all_examples[i] == 'func_3e':
            x0 = np.array([-1., 2.])

            lsm = LSM(f, x0=x0, max_iter=10000)
            x, f_at_x, success = lsm.gradient_descent()

        else:
            lsm = LSM(f, x0=x0, max_iter=100)
            x, f_at_x, success = lsm.gradient_descent()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------newton--------')
        lsm2 = LSM(f, x0=x0)
        x, f_at_x, success = lsm2.newton()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------BFGS--------')
        lsm3 = LSM(f, x0=x0)
        x, f_at_x, success = lsm3.BFGS()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------SR1--------')
        lsm4 = LSM(f, x0=x0)
        x, f_at_x, success = lsm4.SR1()
        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        utils.generate_graph(all_examples[i], [lsm.results, lsm2.results, lsm3.results, lsm4.results],[[-3, 3], [-3, 3]])

    def test_func_3d_ii(self):
        print('-------------func_3d_ii--------------')
        i = 0
        all_examples = ['func_3d_ii']
        f = getattr(examples, all_examples[i])
        x0 = np.array([1., 1.])
        # max_iter = 100
        print('-------gradient_descent--------')
        if all_examples[i] == 'func_3e':
            x0 = np.array([-1., 2.])

            lsm = LSM(f, x0=x0, max_iter=10000)
            x, f_at_x, success = lsm.gradient_descent()

        else:
            lsm = LSM(f, x0=x0, max_iter=100)
            x, f_at_x, success = lsm.gradient_descent()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------newton--------')
        lsm2 = LSM(f, x0=x0)
        x, f_at_x, success = lsm2.newton()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------BFGS--------')
        lsm3 = LSM(f, x0=x0)
        x, f_at_x, success = lsm3.BFGS()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------SR1--------')
        lsm4 = LSM(f, x0=x0)
        x, f_at_x, success = lsm4.SR1()
        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        utils.generate_graph(all_examples[i], [lsm.results, lsm2.results, lsm3.results, lsm4.results],[[-3, 3], [-3, 3]])

    def test_func_3d_i(self):
        print('-------------func_3d_i--------------')
        i = 0
        all_examples = ['func_3d_i']
        f = getattr(examples, all_examples[i])
        x0 = np.array([1., 1.])
        # max_iter = 100
        print('-------gradient_descent--------')
        if all_examples[i] == 'func_3e':
            x0 = np.array([-1., 2.])

            lsm = LSM(f, x0=x0, max_iter=10000)
            x, f_at_x, success = lsm.gradient_descent()

        else:
            lsm = LSM(f, x0=x0, max_iter=100)
            x, f_at_x, success = lsm.gradient_descent()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------newton--------')
        lsm2 = LSM(f, x0=x0)
        x, f_at_x, success = lsm2.newton()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------BFGS--------')
        lsm3 = LSM(f, x0=x0)
        x, f_at_x, success = lsm3.BFGS()

        print('final iteration x value = ', x, ';\tfinal iteration f_at_x value = ', f_at_x,
              ';\tsuccess/failure algorithm output flag = ', success)

        print('-------SR1--------')
        lsm4 = LSM(f, x0=x0)
        x, f_at_x, success = lsm4.SR1()
        print('final iteration x value = ',x, ';\tfinal iteration f_at_x value = ',f_at_x, ';\tsuccess/failure algorithm output flag = ',success)

        utils.generate_graph(all_examples[i], [lsm.results, lsm2.results, lsm3.results, lsm4.results],[[-3, 3], [-3, 3]])




if __name__ == '__main__':
    unittest.main()





