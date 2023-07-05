import matplotlib.pyplot as plt
import numpy as np
import tests.examples as examples
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_constraint_opt_graph_3D(results):

    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)

    tupleList = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    poly3d = [tupleList]
    poly_3d_collection = Poly3DCollection(poly3d, color="pink", alpha=0.5)
    ax.add_collection3d(poly_3d_collection)

    x = np.array(results[0]['x_k'])
    x1, x2, x3 = np.split(x, 3, axis=1)

    x1 = x1.T[0]
    x2 = x2.T[0]
    x3 = x3.T[0]

    ax.plot(x1, x2, x3, label='func_qp Iterations Path', color='blue', linestyle="--")
    ax.scatter(x1[-1], x2[-1], x3[-1], color="blue")

    ax.set_title("func_qp: Feasible Region & Iterations Path", loc='center')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    f_at_x_values = results[0]['f_at_x_k']
    x_vector = results[0]['x_k'][-1]
    x = x_vector[0]
    y = x_vector[1]
    z = x_vector[2]
    txt = '(f=' + str(round(f_at_x_values[-1], 2)) + ', \nx=' + str(round(x, 3)) + ', \ny=' + str(
        round(y, 3)) + ', \nz=' + str(round(z, 3)) + ')'
    ax.text(x + 0.05, y, z + 0.05,'%s' % (str(txt)), color='black')

    plt.legend()
    plt.show()

    fig, ax2 = plt.subplots(1, 1, figsize=(6, 5))

    ax2.set_xlabel('outer iteration')
    ax2.set_ylabel('f_objective(x)')
    ax2.set_title('func_qp: f_objective(x) v.s. Iteration number')

    legend = []
    for r in results:

        legend.append("Newton")

        if legend[-1] == 'Newton':
            c = 'blue'
        elif legend[-1] == 'GD':
            c = 'red'
        elif legend[-1] == 'BFGS':
            c = 'purple'
        elif legend[-1] == 'SR1':
            c = 'green'

        f_at_x_values = r['f_at_x_k']
        ax2.plot(f_at_x_values, color=c)

    ax2.legend(legend)

    plt.show()


def generate_constraint_opt_graph_2D(function_name, results, all_ineq_constraints, limits):

    f = getattr(examples, function_name)

    min_x1 = limits[0][0]
    max_x1 = limits[0][1]
    min_x2 = limits[1][0]
    max_x2 = limits[1][1]

    x_1_values = [x_1 for x_1 in np.arange(int(min_x1), int(max_x1),0.01)]
    x_2_values = [x_2 for x_2 in np.arange(int(min_x2), int(max_x2),0.01)]
    x_1_size = len(x_1_values)
    x_2_size = len(x_2_values)

    z = np.zeros([x_1_size, x_2_size])

    row = 0
    column = 0
    for x_1 in x_1_values:
        for x_2 in x_2_values:
            x_val = np.array((x_1, x_2))
            z[row][column] = f(x_val)[0]
            if column == x_1_size - 1:
                row += 1
                column = 0
            else:
                column += 1

    f = lambda x, y, func: func(np.array([x,y]))

    X1, X2 = np.meshgrid(x_1_values, x_2_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(function_name, fontsize=16)
    ax1.contour(X1, X2, z, levels=60)

    binary = None
    for ineq_constraint in all_ineq_constraints:
        if binary is None:
            f_val, _, _ = f(X1, X2, ineq_constraint)
            binary = (f_val <= 0)
        else:
            f_val, _, _ = f(X1, X2, ineq_constraint)
            binary &= (f_val <= 0)

    ax1.imshow((binary).astype(int),
                    extent=(X1.min(), X1.max(), X2.min(), X2.max()), origin="lower", cmap="Accent")

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Feasible Region & Iterations Path')

    ax2.set_xlabel('outer iteration')
    ax2.set_ylabel('f_objective(x)')
    ax2.set_title('f_objective(x) v.s. Iteration number')

    legend = []
    for r in results:

        legend.append("Newton")

        if legend[-1] == 'Newton':
            c = 'blue'
        elif legend[-1] == 'GD':
            c = 'red'
        elif legend[-1] == 'BFGS':
            c = 'purple'
        elif legend[-1] == 'SR1':
            c = 'green'

        x = np.array(r['x_k'])
        x1, x2 = np.split(x, 2, axis=1)
        ax1.plot(x1, x2, color=c, linestyle='dashed')
        ax1.scatter(x1[-1], x2[-1], color=c)

        f_at_x_values = r['f_at_x_k']
        txt = '(f='+str(round(f_at_x_values[-1],2))+', x='+str(round(x1[-1][0],1))+', y='+str(round(x2[-1][0],1))+')'
        ax1.annotate(txt, (x1[-1] - 0.8, x2[-1] + 0.2), color='white')
        ax2.plot(f_at_x_values, color=c)

    ax1.legend(legend)
    ax2.legend(legend)

    plt.show()


def generate_graph(function_name, results, limits):

    example_names = {'func_3d_i':' - quadratic example where contour lines are circles', 'func_3d_ii':' - quadratic example where contour lines are axis aligned ellipses',
                     'func_3d_iii':' - quadratic example where contour lines are rotated ellipses', 'func_3e':' - Rosenbrock function, contour lines are banana shaped ellipses',
                     'func_3f':' - linear function, contour lines are straight lines', 'func_3g':' - contour lines look like smoothed corner triangles'}

    f = getattr(examples, function_name)

    min_x1 = limits[0][0]
    max_x1 = limits[0][1]
    min_x2 = limits[1][0]
    max_x2 = limits[1][1]

    x_1_values = [x_1 for x_1 in np.arange(int(min_x1), int(max_x1),0.1)]
    x_2_values = [x_2 for x_2 in np.arange(int(min_x2), int(max_x2),0.1)]
    x_1_size = len(x_1_values)
    x_2_size = len(x_2_values)

    z = np.zeros([x_1_size, x_2_size])

    row = 0
    column = 0
    for x_1 in x_1_values:
        for x_2 in x_2_values:
            x_val = np.array((x_1, x_2))
            z[row][column] = f(x_val)[0]
            if column == x_1_size - 1:
                row += 1
                column = 0
            else:
                column += 1

    X1, X2 = np.meshgrid(x_1_values, x_2_values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(function_name + example_names[function_name], fontsize=16)
    ax1.contour(X1, X2, z, levels=60)

    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Contour Lines & Iterations Path')

    ax2.set_xlabel('iteration')
    ax2.set_ylabel('f(x)')
    ax2.set_title('f(x) v.s. Iteration number')

    legend = []
    for r in results:

        legend.append(r['method'])

        if legend[-1] == 'Newton':
            c = 'blue'
        elif legend[-1] == 'GD':
            c = 'red'
        elif legend[-1] == 'BFGS':
            c = 'purple'
        elif legend[-1] == 'SR1':
            c = 'green'

        x = np.array(r['x_k'])
        x1, x2 = np.split(x, 2, axis=1)
        ax1.plot(x1, x2, color=c, linestyle='dashed')
        ax1.scatter(x1[-1], x2[-1], color=c)

        f_at_x_values = r['f_at_x_k']
        ax2.plot(f_at_x_values, color=c)

    ax1.legend(legend)
    ax2.legend(legend)

    plt.show()
