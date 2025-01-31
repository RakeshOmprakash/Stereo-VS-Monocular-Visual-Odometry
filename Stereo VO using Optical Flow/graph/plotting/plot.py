import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column, layout, gridplot
from bokeh.models import Div, WheelZoomTool


def visualize_paths(groundtruthpath, predictedpath, html_tile="", title="Stereo Visual Odometry", file_out="plot.html"):
    output_file(file_out, title=html_tile)
    groundtruthpath = np.array(groundtruthpath)
    predictedpath = np.array(predictedpath)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

    gt_x, gt_y = groundtruthpath.T
    pred_x, pred_y = predictedpath.T
    xs = list(np.array([gt_x, pred_x]).T)
    ys = list(np.array([gt_y, pred_y]).T)

    diff = np.linalg.norm(groundtruthpath - predictedpath, axis=1)
    src = ColumnDataSource(data=dict(gtx=groundtruthpath[:, 0], gty=groundtruthpath[:, 1],
                                        px=predictedpath[:, 0], py=predictedpath[:, 1],
                                        diffx=np.arange(len(diff)), diffy=diff,
                                        disx=xs, disy=ys))

    figure1 = figure(title="Paths", tools=tools, match_aspect=True, width_policy="max", toolbar_location="above",
                  x_axis_label="x", y_axis_label="y")
    figure1.circle("gtx", "gty", source=src, color="blue", hover_fill_color="firebrick", legend_label="GT")
    figure1.line("gtx", "gty", source=src, color="blue", legend_label="GT")

    figure1.circle("px", "py", source=src, color="green", hover_fill_color="firebrick", legend_label="Pred")
    figure1.line("px", "py", source=src, color="green", legend_label="Pred")

    figure1.multi_line("disx", "disy", source=src, legend_label="Error", color="red", line_dash="dashed")
    figure1.legend.click_policy = "hide"

    figure2 = figure(title="Error", tools=tools, width_policy="max", toolbar_location="above",
                  x_axis_label="frame", y_axis_label="error")
    figure2.circle("diffx", "diffy", source=src, hover_fill_color="firebrick", legend_label="Error")
    figure2.line("diffx", "diffy", source=src, legend_label="Error")

    show(layout([Div(text=f"<h1>{title}</h1>"),
                 Div(text="<h2>Paths</h1>"),
                 [figure1, figure2],
                 ], sizing_mode='scale_width'))


def make_residual_plot(x, residual_init, residual_minimized):
    figure1 = figure(title="Initial residuals", x_range=[0, len(residual_init)], x_axis_label="residual", y_axis_label="")
    figure1.line(x, residual_init)

    change = np.abs(residual_minimized) - np.abs(residual_init)
    plot_data = ColumnDataSource(data={"x": x, "residual": residual_minimized, "change": change})
    tooltips = [
        ("change", "@change"),
    ]
    figure2 = figure(title="Optimized residuals", x_axis_label=figure1.xaxis.axis_label, y_axis_label=figure1.yaxis.axis_label,
                  x_range=figure1.x_range, y_range=figure1.y_range, tooltips=tooltips)
    figure2.line("x", "residual", source=plot_data)

    figure3 = figure(title="Change", x_axis_label=figure1.xaxis.axis_label, y_axis_label=figure1.yaxis.axis_label,
                  x_range=figure1.x_range, tooltips=tooltips)
    figure3.line("x", "change", source=plot_data)
    return figure1, figure2, figure3


def plot_residual_results(qs_small, small_residual_init, small_residual_minimized,
                          qs, residual_init, residual_minimized):
    output_file("plot.html", title="Bundle Adjustment")
    x = np.arange(2 * qs_small.shape[0])
    figure1, figure2, figure3 = make_residual_plot(x, small_residual_init, small_residual_minimized)

    x = np.arange(2 * qs.shape[0])
    figure4, figure5, figure6 = make_residual_plot(x, residual_init, residual_minimized)

    show(layout([Div(text="<h1>Bundle Adjustment exercises</h1>"),
                 Div(text="<h2>Bundle adjustment with reduced parameters</h1>"),
                 gridplot([[figure1, figure2, figure3]], toolbar_location='above'),
                 Div(text="<h2>Bundle adjustment with all parameters (with sparsity)</h1>"),
                 gridplot([[figure4, figure5, figure6]], toolbar_location='above')
                 ]))


def plot_sparsity(sparse_mat):
    fig, ax = plt.subplots(figsize=[20, 10])
    plt.title("Sparsity matrix")

    ax.spy(sparse_mat, aspect="auto", markersize=0.02)
    plt.xlabel("Parameters")
    plt.ylabel("Resudals")

    plt.show()
