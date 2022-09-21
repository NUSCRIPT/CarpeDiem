import collections
import colorsys

import matplotlib.pyplot
import matplotlib.path
import matplotlib.patches
import matplotlib.patheffects


def get_distinct_colors(n):
    """
    https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python/answer/Karthik-Kumar-Viswanathan
    """
    hue_partition = 1 / (n + 1)
    colors = [colorsys.hsv_to_rgb(hue_partition * value, 0.8, 0.5)
              for value in range(0, n)]
    return reversed(colors[::2] + colors[1::2])


def text_width(fig, ax, text, fontsize):
    text = ax.text(-100, 0, text, fontsize=fontsize)
    text_bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    text_bb = text_bb.transformed(fig.dpi_scale_trans.inverted())
    width = text_bb.width
    text.remove()
    return width


class Sankey:
    def __init__(self, df,
                 plot_width=8,
                 plot_height=8,
                 gap=0.12,
                 alpha=0.3,
                 fontsize='small',
                 order=None,
                 mapping=None,
                 tag=None,
                 title=None,
                 title_left=None,
                 title_right=None,
                 labels=True,
                 block_width=0.1,
                 block_fontsize=12,
                 flow_color_func=None,
                 colors=None,
                 ax=None
    ):
        self.df = df
        if ax:
            self.plot_width = ax.get_position().width * ax.figure.get_size_inches()[0]
            self.plot_height = ax.get_position().height * ax.figure.get_size_inches()[1]
        else:
            self.plot_width = plot_width
            self.plot_height = plot_height
        self.gap = gap
        self.block_width = block_width
        self.block_fontsize = block_fontsize
        self.alpha = alpha
        self.labels = labels
        self.fontsize = fontsize
        self.order = order
        self.flow_color_func = flow_color_func
        # self.tag = tag
        # self.map = mapping is not None
        # self.mapping = mapping
        self.mapping_colors = {
            'increase': '#1f721c',
            'decrease': '#ddc90f',
            'mistake': '#dd1616',
            'correct': '#dddddd',
            'novel': '#59a8d6',
        }
        # self.title = title
        # self.title_left = title_left
        # self.title_right = title_right

        # self.need_title = any(map(lambda x: x is not None, (title, title_left, title_right)))
        # if self.need_title:
        #     self.plot_height -= 0.5

        self.init_figure(ax)

        # self.flows = collections.Counter(zip(x, y))
        self.init_flows()
        self.init_nodes(order)

        self.init_widths()
        # inches per 1 item in x and y
        self.resolution = (plot_height - gap * (len(order) - 1)) / df.shape[0]
        if colors is not None:
            self.colors = colors
        else:
            self.colors = {
                name: colour
                for name, colour
                in zip(self.nodes[0].keys(),
                    get_distinct_colors(len(self.nodes[0])))
            }

        self.init_offsets()

    def init_figure(self, ax):
        if ax is None:
            self.fig = matplotlib.pyplot.figure()
            self.ax = matplotlib.pyplot.Axes(self.fig, [0, 0, 1, 1])
            self.fig.add_axes(self.ax)
        self.fig = ax.figure
        self.ax = ax

    def init_flows(self):
        self.flows = []
        n_cols = self.df.columns.size
        for i in range(n_cols - 1):
            x, y = self.df.iloc[:, i], self.df.iloc[:, i + 1]
            self.flows.append(collections.Counter(zip(x, y)))

    def init_nodes(self, order):
        self.nodes = []

        for i in range(self.df.columns.size):
            column = collections.OrderedDict()
            counts = self.df.iloc[:, i].value_counts()
            for item in order:
                if item in counts:
                    column[item] = counts[item]
                else:
                    column[item] = 0
            self.nodes.append(column)

        # left_nodes = {}
        # right_nodes = {}
        # left_offset = 0
        # for (left, right), flow in self.flows.items():
        #     if left in left_nodes:
        #         left_nodes[left] += flow
        #     else:
        #         left_nodes[left] = flow
        #     if right in right_nodes:
        #         node = right_nodes[right]
        #         node[0] += flow
        #         if flow > node[2]:
        #             node[1] = left
        #             node[2] = flow
        #     else:
        #         # total_flow, max_incoming_left, max_incoming_flow
        #         right_nodes[right] = [flow, left, flow]

        # self.left_nodes = collections.OrderedDict()
        # self.left_nodes_idx = {}
        # if left_order is None:
        #     key = lambda pair: -pair[1]
        # else:
        #     left_order = list(left_order)
        #     key = lambda pair: left_order.index(pair[0])

        # for name, flow in sorted(left_nodes.items(), key=key):
        #     self.left_nodes[name] = flow
        #     self.left_nodes_idx[name] = len(self.left_nodes_idx)

        # left_names = list(self.left_nodes.keys())
        # self.right_nodes = collections.OrderedDict()
        # self.right_nodes_idx = {}
        # for name, node in sorted(
        #     right_nodes.items(),
        #     key=lambda pair: (left_names.index(pair[1][1]), -pair[1][2])
        # ):
        #     self.right_nodes[name] = node[0]
        #     self.right_nodes_idx[name] = len(self.right_nodes_idx)

    def init_widths(self):
        self.left_stop = self.block_width
        self.right_stop = self.plot_width - self.block_width
        self.stops = []
        n_cols = self.df.columns.size
        self.flow_width = (self.plot_width - self.block_width * (n_cols - 2)) / (n_cols - 1)

        for i in range(1, n_cols):
            stop1 = (self.block_width * i
                     + self.flow_width * (i - 1) + self.flow_width * 7 / 20)
            stop2 = (self.block_width * i
                     + self.flow_width * (i - 1) + self.flow_width * 13 / 20)
            self.stops.append((stop1, stop2))

    def init_offsets(self):
        self.offsets = []
        # self.offsets_l = {}
        # self.offsets_r = {}

        for col in self.nodes:
            offset = 0
            offsets = collections.OrderedDict()
            for name, size in col.items():
                offsets[name] = offset
                offset += size * self.resolution + self.gap
            self.offsets.append(offsets)

    def draw_flow(self, x, left, right, flow, node_offsets_l, node_offsets_r):
        P = matplotlib.path.Path

        left_y = self.offsets[x][left] + node_offsets_l[left]
        right_y = self.offsets[x + 1][right] + node_offsets_r[right]

        flow *= self.resolution

        node_offsets_l[left] += flow
        node_offsets_r[right] += flow
        if self.flow_color_func is not None:
            # if self.color[x] == "left":
            #     if left == right:
            #         color = self.mapping_colors["correct"]
            #     elif left > right:
            #         color = self.mapping_colors["mistake"]
            #     else:
            #         color = self.mapping_colors["increase"]
            # else:
            #     if left == right:
            #         color = self.mapping_colors["correct"]
            #     elif left < right:
            #         color = self.mapping_colors["mistake"]
            #     else:
            #         color = self.mapping_colors["increase"]
            mapping = self.flow_color_func(left, right)
            color = self.mapping_colors[mapping]
            # color = self.colors[left if self.color[x] == "left" else right]
        else:
            color = self.colors[left]

        left_x = self.flow_width * x + self.block_width * (x + 1)
        right_x  = left_x + self.flow_width

        path_data = [
            (P.MOVETO, (left_x, -left_y)),
            (P.LINETO, (left_x, -left_y - flow)),
            (P.CURVE4, (self.stops[x][0], -left_y - flow)),
            (P.CURVE4, (self.stops[x][1], -right_y - flow)),
            (P.CURVE4, (right_x, -right_y - flow)),
            (P.LINETO, (right_x, -right_y)),
            (P.CURVE4, (self.stops[x][1], -right_y)),
            (P.CURVE4, (self.stops[x][0], -left_y)),
            (P.CURVE4, (left_x, -left_y)),
            (P.CLOSEPOLY, (left_x, -left_y)),
        ]
        codes, verts = zip(*path_data)
        path = P(verts, codes)
        patch = matplotlib.patches.PathPatch(
            path,
            facecolor=color,
            alpha=0.9 if flow < .02 else self.alpha,
            edgecolor='none',
        )
        self.ax.add_patch(patch)

    def draw_label(self, label, is_left):
        nodes = self.left_nodes if is_left else self.right_nodes
        offsets = self.offsets_l if is_left else self.offsets_r
        y = offsets[label] + nodes[label] * self.resolution / 2
        if self.need_title:
            y += 0.5

        self.ax.text(
            -.1 if is_left else self.right_stop + .1,
            -y,
            label,
            horizontalalignment='right' if is_left else 'left',
            verticalalignment='center',
            fontsize=self.fontsize,
        )

    def draw_node(self, x, y, size, name):
        if size <= 0:
            return
        y = -list(self.offsets[x].values())[y] - size * self.resolution
        x = self.flow_width * x + self.block_width * x
        color = self.colors[name]
        patch = matplotlib.patches.Rectangle(
            (x, y),
            width=self.block_width,
            height=size * self.resolution,
            facecolor=color,
            edgecolor='none',
        )
        self.ax.add_patch(patch)
        self.ax.text(
            x + self.block_width / 2,
            y + size * self.resolution / 2,
            name,
            color="black",
            va="center",
            ha="center",
            size=self.block_fontsize,
            path_effects=[
                matplotlib.patheffects.Stroke(linewidth=2, foreground="white"),
                matplotlib.patheffects.Normal()
            ]
        )

    def draw(self):
        for x, col in enumerate(self.nodes):
            for y, (name, size) in enumerate(col.items()):
                self.draw_node(x, y, size, name)

        for x, flows in enumerate(self.flows):
            node_offsets_l = collections.Counter()
            node_offsets_r = collections.Counter()

            for (left, right), flow in sorted(
                flows.items(),
                key=lambda x: (self.order.index(x[0][0]), self.order.index(x[0][1]))
            ):
                self.draw_flow(
                    x,
                    left,
                    right,
                    flow,
                    node_offsets_l,
                    node_offsets_r
                )

            # for name in self.left_nodes:
            #     self.draw_label(name, True)
            # for name in self.right_nodes:
            #     self.draw_label(name, False)
            # self.draw_titles()

        # self.ax.axis('equal')
        self.ax.set_ylim(
            -self.resolution * self.df.shape[0] - self.gap * (len(self.order) - 1),
            0
        )
        self.ax.set_xlim(
            0,
            self.block_width * self.df.shape[1] + self.flow_width * (self.df.shape[1] - 1)
        )
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        for k in self.ax.spines.keys():
            self.ax.spines[k].set_visible(False)
        # matplotlib.pyplot.axis('off')
        # self.fig.set_figheight(self.plot_height)
        # self.fig.set_figwidth(self.plot_width)
        # if self.tag:
        #     text_ax = self.fig.add_axes((0.02, 0.95, 0.05, 0.05), frame_on=False)
        #     text_ax.set_axis_off()
        #     matplotlib.pyplot.text(0, 0, self.tag, fontsize=30, transform=text_ax.transAxes)
        #matplotlib.pyplot.tight_layout()


def sankey(df, **kwargs):
    diag = Sankey(df, **kwargs)
    diag.draw()
    return diag.fig