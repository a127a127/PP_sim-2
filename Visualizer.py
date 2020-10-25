import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from tqdm import tqdm

class MappingGraph:
    def __init__(self, hw_config):
        self.Router_num_x = hw_config.Router_num_x
        self.Router_num_y = hw_config.Router_num_y
        self.router_width = self.Router_num_x
        self.router_height = self.Router_num_y
        self.router_size = self.router_width * self.router_height

        self.PE_num_x = hw_config.PE_num_x
        self.PE_num_y = hw_config.PE_num_y
        assert hw_config.PE_num_x == 2
        assert hw_config.PE_num_y == 2
        self.pe_width = self.Router_num_x * self.PE_num_x
        self.pe_height = self.Router_num_y * self.PE_num_y

        assert hw_config.CU_num == 12
        self.CU_num = hw_config.CU_num
        self.CU_num_x = 4
        self.CU_num_y = 3

        self.cu_width = self.Router_num_x * self.PE_num_x * self.CU_num_x
        self.cu_height = self.Router_num_y * self.PE_num_y * self.CU_num_y
        self.cu_size = self.cu_width * self.cu_height
        self.cu_active_array = [0] * self.cu_size

        # Init matplotlib figure
        self.H = 8.64
        self.W = 11.52
        self.fig = None
        self.init_fig()

        self.node_size = int((self.W * 30 / self.cu_width) ** 2)


        # Init networkx graph for Router
        self.router_G = nx.grid_2d_graph(self.router_width, self.router_height)
        self.router_pos = {}

        p = []
        for i in range(0, self.router_width):
            for j in range(0, self.router_height):
                x = i*(self.CU_num_x+1)*self.PE_num_x + self.CU_num_x
                y = 1 + j*(self.CU_num_y+1)*self.PE_num_y + self.CU_num_y
                p.append([x,y])

        nodes = list(self.router_G.nodes)
        for i in range(0, len(nodes)):
            self.router_pos[nodes[i]] = p[i]


        # Init networkx graph for CU
        self.cu_G = nx.grid_2d_graph(self.cu_width, self.cu_height)
        self.cu_pos = {}

        nodes = list(self.cu_G.nodes)
        self.cu_G.remove_edges_from(self.cu_G.edges)
        #edges = list(self.G.edges)
        #G.add_edge((1,1), (2,2))

        p = []
        for i in range(0, self.cu_width):
            for j in range(0, self.cu_height):
                p.append([i + (i//self.CU_num_x),1 + j + (j//self.CU_num_y)])
        #for i in range(0, len(nodes)):
        #    G.nodes[nodes[i]]['pos'] = p[i]

        for i in range(0, len(nodes)):
            self.cu_pos[nodes[i]] = p[i]
        #cmap = plt.cm.get_cmap('rainbow')


        # Empty graph for text
        self.empty_G = nx.Graph()
        self.empty_node = (-1,-1)
        self.empty_G.add_node(self.empty_node)
        self.empty_pos = {self.empty_node: [0, 0]}

        # Weight mapping
        self.weight_mapping = self.allocate_cu_active_array()
        self.cu_color_default = None
        self.max_nlayer = 0

    def init_fig(self):
        if self.fig != None:
            plt.close(self.fig)
        self.fig = plt.figure(None, figsize=(self.W, self.H))

    def allocate_cu_active_array(self):
        return self.cu_active_array.copy()

    def position_idx_to_x_y(self, position_idx):
        x = (position_idx[1] * self.PE_num_x * self.CU_num_x) + (position_idx[3] * self.CU_num_x) + (position_idx[4] // self.CU_num_y)
        y = (position_idx[0] * self.PE_num_y * self.CU_num_y) + (position_idx[2] * self.CU_num_y) + (position_idx[4] % self.CU_num_y)
        return (x, y)

    def position_idx_to_idx(self, position_idx):
        (x, y) = self.position_idx_to_x_y(position_idx)
        return x*self.cu_height + y

    def set_mapping(self, idx, nlayer):
        nlayer = nlayer + 1
        assert (self.weight_mapping[idx] == 0 or self.weight_mapping[idx] == nlayer)
        self.weight_mapping[idx] = nlayer
        if nlayer > self.max_nlayer:
            self.max_nlayer = nlayer

    def ensure_cu_color_default(self):
        if self.cu_color_default == None:
            self.cu_color_default = self.allocate_cu_active_array()
            for k, v in enumerate(self.weight_mapping):
                if v == 0:
                    self.cu_color_default[k] = (0.5, 0.5, 0.5, 0.1)
                else:
                    v = ((v - 1) % 6)
                    if v == 0:
                        self.cu_color_default[k] = (1, 0.3, 0.3, 0.1)
                    elif v == 1:
                        self.cu_color_default[k] = (1, 1, 0.3, 0.1)
                    elif v == 2:
                        self.cu_color_default[k] = (0.3, 1, 0.3, 0.1)
                    elif v == 3:
                        self.cu_color_default[k] = (0.3, 1, 1, 0.1)
                    elif v == 4:
                        self.cu_color_default[k] = (0.3, 0.3, 1, 0.1)
                    elif v == 5:
                        self.cu_color_default[k] = (1, 0.3, 1, 0.1)

    def draw(self, text, filename, active_cu, active_router=None, active_router_edge=None):
        if np.count_nonzero(active_cu) == 0:
            return
        assert len(active_cu) == len(self.weight_mapping)

        # Draw cu graph
        self.ensure_cu_color_default()

        cu_color = self.cu_color_default.copy()
        for k, v in enumerate(active_cu):
            if v != 0:
                t = list(cu_color[k])
                t[3] = 0.8
                cu_color[k] = tuple(t)

        nx.draw(self.cu_G, self.cu_pos, node_color = cu_color, node_shape='s', node_size=self.node_size)

        # Draw PE
        current_axis = plt.gca()
        padding = 0.1
        for i in range(0, self.pe_width):
            for j in range(0, self.pe_height):
                x = (i*(self.CU_num_x+1)) - 0.5 - padding
                y = 1 + (j*(self.CU_num_y+1)) - 0.5 - padding
                current_axis.add_patch(
                    Rectangle((x, y),
                        width = self.CU_num_x + padding*2,
                        height = self.CU_num_y + padding*2,fill=False
                    )
                )

        # Draw Router
        if active_router == None:
            active_router = [0] * self.router_size
        router_color = [(0.2, 0, 0)] * self.router_size
        for k, v in enumerate(active_router):
            if v != 0:
                t = list(router_color[k])
                t[0] = 1
                router_color[k] = tuple(t)

        nx.draw(self.router_G, self.router_pos, node_color = router_color, node_shape='o', node_size=self.node_size)

        # Draw text
        nx.draw(self.empty_G, pos = self.empty_pos, node_color='none', node_shape='s', node_size=self.node_size)
        self.fig.text(.5, 0.03, text, ha='center', fontsize=15)

        plt.savefig(f"{filename}.png", format="PNG")

        self.init_fig()

class Visualizer:
    def weightMappingByCO(hw_config, Computation_order, filename):
        CU_num = hw_config.CU_num

        graph = MappingGraph(hw_config)

        node_color_rd = graph.allocate_cu_active_array()
        node_color_cu = graph.allocate_cu_active_array()
        node_color_agg = graph.allocate_cu_active_array()
        node_color_pe = graph.allocate_cu_active_array()
        node_color_act = graph.allocate_cu_active_array()
        node_color_pool = graph.allocate_cu_active_array()
        node_color_trans = graph.allocate_cu_active_array()

        have_weight_layer = 0
        nlayer_mapping = {}
        for event in tqdm(Computation_order):
            if event.event_type == 'cu_operation':
                if not event.nlayer in nlayer_mapping:
                    nlayer_mapping[event.nlayer] = have_weight_layer
                    have_weight_layer += 1
                idx = graph.position_idx_to_idx(event.position_idx)
                graph.set_mapping(idx, nlayer_mapping[event.nlayer])

        nlayer = 0
        def draw_all():
            graph.draw("Read from edram to input register", f"{filename}-{nlayer}-0-edram_rd_ir", node_color_rd)
            graph.draw("CU operation", f"{filename}-{nlayer}-1-cu_operation", node_color_cu)
            graph.draw("PE shift and add", f"{filename}-{nlayer}-2-pe_saa", node_color_pe)
            graph.draw("Transfer data to aggregator", f"{filename}-{nlayer}-3-trans_to_aggregator", node_color_agg)
            graph.draw("Activation", f"{filename}-{nlayer}-4-activation", node_color_act)
            graph.draw("Pooling", f"{filename}-{nlayer}-5-pooling", node_color_pool)
            graph.draw("Transfer data to next layer", f"{filename}-{nlayer}-6-trans", node_color_trans)

        for event in tqdm(Computation_order):
            if event.nlayer != nlayer:
                draw_all()
                nlayer = event.nlayer
                node_color_rd = graph.allocate_cu_active_array()
                node_color_cu = graph.allocate_cu_active_array()
                node_color_agg = graph.allocate_cu_active_array()
                node_color_pe = graph.allocate_cu_active_array()
                node_color_act = graph.allocate_cu_active_array()
                node_color_pool = graph.allocate_cu_active_array()
                node_color_trans = graph.allocate_cu_active_array()
            if event.event_type == 'edram_rd_ir':
                idx = graph.position_idx_to_idx(event.position_idx)
                assert (node_color_rd[idx] == 1 or node_color_rd[idx] == 0)
                node_color_rd[idx] = 1
            elif event.event_type == 'cu_operation':
                idx = graph.position_idx_to_idx(event.position_idx)
                assert (node_color_cu[idx] == 1 or node_color_cu[idx] == 0)
                node_color_cu[idx] = 1
            elif event.event_type == 'pe_saa':
                position_idx = list(event.position_idx)
                for i in range(CU_num):
                    idx = graph.position_idx_to_idx(position_idx + [i])
                    assert (node_color_pe[idx] == 1 or node_color_pe[idx] == 0)
                    node_color_pe[idx] = 1
            elif event.event_type == 'activation':
                position_idx = list(event.position_idx)
                for i in range(CU_num):
                    idx = graph.position_idx_to_idx(position_idx + [i])
                    assert (node_color_act[idx] == 1 or node_color_act[idx] == 0)
                    node_color_act[idx] = 1
            elif event.event_type == 'pooling':
                position_idx = list(event.position_idx)
                for i in range(CU_num):
                    idx = graph.position_idx_to_idx(position_idx + [i])
                    assert (node_color_pool[idx] == 1 or node_color_pool[idx] == 0)
                    node_color_pool[idx] = 1
            elif event.event_type == 'data_transfer':
                position_idx = event.position_idx
                position_idx_from = list(position_idx[0])
                position_idx_to = list(position_idx[1])
                if position_idx_from == position_idx_to:
                    continue
                is_agg = Computation_order[event.proceeding_event[0]].nlayer == nlayer
                if is_agg:
                    for i in range(CU_num):
                        idx_from = graph.position_idx_to_idx(position_idx_from + [i])
                        idx_to = graph.position_idx_to_idx(position_idx_to + [i])
                        #if nlayer == 4:
                        #print(nlayer, idx_from, idx_to)
                        #print(event)
                        assert node_color_agg[idx_from] != 2
                        #assert node_color_agg[idx_to] != 1
                        node_color_agg[idx_from] = 1
                        node_color_agg[idx_to] = 2
                else:
                    for i in range(CU_num):
                        idx_from = graph.position_idx_to_idx(position_idx_from + [i])
                        idx_to = graph.position_idx_to_idx(position_idx_to + [i])
                        #print(nlayer, idx_from, idx_to)
                        #print(event)
                        assert node_color_trans[idx_from] != 2
                        #assert node_color_trans[idx_to] != 1
                        node_color_trans[idx_from] = 1
                        node_color_trans[idx_to] = 2
        # draw last layer
        draw_all()

        graph.draw('Weight mapping', filename, graph.weight_mapping)
