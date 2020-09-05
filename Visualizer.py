import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm

class Visualizer:
    def weightMappingByCO(hw_config, Computation_order, filename):
        assert hw_config.CU_num == 12
        CU_num_x = 4
        CU_num_y = 3

        Router_num_x = hw_config.Router_num_x
        Router_num_y = hw_config.Router_num_y
        PE_num_x = hw_config.PE_num_x
        PE_num_y = hw_config.PE_num_y

        width = Router_num_x * PE_num_x * CU_num_x
        height = Router_num_y * PE_num_y * CU_num_y

        G = nx.grid_2d_graph(width, height)
        nodes = list(G.nodes)
        edges = list(G.edges)

        p = []
        for i in range(0, width):
            for j in range(0, height):
                p.append([i,j])
        #for i in range(0, len(nodes)):
        #    G.nodes[nodes[i]]['pos'] = p[i]

        pos = {}
        for i in range(0, len(nodes)):
            pos[nodes[i]] = p[i]

        node_color = [0]*width*height

        def position_idx_to_idx(position_idx):
            x = position_idx[1] * PE_num_x * CU_num_x + position_idx[3] * CU_num_x + (position_idx[4] // CU_num_y)
            y = position_idx[0] * PE_num_y * CU_num_y + position_idx[2] * CU_num_y + (position_idx[4] % CU_num_y)
            return x*height + y

        for event in tqdm(Computation_order):
            if event.event_type == 'cu_operation':
                node_color[position_idx_to_idx(event.position_idx)] = (event.nlayer % 6)+1

        cmap = plt.cm.get_cmap('rainbow')
        nx.draw(G, pos, node_color = node_color, cmap = cmap, node_size=100)
        plt.savefig(filename, format="PNG")
