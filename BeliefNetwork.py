import pandas as pd  # for data manipulation
import networkx as nx  # for drawing graphs
import matplotlib.pyplot as plt  # for drawing graphs

# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController


class BeliefBayesianNetwork:
    def __init__(self):
        self.join_tree = None
        self.bbn = None
        self.df = pd.read_csv('HealtRisk.csv')
        self.df = self.df.dropna()
        self.generate(self.df)

    def generate(self, df):

        fat = BbnNode(Variable(0, "Fat", ['low', 'medium', 'high']), self.probs(df, 'Fat'))
        badlifeStyle = BbnNode(Variable(1, "Life", ['bad', 'good']), self.probs(df, 'LifeStyle'))
        young = BbnNode(Variable(2, "Young", ['<30', '>=30']), self.probs(df, 'Young'))
        smoke = BbnNode(Variable(3, "Smoke", ['no', 'low', 'much']), self.probs(df, 'Smoke'))
        pollution = BbnNode(Variable(4, "Pollution", ['village', 'city']), self.probs(df, 'Pollution'))
        body = BbnNode(Variable(5, "Body", ['good', 'bad']), self.probs(df, 'Body', 'Fat', 'LifeStyle', 'Smoke'))
        diabetes = BbnNode(Variable(6, "Diabetes", ['yes', 'no']), self.probs(df, 'Diabetes', 'LifeStyle', 'Young'))
        external = BbnNode(Variable(7, "External", ['yes', 'no']), self.probs(df, 'External', 'Smoke', 'Pollution'))
        generalDiases = BbnNode(Variable(8, "Diases", ['yes', 'no']), self.probs(df, 'Diases', 'Diabetes', 'External'))
        risk = BbnNode(Variable(9, "Risk", ['low', 'medium', 'high']), self.probs(df, 'Risk', 'Body', 'Diases'))

        self.bbn = Bbn() \
            .add_node(fat) \
            .add_node(badlifeStyle) \
            .add_node(young) \
            .add_node(smoke) \
            .add_node(pollution) \
            .add_node(body) \
            .add_node(diabetes) \
            .add_node(external) \
            .add_node(generalDiases) \
            .add_node(risk) \
            .add_edge(Edge(fat, body, EdgeType)) \
            .add_edge(Edge(badlifeStyle, body, EdgeType.DIRECTED)) \
            .add_edge(Edge(smoke, body, EdgeType.DIRECTED)) \
            .add_edge(Edge(badlifeStyle, diabetes, EdgeType.DIRECTED)) \
            .add_edge(Edge(young, diabetes, EdgeType.DIRECTED)) \
            .add_edge(Edge(smoke, external, EdgeType.DIRECTED)) \
            .add_edge(Edge(pollution, external, EdgeType.DIRECTED)) \
            .add_edge(Edge(diabetes, generalDiases, EdgeType.DIRECTED)) \
            .add_edge(Edge(external, generalDiases, EdgeType.DIRECTED)) \
            .add_edge(Edge(generalDiases, risk, EdgeType.DIRECTED)) \
            .add_edge(Edge(body, risk, EdgeType.DIRECTED))

        '''self.bbn = Bbn()\
            .add_node(fat)\
            .add_node(badlifeStyle)\
            .add_node(young)\
            .add_node(smoke)\
            .add_node(pollution)\
            .add_node(body)\
            .add_node(diabetes)\
            .add_node(external)\
            .add_node(generalDiases)\
            .add_node(risk)\
            .add_edge(Edge(fat, body, EdgeType.DIRECTED))\
            .add_edge(Edge(badlifeStyle, body, EdgeType.DIRECTED))\
            .add_edge(Edge(smoke, body, EdgeType.DIRECTED))\
            .add_edge(Edge(badlifeStyle, diabetes, EdgeType.DIRECTED))\
            .add_edge(Edge(young, diabetes, EdgeType.DIRECTED))\
            .add_edge(Edge(smoke, external, EdgeType.DIRECTED))\
            .add_edge(Edge(pollution, external, EdgeType.DIRECTED))\
            .add_edge(Edge(diabetes, generalDiases, EdgeType.DIRECTED))\
            .add_edge(Edge(external, generalDiases, EdgeType.DIRECTED))\
            .add_edge(Edge(generalDiases, risk, EdgeType.DIRECTED))\
            .add_edge(Edge(body, risk, EdgeType.DIRECTED))'''

        print(self.bbn.nodes)
        print(self.bbn.edge_map)
        self.join_tree = InferenceController.apply(self.bbn)

    @staticmethod
    def probs(data, child, parent1=None, parent2=None, parent3=None):
        if parent1 is None:
            # Calculate probabilities
            prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
        else:
            # Check if child node has 1 parent or 2 parents
            if parent2 is None:
                # Caclucate probabilities
                prob = pd.crosstab(data[parent1], data[child], margins=False,
                                   normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:
                if parent3 is None:
                    # Caclucate probabilities
                    prob = pd.crosstab([data[parent1], data[parent2]], data[child], margins=False,
                                       normalize='index').sort_index().to_numpy().reshape(-1).tolist()
                else:
                    prob = pd.crosstab([data[parent1], data[parent2], data[parent3]], data[child], margins=False,
                                       normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        return prob

    def show(self):
        pos = {0: (-1, 2.5),
               1: (-1, 1.75),
               2: (-1, 1),
               3: (-1, .25),
               4: (-1, -0.5),
               5: (-.6, 2),
               6: (-.6, .75),
               7: (-.6, -.25),
               8: (-.3, .25),
               9: (0, 1)
               }
        options = {
            "font_size": 12,
            "node_size": 3000,
            "node_color": "white",
            "edgecolors": "blue",
            "edge_color": "blue",
            "linewidths": 2,
            "node_shape": 'o',
            "width": 2}
        # Generate graph
        n, d = self.bbn.to_nx_graph()
        nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

        # Update margins and print the graph
        ax = plt.gca()
        ax.margins(0.10)
        plt.axis("off")
        #plt.show()

        self.print_probs()

    # Define a function for printing marginal probabilities
    def print_probs(self):
        for node in self.join_tree.get_bbn_nodes():
            potential = self.join_tree.get_bbn_potential(node)
            print("Node:", node)
            print("Values:")
            print(potential)
            print('----------------')

    @staticmethod
    def insertDefinedValue(tree, nod, cat, value):
        ev = EvidenceBuilder() \
            .with_node(tree.get_bbn_node_by_name(nod)) \
            .with_evidence(cat, value) \
            .build()
        tree.set_observation(ev)

    @staticmethod
    def getFinalProb(tree, nodeName):
        node = tree.get_bbn_node_by_name(nodeName)
        potential = tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')

    def query(self, fat, life, young, smoke, pollution):
        tree = self.join_tree.__copy__()

        self.insertDefinedValue(tree, 'Fat', fat, 1)
        self.insertDefinedValue(tree, 'Life', life, 1)
        self.insertDefinedValue(tree, 'Young', young, 1)
        self.insertDefinedValue(tree, 'Smoke', smoke, 1)
        self.insertDefinedValue(tree, 'Pollution', pollution, 1)
        self.getFinalProb(tree, 'Risk')


bbn = BeliefBayesianNetwork()
bbn.show()
# bbn.query('medium', 'bad', '<30', 'no', 'city')
