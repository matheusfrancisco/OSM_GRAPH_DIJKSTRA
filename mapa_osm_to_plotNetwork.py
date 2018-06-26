"""
# Universidade Federal de Santa Catarina
## Trabalho de Tópicos Especiais III
### Nome: Matheus Francisco Batista Machado

"""

import xml.sax
import copy
import networkx as nx
import sys
from matplotlib.lines import Line2D
from matplotlib.widgets import Cursor, Button, CheckButtons
import numpy as np
import math
import calculo_distancia
import pylab as plt


## Node vai armazenar o id, long, latitude
class Node:
    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat
        self.tags = {}

    def __str__(self):
        return str(self.id)

    __repr__ = __str__

## O caminho tem osm o id 
class Way:
    def __init__(self, id, osm):
        self.osm = osm
        self.id = id
        self.nds = []
        self.tags = {}

    def split(self, dividers):

        ## Vai dividir a lista de no usando uma função recursiva
        def slice_array(ar, dividers):
            for i in range(1,len(ar)-1):
                if dividers[ar[i]]>1:
                    
                    left = ar[:i+1]
                    right = ar[i:]

                    rightsliced = slice_array(right, dividers)

                    return [left]+rightsliced
            return [ar]

        slices = slice_array(self.nds, dividers)

        # create a way object for each node-array slice
        # Cria um caminho  para cada node.
        ret = []
        i=0
        for slice in slices:
            littleway = copy.copy( self )
            littleway.id += "-%d"%i
            littleway.nds = slice
            ret.append( littleway )
            i += 1

        return ret


class Relation:
    def __init__(self, id, osm):
        self.osm = osm
        self.id = id
        self.nds = []
        self.tags = {}

## Classe OSM
##	
class OSM:
    def __init__(self, filename_or_stream):
        """ File can be either a filename or stream/file object."""
        nodes = {}
        ways = {}
        bounds = {}

        superself = self

        class OSMHandler(xml.sax.ContentHandler):
            @classmethod
            def setDocumentLocator(self,loc):
                pass

            @classmethod
            def startDocument(self):
                pass

            @classmethod
            def endDocument(self):
                pass

            @classmethod
            def startElement(self, name, attrs):
                if name=='node':
                    self.currElem = Node(attrs['id'], float(attrs['lon']), float(attrs['lat']))
                elif name=='way':
                    self.currElem = Way(attrs['id'], superself)
                elif name=='tag':
                    self.currElem.tags[attrs['k']] = attrs['v']
                elif name=='nd':
                    self.currElem.nds.append( attrs['ref'] )
                elif name == 'bounds':
                    bounds['maxlon'] = float(attrs['maxlon'])
                    bounds['minlon'] = float(attrs['minlon'])
                    bounds['maxlat'] = float(attrs['maxlat'])
                    bounds['minlat'] = float(attrs['minlat'])
                elif name == 'relation':
                    self.currElem = Relation(attrs['id'], superself)

            @classmethod
            def endElement(self,name):
                if name=='node':
                    nodes[self.currElem.id] = self.currElem
                elif name=='way':
                    ways[self.currElem.id] = self.currElem

            @classmethod
            def characters(self, chars):
                pass

        xml.sax.parse(filename_or_stream, OSMHandler)
        self.nodes = nodes
        self.ways = ways
        self.bounds = bounds

        #"""
        #count times each node is used
        node_histogram = dict.fromkeys( list(self.nodes.keys()), 0 )
        for way in list(self.ways.values()):
            if len(way.nds) < 2:       #if a way has only one node, delete it out of the osm collection
                del self.ways[way.id]
            else:
                for node in way.nds:
                    node_histogram[node] += 1

        new_ways = {}
        for id, way in self.ways.items():
            split_ways = way.split(node_histogram)
            for split_way in split_ways:
                new_ways[split_way.id] = split_way
        self.ways = new_ways

## 
def read_osm(filename_or_stream, only_roads=True):
    osm = OSM(filename_or_stream)
    G = nx.Graph()

    for w in osm.ways.values():
        if only_roads and 'highway' not in w.tags:
            continue

        actual_nodes = w.nds
        nds = actual_nodes
        nds1 = actual_nodes[1:]
        weights = [calculo_distancia.calc_distance(osm.nodes[edge[0]], osm.nodes[edge[1]]) for edge in zip(nds, nds1)]

        for i, edge in enumerate(zip(nds, nds1)):
            G.add_edge(edge[0], edge[1], weight=weights[i])

    return G, osm





## MatplotiLibMAP Onde ira desenhar o MAPA
class MatplotLibMap:
    ## Definindo os tipos de cores para as ruas
    renderingRules = {
        'primary': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           =  (0.933, 0.51, 0.933), 
                zorder          = 400,
        ),
        'primary_link': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = (0.85, 0.44, 0.84), 
                zorder          = 300,
        ),
        'secondary': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = (0.85, 0.75, 0.85), 
                zorder          = 200,
        ),
        'secondary_link': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = (0.85, 0.75, 0.85), 
                zorder          = 200,
        ),
        'tertiary': dict(
                linestyle       = '-',
                linewidth       = 4,
                color           = (1.0, 0.0, 0.0), 
                zorder          = 100,
        ),
        'tertiary_link': dict(
                linestyle       = '-',
                linewidth       = 4,
                color           = (1.0, 0.0, 0.0), 
                zorder          = 100,
        ),
        'residential': dict(
                linestyle       = '-',
                linewidth       = 1,
                color           = (1.0, 1.0, 0.0), 
                zorder          = 50,
        ),
        'unclassified': dict(
                linestyle       = ':',
                linewidth       = 1,
                color           = (0.5,0.5,0.5),
                zorder          = 10,
        ),
        'calculated_path': dict(
                linestyle       = '-',
                linewidth       = 4,
                color           = (1.0,0.0,0.0),
                zorder          = 2000,
        ),
        'correct_path': dict(
                linestyle       = '-',
                linewidth       = 6,
                color           = (0.6,0.8,0.0),
                zorder          = 1900,
        ),
        'default': dict(
                linestyle       = '-',
                linewidth       = 3,
                color           = (1.0, 0.48, 0.0),
                zorder          = 500,
                ),

        'other': dict(
                linestyle       = '-',
                linewidth       = 3,
                color           = (0.6, 0.6, 0.6),
                zorder          = 500,
                ),
        }

    #INIT Distancia entre dois nós do gráfico
    def __init__(self, osm, graph):
        self._node1 = None
        self._node2 = None
        self._mouse_click1 = None
        self._mouse_click2 = None
        self._node_map = {}
        self._graph = graph
        self._osm = None
        self._fig = None

        # Matplotlib membros de dados
        self._node_plots = []
        self._osm = osm
        #lisa de lat e lon
        self.setup_figure()

    @property
    def node1(self):
        return self._node1

    @property
    def node2(self):
        return self._node2

    def setup_figure(self):
        # Pega os  min long no lat
        minX = float(self._osm.bounds['minlon'])
        maxX = float(self._osm.bounds['maxlon'])
        minY = float(self._osm.bounds['minlat'])
        maxY = float(self._osm.bounds['maxlat'])

        if self._fig is not None:
            plt.close(self._fig)

        self._fig = plt.figure( figsize=(28,12), dpi=80, facecolor='grey', edgecolor='k')
        self._fig.canvas.set_window_title("Dijkstra Algoritmo OSM")

        self._render_axes0 = self._fig.add_subplot(231, autoscale_on = True, xlim = (minX,maxX), ylim = (minY,maxY))
        self._render_axes0.xaxis.set_visible(False)
        self._render_axes0.yaxis.set_visible(False)
        plt.title("Dijkstra")

        self._render_axes3 = self._fig.add_subplot(234, autoscale_on = True, xlim = (minX,maxX), ylim = (minY,maxY))
        self._render_axes3.xaxis.set_visible(False)
        self._render_axes3.yaxis.set_visible(False)
        plt.title("Resultado de Dijkstra")

        self._axes = {}
        self._axes['dijkstra'] = {}


        self._axes['dijkstra']['main'] = self._render_axes0
        self._axes['dijkstra']['paths_considered'] = self._render_axes3


        self._main_rendering_axes = [self._render_axes0]

        for rendering_axes in self._main_rendering_axes:
            self.render(rendering_axes)

        self._fig.tight_layout()
        plt.show()

    def _get_axes(self, algo, graph_type):
        return self._axes[algo][graph_type]

    def render(self, axes, plot_nodes=False):
        plt.sca(axes)
        for idx, nodeID in enumerate(self._osm.ways.keys()):
            wayTags = self._osm.ways[nodeID].tags
            wayType = None
            if 'highway' in wayTags.keys():
                wayType = wayTags['highway']

            if wayType in [
                           'primary',
                           'primary_link',
                           'unclassified',
                           'secondary',
                           'secondary_link',
                           'tertiary',
                           'tertiary_link',
                           'residential',
                           'trunk',
                           'trunk_link',
                           'motorway',
                           'motorway_link'
                            ]:
                oldX = None
                oldY = None
                
                if wayType in list(MatplotLibMap.renderingRules.keys()):
                    thisRendering = MatplotLibMap.renderingRules[wayType]
                else:
                    thisRendering = MatplotLibMap.renderingRules['default']

                for nCnt, nID in enumerate(self._osm.ways[nodeID].nds):
                    y = float(self._osm.nodes[nID].lat)
                    x = float(self._osm.nodes[nID].lon)

                    self._node_map[(x, y)] = nID

                    if oldX is None:
                        pass
                    else:
                        plt.plot([oldX,x],[oldY,y],
                                marker          = '',
                                linestyle       = thisRendering['linestyle'],
                                linewidth       = thisRendering['linewidth'],
                                color           = thisRendering['color'],
                                solid_capstyle  = 'round',
                                solid_joinstyle = 'round',
                                zorder          = thisRendering['zorder'],
                                picker=2
                        )

                        if plot_nodes == True and (nCnt == 0 or nCnt == len(self._osm.ways[nodeID].nds) - 1):
                            plt.plot(x, y,'ro', zorder=5)

                    oldX = x
                    oldY = y

        self._fig.canvas.mpl_connect('pick_event', self.__onclick__)
        plt.draw()

    def __clear_button_clicked__(self, event):
        self._node1 = None
        self._node2 = None
        self._mouse_click1 = None
        self._mouse_click2 = None
        self.render(self._osm, plot_nodes=False)

    def __onclick__(self, event):
        threshold = 0.001
        print("Mouse clicado")

        if self._node1 is not None and self._node2 is not None:
            return None

        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            point = (float(np.take(xdata, ind)[0]), float(np.take(ydata, ind)[0]))
            node_id = self._node_map[point]

            if self._node1 is None:
                self._node1 = Node(node_id, point[0], point[1])
                self._mouse_click1 = (event.mouseevent.xdata, event.mouseevent.ydata)

                for axes in self._main_rendering_axes:
                    plt.sca(axes)
                    plt.plot(self._mouse_click1[0], self._mouse_click1[1], 'ro', zorder=100)

                plt.draw()
                return self._node1
            else:
                if abs(point[0] - self._node1.lon) < threshold and abs(point[1] - self._node1.lat) < threshold:
                    return None

                self._node2 = Node(node_id, point[0], point[1])
                self._mouse_click2 = (event.mouseevent.xdata, event.mouseevent.ydata)
                print("Todos os pontos marcados")

                for axes in self._main_rendering_axes:
                    plt.sca(axes)
                    plt.plot(self._mouse_click2[0], self._mouse_click2[1], 'ro', zorder=100)

                plt.draw()

                path_dijkstra, paths_considered_dijkstra = calculo_distancia.dijkstra(self._graph, self._node1.id, self._node2.id)
                
                self.plot_path(self._get_axes('dijkstra', 'main'), path_dijkstra, MatplotLibMap.renderingRules['correct_path'], animate=False)
                self.plot_considered_paths(self._get_axes('dijkstra', 'paths_considered'), path_dijkstra, (paths_considered_dijkstra, 'green'))

                plt.savefig("mapa.png")
                return self._node2

    def plot_path(self, axes, path, rendering_style=None, animate=False):
        plt.sca(axes)
        edges = zip(path, path[1:])

        if rendering_style is None:
            thisRendering = MatplotLibMap.renderingRules['calculated_path']
        else:
            thisRendering = rendering_style

        for i, edge in enumerate(edges):
            node_from = self._osm.nodes[edge[0]]
            node_to = self._osm.nodes[edge[1]]
            x_from = node_from.lon
            y_from = node_from.lat
            x_to = node_to.lon
            y_to = node_to.lat


            plt.plot([x_from,x_to],[y_from,y_to],
                    marker          = '',
                    linestyle       = thisRendering['linestyle'],
                    linewidth       = thisRendering['linewidth'],
                    color           = thisRendering['color'],
                    solid_capstyle  = 'round',
                    solid_joinstyle = 'round',
                    zorder          = thisRendering['zorder'],
                    )

            if animate:
                plt.draw()

        plt.draw()
    ## Plota considerando as distancias 
    def plot_considered_paths(self, axes, path, *paths_considered_tuples):
        plt.sca(axes)
        edges = zip(path, path[1:])

        for path_considered_tuple in paths_considered_tuples:
            paths_considered = path_considered_tuple[0]
            color = path_considered_tuple[1]
            for i, edge in enumerate(paths_considered):
                node_from = self._osm.nodes[edge[0]]
                node_to = self._osm.nodes[edge[1]]
                x_from = node_from.lon
                y_from = node_from.lat
                x_to = node_to.lon
                y_to = node_to.lat

                plt.plot([x_from,x_to],[y_from,y_to],
                        marker          = '',
                        linestyle       = '-',
                        linewidth       = 1,
                        color           = color,
                        solid_capstyle  = 'round',
                        solid_joinstyle = 'round',
                        zorder          = 0,
                        )

        for i, edge in enumerate(edges):
            node_from = self._osm.nodes[edge[0]]
            node_to = self._osm.nodes[edge[1]]
            x_from = node_from.lon
            y_from = node_from.lat
            x_to = node_to.lon
            y_to = node_to.lat


            plt.plot([x_from,x_to],[y_from,y_to],
                    marker          = '',
                    linestyle       = '-',
                    linewidth       = 3,
                    color           = 'black',
                    solid_capstyle  = 'round',
                    solid_joinstyle = 'round',
                    zorder          = 1,
                    )

        plt.draw()


## Converte coordenadas
def convert_to_pixel_coords(lat, lon):
    map_width = 400
    map_height = 400

    x = (lon + 180.0) * (map_width / 360.0)
    lat_rad = abs(lat * math.pi / 180.0)

    merc_n = math.log(math.tan((math.pi / 4.0) + (lat_rad / 2.0)))

    y = 180.0/math.pi*math.log(math.tan(math.pi/4.0 + lat *(math.pi/180.0)/2.0))
    return x, y


class MapInfo:
    def __init__(self):
        self.map_shiftX = 0
        self.map_shiftY = 0

    # Conveter lat para y
    def convert_lat_to_y(self, lat):
        y = 0
        w = 2000
        SCALE = 9000
        lat_rad = math.radians(lat)
        y = (w / (2 * math.pi) * math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0)) * SCALE)
        y += self.map_shiftY
        return y

    # Convert lon para x 
    def convert_lon_to_x(self, lon):
        x = 0
        w = 2000
        SCALE = 9000
        lon_rad = math.radians(lon)

        x = (w / (2.0 * math.pi)) * (lon_rad) * SCALE
        x -= self.map_shiftX

        return x

def get_points_from_node_ids(osm, path):
    edges = zip(path, path[1:])

    path_coords = []

    for i, edge in enumerate(edges):
        node_from = osm.nodes[edge[0]]
        node_to = osm.nodes[edge[1]]
        x_from = node_from.lon
        y_from = node_from.lat
        x_to = node_to.lon
        y_to = node_to.lat

        x_pixel, y_pixel = convert_to_pixel_coords(y_from, x_from)
        path_coords.append([x_pixel, y_pixel, 0.1])

        if i == len(path) - 2:
            x_pixel, y_pixel = convert_to_pixel_coords(y_to, x_to)
            path_coords.append([x_pixel, y_pixel, 0.1])

    return np.array(path_coords).astype(np.float32)

def get_vbo(osm):
    road_vbos = []  # A list of individual vbo's
    other_vbos = []

    for idx, nodeID in enumerate(osm.ways.keys()):
        vbo = []
        wayTags = osm.ways[nodeID].tags
        wayType = None
        thisRendering = MatplotLibMap.renderingRules['other']
        if 'highway' in wayTags.keys():
            wayType = wayTags['highway']

        if wayType in [
            'primary',
            'primary_link',
            'unclassified',
            'secondary',
            'secondary_link',
            'tertiary',
            'tertiary_link',
            'residential',
            'trunk',
            'trunk_link',
            'motorway',
            'motorway_link'
        ]:

            if wayType in list(MatplotLibMap.renderingRules.keys()):
                thisRendering = MatplotLibMap.renderingRules[wayType]

        for nCnt, nID in enumerate(osm.ways[nodeID].nds):
            y = float(osm.nodes[nID].lat)
            x = float(osm.nodes[nID].lon)

            vbo.append([x,y, abs(thisRendering['zorder'] / 10000000.0)])

        if len(vbo) > 0:
            if wayType is not None:
                road_vbos.append((vbo, thisRendering['color']))
            else:
                if 'building' in wayTags:
                    other_vbos.append((vbo, thisRendering['color']))

    return road_vbos, other_vbos

def main():
	grafo , osm = read_osm(sys.argv[1])
	print(osm.bounds)
	print("=============================")
	#print(osm.nodes)

	matplotmap = MatplotLibMap(osm, grafo)

if __name__ == '__main__':
	main()


