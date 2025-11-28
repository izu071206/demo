"""
CFG (Control Flow Graph) Properties Extractor
Trích xuất các thuộc tính từ CFG của binary
"""

import angr
import networkx as nx
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CFGExtractor:
    """Trích xuất CFG properties từ binary"""
    
    def __init__(self):
        """Initialize CFG extractor"""
        pass
    
    def extract_cfg(self, binary_path: str) -> Optional[nx.DiGraph]:
        """
        Trích xuất CFG từ binary sử dụng angr
        
        Args:
            binary_path: Path to binary file
            
        Returns:
            NetworkX DiGraph representing CFG
        """
        try:
            # Load binary với angr
            project = angr.Project(binary_path, auto_load_libs=False)
            
            # Trích xuất CFG
            cfg = project.analyses.CFGFast()
            
            # Convert sang NetworkX graph
            graph = nx.DiGraph()
            
            for node_addr, node in cfg.graph.nodes(data=True):
                graph.add_node(node_addr)
            
            for src, dst in cfg.graph.edges():
                graph.add_edge(src, dst)
            
            return graph
        
        except Exception as e:
            logger.error(f"Error extracting CFG from {binary_path}: {e}")
            return None
    
    def calculate_metrics(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Tính toán các metrics từ CFG
        
        Args:
            graph: NetworkX DiGraph
            
        Returns:
            Dictionary of metrics
        """
        if graph is None or len(graph) == 0:
            return self._empty_metrics()
        
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = graph.number_of_nodes()
        metrics['num_edges'] = graph.number_of_edges()
        
        # Average degree
        if metrics['num_nodes'] > 0:
            metrics['avg_degree'] = 2 * metrics['num_edges'] / metrics['num_nodes']
        else:
            metrics['avg_degree'] = 0.0
        
        # Cyclomatic complexity: E - N + 2P
        # P = number of connected components
        if nx.is_weakly_connected(graph):
            p = 1
        else:
            p = nx.number_weakly_connected_components(graph)
        
        metrics['cyclomatic_complexity'] = metrics['num_edges'] - metrics['num_nodes'] + 2 * p
        
        # Number of loops (strongly connected components with size > 1)
        sccs = list(nx.strongly_connected_components(graph))
        metrics['num_loops'] = sum(1 for scc in sccs if len(scc) > 1)
        
        # Depth metrics
        try:
            if len(graph) > 0:
                # Longest path length
                try:
                    longest_path = nx.dag_longest_path_length(graph)
                    metrics['max_depth'] = longest_path
                except:
                    metrics['max_depth'] = 0.0
                
                # Average path length (if possible)
                try:
                    avg_path_length = nx.average_shortest_path_length(graph.to_undirected())
                    metrics['avg_path_length'] = avg_path_length
                except:
                    metrics['avg_path_length'] = 0.0
            else:
                metrics['max_depth'] = 0.0
                metrics['avg_path_length'] = 0.0
        except:
            metrics['max_depth'] = 0.0
            metrics['avg_path_length'] = 0.0
        
        # Clustering coefficient
        try:
            undirected = graph.to_undirected()
            clustering = nx.average_clustering(undirected)
            metrics['clustering_coefficient'] = clustering
        except:
            metrics['clustering_coefficient'] = 0.0
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Trả về metrics rỗng"""
        return {
            'num_nodes': 0.0,
            'num_edges': 0.0,
            'avg_degree': 0.0,
            'cyclomatic_complexity': 0.0,
            'num_loops': 0.0,
            'max_depth': 0.0,
            'avg_path_length': 0.0,
            'clustering_coefficient': 0.0
        }
    
    def extract_features(self, binary_path: str) -> Dict[str, float]:
        """
        Trích xuất CFG features từ binary
        
        Args:
            binary_path: Path to binary file
            
        Returns:
            Dictionary of CFG metrics
        """
        graph = self.extract_cfg(binary_path)
        if graph is None:
            return self._empty_metrics()
        
        return self.calculate_metrics(graph)

