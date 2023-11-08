"""
Graph parsing utilities from a very simplify version of the DOT grammar
https://www.graphviz.org/doc/info/lang.html
graph       :  '{' stmt_list '}'
stmt_list   : (node_id | edge_stmt) '\n' stmt_list
edge_stmt 	:  node_id  edgeRHS
edgeRHS 	: 	'->' node_id  [ edgeRHS ]
node_id 	: 	ID
ID          : [a-zA-Z0-9_]+

ID are case un-sensitives
"""
import re
from collections import deque
from pathlib import Path


def wrong_format_msg(got_str_list: list[str], expected_str_list: list[str]):
    msg = ""
    for got_str, expected_str in zip(got_str_list, expected_str_list):
        msg += f"got: {got_str} expected: {expected_str}\n"
    return msg


def parsing_graph_from_path(path: Path):
    with open(path, mode="r") as fd_r:
        txt_graph = fd_r.read()
    return parsing_graph_from_text(txt_graph)


def parsing_graph_from_list(graph: list):
    outgoing_edges: dict[str, list[str]] = dict()
    ingoing_edges: dict[str, list[str]] = dict()

    def maybe_add(node):
        if outgoing_edges.get(node) is None:
            outgoing_edges[node] = []
            ingoing_edges[node] = []

    for nodes_list in graph:
        maybe_add(nodes_list[0])
        for i in range(1, len(nodes_list)):
            maybe_add(nodes_list[i])
            outgoing_edges[nodes_list[i - 1]].append(nodes_list[i])
            ingoing_edges[nodes_list[i]].append(nodes_list[i - 1])
    # remove duplicates while preserving insertion order
    for key in outgoing_edges:
        outgoing_edges[key] = list(dict.fromkeys(outgoing_edges[key]))
        ingoing_edges[key] = list(dict.fromkeys(ingoing_edges[key]))

    return outgoing_edges, ingoing_edges


# very crude parsing, pattern matching would be better but...its python 3.9
def parsing_graph_from_text(graph: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    node_id_pattern = r"^([a-z0-9_]+)$"
    arrow_pattern = "->"
    arrow_space_pattern = fr" * {arrow_pattern} *"
    outgoing_edges: dict[str, list[str]] = dict()
    ingoing_edges: dict[str, list[str]] = dict()

    def maybe_add(node: str):
        if outgoing_edges.get(node) is None:
            valid = bool(re.search(node_id_pattern, node))
            if not valid:
                maybe_add_error_msg = wrong_format_msg([node], [node_id_pattern])
                raise ValueError(maybe_add_error_msg)
            outgoing_edges[node] = []
            ingoing_edges[node] = []

    txt_graph = graph.lower()
    # remove trailing whitespaces
    txt_graph = txt_graph.strip()
    first_char = txt_graph[0]
    last_char = txt_graph[-1]
    if first_char != '{' or last_char != '}':
        char_error_msg = wrong_format_msg([first_char, last_char], ['{', '}'])
        raise ValueError(char_error_msg)
    txt_graph = txt_graph[1:-1]
    # remove any trailing whitespace characters
    txt_graph = txt_graph.strip()
    # remove any space btw arrow sides
    txt_graph = re.sub(arrow_space_pattern, arrow_pattern, txt_graph)
    # break down statement
    stmt_list = txt_graph.split(sep="\n")
    for stmt in stmt_list:
        stmt = stmt.strip()
        nodes = stmt.split(sep="->")
        maybe_add(nodes[0])
        for i in range(1, len(nodes)):
            maybe_add(nodes[i])
            outgoing_edges[nodes[i - 1]].append(nodes[i])
            ingoing_edges[nodes[i]].append(nodes[i - 1])
    for key in outgoing_edges:
        outgoing_edges[key] = list(dict.fromkeys(outgoing_edges[key]))
        ingoing_edges[key] = list(dict.fromkeys(ingoing_edges[key]))
    return outgoing_edges, ingoing_edges
