#this is just because if it will be sometime useful
from __future__ import absolute_import

import sympy as sym


def _parse_sympy_obj(ob):
    #while ob.args:
    print()


class TreeNode:
    def __init__(self, start_pos, end_pos, priority, st, bracket_ids):
        self.start = start_pos
        self.end = end_pos
        self.priority = priority
        self.parent = None
        self.children = []
        # self.
        self.args = st[bracket_ids[start_pos]:bracket_ids[end_pos] + 1]
        if start_pos == 0:
            de = 0
        else:
            de = bracket_ids[start_pos - 1] + 1
        self.type = st[de:bracket_ids[start_pos]]
        if self.type[0] == ',':
            self.type = self.type[2:]
        self.val = None


class Fraction:
    def __init__(self, num, den):
        self.num = num
        self.den = den


def _map_val(node: TreeNode, child_nodes=None):
    if node.type == "Symbol":
        ret = sym.Symbol(node.args[2:-2])
    elif node.type == "Integer":
        ret = int(node.args[1:-1])
    elif node.type == "Mul":
        arg1 = child_nodes[0].val
        arg2 = child_nodes[1].val
        ret = sym.sympify(f"( {arg1} ) * ( {arg2} )")
    elif node.type == "Pow":
        if child_nodes[1].val == -1:
            ret = sym.sympify(f"1 / {child_nodes[0].val}", evaluate=False)
    elif node.type == "Add":
        arg1 = child_nodes[0].val
        arg2 = child_nodes[1].val
        arg1minus = arg1.args and arg1.args[0] == -1
        arg2minus = arg2.args and arg2.args[0] == -1
        if arg1minus and arg2minus:
            print("Warning - two negative add")
            ret = sym.sympify(f"{arg2} + {arg1}", evaluate=False)
        elif arg1minus:
            arg1 = arg1.args[1]
            ret = sym.sympify(f"{arg2} - {arg1}", locals={str(arg1): arg1, str(arg2): arg2}, evaluate=False)
        elif arg2minus:
            arg2 = arg2.args[1]
            ret = sym.sympify(f"{arg1} - {arg2}", locals={str(arg1): arg1, str(arg2): arg2}, evaluate=False)
        else:
            ret = sym.sympify(f"{arg1} + {arg2}", evaluate=False)

    return ret


def _create_tree_rec(brackets, res_srept, pos, priority, nodes):
    cur_symbol = brackets[pos][1]
    child_nodes = []
    start_pos = pos - 1
    # no backwards pass
    if cur_symbol == 'r':
        end_pos = pos
        new_node = TreeNode(start_pos, end_pos,
                            priority, res_srept, [bracket[0] for bracket in brackets])
        new_node.val = _map_val(new_node)
        nodes[new_node] = new_node
        print(f"{brackets[new_node.start][0]} {brackets[new_node.end][0]}")
        return new_node

    while cur_symbol == 'l':
        child_node = _create_tree_rec(brackets, res_srept, pos + 1, priority + 1, nodes)
        child_nodes.append(child_node)
        pos = child_node.end + 1
        if pos == len(brackets):
            return
        cur_symbol = brackets[pos][1]

    end_pos = pos
    new_node = TreeNode(start_pos, end_pos,
                        priority, res_srept, [bracket[0] for bracket in brackets])
    new_node.val = _map_val(new_node, child_nodes)
    nodes[new_node] = new_node
    for c_node in child_nodes:
        c_node.parent = new_node
        new_node.children.append(c_node)

    print(f"{brackets[new_node.start][0]} {brackets[new_node.end][0]}")
    return new_node


def _create_tree(res_srept):
    l_ids = [idx for idx, c in enumerate(res_srept) if c == '(']
    r_ids = [idx for idx, c in enumerate(res_srept) if c == ')']
    bracket_ids = sorted(l_ids + r_ids)
    brackets = []
    for id in bracket_ids:
        type = 'l' if id in l_ids else 'r'
        brackets.append((id, type))
    nodes = {}
    _create_tree_rec(brackets, res_srept, 0, 0, nodes)
    prior_q = sorted(nodes, key=lambda x: x.priority, reverse=True)
    return nodes


def _recompose_tree(st, tree):
    prior_q = sorted(tree, key=lambda x: x.priority, reverse=True)
    for node in prior_q:
        if node.children_vals:
            if node.type == "Add":
                if node.children_vals[0] < 0 and node.children_vals[1] > 0:
                    node.children_vals = [node.children_vals[1], node.children_vals[0]]
                node.computed_vals = 1
            if node.type == "Mul":
                if node.children_vals[0] < 0 and node.children_vals[1] > 0:
                    node.children_vals = [node.children_vals[1], node.children_vals[0]]
                    node.computed_vals = -1
                else:
                    node.computed_vals = 1
        else:
            if node.type == "Integer":
                node.computed_vals = int(node.args[1:-1])
            elif node.type == "Symbol":
                node.computed_vals = 1
        parent = node.parent
        if parent:
            parent.children_vals.append(node.computed_vals)
    print()


