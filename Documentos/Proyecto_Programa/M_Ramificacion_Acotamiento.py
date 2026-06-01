import streamlit as st
import numpy as np
import math
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from PIL import Image
import io
import itertools
import pandas as pd

st.set_page_config(
    layout="wide",
    page_title="Ramificacion y Acotamiento",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #f8f7f4;
        color: #1a1a2e;
    }
    .stApp { background-color: #f8f7f4; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e8e4dc;
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1a1a2e;
        font-weight: 600;
        border-bottom: 2px solid #c8b8a2;
        padding-bottom: 4px;
    }

    /* Encabezados principales */
    h1 { color: #1a1a2e !important; font-weight: 600 !important; letter-spacing: -0.5px; }
    h2 { color: #2d3561 !important; font-weight: 500 !important; }
    h3 { color: #2d3561 !important; font-weight: 500 !important; }

    /* Tarjetas de resultado */
    .result-card {
        background: #ffffff;
        border: 2px solid #2d3561;
        border-left: 6px solid #2d3561;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        font-family: 'IBM Plex Mono', monospace;
    }
    .result-card .label { font-size: 11px; letter-spacing: 2px; color: #888; text-transform: uppercase; margin-bottom: 6px; }
    .result-card .value { font-size: 22px; font-weight: 600; color: #1a1a2e; }
    .result-card .vars  { font-size: 13px; color: #555; margin-top: 6px; }

    /* Cota card */
    .bound-card {
        background: #f0f4ff;
        border: 1px solid #aab4e8;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Tabla */
    .stTable table { font-size: 12px; }
    .stTable thead tr th {
        background-color: #2d3561 !important;
        color: white !important;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .stTable tbody tr:nth-child(even) { background-color: #f4f2ee; }

    /* Botón principal */
    .stButton > button[kind="primary"] {
        background-color: #2d3561;
        color: white;
        border-radius: 6px;
        border: none;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        letter-spacing: 0.5px;
        padding: 10px 24px;
        transition: background 0.2s;
    }
    .stButton > button[kind="primary"]:hover { background-color: #1a1a2e; }

    /* Botón de descarga */
    .stDownloadButton > button {
        background-color: #ffffff;
        border: 2px solid #2d3561;
        color: #2d3561;
        border-radius: 6px;
        font-weight: 500;
    }

    /* Separador de sección */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 3px;
        color: #888;
        text-transform: uppercase;
        padding: 16px 0 8px 0;
        border-bottom: 1px solid #ddd;
        margin-bottom: 16px;
    }

    /* Iteración card */
    .iter-card {
        background: #ffffff;
        border: 1px solid #e0ddd5;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 4px solid #c8b8a2;
    }
    .iter-card.branched  { border-left-color: #2d3561; }
    .iter-card.optimal   { border-left-color: #4a7c59; }
    .iter-card.infeasible{ border-left-color: #c0392b; }
    .iter-card.pruned    { border-left-color: #c8b8a2; }

    .iter-title { font-weight: 600; font-size: 14px; color: #1a1a2e; margin-bottom: 4px; }
    .iter-body  { font-size: 12px; color: #555; font-family: 'IBM Plex Mono', monospace; line-height: 1.6; }

    /* Indicadores knapsack */
    .knap-item {
        display: inline-block;
        background: #eef1fb;
        border: 1px solid #aab4e8;
        border-radius: 4px;
        padding: 4px 10px;
        margin: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
    }

    /* Formulario */
    .stNumberInput input, .stSelectbox select {
        border-radius: 4px;
        border: 1px solid #c8b8a2;
        font-size: 13px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #e0ddd5; }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #888;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 13px;
        font-weight: 500;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff;
        border-bottom: 3px solid #2d3561;
        color: #2d3561;
        border-radius: 4px 4px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MODELO / ALGORITMO


class Node:
    def __init__(self, id, bounds, parent_id=None, branch_desc="Ninguna", depth=0):
        self.id          = id
        self.bounds      = bounds
        self.parent_id   = parent_id
        self.branch_desc = branch_desc
        self.depth       = depth
        self.z      = None
        self.x      = None
        self.status = "No resuelto"
        self.branch_var  = None
        self.branch_val  = None

def is_integer(val, tol=1e-5):
    return abs(val - round(val)) < tol

def solve_relaxation(c, A, b, senses, bounds, is_max):
    c_opt = [-x for x in c] if is_max else list(c)
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for i in range(len(senses)):
        if senses[i] == "<=":
            A_ub.append(A[i]); b_ub.append(b[i])
        elif senses[i] == ">=":
            A_ub.append([-v for v in A[i]]); b_ub.append(-b[i])
        elif senses[i] == "==":
            A_eq.append(A[i]); b_eq.append(b[i])
    res = linprog(
        c_opt,
        A_ub=A_ub or None, b_ub=b_ub or None,
        A_eq=A_eq or None, b_eq=b_eq or None,
        bounds=bounds, method='highs'
    )
    if res.success:
        z = -res.fun if is_max else res.fun
        return True, z, res.x
    return False, None, None

def get_accumulated_constraints(n_id, nodes_dict):
    curr = nodes_dict[n_id]
    cons = []
    while curr.parent_id is not None:
        cons.append(curr.branch_desc)
        curr = nodes_dict[curr.parent_id]
    return " | ".join(reversed(cons)) if cons else "Modelo original"

def branch_and_bound(c, A, b, senses, var_types, is_max):
    num_vars = len(c)
    initial_bounds = []
    for t in var_types:
        if t == 'Binaria': initial_bounds.append((0, 1))
        else:              initial_bounds.append((0, None))

    root = Node(id=0, bounds=initial_bounds, depth=0)
    nodes = [root]
    active = [root]
    best_z = -float('inf') if is_max else float('inf')
    best_x = None
    counter = 0
    iteration_log = []

    while active:
        current = active.pop()
        success, z, x = solve_relaxation(c, A, b, senses, current.bounds, is_max)
        log_entry = {"nodo": current.id, "profundidad": current.depth, "restriccion": current.branch_desc}

        if not success:
            current.status = "Infactible"
            log_entry["estado"] = "Infactible"
            log_entry["z"] = None
            iteration_log.append(log_entry)
            continue

        current.z = z
        current.x = np.round(x, 6)
        log_entry["z"] = z
        log_entry["x"] = list(np.round(x, 4))

        if (is_max and z <= best_z) or (not is_max and z >= best_z):
            current.status = "Agotado"
            log_entry["estado"] = "Agotado (cota superada)"
            iteration_log.append(log_entry)
            continue

        is_int = True
        branch_idx, branch_val = -1, None
        for i in range(num_vars):
            if var_types[i] in ['Entera', 'Binaria']:
                if not is_integer(current.x[i]):
                    is_int = False
                    branch_idx = i
                    branch_val = current.x[i]
                    break

        if is_int:
            current.status = "Solucion Entera"
            best_z = z
            best_x = current.x.copy()
            log_entry["estado"] = "Solucion Entera Optima"
            iteration_log.append(log_entry)
        else:
            current.status = "Ramificado"
            current.branch_var = branch_idx
            current.branch_val = branch_val
            log_entry["estado"] = f"Ramificado en x{branch_idx+1}={branch_val:.4f}"
            iteration_log.append(log_entry)

            lb_l = list(current.bounds)
            lb_r = list(current.bounds)

            if var_types[branch_idx] == 'Binaria':
                lb_l[branch_idx] = (0, 0)
                lb_r[branch_idx] = (1, 1)
                dl = f"x{branch_idx+1}=0"
                dr = f"x{branch_idx+1}=1"
            else:
                fv = math.floor(branch_val)
                lb_l[branch_idx] = (lb_l[branch_idx][0], fv)
                lb_r[branch_idx] = (fv + 1, lb_r[branch_idx][1])
                dl = f"x{branch_idx+1}<={fv}"
                dr = f"x{branch_idx+1}>={fv+1}"

            counter += 1
            nl = Node(counter, lb_l, current.id, dl, current.depth+1)
            counter += 1
            nr = Node(counter, lb_r, current.id, dr, current.depth+1)
            nodes.extend([nl, nr])
            active.extend([nl, nr])

    nodes.sort(key=lambda n: n.id)
    return nodes, best_z, best_x, iteration_log

# Problema Mochila

def knapsack_bb(items, capacity):
  
    n = len(items)
    
    ratios = sorted(range(n), key=lambda i: items[i]["value"]/items[i]["weight"] if items[i]["weight"] else 0, reverse=True)

    def upper_bound(idx, curr_val, curr_w, included):
        val = curr_val
        w   = curr_w
        for i in range(idx, n):
            j = ratios[i]
            if w + items[j]["weight"] <= capacity:
                w   += items[j]["weight"]
                val += items[j]["value"]
            else:
                frac = (capacity - w) / items[j]["weight"]
                val += frac * items[j]["value"]
                break
        return val

    best_val = 0
    best_sel = []
    tree     = []
    ilog     = []
    node_id  = [0]

    def bb(idx, curr_val, curr_w, included):
        nonlocal best_val, best_sel
        nid = node_id[0]; node_id[0] += 1
        ub  = upper_bound(idx, curr_val, curr_w, included)
        entry = {
            "id": nid, "idx": idx,
            "curr_val": curr_val, "curr_w": curr_w,
            "ub": ub, "included": included[:]
        }

        if ub <= best_val:
            entry["status"] = "Agotado"
            tree.append(entry); ilog.append(entry); return

        if idx == n:
            if curr_val > best_val:
                best_val = curr_val
                best_sel = included[:]
            entry["status"] = "Solucion Entera"
            tree.append(entry); ilog.append(entry); return

        entry["status"] = "Ramificado"
        tree.append(entry); ilog.append(entry)

        j = ratios[idx]
        if curr_w + items[j]["weight"] <= capacity:
            bb(idx+1, curr_val+items[j]["value"], curr_w+items[j]["weight"], included+[j])
        bb(idx+1, curr_val, curr_w, included)

    bb(0, 0, 0, [])
    return best_val, best_sel, tree, ilog

# Regiones factibles 

def find_feasible_corners(c, A, b, senses):
    lines_A = [row[:] for row in A] + [[1,0],[0,1],[-1,0],[0,-1]]
    lines_b = list(b)              + [0,   0,   0,    0   ]
    corners = []
    for i, j in itertools.combinations(range(len(lines_A)), 2):
        try:
            Am = np.array([lines_A[i], lines_A[j]])
            bv = np.array([lines_b[i], lines_b[j]])
            x  = np.linalg.solve(Am, bv)
            x  = np.round(x, 6)
            if x[0] < -1e-5 or x[1] < -1e-5: continue
            ok = True
            for k in range(len(A)):
                v = A[k][0]*x[0]+A[k][1]*x[1]
                if senses[k]=="<=" and v>b[k]+1e-5: ok=False
                elif senses[k]==">=" and v<b[k]-1e-5: ok=False
                elif senses[k]=="==" and abs(v-b[k])>1e-5: ok=False
            if ok and not any(np.allclose(x, q) for q in corners):
                corners.append(x)
        except np.linalg.LinAlgError:
            pass
    return corners


# ──────────────────────────────────────────────
# COMPONENTES DE VISUALIZACIÓN


def _assign_positions(nodes):
    
    id_map = {n.id: n for n in nodes}
    children = {n.id: [] for n in nodes}
    for n in nodes:
        if n.parent_id is not None:
            children[n.parent_id].append(n.id)

    
    levels = {}
    queue = [nodes[0].id]
    levels[nodes[0].id] = 0
    while queue:
        cur = queue.pop(0)
        for ch in children[cur]:
            levels[ch] = levels[cur] + 1
            queue.append(ch)

    max_level = max(levels.values()) if levels else 0
    level_nodes = {}
    for nid, lv in levels.items():
        level_nodes.setdefault(lv, []).append(nid)

    pos = {}
    for lv, nids in level_nodes.items():
        for k, nid in enumerate(nids):
            x = (k - (len(nids)-1)/2.0) * 2.2
            y = -(lv * 2.0)
            pos[nid] = (x, y)
    return pos, children, max_level


def build_tree_figure(nodes, num_vars, title="Arbol de Ramificacion"):
    
    STATUS_COLORS = {
        "Solucion Entera": ("#e8f5e9", "#388e3c"),
        "Infactible":      ("#ffebee", "#c62828"),
        "Agotado":         ("#f5f5f5", "#9e9e9e"),
        "Ramificado":      ("#e3f2fd", "#1565c0"),
        "No resuelto":     ("#fafafa", "#757575"),
    }
    id_map = {n.id: n for n in nodes}
    pos, children, max_level = _assign_positions(nodes)

    
    all_x = [p[0] for p in pos.values()]
    width  = max(10, (max(all_x) - min(all_x) + 2.5) * 1.1)
    height = max(5,  (max_level + 1) * 2.4)

    fig, ax = plt.subplots(figsize=(width, height), facecolor="#f8f7f4")
    ax.set_facecolor("#f8f7f4")
    ax.axis("off")
    ax.set_aspect("equal")

    BOX_W, BOX_H = 1.9, 0.85

    
    for n in nodes:
        if n.parent_id is not None and n.parent_id in pos and n.id in pos:
            px, py = pos[n.parent_id]
            cx, cy = pos[n.id]
            ax.annotate("",
                xy=(cx, cy + BOX_H/2),
                xytext=(px, py - BOX_H/2),
                arrowprops=dict(arrowstyle="-|>", color="#999999",
                                lw=1.2, mutation_scale=10))
            
            mx = (px + cx) / 2
            my = (py + cy) / 2
            ax.text(mx, my, n.branch_desc, fontsize=6.5, ha="center", va="center",
                    color="#2d3561",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#cccccc", lw=0.6))

    
    for n in nodes:
        if n.id not in pos:
            continue
        x, y = pos[n.id]
        fc, ec = STATUS_COLORS.get(n.status, ("#fafafa", "#757575"))

        
        box = FancyBboxPatch((x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
                             boxstyle="round,pad=0.05",
                             linewidth=1.5, edgecolor=ec, facecolor=fc,
                             zorder=3)
        ax.add_patch(box)

        
        lines = [f"Nodo {n.id}"]
        if n.id != 0 and n.branch_desc:
            lines.append(f"[{n.branch_desc}]")
        if n.z is not None:
            lines.append(f"Z = {n.z:.3f}")
            xvals = "  ".join([f"x{i+1}={n.x[i]:.2f}" for i in range(num_vars)])
            lines.append(xvals)
        lines.append(n.status)

        txt = "\n".join(lines)
        ax.text(x, y, txt, fontsize=6.5, ha="center", va="center",
                color="#1a1a2e", zorder=4,
                multialignment="center",
                fontfamily="monospace")

    
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - BOX_W, max(xs) + BOX_W)
        ax.set_ylim(min(ys) - BOX_H*1.5, max(ys) + BOX_H*1.5)

    ax.set_title(title, fontsize=11, fontweight="bold", color="#1a1a2e", pad=10)
    fig.tight_layout(pad=1)
    return fig


def build_tree_graph(nodes, num_vars, best_x, is_max):
    
    return build_tree_figure(nodes, num_vars)

def plot_2d(c, A, b, senses, nodes, best_x, is_max, var_types):
    corners = find_feasible_corners(c, A, b, senses)

    max_val = 5
    for i in range(len(A)):
        for j in range(2):
            if A[i][j] > 0: max_val = max(max_val, b[i]/A[i][j])
    if best_x is not None: max_val = max(max_val, float(best_x[0]), float(best_x[1]))
    max_val = math.ceil(max_val * 1.3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='#f8f7f4')
    ax = axes[0]
    ax.set_facecolor('#ffffff')

    d = np.linspace(0, max_val, 500)
    xg, yg = np.meshgrid(d, d)
    region = np.ones_like(xg, dtype=bool)
    for i in range(len(A)):
        a1, a2 = A[i][0], A[i][1]
        if senses[i]=="<=": region &= (a1*xg+a2*yg <= b[i])
        elif senses[i]==">=": region &= (a1*xg+a2*yg >= b[i])

    ax.contourf(xg, yg, region.astype(float), levels=[0.5,1.5], colors=['#dce8f7'], alpha=0.6)
    ax.contour(xg, yg, region.astype(float), levels=[0.5], colors=['#4a7cb4'], linewidths=1, linestyles='--')

    palette = ['#2d3561','#c0392b','#4a7c59','#e67e22','#8e44ad']
    xs = np.linspace(0, max_val, 500)
    for i in range(len(A)):
        a1, a2 = A[i][0], A[i][1]
        col = palette[i % len(palette)]
        sign = senses[i]
        label = f"R{i+1}: {a1}x1 + {a2}x2 {sign} {b[i]}"
        if a2 != 0:
            ax.plot(xs, (b[i]-a1*xs)/a2, color=col, lw=2, label=label)
        elif a1 != 0:
            ax.axvline(b[i]/a1, color=col, lw=2, label=label)

    if corners:
        cx = [p[0] for p in corners]; cy = [p[1] for p in corners]
        ax.scatter(cx, cy, color='#2d3561', s=60, zorder=5, marker='o', label='Vertices factibles')

    root_node = nodes[0]
    if root_node.x is not None:
        ax.scatter([root_node.x[0]], [root_node.x[1]], color='#2980b9', s=120, zorder=6,
                   marker='D', label='Optimo relajado')

    if best_x is not None:
        ax.scatter([best_x[0]], [best_x[1]], color='#c0392b', s=200, zorder=7,
                   marker='*', label='Optimo entero')

    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
    ax.set_xlabel('x1', fontsize=11); ax.set_ylabel('x2', fontsize=11)
    ax.set_title('Region Factible — Solucion Grafica', fontsize=12, fontweight='bold', color='#1a1a2e', pad=12)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines[['top','right']].set_visible(False)

    
    ax2 = axes[1]
    ax2.set_facecolor('#f8f7f4')
    ax2.axis('off')

    evals = [(pt, c[0]*pt[0]+c[1]*pt[1]) for pt in corners]
    evals.sort(key=lambda e: e[1], reverse=is_max)

    y_pos = 0.97
    ax2.text(0, y_pos, 'Evaluacion de Vertices', fontsize=11, fontweight='bold',
             color='#1a1a2e', va='top', transform=ax2.transAxes)
    y_pos -= 0.08
    ax2.text(0, y_pos, f'Tipo: {"Maximizacion" if is_max else "Minimizacion"}',
             fontsize=9, color='#666', va='top', transform=ax2.transAxes)
    y_pos -= 0.07

    for rank, (pt, val) in enumerate(evals):
        bg = '#e8f5e9' if rank == 0 else '#ffffff'
        brd= '#388e3c' if rank == 0 else '#ddd'
        box = FancyBboxPatch((0, y_pos-0.08), 1.0, 0.09,
                             boxstyle="round,pad=0.01", linewidth=1,
                             edgecolor=brd, facecolor=bg,
                             transform=ax2.transAxes, clip_on=False)
        ax2.add_patch(box)
        star = " [optimo]" if rank == 0 else ""
        ax2.text(0.03, y_pos-0.02,
                 f"V{rank+1}  ({pt[0]:.3f}, {pt[1]:.3f})  ->  Z = {val:.4f}{star}",
                 fontsize=9, va='top', transform=ax2.transAxes,
                 fontfamily='monospace', color='#1a1a2e')
        y_pos -= 0.12

    y_pos -= 0.05
    if root_node.z is not None:
        ax2.text(0, y_pos, f'Z optimo relajado: {root_node.z:.4f}',
                 fontsize=9, color='#2980b9', va='top', transform=ax2.transAxes,
                 fontfamily='monospace')
        y_pos -= 0.07
    if best_x is not None:
        ax2.text(0, y_pos, f'Z optimo entero:   {c[0]*best_x[0]+c[1]*best_x[1]:.4f}',
                 fontsize=9, color='#c0392b', va='top', transform=ax2.transAxes,
                 fontfamily='monospace')

    fig.tight_layout(pad=2)
    return fig



# SIDEBAR — Configuración


with st.sidebar:
    st.markdown("## Configuracion del Modelo")

    model_type = st.selectbox(
        "Tipo de Modelo",
        ["Entero Puro ", "Mixto ", "Binario ", "Mochila "]
    )
    is_knapsack = model_type == "Mochila (Knapsack)"

    if not is_knapsack:
        opt_type = st.selectbox("Objetivo", ["Maximizar", "Minimizar"])
        is_max   = opt_type == "Maximizar"

        st.markdown("---")
        num_vars = st.number_input("Variables de decision", min_value=2, max_value=8, value=2)
        num_cons = st.number_input("Restricciones", min_value=1, max_value=10, value=2)

        st.markdown("### Funcion Objetivo")
        c = []
        cols_c = st.columns(num_vars)
        for i in range(num_vars):
            c.append(cols_c[i].number_input(f"c{i+1}", value=5.0 if i==0 else 4.0, key=f"c_{i}", step=0.5))

        st.markdown("### Tipos de Variables")
        var_types = []
        cols_t = st.columns(num_vars)
        if model_type == "Entero Puro ":
            default_type = "Entera"
        elif model_type == "Binario ":
            default_type = "Binaria"
        else:
            default_type = "Entera"

        type_options = ["Entera", "Continua", "Binaria"]
        for i in range(num_vars):
            vt = cols_t[i].selectbox(f"x{i+1}", type_options,
                                     index=type_options.index(default_type),
                                     key=f"t_{i}")
            var_types.append(vt)

        st.markdown("### Restricciones")
        A, b, senses = [], [], []
        default_A = [[6,4],[1,2]]
        default_b = [24, 6]
        for i in range(num_cons):
            st.markdown(f"**Restriccion {i+1}**")
            cols_r = st.columns(num_vars + 2)
            row = []
            for j in range(num_vars):
                dv = default_A[i][j] if i < len(default_A) and j < len(default_A[i]) else 1.0
                row.append(cols_r[j].number_input(f"a{i+1}{j+1}", value=float(dv), key=f"a_{i}_{j}", step=0.5, label_visibility="collapsed"))
            A.append(row)
            senses.append(cols_r[-2].selectbox("", ["<=",">=","=="], key=f"s_{i}", label_visibility="collapsed"))
            dbv = float(default_b[i]) if i < len(default_b) else 10.0
            b.append(cols_r[-1].number_input("RHS", value=dbv, key=f"b_{i}", step=0.5, label_visibility="collapsed"))

    else:
        st.markdown("### Articulos de la Mochila")
        num_items = st.number_input("Numero de articulos", min_value=1, max_value=12, value=4)
        capacity  = st.number_input("Capacidad (W)", min_value=1, value=10, step=1)

        items = []
        for i in range(num_items):
            st.markdown(f"**Articulo {i+1}**")
            col1, col2 = st.columns(2)
            v = col1.number_input(f"Valor c{i+1}", value=float([6,5,4,3][i % 4]), key=f"kv_{i}", step=0.5)
            w = col2.number_input(f"Peso a{i+1}",  value=float([4,3,2,3][i % 4]),  key=f"kw_{i}", step=0.5)
            items.append({"name": f"x{i+1}", "value": v, "weight": w})

    solve_btn = st.button("Resolver Modelo", type="primary", use_container_width=True)


# ──────────────────────────────────────────────
# AREA PRINCIPAL
# ──────────────────────────────────────────────

st.markdown("# Ramificacion y Acotamiento")
st.markdown(
    f"<div class='section-header'>"
    f"Metodo | {model_type}  —  "
    f"{'Maximizacion' if (not is_knapsack and is_max) else ('Minimizacion' if not is_knapsack else 'Maximizacion')} | "
    f"Solucion exacta para programacion entera"
    f"</div>",
    unsafe_allow_html=True
)

# ── Formulacion del modelo ───────────────────
with st.expander("Ver Formulacion del Modelo", expanded=False):
    if not is_knapsack:
        fo_terms = " + ".join([f"{c[i]}*x{i+1}" for i in range(num_vars)])
        st.markdown(f"**{'Max' if is_max else 'Min'} Z = {fo_terms}**")
        st.markdown("**Sujeto a:**")
        for i in range(num_cons):
            lhs = " + ".join([f"{A[i][j]}*x{j+1}" for j in range(num_vars)])
            st.markdown(f"- {lhs} {senses[i]} {b[i]}")
        type_str = {v: t for v, t in zip([f"x{i+1}" for i in range(num_vars)], var_types)}
        for vn, vt in type_str.items():
            st.markdown(f"- {vn}: {vt}")
    else:
        terms = " + ".join([f"{it['value']}*{it['name']}" for it in items])
        st.markdown(f"**Max Z = {terms}**")
        wterms = " + ".join([f"{it['weight']}*{it['name']}" for it in items])
        st.markdown(f"**Sujeto a:** {wterms} <= {capacity}")
        st.markdown("**Donde:** x_j ∈ {0, 1}")


if solve_btn:
    with st.spinner("Resolviendo..."):

        if not is_knapsack:
            nodes, best_z, best_x, ilog = branch_and_bound(c, A, b, senses, var_types, is_max)
            nodes_dict = {n.id: n for n in nodes}
        else:
            best_z, best_sel, knap_tree, ilog = knapsack_bb(items, capacity)

    # ── Resultado ─────────────────────────────
    if not is_knapsack:
        if best_x is not None:
            vars_str = "  |  ".join([f"x{i+1} = {best_x[i]:.4f}" for i in range(num_vars)])
            st.markdown(f"""
            <div class='result-card'>
                <div class='label'>Solucion Optima Global</div>
                <div class='value'>Z = {best_z:.4f}</div>
                <div class='vars'>{vars_str}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("No se encontro solucion factible entera para el modelo.")
    else:
        sel_names = [items[i]['name'] for i in best_sel]
        sel_vals  = [items[i]['value'] for i in best_sel]
        sel_ws    = [items[i]['weight'] for i in best_sel]
        st.markdown(f"""
        <div class='result-card'>
            <div class='label'>Solucion Optima — Mochila</div>
            <div class='value'>Z = {best_z:.4f}</div>
            <div class='vars'>
                Articulos seleccionados: {', '.join(sel_names)}<br>
                Peso total: {sum(sel_ws):.2f} / {capacity}  |  Valor total: {sum(sel_vals):.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    LEGEND_HTML = """
    <div style='display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap;'>
      <span style='background:#e3f2fd;border:1px solid #1565c0;padding:3px 12px;border-radius:4px;font-size:12px;'>Ramificado</span>
      <span style='background:#e8f5e9;border:1px solid #388e3c;padding:3px 12px;border-radius:4px;font-size:12px;'>Solucion Entera</span>
      <span style='background:#ffebee;border:1px solid #c62828;padding:3px 12px;border-radius:4px;font-size:12px;'>Infactible</span>
      <span style='background:#f5f5f5;border:1px solid #9e9e9e;padding:3px 12px;border-radius:4px;font-size:12px;'>Agotado</span>
    </div>
    """

    def render_color_table(rows, df_cols):
        df = pd.DataFrame(rows)
        def color_row(row):
            est = row.get("Estado", row.get("Estado / Cota", ""))
            if "Entera" in est:    return ["background-color: #e8f5e9"]*len(row)
            if "Infactible" in est: return ["background-color: #ffebee"]*len(row)
            if "Agotado" in est:   return ["background-color: #f5f5f5"]*len(row)
            if "Ramificado" in est: return ["background-color: #e3f2fd"]*len(row)
            return [""]*len(row)
        st.dataframe(df.style.apply(color_row, axis=1), use_container_width=True, hide_index=True)

    # ── RAMA: modelos normales (PIP / MIP / BIP) ──────────────────────────
    if not is_knapsack:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Arbol de Ramificacion",
            "Grafico 2D",
            "Tabla de Subproblemas",
            "Iteraciones Detalladas",
            "Analisis de Cotas",
        ])

        # TAB 1 — Arbol
        with tab1:
            st.markdown("<div class='section-header'>Arbol de Ramificacion y Acotamiento</div>", unsafe_allow_html=True)
            st.markdown(LEGEND_HTML, unsafe_allow_html=True)
            fig_tree = build_tree_graph(nodes, num_vars, best_x, is_max)
            st.pyplot(fig_tree, use_container_width=True)
            plt.close(fig_tree)

        # TAB 2 — Grafico 2D
        with tab2:
            st.markdown("<div class='section-header'>Representacion Grafica (2 variables)</div>", unsafe_allow_html=True)
            if num_vars == 2:
                fig2d = plot_2d(c, A, b, senses, nodes, best_x, is_max, var_types)
                st.pyplot(fig2d, use_container_width=True)
            else:
                st.info("El grafico 2D solo esta disponible para modelos con exactamente 2 variables de decision.")

        # TAB 3 — Tabla
        with tab3:
            st.markdown("<div class='section-header'>Tabla de Subproblemas</div>", unsafe_allow_html=True)
            fo_str = ("Max " if is_max else "Min ") + "Z = " + " + ".join([f"{c[i]}x{i+1}" for i in range(num_vars)])
            rows = []
            for n in nodes:
                acc = get_accumulated_constraints(n.id, nodes_dict)
                if n.status == "Infactible":
                    zs, xs = "Infactible", "—"
                else:
                    zs = f"{n.z:.4f}" if n.z is not None else "—"
                    xs = "  ".join([f"x{i+1}={n.x[i]:.3f}" for i in range(num_vars)]) if n.x is not None else "—"
                if n.status == "Solucion Entera":  cota = f"Z* = {n.z:.4f} (actualizada)"
                elif n.status == "Agotado":         cota = "Descartado"
                elif n.status == "Infactible":      cota = "Infactible"
                elif n.status == "Ramificado":      cota = f"Cota sup. = {n.z:.4f}" if n.z else "—"
                else:                               cota = "—"
                rows.append({
                    "Nodo": f"Nodo {n.id}", "F.O.": fo_str,
                    "Restricciones": acc, "Variables": xs,
                    "Z": zs, "Estado": n.status, "Cota": cota,
                })
            render_color_table(rows, list(rows[0].keys()) if rows else [])

        # TAB 4 — Iteraciones
        with tab4:
            st.markdown("<div class='section-header'>Iteraciones Paso a Paso</div>", unsafe_allow_html=True)
            for entry in ilog:
                css = "branched"   if "Ramificado"  in entry["estado"] else \
                      "optimal"    if "Entera"       in entry["estado"] else \
                      "infeasible" if "Infactible"   in entry["estado"] else "pruned"
                z_str = f"Z = {entry['z']:.4f}" if entry.get('z') is not None else "Sin solucion"
                x_str = "  ".join([f"x{i+1}={v:.3f}" for i,v in enumerate(entry.get('x',[]))]) if entry.get('x') else ""
                st.markdown(f"""
                <div class='iter-card {css}'>
                    <div class='iter-title'>Iteracion — Nodo {entry['nodo']}  (profundidad {entry.get('profundidad',0)})</div>
                    <div class='iter-body'>
                        Restriccion aplicada: {entry.get('restriccion','—')}<br>
                        {z_str}  |  {x_str}<br>
                        Estado: {entry['estado']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # TAB 5 — Analisis de Cotas
        with tab5:
            st.markdown("<div class='section-header'>Analisis de Cotas — Evolucion del Algoritmo</div>", unsafe_allow_html=True)
            z_vals     = [e['z'] for e in ilog if e.get('z') is not None]
            best_track = []
            curr_best  = -float('inf') if is_max else float('inf')
            for z in z_vals:
                if (is_max and z > curr_best) or (not is_max and z < curr_best):
                    curr_best = z
                best_track.append(curr_best)
            if z_vals:
                fig_ev, ax_ev = plt.subplots(figsize=(10, 4), facecolor='#f8f7f4')
                ax_ev.set_facecolor('#ffffff')
                ax_ev.plot(range(len(z_vals)), z_vals, 'o--', color='#aab4e8', lw=1.5, ms=5, label='Z en cada nodo')
                ax_ev.plot(range(len(best_track)), best_track, 's-', color='#2d3561', lw=2, ms=6, label='Mejor cota conocida')
                if best_z is not None and best_z != -float('inf'):
                    ax_ev.axhline(best_z, color='#4a7c59', ls=':', lw=2, label=f'Z* = {best_z:.4f}')
                ax_ev.set_xlabel('Iteracion (nodo evaluado)', fontsize=10)
                ax_ev.set_ylabel('Valor Z', fontsize=10)
                ax_ev.set_title('Evolucion de la Cota durante B&B', fontsize=11, fontweight='bold', color='#1a1a2e')
                ax_ev.legend(fontsize=9)
                ax_ev.grid(True, alpha=0.25, linestyle='--')
                ax_ev.spines[['top','right']].set_visible(False)
                st.pyplot(fig_ev, use_container_width=True)
            st.markdown("---")
            st.markdown("**Estadisticas del Arbol de Busqueda**")
            total      = len(nodes)
            n_branched = sum(1 for n in nodes if n.status=="Ramificado")
            n_integer  = sum(1 for n in nodes if n.status=="Solucion Entera")
            n_inf      = sum(1 for n in nodes if n.status=="Infactible")
            n_pruned   = sum(1 for n in nodes if n.status=="Agotado")
            max_depth  = max((n.depth for n in nodes), default=0)
            col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
            col_s1.metric("Total de nodos", total)
            col_s2.metric("Ramificados",    n_branched)
            col_s3.metric("Sol. enteras",   n_integer)
            col_s4.metric("Infactibles",    n_inf)
            col_s5.metric("Agotados",       n_pruned)
            st.markdown(f"**Profundidad maxima del arbol:** {max_depth}")
            fig_bar, ax_bar = plt.subplots(figsize=(7, 3), facecolor='#f8f7f4')
            ax_bar.set_facecolor('#ffffff')
            bars = ax_bar.bar(['Ramificados','Sol. Enteras','Infactibles','Agotados'],
                              [n_branched, n_integer, n_inf, n_pruned],
                              color=['#4a7cb4','#4a7c59','#c0392b','#9e9e9e'], width=0.5, edgecolor='white')
            for bar, val in zip(bars, [n_branched, n_integer, n_inf, n_pruned]):
                ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                           str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax_bar.set_ylabel('Cantidad de nodos', fontsize=10)
            ax_bar.set_title('Distribucion de nodos por estado', fontsize=11, fontweight='bold', color='#1a1a2e')
            ax_bar.spines[['top','right']].set_visible(False)
            ax_bar.grid(True, axis='y', alpha=0.25, linestyle='--')
            st.pyplot(fig_bar, use_container_width=True)

    
    else:
        tab1, tab3, tab4, tab5 = st.tabs([
            "Arbol de Ramificacion",
            "Tabla de Subproblemas",
            "Iteraciones Detalladas",
            "Analisis de Cotas",
        ])

        
        with tab1:
            st.markdown("<div class='section-header'>Arbol de Ramificacion y Acotamiento — Mochila</div>", unsafe_allow_html=True)
            st.markdown(LEGEND_HTML, unsafe_allow_html=True)

            
            id_map = {e['id']: e for e in knap_tree}
            parent_stack = []
            for entry in knap_tree:
                depth = len(entry["included"])
                while len(parent_stack) > depth:
                    parent_stack.pop()
                entry["_parent"] = parent_stack[-1] if parent_stack else None
                if entry["status"] == "Ramificado":
                    parent_stack.append(entry["id"])

            
            STATUS_K = {
                "Solucion Entera": ("#e8f5e9","#388e3c"),
                "Agotado":         ("#f5f5f5","#9e9e9e"),
                "Ramificado":      ("#e3f2fd","#1565c0"),
            }

            
            class KNode:
                pass
            knodes = []
            for entry in knap_tree:
                kn = KNode()
                kn.id        = entry["id"]
                kn.parent_id = entry["_parent"]
                kn.depth     = len(entry["included"])
                kn.status    = entry["status"]
                kn.branch_desc = ""
                sel = [items[i]["name"] for i in entry["included"]] if entry["included"] else ["ninguno"]
                kn.z  = entry["curr_val"]
                kn.x  = None
                
                if entry["_parent"] is not None:
                    par = id_map[entry["_parent"]]
                    new_its = set(entry["included"]) - set(par["included"])
                    kn.branch_desc = f"+{items[list(new_its)[0]]['name']}" if new_its else "excluir"
                knodes.append(kn)

            
            BOX_W_K, BOX_H_K = 2.2, 1.0
            pos_k, children_k, max_lv_k = _assign_positions(knodes)
            all_x_k = [p[0] for p in pos_k.values()]
            fig_width_k  = max(10, (max(all_x_k) - min(all_x_k) + 3.0) * 1.1)
            fig_height_k = max(5, (max_lv_k + 1) * 2.6)

            fig_k, ax_k = plt.subplots(figsize=(fig_width_k, fig_height_k), facecolor="#f8f7f4")
            ax_k.set_facecolor("#f8f7f4")
            ax_k.axis("off")

            for kn in knodes:
                if kn.parent_id is not None and kn.parent_id in pos_k and kn.id in pos_k:
                    px, py = pos_k[kn.parent_id]
                    cx, cy = pos_k[kn.id]
                    ax_k.annotate("", xy=(cx, cy + BOX_H_K/2), xytext=(px, py - BOX_H_K/2),
                        arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.2, mutation_scale=10))
                    mx, my = (px+cx)/2, (py+cy)/2
                    ax_k.text(mx, my, kn.branch_desc, fontsize=6.5, ha="center", va="center",
                              color="#2d3561",
                              bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#cccccc", lw=0.6))

            for kn in knodes:
                if kn.id not in pos_k: continue
                x, y = pos_k[kn.id]
                fc, ec = STATUS_K.get(kn.status, ("#fafafa","#757575"))
                entry = id_map[kn.id]
                box = FancyBboxPatch((x - BOX_W_K/2, y - BOX_H_K/2), BOX_W_K, BOX_H_K,
                    boxstyle="round,pad=0.05", linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3)
                ax_k.add_patch(box)
                sel = [items[i]["name"] for i in entry["included"]] if entry["included"] else ["ninguno"]
                txt = (f"Nodo {kn.id}\n"
                       f"Inc: {sel}\n"
                       f"UB={entry['ub']:.2f}  W={entry['curr_w']:.1f}  V={entry['curr_val']:.1f}\n"
                       f"{kn.status}")
                ax_k.text(x, y, txt, fontsize=6, ha="center", va="center",
                          color="#1a1a2e", zorder=4, multialignment="center", fontfamily="monospace")

            if pos_k:
                xs_k = [p[0] for p in pos_k.values()]
                ys_k = [p[1] for p in pos_k.values()]
                ax_k.set_xlim(min(xs_k)-BOX_W_K, max(xs_k)+BOX_W_K)
                ax_k.set_ylim(min(ys_k)-BOX_H_K*1.5, max(ys_k)+BOX_H_K*1.5)

            ax_k.set_title("Arbol de Ramificacion — Mochila", fontsize=11, fontweight="bold", color="#1a1a2e")
            fig_k.tight_layout(pad=1)
            st.pyplot(fig_k, use_container_width=True)
            plt.close(fig_k)

            
            st.markdown("---")
            st.markdown("**Cocientes c_j / a_j (criterio de ordenamiento por Inspeccion)**")
            ratios_data = []
            for it in items:
                ratio = it['value']/it['weight'] if it['weight'] != 0 else float('inf')
                ratios_data.append({
                    "Articulo": it['name'],
                    "Valor (c_j)": it['value'],
                    "Peso (a_j)":  it['weight'],
                    "Cociente c_j/a_j": f"{ratio:.4f}",
                    "Seleccionado": "Si" if items.index(it) in best_sel else "No"
                })
            ratios_df = pd.DataFrame(ratios_data)
            def color_sel(row):
                if row["Seleccionado"] == "Si": return ["background-color: #e8f5e9"]*len(row)
                return [""]*len(row)
            st.dataframe(ratios_df.style.apply(color_sel, axis=1), use_container_width=True, hide_index=True)

        
        with tab3:
            st.markdown("<div class='section-header'>Tabla de Subproblemas — Mochila</div>", unsafe_allow_html=True)
            fo_knap = "Max Z = " + " + ".join([f"{it['value']}*{it['name']}" for it in items])
            rows_k = []
            for entry in knap_tree:
                sel_names_e = [items[i]['name'] for i in entry["included"]] if entry["included"] else ["ninguno"]
                peso_used   = entry["curr_w"]
                val_used    = entry["curr_val"]
                # Cota
                if entry["status"] == "Solucion Entera":
                    cota_k = f"Z* = {entry['curr_val']:.2f} (actualizada)"
                elif entry["status"] == "Agotado":
                    cota_k = f"Descartado (UB={entry['ub']:.2f} <= Z*)"
                else:
                    cota_k = f"UB = {entry['ub']:.4f}"
                rows_k.append({
                    "Nodo":          f"Nodo {entry['id']}",
                    "F.O.":         fo_knap,
                    "Articulos incluidos": str(sel_names_e),
                    "Peso acum.":   f"{peso_used:.2f} / {capacity}",
                    "Valor acum.":  f"{val_used:.2f}",
                    "UB (Cota sup.)": f"{entry['ub']:.4f}",
                    "Estado / Cota": entry["status"] + " | " + cota_k,
                })
            render_color_table(rows_k, list(rows_k[0].keys()) if rows_k else [])

        
        with tab4:
            st.markdown("<div class='section-header'>Iteraciones Paso a Paso — Mochila</div>", unsafe_allow_html=True)
            for entry in knap_tree:
                css = "branched"   if entry["status"] == "Ramificado"     else \
                      "optimal"    if entry["status"] == "Solucion Entera" else \
                      "pruned"
                sel_e = [items[i]['name'] for i in entry["included"]] if entry["included"] else ["ninguno"]
                decision = ""
                if entry["_parent"] is not None:
                    par = id_map[entry["_parent"]]
                    new_items = set(entry["included"]) - set(par["included"])
                    if new_items:
                        it_idx = list(new_items)[0]
                        decision = f"Incluir {items[it_idx]['name']} (c={items[it_idx]['value']}, a={items[it_idx]['weight']})"
                    else:
                        decision = "Excluir articulo"
                else:
                    decision = "Nodo raiz (sin restriccion)"

                st.markdown(f"""
                <div class='iter-card {css}'>
                    <div class='iter-title'>Iteracion — Nodo {entry['id']}  (nivel {len(entry['included'])})</div>
                    <div class='iter-body'>
                        Decision: {decision}<br>
                        Articulos incluidos: {sel_e}<br>
                        Peso acumulado: {entry['curr_w']:.2f} / {capacity}  |  Valor acumulado: {entry['curr_val']:.2f}<br>
                        Cota superior (UB): {entry['ub']:.4f}<br>
                        Estado: {entry['status']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        
        with tab5:
            st.markdown("<div class='section-header'>Analisis de Cotas — Evolucion del Algoritmo</div>", unsafe_allow_html=True)

            ub_vals    = [e['ub'] for e in knap_tree]
            val_vals   = [e['curr_val'] for e in knap_tree]
            best_track_k = []
            curr_bk = 0
            for e in knap_tree:
                if e['status'] == 'Solucion Entera' and e['curr_val'] > curr_bk:
                    curr_bk = e['curr_val']
                best_track_k.append(curr_bk)

            if ub_vals:
                fig_ek, ax_ek = plt.subplots(figsize=(10, 4), facecolor='#f8f7f4')
                ax_ek.set_facecolor('#ffffff')
                ax_ek.plot(range(len(ub_vals)), ub_vals, 'o--', color='#aab4e8', lw=1.5, ms=5, label='Cota superior (UB)')
                ax_ek.plot(range(len(best_track_k)), best_track_k, 's-', color='#2d3561', lw=2, ms=6, label='Mejor valor entero conocido')
                ax_ek.axhline(best_z, color='#4a7c59', ls=':', lw=2, label=f'Z* = {best_z:.4f}')
                ax_ek.set_xlabel('Nodo evaluado', fontsize=10)
                ax_ek.set_ylabel('Valor', fontsize=10)
                ax_ek.set_title('Evolucion de UB y mejor valor entero — Mochila', fontsize=11, fontweight='bold', color='#1a1a2e')
                ax_ek.legend(fontsize=9)
                ax_ek.grid(True, alpha=0.25, linestyle='--')
                ax_ek.spines[['top','right']].set_visible(False)
                st.pyplot(fig_ek, use_container_width=True)

            st.markdown("---")
            st.markdown("**Estadisticas del Arbol**")
            kn_total   = len(knap_tree)
            kn_branch  = sum(1 for e in knap_tree if e['status']=="Ramificado")
            kn_int     = sum(1 for e in knap_tree if e['status']=="Solucion Entera")
            kn_pruned  = sum(1 for e in knap_tree if e['status']=="Agotado")
            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            col_k1.metric("Total de nodos", kn_total)
            col_k2.metric("Ramificados",    kn_branch)
            col_k3.metric("Sol. enteras",   kn_int)
            col_k4.metric("Agotados",       kn_pruned)

            fig_bk, ax_bk = plt.subplots(figsize=(6, 3), facecolor='#f8f7f4')
            ax_bk.set_facecolor('#ffffff')
            bars_k = ax_bk.bar(['Ramificados','Sol. Enteras','Agotados'],
                               [kn_branch, kn_int, kn_pruned],
                               color=['#4a7cb4','#4a7c59','#9e9e9e'], width=0.5, edgecolor='white')
            for bar, val in zip(bars_k, [kn_branch, kn_int, kn_pruned]):
                ax_bk.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                          str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax_bk.set_ylabel('Cantidad de nodos', fontsize=10)
            ax_bk.set_title('Distribucion de nodos', fontsize=11, fontweight='bold', color='#1a1a2e')
            ax_bk.spines[['top','right']].set_visible(False)
            ax_bk.grid(True, axis='y', alpha=0.25, linestyle='--')
            st.pyplot(fig_bk, use_container_width=True)

   
    if not is_knapsack and best_x is not None:
        st.markdown("---")
        pdf_imgs = []

        if num_vars == 2:
            fig2d_dl = plot_2d(c, A, b, senses, nodes, best_x, is_max, var_types)
            buf1 = io.BytesIO()
            fig2d_dl.savefig(buf1, format='png', bbox_inches='tight', dpi=150)
            buf1.seek(0)
            pdf_imgs.append(Image.open(buf1).convert('RGB'))
            plt.close(fig2d_dl)

        fig_tree_dl = build_tree_graph(nodes, num_vars, best_x, is_max)
        buf2 = io.BytesIO()
        fig_tree_dl.savefig(buf2, format='png', bbox_inches='tight', dpi=150)
        buf2.seek(0)
        pdf_imgs.append(Image.open(buf2).convert('RGB'))
        plt.close(fig_tree_dl)

        if pdf_imgs:
            pdf_buf = io.BytesIO()
            pdf_imgs[0].save(pdf_buf, format='PDF', save_all=True, append_images=pdf_imgs[1:])
            st.download_button(
                label="Descargar Reporte PDF (Grafico + Arbol)",
                data=pdf_buf.getvalue(),
                file_name="Reporte_BnB.pdf",
                mime="application/pdf",
            )

else:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#888;'>
        <div style='font-size:48px; margin-bottom:16px; opacity:0.3;'>◆</div>
        <div style='font-size:16px; font-weight:500; color:#555;'>Configure el modelo en el panel lateral y presione <strong>Resolver Modelo</strong></div>
        <div style='font-size:13px; color:#aaa; margin-top:8px;'>Soporta modelos Entero Puro (PIP), Mixto (MIP), Binario (BIP) y Mochila (Knapsack)</div>
    </div>
    """, unsafe_allow_html=True)

    # Referencia teorica
    with st.expander("Referencia: Notacion y Formulas del Metodo"):
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("""
**Notacion principal**

- **x_j**: Variable de decision fraccionaria seleccionada para ramificar
- **x_{Bj}**: Valor fraccionario de x_j en la solucion relajada
- **[x_{Bj}]**: Parte entera (floor) del valor fraccionario
- **Z**: Valor de F.O. en el subproblema actual
- **Z_cota**: Mejor solucion entera conocida (cota)

**Reglas de Ramificacion**

*Entero puro / Mixto:*
- Rama izquierda: x_j <= [x_{Bj}]
- Rama derecha:   x_j >= [x_{Bj}] + 1

*Binario:*
- Rama izquierda: x_j = 0
- Rama derecha:   x_j = 1
""")
        with col_r2:
            st.markdown("""
**Reglas de Acotamiento (Sondeo)**

*Actualizar Z_cota:*
- Maximizacion: si Z_cota < Z → Z_cota = Z
- Minimizacion: si Z_cota > Z → Z_cota = Z

*Descartar subproblema:*
- Maximizacion: descartar si Z_cota >= Z
- Minimizacion: descartar si Z_cota <= Z

**Mochila**

- F.O.: max z = c1*x1 + ... + cn*xn
- Restriccion: a1*x1 + ... + an*xn <= W
- x_j ∈ {0, 1}
- Cota superior por cociente c_j/a_j (Inspeccion)
""")