
import math
import time
import threading
import numpy as np
from scipy.optimize import linprog
import pygame
import pygame.font


BG          = (245, 244, 240)
TEXT_DARK   = (30,  30,  28)
TEXT_MED    = (90,  88,  84)
TEXT_LIGHT  = (160, 158, 152)

NODE_OPEN   = {"fill": (230, 241, 251), "border": (24,  95, 165), "text": (12,  68, 124)}
NODE_INT    = {"fill": (234, 243, 222), "border": (59, 109,  17), "text": (39,  80,  10)}
NODE_PRUNE  = {"fill": (252, 235, 235), "border": (163, 45,  45), "text": (121, 31,  31)}
NODE_ACTIVE = {"fill": (250, 238, 218), "border": (133, 79,  11), "text": (99,  56,   6)}
NODE_BEST   = {"fill": (209, 250, 229), "border": (16, 185, 129), "text": ( 6, 95,  70)}

EDGE_LEFT   = (24,  95, 165)
EDGE_RIGHT  = (163, 45,  45)
PANEL_BG    = (255, 255, 255)
PANEL_BORDER= (200, 198, 192)
LOG_BG      = (250, 249, 245)


def solve_lp(bounds):

    c  = [-5, -4]
    Au = [[1, 1], [10, 6]]
    bu = [5, 45]
    bds = [(b[0], b[1]) for b in bounds]
    res = linprog(c, A_ub=Au, b_ub=bu, bounds=bds, method='highs')
    if not res.success:
        return None
    return -res.fun, res.x


def is_int(v, tol=1e-5):
    return abs(v - round(v)) < tol



class BBNode:
    _counter = 0

    def __init__(self, bounds, parent_id=None, side=None, depth=0):
        BBNode._counter += 1
        self.id        = BBNode._counter
        self.bounds    = bounds
        self.parent_id = parent_id
        self.side      = side         
        self.depth     = depth
        self.status    = 'pending'     
        self.z         = None
        self.x         = None
        self.label     = f"Nodo {self.id}"
        # Layout
        self.cx = 0
        self.cy = 0
        self.children  = []           
        self.visible   = False


class BranchAndBoundViz:
    def __init__(self):
        self.nodes: dict[int, BBNode] = {}
        self.root_id   = None
        self.best_z    = -math.inf
        self.best_x    = None
        self.steps     = []            
        self.step_idx  = 0
        self.log_lines = []            
        self._build()

   
    def _build(self):
        BBNode._counter = 0
        self._add_steps([[0, None], [0, None]], None, None, 0)

    def _add_steps(self, bounds, parent_id, side, depth):
        node = BBNode(bounds, parent_id, side, depth)
        nid  = node.id
        self.nodes[nid] = node
        if parent_id is not None:
            self.nodes[parent_id].children.append(nid)
        if self.root_id is None:
            self.root_id = nid

        def reveal(n=node):
            n.visible = True
            self._layout()

        def activate(n=node):
            n.status = 'open'

        res = solve_lp(bounds)

        if res is None:
            def mark_inf(n=node):
                n.status  = 'prune_inf'
                n.label   = f"N{n.id}\nInfactible"
                self._log("✗ Podado — Infactible", NODE_PRUNE["border"])
            self.steps += [reveal, activate, mark_inf]
            return

        z, x = res
        zr = round(z, 2)
        x1r = round(x[0], 3)
        x2r = round(x[1], 3)

        if z <= self.best_z + 1e-8:
            def mark_bound(n=node, zz=zr):
                n.status = 'prune_bound'
                n.label  = f"N{n.id}  Z={zz}\nPodado ≤ Z*"
                self._log(f"✗ N{n.id} Podado cota Z={zz}", NODE_PRUNE["border"])
            self.steps += [reveal, activate, mark_bound]
            return

        all_int = is_int(x[0]) and is_int(x[1])

        if all_int:
            if z > self.best_z:
                self.best_z = z
                self.best_x = [round(v) for v in x]
            bz = round(self.best_z, 2)
            bx = list(self.best_x)

            def mark_int(n=node, zz=zr, xx=[x1r, x2r], bz=bz, bx=bx):
                n.status = 'integer'
                n.label  = f"N{n.id}  Z={zz}\nx=[{xx[0]},{xx[1]}]"
                self._log(f"★ N{n.id} Entera Z={zz}  x={xx}", NODE_INT["border"])
            self.steps += [reveal, activate, mark_int]
            return

        
        bi = next(i for i, v in enumerate(x) if not is_int(v))
        bval = x[bi]

        def mark_open(n=node, zz=zr, xx=[x1r, x2r], bvar=bi, bv=bval):
            n.z      = zz
            n.x      = xx
            n.status = 'open'
            n.label  = f"N{n.id}  Z={zz}\nx=[{xx[0]},{xx[1]}]"
            self._log(
                f"→ N{n.id}  Z={zz}  x=[{xx[0]},{xx[1]}]  "
                f"ramifica x{bvar+1}={round(bv,3)}",
                NODE_OPEN["border"]
            )
        self.steps += [reveal, activate, mark_open]

        
        left  = [list(b) for b in bounds]
        left[bi][1] = math.floor(bval)
        self._add_steps(left, nid, 'left', depth + 1)

        
        right = [list(b) for b in bounds]
        right[bi][0] = math.ceil(bval)
        self._add_steps(right, nid, 'right', depth + 1)

    def _log(self, text, color=TEXT_DARK):
        self.log_lines.append((text, color))
        if len(self.log_lines) > 200:
            self.log_lines.pop(0)

   
    def _layout(self):
        visible = [n for n in self.nodes.values() if n.visible]
        if not visible:
            return

        
        leaves = self._leaf_order(self.root_id, visible_only=True)
        NW = 110

        for idx, nid in enumerate(leaves):
            self.nodes[nid].cx = 40 + idx * (NW + 16) + NW // 2

        
        def assign_parent(nid):
            n = self.nodes[nid]
            vis_ch = [c for c in n.children if self.nodes[c].visible]
            if vis_ch:
                for c in vis_ch:
                    assign_parent(c)
                n.cx = sum(self.nodes[c].cx for c in vis_ch) // len(vis_ch)
            n.cy = 60 + n.depth * 80

        if self.root_id in self.nodes:
            assign_parent(self.root_id)

    def _leaf_order(self, nid, visible_only=False):
        n = self.nodes.get(nid)
        if n is None:
            return []
        if visible_only and not n.visible:
            return []
        vis_ch = [c for c in n.children if self.nodes.get(c) and (not visible_only or self.nodes[c].visible)]
        if not vis_ch:
            return [nid]
        result = []
        for c in vis_ch:
            result += self._leaf_order(c, visible_only)
        return result

    def advance(self):
        if self.step_idx < len(self.steps):
            self.steps[self.step_idx]()
            self.step_idx += 1
            return True
        return False

    def done(self):
        return self.step_idx >= len(self.steps)



class Renderer:
    NW, NH = 112, 46
    FONT_SIZE_BIG  = 14
    FONT_SIZE_MED  = 12
    FONT_SIZE_SM   = 11
    LOG_LINES      = 14
    PANEL_W        = 260

    def __init__(self, width=1100, height=700):
        pygame.init()
        pygame.display.set_caption("Branch and Bound — Visualización")
        self.W, self.H = width, height
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)

        # Fuentes
        mono = pygame.font.match_font("consolas,monospace,dejavusansmono")
        sans = pygame.font.match_font("segoeui,helvetica,arial")
        self.f_node   = pygame.font.Font(mono, self.FONT_SIZE_SM)
        self.f_log    = pygame.font.Font(mono, self.FONT_SIZE_MED)
        self.f_title  = pygame.font.Font(sans, self.FONT_SIZE_BIG)
        self.f_hint   = pygame.font.Font(sans, self.FONT_SIZE_SM)

        self.bb       = BranchAndBoundViz()
        self.auto     = False
        self.speed    = 3              
        self._last_auto = 0.0
        self.scroll_y = 0             
        self.log_scroll = 0

        
        self.tree_surf_w = 1800
        self.tree_surf_h = 900
        self.tree_surf   = pygame.Surface((self.tree_surf_w, self.tree_surf_h))

        
        self.bb.advance()

   
    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.VIDEORESIZE:
                    self.W, self.H = event.w, event.h
                    self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
                elif event.type == pygame.MOUSEWHEEL:
                    self.scroll_y = max(0, self.scroll_y - event.y * 20)

            
            now = time.time()
            if self.auto and not self.bb.done():
                if now - self._last_auto >= 1.0 / self.speed:
                    self.bb.advance()
                    self._last_auto = now
                    if self.bb.done():
                        self.auto = False

            self._draw()
            clock.tick(60)

        pygame.quit()

    def _handle_key(self, key):
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key == pygame.K_SPACE:
            if not self.bb.done():
                self.bb.advance()
        if key == pygame.K_a:
            self.auto = not self.auto
            self._last_auto = time.time()
        if key == pygame.K_r:
            self.bb     = BranchAndBoundViz()
            self.auto   = False
            self.scroll_y = 0
            self.bb.advance()
        if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(self.speed + 1, 8)
        if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(self.speed - 1, 1)
        return True

    
    def _draw(self):
        self.screen.fill(BG)

        tree_area_w = self.W - self.PANEL_W - 16
        tree_area_h = self.H - 40

        
        self._draw_panel(tree_area_w + 8, 8, self.PANEL_W - 16, self.H - 16)

        
        self._draw_statusbar()

        
        self._draw_tree(tree_area_w, tree_area_h)

        pygame.display.flip()

    def _draw_tree(self, area_w, area_h):
       
        clip = pygame.Rect(8, 8, area_w - 8, area_h - 8)
        self.screen.set_clip(clip)

        self.tree_surf.fill(BG)

        
        pygame.draw.rect(self.tree_surf, (240, 239, 234), (0, 0, self.tree_surf_w, self.tree_surf_h))

        
        for nid, node in self.bb.nodes.items():
            if not node.visible or node.parent_id is None:
                continue
            parent = self.bb.nodes.get(node.parent_id)
            if parent and parent.visible:
                color = EDGE_LEFT if node.side == 'left' else EDGE_RIGHT
                x0, y0 = parent.cx, parent.cy + self.NH // 2
                x1, y1 = node.cx,  node.cy  - self.NH // 2
                pygame.draw.line(self.tree_surf, color, (x0, y0 - self.scroll_y), (x1, y1 - self.scroll_y), 1)
                
                label = f"≤{node.bounds[0][1]}" if node.side == 'left' else f"≥{node.bounds[0][0]}"
                mx = (x0 + x1) // 2 + (-18 if node.side == 'left' else 6)
                my = (y0 + y1) // 2 - self.scroll_y
                surf = self.f_hint.render(label, True, color)
                self.tree_surf.blit(surf, (mx, my))

        
        for nid, node in self.bb.nodes.items():
            if node.visible:
                self._draw_node(self.tree_surf, node)

        
        self.screen.blit(self.tree_surf, (8, 8))
        self.screen.set_clip(None)

        
        pygame.draw.rect(self.screen, PANEL_BORDER, (8, 8, area_w - 8, area_h - 8), 1, border_radius=6)

    def _draw_node(self, surf, node):
        col = NODE_OPEN
        if node.status == 'integer':    col = NODE_INT
        if node.status == 'prune_inf':  col = NODE_PRUNE
        if node.status == 'prune_bound':col = NODE_PRUNE
        if node.status == 'pending':    col = {"fill": BG, "border": TEXT_LIGHT, "text": TEXT_LIGHT}

        
        last_visible = max((n for n in self.bb.nodes.values() if n.visible), key=lambda n: n.id, default=None)
        if last_visible and node.id == last_visible.id and not self.bb.done():
            col = NODE_ACTIVE

        
        if self.bb.best_x and node.status == 'integer' and node.z == round(self.bb.best_z, 2):
            col = NODE_BEST

        x = node.cx - self.NW // 2
        y = node.cy - self.NH // 2 - self.scroll_y

        
        shadow = pygame.Surface((self.NW, self.NH), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 18))
        surf.blit(shadow, (x + 2, y + 2))

        rect = pygame.Rect(x, y, self.NW, self.NH)
        pygame.draw.rect(surf, col["fill"],   rect, border_radius=8)
        pygame.draw.rect(surf, col["border"], rect, 1, border_radius=8)

        lines = node.label.split('\n')
        total_h = len(lines) * 14
        start_y = y + (self.NH - total_h) // 2 + 6
        for i, ln in enumerate(lines):
            s = self.f_node.render(ln, True, col["text"])
            surf.blit(s, (x + (self.NW - s.get_width()) // 2, start_y + i * 14))

    def _draw_panel(self, px, py, pw, ph):
        panel_rect = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(self.screen, PANEL_BG,    panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel_rect, 1, border_radius=8)

        y = py + 12
        
        t = self.f_title.render("Branch & Bound", True, TEXT_DARK)
        self.screen.blit(t, (px + 12, y))
        y += 22

        t2 = self.f_hint.render("Max 5x₁ + 4x₂", True, TEXT_MED)
        self.screen.blit(t2, (px + 12, y))
        y += 14
        t3 = self.f_hint.render("s.a. x₁+x₂≤5,  10x₁+6x₂≤45", True, TEXT_MED)
        self.screen.blit(t3, (px + 12, y))
        y += 20

        pygame.draw.line(self.screen, PANEL_BORDER, (px + 12, y), (px + pw - 12, y))
        y += 10

        
        if self.bb.best_z > -math.inf:
            bz = round(self.bb.best_z, 2)
            bx = self.bb.best_x
            lbl = self.f_title.render(f"Z* = {bz}", True, NODE_INT["border"])
            self.screen.blit(lbl, (px + 12, y))
            y += 20
            lx = self.f_hint.render(f"x = {bx}", True, NODE_INT["text"])
            self.screen.blit(lx, (px + 12, y))
            y += 20
        else:
            lbl = self.f_hint.render("Sin solución aún…", True, TEXT_LIGHT)
            self.screen.blit(lbl, (px + 12, y))
            y += 20

        pygame.draw.line(self.screen, PANEL_BORDER, (px + 12, y), (px + pw - 12, y))
        y += 10

        
        total = len(self.bb.steps)
        done  = self.bb.step_idx
        pct   = done / total if total else 0
        bar_w = pw - 24
        pygame.draw.rect(self.screen, (220, 218, 212), (px + 12, y, bar_w, 6), border_radius=3)
        pygame.draw.rect(self.screen, NODE_INT["border"], (px + 12, y, int(bar_w * pct), 6), border_radius=3)
        y += 14
        pt = self.f_hint.render(f"Paso {done}/{total}", True, TEXT_MED)
        self.screen.blit(pt, (px + 12, y))
        y += 20

        pygame.draw.line(self.screen, PANEL_BORDER, (px + 12, y), (px + pw - 12, y))
        y += 10

        
        log_h = ph - (y - py) - 12
        lt = self.f_title.render("Log", True, TEXT_DARK)
        self.screen.blit(lt, (px + 12, y))
        y += 18

        log_clip = pygame.Rect(px + 8, y, pw - 16, log_h - 18)
        self.screen.set_clip(log_clip)

        line_h = 16
        visible_lines = log_h // line_h - 2
        start = max(0, len(self.bb.log_lines) - visible_lines)
        lines = self.bb.log_lines[start:]

        for i, (text, color) in enumerate(lines):
            s = self.f_log.render(text[:35], True, color)
            self.screen.blit(s, (px + 10, y + i * line_h))

        self.screen.set_clip(None)

    def _draw_statusbar(self):
        y = self.H - 30
        hints = [
            "ESPACIO paso",
            "A auto",
            "R reiniciar",
            "+/- velocidad",
            "Q salir",
            "Scroll ↕ árbol",
        ]
        x = 16
        for h in hints:
            surf = self.f_hint.render(h, True, TEXT_LIGHT)
            self.screen.blit(surf, (x, y + 8))
            x += surf.get_width() + 20

        
        spd = self.f_hint.render(f"vel: {self.speed}x {'▶▶' if self.auto else '⏸'}", True, TEXT_MED)
        self.screen.blit(spd, (self.W - self.PANEL_W - 30, y + 8))



if __name__ == "__main__":
    app = Renderer(width=1200, height=750)
    app.run()