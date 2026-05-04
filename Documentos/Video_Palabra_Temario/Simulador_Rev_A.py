import pygame
import sys


class Actividad:
    def __init__(self, id, t, x, y):
        self.id = id
        self.t = t         
        
        self.ls = None     
        self.lf = None     
        
        self.sucesores = [] 
        
        
        self.x = x
        self.y = y
        self.procesado = False
        self.activo = False


A = Actividad('A', 2, 200, 300)
B = Actividad('B', 5, 450, 150)
C = Actividad('C', 3, 450, 450)
D = Actividad('D', 4, 700, 300)


A.sucesores = [B, C]
B.sucesores = [D]
C.sucesores = [D]
D.sucesores = [] 

nodos = [A, B, C, D]


duracion_proyecto = 12 

# ORDENAMIENTO TOPOLÓGICO INVERSO
top_sort_inverso = [D, C, B, A]



pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulador CPM: Revisión Hacia Atrás (Nuevo Grafo)")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (100, 150, 255)
ORANGE = (255, 165, 0)
GREEN = (100, 255, 100)
RED = (255, 50, 50)

fuente = pygame.font.SysFont("arial", 20, bold=True)
fuente_peq = pygame.font.SysFont("arial", 16, bold=True)
fuente_calc = pygame.font.SysFont("arial", 18)

def dibujar_red():
    screen.fill(WHITE)
    
    # Título e Instrucciones
    titulo = fuente.render("CPM: Revisión Hacia Atrás", True, BLACK)
    instruccion = fuente_peq.render("Presiona ESPACIO para procesar el siguiente nodo", True, GRAY)
    screen.blit(titulo, (50, 30))
    screen.blit(instruccion, (50, 60))

    
    if paso_actual > 0 and paso_actual <= len(top_sort_inverso):
        nodo_calculado = top_sort_inverso[paso_actual - 1]
        texto_calc1 = fuente_calc.render(f"Último cálculo en {nodo_calculado.id}:", True, BLACK)
        texto_calc2 = fuente_calc.render(f"LF = {nodo_calculado.lf}", True, RED)
        texto_calc3 = fuente_calc.render(f"LS = {nodo_calculado.lf} - {nodo_calculado.t} = {nodo_calculado.ls}", True, RED)
        screen.blit(texto_calc1, (750, 30))
        screen.blit(texto_calc2, (750, 60))
        screen.blit(texto_calc3, (750, 90))

    # Dibujar Aristas
    for nodo in nodos:
        for sucesor in nodo.sucesores:
            pygame.draw.line(screen, GRAY, (nodo.x, nodo.y), (sucesor.x, sucesor.y), 4)

    # Dibujar Nodos
    for nodo in nodos:
        color_nodo = BLUE
        if nodo.activo:
            color_nodo = ORANGE
        elif nodo.procesado:
            color_nodo = GREEN

        pygame.draw.circle(screen, color_nodo, (nodo.x, nodo.y), 45)
        pygame.draw.circle(screen, BLACK, (nodo.x, nodo.y), 45, 3) 
        
        texto_id = fuente.render(f"{nodo.id} (t={nodo.t})", True, BLACK)
        screen.blit(texto_id, (nodo.x - 30, nodo.y - 12))

        if nodo.procesado or nodo.activo:
            texto_backward = fuente_peq.render(f"[LS={nodo.ls} , LF={nodo.lf}]", True, RED)
            screen.blit(texto_backward, (nodo.x - 55, nodo.y + 55))
        else:
            texto_vacio = fuente_peq.render("[LS=? , LF=?]", True, GRAY)
            screen.blit(texto_vacio, (nodo.x - 45, nodo.y + 55))

paso_actual = 0
reloj = pygame.time.Clock()

ejecutando = True
while ejecutando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False
            
        if evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_SPACE:
                if paso_actual < len(top_sort_inverso):
                    for n in nodos:
                        n.activo = False
                        
                    nodo_actual = top_sort_inverso[paso_actual]
                    nodo_actual.activo = True
                    
                    
                    if len(nodo_actual.sucesores) == 0:
                        nodo_actual.lf = duracion_proyecto
                    else:
                        tiempos_sucesores = [s.ls for s in nodo_actual.sucesores if s.ls is not None]
                        nodo_actual.lf = min(tiempos_sucesores)
                    
                    nodo_actual.ls = nodo_actual.lf - nodo_actual.t
                    
                    nodo_actual.procesado = True
                    paso_actual += 1
                else:
                    for n in nodos:
                        n.activo = False

    dibujar_red()
    pygame.display.flip()
    reloj.tick(30)

pygame.quit()
sys.exit()