import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

class STEMRenderer:
    def __init__(self, env):
        self.env = env
        pygame.init()
        self.display = (800, 800)
        self.screen = pygame.display.set_mode(
            self.display, pygame.DOUBLEBUF|pygame.OPENGL)
        gluOrtho2D(0, env.grid_size, 0, env.grid_size)
        
        self.colors = {
            0: (0.95, 0.95, 0.95),  # Empty
            1: (0.2, 0.5, 0.8),     # Agent
            2: (0.9, 0.6, 0.9),     # Student
            3: (0.8, 0.2, 0.2),     # Barrier
            4: (0.4, 0.8, 0.4),     # Resource
            5: (0.4, 0.4, 0.4)      # Dropout
        }

    def render(self, stats=None):
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Draw grid
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                glColor3f(*self.colors[self.env.grid[x, y]])
                glBegin(GL_QUADS)
                glVertex2f(x, y)
                glVertex2f(x+1, y)
                glVertex2f(x+1, y+1)
                glVertex2f(x, y+1)
                glEnd()
        
        pygame.display.flip()
        pygame.time.wait(50)
    
    def close(self):
        pygame.quit()
