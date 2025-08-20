from enum import IntEnum
import pygame

Vector2D = pygame.Vector2


class CollisionType(IntEnum):
    ENTITY = 1
    AGENT = 2
    OBSTACLE = 3
