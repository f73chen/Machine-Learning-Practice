import numpy as np


class Blob:
    # starts randomly somewhere on the 10x10 grid

    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    # overload the subtraction method to get distance between two Blobs
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    # moving the Blob
    def move(self, x=False, y=False):
        # if no x given, move x randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # if no y given, move y randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # fix location if out of bounds
        if self.x < 0:
            self.x = 0
        elif self.x > self.size - 1:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size - 1:
            self.y = self.size - 1

    # moves the Blob according to one of four action choices
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
