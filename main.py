from ray.frame import Frame
import os

frame = Frame(1000, 1000)
frame.cast_rays()

frame.write("test.png")
