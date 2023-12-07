from ray.frame import Frame
import params
import sys
from numba import jit, config

config.DISABLE_JIT = params.do_jit

frame = Frame(params.window_width, params.window_height)
frame.build_scene(sys.argv[1])

print("rendering scene")
print("number of geometries ", len(frame.scene.geometry))
frame.cast_rays()
frame.add_background()
print(" done")

frame.write("test.png")
