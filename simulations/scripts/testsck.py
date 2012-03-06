## Test SCK foreground generation

from simulations import foregroundsck

s = foregroundsck.Synchrotron()
sf = s.getfield()

x = foregroundsck.ExtraGalacticFreeFree()
xf = x.getfield()

g = foregroundsck.GalacticFreeFree()
gf = g.getfield()

p = foregroundsck.PointSources()
pf = p.getfield()

