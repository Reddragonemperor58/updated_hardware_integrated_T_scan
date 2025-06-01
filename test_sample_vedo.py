# test_simple_vedo.py
import vedo
from vedo import Box, show, settings, Mesh
import vtk

print(f"Vedo version in test_simple_vedo.py: {vedo.__version__}")
print(f"VTK version in test_simple_vedo.py: {vtk.VTK_VERSION}")

b = Box()
b.color('red')
b.name = "MySimpleBox"

print(f"Box object: {b}")
print(f"Is 'b' a vedo.Mesh? {isinstance(b, Mesh)}")

valid_mapper_and_input = False
try:
    print("Accessing b.mapper (as property)...")
    m = b.mapper  # ACCESS AS PROPERTY
    print(f"b.mapper is: {m}")
    print(f"Type of m: {type(m)}")

    if not m:
        print("ERROR: Simple Box has NO MAPPER (mapper property is None).")
    elif not isinstance(m, vtk.vtkMapper):
        print(f"ERROR: b.mapper did not return a VTK mapper object. Got type: {type(m)}")
    elif m.GetInput() is None: # GetInput() IS a method of vtkMapper
        print("ERROR: Simple Box mapper has NO INPUT DATA OBJECT.")
    else:
        print("SUCCESS: Simple Box mapper AND ITS INPUT SEEM VALID.")
        valid_mapper_and_input = True
except AttributeError:
    print("ERROR: b.mapper does not exist as an attribute/property.")
except Exception as e:
    print(f"ERROR: Simple Box General exception during mapper check: {e}")

if valid_mapper_and_input:
    show(b, axes=1, title="Simple Vedo Box Test - Property Access")
else:
    print("Not showing box due to mapper/input issue.")