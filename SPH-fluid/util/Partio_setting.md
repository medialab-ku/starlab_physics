# export_mesh.py info

In run_simulation.py

```angular2html
~~~
runSim = False
PRINTMESH = True # True : export, False : not export

if PRINTMESH:
    exporter = Exporter(folder="./data/output", frameInterval=5) 
    # param : folder : output folder
    #         frameInterval : export interval
    
    # exporter.set_faces(ps.faces_st)
    # uncomment the line when you want to export obj with faces 

~~~ # in window.running loop
        if PRINTMESH:
            # exporter.export_mesh("scene.obj", ps.x, MODE="PARTICLE")
            # uncomment the line when you want to export obj with faces 
            
            # exporter.export_ply("scene.obj", ps.x, MODE="MULTI")
            # exporting the line when you want to export obj only with vertices
            
            exporter.export_bgeo("scene.bgeo", ps.x, MODE="MULTI")
            # bgeo export with partio 
            # Param : MODE = "SINGLE" : overwrite scene.bgeo every loop
            # Param : MODE = "MULTI" : scene + frame(int) with frame interval + .bgeo 
            # ex) frameInterval = 5 -> scene5.bgeo, scene10.bgeo, scene15.bgeo, scene 20.bgeo...
```

## Partio error 

when you use partio error occur (import partio error)

```angular2html
cd SPH-fluid/util
git clone https://github.com/wdas/partio.git
```

then change folder name to Partio (from partio)

```
cd Partio
sudo apt install swig
mkdir build && cd build
cmake ..
make -j$(nproc)
```

there will be _partio.so, partio.py in build/src/py

and also libpartio.so, libpartio.so.1 in build/src/lib

copy all of them and paste to util folder

then you will face the error about library  
goto edit configuration in Pycharm and paste below to Environment Variable(E) 

```
PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=/home/media/Desktop/starlab_physics/SPH-fluid/util:$LD_LIBRARY_PATH
```

then all error will be fixed








