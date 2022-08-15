# Install DART (python)
## Dart dependencies
```sh
    sudo apt-get -y update
    # dart dependencies (see: https://dartsim.github.io/install_dartpy_on_ubuntu.html)
    sudo apt-get -y install build-essential cmake pkg-config git #install build tools
    sudo apt-get install libpoco-dev 
    sudo apt-get install libeigen3-dev  #install Eigen 3.4.0.
    sudo apt-get install libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev
    sudo apt-get install libboost-filesystem-dev #required since Ubuntu 22.04 (?)
    sudo apt-get install libtinyxml2-dev liburdfdom-dev
    sudo apt-get install libxi-dev libxmu-dev freeglut3-dev libopenscenegraph-dev # install visualization libs
    sudo apt-get install pybind11-dev # install pybind for Ubuntu 19.04 and newer
    sudo apt-get install libbullet-dev #(optional if Bullet Collision Detector is used)
    sudo apt-get install libnlopt-cxx-dev
```

## install dart
```sh
cd ~/install
git clone https://github.com/dartsim/dart.git
cd dart
git checkout v6.12.1
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release -DDART_BUILD_DARTPY=ON
make -j dartpy
sudo make install dartpy
```
## test darpy
```sh
import dartpy as dart
world = dart.simulation.World()
world.step()
```

```sh
# tests if Dartpy can be run with viewer
import dartpy as dart

if __name__== "__main__":
    world = dart.io.SkelParser.readWorld("dart://sample/skel/cubes.skel")
    world.setGravity([0, -9.81, 0])

    node = dart.gui.osg.RealTimeWorldNode(world)

    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)

    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([0.8, 0.0, 0.8], [0, -0.25, 0], [0, 0.5, 0])
    viewer.run()

```
## Troubleshooting
### Cannot import dartpy
Make sure the dartpy (.so file) is in the python path.
If you built with `-DCMAKE_INSTALL_PREFIX=/usr` it should be located under 
```sh
/usr/dartpycpython-310-x86_64-linux-gnu.so
```
If this is not in your python path you can add it to the path, or copy the file to a location in your python path, e.g.
```sh
cp /usr/dartpycpython-310-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/
```

# Install libfranka
```sh
cd ~/install
git clone --recursive https://github.com/frankaemika/libfranka
cd libfranka
#git checkout tags/0.8.0
git submodule update
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
sudo make install
```

## Troubleshooting
### std import error when building with tag/0.8.0
add required libraries

# Install libvisiontransfer
wget tar \

# Install pyDataverse 
```sh
cd data/darus_data_download
pip install --user -r requirements.txt
```