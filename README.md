# DensNearOdOH_HPC

<img src ="https://github.com/chen3262/DensNearOdOH_HPC/blob/master/pic.png" width="600">

This is a CUDA program to calculate regional molardular densities from [GROMACS](http://www.gromacs.org/) trajectory files (.trr) trajectory files (.trr). File transferring and calculations on GPU are overlapped to minimize overhead. Performance optimzations was based on [Nividia GTX 780](http://www.nvidia.com/gtx-700-graphics-cards/gtx-780/). This code requires the [XTC Libray](http://www.gromacs.org/Developer_Zone/Programming_Guide/XTC_Library)

## Requirements
[xdrfile](http://www.gromacs.org/Developer_Zone/Programming_Guide/XTC_Library)>=1.1.4 is required.

## Compilation

After compiling the xdrfile-1.1.4 library on the local machine, cd into this repository. Then:

```bash
nvcc -arch=sm_30 Dist.cu DensNearOdOH.cu -o DensNearOdOH.exe -I "path to xdrfile header files" -L "path to xdrfile library" -lxdrfile
```

## Testing

To test your build, do:

```bash
./DensNearOdOH.exe TRR GRO molinfo output.dens log.txt
```

## License

Copyright (C) 2017 Si-Han Chen chen.3262@osu.edu
