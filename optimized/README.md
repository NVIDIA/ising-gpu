# Optimized CUDA implementation

To compile the code simply adjust the Makefile to point to your CUDA
installation and specify the CUDA architecture you want  to compile for. A
simple `make` should be enough to produce the ``cuIsing`` binary.

## Usage

<PRE>
Usage: cuIsing [options]
options:
        -x|--x &lt;HORIZ_DIM&gt;
                Specifies the horizontal dimension of the entire  lattice  (black+white  spins),
                per GPU. This dimension must be a multiple of 2048.

        -y|--y &lt;VERT_DIM&gt;
                Specifies the vertical dimension of the entire lattice (black+white spins),  per
                GPU. This dimension must be a multiple of 16.

        -n|--n &lt;NSTEPS&gt;
                Specifies the number of iteration to run.
                Defualt: 1

        -d|--devs &lt;NUM_DEVICES&gt;
                Specifies the number of GPUs to use. Will use devices with ids [0, NUM_DEVS-1].
                Defualt: 1.

        -s|--seed &lt;SEED&gt;
                Specifies the seed used to generate random numbers.
                Default: 463463564571

        -a|--alpha &lt;ALPHA&gt;
                Specifies the temperature in T_CRIT units.  If both this  option  and  '-t'  are
                specified then the '-t' option is used.
                Default: 0.100000

        -t|--temp &lt;TEMP&gt;
                Specifies the temperature in absolute units.  If both this option and  '-a'  are
                specified then this option is used.
                Default: 0.226919

        -p|--print &lt;STAT_FREQ&gt;
                Specifies the frequency, in no.  of  iteration,  with  which  the  magnetization
                statistics is printed.  If this option is used together to the '-e' option, this
                option is ignored.
                Default: only at the beginning and at end of the simulation

        -e|--exppr
                Prints the magnetization at time steps in the series 0 &lt;= 2^(x/4) &lt; NSTEPS.   If
                this option is used  together  to  the  '-p'  option,  the  latter  is  ignored.
                Default: disabled

        -c|--corr
                Dumps to a  file  named  corr_{X}x{Y}_T_{TEMP}  the  correlation  of each  point
                with the  128 points on the right and below.  The correlation is computed  every
                time the magnetization is printed on screen (based on either the  '-p'  or  '-e'
                option) and it is written in the file one line per measure.
                Default: disabled

        -m|--magn &lt;TGT_MAGN&gt;
                Specifies the magnetization value at which the simulation is  interrupted.   The
                magnetization of the system is checked against TGT_MAGN every STAT_FREQ, if  the
                '-p' option is specified, or according to the exponential  timestep  series,  if
                the '-e' option is specified.  If neither '-p' not '-e' are specified then  this
                option is ignored.
                Default: unset

        -J|--J &lt;PROB&gt;
                Specifies the probability [0.0-1.0] that links  connecting  any  two  spins  are
                anti-ferromagnetic. 
                Default: 0.0

           --xsl &lt;HORIZ_SUB_DIM&gt;
                Specifies the horizontal dimension of each sub-lattice (black+white spins),  per
                GPU.  This dimension must be a divisor of the horizontal dimension of the entire
                lattice per  GPU  (specified  with  the  '-x'  option) and a multiple of 2048.
                Default: sub-lattices are disabled.

           --ysl &lt;VERT_SUB_DIM&gt;
                Specifies the vertical  dimension of each  sub-lattice (black+white spins),  per
                GPU.  This dimension must be a divisor of the vertical dimension of  the  entire
                lattice per  GPU  (specified  with  the  '-y'  option) and a multiple of 16.

        -o|--o
                Enables the file dump of  the lattice  every time  the magnetization is printed.
                Default: off
</PRE>

For example, to run 128 update steps on a 65536^2 lattice using two GPUs connected via NVLink and printing the magnetization every 16 steps:

<PRE>
$ ./cuIsing -y 32768 -x 65536 -n 128 -p 16 -d 2 -t 1.5  

Using GPUs:
         0 (Tesla V100-DGXS-16GB, 80 SMs, 2048 th/SM max, CC 7.0, ECC on)
         1 (Tesla V100-DGXS-16GB, 80 SMs, 2048 th/SM max, CC 7.0, ECC on)

GPUs direct access matrix:
          0   1
GPU  0:   V   V
GPU  1:   V   V

Run configuration:
        spin/word: 16
        spins: 4294967296
        seed: 463463564571
        iterations: 128
        block (X, Y): 16, 16
        tile  (X, Y): 32, 16
        grid  (X, Y): 32, 2048
        print magn. every 16 steps
        temp: 1.500000 (0.661030*T_crit)
        temp update not set
        not using Hamiltonian buffer

        local lattice size:         32768 x    65536
        total lattice size:         65536 x    65536
        local lattice shape: 2 x    32768 x     2048 (   134217728 ulls)
        total lattice shape: 2 x    65536 x     2048 (   268435456 ulls)
        memory: 2048.00 MB (1024.00 MB per GPU)

Setting up multi-gpu configuration:
        GPU  0 done
        GPU  1 done

Initial magnetization:  0.000000, up_s:   2147484090, dw_s:   2147483206
        magnetization:  0.000043, up_s:   2147575418, dw_s:   2147391878 (iter:       16)
        magnetization:  0.000074, up_s:   2147641872, dw_s:   2147325424 (iter:       32)
        magnetization:  0.000057, up_s:   2147605659, dw_s:   2147361637 (iter:       48)
        magnetization:  0.000101, up_s:   2147701147, dw_s:   2147266149 (iter:       64)
        magnetization:  0.000035, up_s:   2147558546, dw_s:   2147408750 (iter:       80)
        magnetization:  0.000006, up_s:   2147471275, dw_s:   2147496021 (iter:       96)
        magnetization:  0.000060, up_s:   2147612509, dw_s:   2147354787 (iter:      112)
        magnetization:  0.000091, up_s:   2147678887, dw_s:   2147288409 (iter:      128)
Final   magnetization:  0.000091, up_s:   2147678887, dw_s:   2147288409 (iter:      128)

Kernel execution time for 128 update steps: 7.174555E+02 ms, 766.26 flips/ns (BW: 1150.32 GB/s)

</PRE>

</PRE>

Or, to run concurrently 1024 independent sub-lattices of size 2048^2 using two GPUs connected via NVLink and printing the magnetization every 16 steps:

<PRE>
$ ./cuIsing -y 32768 -x 65536 -n 128 -p 16 -d 2 -t 1.5 --xsl 2048 --ysl 2048

Using GPUs:
         0 (Tesla V100-DGXS-16GB, 80 SMs, 2048 th/SM max, CC 7.0, ECC on)
         1 (Tesla V100-DGXS-16GB, 80 SMs, 2048 th/SM max, CC 7.0, ECC on)

GPUs direct access matrix:
          0   1
GPU  0:   V   V
GPU  1:   V   V

Run configuration:
        spin/word: 16
        spins: 4294967296
        seed: 463463564571
        iterations: 128
        block (X, Y): 16, 16
        tile  (X, Y): 32, 16
        grid  (X, Y): 32, 2048
        print magn. every 16 steps
        temp: 1.500000 (0.661030*T_crit)
        temp update not set
        not using Hamiltonian buffer

        using sub-lattices:
                no. of sub-lattices per GPU:      512
                no. of sub-lattices (total):     1024
                sub-lattices size:              2048 x    2048

        local lattice size:         32768 x    65536
        total lattice size:         65536 x    65536
        local lattice shape: 2 x    32768 x     2048 (   134217728 ulls)
        total lattice shape: 2 x    65536 x     2048 (   268435456 ulls)
        memory: 2048.00 MB (1024.00 MB per GPU)

Setting up multi-gpu configuration:
        GPU  0 done
        GPU  1 done

Initial magnetization:  0.000000, up_s:   2147484090, dw_s:   2147483206
        magnetization:  0.000052, up_s:   2147594634, dw_s:   2147372662 (iter:       16)
        magnetization:  0.000069, up_s:   2147631783, dw_s:   2147335513 (iter:       32)
        magnetization:  0.000031, up_s:   2147550893, dw_s:   2147416403 (iter:       48)
        magnetization:  0.000068, up_s:   2147630364, dw_s:   2147336932 (iter:       64)
        magnetization:  0.000008, up_s:   2147500244, dw_s:   2147467052 (iter:       80)
        magnetization:  0.000059, up_s:   2147357073, dw_s:   2147610223 (iter:       96)
        magnetization:  0.000000, up_s:   2147482936, dw_s:   2147484360 (iter:      112)
        magnetization:  0.000010, up_s:   2147461873, dw_s:   2147505423 (iter:      128)
Final   magnetization:  0.000010, up_s:   2147461873, dw_s:   2147505423 (iter:      128)

Kernel execution time for 128 update steps: 7.147521E+02 ms, 769.16 flips/ns (BW: 1154.67 GB/s)
</PRE>
## Visualizing results

Running the code with the '-o' option enables the lattice dump at every timestep in which the
magnetization is printed on screen (depends on either the '-p' and '-e' options). The file name
has the following format:

<PRE>
lattice_&lt;LOCAL_Y&gt;x&lt;LOCAL_X&gt;_T_&lt;TEMP&gt;_IT_&lt;IT_NUMBER&gt;_&lt;GPU_ID&gt;.txt
</PRE>

The included `plotLattice.py` script allows to create an image from those output files. For example,
the following command:

<PRE>
$ ./plotLattice.py lattice_8192x8192_T_1.500000_IT_00001024_0.txt
</PRE>

will generate an image file named `lattice_8192x8192_T_1.500000_IT_00001024_0.txt.png` like:

![image_1](images/lattice_8192x8192_T_1.500000_IT_00001024_0.txt.png)

## Contacts

For comments, questions or anything related, write to Mauro Bisson at maurob@nvidia.com.
