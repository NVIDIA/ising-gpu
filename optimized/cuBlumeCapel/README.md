# A CUDA implementation for the Blume-Capel model supporting Multi-Node NVLink

A high performance Blume Capel model implementation for GPU. The code can run on
multiple GPUs connected to the same node or on multiple nodes connected via
NVLink (MNNVL).

To compile the code to run on single node, adjust the Makefile to point to your CUDA
installation, specify the CUDA architecture you want to compile for and then
run `make`. That should be enough to produce the ``cuBlume`` binary.

For multi-node, in addition to the Makefile adjustment above, also modify it to 
point to your MPI installation and then compile it with `make USE_MNNVL=1`.

When running on a single node, the code uses managed memory. On multiple nodes
with MNNVL, it uses [fabric memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fabric-memory).

When more than one GPU is used, the spin system is partitioned vertically.

## Usage

<PRE>
Usage: cuBlume [options]
options:
        -x|--x &lt;HORIZ_DIM&gt;
                Specifies the horizontal dimension of the entire  lattice  (black+white  spins).
                This dimension must be a multiple of 2048.

        -y|--y &lt;VERT_DIM&gt;
                Specifies the vertical dimension of the per-GPU lattice.  This dimension must be
                a multiple of 16.

        -n|--n &lt;NSTEPS&gt;
                Specifies the number of iteration to run.
                Defualt: 1

        -g|--gpus &lt;NUM_DEVICES&gt;
                Specifies the number of GPUs to use. Will use devices with ids [0, NUM_DEVS-1].
                Defualt: 1.

        -s|--seed &lt;SEED&gt;
                Specifies the seed used to generate random numbers.
                Default: 463463564571

        -a|--alpha &lt;ALPHA&gt;
                Specifies the temperature in T_CRIT units.  If both this  option  and  '-t'  are
                specified then the '-t' option is used.
                Default: 0.100000

        -d|--delta &lt;DELTA&gt;
                Specifies the delta parameter for the Blume-Capel model.
                Default: 1.000000

        -t|--temp &lt;TEMP_0&gt;[[,&lt;IT_1&gt;:&lt;TEMP_1&gt;]...]
                Specifies the temperature(s), in absolute  units.   It  is  possible  to  use  a
                temperature-changing   protocol   by   specifying   a   sequence   of    couples
                &lt;IT_i&gt;:&lt;TEMP_i&gt; after the first temperature &lt;TEMP_0&gt;. The value &lt;IT_i&gt; specifies
                the time step at which the temperature  changes  from  &lt;TEMP_i-1&gt;  to  &lt;TEMP_i&gt;.
                Temperature &lt;TEMP_0&gt; is the starting temperature and thus  does  not  require  a
                time step specification.
                Default: 0.226919

        -p|--print &lt;STAT_FREQ&gt;
                Specifies the frequency, in no.  of  iteration,  with  which  the  magnetization
                statistics is printed.  If this option is used together to the '-e' option, this
                option is ignored.
                Default: only at the beginning and at end of the simulation

        --pexp
                Prints statistics every power-of-2 time steps.  This  option  overrides  the  -p
                option.
                Default: disabled

        -c|--corr
                Dumps  to  a  file  named  corr_{TYPE}_{X}x{Y}_T_{TEMP} the correlation o   each
                point with the vertical and horizontal neighbors at distance r &lt;= 256.   Beyond
                that, distance as chosen according to an exponential rule, with 32  values  per
                power of 2.  The  correlation  is  computed  every  time  the  magnetization  is
                printed on screen (based  on  either  the  '-p'  or  '-e'  options)  and  it  is
                written in the  file one line per measure.
                Default: full correlation (see --corrfull option)

        --corrfull
                Compute the correlation for each spin in the system.

        --corrdiag
                Compute the correlation only for diagonal spins.

        --corrchkb
                Computes the correlation for only one spin (the top-left one)  for each block of
                16x16 spins (checkerboard pattern).

        --corrmixd
                Computes the correlation using a mix of full and checkerboard modes.   The  full
                correlation is used for  all distances  r <= 32. Then,  for each spin in a 16x16
                square, it is computed for each r > 32.

        --writechkp &lt;CHECKPOINT_FILE_PATH&gt;
                Enables write of checkpoint file at the end of the simulation.  The file can  be
                later used to resume the simulation with the '-r' option.  This option and  '-r'
                can be used together to break down a  large  run  into  multiple  smaller  runs.
                When running with multiple processes,  the file name must contain either '%i' or
                '%d' which will be substituted with the process number.
                
        --readchkp &lt;CHECKPOINT_FILE_PATH&gt;
                Enables the restart of a simulation from the state in a checkpoint file.  Please
                note that in order for that to work, the non-checkpoint  command  lines  options
                used in the run where the checkpoint file was created must match with those used
                in the run where the checkpoint file is read.  This option and '-r' can be  used
                together  to  break   down   a   large   run   into   multiple   smaller   runs.
                When running with multiple processes,  the file name must contain either '%i' or
                '%d' which will be substituted with the process number.
        -o|--o
                Enables the file dump of  the lattice  every time  the magnetization is printed.
                Default: off
</PRE>

For example, to run 102400 steps on a 16384^2 lattice using one GPU, using temperature 1.5 and
printing the statistics every 10240 steps:

<PRE>
$ ./cuBlume -y 32768 -x 32768 -n 1024 -p 128 -g 1 -t 1.5

Using GPUs:
         0 (NVIDIA RTX 6000 Ada Generation, 48 GB, 142 SMs, 1536 th/SM max, CC 8.9, ECC off)

Run configuration:
        word size: 16
        bits per spin: 4 (mask: 0xF)
        spins/word: 32
        spins: 1073741824 (~1.07E+09)
        seed: 463463564571
        block size (X, Y): 16, 16
        tile  size (X, Y): 32, 16
        grid size 1D: 32768
        virtual grid size 2D (X, Y): 16, 2048
        spins per tile (X, Y): 1024, 512

        iterations:
                beg: 1
                end: 1024
                tot: 1024

        print stats every 128 steps
        delta: 1
        temperature: 1.5 (0.661030190265538*T_crit)

        no. of  processes: 1
        GPUs  per process: 1
        total no. of GPUs: 1
        GPUs  memory type: managed

        per-GPU lattice size:         32768 x    32768 spins
        per-GPU lattice shape: 2 x    32768 x      512 ull2s (    33554432 total)

        total lattice size:         32768 x    32768 spins
        total lattice shape: 2 x    32768 x      512 ull2s (    33554432 total)

        total memory: 0.50 GB (0.50 GB per GPU)

Setting up GPUs:
        GPU  0 done in 0.020104 secs

Initializing spin lattice... done in 0.058671 secs

[Switching to temperature: 1.5]

Running simulation...

        Step   MC SW          Magn.          N(-1)           N(0)           N(1)     SD value     flips/ns         GB/s          ERT

           0           7.080846E-06      357903413      357927395      357911016     5.716485
         128    *      1.601530E-04      376546141      320821505      376374178     1.000418       511.37       769.55        2.17s
         256    *      5.741259E-04      376809831      320738625      376193368     0.999816       509.93       767.38        2.18s
         384    *      1.082895E-04      376545445      320767209      376429170     0.999965       509.44       766.64        2.18s
         512    *      1.646699E-04      376582881      320752875      376406068     1.000123       507.92       764.36        2.18s
         640    *      1.317356E-04      376417378      320765618      376558828     0.999747       510.41       768.11        2.18s
         768    *      3.697937E-04      376286661      320771439      376683724     1.000044       508.58       765.36        2.18s
         896    *      3.665267E-04      376673596      320788187      376280041     0.999778       509.95       767.42        2.18s
        1024    *      1.519648E-04      376579698      320745599      376416527     1.000010       504.67       759.46        2.18s

Done in 2.184835E+03 ms (stats overhead: 1.15%, spins/ns: 503.25, BW: 757.33 GB/s)
</PRE>

To run 128 steps on a 2^20x2^20 lattice using 8 H100 GPUs:

<PRE>
$ ./cuBlume -y $((2**20 / 8)) -x $((2**20)) -n 128 -p 32 -t 1.5 -g 8

Using GPUs:
         0 (NVIDIA H100 80GB HBM3, 80 GB, 132 SMs, 2048 th/SM max, CC 9.0, ECC on)
         1 (NVIDIA H100 80GB HBM3, 80 GB, 132 SMs, 2048 th/SM max, CC 9.0, ECC on)
         2 (NVIDIA H100 80GB HBM3, 80 GB, 132 SMs, 2048 th/SM max, CC 9.0, ECC on)
         3 (NVIDIA H100 80GB HBM3, 80 GB, 132 SMs, 2048 th/SM max, CC 9.0, ECC on)
         4 (NVIDIA H100 80GB HBM3, 80 GB, 132 SMs, 2048 th/SM max, CC 9.0, ECC on)
         5 (NVIDIA H100 80GB HBM3, 80 GB, 132 SMs, 2048 th/SM max, CC 9.0, ECC on)
         6 (NVIDIA H100 80GB HBM3, 80 GB, 132 SMs, 2048 th/SM max, CC 9.0, ECC on)
         7 (NVIDIA H100 80GB HBM3, 80 GB, 132 SMs, 2048 th/SM max, CC 9.0, ECC on)

Run configuration:
        word size: 16
        bits per spin: 4 (mask: 0xF)
        spins/word: 32
        spins: 1099511627776 (~1.10E+12)
        seed: 463463564571
        block size (X, Y): 16, 16
        tile  size (X, Y): 32, 16
        grid size 1D: 4194304
        virtual grid size 2D (X, Y): 512, 8192
        spins per tile (X, Y): 1024, 512

        iterations:
                beg: 1
                end: 128
                tot: 128

        print stats every 32 steps
        delta: 1
        temperature: 1.5 (0.661030190265538*T_crit)

        no. of  processes: 1
        GPUs  per process: 8
        total no. of GPUs: 8
        GPUs  memory type: managed

        per-GPU lattice size:        131072 x  1048576 spins
        per-GPU lattice shape: 2 x   131072 x    16384 ull2s (  4294967296 total)

        total lattice size:       1048576 x  1048576 spins
        total lattice shape: 2 x  1048576 x    16384 ull2s ( 34359738368 total)

        total memory: 512.00 GB (64.00 GB per GPU)

Setting up GPUs:
        GPU  0 done in 1.094278 secs
        GPU  1 done in 1.260294 secs
        GPU  2 done in 1.268412 secs
        GPU  3 done in 1.259265 secs
        GPU  4 done in 1.269356 secs
        GPU  5 done in 1.279294 secs
        GPU  6 done in 1.286008 secs
        GPU  7 done in 1.288558 secs

Initializing spin lattice... done in 6.611633 secs

[Switching to temperature: 1.5]

Running simulation...

        Step   MC SW          Magn.          N(-1)           N(0)           N(1)     SD value     flips/ns         GB/s          ERT

           0           5.335123E-07   366503778958   366503483257   366504365561     5.717101
          32    *      3.692141E-06   384166621778   331174324668   384170681330     1.001240      6375.69      9567.43       22.45s
          64    *      7.972785E-07   385202476551   329105798057   385203353168     1.000216      6375.91      9567.76       22.45s
          96    *      2.314373E-07   385421280026   328668813256   385421534494     1.000065      6376.48      9568.61       22.45s
         128    *      5.507602E-06   385491919415   328533844618   385485863743     1.000011      6376.44      9568.55       22.45s

Done in 2.244686E+04 ms (stats overhead: 1.70%, spins/ns: 6269.81, BW: 9408.54 GB/s)
</PRE>

## Contacts

For comments, questions or anything related, write to Mauro Bisson at maurob@nvidia.com.
