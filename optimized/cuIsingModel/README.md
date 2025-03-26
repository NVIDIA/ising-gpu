# A CUDA implementation for the Ising model supporting Multi-Node NVLink

A high performance Ising model implementation for GPU. The code can run on
multiple GPUs connected to the same node or on multiple nodes connected via
NVLink (MNNVL).

To compile the code to run on single node, adjust the Makefile to point to your CUDA
installation, specify the CUDA architecture you want to compile for and then
run `make`. That should be enough to produce the ``cuIsing`` binary.

For multi-node, in addition to the Makefile adjustment above, also modify it to 
point to your MPI installation and then compile it with `make USE_MNNVL=1`.

When running on a single node, the code uses managed memory. On multiple nodes
with MNNVL, it uses [fabric memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fabric-memory).

When more than one GPU is used, the spin system is partitioned vertically.

## Usage

<PRE>
Usage: cuIsing [options]
options:
        -x|--x &lt;HORIZ_DIM&gt;
                Specifies the horizontal dimension of the entire  lattice  (black+white  spins).
                This dimension must be a multiple of 4096.

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
                statistics is printed. If this option is used with --pexp, this option is ignored.
                Default: only at the beginning and at end of the simulation

        --pexp
                Prints statistics every power-of-2 time steps.  This  option  overrides  the  -p
                option.
                Default: disabled

        -c|--corr &lt;CORR_FILE_PATH&gt;
                Enables correlation and writes to file CORR_FILE_PATH  the  correlation of  each
                point with the vertical and  orizontal  neighbors at distance r <= 256.   Beyond
                that, distance as chosen according to an  exponential rule, with 32  values  per
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
                correlation is used for  all distances  r &lt;= 32. Then,  for each spin in a 16x16
                square, it is computed for each r &gt; 32.

        --writechkp &lt;CHECKPOINT_FILE_PATH&gt;
                Enables write of checkpoint file at the end of the simulation.  The file can  be
                later used to resume the simulation with the '-r' option.  This option and  '-r'
                can be used together to break down a  large  run  into  multiple  smaller  runs.

        --readchkp &lt;CHECKPOINT_FILE_PATH&gt;
                Enables the restart of a simulation from the state in a checkpoint file.  Please
                note that in order for that to work, the non-checkpoint  command  lines  options
                used in the run where the checkpoint file was created must match with those used
                in the run where the checkpoint file is read.  This option and '-r' can be  used
                together  to  break   down   a   large   run   into   multiple   smaller   runs.

        -o|--o
                Enables the file dump of  the lattice  every time  the magnetization is printed.
                Default: off
</PRE>

For example, to run 102400 steps on a 16384^2 lattice using one GPU, using temperature 1.5 and
printing the statistics every 10240 steps:

<PRE>
$ ./cuIsing -y 16384 -x 16384 -n 102400 -p 10240 -t 1.5

Using GPUs:
         0 (NVIDIA RTX 6000 Ada Generation, 48 GB, 142 SMs, 1536 th/SM max, CC 8.9, ECC off)

Run configuration:
        word size: 16
        bits per spin: 1 (mask: 0x1)
        spins/word: 128
        spins: 268435456 (~2.68E+08)
        seed: 463463564571
        block size (X, Y): 16, 16
        tile  size (X, Y): 16, 16
        grid  size (X, Y): 4, 1024
        spins per tile (X, Y): 2048, 2048

        iterations:
                beg: 1
                end: 102400
                tot: 102400

        print stats every 10240 steps
        temp: 1.5 (0.661030190265538*T_crit)

        local lattice size:         16384 x    16384 spins
        local lattice shape: 2 x    16384 x       64 ull2s (     2097152 total)

        total lattice size:         16384 x    16384 spins
        total lattice shape: 2 x    16384 x       64 ull2s (     2097152 total)

        total memory: 0.03 GB (0.03 GB per GPU)

        random-bit table:
                size of element: 32-bit
                no. of elements: 16
                bits per lookup: 4

Setting up GPUs:
        GPU  0 done in 0.001597 secs

Initializing spin lattice... done in 0.011790 secs

Running simulation...

        Step          Magn.          N(-1)           N(1)     SD value     flips/ns         GB/s          ERT

           0   8.381903E-05      134206478      134228978    16.936372
       10240   5.389421E-02      141451286      126984170     1.000717      1399.28       527.46       19.65s
       20480   6.544993E-02      143002269      125433187     0.999917      1392.13       524.77       19.70s
       30720   7.027917E-02      143650439      124785017     1.000416      1387.92       523.18       19.74s
       40960   7.348213E-02      144080332      124355124     0.998606      1385.64       522.32       19.76s
       51200   7.878675E-02      144792307      123643149     1.000069      1385.46       522.25       19.78s
       61440   8.068839E-02      145047541      123387915     0.997942      1384.75       521.99       19.79s
       71680   7.845285E-02      144747491      123687965     1.000395      1383.86       521.65       19.80s
       81920   7.937136E-02      144870771      123564685     1.000686      1378.90       519.78       19.82s
       92160   7.773913E-02      144651698      123783758     0.998647      1375.31       518.43       19.84s
      102400   8.023911E-02      144987239      123448217     1.000491      1371.54       517.00       19.86s

Final energy: -1.949967

Done in 1.986138E+04 ms (stats overhead: 0.05%, spins/ns: 1383.98, BW: 521.70 GB/s)
</PRE>

Run 307200 steps on a 16384^2 lattice using one GPU, in three distinct runs
each of 102400 steps using checkpointing:

<PRE>
$ ./cuIsing -y 16384 -x 16384 -n 102400 -p 10240 -t 1.5 -w chkpfile

Using GPUs:
         0 (NVIDIA RTX 6000 Ada Generation, 48 GB, 142 SMs, 1536 th/SM max, CC 8.9, ECC off)

Run configuration:
        word size: 16
        bits per spin: 1 (mask: 0x1)
        spins/word: 128
        spins: 268435456 (~2.68E+08)
        seed: 463463564571
        block size (X, Y): 16, 16
        tile  size (X, Y): 16, 16
        grid  size (X, Y): 4, 1024
        spins per tile (X, Y): 2048, 2048

        iterations:
                beg: 1
                end: 102400
                tot: 102400

        print stats every 10240 steps
        temp: 1.5 (0.661030190265538*T_crit)

        local lattice size:         16384 x    16384 spins
        local lattice shape: 2 x    16384 x       64 ull2s (     2097152 total)

        total lattice size:         16384 x    16384 spins
        total lattice shape: 2 x    16384 x       64 ull2s (     2097152 total)

        total memory: 0.03 GB (0.03 GB per GPU)

        random-bit table:
                size of element: 32-bit
                no. of elements: 16
                bits per lookup: 4

Setting up GPUs:
        GPU  0 done in 0.001700 secs

Initializing spin lattice... done in 0.012194 secs

Running simulation...

        Step          Magn.          N(-1)           N(1)     SD value     flips/ns         GB/s          ERT

           0   8.381903E-05      134206478      134228978    16.936372
       10240   5.389421E-02      141451286      126984170     1.000717      1351.59       509.49       20.34s
       20480   6.544993E-02      143002269      125433187     0.999917      1352.59       509.86       20.34s
       30720   7.027917E-02      143650439      124785017     1.000416      1347.67       508.01       20.36s
       40960   7.348213E-02      144080332      124355124     0.998606      1349.08       508.54       20.36s
       51200   7.878675E-02      144792307      123643149     1.000069      1351.91       509.61       20.36s
       61440   8.068839E-02      145047541      123387915     0.997942      1355.41       510.93       20.35s
       71680   7.845285E-02      144747491      123687965     1.000395      1353.09       510.05       20.34s
       81920   7.937136E-02      144870771      123564685     1.000686      1352.15       509.70       20.34s
       92160   7.773913E-02      144651698      123783758     0.998647      1347.46       507.93       20.35s
      102400   8.023911E-02      144987239      123448217     1.000491      1345.72       507.27       20.36s

Final energy: -1.949967

Done in 2.035810E+04 ms (stats overhead: 0.05%, spins/ns: 1350.21, BW: 508.97 GB/s)

Writing checkpoint to file chkpfile... done in 0.083085 secs
</PRE>
<PRE>
$ ./cuIsing -y 16384 -x 16384 -n 102400 -p 10240 -t 1.5 -w chkpfile -r chkpfile

Using GPUs:
         0 (NVIDIA RTX 6000 Ada Generation, 48 GB, 142 SMs, 1536 th/SM max, CC 8.9, ECC off)

Reading checkpoint from file chkpfile... done in 0.010425 secs

Run configuration:
        word size: 16
        bits per spin: 1 (mask: 0x1)
        spins/word: 128
        spins: 268435456 (~2.68E+08)
        seed: 463463564571
        block size (X, Y): 16, 16
        tile  size (X, Y): 16, 16
        grid  size (X, Y): 4, 1024
        spins per tile (X, Y): 2048, 2048

        iterations:
                beg: 102401
                end: 204800
                tot: 102400

        print stats every 10240 steps
        temp: 1.5 (0.661030190265538*T_crit)

        local lattice size:         16384 x    16384 spins
        local lattice shape: 2 x    16384 x       64 ull2s (     2097152 total)

        total lattice size:         16384 x    16384 spins
        total lattice shape: 2 x    16384 x       64 ull2s (     2097152 total)

        total memory: 0.03 GB (0.03 GB per GPU)

        random-bit table:
                size of element: 32-bit
                no. of elements: 16
                bits per lookup: 4

Setting up GPUs:
        GPU  0 done in 0.003768 secs

Running simulation...

        Step          Magn.          N(-1)           N(1)     SD value     flips/ns         GB/s          ERT

      102400   8.023911E-02      144987239      123448217     1.000491
      112640   8.427709E-02      145529207      122906249     0.999487      1369.15       516.10       20.08s
      122880   8.961894E-02      146246178      122189278     1.001249      1362.94       513.76       20.13s
      133120   8.933730E-02      146208378      122227078     0.999772      1356.70       511.41       20.17s
      143360   8.894347E-02      146155518      122279938     1.000053      1356.84       511.46       20.20s
      153600   8.961185E-02      146245227      122190229     1.000030      1352.37       509.78       20.22s
      163840   8.997627E-02      146294138      122141318     0.999970      1352.44       509.81       20.24s
      174080   8.834548E-02      146075257      122360199     1.000698      1352.11       509.68       20.26s
      184320   8.784929E-02      146008660      122426796     1.000313      1349.95       508.87       20.27s
      194560   9.042334E-02      146354143      122081313     1.000820      1348.24       508.22       20.28s
      204800   9.108921E-02      146443515      121991941     1.000014      1346.63       507.62       20.30s

Final energy: -1.950272

Done in 2.029726E+04 ms (stats overhead: 0.05%, spins/ns: 1354.26, BW: 510.49 GB/s)

Writing checkpoint to file chkpfile... done in 0.082859 secs
</PRE>
<PRE>
$ ./cuIsing -y 16384 -x 16384 -n 102400 -p 10240 -t 1.5 -r chkpfile

Using GPUs:
         0 (NVIDIA RTX 6000 Ada Generation, 48 GB, 142 SMs, 1536 th/SM max, CC 8.9, ECC off)

Reading checkpoint from file chkpfile... done in 0.010423 secs

Run configuration:
        word size: 16
        bits per spin: 1 (mask: 0x1)
        spins/word: 128
        spins: 268435456 (~2.68E+08)
        seed: 463463564571
        block size (X, Y): 16, 16
        tile  size (X, Y): 16, 16
        grid  size (X, Y): 4, 1024
        spins per tile (X, Y): 2048, 2048

        iterations:
                beg: 204801
                end: 307200
                tot: 102400

        print stats every 10240 steps
        temp: 1.5 (0.661030190265538*T_crit)

        local lattice size:         16384 x    16384 spins
        local lattice shape: 2 x    16384 x       64 ull2s (     2097152 total)

        total lattice size:         16384 x    16384 spins
        total lattice shape: 2 x    16384 x       64 ull2s (     2097152 total)

        total memory: 0.03 GB (0.03 GB per GPU)

        random-bit table:
                size of element: 32-bit
                no. of elements: 16
                bits per lookup: 4

Setting up GPUs:
        GPU  0 done in 0.003810 secs

Running simulation...

        Step          Magn.          N(-1)           N(1)     SD value     flips/ns         GB/s          ERT

      204800   9.108921E-02      146443515      121991941     1.000014
      215040   8.998523E-02      146295341      122140115     0.999673      1354.50       510.58       20.30s
      225280   8.892218E-02      146152661      122282795     0.999000      1344.27       506.73       20.38s
      235520   9.020317E-02      146324593      122110863     1.000224      1343.46       506.42       20.41s
      245760   9.139725E-02      146484859      121950597     0.999815      1342.54       506.07       20.43s
      256000   9.055272E-02      146371509      122063947     0.999528      1341.84       505.81       20.44s
      266240   8.986650E-02      146279405      122156051     1.000316      1339.66       504.99       20.45s
      276480   9.154957E-02      146505303      121930153     1.001214      1335.71       503.50       20.47s
      286720   9.230582E-02      146606805      121828651     0.999690      1336.05       503.63       20.49s
      296960   9.236395E-02      146614608      121820848     0.998615      1333.45       502.65       20.50s
      307200   9.218215E-02      146590207      121845249     1.000438      1332.56       502.31       20.51s

Final energy: -1.950339

Done in 2.051432E+04 ms (stats overhead: 0.05%, spins/ns: 1339.93, BW: 505.09 GB/s)
</PRE>


To run 128 steps on a 2^20x2^20 lattice using 8 H100 GPUs:

<PRE>
$ ./cuIsing -y $((2**20 / 8)) -x $((2**20)) -n 128 -p 128 -t 1.5 -g 8

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
	bits per spin: 1 (mask: 0x1)
	spins/word: 128
	spins: 1099511627776 (~1.10E+12)
	seed: 463463564571
	block size (X, Y): 16, 16
	tile  size (X, Y): 16, 16
	grid  size (X, Y): 256, 8192
	spins per tile (X, Y): 2048, 2048

	iterations:
		beg: 1
		end: 128
		tot: 128

	print stats every 128 steps
	temp: 1.5 (0.661030190265538*T_crit)

	local lattice size:        131072 x  1048576 spins
	local lattice shape: 2 x   131072 x     4096 ull2s (  1073741824 total)

	total lattice size:       1048576 x  1048576 spins
	total lattice shape: 2 x  1048576 x     4096 ull2s (  8589934592 total)

	total memory: 128.00 GB (16.00 GB per GPU)

	random-bit table:
		size of element: 32-bit
		no. of elements: 16
		bits per lookup: 4

Setting up GPUs:
	GPU  0 done in 0.001748 secs
	GPU  1 done in 0.166805 secs
	GPU  2 done in 0.166164 secs
	GPU  3 done in 0.166996 secs
	GPU  4 done in 0.186960 secs
	GPU  5 done in 0.187743 secs
	GPU  6 done in 0.182130 secs
	GPU  7 done in 0.192766 secs

Initializing spin lattice... done in 3.404245 secs

Running simulation...

        Step          Magn.        N(-1)         N(1)     SD value     flips/ns         GB/s          ERT

           0   7.547405E-07 549755398965 549756228811    16.936123
         128   3.269196E-05 549737841294 549773786482     1.000580     10306.30      3867.38       13.78s

Final energy: -1.908699

Done in 1.377803E+04 ms (stats overhead: 0.90%, spins/ns: 10214.63, BW: 3832.98 GB/s)
</PRE>

## Contacts

For comments, questions or anything related, write to Mauro Bisson at maurob@nvidia.com.
