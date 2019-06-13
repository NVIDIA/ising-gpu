### Basic Implementation using CUDA C

### Basic Usage
Compile binary with `make`.

Example run command:

`./ising_basic -x <rows> -y <columns> -n <number of iterations> `

Run `./ising_basic --help` for more options.

### Visualizing Results
`-o` flag enables output of final lattice configuration to text file `final.txt`. Use provided `plot_ising.py` to visualize output.

For example:
```
$ ./ising_basic -x 2048 -y 2048 -n 100 -a 0.5 -o
...
Writing lattice to final.txt...

$ python plot_ising.py
```

This will produce the following output:

![sample_plot.png](sample_plot.png)
