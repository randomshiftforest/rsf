# Random Shift Forest

## Install

Make sure to have the latest Python toolchain installed. Running the `gen` script downloads all necessary Python libraries and fetches the datasets and libraries used and puts them in a directory named `in`.

```
$ chmod +x gen.sh
$ ./gen.sh
```

## Run

Make sure to have the latest Rust toolchain installed. Running `cargo test` runs all the experiments and puts the results in a directory named `out`.

```
$ RUSTFLAGS="-C target-cpu=native" cargo test
```
