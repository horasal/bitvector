## bitvector, a simple bitvector implementation in Rust-lang

This crate is a modification of [librustc_data_structures/bitvec.rs](https://github.com/rust-lang/rust/blob/master/src/librustc_data_structures/bitvec.rs) for set operator support.

### Documentation

Visit the [online documentation](http://zhaihj.github.io/doc/bitvector/index.html) or run

```cargo doc```

to generate a local copy.

### How to use

Add the following line to `[dependencies]` in your `Cargo.toml` file:

```toml
bitvector = 0.1
```

Then you can use the `BitVector`:

```rust
extern crate bitvector;
use bitvector::*;

fn main() {
    let mut test_vec = BitVector::new(50);
    for i in vec![0,1,3,5,7,11,13,17,19,23] { test_vec.insert(i); }

    let mut test_vec2 = BitVector::new(50);
    for i in vec![3,5,9,13,19,40,45] { test_vec2.insert(i); }

    // union of test_vec and test_vec2
    // other possible operators: intersection, difference
    let union = test_vec.union(&test_vec2);


    assert!(union.contains(3));
    assert!(union.contains(5));
    assert!(union.contains(13));
    assert!(union.contains(19));
    assert!(union.contains(9));
    assert!(union.contains(40));

    // all the operators also have `*_inplace` variants
    // which directly modify `self` to avoid extra memory operations.
    test_vec.union_inplace(&test_vec2);

    assert!(test_vec.contains(3));
    assert!(test_vec.contains(5));
    assert!(test_vec.contains(13));
    assert!(test_vec.contains(19));
    assert!(test_vec.contains(0));
    assert!(test_vec.contains(9));
    assert!(test_vec.contains(40));
}
```


### Performance Comparison against std/collections/Set

```
test bench::bench_bitset_operator         ... bench:         222 ns/iter (+/- 22)
test bench::bench_bitset_operator_inplace ... bench:         122 ns/iter (+/- 5)
test bench::bench_btreeset_operator       ... bench:       1,675 ns/iter (+/- 115)
test bench::bench_hashset_operator        ... bench:       1,748 ns/iter (+/- 37)
```

### LICENSE

MIT
