extern crate bitvector;
#[cfg(test)]
extern crate rand;

use bitvector::*;
use rand::distributions::Uniform;
use rand::Rng;
use std::collections::HashSet;

#[test]
fn fuzz_test() {
    let mut rng = rand::thread_rng();
    for _ in 0..5 {
        let v1: Vec<usize> = rng
            .sample_iter(Uniform::new(1, 10_000_000))
            .take(rng.gen_range(10, 10_000))
            .collect();
        let v2: Vec<usize> = rng
            .sample_iter(Uniform::new(1, 10_000_000))
            .take(rng.gen_range(10, 10_000))
            .collect();

        // reference hashset
        let r1 = v1.iter().cloned().collect::<HashSet<usize>>();
        let r2 = v2.iter().cloned().collect::<HashSet<usize>>();

        // bitvector
        let b1 = v1.iter().cloned().collect::<BitVector>();
        let b2 = v2.iter().cloned().collect::<BitVector>();

        // Step1. Check if collect::<BitVector>() works
        for i in 0..=b1.capacity() {
            assert_eq!(b1.contains(i), r1.contains(&i));
        }

        for i in 0..=b2.capacity() {
            assert_eq!(b2.contains(i), r2.contains(&i));
        }

        // STep2. Check if iter() works
        let v1_reconstruct = b1.iter().collect::<HashSet<usize>>();
        for i in 0..=b1.capacity() {
            assert_eq!(b1.contains(i), v1_reconstruct.contains(&i));
            assert_eq!(r1.contains(&i), v1_reconstruct.contains(&i));
        }
        let v2_reconstruct = b2.iter().collect::<HashSet<usize>>();
        for i in 0..=b2.capacity() {
            assert_eq!(b2.contains(i), v2_reconstruct.contains(&i));
            assert_eq!(r2.contains(&i), v2_reconstruct.contains(&i));
        }

        // Step 3.1 test union
        let mut u_b = b1.union(&b2).iter().collect::<Vec<usize>>();
        let mut u_r = r1.union(&r2).cloned().collect::<Vec<usize>>();
        u_b.sort();
        u_r.sort();
        assert_eq!(u_b, u_r);

        // Step 3.2 test intersection
        let mut u_b = b1.intersection(&b2).iter().collect::<Vec<usize>>();
        let mut u_r = r1.intersection(&r2).cloned().collect::<Vec<usize>>();
        u_b.sort();
        u_r.sort();
        assert_eq!(u_b, u_r);

        // Step 3.3 test difference
        let mut u_b = b1.difference(&b2).iter().collect::<Vec<usize>>();
        let mut u_r = r1.difference(&r2).cloned().collect::<Vec<usize>>();
        u_b.sort();
        u_r.sort();
        assert_eq!(u_b, u_r);
    }
}
