extern crate bitvector;
use bitvector::*;

fn main() {
    let mut test_vec = BitVector::new(50);
    for i in vec![0, 1, 3, 5, 7, 11, 13, 17, 19, 23] {
        test_vec.insert(i);
    }

    let mut test_vec2 = BitVector::new(50);
    for i in vec![3, 5, 9, 13, 19, 40, 45] {
        test_vec2.insert(i);
    }

    let union = test_vec.intersection(&test_vec2);

    println!("{} | {} = {}", test_vec, test_vec2, union);

    assert!(union.contains(3));
    assert!(union.contains(5));
    assert!(union.contains(13));
    assert!(union.contains(19));
    assert!(!union.contains(0));
    assert!(!union.contains(9));
    assert!(!union.contains(40));

    test_vec.union_inplace(&test_vec2);

    assert!(test_vec.contains(3));
    assert!(test_vec.contains(5));
    assert!(test_vec.contains(13));
    assert!(test_vec.contains(19));
    assert!(test_vec.contains(0));
    assert!(test_vec.contains(9));
    assert!(test_vec.contains(40));
}
