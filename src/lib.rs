#![cfg_attr(feature = "unstable", feature(test))]

use std::ops::*;
use std::fmt;
use std::iter::FromIterator;

#[derive(Clone, Debug)]
pub struct BitVector {
    bits: usize,
    vector: Vec<u64>,
}

impl fmt::Display for BitVector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "["));
        try!(write!(f, "{}", self.iter().fold(String::new(), 
                             |x0, x| x0 + &format!("{}, ", x))));
        write!(f, "]")
    }
}

impl PartialEq for BitVector {
    fn eq(&self, other: &BitVector) -> bool { self.vector == other.vector }
}

impl BitVector {
    pub fn new(bits: usize) -> Self {
        BitVector {
            bits: bits,
            vector: vec![0; u64s(bits)],
        }
    }

    pub fn clear(&mut self) {
        for p in &mut self.vector { *p = 0; }
    }

    pub fn contains(&self, bit: usize) -> bool {
        let (word, mask) = word_mask(bit);
        (self.vector[word] & mask) != 0
    }

    pub fn as_slice(&self) -> &[u64] { self.vector.as_slice() }

    pub fn eq_left(&self, other: &BitVector, bit: usize) -> bool {
        let (word, offset) = word_offset(bit - 1);
        /*
         * We can also use slice comparison, which only take 1 line.
         * However, it has been reported that the `Eq` implementation of slice
         * is extremly slow.
         *
         * self.vector.as_slice()[0 .. word] == other.vector.as_slice[0 .. word]
         */
        for i in 0 .. word {
            if self.vector[i] != other.vector[i] {
                return false
            }
        }

        (self.vector[word] << (63 - offset)) == (other.vector[word] << (63 - offset))
    }

    pub fn insert(&mut self, bit: usize) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &mut self.vector[word];
        let value = *data;
        let new_value = value | mask;
        *data = new_value;
        new_value != value
    }

    pub fn insert_all(&mut self, all: &BitVector) -> bool {
        assert!(self.vector.len() == all.vector.len());
        let mut changed = false;
        for (i, j) in self.vector.iter_mut().zip(&all.vector) {
            let value = *i;
            *i = value | *j;
            if value != *i {
                changed = true;
            }
        }
        changed
    }

    pub fn len(&self) -> usize { self.bits }

    pub fn union(&self, other: &BitVector) -> BitVector {
        assert_eq!(self.len(), other.len());
        BitVector {
            bits: self.len(),
            vector: self.vector.iter().enumerate()
            .map(|(i,x)| x | other.vector[i]).collect()
        }
    }

    pub fn intersection(&self, other: &BitVector) -> BitVector {
        assert_eq!(self.len(), other.len());
        BitVector {
            bits: self.len(),
            vector: self.vector.iter().enumerate()
            .map(|(i,x)| x & other.vector[i]).collect()
        }
    }

    pub fn difference(&self, other: &BitVector) -> BitVector {
        assert_eq!(self.len(), other.len());
        BitVector {
            bits: self.len(),
            vector: self.vector.iter().enumerate()
            .map(|(i,x)| (x ^ other.vector[i]) & x).collect()
        }
    }

    pub fn difference_d(&self, other: &BitVector) -> BitVector {
        assert_eq!(self.len(), other.len());
        BitVector {
            bits: self.len(),
            vector: self.vector.iter().enumerate()
            .map(|(i,x)| x ^ other.vector[i]).collect()
        }
    }

    pub fn union_inplace(&mut self, other: &BitVector) -> &mut BitVector {
        assert_eq!(self.len(), other.len());
        for (i,v) in self.vector.iter_mut().enumerate() {
            *v |= other.vector[i]
        }
        self
    }

    pub fn intersection_inplace(&mut self, other: &BitVector) -> &mut BitVector {
        assert_eq!(self.len(), other.len());
        for (i,v) in self.vector.iter_mut().enumerate() {
            *v &= other.vector[i]
        }
        self
    }

    pub fn difference_inplace(&mut self, other: &BitVector) -> &mut BitVector {
        assert_eq!(self.len(), other.len());
        for (i,v) in self.vector.iter_mut().enumerate() {
            *v = (*v ^ other.vector[i]) & *v
        }
        self
    }

    pub fn difference_d_inplace(&mut self, other: &BitVector) -> &mut BitVector {
        assert_eq!(self.len(), other.len());
        for (i,v) in self.vector.iter_mut().enumerate() {
            *v ^= other.vector[i]
        }
        self
    }

    fn grow(&mut self, num_bits: usize) {
        let num_words = u64s(num_bits);
        if self.vector.len() < num_words {
            self.vector.resize(num_words, 0)
        }
    }

    pub fn iter<'a>(&'a self) -> BitVectorIter<'a> {
        BitVectorIter {
            iter: self.vector.iter(),
            current: 0,
            idx: 0,
        }
    }
}

pub struct BitVectorIter<'a> {
    iter: ::std::slice::Iter<'a, u64>,
    current: u64,
    idx: usize,
}

impl<'a> Iterator for BitVectorIter<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        while self.current == 0 {
            self.current = if let Some(&i) = self.iter.next() {
                if i == 0 {
                    self.idx += 64;
                    continue;
                } else {
                    self.idx = u64s(self.idx) * 64;
                    i
                }
            } else {
                return None;
            }
        }
        let offset = self.current.trailing_zeros() as usize;
        self.current >>= offset;
        self.current >>= 1; // shift otherwise overflows for 0b1000_0000_…_0000
        self.idx += offset + 1;
        return Some(self.idx - 1);
    }
}

impl FromIterator<bool> for BitVector {
    fn from_iter<I>(iter: I) -> BitVector where I: IntoIterator<Item=bool> {
        let iter = iter.into_iter();
        let (len, _) = iter.size_hint();
        // Make the minimum length for the bitvector 64 bits since that's
        // the smallest non-zero size anyway.
        let len = if len < 64 { 64 } else { len };
        let mut bv = BitVector::new(len);
        for (idx, val) in iter.enumerate() {
            if idx > len {
                bv.grow(idx);
            }
            if val {
                bv.insert(idx);
            }
        }

        bv
    }
}

impl<'a> BitAnd for &'a BitVector {
    type Output = BitVector;
    fn bitand(self, rhs: Self) -> BitVector {
        self.intersection(rhs)
    }
}

impl<'a> BitAndAssign for &'a mut BitVector {
    fn bitand_assign(&mut self, rhs: Self) {
        self.intersection_inplace(rhs);
    }
}

impl<'a> BitOr for &'a BitVector {
    type Output = BitVector;
    fn bitor(self, rhs: Self) -> BitVector {
        self.union(rhs)
    }
}

impl<'a> BitOrAssign for &'a mut BitVector {
    fn bitor_assign(&mut self, rhs: Self) {
        self.union_inplace(rhs);
    }
}

impl<'a> BitXor for &'a BitVector {
    type Output = BitVector;
    fn bitxor(self, rhs: Self) -> BitVector {
        self.difference(rhs)
    }
}

impl<'a> BitXorAssign for &'a mut BitVector {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.difference_inplace(rhs);
    }
}

impl BitAnd for BitVector {
    type Output = BitVector;
    fn bitand(self, rhs: Self) -> BitVector {
        self.intersection(&rhs)
    }
}

impl BitAndAssign for BitVector {
    fn bitand_assign(&mut self, rhs: Self) {
        self.intersection_inplace(&rhs);
    }
}

impl BitOr for BitVector {
    type Output = BitVector;
    fn bitor(self, rhs: Self) -> BitVector {
        self.union(&rhs)
    }
}

impl BitOrAssign for BitVector {
    fn bitor_assign(&mut self, rhs: Self) {
        self.union_inplace(&rhs);
    }
}

impl BitXor for BitVector {
    type Output = BitVector;
    fn bitxor(self, rhs: Self) -> BitVector {
        self.difference(&rhs)
    }
}

impl BitXorAssign for BitVector {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.difference_inplace(&rhs);
    }
}


fn u64s(elements: usize) -> usize {
    (elements + 63) / 64
}

fn word_offset(index: usize) -> (usize, usize) {
    (index / 64, index % 64)
}

fn word_mask(index: usize) -> (usize, u64) {
    let word = index / 64;
    let mask = 1 << (index % 64);
    (word, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn union_two_vecs() {
        let mut vec1 = BitVector::new(65);
        let mut vec2 = BitVector::new(65);
        assert!(vec1.insert(3));
        assert!(!vec1.insert(3));
        assert!(vec2.insert(5));
        assert!(vec2.insert(64));
        assert!(vec1.insert_all(&vec2));
        assert!(!vec1.insert_all(&vec2));
        assert!(vec1.contains(3));
        assert!(!vec1.contains(4));
        assert!(vec1.contains(5));
        assert!(!vec1.contains(63));
        assert!(vec1.contains(64));
    }

    #[test]
    fn bitvector_union() {
        let mut vec1 = BitVector::new(65);
        let mut vec2 = BitVector::new(65);
        assert!(vec1.insert(3));
        assert!(!vec1.insert(3));
        assert!(vec2.insert(5));
        assert!(vec2.insert(64));

        let vec1 = vec1.union(&vec2);

        assert!(vec1.contains(3));
        assert!(!vec1.contains(4));
        assert!(vec1.contains(5));
        assert!(!vec1.contains(63));
        assert!(vec1.contains(64));
    }

    #[test]
    fn bitvector_intersection() {
        let mut vec1 = BitVector::new(65);
        let mut vec2 = BitVector::new(65);
        assert!(vec1.insert(3));
        assert!(!vec1.insert(3));
        assert!(vec1.insert(5));
        assert!(vec2.insert(5));
        assert!(!vec2.insert(5));
        assert!(vec2.insert(64));

        let vec1 = vec1.intersection(&vec2);

        assert!(!vec1.contains(3));
        assert!(!vec1.contains(4));
        assert!(vec1.contains(5));
        assert!(!vec1.contains(63));
        assert!(!vec1.contains(64));
    }

    #[test]
    fn bitvector_difference() {
        let mut vec1 = BitVector::new(65);
        let mut vec2 = BitVector::new(65);
        assert!(vec1.insert(3));
        assert!(!vec1.insert(3));
        assert!(vec1.insert(5));
        assert!(vec2.insert(5));
        assert!(!vec2.insert(5));
        assert!(vec2.insert(64));

        let vec1 = vec1.difference(&vec2);

        assert!(vec1.contains(3));
        assert!(!vec1.contains(4));
        assert!(!vec1.contains(5));
        assert!(!vec1.contains(63));
        assert!(!vec1.contains(64));
    }

    #[test]
    fn bitvector_union_inplace() {
        let mut vec1 = BitVector::new(65);
        let mut vec2 = BitVector::new(65);
        assert!(vec1.insert(3));
        assert!(!vec1.insert(3));
        assert!(vec2.insert(5));
        assert!(vec2.insert(64));

        let vec1 = vec1.union_inplace(&vec2);

        assert!(vec1.contains(3));
        assert!(!vec1.contains(4));
        assert!(vec1.contains(5));
        assert!(!vec1.contains(63));
        assert!(vec1.contains(64));
    }

    #[test]
    fn bitvector_intersection_inplace() {
        let mut vec1 = BitVector::new(65);
        let mut vec2 = BitVector::new(65);
        assert!(vec1.insert(3));
        assert!(!vec1.insert(3));
        assert!(vec1.insert(5));
        assert!(vec2.insert(5));
        assert!(!vec2.insert(5));
        assert!(vec2.insert(64));

        let vec1 = vec1.intersection_inplace(&vec2);

        assert!(!vec1.contains(3));
        assert!(!vec1.contains(4));
        assert!(vec1.contains(5));
        assert!(!vec1.contains(63));
        assert!(!vec1.contains(64));
    }

    #[test]
    fn bitvector_difference_inplace() {
        let mut vec1 = BitVector::new(65);
        let mut vec2 = BitVector::new(65);
        assert!(vec1.insert(3));
        assert!(!vec1.insert(3));
        assert!(vec1.insert(5));
        assert!(vec2.insert(5));
        assert!(!vec2.insert(5));
        assert!(vec2.insert(64));

        let vec1 = vec1.difference_inplace(&vec2);

        assert!(vec1.contains(3));
        assert!(!vec1.contains(4));
        assert!(!vec1.contains(5));
        assert!(!vec1.contains(63));
        assert!(!vec1.contains(64));
    }

    #[test]
    fn bitvector_operator_overload() {
        let mut vec1 = BitVector::new(65);
        let mut vec2 = BitVector::new(65);
        assert!(vec1.insert(3));
        assert!(!vec1.insert(3));
        assert!(vec1.insert(5));
        assert!(vec2.insert(5));
        assert!(!vec2.insert(5));
        assert!(vec2.insert(64));

        let inter = &vec1 & &vec2;
        let union = &vec1 | &vec2;
        let diff = &vec1 ^ &vec2;

        assert!(union.contains(3));
        assert!(!union.contains(4));
        assert!(union.contains(5));
        assert!(!union.contains(63));
        assert!(union.contains(64));

        assert!(!inter.contains(3));
        assert!(!inter.contains(4));
        assert!(inter.contains(5));
        assert!(!inter.contains(63));
        assert!(!inter.contains(64));

        assert!(diff.contains(3));
        assert!(!diff.contains(4));
        assert!(!diff.contains(5));
        assert!(!diff.contains(63));
        assert!(!diff.contains(64));
    }

    #[test]
    fn bitvec_iter_works() {
        let mut bitvec = BitVector::new(100);
        bitvec.insert(1);
        bitvec.insert(10);
        bitvec.insert(19);
        bitvec.insert(62);
        bitvec.insert(63);
        bitvec.insert(64);
        bitvec.insert(65);
        bitvec.insert(66);
        bitvec.insert(99);
        assert_eq!(bitvec.iter().collect::<Vec<_>>(),
                   [1, 10, 19, 62, 63, 64, 65, 66, 99]);
    }


    #[test]
    fn bitvec_iter_works_2() {
        let mut bitvec = BitVector::new(319);
        bitvec.insert(0);
        bitvec.insert(127);
        bitvec.insert(191);
        bitvec.insert(255);
        bitvec.insert(319);
        assert_eq!(bitvec.iter().collect::<Vec<_>>(), [0, 127, 191, 255, 319]);
    }

    #[test]
    fn eq_left() {
        let mut bitvec = BitVector::new(50);
        for i in vec![0,1,3,5,11,12,19,23] { bitvec.insert(i); }
        let mut bitvec2 = BitVector::new(50);
        for i in vec![0,1,3,5,7,11,13,17,19,23] { bitvec2.insert(i); }

        assert!(bitvec.eq_left(&bitvec2, 1));
        assert!(bitvec.eq_left(&bitvec2, 2));
        assert!(bitvec.eq_left(&bitvec2, 3));
        assert!(bitvec.eq_left(&bitvec2, 4));
        assert!(bitvec.eq_left(&bitvec2, 5));
        assert!(bitvec.eq_left(&bitvec2, 6));
        assert!(bitvec.eq_left(&bitvec2, 7));
        assert!(!bitvec.eq_left(&bitvec2, 8));
        assert!(!bitvec.eq_left(&bitvec2, 9));
        assert!(!bitvec.eq_left(&bitvec2, 50));
    }

    #[test]
    fn eq() {
        let mut bitvec = BitVector::new(50);
        for i in vec![0,1,3,5,11,12,19,23] { bitvec.insert(i); }
        let mut bitvec2 = BitVector::new(50);
        for i in vec![0,1,3,5,7,11,13,17,19,23] { bitvec2.insert(i); }
        let mut bitvec3 = BitVector::new(50);
        for i in vec![0,1,3,5,11,12,19,23] { bitvec3.insert(i); }

        assert!(bitvec != bitvec2);
        assert!(bitvec == bitvec3);
        assert!(bitvec2 != bitvec3);
    }

    #[test]
    fn grow() {
        let mut vec1 = BitVector::new(65);
        for index in 0 .. 65 {
            assert!(vec1.insert(index));
            assert!(!vec1.insert(index));
        }
        vec1.grow(128);

        // Check if the bits set before growing are still set
        for index in 0 .. 65 {
            assert!(vec1.contains(index));
        }

        // Check if the new bits are all un-set
        for index in 65 .. 128 {
            assert!(!vec1.contains(index));
        }

        // Check that we can set all new bits without running out of bounds
        for index in 65 .. 128 {
            assert!(vec1.insert(index));
            assert!(!vec1.insert(index));
        }
    }
}

#[cfg(all(feature = "unstable", test))]
mod bench {
    extern crate test;
    use std::collections::{HashSet, BTreeSet};
    use self::test::Bencher;
    use super::*;
    #[bench]
    fn bench_bitset_operator(b: &mut Bencher) {

        b.iter(|| { 
            let mut vec1 = BitVector::new(65);
            let mut vec2 = BitVector::new(65);
            for i in vec![0,1,2,10,15,18,25,31,40,42,60,64] {
                vec1.insert(i);
            };
            for i in vec![3,5,7,12,13,15,21,25,30,29,42,50,61,62,63,64] {
                vec2.insert(i);
            };
            vec1.intersection(&vec2);
            vec1.union(&vec2);
            vec1.difference(&vec2);
        });
    }

    #[bench]
    fn bench_bitset_operator_inplace(b: &mut Bencher) {
        b.iter(|| { 
            let mut vec1 = BitVector::new(65);
            let mut vec2 = BitVector::new(65);
            for i in vec![0,1,2,10,15,18,25,31,40,42,60,64] {
                vec1.insert(i);
            };
            for i in vec![3,5,7,12,13,15,21,25,30,29,42,50,61,62,63,64] {
                vec2.insert(i);
            };
            vec1.intersection_inplace(&vec2);
            vec1.union_inplace(&vec2);
            vec1.difference_inplace(&vec2);
        });
    }

    #[bench]
    fn bench_hashset_operator(b: &mut Bencher) {
        b.iter(|| {
            let mut vec1 = HashSet::with_capacity(65);
            let mut vec2 = HashSet::with_capacity(65);
            for i in vec![0,1,2,10,15,18,25,31,40,42,60,64] {
                vec1.insert(i);
            };
            for i in vec![3,5,7,12,13,15,21,25,30,29,42,50,61,62,63,64] {
                vec2.insert(i);
            };

            vec1.intersection(&vec2).cloned().collect::<HashSet<_>>();
            vec1.union(&vec2).cloned().collect::<HashSet<_>>();
            vec1.difference(&vec2).cloned().collect::<HashSet<_>>();
        });
    }

    #[bench]
    fn bench_btreeset_operator(b: &mut Bencher) {
        b.iter(|| {
            let mut vec1 = BTreeSet::new();
            let mut vec2 = BTreeSet::new();
            for i in vec![0,1,2,10,15,18,25,31,40,42,60,64] {
                vec1.insert(i);
            };
            for i in vec![3,5,7,12,13,15,21,25,30,29,42,50,61,62,63,64] {
                vec2.insert(i);
            };

            vec1.intersection(&vec2).cloned().collect::<HashSet<_>>();
            vec1.union(&vec2).cloned().collect::<HashSet<_>>();
            vec1.difference(&vec2).cloned().collect::<HashSet<_>>();
        });
    }
}
