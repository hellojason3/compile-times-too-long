
use std::collections::HashMap;
use ark_ff::{Fp256, MontBackend, MontConfig, PrimeField};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::FieldVar;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::prelude::UInt8;
use ark_r1cs_std::{R1CSVar, ToBitsGadget, ToBytesGadget};
use ark_relations::lc;
use ark_relations::r1cs::ConstraintSystem;
use ark_std::UniformRand;
use ark_std::rand::Rng;
#[derive(MontConfig)]
#[modulus = "3618502788666131213697322783095070105623107215331596699973092056135872020481"]
#[generator = "3"]
pub struct FpMontConfig;

pub type Fp = Fp256<MontBackend<FpMontConfig, 4>>;

#[derive(MontConfig)]
#[modulus = "3618502788666131213697322783095070105526743751716087489154079457884512865583"]
#[generator = "3"]
pub struct FrMontConfig;

pub type Fr = Fp256<MontBackend<FrMontConfig, 4>>;

pub type Fvp = FpVar<Fp>;

pub trait SimpleField:
Clone
+ Sized
+ Add<Self, Output = Self>
+ Sub<Self, Output = Self>
+ Mul<Self, Output = Self>
+ AddAssign<Self>
+ SubAssign<Self>
+ MulAssign<Self>
+ for<'a> Add<&'a Self, Output = Self>
+ for<'a> Sub<&'a Self, Output = Self>
+ for<'a> Mul<&'a Self, Output = Self>
+ for<'a> AddAssign<&'a Self>
+ for<'a> SubAssign<&'a Self>
+ for<'a> MulAssign<&'a Self>
{
    type Value: PrimeField + SimpleField;
    type BooleanType: Clone + Debug;
    type ByteType: Clone + Debug;

    fn zero() -> Self;
    fn one() -> Self;
    fn two() -> Self;
    fn three() -> Self;
    fn four() -> Self;
    fn get_value(&self) -> Self::Value;
    fn negate(&self) -> Self;
    fn inv(&self) -> Self;
    fn mul_by_constant(&self, n: impl Into<num_bigint::BigUint>) -> Self;
    fn powers<Exp: AsRef<[u64]>>(&self, n: Exp) -> Self;
    fn powers_felt(&self, n: &Self) -> Self;
    fn from_constant(value: impl Into<num_bigint::BigUint>) -> Self;
    fn from_boolean(value: Self::BooleanType) -> Self;
    fn into_boolean(&self) -> Self::BooleanType;
    fn from_biguint(value: num_bigint::BigUint) -> Self;
    fn into_biguint(&self) -> num_bigint::BigUint;
    fn into_constant<T: TryFrom<num_bigint::BigUint>>(&self) -> T
    where
        <T as TryFrom<num_bigint::BigUint>>::Error: Debug;
    fn assert_equal(&self, other: &Self);
    fn assert_not_equal(&self, other: &Self);
    fn div_rem(&self, other: &Self) -> (Self, Self);
    fn div2_rem(&self) -> (Self, Self);
    fn rsh(&self, n: usize) -> Self;
    fn rsh_rem(&self, n: usize) -> (Self, Self);
    fn lsh(&self, n: usize) -> Self;
    fn field_div(&self, other: &Self) -> Self;
    fn select(cond: &Self::BooleanType, true_value: Self, false_value: Self) -> Self;
    fn is_equal(&self, other: &Self) -> Self::BooleanType;
    fn is_not_equal(&self, other: &Self) -> Self::BooleanType;
    fn is_zero(&self) -> Self::BooleanType;
    fn is_one(&self) -> Self::BooleanType;
    fn and(lhs: &Self::BooleanType, rhs: &Self::BooleanType) -> Self::BooleanType;
    fn or(lhs: &Self::BooleanType, rhs: &Self::BooleanType) -> Self::BooleanType;
    fn xor(lhs: &Self::BooleanType, rhs: &Self::BooleanType) -> Self::BooleanType;
    fn not(value: &Self::BooleanType) -> Self::BooleanType;
    fn greater_than(&self, other: &Self) -> Self::BooleanType;
    fn less_than(&self, other: &Self) -> Self::BooleanType;
    fn lte(&self, other: &Self) -> Self::BooleanType;
    fn gte(&self, other: &Self) -> Self::BooleanType;
    fn assert_true(value: Self::BooleanType);
    fn assert_false(value: Self::BooleanType);
    fn assert_gt(&self, other: &Self);
    fn assert_lt(&self, other: &Self);
    fn assert_lte(&self, other: &Self);
    fn assert_gte(&self, other: &Self);
    fn to_le_bytes(&self) -> Vec<Self::ByteType>;
    fn to_be_bytes(&self) -> Vec<Self::ByteType>;
    fn from_be_bytes(bytes: &[Self::ByteType]) -> Self;
    fn from_le_bytes(bytes: &[Self::ByteType]) -> Self;
    fn to_le_bits(&self) -> Vec<Self::BooleanType>;
    fn to_be_bits(&self) -> Vec<Self::BooleanType>;
    fn from_le_bits(bits: &[Self::BooleanType]) -> Self;
    fn from_be_bits(bits: &[Self::BooleanType]) -> Self;
    fn reverse_bits(&self, n: usize) -> Self;
    fn construct_byte(value: u8) -> Self::ByteType;
    fn construct_bool(value: bool) -> Self::BooleanType;

    fn sort(values: Vec<Self>) -> Vec<Self>;
    fn slice(values: &[Self], start: &Self, end: &Self) -> Vec<Self>;
    fn skip(values: &[Self], n: &Self) -> Vec<Self>;
    fn take(values: &[Self], n: &Self) -> Vec<Self>;
    fn range(start: &Self, end: &Self) -> Vec<Self>;
    fn at(values: &[Self], i: &Self) -> Self;
}


impl<F: PrimeField + SimpleField> SimpleField for FpVar<F> {
    type Value = F;
    type BooleanType = Boolean<F>;
    type ByteType = UInt8<F>;

    fn zero() -> Self {
        FpVar::Constant(SimpleField::zero())
    }

    fn one() -> Self {
        FpVar::Constant(SimpleField::one())
    }

    fn two() -> Self {
        FpVar::Constant(SimpleField::two())
    }

    fn negate(&self) -> Self {
        FieldVar::<F, F>::negate(self).unwrap()
    }

    fn is_zero(&self) -> Self::BooleanType {
        EqGadget::is_eq(self, &SimpleField::zero()).unwrap()
    }

    fn is_one(&self) -> Self::BooleanType {
        EqGadget::is_eq(self, &SimpleField::one()).unwrap()
    }

    fn inv(&self) -> Self {
        if self.is_constant() {
            FpVar::Constant(self.value().unwrap().inv().clone())
        } else {
            let is_zero = <Self as SimpleField>::is_zero(self);
            let inv = FpVar::new_witness(self.cs(), || Ok(self.value()?.inv())).unwrap();
            self.mul_equals(&inv, &Self::from_boolean(Self::not(&is_zero)))
                .unwrap();
            inv
        }
    }

    fn from_constant(value: impl Into<num_bigint::BigUint>) -> Self {
        FpVar::Constant(F::from_constant(value))
    }

    fn from_boolean(value: Self::BooleanType) -> Self {
        Self::from(value)
    }

    fn into_boolean(&self) -> Self::BooleanType {
        Self::assert_true(Self::or(
            &SimpleField::is_zero(self),
            &SimpleField::is_one(self),
        ));
        Boolean::select(
            &SimpleField::is_zero(self),
            &Boolean::<F>::FALSE,
            &Boolean::<F>::TRUE,
        )
            .unwrap()
    }

    fn from_biguint(value: num_bigint::BigUint) -> Self {
        FpVar::Constant(F::from_biguint(value))
    }

    fn into_biguint(&self) -> num_bigint::BigUint {
        self.value().unwrap().try_into().unwrap()
    }

    fn into_constant<T: TryFrom<num_bigint::BigUint>>(&self) -> T
    where
        <T as TryFrom<num_bigint::BigUint>>::Error: Debug,
    {
        self.into_biguint().try_into().unwrap()
    }

    fn powers<Exp: AsRef<[u64]>>(&self, n: Exp) -> Self {
        self.pow_by_constant(n).unwrap()
    }

    fn assert_equal(&self, other: &Self) {
        assert!(ark_r1cs_std::eq::EqGadget::enforce_equal(self, other).is_ok());
    }

    fn assert_not_equal(&self, other: &Self) {
        assert!(ark_r1cs_std::eq::EqGadget::enforce_not_equal(self, other).is_ok());
    }

    fn powers_felt(&self, n: &Self) -> Self {
        ark_r1cs_std::bits::ToBitsGadget::to_bits_le(n)
            .and_then(|bits| FieldVar::pow_le(self, &bits))
            .unwrap()
    }

    fn div_rem(&self, other: &Self) -> (Self, Self) {
        if let (FpVar::Constant(dividend), FpVar::Constant(divisor)) = (self, other) {
            let (quotient, remainder) = dividend.div_rem(divisor);
            return (FpVar::Constant(quotient), FpVar::Constant(remainder));
        }

        let cs = self.cs().or(other.cs());

        let quotient = Self::new_witness(cs.clone(), || {
            Ok(self.value().unwrap().div_rem(&other.value().unwrap()).0)
        })
            .unwrap();

        let remainder = Self::new_witness(cs.clone(), || {
            Ok(self.value().unwrap().div_rem(&other.value().unwrap()).1)
        })
            .unwrap();

        (quotient.clone() * other + &remainder)
            .enforce_equal(self)
            .unwrap();

        (quotient, remainder)
    }

    fn div2_rem(&self) -> (Self, Self) {
        self.div_rem(&Self::two())
        // let bits = self.to_bits_le().unwrap();
        // if bits.is_empty() {
        //     return (SimpleField::zero(), SimpleField::zero());
        // }
        //
        // let (left, right) = bits.split_at(1);
        // (
        //     Boolean::le_bits_to_fp_var(right).unwrap(),
        //     Boolean::le_bits_to_fp_var(left).unwrap(),
        // )
    }

    fn select(cond: &Self::BooleanType, a: Self, b: Self) -> Self {
        cond.select(&a, &b).unwrap()
    }

    fn is_equal(&self, other: &Self) -> Self::BooleanType {
        EqGadget::is_eq(self, other).unwrap()
    }

    fn and(lhs: &Self::BooleanType, rhs: &Self::BooleanType) -> Self::BooleanType {
        Boolean::and(lhs, rhs).unwrap()
    }

    fn or(lhs: &Self::BooleanType, rhs: &Self::BooleanType) -> Self::BooleanType {
        Boolean::or(lhs, rhs).unwrap()
    }

    fn xor(lhs: &Self::BooleanType, rhs: &Self::BooleanType) -> Self::BooleanType {
        Boolean::xor(lhs, rhs).unwrap()
    }

    fn not(value: &Self::BooleanType) -> Self::BooleanType {
        Boolean::not(value)
    }

    fn assert_gt(&self, other: &Self) {
        FpVar::<F>::enforce_cmp(self, other, core::cmp::Ordering::Greater, false).unwrap();
    }

    fn assert_lt(&self, other: &Self) {
        FpVar::<F>::enforce_cmp(self, other, core::cmp::Ordering::Less, false).unwrap();
    }

    fn assert_gte(&self, other: &Self) {
        FpVar::<F>::enforce_cmp(self, other, core::cmp::Ordering::Greater, true).unwrap();
    }

    fn assert_lte(&self, other: &Self) {
        FpVar::<F>::enforce_cmp(self, other, core::cmp::Ordering::Less, true).unwrap();
    }

    fn field_div(&self, other: &Self) -> Self {
        self.mul(other.inv())
    }

    fn rsh(&self, n: usize) -> Self {
        // let bits = self.to_bits_le().unwrap();
        // if bits.is_empty() || n >= bits.len() {
        //     return SimpleField::zero();
        // }
        //
        // Boolean::le_bits_to_fp_var(&bits[n..]).unwrap()
        self.div_rem(&Self::from_biguint(num_bigint::BigUint::from(1u64) << n)).0
    }

    fn rsh_rem(&self, n: usize) -> (Self, Self) {
        self.div_rem(&Self::from_biguint(num_bigint::BigUint::from(1u64) << n))
        // let bits = self.to_bits_le().unwrap();
        // if bits.is_empty() || n >= bits.len() {
        //     return (SimpleField::zero(), self.clone());
        // }
        //
        // (
        //     Boolean::le_bits_to_fp_var(&bits[n..]).unwrap(),
        //     Boolean::le_bits_to_fp_var(&bits[..n]).unwrap(),
        // )
    }

    fn greater_than(&self, other: &Self) -> Self::BooleanType {
        FpVar::<F>::is_cmp_unchecked(self, other, core::cmp::Ordering::Greater, false).unwrap()
    }

    fn less_than(&self, other: &Self) -> Self::BooleanType {
        FpVar::<F>::is_cmp_unchecked(self, other, core::cmp::Ordering::Less, false).unwrap()
    }

    fn lte(&self, other: &Self) -> Self::BooleanType {
        FpVar::<F>::is_cmp_unchecked(self, other, core::cmp::Ordering::Less, true).unwrap()
    }

    fn gte(&self, other: &Self) -> Self::BooleanType {
        FpVar::<F>::is_cmp_unchecked(self, other, core::cmp::Ordering::Greater, true).unwrap()
    }

    fn three() -> Self {
        FpVar::Constant(SimpleField::three())
    }

    fn four() -> Self {
        FpVar::Constant(SimpleField::four())
    }

    fn is_not_equal(&self, other: &Self) -> Self::BooleanType {
        EqGadget::is_neq(self, other).unwrap()
    }

    fn assert_true(value: Self::BooleanType) {
        value.enforce_equal(&Boolean::<F>::TRUE).unwrap()
    }

    fn assert_false(value: Self::BooleanType) {
        value.enforce_equal(&Boolean::<F>::FALSE).unwrap()
    }

    fn lsh(&self, n: usize) -> Self {
        self * &Self::from_biguint(num_bigint::BigUint::from(1u64) << n)
        // let bits = self.to_bits_le().unwrap();
        // if bits.is_empty() || n >= bits.len() {
        //     return SimpleField::zero();
        // }
        //
        // return Boolean::le_bits_to_fp_var(
        //     &core::iter::repeat(Boolean::<F>::FALSE)
        //         .take(n)
        //         .chain(bits.iter().cloned())
        //         .collect::<Vec<_>>(),
        // )
        // .unwrap();
    }

    fn to_le_bytes(&self) -> Vec<Self::ByteType> {
        ToBytesGadget::to_bytes(self).unwrap()
    }

    fn to_be_bytes(&self) -> Vec<Self::ByteType> {
        ToBytesGadget::to_bytes(self)
            .unwrap()
            .into_iter()
            .rev()
            .collect()
    }

    fn to_le_bits(&self) -> Vec<Self::BooleanType> {
        ToBitsGadget::to_bits_le(self).unwrap()
    }

    fn to_be_bits(&self) -> Vec<Self::BooleanType> {
        ToBitsGadget::to_bits_be(self).unwrap()
    }

    fn construct_byte(value: u8) -> Self::ByteType {
        UInt8::<F>::constant(value)
    }

    fn construct_bool(value: bool) -> Self::BooleanType {
        Boolean::<F>::constant(value)
    }

    fn from_be_bytes(bytes: &[Self::ByteType]) -> Self {
        Boolean::le_bits_to_fp_var(
            &bytes
                .iter()
                .rev()
                .flat_map(|b| b.to_bits_le().unwrap())
                .collect::<Vec<_>>(),
        )
            .unwrap()
    }

    fn from_le_bytes(bytes: &[Self::ByteType]) -> Self {
        Boolean::le_bits_to_fp_var(
            &bytes
                .iter()
                .flat_map(|b| b.to_bits_le().unwrap())
                .collect::<Vec<_>>(),
        )
            .unwrap()
    }

    fn from_le_bits(bits: &[Self::BooleanType]) -> Self {
        Boolean::le_bits_to_fp_var(&bits).unwrap()
    }

    fn from_be_bits(bits: &[Self::BooleanType]) -> Self {
        Boolean::le_bits_to_fp_var(
            bits.into_iter()
                .cloned()
                .rev()
                .collect::<Vec<_>>()
                .as_slice(),
        )
            .unwrap()
    }

    // Unsafe
    fn reverse_bits(&self, n: usize) -> Self {
        if self.is_constant() {
            return FpVar::Constant(self.value().unwrap().reverse_bits(n));
        }

        let r = Self::new_witness(self.cs(), || Ok(self.value().unwrap().reverse_bits(n))).unwrap();

        // TODO: we should have been able to write this in circuits but arkworks' to_non_unique_bits looks buggy
        r
    }

    fn get_value(&self) -> Self::Value {
        self.value().unwrap().clone()
    }

    // Unsafe
    fn sort(values: Vec<Self>) -> Vec<Self> {
        if values.is_empty() {
            return vec![];
        }

        let cs = values.cs();

        // All should be constants
        if cs.is_none() {
            let mut new_values = values.value().unwrap();
            new_values.sort();
            return new_values.into_iter().map(|v| FpVar::Constant(v)).collect();
        }

        let new_values = Vec::<Self>::new_witness(cs, || {
            let mut values = values.value().unwrap();
            values.sort();
            Ok(values)
        })
            .unwrap();

        // TODO: check new_values come from the original vector
        new_values.iter().reduce(|prev, next| {
            SimpleField::assert_lte(prev, next);
            next
        });

        new_values
    }

    // Unsafe
    fn slice(values: &[Self], start: &Self, end: &Self) -> Vec<Self> {
        let cs = values.cs();

        // All should be constants
        if cs.is_none() {
            let start = start.get_value();
            let end = end.get_value();
            return values[start.into_constant()..end.into_constant()].to_vec();
        }

        let slice = Vec::<Self>::new_witness(cs, || {
            let start = start.get_value();
            let end = end.get_value();
            let values = values.value().unwrap();
            Ok(values[start.into_constant()..end.into_constant()].to_vec())
        })
            .unwrap();

        // TODO: check new slice comes from the original slice
        slice
    }

    // Unsafe
    fn skip(values: &[Self], n: &Self) -> Vec<Self> {
        let cs = values.cs();

        // All should be constants
        if cs.is_none() {
            let start = n.get_value();
            return values[start.into_constant()..].to_vec();
        }

        let slice = Vec::<Self>::new_witness(cs, || {
            let start = n.get_value();
            let values = values.value().unwrap();
            Ok(values[start.into_constant()..].to_vec())
        })
            .unwrap();

        // TODO: check new slice comes from the original slice
        slice
    }

    // Unsafe
    fn take(values: &[Self], n: &Self) -> Vec<Self> {
        let cs = values.cs();

        // All should be constants
        if cs.is_none() {
            let start = n.get_value();
            return values[..start.into_constant()].to_vec();
        }

        let slice = Vec::<Self>::new_witness(cs, || {
            let start = n.get_value();
            let values = values.value().unwrap();
            Ok(values[..start.into_constant()].to_vec())
        })
            .unwrap();

        // TODO: check new slice comes from the original slice
        slice
    }

    // Unsafe
    fn range(start: &Self, end: &Self) -> Vec<Self> {
        let cs = start.cs().or(end.cs());

        if cs.is_none() {
            return (start.value().unwrap().into_constant::<usize>()
                ..end.value().unwrap().into_constant::<usize>())
                .map(|v| Self::from_constant(v))
                .collect::<Vec<_>>();
        }

        let range = Vec::<Self>::new_witness(cs, || {
            Ok((start.value().unwrap().into_constant::<usize>()
                ..end.value().unwrap().into_constant::<usize>())
                .map(|v| F::from_constant(v))
                .collect::<Vec<_>>())
        })
            .unwrap();

        if let Some(first) = range.first() {
            first.assert_equal(start);
        }

        if let Some(last) = range.last() {
            last.assert_equal(&(end.clone() - &SimpleField::one()))
        }

        range.iter().reduce(|prev, next| {
            next.assert_equal(&(prev + &SimpleField::one()));
            next
        });

        return range;
    }

    // Unsafe
    fn at(values: &[Self], i: &Self) -> Self {
        let cs = values.cs();
        if i.is_constant() || cs.is_none() {
            return values[i.into_constant::<usize>()].clone();
        }

        let value = Self::new_witness(cs, || {
            Ok(<Self as R1CSVar<F>>::value(&values[i.into_constant::<usize>()]).unwrap())
        })
            .unwrap();

        // TODO: check value equals to element at i position
        value
    }

    fn mul_by_constant(&self, n: impl Into<num_bigint::BigUint>) -> Self {
        self.mul(&Self::from_constant(n))
    }
}