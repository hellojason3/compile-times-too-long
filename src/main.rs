mod filed;
mod generated_code;

use std::env;

use std::collections::HashMap;
use ark_ff::{BigInteger, Fp256, MontBackend, MontConfig, PrimeField};
use std::fmt::Debug;
use std::io::Write;
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
use crate::filed::{Fp, SimpleField};
use crate::generated_code::get_f;
fn get_name(idx: usize) -> String {
    let prefix = "a";
    format!("{}_{}", prefix, idx)
}

struct Pow {
    a: usize,
    b: usize,
    c: usize
}

fn generate_code(rounds: usize) {
    let mut pows = String::new();
    let mut rng = rand::thread_rng();
    for i in 2..rounds+2 {
        let idx1 = rng.gen_range(0..i-1);
        let idx2 = rng.gen_range(0..i-1);
        pows.push_str(&format!("let a{} = a{}.clone() * &a{};\n", i, idx1, idx2));
    }
    pows.push_str(&format!("a{}\n", rounds, ));
    let write_file_name = "src/generated_code.rs";

    let content = format!("
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::fields::fp::FpVar;
use ark_relations::r1cs::ConstraintSystem;
use ark_std::UniformRand;
use crate::filed::{{Fp, Fvp}};

pub fn get_f() -> Fvp{{
    let mut rng = &mut ark_std::test_rng();
    let random_point = Fp::rand(&mut rng);
    let cs = ConstraintSystem::<Fp>::new_ref();
    let a0 = FpVar::<Fp>::new_witness(cs.clone(), || Ok(random_point)).unwrap();
    let a1 = FpVar::<Fp>::new_witness(cs.clone(), || Ok(random_point)).unwrap();
    {}

}}", pows);

    let mut file = std::fs::File::create(write_file_name).unwrap();
    file.write_all(content.as_bytes()).unwrap();
    file.flush().unwrap();
}
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 {
        let rounds = args[1].parse::<usize>().unwrap();
        println!("rounds: {}", rounds);
        generate_code(rounds);
    }else {
        let a = get_f();
        println!("{}", hex::encode(a.value().unwrap().into_bigint().to_bytes_le()));
    }
}
