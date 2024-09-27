use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::fields::fp::FpVar;
use ark_relations::r1cs::ConstraintSystem;
use ark_std::UniformRand;
use crate::filed::{{Fp, Fvp}};

pub fn get_f() -> Fvp{
    let mut rng = &mut ark_std::test_rng();
    let random_point = Fp::rand(&mut rng);
    let cs = ConstraintSystem::<Fp>::new_ref();
    let a0 = FpVar::<Fp>::new_witness(cs.clone(), || Ok(random_point)).unwrap();
    let a1 = FpVar::<Fp>::new_witness(cs.clone(), || Ok(random_point)).unwrap();


}