use std::collections::HashMap;

use self::arch::{Instr, VReg};

pub mod arch;

/// [`codegem::ir`] contains the intermediate representation (IR) that programs using this crate
/// interface with, as well as a builder struct to create the intermediate representation.
pub mod ir;

pub mod regalloc;

pub trait RegisterAllocator: Default {
    fn add_use(&mut self, reg: VReg);

    fn add_def(&mut self, reg: VReg);

    fn force_same(&mut self, reg: VReg, constraint: VReg);

    fn next_live_step(&mut self);

    fn allocate_regs<I>(self) -> HashMap<VReg, VReg>
    where
        I: Instr;
}
