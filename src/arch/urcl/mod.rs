use std::{collections::HashMap, fmt::Display, fs::File, io::Write};

use crate::{
    ir::{Operation, Terminator, Type, Value},
    regalloc::RegisterAllocator,
};

use super::{Instr, InstructionSelector, Location, VCode, VCodeGenerator, VReg, Function};

pub const URCL_REGISTER_ZERO: usize = 0;
pub const URCL_REGISTER_PC: usize = 1;
pub const URCL_REGISTER_SP: usize = 2;
pub const URCL_REGISTER_R1: usize = 3;
pub const URCL_REGISTER_R2: usize = 4;
pub const URCL_REGISTER_R3: usize = 5;
pub const URCL_REGISTER_R4: usize = 6;
pub const URCL_REGISTER_R5: usize = 7;
pub const URCL_REGISTER_R6: usize = 8;
pub const URCL_REGISTER_R7: usize = 9;
pub const URCL_REGISTER_R8: usize = 10;

pub enum UrclInstruction {
    PhiPlaceholder {
        rd: VReg,
        options: Vec<(Location, VReg)>,
    },

    Imm {
        rd: VReg,
        value: u64,
    },

    Add {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Sub {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Mlt {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Div {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Mod {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Bsl {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Bsr {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Bre {
        location: Location,
        rx: VReg,
        ry: VReg,
    },

    Bne {
        location: Location,
        rx: VReg,
        ry: VReg,
    },

    Brl {
        location: Location,
        rx: VReg,
        ry: VReg,
    },

    Ble {
        location: Location,
        rx: VReg,
        ry: VReg,
    },

    Brg {
        location: Location,
        rx: VReg,
        ry: VReg,
    },

    Bge {
        location: Location,
        rx: VReg,
        ry: VReg,
    },

    Jmp {
        location: Location,
    },

    And {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Or {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Xor {
        rd: VReg,
        rx: VReg,
        ry: VReg,
    },

    Lod {
        rd: VReg,
        rx: VReg,
    },

    Str {
        rd: VReg,
        rx: VReg,
    },

    Cal {
        location: Location
    },

    Ret,

    Hlt,

    Sete {
        rd: VReg,
        rx: VReg,
        ry: VReg
    },

    Setne {
        rd: VReg,
        rx: VReg,
        ry: VReg
    },

    Setg {
        rd: VReg,
        rx: VReg,
        ry: VReg
    },

    Setge {
        rd: VReg,
        rx: VReg,
        ry: VReg
    },

    Setl {
        rd: VReg,
        rx: VReg,
        ry: VReg
    },

    Setle {
        rd: VReg,
        rx: VReg,
        ry: VReg
    },
}


impl Display for UrclInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UrclInstruction::PhiPlaceholder { rd, .. } => write!(f, "phi {} ...", rd),

            UrclInstruction::Imm { rd, value } => write!(f, "imm {} {}", rd, value),

            UrclInstruction::Add { rd, rx, ry } => write!(f, "add {} {} {}", rd, rx, ry),

            UrclInstruction::Jmp { location } => write!(f, "jmp {}", location),

            UrclInstruction::Bne { rx, ry, location } => write!(f, "bne {} {} {}", location, rx, ry),

            UrclInstruction::Bre { location, rx, ry } => write!(f, "bre {} {} {}", location, rx, ry),

            UrclInstruction::Brl { location, rx, ry } => write!(f, "brl {} {} {}", location, rx, ry),
            
            UrclInstruction::Ble { location, rx, ry } => write!(f, "ble {} {} {}", location, rx, ry),

            UrclInstruction::Brg { location, rx, ry } => write!(f, "brg {} {} {}", location, rx, ry),

            UrclInstruction::Bge { location, rx, ry } => write!(f, "bge {} {} {}", location, rx, ry),

            UrclInstruction::Sub { rd, rx, ry } => write!(f, "sub {} {} {}", rd, rx, ry),

            UrclInstruction::And { rd, rx, ry } => write!(f, "and {} {} {}", rd, rx, ry),

            UrclInstruction::Or { rd, rx, ry } => write!(f, "or {} {} {}", rd, rx, ry),

            UrclInstruction::Xor { rd, rx, ry } => write!(f, "xor {} {} {}", rd, rx, ry),

            UrclInstruction::Mod { rd, rx, ry } => write!(f, "mod {} {} {}", rd, rx, ry),

            UrclInstruction::Div { rd, rx, ry } => write!(f, "div {} {} {}", rd, rx, ry),

            UrclInstruction::Mlt { rd, rx, ry } => write!(f, "Mlt {} {} {}", rd, rx, ry),

            UrclInstruction::Bsl { rd, rx, ry } => write!(f, "bsl {} {} {}", rd, rx, ry),

            UrclInstruction::Bsr { rd, rx, ry } => write!(f, "bsr {} {} {}", rd, rx, ry),

            UrclInstruction::Lod { rd,  rx } => write!(f, "lod {} {}", rd, rx),

            UrclInstruction::Str { rd, rx } => write!(f, "str {} {}", rd, rx),

            UrclInstruction::Hlt => write!(f, "hlt"),

            UrclInstruction::Cal { location } => write!(f, "cal {}", location),

            UrclInstruction::Ret => write!(f, "ret"),

            UrclInstruction::Sete { rd, rx, ry } => write!(f, "Sete {} {} {}", rd, rx, ry),

            UrclInstruction::Setne { rd, rx, ry } => write!(f, "setne {} {} {}", rd, rx, ry),

            UrclInstruction::Setg { rd, rx, ry } => write!(f, "setg {} {} {}", rd, rx, ry),

            UrclInstruction::Setge { rd, rx, ry } => write!(f, "setge {} {} {}", rd, rx, ry),

            UrclInstruction::Setl { rd, rx, ry } => write!(f, "setl {} {} {}", rd, rx, ry),

            UrclInstruction::Setle { rd, rx, ry } => write!(f, "setle {} {} {}", rd, rx, ry),
        }
    }
}

impl Instr for UrclInstruction {
    fn get_regs() -> Vec<VReg> {
        vec![
            VReg::RealRegister(URCL_REGISTER_R1),
            VReg::RealRegister(URCL_REGISTER_R2),
            VReg::RealRegister(URCL_REGISTER_R3),
            VReg::RealRegister(URCL_REGISTER_R4),
            VReg::RealRegister(URCL_REGISTER_R5),
            VReg::RealRegister(URCL_REGISTER_R6),
            VReg::RealRegister(URCL_REGISTER_R7),
            VReg::RealRegister(URCL_REGISTER_R8),
        ]
    }

    fn get_arg_regs() -> Vec<VReg> {
        vec![]
    }

    fn collect_registers<A>(&self, alloc: &mut A)
    where
        A: RegisterAllocator,
    {
        match self {
            UrclInstruction::PhiPlaceholder { .. } => (),

            UrclInstruction::Imm { rd, .. } => {
                alloc.add_def(*rd);
            }

            UrclInstruction::Jmp { .. } => (),

            UrclInstruction::Lod { rd, rx } => {
                alloc.add_def(*rd);
                alloc.add_use(*rx);
            },

            UrclInstruction::Str { rd, rx } => {
                alloc.add_use(*rd);
                alloc.add_use(*rx);
            },

            UrclInstruction::Brl { rx, ry, .. } => {
                alloc.add_use(*rx);
                alloc.add_use(*ry);
            }

            UrclInstruction::Ble { rx, ry, .. } => {
                alloc.add_use(*rx);
                alloc.add_use(*ry);
            }

            UrclInstruction::Brg { rx, ry, .. } => {
                alloc.add_use(*rx);
                alloc.add_use(*ry);
            }

            UrclInstruction::Bge { rx, ry, .. } => {
                alloc.add_use(*rx);
                alloc.add_use(*ry);
            }

            UrclInstruction::Bre { rx, ry, .. } => {
                alloc.add_use(*rx);
                alloc.add_use(*ry);
            }

            UrclInstruction::Bne { rx, ry, .. } => {
                alloc.add_use(*rx);
                alloc.add_use(*ry);
            }



            UrclInstruction::Hlt => (),

            UrclInstruction::Cal { .. } => (),

            UrclInstruction::Ret => (),

            UrclInstruction::Sub { rd, rx, ry }
            | UrclInstruction::Mlt { rd, rx, ry }
            | UrclInstruction::Add { rd, rx, ry }
            | UrclInstruction::Div { rd, rx, ry }
            | UrclInstruction::Mod { rd, rx, ry }
            | UrclInstruction::And { rd, rx, ry }
            | UrclInstruction::Bsl { rd, rx, ry }
            | UrclInstruction::Bsr { rd, rx, ry }
            | UrclInstruction::Xor { rd, rx, ry }
            | UrclInstruction::Or { rd, rx, ry }
            | UrclInstruction::Sete { rd, rx, ry }
            | UrclInstruction::Setne { rd, rx, ry }
            | UrclInstruction::Setg { rd, rx, ry }
            | UrclInstruction::Setge { rd, rx, ry }
            | UrclInstruction::Setl { rd, rx, ry }
            | UrclInstruction::Setle { rd, rx, ry } => {
                alloc.add_def(*rd);
                alloc.add_use(*rx);
                alloc.add_use(*ry);
            }
        }
    }




    fn apply_reg_allocs(&mut self, alloc: &HashMap<VReg, VReg>) {
        match self {
            UrclInstruction::PhiPlaceholder { .. } => (),

            UrclInstruction::Imm { rd, .. } => {
                apply_alloc(alloc, rd);
            }

            UrclInstruction::Jmp { .. } => (),

            UrclInstruction::Bne { rx, ry, .. } => {
                apply_alloc(alloc, rx);
                apply_alloc(alloc, ry);
            }

            UrclInstruction::Lod { rd, rx } => {
                apply_alloc(alloc, rd);
                apply_alloc(alloc, rx);
            },

            UrclInstruction::Str { rd, rx } => {
                apply_alloc(alloc, rd);
                apply_alloc(alloc, rx);
            },

            UrclInstruction::Brl { rx, ry, .. } => {
                apply_alloc(alloc, rx);
                apply_alloc(alloc, ry);
            }

            UrclInstruction::Ble { rx, ry, ..} => {
                apply_alloc(alloc, rx);
                apply_alloc(alloc, ry);
            }

            UrclInstruction::Brg { rx, ry, .. } => {
                apply_alloc(alloc, rx);
                apply_alloc(alloc, ry);
            }

            UrclInstruction::Bge { rx, ry, .. } => {
                apply_alloc(alloc, rx);
                apply_alloc(alloc, ry);
            }

            UrclInstruction::Bre { rx, ry, .. } => {
                apply_alloc(alloc, rx);
                apply_alloc(alloc, ry);
            }

            UrclInstruction::Hlt => (),

            UrclInstruction::Cal { .. } => (),

            UrclInstruction::Ret => (),

            UrclInstruction::Add { rd, rx, ry }
            | UrclInstruction::Sete { rd, rx, ry }
            | UrclInstruction::Setne { rd, rx, ry }
            | UrclInstruction::Setg { rd, rx, ry }
            | UrclInstruction::Setge { rd, rx, ry }
            | UrclInstruction::Setl { rd, rx, ry }
            | UrclInstruction::Setle { rd, rx, ry }
            | UrclInstruction::Sub { rd, rx, ry }
            | UrclInstruction::Mlt { rd, rx, ry }
            | UrclInstruction::Div { rd, rx, ry }
            | UrclInstruction::Mod { rd, rx, ry }
            | UrclInstruction::And { rd, rx, ry }
            | UrclInstruction::Or { rd, rx, ry }
            | UrclInstruction::Xor { rd, rx, ry }
            | UrclInstruction::Bsl { rd, rx, ry }
            | UrclInstruction::Bsr { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                apply_alloc(alloc, rx);
                apply_alloc(alloc, ry);
            }
        }
    }

    fn mandatory_transforms(_vcode: &mut VCode<Self>) {
        // TODO
    }

    fn emit_assembly(vcode: &VCode<Self>) -> std::io::Result<()> {
        let mut file = File::create(format!("{}.urcl", vcode.name))?;

        writeln!(file, "minreg 8")?;
        writeln!(file, "bits 64")?;
        for func in vcode.functions.iter() {
            writeln!(file, ".{}", func.name)?;
            for (i, labelled) in func.labels.iter().enumerate() {
                writeln!(file, ".L{}", i)?;
                for instruction in labelled.instructions.iter() {
                    match instruction {
                        UrclInstruction::PhiPlaceholder { .. } => (),

                        UrclInstruction::Imm { rd, value } => {
                            writeln!(file, "    imm {} {}", register(*rd), value)?;
                        }

                        UrclInstruction::Add { rd, rx, ry } => {
                            writeln!(file, "    add {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::Jmp { location } => {
                            match *location {
                                Location::InternalLabel(_) => {
                                    writeln!(file, "    jmp {}", location)?;
                                }
                                Location::Function(f) => {
                                    writeln!(file, "    jmp {}", vcode.functions[f].name)?;
                                }
                            }
                        }

                        UrclInstruction::Bne { rx, ry, location } => {
                            match *location {
                                Location::InternalLabel(_) => {
                                    writeln!(file, "    bne {} {} {}", location, register(*rx), register(*ry) )?;
                                }
                                Location::Function(f) => {
                                    writeln!(file, "    bne {} {} {}", vcode.functions[f].name, register(*rx), register(*ry) )?;
                                }
                            }
                        }

                        UrclInstruction::Lod { rd,  rx } => {
                            writeln!(file, "    lod {} {}", rd, rx)?;
                        }

                        UrclInstruction::Str { rd, rx} => {
                            writeln!(file, "    str {} {}", rd, rx)?;
                        }

                        UrclInstruction::Brl { rx, ry, location } => {
                            match *location {
                                Location::InternalLabel(_) => {
                                    writeln!(file, "    brl {} {} {}", location, register(*rx), register(*ry) )?;
                                }
                                Location::Function(f) => {
                                    writeln!(file, "    brl {} {} {}", vcode.functions[f].name, register(*rx), register(*ry) )?;
                                }
                            }
                        }

                        UrclInstruction::Ble { rx, ry, location } => {
                            match *location {
                                Location::InternalLabel(_) => {
                                    writeln!(file, "    ble {} {} {}", location, register(*rx), register(*ry) )?;
                                }
                                Location::Function(f) => {
                                    writeln!(file, "    ble {} {} {}", vcode.functions[f].name, register(*rx), register(*ry) )?;
                                }
                            }
                        }

                        UrclInstruction::Brg { rx, ry, location } => {
                            match *location {
                                Location::InternalLabel(_) => {
                                    writeln!(file, "    brg {} {} {}", location, register(*rx), register(*ry) )?;
                                }
                                Location::Function(f) => {
                                    writeln!(file, "    brg {} {} {}", vcode.functions[f].name, register(*rx), register(*ry) )?;
                                }
                            }
                        }

                        UrclInstruction::Bge { rx, ry, location } => {
                            match *location {
                                Location::InternalLabel(_) => {
                                    writeln!(file, "    bge {} {} {}", location, register(*rx), register(*ry) )?;
                                }
                                Location::Function(f) => {
                                    writeln!(file, "    bge {} {} {}", vcode.functions[f].name, register(*rx), register(*ry) )?;
                                }
                            }
                        }

                        UrclInstruction::Bre { rx, ry, location } => {
                            match *location {
                                Location::InternalLabel(_) => {
                                    writeln!(file, "    bre {} {} {}", location, register(*rx), register(*ry) )?;
                                }
                                Location::Function(f) => {
                                    writeln!(file, "    bre {} {} {}", vcode.functions[f].name, register(*rx), register(*ry) )?;
                                }
                            }
                        }

                        UrclInstruction::Sub { rd, rx, ry } => {
                            writeln!(file, "    sub {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::Mlt { rd, rx, ry } => {
                            writeln!(file, "    mlt {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::Div { rd, rx, ry } => {
                            writeln!(file, "    div {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::Mod { rd, rx, ry } => {
                            writeln!(file, "    mod {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::And { rd, rx, ry } => {
                            writeln!(file, "    and {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::Or { rd, rx, ry } => {
                            writeln!(file, "    or {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::Xor { rd, rx, ry } => {
                            writeln!(file, "    xor {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::Bsr { rd, rx, ry } => {
                            writeln!(file, "    bsr {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::Bsl { rd, rx, ry } => {
                            writeln!(file, "    bsl {} {} {}", register(*rd), register(*rx), register(*ry))?;
                        }

                        UrclInstruction::Hlt => {
                            writeln!(file, "    hlt")?;
                        }

                        UrclInstruction::Cal { location } => {
                            writeln!(file, "    cal {}", location)?;
                        }

                        UrclInstruction::Ret => {
                            writeln!(file, "    ret")?;
                        }

                        UrclInstruction::Sete { rd, rx, ry } => {
                            writeln!(file, "    Sete {} {} {}", rd, rx, ry)?;
                        }

                        UrclInstruction::Setne { rd, rx, ry } => {
                            writeln!(file, "    setne {} {} {}", rd, rx, ry)?;
                        }

                        UrclInstruction::Setg { rd, rx, ry } => {
                            writeln!(file, "    setg {} {} {}", rd, rx, ry)?;
                        }

                        UrclInstruction::Setge { rd, rx, ry } => {
                            writeln!(file, "    setge {} {} {}", rd, rx, ry)?;
                        }

                        UrclInstruction::Setl { rd, rx, ry } => {
                            writeln!(file, "    setl {} {} {}", rd, rx, ry)?;
                        }

                        UrclInstruction::Setle { rd, rx, ry } => {
                            writeln!(file, "    setle {} {} {}", rd, rx, ry)?;
                        }
                    }
                }
            }

            writeln!(file)?;
        }
        Ok(())
    }
}

fn apply_alloc(alloc: &HashMap<VReg, VReg>, reg: &mut VReg) {
    if let Some(new) = alloc.get(reg) {
        *reg = *new;
    }
}

fn register(reg: VReg) -> String {
    match reg {
        VReg::RealRegister(reg) => {
            String::from(match reg {
                v if v == URCL_REGISTER_ZERO => "r0",
                v if v == URCL_REGISTER_PC => "pc",
                v if v == URCL_REGISTER_SP => "sp",
                v if v == URCL_REGISTER_R1 => "r1",
                v if v == URCL_REGISTER_R2 => "r2",
                v if v == URCL_REGISTER_R3 => "r3",
                v if v == URCL_REGISTER_R4 => "r4",
                v if v == URCL_REGISTER_R5 => "r5",
                v if v == URCL_REGISTER_R6 => "r6",
                v if v == URCL_REGISTER_R7 => "r7",
                v if v == URCL_REGISTER_R8 => "r8",
                _ => unreachable!(),
            })
        }

        VReg::Virtual(_) => unreachable!(),

        VReg::Spilled(s) => format!("-{}(fp)", 8 * s),
    }
}

#[derive(Default)]
pub struct UrclSelector {
    value_map: HashMap<Value, VReg>,
    vreg_index: usize,
}

impl InstructionSelector for UrclSelector {
    type Instruction = UrclInstruction;

    fn select_instr(
        &mut self,
        gen: &mut VCodeGenerator<Self::Instruction, Self>,
        result: Option<Value>,
        _type_: Type,
        op: Operation,
    ) {
        let rd = match result {
            Some(val) => {
                let dest = VReg::Virtual(self.vreg_index);
                self.vreg_index += 1;
                self.value_map.insert(val, dest);
                dest
            }

            None => VReg::RealRegister(URCL_REGISTER_ZERO),
        };

        match op {
            Operation::Identity(value) => {
                if let Some(&rx) = self.value_map.get(&value) {
                    gen.push_instruction(UrclInstruction::Add { rd, rx, ry: VReg::RealRegister(URCL_REGISTER_ZERO) });
                }
            }

            Operation::Integer(_signed, mut value) => {
                // TODO: better way to do this
                while value.len() < 8 {
                    value.push(0);
                }
                let value = u64::from_le_bytes(value[..8].try_into().unwrap());
                gen.push_instruction(UrclInstruction::Imm { rd, value });
            }

            Operation::Add(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Add { rd, rx, ry });
                    }
                }
            }

            Operation::Sub(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Sub { rd, rx, ry });
                    }
                }
            },
            Operation::Mul(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Mlt { rd, rx, ry });
                    }
                }
            },
            Operation::Div(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Div { rd, rx, ry });
                    }
                }
            },
            Operation::Mod(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Mod { rd, rx, ry });
                    }
                }
            },
            Operation::Bsl(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Bsl { rd, rx, ry });
                    }
                }
            },
            Operation::Bsr(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Bsr { rd, rx, ry });
                    }
                }
            },
            Operation::Eq(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Sete { rd, rx, ry });
                    }
                }
            },
            Operation::Ne(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Setne { rd, rx, ry });
                    }
                }
            },
            Operation::Lt(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Setl { rd, rx, ry });
                    }
                }
            },
            Operation::Le(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Setle { rd, rx, ry });
                    }
                }
            },
            Operation::Gt(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Setg { rd, rx, ry });
                    }
                }
            },
            Operation::Ge(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Setge { rd, rx, ry });
                    }
                }
            },
            Operation::BitAnd(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::And { rd, rx, ry });
                    }
                }
            },
            Operation::BitOr(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Or { rd, rx, ry });
                    }
                }
            },
            Operation::BitXor(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Xor { rd, rx, ry });
                    }
                }
            },

            Operation::Phi(mapping) => {
                gen.push_instruction(UrclInstruction::PhiPlaceholder {
                    rd,
                    options: mapping
                        .into_iter()
                        .filter_map(|(b, v)| {
                            if let Some(&l) = gen.label_map().get(&b) {
                                if let Some(&r) = self.value_map.get(&v) {
                                    return Some((Location::InternalLabel(l), r));
                                }
                            }

                            None
                        })
                        .collect(),
                });
            }

            Operation::GetVar(_) => unreachable!(),
            Operation::SetVar(_, _) => unreachable!(),

            Operation::Call(_, _) => todo!(),
            Operation::CallIndirect(_, _) => todo!(),
        }
    }

    fn select_term(&mut self, gen: &mut VCodeGenerator<Self::Instruction, Self>, op: Terminator) {
        match op {
            Terminator::NoTerminator => (),

            Terminator::ReturnVoid => {
                gen.push_instruction(UrclInstruction::Ret);
            }

            Terminator::Return(_) => {
                gen.push_instruction(UrclInstruction::Hlt);
            }

            Terminator::Jump(label) => {
                if let Some(&label) = gen.label_map().get(&label) {
                    gen.push_instruction(UrclInstruction::Jmp {
                        location: Location::InternalLabel(label),
                    });
                }
            }

            Terminator::Branch(v, l1, l2) => {
                if let Some(&rx) = self.value_map.get(&v) {
                    if let Some(&l1) = gen.label_map().get(&l1) {
                        gen.push_instruction(UrclInstruction::Bne {
                            rx,
                            ry: VReg::RealRegister(URCL_REGISTER_ZERO),
                            location: Location::InternalLabel(l1),
                        });
                    }
                    if let Some(&l2) = gen.label_map().get(&l2) {
                        gen.push_instruction(UrclInstruction::Jmp {
                            location: Location::InternalLabel(l2),
                        });
                    }
                }
            }
        }
    }

    fn post_function_generation(&mut self, _func: &mut Function<Self::Instruction>, _gen: &mut VCodeGenerator<Self::Instruction, Self>) {
    }

    fn post_generation(&mut self, vcode: &mut VCode<Self::Instruction>) {
        for func in vcode.functions.iter_mut() {
            let mut v = Vec::new();
            for (i, labelled) in func.labels.iter().enumerate() {
                for (j, instr) in labelled.instructions.iter().enumerate() {
                    if let UrclInstruction::PhiPlaceholder { .. } = instr {
                        v.push((i, j));
                    }
                }
            }

            for (label_index, instr_index) in v.into_iter().rev() {
                let phi = func.labels[label_index].instructions.remove(instr_index);
                if let UrclInstruction::PhiPlaceholder { rd, options } = phi {
                    for (label, rx) in options {
                        if let Location::InternalLabel(label) = label {
                            let labelled = &mut func.labels[label];
                            labelled.instructions.insert(
                                labelled.instructions.len() - 1,
                                UrclInstruction::Add {
                                    rd,
                                    rx,
                                    ry: VReg::RealRegister(URCL_REGISTER_ZERO),
                                },
                            );
                        }
                    }
                }
            }
        }
    }
}
