use std::{collections::HashMap, fmt::Display, fs::File, io::Write};

use crate::codegen::{
    ir::{Operation, Terminator, Type, Value},
    RegisterAllocator,
};

use super::{Instr, InstructionSelector, Location, VCode, VCodeGenerator, VReg};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstVal {
    Reg(VReg),
    Imm(u64),
}

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
        rx: InstVal,
        ry: InstVal,
    },

    Sub {
        rd: VReg,
        rx: InstVal,
        ry: InstVal,
    },

    Mlt {
        rd: VReg,
        rx: InstVal,
        ry: InstVal,
    },

    Div {
        rd: VReg,
        rx: InstVal,
        ry: InstVal,
    },

    Mod {
        rd: VReg,
        rx: InstVal,
        ry: InstVal,
    },

    Bsl {
        rd: VReg,
        rx: InstVal,
        ry: InstVal,
    },

    Bsr {
        rd: VReg,
        rx: InstVal,
        ry: InstVal,
    },

    Bre {
        location: Location,
        rx: InstVal,
        ry: InstVal,
    },

    Bne {
        location: Location,
        rx: InstVal,
        ry: InstVal,
    },

    Brl {
        location: Location,
        rx: InstVal,
        ry: InstVal,
    },

    Ble {
        location: Location,
        rx: InstVal,
        ry: InstVal,
    },

    Brg {
        location: Location,
        rx: InstVal,
        ry: InstVal,
    },

    Bge {
        location: Location,
        rx: InstVal,
        ry: InstVal,
    },

    Jmp {
        location: Location,
    },

    And {
        rd: VReg,
        rx: InstVal,
        ry: InstVal,
    },

    Or {
        rd: VReg,
        rx: InstVal,
        ry: InstVal,
    },

    Xor {
        rd: VReg,
        rx: InstVal,
        ry: InstVal,
    },

    Lod {
        rd: VReg,
        rx: InstVal,
    },

    Str {
        rd: VReg,
        rx: InstVal,
    },

    Cal {
        location: Location
    },

    Ret,

    Hlt,
}

impl Display for InstVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstVal::Imm(v) => write!(f, "{}", v),
            InstVal::Reg(v) => write!(f, "{}", v),
        }
    }
}

impl InstVal {
    pub fn is_reg(self) -> bool {
        if let InstVal::Reg(_) = self {
            return true;
        }
        false
    }

    pub fn is_imm(self) -> bool {
        return !self.is_reg();
    }

    pub fn get_reg(self) -> Option<VReg> {
        match self {
            InstVal::Reg(r) => Some(r),
            InstVal::Imm(_) => None,
        }
    }

    pub fn get_imm(self) -> Option<u64> {
        match self {
            InstVal::Reg(_) => None,
            InstVal::Imm(v) => Some(v),
        }
    }

    fn apply_instval_reg<A>(self, alloc: &mut A)
    where
        A: RegisterAllocator,
    {
        if self.is_reg() {
            alloc.add_use(self.get_reg().unwrap());
        }
    }

    fn apply_reg_alloc(self, alloc: &HashMap<VReg, VReg>) {
        if self.is_reg() {
            apply_alloc(alloc, &mut self.get_reg().unwrap());
        }
    }
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

    fn collect_registers<A>(&self, alloc: &mut A)
    where
        A: RegisterAllocator,
    {
        match self {
            UrclInstruction::PhiPlaceholder { .. } => (),

            UrclInstruction::Imm { rd, .. } => {
                alloc.add_def(*rd);
            }

            UrclInstruction::Add { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Jmp { .. } => (),

            UrclInstruction::Lod { rd, rx } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
            },

            UrclInstruction::Str { rd, rx } => {
                alloc.add_use(*rd);
                rx.apply_instval_reg(alloc);
            },

            UrclInstruction::Brl { rx, ry, .. } => {
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Ble { rx, ry, .. } => {
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Brg { rx, ry, .. } => {
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Bge { rx, ry, .. } => {
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Bre { rx, ry, .. } => {
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Bne { rx, ry, .. } => {
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Sub { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Mlt { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Div { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Mod { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::And { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Or { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Xor { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Bsl { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Bsr { rd, rx, ry } => {
                alloc.add_def(*rd);
                rx.apply_instval_reg(alloc);
                ry.apply_instval_reg(alloc);
            }

            UrclInstruction::Hlt => (),

            UrclInstruction::Cal { .. } => (),

            UrclInstruction::Ret => ()
        }
    }




    fn apply_reg_allocs(&mut self, alloc: &HashMap<VReg, VReg>) {
        match self {
            UrclInstruction::PhiPlaceholder { .. } => (),

            UrclInstruction::Imm { rd, .. } => {
                apply_alloc(alloc, rd);
            }

            UrclInstruction::Add { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Jmp { .. } => (),

            UrclInstruction::Bne { rx, ry, .. } => {
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Lod { rd, rx } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
            },

            UrclInstruction::Str { rd, rx } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
            },

            UrclInstruction::Brl { rx, ry, .. } => {
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Ble { rx, ry, ..} => {
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Brg { rx, ry, .. } => {
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Bge { rx, ry, .. } => {
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Bre { rx, ry, .. } => {
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Sub { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Mlt { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Div { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Mod { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::And { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Or { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Xor { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Bsl { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Bsr { rd, rx, ry } => {
                apply_alloc(alloc, rd);
                rx.apply_reg_alloc(alloc);
                ry.apply_reg_alloc(alloc);
            }

            UrclInstruction::Hlt => (),

            UrclInstruction::Cal { .. } => (),

            UrclInstruction::Ret => (),
        }
    }

    fn mandatory_transforms(_vcode: &mut VCode<Self>) {
        // TODO
    }

    fn emit_assembly(vcode: &VCode<Self>) {
        match File::create(format!("{}.urcl", vcode.name)) {
            Ok(mut file) => {
                let _ = writeln!(file, "minreg 8");
                let _ = writeln!(file, "bits 64");
                for func in vcode.functions.iter() {
                    let _ = writeln!(file, ".{}", func.name);
                    for (i, labelled) in func.labels.iter().enumerate() {
                        let _ = writeln!(file, ".L{}", i);
                        for instruction in labelled.instructions.iter() {
                            match instruction {
                                UrclInstruction::PhiPlaceholder { .. } => (),

                                UrclInstruction::Imm { rd, value } => {
                                    let _ = writeln!(file, "    imm {} {}", register(*rd), value);
                                }

                                UrclInstruction::Add { rd, rx, ry } => {
                                    let _ = writeln!(file, "    add {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::Jmp { location } => {
                                    match *location {
                                        Location::InternalLabel(_) => {
                                            let _ = writeln!(file, "    jmp {}", location);
                                        }
                                        Location::Function(f) => {
                                            let _ = writeln!(file, "    jmp {}", vcode.functions[f].name);
                                        }
                                    }
                                }

                                UrclInstruction::Bne { rx, ry, location } => {
                                    match *location {
                                        Location::InternalLabel(_) => {
                                            let _ = writeln!(file, "    bne {} {} {}", location, register_instval(*rx), register_instval(*ry) );
                                        }
                                        Location::Function(f) => {
                                            let _ = writeln!(file, "    bne {} {} {}", vcode.functions[f].name, register_instval(*rx), register_instval(*ry) );
                                        }
                                    }
                                }

                                UrclInstruction::Lod { rd,  rx } => {
                                    let _ = writeln!(file, "    lod {} {}", rd, rx);
                                }

                                UrclInstruction::Str { rd, rx} => {
                                    let _ = writeln!(file, "    str {} {}", rd, rx);
                                }

                                UrclInstruction::Brl { rx, ry, location } => {
                                    match *location {
                                        Location::InternalLabel(_) => {
                                            let _ = writeln!(file, "    brl {} {} {}", location, register_instval(*rx), register_instval(*ry) );
                                        }
                                        Location::Function(f) => {
                                            let _ = writeln!(file, "    brl {} {} {}", vcode.functions[f].name, register_instval(*rx), register_instval(*ry) );
                                        }
                                    }
                                }

                                UrclInstruction::Ble { rx, ry, location } => {
                                    match *location {
                                        Location::InternalLabel(_) => {
                                            let _ = writeln!(file, "    ble {} {} {}", location, register_instval(*rx), register_instval(*ry) );
                                        }
                                        Location::Function(f) => {
                                            let _ = writeln!(file, "    ble {} {} {}", vcode.functions[f].name, register_instval(*rx), register_instval(*ry) );
                                        }
                                    }
                                }

                                UrclInstruction::Brg { rx, ry, location } => {
                                    match *location {
                                        Location::InternalLabel(_) => {
                                            let _ = writeln!(file, "    brg {} {} {}", location, register_instval(*rx), register_instval(*ry) );
                                        }
                                        Location::Function(f) => {
                                            let _ = writeln!(file, "    brg {} {} {}", vcode.functions[f].name, register_instval(*rx), register_instval(*ry) );
                                        }
                                    }
                                }

                                UrclInstruction::Bge { rx, ry, location } => {
                                    match *location {
                                        Location::InternalLabel(_) => {
                                            let _ = writeln!(file, "    bge {} {} {}", location, register_instval(*rx), register_instval(*ry) );
                                        }
                                        Location::Function(f) => {
                                            let _ = writeln!(file, "    bge {} {} {}", vcode.functions[f].name, register_instval(*rx), register_instval(*ry) );
                                        }
                                    }
                                }

                                UrclInstruction::Bre { rx, ry, location } => {
                                    match *location {
                                        Location::InternalLabel(_) => {
                                            let _ = writeln!(file, "    bre {} {} {}", location, register_instval(*rx), register_instval(*ry) );
                                        }
                                        Location::Function(f) => {
                                            let _ = writeln!(file, "    bre {} {} {}", vcode.functions[f].name, register_instval(*rx), register_instval(*ry) );
                                        }
                                    }
                                }

                                UrclInstruction::Sub { rd, rx, ry } => {
                                    let _ = writeln!(file, "    sub {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::Mlt { rd, rx, ry } => {
                                    let _ = writeln!(file, "    mlt {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::Div { rd, rx, ry } => {
                                    let _ = writeln!(file, "    div {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::Mod { rd, rx, ry } => {
                                    let _ = writeln!(file, "    mod {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::And { rd, rx, ry } => {
                                    let _ = writeln!(file, "    and {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::Or { rd, rx, ry } => {
                                    let _ = writeln!(file, "    or {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::Xor { rd, rx, ry } => {
                                    let _ = writeln!(file, "    xor {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::Bsr { rd, rx, ry } => {
                                    let _ = writeln!(file, "    bsr {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::Bsl { rd, rx, ry } => {
                                    let _ = writeln!(file, "    bsl {} {} {}", register(*rd), register_instval(*rx), register_instval(*ry));
                                }

                                UrclInstruction::Hlt => {
                                    let _ = writeln!(file, "    hlt");
                                }

                                UrclInstruction::Cal { location } => {
                                    let _ = writeln!(file, "    cal {}", location);
                                }

                                UrclInstruction::Ret => {
                                    let _ = writeln!(file, "    ret");
                                }
                            }
                        }
                    }

                    let _ = writeln!(file);
                }
            }
            Err(e) => {
                eprintln!("Could not open file `{}`: {}", vcode.name, e);
            }
        }
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

fn register_instval(instval: InstVal) -> String {
    if instval.is_reg() {
        return register(instval.get_reg().unwrap());
    } else {
        return instval.get_imm().unwrap().to_string();
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
                    gen.push_instruction(UrclInstruction::Add { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(VReg::RealRegister(URCL_REGISTER_ZERO)) });
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
                        gen.push_instruction(UrclInstruction::Add { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
                    }
                }
            }

            Operation::Sub(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Sub { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
                    }
                }
            },
            Operation::Mul(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Mlt { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
                    }
                }
            },
            Operation::Div(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Div { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
                    }
                }
            },
            Operation::Mod(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Mod { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
                    }
                }
            },
            Operation::Bsl(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Bsl { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
                    }
                }
            },
            Operation::Bsr(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Bsr { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
                    }
                }
            },
            Operation::Eq(_, _) => todo!(),
            Operation::Ne(_, _) => todo!(),
            Operation::Lt(_, _) => todo!(),
            Operation::Le(_, _) => todo!(),
            Operation::Gt(_, _) => todo!(),
            Operation::Ge(_, _) => todo!(),
            Operation::BitAnd(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::And { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
                    }
                }
            },
            Operation::BitOr(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Or { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
                    }
                }
            },
            Operation::BitXor(a, b) => {
                if let Some(&rx) = self.value_map.get(&a) {
                    if let Some(&ry) = self.value_map.get(&b) {
                        gen.push_instruction(UrclInstruction::Xor { rd, rx: InstVal::Reg(rx), ry: InstVal::Reg(ry) });
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
                            rx: InstVal::Reg(rx),
                            ry: InstVal::Reg(VReg::RealRegister(URCL_REGISTER_ZERO)),
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
                                    rx: InstVal::Reg(rx),
                                    ry: InstVal::Reg(VReg::RealRegister(URCL_REGISTER_ZERO)),
                                },
                            );
                        }
                    }
                }
            }
        }
    }
}
