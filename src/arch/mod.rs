use std::{collections::HashMap, fmt::Display, marker::PhantomData, io::{Write, self}};

use crate::regalloc::RegAllocMapping;

use super::{
    ir::{BasicBlockId, FunctionId, Operation, Terminator, Type, Value},
    regalloc::RegisterAllocator,
};

/// [`rv64`] provides a backend for the RISC V 64GC architecture.
pub mod rv64;

/// [`x64`] provides a backend for the x86_64 architecture.
pub mod x64;

/// [`urcl`] provides a backend for the URCL intermediate language, which can be then used to
/// compile to a variety of targets.
pub mod urcl;

/// [`VReg`] is a representation of a register in [`VCode`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VReg {
    /// [`VReg::RealRegister`] represents a register that exists in the target architecture.
    RealRegister(usize),

    /// [`VReg::Virtual`] represents a virtual register, of which there are unlimited. These are
    /// converted into real registers or spilled registers depending on the register allocator.
    Virtual(usize),

    /// [`VReg::Spilled`] represents a spilled value on the stack. This variant is used when the
    /// register allocator could not fit a value into the limited registers of the target
    /// architecture.
    Spilled(usize),
}

impl Display for VReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VReg::RealRegister(r) => write!(f, "%r{}", r),
            VReg::Virtual(v) => write!(f, "${}", v),
            VReg::Spilled(s) => write!(f, "[spilled #{}]", s),
        }
    }
}

/// [`Location`] is a representation of a location in [`VCode`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Location {
    /// [`Location::InternalLabel`] represents a label pointing to some code within the current
    /// function.
    InternalLabel(usize),

    /// [`Location::Function`] represents a label pointing to code that corresponds to another
    /// function.
    Function(usize),
}

impl Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Location::InternalLabel(label) => write!(f, ".L{}", label),
            Location::Function(func) => write!(f, "F{}", func),
        }
    }
}

// TODO: document the rest of this and figure out what should be public and what shouldn't be.

pub struct VCode<I>
where
    I: Instr,
{
    pub name: String,
    pub functions: Vec<Function<I>>,
}

pub struct Function<I>
where
    I: Instr,
{
    pub name: String,
    pub arg_count: usize,
    pub pre_labels: Vec<I>,
    pub pre_return: Vec<I>,
    pub labels: Vec<LabelledInstructions<I>>,
}

pub struct LabelledInstructions<I>
where
    I: Instr,
{
    pub instructions: Vec<I>,
}

impl<I> Display for VCode<I>
where
    I: Display + Instr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, ";;; module = {}\n", self.name)?;

        for (i, func) in self.functions.iter().enumerate() {
            write!(f, "F{}:\n{}", i, func)?;
        }

        Ok(())
    }
}

impl<I> Display for Function<I>
where
    I: Display + Instr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}:", self.name)?;
        for (i, labelled) in self.labels.iter().enumerate() {
            write!(f, ".L{}:\n{}", i, labelled)?;
        }

        Ok(())
    }
}

impl<I> Display for LabelledInstructions<I>
where
    I: Display + Instr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for instr in self.instructions.iter() {
            writeln!(f, "    {}", instr)?;
        }

        Ok(())
    }
}

/// [`Instr`] is one of the two traits necessary to implement to define your own backend (the other
/// trait being [`InstructionSelector`]). This trait represents an instruction in the target
/// architecture that can be generated by `codegem`. It also contains various functions related to
/// the generated [`VCode`] such as [`Instr::mandatory_transforms`] and [`Instr::emit_assembly`].
pub trait Instr: Sized {
    /// Gets the registers available for selection by the register allocator.
    fn get_regs() -> Vec<VReg>;

    /// Gets the registers that can be used as function arguments.
    fn get_arg_regs() -> Vec<VReg>;

    /// Transforms the given [`VCode`] into assembly written to a file.
    fn emit_assembly(file: &mut impl Write, vcode: &VCode<Self>) -> io::Result<()>;

    /// Collects the registers in an instruction for register allocation.
    fn collect_registers<A>(&self, alloc: &mut A)
    where
        A: RegisterAllocator;

    /// Performs transformations on the [`VCode`] that occur before register allocation
    /// applications.
    fn pre_regalloc_apply_transforms(func: &mut Function<Self>, alloc: &HashMap<VReg, RegAllocMapping>);

    /// Applies the results of register allocation.
    fn apply_reg_allocs(&mut self, alloc: &HashMap<VReg, RegAllocMapping>);

    /// Performs transformations on the [`VCode`] that occur after register allocation.
    fn post_regalloc_transforms(vcode: &mut VCode<Self>);
}

/// [`InstructionSelector`] is the other trait that has to be implemented for a backend. This trait
/// selects instructions to use based on the [`Operation`]s and
/// [`Terminator`]s in the provided IR.
pub trait InstructionSelector: Default {
    /// [`InstructionSelector::Instruction`] is the type to use for instructions.
    type Instruction: Instr;

    /// Selects the instructions that occur before the function starts executing.
    fn select_pre_function_instructions(&mut self, gen: &mut VCodeGenerator<Self::Instruction, Self>);

    /// Selects an instruction to use for the given [`Operation`].
    fn select_instr(
        &mut self,
        gen: &mut VCodeGenerator<Self::Instruction, Self>,
        result: Option<Value>,
        type_: Type,
        op: Operation,
    );

    /// Selects an instruction to use for a given [`Terminator`].
    fn select_term(&mut self, gen: &mut VCodeGenerator<Self::Instruction, Self>, op: Terminator);

    /// Performs post generation transforms on the [`Function`].
    fn post_function_generation(&mut self, func: &mut Function<Self::Instruction>, gen: &mut VCodeGenerator<Self::Instruction, Self>);

    /// Performs post generation transforms on the [`VCode`].
    fn post_generation(&mut self, vcode: &mut VCode<Self::Instruction>);
}

/// [`VCodeGenerator`] is an internal structure used to generate [`VCode`] for a given backend.
pub struct VCodeGenerator<I, S>
where
    S: InstructionSelector<Instruction = I>,
    I: Instr,
{
    internal: VCode<I>,
    _phantom: PhantomData<S>,
    func_map: HashMap<FunctionId, usize>,
    label_map: HashMap<BasicBlockId, usize>,
    current_function: Option<usize>,
    current_block: Option<usize>,
    value_map: HashMap<Value, VReg>,
    vreg_index: usize,
}

impl<I, S> VCodeGenerator<I, S>
where
    S: InstructionSelector<Instruction = I>,
    I: Instr,
{
    pub(crate) fn new_module(name: &str) -> Self {
        Self {
            internal: VCode {
                name: name.to_owned(),
                functions: Vec::new(),
            },
            _phantom: PhantomData::default(),
            func_map: HashMap::new(),
            label_map: HashMap::new(),
            current_function: None,
            current_block: None,
            value_map: HashMap::new(),
            vreg_index: 0,
        }
    }

    /// Gets the [`VReg`] associated with the given [`Value`].
    pub fn get_vreg(&mut self, value: Value) -> VReg {
        let args = I::get_arg_regs();
        let arg_count = self.current_function.and_then(|v| self.internal.functions.get(v)).map(|v| v.arg_count).unwrap_or(0);
        if value.0 < arg_count {
            if value.0 < args.len() {
                args[value.0]
            } else {
                todo!();
            }
        } else if let Some(&reg) = self.value_map.get(&value) {
            reg
        } else {
            let reg = VReg::Virtual(self.vreg_index);
            self.vreg_index += 1;
            self.value_map.insert(value, reg);
            reg
        }
    }

    /// Creates a new [`VReg`] that is unassociated with any [`Value`].
    pub fn new_unassociated_vreg(&mut self) -> VReg {
        let reg = VReg::Virtual(self.vreg_index);
        self.vreg_index += 1;
        reg
    }

    /// Returns the map from IR [`FunctionId`]s to indexes into the [`VCode`]'s list of functions.
    pub fn func_map(&self) -> &HashMap<FunctionId, usize> {
        &self.func_map
    }

    /// Returns the map from IR [`BasicBlockId`]s to indexes into the current function's list of
    /// labels.
    pub fn label_map(&self) -> &HashMap<BasicBlockId, usize> {
        &self.label_map
    }

    pub(crate) fn add_function(&mut self, name: &str, id: FunctionId, arg_count: usize) {
        let f = self.internal.functions.len();
        self.internal.functions.push(Function {
            name: name.to_owned(),
            arg_count,
            pre_labels: Vec::new(),
            pre_return: Vec::new(),
            labels: Vec::new(),
        });
        self.func_map.insert(id, f);
    }

    pub(crate) fn switch_to_function(&mut self, id: FunctionId) {
        self.current_function = self.func_map.get(&id).cloned();
        self.label_map.clear();
        self.value_map.clear();
        self.vreg_index = 0;
    }

    pub(crate) fn push_label(&mut self, id: BasicBlockId) {
        if let Some(func) = self
            .current_function
            .and_then(|v| self.internal.functions.get_mut(v))
        {
            let label = func.labels.len();
            func.labels.push(LabelledInstructions {
                instructions: Vec::new(),
            });
            self.label_map.insert(id, label);
        }
    }

    pub fn push_prelabel_instruction(&mut self, instruction: I) {
        if let Some(func) = self
            .current_function
            .and_then(|v| self.internal.functions.get_mut(v))
        {
            func.pre_labels.push(instruction);
        }
    }

    pub fn push_prereturn_instruction(&mut self, instruction: I) {
        if let Some(func) = self
            .current_function
            .and_then(|v| self.internal.functions.get_mut(v))
        {
            func.pre_return.push(instruction);
        }
    }

    pub(crate) fn switch_to_label(&mut self, id: BasicBlockId) {
        self.current_block = self.label_map.get(&id).cloned();
    }

    /// Pushes an instruction into the [`VCode`] being generated.
    pub fn push_instruction(&mut self, instruction: I) {
        if let Some(func) = self
            .current_function
            .and_then(|v| self.internal.functions.get_mut(v))
        {
            if let Some(labelled) = self.current_block.and_then(|v| func.labels.get_mut(v)) {
                labelled.instructions.push(instruction);
            }
        }
    }

    pub(crate) fn post_function(&mut self, selector: &mut S) {
        let mut internal = VCode {
            name: String::new(),
            functions: Vec::new(),
        };
        std::mem::swap(&mut internal, &mut self.internal);
        if let Some(func) = self.current_function.and_then(|v| internal.functions.get_mut(v)) {
            selector.post_function_generation(func, self);
        }
        std::mem::swap(&mut internal, &mut self.internal);
    }

    pub(crate) fn build(mut self, mut selector: S) -> VCode<I> {
        selector.post_generation(&mut self.internal);
        self.internal
    }
}

impl<I> VCode<I>
where
    I: Instr,
{
    /// Performs register allocation on the [`VCode`].
    pub fn allocate_regs<A>(&mut self)
    where
        A: RegisterAllocator,
    {
        for func in self.functions.iter_mut() {
            let mut allocator = A::default();

            for labelled in func.labels.iter() {
                for instr in labelled.instructions.iter() {
                    instr.collect_registers(&mut allocator);
                    allocator.next_live_step();
                }
            }

            let allocations = allocator.allocate_regs::<I>();
            I::pre_regalloc_apply_transforms(func, &allocations);
            for labelled in func.labels.iter_mut() {
                for instr in labelled.instructions.iter_mut() {
                    instr.apply_reg_allocs(&allocations);
                }
            }
        }

        I::post_regalloc_transforms(self);
    }

    /// Emits assembly for the [`VCode`]. Register allocation via [`VCode::allocate_regs`] must
    /// have been done beforehand, as well as any optional transforms.
    pub fn emit_assembly(&self, file: &mut impl Write) -> io::Result<()> {
        I::emit_assembly(file, self)
    }
}
