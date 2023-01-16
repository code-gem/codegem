#![warn(missing_docs)]

use std::{fmt::Display, collections::{HashSet, HashMap}};

use super::arch::{Instr, InstructionSelector, VCode, VCodeGenerator};

/// [`Module`] represents a module of SSA intermediate representation.
#[derive(Default)]
pub struct Module {
    name: String,
    functions: Vec<Function>,
}

impl Module {
    /// Lowers the module into [`VCode`]. The first generic argument can be ignored, whereas the
    /// second generic argument is the instruction selector to use, which usually implies the first
    /// generic argument.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # use codegem::arch::rv64::*;
    /// # fn test(module: Module) {
    /// let vcode = module.lower_to_vcode::<_, RvSelector>();
    /// # }
    /// ```
    pub fn lower_to_vcode<I, S>(self) -> VCode<I>
    where
        S: InstructionSelector<Instruction = I>,
        I: Instr,
    {
        let mut gen = VCodeGenerator::<I, S>::new_module(&self.name);
        let mut selector = S::default();

        for (i, function) in self.functions.iter().enumerate() {
            gen.add_function(&function.name, function.linkage, FunctionId(i), function.arg_types.len());
        }

        for (f, func) in self.functions.into_iter().enumerate() {
            gen.switch_to_function(FunctionId(f));
            for (i, block) in func.blocks.iter().enumerate() {
                if block.deleted {
                    continue;
                }

                gen.push_label(BasicBlockId(FunctionId(f), i));
            }

            for (i, block) in func.blocks.into_iter().enumerate() {
                if block.deleted {
                    continue;
                }

                gen.switch_to_label(BasicBlockId(FunctionId(f), i));
                if i == 0 {
                    selector.select_pre_function_instructions(&mut gen);
                }

                for instr in block.instructions {
                    selector.select_instr(&mut gen, instr.yielded, instr.operation);
                }
                selector.select_term(&mut gen, block.terminator);
            }

            gen.post_function(&mut selector);
        }

        gen.build(selector)
    }
}

impl Display for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "/* module {} */", self.name)?;
        for (i, func) in self.functions.iter().enumerate() {
            writeln!(f, "\n\n@{}: {}", i, func)?;
        }

        Ok(())
    }
}

/// [`Type`] represents a type in the IR.
#[derive(Clone)]
pub enum Type {
    /// A `void` type analogous to `void` in C or `()` in Rust.
    Void,

    /// A signed or unsigned integer type with a given bitwidth.
    Integer(bool, u8),
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Integer(signed, width) => {
                write!(f, "{}{}", if *signed { "i" } else { "u" }, width)
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// [`Linkage`] is the linkage for a given [`Function`].
pub enum Linkage {
    /// An external linkage indicates that the function is written elsewhere.
    External,

    /// A private function is internal to the module and not exposed to outside modules.
    Private,

    /// A public function is internal to the module but exposed to outside modules.
    Public,
}

impl Display for Linkage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Linkage::External => write!(f, "external"),
            Linkage::Private => write!(f, "private"),
            Linkage::Public => write!(f, "public"),
        }
    }
}

struct Function {
    name: String,
    linkage: Linkage,
    arg_types: Vec<Type>,
    ret_type: Type,
    variables: Vec<Variable>,
    blocks: Vec<BasicBlock>,
    value_index: usize,
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "function {} {} @{}(", self.linkage, self.ret_type, self.name)?;
        let mut first = true;
        for (i, arg_type) in self.arg_types.iter().enumerate() {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "%{}: {}", i, arg_type)?;
        }
        writeln!(f, ") {{")?;

        for (i, var) in self.variables.iter().enumerate() {
            writeln!(f, "    #{} : {} // {}", i, var.type_, var.name)?;
        }

        for (i, block) in self.blocks.iter().enumerate() {
            if !block.deleted {
                write!(f, "{}: // preds =", i)?;
                for pred in block.predecessors.iter() {
                    write!(f, " {}", pred)?;
                }
                write!(f, "\n{}", block)?;
            }
        }
        write!(f, "}}")
    }
}

/// [`FunctionId`] represents a reference to a function in IR.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct FunctionId(usize);

impl Display for FunctionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@{}", self.0)
    }
}

struct Variable {
    name: String,
    type_: Type,
}

/// [`VariableId`] represents a reference to a variable in an IR function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct VariableId(usize);

impl Display for VariableId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

struct BasicBlock {
    deleted: bool,
    predecessors: HashSet<BasicBlockId>,
    instructions: Vec<Instruction>,
    terminator: Terminator,
}

impl Display for BasicBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for instruction in self.instructions.iter() {
            writeln!(f, "    {}", instruction)?;
        }
        writeln!(f, "    {}", self.terminator)
    }
}

/// [`BasicBlockId`] represents a reference to a basic block in an IR function. See
/// [`ModuleBuilder::push_block`] for details on what a basic block is.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BasicBlockId(FunctionId, usize);

impl Display for BasicBlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${}", self.1)
    }
}

struct Instruction {
    yielded: Option<Value>,
    //type_: Type,
    operation: Operation,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(yielded) = &self.yielded {
            write!(f, "{} = ", yielded)?;
        }

        write!(f, "{}", self.operation)
    }
}

/// [`ToIntegerOperation`] converts the given value into an integer operation.
pub trait ToIntegerOperation {
    /// Converts an integer into an integer operation.
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// let op = 69i32.to_integer_operation();
    /// ```
    fn to_integer_operation(self) -> Operation;
}

impl ToIntegerOperation for i8 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

impl ToIntegerOperation for u8 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

impl ToIntegerOperation for i16 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

impl ToIntegerOperation for u16 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

impl ToIntegerOperation for i32 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

impl ToIntegerOperation for u32 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

impl ToIntegerOperation for i64 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

impl ToIntegerOperation for u64 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

impl ToIntegerOperation for i128 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

impl ToIntegerOperation for u128 {
    fn to_integer_operation(self) -> Operation {
        Operation::Integer(true, self.to_le_bytes().to_vec())
    }
}

/// [`Operation`] is an operation that can be performed by the IR.
pub enum Operation {
    /// Does nothing to the given value, returning a new value that has the same contents as the
    /// passed in value. This is used internally.
    Identity(Value),

    /// Creates an integer from the little endian bytes passed in and sign extending it if the
    /// first parameter is true.
    Integer(bool, Vec<u8>),

    /// Performs an addition on the two values.
    Add(Value, Value),

    /// Performs a subtraction on the two values.
    Sub(Value, Value),

    /// Performs a multiplication on the two values.
    Mul(Value, Value),

    /// Performs a division on the two values.
    Div(Value, Value),

    /// Performs a modulus on the two values.
    Mod(Value, Value),

    /// Performs a left logical bit shift on the two values.
    Bsl(Value, Value),

    /// Performs a right logical bit shift on the two values.
    Bsr(Value, Value),

    /// Performs an equality check on the two values.
    Eq(Value, Value),

    /// Performs an inverted equality check on the two values.
    Ne(Value, Value),

    /// Performs a comparison check on the two values, returning true if less than and false
    /// otherwise.
    Lt(Value, Value),

    /// Performs a comparison check on the two values, returning true if less than or equal to
    /// and false otherwise.
    Le(Value, Value),

    /// Performs a comparison check on the two values, returning true if greater than and false
    /// otherwise.
    Gt(Value, Value),

    /// Performs a comparison check on the two values, returning true if less than or equal to
    /// and false otherwise.
    Ge(Value, Value),

    /// Performs a bitwise and on the two values.
    BitAnd(Value, Value),

    /// Performs a bitwise or on the two values.
    BitOr(Value, Value),

    /// Performs a bitwise xor on the two values.
    BitXor(Value, Value),

    /// Performs a phi operation.
    ///
    /// The phi operation is a concept stolen from LLVM. It allows a basic block to choose which
    /// value to yield depending on which basic block was the predecessor of the basic block the
    /// phi node resides in.
    ///
    /// # Example generated code
    /// ```ir
    /// @0: function i32 @main() {
    /// 0:
    ///     %0 = i32 iconst 0000000000000001
    ///     branch %0, $1, $2
    /// 1: // predecessor to $3
    ///     %1 = i32 iconst 0000000000000045
    ///     jump $3
    /// 2: // also a predecessor to $3
    ///     %2 = i32 iconst 00000000000001a4
    ///     jump $3
    /// 3:
    ///     %3 = i32 phi $1 => %1, $2 => %2 // %3 has to choose between %1 and %2 depending on
    ///                                     // which basic block entered
    ///     ret %3
    /// }
    /// ```
    Phi(Vec<(BasicBlockId, Value)>),

    /// Gets a variable.
    GetVar(VariableId),

    /// Sets a variable.
    SetVar(VariableId, Value),

    /// Calls a function with the given arguments.
    Call(FunctionId, Vec<Value>),

    /// Calls a function value with the given arguments.
    CallIndirect(Value, Vec<Value>),
}

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operation::Identity(val) => write!(f, "id {}", val),

            Operation::Integer(signed, val) => {
                if *signed {
                    write!(f, "iconst ")?;
                } else {
                    write!(f, "uconst ")?;
                }

                if val.is_empty() {
                    write!(f, "0")
                } else {
                    for byte in val.iter().rev() {
                        write!(f, "{:02x}", byte)?;
                    }
                    Ok(())
                }
            }

            Operation::Add(a, b) => write!(f, "addi {}, {}", a, b),
            Operation::Sub(a, b) => write!(f, "subi {}, {}", a, b),
            Operation::Mul(a, b) => write!(f, "muli {}, {}", a, b),
            Operation::Div(a, b) => write!(f, "divi {}, {}", a, b),
            Operation::Mod(a, b) => write!(f, "mod {}, {}", a, b),
            Operation::Bsl(a, b) => write!(f, "shiftl {}, {}", a, b),
            Operation::Bsr(a, b) => write!(f, "shiftr {}, {}", a, b),
            Operation::Eq(a, b) => write!(f, "eqi {}, {}", a, b),
            Operation::Ne(a, b) => write!(f, "neqi {}, {}", a, b),
            Operation::Lt(a, b) => write!(f, "lti {}, {}", a, b),
            Operation::Le(a, b) => write!(f, "leqi {}, {}", a, b),
            Operation::Gt(a, b) => write!(f, "gti {}, {}", a, b),
            Operation::Ge(a, b) => write!(f, "geqi {}, {}", a, b),
            Operation::BitAnd(a, b) => write!(f, "andi {}, {}", a, b),
            Operation::BitOr(a, b) => write!(f, "ori {}, {}", a, b),
            Operation::BitXor(a, b) => write!(f, "xori {}, {}", a, b),

            Operation::Phi(maps) => {
                write!(f, "phi ")?;
                let mut first = true;
                for (block, value) in maps {
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }

                    write!(f, "{} => {}", block, value)?;
                }
                Ok(())
            }

            Operation::GetVar(var) => write!(f, "get {}", var),
            Operation::SetVar(var, val) => write!(f, "set {}, {}", var, val),

            Operation::Call(func, args) => {
                write!(f, "call {}(", func)?;
                let mut first = true;
                for arg in args {
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }

            Operation::CallIndirect(func, args) => {
                write!(f, "icall {}(", func)?;
                let mut first = true;
                for arg in args {
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// [`Terminator`] terminates a given basic block. For information on basic blocks, see
/// [`ModuleBuilder::push_block`].
pub enum Terminator {
    /// No terminator has been added to the block yet. Note that compiling blocks with this as its
    /// terminator results in undefined behaviour.
    NoTerminator,

    /// The block ends with a return with no value.
    ReturnVoid,

    /// The block ends with a return with a value.
    Return(Value),

    /// The block ends with a jump to another block.
    Jump(BasicBlockId),

    /// The block ends with a branch to two different blocks depending on the truthiness of the
    /// value. If the value is true, it jumps to the first block; otherwise, it jumps to the second
    /// block.
    Branch(Value, BasicBlockId, BasicBlockId),
}

impl Display for Terminator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Terminator::NoTerminator => write!(f, "noterm"),
            Terminator::ReturnVoid => write!(f, "ret void"),
            Terminator::Return(v) => write!(f, "ret {}", v),
            Terminator::Jump(b) => write!(f, "jump {}", b),
            Terminator::Branch(c, t, e) => write!(f, "branch {}, {}, {}", c, t, e),
        }
    }
}

/// [`Value`] represents a reference to a value in an IR function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Value(pub(crate) usize);

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// [`ModuleBuilder`] is used to build a module.
///
/// # Example
/// ```rust
/// # fn testy() -> Option<()> {
/// # use codegem::ir::*;
/// let mut builder = ModuleBuilder::default()
///     .with_name("uwu");
/// let main = builder.new_function("main", &[], &Type::Void);
/// builder.switch_to_function(main);
/// let entry = builder.push_block()?;
/// builder.switch_to_block(entry);
/// let val = builder.push_instruction(
///     &Type::Integer(true, 32),
///     69u32.to_integer_operation()
/// )?;
/// builder.set_terminator(Terminator::Return(val));
/// let module = builder.build();
/// # Some(())
/// # }
/// ```
#[derive(Default)]
pub struct ModuleBuilder {
    internal: Module,
    current_function: Option<usize>,
    current_block: Option<usize>,
}

impl ModuleBuilder {
    /// Sets the name of the module being built.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// let builder = ModuleBuilder::default()
    ///     .with_name("uwu");
    /// ```
    pub fn with_name(mut self, name: &str) -> Self {
        self.internal.name = name.to_owned();
        self
    }

    /// Consumes the builder and returns a module, performing some mandatory transformations such
    /// as dead code elimination and Phi operation lowering, as well as checking for malformed IR.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// let empty_module = ModuleBuilder::default()
    ///     .build();
    /// ```
    // TODO: return Result<Module, MalformedIrError> on malformed IR instead of panicking.
    pub fn build(mut self) -> Module {
        for (f, func) in self.internal.functions.iter_mut().enumerate() {
            let mut blocks_prec = vec![HashSet::new(); func.blocks.len()];
            for (i, block) in func.blocks.iter().enumerate() {
                match &block.terminator {
                    Terminator::NoTerminator => (),
                    Terminator::ReturnVoid => (),
                    Terminator::Return(_) => (),

                    Terminator::Jump(next) => {
                        let this = BasicBlockId(FunctionId(f), i);
                        blocks_prec[next.1].insert(this);
                    }

                    Terminator::Branch(_, on_true, on_false) => {
                        let this = BasicBlockId(FunctionId(f), i);
                        blocks_prec[on_true.1].insert(this);
                        blocks_prec[on_false.1].insert(this);
                    }
                }
            }

            for (block, prec) in func.blocks.iter_mut().zip(blocks_prec.into_iter()) {
                block.predecessors = prec;
            }

            let mut removed = Vec::new();
            while {
                removed.clear();

                for (i, block) in func.blocks.iter_mut().enumerate() {
                    if block.deleted || i == 0 {
                        continue;
                    }

                    if block.predecessors.is_empty() {
                        block.deleted = true;
                        removed.push(BasicBlockId(FunctionId(f), i));
                    }
                }

                for block in func.blocks.iter_mut() {
                    for remove in removed.iter() {
                        block.predecessors.remove(remove);
                    }
                }

                !removed.is_empty()
            } {}

            for block in func.blocks.iter() {
                if block.deleted {
                    continue;
                }

                if let Terminator::Branch(_, a, b) = block.terminator {
                    if func.blocks[a.1].predecessors.len() > 1 || func.blocks[b.1].predecessors.len() > 1 {
                        panic!("malformed ir");
                    }
                }
            }

            let mut var_map = HashMap::new();
            for arg in 0..func.arg_types.len() {
                var_map.insert((VariableId(arg), 0), Value(arg));
            }

            let mut phi_to_var_map = HashMap::new();
            for (i, block) in func.blocks.iter_mut().enumerate() {
                if block.deleted {
                    continue;
                }

                match block.predecessors.len() {
                    0 => (),

                    1 => {
                        let prev = block.predecessors.iter().next().unwrap().1;
                        for var in 0..func.variables.len() {
                            let var = VariableId(var);
                            if let Some(&val) = var_map.get(&(var, prev)) {
                                var_map.insert((var, i), val);
                            }
                        }
                    }

                    _ => {
                        for var in 0..func.variables.len() {
                            let var = VariableId(var);
                            let operation = Operation::Phi(Vec::new());
                            let val = Value(func.value_index);
                            func.value_index += 1;
                            let phi = Instruction {
                                yielded: Some(val),
                                operation,
                            };
                            block.instructions.insert(0, phi);
                            var_map.insert((var, i), val);
                            phi_to_var_map.insert(val, var);
                        }
                    }
                }

                let mut to_remove = Vec::new();
                for (j, instruction) in block.instructions.iter_mut().enumerate() {
                    match instruction.operation {
                        Operation::GetVar(var) => {
                            match var_map.get(&(var, i)) {
                                Some(&val) => {
                                    instruction.operation = Operation::Identity(val);
                                }

                                None => {
                                    panic!("malformed ir {}", var);
                                }
                            }
                        }

                        Operation::SetVar(var, val) => {
                            var_map.insert((var, i), val);
                            to_remove.push(j);
                        }

                        _ => (),
                    }
                }

                for remove in to_remove.into_iter().rev() {
                    block.instructions.remove(remove);
                }
            }

            for block in func.blocks.iter_mut() {
                if block.deleted {
                    continue;
                }

                if block.predecessors.len() > 1 {
                    for instruction in block.instructions.iter_mut() {
                        if let Operation::Phi(mapping) = &mut instruction.operation {
                            if let Some(&var) = instruction.yielded.as_ref().and_then(|v| phi_to_var_map.get(v)) {
                                *mapping = block.predecessors.iter().filter_map(|&v| var_map.get(&(var, v.1)).map(|&u| (v, u))).collect();
                            }
                        }
                    }
                }
            }
        }

        self.internal
    }

    /// Adds a new function to the module being built. Returns a [`FunctionId`], which can be used
    /// to reference the built function.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) {
    /// let main_func = builder.new_function("main", &[], &Type::Void);
    /// # }
    /// ```
    pub fn new_function(
        &mut self,
        name: &str,
        linkage: Linkage,
        args: &[(&str, Type)],
        ret_type: &Type,
    ) -> FunctionId {
        let id = self.internal.functions.len();
        self.internal.functions.push(Function {
            name: name.to_owned(),
            linkage,
            arg_types: args.iter().map(|(_, t)| t.clone()).collect(),
            ret_type: ret_type.clone(),
            variables: args
                .iter()
                .map(|(n, t)| Variable {
                    name: (*n).to_owned(),
                    type_: t.clone(),
                })
                .collect(),
            blocks: Vec::new(),
            value_index: args.len(),
        });
        FunctionId(id)
    }

    /// Switches the function to which the builder is currently adding blocks and instructions.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) {
    /// let main_func = builder.new_function("main", &[], &Type::Void);
    /// builder.switch_to_function(main_func);
    /// # }
    /// ```
    pub fn switch_to_function(&mut self, id: FunctionId) {
        self.current_function = Some(id.0);
        self.current_block = None;
    }

    /// Adds a new basic block to the current function. Returns a [`BasicBlockId`], which can be used
    /// to reference the built basic block. Returns None if there is no function currently
    /// selected.
    ///
    /// A basic block is a single strand of code that does not have any control flow within it.
    /// Function calls do not count as control flow in this case. Each basic block contains
    /// instructions and ends with a single terminator, which may jump to one or more basic blocks,
    /// or exit from the current function or program.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) -> Option<()> {
    /// let entry_block = builder.push_block()?;
    /// # None
    /// # }
    /// ```
    pub fn push_block(&mut self) -> Option<BasicBlockId> {
        if let Some(func_id) = self.current_function {
            let func = self.internal.functions.get_mut(func_id)?;
            let block_id = func.blocks.len();
            func.blocks.push(BasicBlock {
                deleted: false,
                predecessors: HashSet::new(),
                instructions: Vec::new(),
                terminator: Terminator::NoTerminator,
            });
            Some(BasicBlockId(FunctionId(func_id), block_id))
        } else {
            None
        }
    }

    /// Switches the basic block to which the builder is currently adding blocks and instructions.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) -> Option<()> {
    /// let entry_block = builder.push_block()?;
    /// builder.switch_to_block(entry_block);
    /// # None
    /// # }
    /// ```
    pub fn switch_to_block(&mut self, id: BasicBlockId) {
        match self.current_function {
            Some(x) if id.0 .0 == x => self.current_block = Some(id.1),
            _ => self.current_block = None,
        }
    }

    /// Pushes an instruction to the current block in the current function. Returns an optional
    /// [`Value`] depending on if there is a selected function and block, and if the instruction
    /// does yield a value. (Some [`Operation`]s, such as [`Operation::SetVar`], do not return
    /// anything.)
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) -> Option<()> {
    /// builder.push_instruction(
    ///     69i32.to_integer_operation()
    /// );
    /// # None
    /// # }
    pub fn push_instruction(&mut self, instr: Operation) -> Option<Value> {
        if let Some(func_id) = self.current_function {
            if let Some(block_id) = self.current_block {
                let yielded = match &instr {
                    Operation::SetVar(_, _) => false,
                    Operation::Call(f, _) => {
                        if let Some(f) = self.internal.functions.get(f.0) {
                            !matches!(f.ret_type, Type::Void)
                        } else {
                            false
                        }
                    }

                    _ => true,
                };
                let func = self.internal.functions.get_mut(func_id)?;
                let block = func.blocks.get_mut(block_id)?;
                let yielded = if yielded {
                    func.value_index += 1;
                    Some(Value(func.value_index - 1))
                } else {
                    None
                };
                block.instructions.push(Instruction {
                    yielded,
                    operation: instr,
                });
                if let Some(_) = yielded {
                    return yielded;
                } else {
                    return None;
                }
            }
        }

        None
    }

    /// Sets the [`Terminator`] of the current basic block.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) {
    /// builder.set_terminator(Terminator::ReturnVoid);
    /// # }
    pub fn set_terminator(&mut self, terminator: Terminator) {
        if let Some(func) = self.current_function.and_then(|v| self.internal.functions.get_mut(v)) {
            if let Some(block) = self.current_block.and_then(|v| func.blocks.get_mut(v)) {
                block.terminator = terminator;
            }
        }
    }

    /// Pushes a variable of the given name and type to the current function. This variable can be
    /// used anywhere in the function it is defined in. Currently, creating an [`Operation::GetVar`]
    /// instruction before an [`Operation::SetVar`] instruction is undefined behaviour. Returns a
    /// [`VariableId`] if there is a currently selected function, which can be used to reference
    /// the variable.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) {
    /// let i = builder.push_variable("i", &Type::Integer(true, 32));
    /// # }
    /// ```
    pub fn push_variable(&mut self, name: &str, type_: &Type) -> Option<VariableId> {
        if let Some(func_id) = self.current_function {
            let func = self.internal.functions.get_mut(func_id)?;
            let id = func.variables.len();
            func.variables.push(Variable {
                name: name.to_owned(),
                type_: type_.clone(),
            });
            Some(VariableId(id))
        } else {
            None
        }
    }

    /// Gets the currently selected function.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) {
    /// let current = builder.get_function();
    /// # }
    /// ```
    pub fn get_function(&self) -> Option<FunctionId> {
        self.current_function.map(FunctionId)
    }

    /// Gets the arguments of the given function as variables
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) -> Option<()> {
    /// let current = builder.get_function()?;
    /// let args = builder.get_function_args(current)?;
    /// # None
    /// # }
    /// ```
    pub fn get_function_args(&self, func: FunctionId) -> Option<Vec<VariableId>> {
        self.internal
            .functions
            .get(func.0)
            .map(|f| (0..f.arg_types.len()).into_iter().map(VariableId).collect())
    }

    /// Gets the currently selected basic block.
    ///
    /// # Example
    /// ```rust
    /// # use codegem::ir::*;
    /// # fn testy(builder: &mut ModuleBuilder) -> Option<()> {
    /// let current = builder.get_block()?;
    /// # None
    /// # }
    /// ```
    pub fn get_block(&self) -> Option<BasicBlockId> {
        if let Some(f) = self.get_function() {
            self.current_block.map(|b| BasicBlockId(f, b))
        } else {
            None
        }
    }
}
