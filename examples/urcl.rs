use codegem::ir::*;
use codegem::arch::urcl::UrclSelector;
use codegem::regalloc::RegAlloc;

fn main() -> Result<(), ModuleCreationError> {
    let mut builder = ModuleBuilder::default().with_name("test");
    let main_func = builder.new_function("main",  Linkage::Public, &[], &Type::Void);
    builder.switch_to_function(main_func);
    let entry = builder.push_block().unwrap();
    builder.switch_to_block(entry);
    let val = builder.push_instruction(69u32.to_integer_operation())?.unwrap();
    let val2 = builder.push_instruction(69u32.to_integer_operation())?.unwrap();
    let val3 = builder.push_instruction(Operation::Add(val, val2))?.unwrap();
    builder.set_terminator(Terminator::Return(val3))?;
    let module = builder.build()?;
    let mut vcode = module.lower_to_vcode::<_, UrclSelector>();
    vcode.allocate_regs::<RegAlloc>();
    vcode.emit_assembly(&mut std::fs::File::create("out.s").unwrap()).unwrap();
    Ok(())
}