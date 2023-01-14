use std::fs::File;

use codegem::ir::*;
use codegem::arch::urcl::UrclSelector;
use codegem::regalloc::RegAlloc;

fn main() {
    let mut builder = ModuleBuilder::default().with_name("test");
    let main_func = builder.new_function("main", &[], &Type::Void);
    builder.switch_to_function(main_func);
    let entry = builder.push_block().unwrap();
    builder.switch_to_block(entry);
    let val = builder.push_instruction(&Type::Integer(true, 32), Operation::Integer(
        true,
        69u32.to_le_bytes().to_owned().to_vec())).unwrap();
    
    let val2 = builder.push_instruction(&Type::Integer(true, 32), Operation::Integer(
            true,
            69u32.to_le_bytes().to_owned().to_vec())).unwrap();
    
    let val3 = builder.push_instruction(&Type::Integer(true, 32), Operation::Add(val, val2)).unwrap();
    builder.set_terminator(Terminator::Return(val3));
    let module = builder.build();
    let mut vcode = module.lower_to_vcode::<_, UrclSelector>();
    vcode.allocate_regs::<RegAlloc>();
    vcode.emit_assembly(&mut File::create("test.urcl").unwrap()).unwrap();
}
