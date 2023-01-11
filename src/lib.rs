#![doc = include_str!("../README.md")]

#![deny(rustdoc::missing_doc_code_examples)]

/// [`arch`] contains the default architectures supported by `codegem`. An architecture does not
/// need to be in this module to be supported, as the traits that must be implemented are public
/// and can be implemented by other crates wishing to extend `codegem`'s functionality.
pub mod arch;

/// [`ir`] contains the intermediate representation (IR) that programs using this crate
/// interface with, as well as a builder struct to create the intermediate representation.
pub mod ir;

/// [`regalloc`] contains the default register allocator. One can implement their own register
/// allocator by implementing the [`regalloc::RegisterAllocator`] trait, if one does not have a skill issue.
pub mod regalloc;

