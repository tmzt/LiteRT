#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    improper_ctypes,
    clippy::useless_transmute,
    clippy::transmute_int_to_bool,
    clippy::missing_safety_doc
)]
#![doc = "Raw FFI bindings to LiteRT-LM C API. Use the safe `litert-lm` crate for idiomatic Rust."]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// chitin local: v0.13.1 renamed many types/constants with the
// `LiteRtLm` / `kLiteRtLm` prefix (vs v0.10.2's bare names). The
// safe `litertlm` wrapper crate was written against v0.10.2 names,
// so re-export aliases so the wrapper compiles unchanged against
// our updated bindings.
pub use LiteRtLmInputData as InputData;
pub use LiteRtLmInputDataType as InputDataType;
pub use LiteRtLmSamplerType as Type;
pub const kInputText:     LiteRtLmInputDataType = kLiteRtLmInputDataTypeText;
pub const kInputImage:    LiteRtLmInputDataType = kLiteRtLmInputDataTypeImage;
pub const kInputImageEnd: LiteRtLmInputDataType = kLiteRtLmInputDataTypeImageEnd;
pub const kInputAudio:    LiteRtLmInputDataType = kLiteRtLmInputDataTypeAudio;
pub const kInputAudioEnd: LiteRtLmInputDataType = kLiteRtLmInputDataTypeAudioEnd;
pub const kTypeUnspecified: LiteRtLmSamplerType = kLiteRtLmSamplerTypeUnspecified;
pub const kTopK:    LiteRtLmSamplerType = kLiteRtLmSamplerTypeTopK;
pub const kTopP:    LiteRtLmSamplerType = kLiteRtLmSamplerTypeTopP;
pub const kGreedy:  LiteRtLmSamplerType = kLiteRtLmSamplerTypeGreedy;
