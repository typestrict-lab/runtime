//! typestrict-serve: a typestrict.ai model on CPU behind Jev's decision API.
//!
//! - [`manifest`]: `decision_ai.json`, the model package's manifest.
//! - [`render`]: content to text, byte for byte as training rendered it.
//! - [`reader`]: the option reader (the Dinah protocol), a port of the typestrict.ai playground's `reader.ts`.
//! - [`contract`]: Jev's request and answer shapes.
//! - [`server`]: the HTTP routes.

pub mod contract;
pub mod manifest;
pub mod reader;
pub mod render;
pub mod server;
